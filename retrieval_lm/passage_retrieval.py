# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import json
import pickle
import time
import glob

import numpy as np
import torch

import src.index
import src.contriever
import src.utils
import src.slurm
import src.data
import src.normalize_text

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class Retriever:
    def __init__(self, args, model=None, tokenizer=None):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer

    def embed_queries(self, args, queries):
        embeddings, batch_question = [], []
        with torch.no_grad():
            for k, q in enumerate(queries):
                if args.lowercase:
                    q = q.lower()
                if args.normalize_text:
                    q = src.normalize_text.normalize(q)
                batch_question.append(q)

                if (
                    len(batch_question) == args.per_gpu_batch_size
                    or k == len(queries) - 1
                ):
                    encoded_batch = self.tokenizer.batch_encode_plus(
                        batch_question,
                        return_tensors="pt",
                        max_length=args.question_maxlength,
                        padding=True,
                        truncation=True,
                    )
                    encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
                    output = self.model(**encoded_batch)
                    embeddings.append(output.cpu())

                    batch_question = []

        embeddings = torch.cat(embeddings, dim=0)
        print(f"Questions embeddings shape: {embeddings.size()}")

        return embeddings.numpy()

    def index_encoded_data(self, index, embedding_files, indexing_batch_size):
        allids = []
        allembeddings = np.array([])
        for file_path in embedding_files:
            print(f"Loading file {file_path}")
            with open(file_path, "rb") as fin:
                ids, embeddings = pickle.load(fin)

            allembeddings = (
                np.vstack((allembeddings, embeddings))
                if allembeddings.size
                else embeddings
            )
            allids.extend(ids)
            while allembeddings.shape[0] > indexing_batch_size:
                allembeddings, allids = self.add_embeddings(
                    index, allembeddings, allids, indexing_batch_size
                )

        while allembeddings.shape[0] > 0:
            allembeddings, allids = self.add_embeddings(
                index, allembeddings, allids, indexing_batch_size
            )

        print("Data indexing completed.")

    def add_embeddings(self, index, embeddings, ids, indexing_batch_size):
        end_idx = min(indexing_batch_size, embeddings.shape[0])
        ids_toadd = ids[:end_idx]
        embeddings_toadd = embeddings[:end_idx]
        ids = ids[end_idx:]
        embeddings = embeddings[end_idx:]
        index.index_data(ids_toadd, embeddings_toadd)
        return embeddings, ids

    def add_passages(self, top_passages_and_scores):
        # Fetch passages from the database
        doc_ids = top_passages_and_scores[0][0]
        return src.data.fetch_passages_from_db(self.args.db_path, doc_ids)

    def setup_retriever(self):
        print(f"Loading model from: {self.args.model_name_or_path}")
        self.model, self.tokenizer, _ = src.contriever.load_retriever(
            self.args.model_name_or_path
        )
        self.model.eval()
        self.model = self.model.cuda()
        if not self.args.no_fp16:
            self.model = self.model.half()

        self.index = src.index.Indexer(
            self.args.projection_size, self.args.n_subquantizers, self.args.n_bits
        )

        # index all passages
        input_paths = glob.glob(self.args.passages_embeddings)
        input_paths = sorted(input_paths)
        embeddings_dir = os.path.dirname(input_paths[0])
        index_path = os.path.join(embeddings_dir, "index.faiss")
        if self.args.save_or_load_index and os.path.exists(index_path):
            self.index.deserialize_from(embeddings_dir)
        else:
            print(f"Indexing passages from files {input_paths}")
            start_time_indexing = time.time()
            self.index_encoded_data(
                self.index, input_paths, self.args.indexing_batch_size
            )
            print(f"Indexing time: {time.time()-start_time_indexing:.1f} s.")
            if self.args.save_or_load_index:
                self.index.serialize(embeddings_dir)

        print("loading passages")
        src.data.load_passages_to_db(
            self.args.passages, self.args.db_path, force_reload=args.force_reload
        )

        print("passages have been loaded")

    def search_document(self, query, top_n=10):
        questions_embedding = self.embed_queries(self.args, [query])

        # get top k results
        start_time_retrieval = time.time()
        top_ids_and_scores = self.index.search_knn(
            questions_embedding, self.args.n_docs
        )
        print(f"Search time: {time.time()-start_time_retrieval:.1f} s.")

        return self.add_passages(self.passage_id_map, top_ids_and_scores)[:top_n]


def load_data(data_path):
    if data_path.endswith(".json"):
        with open(data_path, "r") as fin:
            data = json.load(fin)
    elif data_path.endswith(".jsonl"):
        data = []
        with open(data_path, "r") as fin:
            for example in fin:
                example = json.loads(example)
                data.append(example)
    return data


def main(args):
    retriever = Retriever(args)
    retriever.setup_retriever()
    print(retriever.search_document(args.query, args.n_docs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help=".json file containing question and answers, similar format to reader data",
    )
    parser.add_argument(
        "--passages", type=str, default=None, help="Path to passages (.tsv file)"
    )
    parser.add_argument(
        "--passages_embeddings",
        type=str,
        default=None,
        help="Glob path to encoded passages",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Results are written to outputdir with data suffix",
    )
    parser.add_argument(
        "--n_docs",
        type=int,
        default=100,
        help="Number of documents to retrieve per questions",
    )
    parser.add_argument(
        "--validation_workers",
        type=int,
        default=32,
        help="Number of parallel processes to validate results",
    )
    parser.add_argument(
        "--per_gpu_batch_size",
        type=int,
        default=64,
        help="Batch size for question encoding",
    )
    parser.add_argument(
        "--save_or_load_index",
        action="store_true",
        help="If enabled, save index and load index if it exists",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="path to directory containing model weights and config file",
    )
    parser.add_argument("--no_fp16", action="store_true", help="inference in fp32")
    parser.add_argument(
        "--question_maxlength",
        type=int,
        default=512,
        help="Maximum number of tokens in a question",
    )
    parser.add_argument(
        "--indexing_batch_size",
        type=int,
        default=1000000,
        help="Batch size of the number of passages indexed",
    )
    parser.add_argument("--projection_size", type=int, default=768)
    parser.add_argument(
        "--n_subquantizers",
        type=int,
        default=0,
        help="Number of subquantizer used for vector quantization, if 0 flat index is used",
    )
    parser.add_argument(
        "--n_bits", type=int, default=8, help="Number of bits per subquantizer"
    )
    parser.add_argument("--lang", nargs="+")
    parser.add_argument("--dataset", type=str, default="none")
    parser.add_argument(
        "--lowercase", action="store_true", help="lowercase text before encoding"
    )
    parser.add_argument("--normalize_text", action="store_true", help="normalize text")
    parser.add_argument(
        "--force_reload",
        action="store_true",
        help="Force re-loading passages into the database even if already loaded.",
    )
    parser.add_argument(
        "--db_path",
        type=str,
        default="passages.db",
        help="Path to SQLite database storing the passages.",
    )

    args = parser.parse_args()
    src.slurm.init_distributed_mode(args)
    main(args)

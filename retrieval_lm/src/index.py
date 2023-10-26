import os
import pickle
from typing import List, Tuple

import faiss
import numpy as np
from tqdm import tqdm


class Indexer(object):
    def __init__(self, vector_sz, n_subquantizers=0, n_bits=8, use_gpu=False):
        self.use_gpu = use_gpu

        if n_subquantizers > 0:
            self.index = faiss.IndexPQ(
                vector_sz, n_subquantizers, n_bits, faiss.METRIC_INNER_PRODUCT
            )
        else:
            self.index = faiss.IndexFlatIP(vector_sz)

        self.index_id_to_db_id = []

        # If using GPU, set up GPU resources and transfer index to GPU
        if self.use_gpu:
            self.res = faiss.StandardGpuResources()
            self.gpu_index = faiss.index_cpu_to_gpu(self.res, 0, self.index)
        else:
            self.gpu_index = None

    def index_data(self, ids, embeddings, chunk_size=100000):
        total_len = len(embeddings)
        for i in range(0, total_len, chunk_size):
            end_idx = min(i + chunk_size, total_len)
            chunk_ids = ids[i:end_idx]
            chunk_embeddings = embeddings[i:end_idx].astype("float32")

            self._update_id_mapping(chunk_ids)

            # Use GPU index if set, otherwise use the regular index
            index_to_use = self.gpu_index if self.use_gpu else self.index
            if not index_to_use.is_trained:
                index_to_use.train(chunk_embeddings)
            index_to_use.add(chunk_embeddings)

            # Optional print statement for tracking progress
            print(f"Indexed {end_idx}/{total_len} embeddings.")

    def search_knn(
        self, query_vectors: np.array, top_docs: int, index_batch_size: int = 2048
    ) -> List[Tuple[List[object], List[float]]]:
        query_vectors = query_vectors.astype("float32")
        result = []

        # Use GPU index if set, otherwise use the regular index
        index_to_use = self.gpu_index if self.use_gpu else self.index

        nbatch = (len(query_vectors) - 1) // index_batch_size + 1
        for k in tqdm(range(nbatch)):
            start_idx = k * index_batch_size
            end_idx = min((k + 1) * index_batch_size, len(query_vectors))
            q = query_vectors[start_idx:end_idx]
            scores, indexes = index_to_use.search(q, top_docs)
            # convert to external ids
            db_ids = [
                [str(self.index_id_to_db_id[i]) for i in query_top_idxs]
                for query_top_idxs in indexes
            ]
            result.extend([(db_ids[i], scores[i]) for i in range(len(db_ids))])
        return result

    def serialize(self, dir_path):
        index_file = os.path.join(dir_path, "index.faiss")
        meta_file = os.path.join(dir_path, "index_meta.faiss")
        print(f"Serializing index to {index_file}, meta data to {meta_file}")

        # Use GPU index if set, otherwise use the regular index
        index_to_use = self.gpu_index if self.use_gpu else self.index
        faiss.write_index(index_to_use, index_file)

        with open(meta_file, mode="wb") as f:
            pickle.dump(self.index_id_to_db_id, f)

    def deserialize_from(self, dir_path):
        index_file = os.path.join(dir_path, "index.faiss")
        meta_file = os.path.join(dir_path, "index_meta.faiss")
        print(f"Loading index from {index_file}, meta data from {meta_file}")

        # If using GPU, load index into GPU, otherwise load regularly
        if self.use_gpu:
            self.gpu_index = faiss.read_index(index_file, faiss.IO_FLAG_ONDISK_SAME_DIR)
            self.gpu_index = faiss.index_cpu_to_gpu(self.res, 0, self.gpu_index)
        else:
            self.index = faiss.read_index(index_file, faiss.IO_FLAG_ONDISK_SAME_DIR)
            self.gpu_index = None

        print(
            "Loaded index of type %s and size %d"
            % (
                type(self.gpu_index if self.use_gpu else self.index),
                (self.gpu_index if self.use_gpu else self.index).ntotal,
            )
        )

        with open(meta_file, "rb") as reader:
            self.index_id_to_db_id = pickle.load(reader)
        assert (
            len(self.index_id_to_db_id)
            == (self.gpu_index if self.use_gpu else self.index).ntotal
        ), "Deserialized index_id_to_db_id should match faiss index size"

    def _update_id_mapping(self, db_ids: List):
        self.index_id_to_db_id.extend(db_ids)

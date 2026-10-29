import faiss
import numpy as np


class RetrievalIndex:
    def __init__(self, embeddings, eligible, threads=4):
        # Single-thread HNSW construction is deterministic and avoids mixed
        # PyTorch/FAISS OpenMP deadlocks seen on macOS ARM64. Queries are single-row.
        faiss.omp_set_num_threads(1)
        self.ids = np.asarray(eligible, dtype=np.int64)
        vectors = np.ascontiguousarray(embeddings[self.ids], dtype=np.float32)
        self.exact = faiss.IndexFlatIP(vectors.shape[1])
        self.approximate = faiss.IndexHNSWFlat(vectors.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
        self.approximate.hnsw.efConstruction = 120
        self.approximate.hnsw.efSearch = 128
        self.exact.add(vectors)
        self.approximate.add(vectors)

    def search(self, query, k, seen=frozenset(), approximate=True):
        if k < 1 or not len(self.ids):
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        # Oversample enough to compensate for every seen item in this catalog.
        size = min(len(self.ids), k + len(seen))
        index = self.approximate if approximate else self.exact
        scores, local = index.search(np.asarray(query, dtype=np.float32).reshape(1, -1), size)
        pairs = [(int(self.ids[j]), float(s)) for j, s in zip(local[0], scores[0]) if j >= 0 and self.ids[j] not in seen]
        pairs = sorted(pairs, key=lambda x: (-x[1], x[0]))[:k]
        return np.array([x[0] for x in pairs], dtype=np.int64), np.array([x[1] for x in pairs])

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FAISS-backed vector index.
Supports IndexFlatIP (exact) and IndexHNSWFlat (approximate, production).
Raises ImportError at import time if faiss is not installed,
allowing callers to fall back to NumpyVectorIndex.
"""

from __future__ import annotations

import os
from typing import List

import faiss
import numpy as np

from RAG.index.base import SearchResult, VectorIndex


class FaissVectorIndex(VectorIndex):
    """
    FAISS index with two modes:
      flat_ip  — IndexFlatIP, exact inner-product search, suitable for < 1M vectors.
      hnsw     — IndexHNSWFlat, approximate, sub-linear query time, O(log N) build.

    Vectors are normalised before insertion so inner product == cosine similarity.
    """

    def __init__(
        self,
        dimension: int,
        index_type: str = "flat_ip",
        hnsw_m: int = 32,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 64,
    ) -> None:
        self._dim = dimension
        self._index_type = index_type
        self._index = self._build_index(
            dimension, index_type, hnsw_m, hnsw_ef_construction, hnsw_ef_search
        )
        self._size = 0

    # ── VectorIndex interface ──────────────────────────────────────────

    def add(self, vectors: np.ndarray) -> None:
        if vectors.ndim != 2 or vectors.shape[1] != self._dim:
            raise ValueError(
                f"Expected shape (n, {self._dim}), got {vectors.shape}"
            )
        normed = self._normalise(vectors.astype(np.float32))
        self._index.add(normed)
        self._size += len(vectors)

    def search(self, query: np.ndarray, k: int) -> List[SearchResult]:
        if self._size == 0:
            return []
        q = self._normalise(query.astype(np.float32).reshape(1, -1))
        actual_k = min(k, self._size)
        distances, indices = self._index.search(q, actual_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:           # FAISS pads with -1 when < k results exist
                continue
            results.append(SearchResult(index=int(idx), score=float(dist)))
        return results

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self._index, os.path.join(path, "index.faiss"))

    def load(self, path: str) -> None:
        self._index = faiss.read_index(os.path.join(path, "index.faiss"))
        self._size = self._index.ntotal

    @property
    def size(self) -> int:
        return self._size

    @property
    def dimension(self) -> int:
        return self._dim

    # ── Internal ───────────────────────────────────────────────────────

    @staticmethod
    def _normalise(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        return vectors / norms

    @staticmethod
    def _build_index(
        dim: int, index_type: str, m: int, ef_construction: int, ef_search: int
    ) -> faiss.Index:
        if index_type == "hnsw":
            index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = ef_construction
            index.hnsw.efSearch = ef_search
            return index
        # default: flat_ip
        return faiss.IndexFlatIP(dim)

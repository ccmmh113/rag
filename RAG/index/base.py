#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Abstract vector index interface.
Decouples retrieval logic from the underlying ANN backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class SearchResult:
    """Single result from an ANN search."""
    index: int          # position in the original document list
    score: float        # similarity score (higher = better)


class VectorIndex(ABC):
    """
    Abstract ANN index. Implementations must be:
    - add()    : ingest vectors (called once during indexing)
    - search() : top-k retrieval at query time
    - save/load: persistence
    """

    @abstractmethod
    def add(self, vectors: np.ndarray) -> None:
        """
        Add vectors to the index.
        vectors: float32 array of shape (n, dim).
        IDs are implicit: position 0, 1, 2 ... n-1.
        """

    @abstractmethod
    def search(self, query: np.ndarray, k: int) -> List[SearchResult]:
        """
        Return up to k nearest neighbours for query vector.
        query: float32 array of shape (dim,).
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist index to disk."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load index from disk."""

    @property
    @abstractmethod
    def size(self) -> int:
        """Number of vectors currently stored."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding dimension."""

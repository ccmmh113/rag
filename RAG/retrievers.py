#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import bm25s
import numpy as np
from bm25s.tokenization import Tokenizer

from RAG.index.base import VectorIndex
from RAG.schema import Document
from RAG.types import EmbeddingModel


def apply_metadata_filter(
    documents: Iterable[Document],
    metadata_filter: Optional[Dict[str, object]] = None,
) -> List[Document]:
    if not metadata_filter:
        return list(documents)
    return [
        doc
        for doc in documents
        if all(doc.metadata.get(key) == value for key, value in metadata_filter.items())
    ]


class DenseRetriever:
    def __init__(
        self,
        documents: Sequence[Document],
        index: VectorIndex,
        embedding_model: EmbeddingModel,
    ) -> None:
        if len(documents) != index.size:
            raise ValueError(
                f"documents and index must have the same length: "
                f"{len(documents)} vs {index.size}"
            )
        self.documents = list(documents)
        self._index = index
        self.embedding_model = embedding_model

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        metadata_filter: Optional[Dict[str, object]] = None,
    ) -> List[Document]:
        query_vector = np.array(self.embedding_model.get_embedding(query), dtype=np.float32)
        search_k = top_k * 4 if metadata_filter else top_k
        results = self._index.search(query_vector, search_k)
        candidates: List[Document] = []
        for r in results:
            if r.index >= len(self.documents):
                continue
            doc = self.documents[r.index]
            if metadata_filter and not all(
                doc.metadata.get(key) == value for key, value in metadata_filter.items()
            ):
                continue
            candidates.append(
                Document(text=doc.text, score=r.score, metadata=dict(doc.metadata))
            )
            if len(candidates) >= top_k:
                break
        return candidates


class BM25Retriever:
    def __init__(self, documents: Sequence[Document], k1: float = 1.5, b: float = 0.75) -> None:
        self.documents = list(documents)
        self.k1 = k1
        self.b = b
        self._tokenizer = Tokenizer(
            lower=False,
            stopwords=[],
            splitter=self._tokenize,
        )
        corpus_texts = [doc.text for doc in self.documents]
        corpus_tokens = self._tokenizer.tokenize(corpus_texts)
        self._retriever = bm25s.BM25(k1=k1, b=b)
        self._retriever.index(corpus_tokens)

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        metadata_filter: Optional[Dict[str, object]] = None,
    ) -> List[Document]:
        if not self.documents:
            return []
        search_k = top_k * 4 if metadata_filter else top_k
        search_k = min(search_k, len(self.documents))
        query_tokens = self._tokenizer.tokenize([query], update_vocab=False)
        results, scores = self._retriever.retrieve(query_tokens, k=search_k)
        candidates: List[Document] = []
        for idx, score in zip(results[0], scores[0]):
            doc = self.documents[int(idx)]
            if metadata_filter and not all(
                doc.metadata.get(key) == value for key, value in metadata_filter.items()
            ):
                continue
            candidates.append(
                Document(text=doc.text, score=float(score), metadata=dict(doc.metadata))
            )
            if len(candidates) >= top_k:
                break
        return candidates

    def _tokenize(self, text: str) -> List[str]:
        lowered = text.lower()
        words = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", lowered)
        bigrams = [
            lowered[i : i + 2]
            for i in range(len(lowered) - 1)
            if "\u4e00" <= lowered[i] <= "\u9fff" and "\u4e00" <= lowered[i + 1] <= "\u9fff"
        ]
        return words + bigrams


@dataclass
class HybridRetrievalConfig:
    dense_top_k: int = 50
    sparse_top_k: int = 50
    final_top_k: int = 50
    fusion: str = "rrf"
    rrf_k: int = 60
    dense_weight: float = 0.5
    sparse_weight: float = 0.5
    parallel: bool = True


class HybridRetriever:
    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: BM25Retriever,
        config: Optional[HybridRetrievalConfig] = None,
    ) -> None:
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.config = config or HybridRetrievalConfig()

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        metadata_filter: Optional[Dict[str, object]] = None,
    ) -> List[Document]:
        final_top_k = top_k or self.config.final_top_k
        if self.config.parallel:
            with ThreadPoolExecutor(max_workers=2) as executor:
                dense_future = executor.submit(
                    self.dense_retriever.retrieve,
                    query,
                    self.config.dense_top_k,
                    metadata_filter,
                )
                sparse_future = executor.submit(
                    self.sparse_retriever.retrieve,
                    query,
                    self.config.sparse_top_k,
                    metadata_filter,
                )
                dense_docs = dense_future.result()
                sparse_docs = sparse_future.result()
        else:
            dense_docs = self.dense_retriever.retrieve(query, self.config.dense_top_k, metadata_filter)
            sparse_docs = self.sparse_retriever.retrieve(query, self.config.sparse_top_k, metadata_filter)

        if self.config.fusion == "weighted":
            fused = self._weighted_fusion(dense_docs, sparse_docs)
        else:
            fused = self._rrf_fusion(dense_docs, sparse_docs)
        return fused[:final_top_k]

    def _rrf_fusion(self, dense_docs: List[Document], sparse_docs: List[Document]) -> List[Document]:
        by_id: Dict[str, Document] = {}
        scores: Dict[str, float] = defaultdict(float)
        for rank, doc in enumerate(dense_docs, start=1):
            by_id.setdefault(doc.identity, doc)
            scores[doc.identity] += self.config.dense_weight / (self.config.rrf_k + rank)
        for rank, doc in enumerate(sparse_docs, start=1):
            by_id.setdefault(doc.identity, doc)
            scores[doc.identity] += self.config.sparse_weight / (self.config.rrf_k + rank)
        return self._sort_fused(by_id, scores)

    def _weighted_fusion(self, dense_docs: List[Document], sparse_docs: List[Document]) -> List[Document]:
        dense_scores = self._normalize_scores(dense_docs)
        sparse_scores = self._normalize_scores(sparse_docs)
        by_id: Dict[str, Document] = {}
        scores: Dict[str, float] = defaultdict(float)
        for doc in dense_docs:
            by_id.setdefault(doc.identity, doc)
            scores[doc.identity] += self.config.dense_weight * dense_scores[doc.identity]
        for doc in sparse_docs:
            by_id.setdefault(doc.identity, doc)
            scores[doc.identity] += self.config.sparse_weight * sparse_scores[doc.identity]
        return self._sort_fused(by_id, scores)

    def _normalize_scores(self, docs: List[Document]) -> Dict[str, float]:
        if not docs:
            return {}
        raw = [doc.score for doc in docs]
        low, high = min(raw), max(raw)
        if high == low:
            return {doc.identity: 1.0 for doc in docs}
        return {doc.identity: (doc.score - low) / (high - low) for doc in docs}

    def _sort_fused(self, by_id: Dict[str, Document], scores: Dict[str, float]) -> List[Document]:
        fused = []
        for identity, score in scores.items():
            doc = by_id[identity]
            fused.append(Document(text=doc.text, score=float(score), metadata=dict(doc.metadata)))
        fused.sort(key=lambda doc: doc.score, reverse=True)
        return fused

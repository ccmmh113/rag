#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ContextCompressor — reduces context size when it exceeds token budget.

Pipeline position:  retrieve → rerank → compress → build_context

Two strategies:
  relevance_filter  Fast, no LLM needed. Drops chunks below rerank_score threshold.
  summary           Uses LLM to summarise each chunk to a fixed token budget.
  hybrid            Filter first, then summarise the survivors if still over budget.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, TYPE_CHECKING

from RAG.chunking import count_tokens
from RAG.core.config import CompressionConfig
from RAG.schema import Document

if TYPE_CHECKING:
    from RAG.LLM import BaseModel


class BaseCompressor(ABC):
    """Abstract compressor interface."""

    @abstractmethod
    def compress(
        self,
        documents: List[Document],
        query: str,
        token_budget: int,
    ) -> List[Document]:
        """Return documents trimmed to fit within token_budget."""


class RelevanceFilterCompressor(BaseCompressor):
    """
    Drop chunks whose rerank_score (or score) is below the threshold
    until the context fits within the token budget.
    Runs in O(n) — no LLM call required.
    """

    def __init__(self, threshold: float = 0.3) -> None:
        self._threshold = threshold

    def compress(
        self,
        documents: List[Document],
        query: str,
        token_budget: int,
    ) -> List[Document]:
        filtered = self._filter_by_threshold(documents)
        if not filtered:
            filtered = documents[:1]

        total = sum(count_tokens(d.text) for d in filtered)
        if total <= token_budget:
            return filtered

        # Sort by effective score descending, drop from the tail
        scored = sorted(
            filtered,
            key=lambda d: d.rerank_score if d.rerank_score is not None else d.score,
            reverse=True,
        )
        kept: List[Document] = []
        used = 0
        for doc in scored:
            t = count_tokens(doc.text)
            if used + t <= token_budget:
                kept.append(doc)
                used += t
        return kept or scored[:1]

    def _filter_by_threshold(self, documents: List[Document]) -> List[Document]:
        # RRF scores are tiny and not comparable to reranker logits, so only
        # apply the absolute relevance threshold when rerank scores exist.
        if not any(doc.rerank_score is not None for doc in documents):
            return documents

        kept = [
            doc
            for doc in documents
            if (doc.rerank_score if doc.rerank_score is not None else doc.score) >= self._threshold
        ]
        return kept or documents[:1]


class SummaryCompressor(BaseCompressor):
    """
    Ask the LLM to summarise each chunk to `summary_max_tokens` tokens.
    Falls back to returning the original document if the LLM is unavailable.
    """

    def __init__(self, llm: "BaseModel", summary_max_tokens: int = 200) -> None:
        self._llm = llm
        self._summary_max_tokens = summary_max_tokens

    def compress(
        self,
        documents: List[Document],
        query: str,
        token_budget: int,
    ) -> List[Document]:
        total = sum(count_tokens(d.text) for d in documents)
        if total <= token_budget:
            return documents

        compressed: List[Document] = []
        for doc in documents:
            summary = self._summarise(doc.text, query)
            compressed.append(
                Document(
                    text=summary,
                    score=doc.score,
                    metadata=dict(doc.metadata),
                    rerank_score=doc.rerank_score,
                )
            )
        return compressed

    def _summarise(self, text: str, query: str) -> str:
        prompt = (
            f"请用不超过{self._summary_max_tokens}个token对以下内容进行摘要，"
            f"保留与问题「{query}」最相关的信息。\n\n{text}"
        )
        try:
            return self._llm.chat([{"role": "user", "content": prompt}])
        except Exception:
            return text


class ContextCompressor:
    """
    Strategy-selecting compressor facade.
    Reads strategy from CompressionConfig; delegates to the matching implementation.
    """

    def __init__(
        self,
        config: Optional[CompressionConfig] = None,
        llm: Optional["BaseModel"] = None,
    ) -> None:
        self._config = config or CompressionConfig()
        self._llm = llm
        self._relevance = RelevanceFilterCompressor(
            threshold=self._config.relevance_threshold
        )
        self._summary = (
            SummaryCompressor(llm, self._config.summary_max_tokens) if llm else None
        )

    def compress(
        self,
        documents: List[Document],
        query: str,
        token_budget: int,
    ) -> List[Document]:
        strategy = self._config.strategy

        if strategy == "relevance_filter" or self._summary is None:
            return self._relevance.compress(documents, query, token_budget)

        if strategy == "summary":
            return self._summary.compress(documents, query, token_budget)

        # hybrid: filter first, then summarise survivors
        filtered = self._relevance.compress(documents, query, token_budget * 2)
        return self._summary.compress(filtered, query, token_budget)

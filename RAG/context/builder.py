#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ContextBuilder — assembles the final prompt context from ranked documents.

Responsibilities:
  - Parent-child expansion: replace child chunk text with full parent section
  - Exact deduplication: skip identical text
  - Semantic deduplication: skip near-duplicate chunks (Jaccard / cosine)
  - Overlap removal: strip leading overlap carried from previous chunk
  - Token budget: hard cap on total context size
  - Citation list: structured attribution for every included chunk
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Set

import numpy as np

from RAG.chunking import count_tokens, trim_to_tokens
from RAG.schema import Document
from RAG.types import EmbeddingModel


@dataclass
class ContextBuilderConfig:
    max_tokens: int = 3000
    min_score: float = 0.0
    overlap_chars: int = 160
    semantic_dedup_threshold: float = 0.92
    enable_semantic_dedup: bool = True


@dataclass
class BuiltContext:
    context: str
    documents: List[Document]
    citations: List[dict]
    token_count: int


class ContextBuilder:
    def __init__(
        self,
        config: Optional[ContextBuilderConfig] = None,
        embedding_model: Optional[EmbeddingModel] = None,
        parent_map: Optional[dict] = None,
    ) -> None:
        self.config = config or ContextBuilderConfig()
        self.embedding_model = embedding_model
        self.parent_map: dict = parent_map or {}

    def prepare_documents(
        self,
        documents: Sequence[Document],
        resolve_parents: bool = True,
    ) -> List[Document]:
        prepared: List[Document] = []
        selected_effective: List[Document] = []
        seen_hashes: Set[str] = set()
        seen_parent_ids: Set[str] = set()

        for doc in documents:
            effective_score = doc.rerank_score if doc.rerank_score is not None else doc.score
            if effective_score < self.config.min_score:
                continue

            parent_id = doc.metadata.get("parent_id")
            if parent_id and parent_id in seen_parent_ids:
                continue

            raw_text = self._resolve_parent(doc) if resolve_parents else doc.text
            normalised = self._normalise_text(raw_text)
            if normalised in seen_hashes:
                continue

            effective_doc = Document(
                text=raw_text,
                score=doc.score,
                metadata=dict(doc.metadata),
                rerank_score=doc.rerank_score,
            )
            if self._is_semantic_duplicate(effective_doc, selected_effective):
                continue

            prepared.append(effective_doc)
            selected_effective.append(effective_doc)
            seen_hashes.add(normalised)
            if parent_id:
                seen_parent_ids.add(parent_id)

        return prepared

    def build(
        self,
        documents: Sequence[Document],
        resolve_parents: bool = True,
        max_tokens: Optional[int] = None,
    ) -> BuiltContext:
        prepared = self.prepare_documents(documents, resolve_parents=resolve_parents)
        selected: List[Document] = []
        citations: List[dict] = []
        total_tokens = 0
        rendered_parts: List[str] = []
        previous_rendered_text = ""
        token_budget = max_tokens if max_tokens is not None else self.config.max_tokens

        for doc in prepared:
            raw_text = doc.text
            text = self._remove_overlap(raw_text, previous_rendered_text)
            source = doc.metadata.get("source", "unknown")
            chunk_id = doc.metadata.get("chunk_id", "unknown")
            header = f"[source={source} chunk={chunk_id}]"
            available = token_budget - total_tokens - count_tokens(header) - 2
            if available <= 0:
                break
            text = trim_to_tokens(text, available)
            part = f"{header}\n{text}"
            part_tokens = count_tokens(part)
            if part_tokens <= 0:
                continue

            rendered_parts.append(part)
            selected.append(doc)
            previous_rendered_text = text
            total_tokens += part_tokens
            citations.append({
                "source": source,
                "chunk_id": chunk_id,
                "parent_id": doc.metadata.get("parent_id"),
                "section": doc.metadata.get("section"),
                "page": doc.metadata.get("page"),
                "score": doc.score,
                "rerank_score": doc.rerank_score,
            })
            if total_tokens >= token_budget:
                break

        return BuiltContext(
            context="\n\n".join(rendered_parts),
            documents=selected,
            citations=citations,
            token_count=total_tokens,
        )

    def _resolve_parent(self, doc: Document) -> str:
        parent_id = doc.metadata.get("parent_id")
        if parent_id and parent_id in self.parent_map:
            return self.parent_map[parent_id]
        return doc.text

    def _normalise_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip().lower()

    def _remove_overlap(self, text: str, previous_text: str) -> str:
        if not previous_text or self.config.overlap_chars <= 0:
            return text
        tail = previous_text[-self.config.overlap_chars:]
        max_len = min(len(tail), len(text), self.config.overlap_chars)
        for size in range(max_len, 30, -1):
            if text.startswith(tail[-size:]):
                return text[size:].lstrip()
        return text

    def _is_semantic_duplicate(self, doc: Document, selected: Sequence[Document]) -> bool:
        if not self.config.enable_semantic_dedup:
            return False
        return any(
            self._similarity(doc.text, existing.text) >= self.config.semantic_dedup_threshold
            for existing in selected
        )

    def _similarity(self, left: str, right: str) -> float:
        if self.embedding_model is not None:
            lv = np.array(self.embedding_model.get_embedding(left))
            rv = np.array(self.embedding_model.get_embedding(right))
            mag = float(np.linalg.norm(lv) * np.linalg.norm(rv))
            return float(np.dot(lv, rv) / mag) if mag else 0.0
        lt, rt = self._terms(left), self._terms(right)
        if not lt or not rt:
            return 0.0
        return len(lt & rt) / len(lt | rt)

    def _terms(self, text: str) -> set:
        return set(re.findall(r"[a-z0-9_]+|[一-鿿]", text.lower()))

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ChunkMetadata:
    doc_id: str
    source: str
    chunk_id: int
    section: str
    page: Optional[int]
    created_at: str
    token_count: int
    parent_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Document:
    text: str
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    rerank_score: Optional[float] = None

    @property
    def identity(self) -> str:
        stable_id = self.metadata.get("chunk_uid") or self.metadata.get("identity")
        if stable_id:
            return str(stable_id)

        doc_id = self.metadata.get("doc_id", "")
        chunk_id = self.metadata.get("chunk_id", "")
        source = self.metadata.get("source", "")
        parent_id = self.metadata.get("parent_id")
        child_chunk_id = self.metadata.get("child_chunk_id")
        page = self.metadata.get("page")

        if parent_id is not None:
            child_id = child_chunk_id if child_chunk_id is not None else chunk_id
            return f"{doc_id}:{source}:{page}:{parent_id}:{child_id}"
        return f"{doc_id}:{source}:{page}:{chunk_id}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "score": self.score,
            "metadata": self.metadata,
            "rerank_score": self.rerank_score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        return cls(
            text=data["text"],
            score=float(data.get("score", 0.0)),
            metadata=dict(data.get("metadata", {})),
            rerank_score=data.get("rerank_score"),
        )

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MemoryManager — facade over Short-term / Long-term memory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

from RAG.memory.long_term import LongTermMemory, MemoryEntry
from RAG.memory.short_term import ShortTermMemory

if TYPE_CHECKING:
    from RAG.types import EmbeddingModel


@dataclass
class MemoryContext:
    history_messages: List[dict]
    preferences: List[str]


class MemoryManager:

    def __init__(
        self,
        db_path: Optional[str] = "storage/ltm",
        embedding_model: Optional["EmbeddingModel"] = None,
        short_term_max_turns: int = 5,
        short_term_max_tokens: int = 1500,
        ltm_similarity_threshold: float = 0.85,
    ) -> None:
        self.short_term = ShortTermMemory(
            max_turns=short_term_max_turns,
            max_tokens=short_term_max_tokens,
        )
        if db_path is None:
            self.long_term = None
        else:
            os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
            self.long_term = LongTermMemory(
                db_path=db_path,
                embedding_model=embedding_model,
                similarity_threshold=ltm_similarity_threshold,
            )

    # ── Query hooks ────────────────────────────────────────────────────

    def before_query(self, query: str) -> MemoryContext:
        if self.long_term is not None:
            prefs = self.long_term.search(query, k=5, entry_type="preference")
        else:
            prefs = []
        return MemoryContext(
            history_messages=self.short_term.to_messages(),
            preferences=[p.content for p in prefs],
        )

    def after_query(self, query: str, context: str, answer: str) -> None:
        self.short_term.add(query, answer, context=context)

    # ── Long-term ops ──────────────────────────────────────────────────

    def save(self, content: str, entry_type: str = "note") -> str:
        if self.long_term is None:
            return ""
        return self.long_term.save(content, entry_type)

    def forget(self, entry_id: str) -> None:
        if self.long_term is None:
            return
        self.long_term.forget(entry_id)

    def list_all(self, entry_type: Optional[str] = None) -> List[MemoryEntry]:
        if self.long_term is None:
            return []
        return self.long_term.list_all(entry_type)

    def expire_old(self, days: int = 90) -> int:
        if self.long_term is None:
            return 0
        return self.long_term.expire_old(days)

    def close(self) -> None:
        if self.long_term is not None:
            self.long_term.close()

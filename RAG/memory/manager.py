#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MemoryManager — unified facade over Short-term / Working / Long-term memory.

LongTermMemory is SQLite-backed; no manual save/load required.
WorkingMemory requires an explicit new_session() call before use.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

from RAG.memory.long_term import LongTermMemory, MemoryEntry
from RAG.memory.short_term import ShortTermMemory
from RAG.memory.working import WorkingMemory

if TYPE_CHECKING:
    from RAG.index.base import VectorIndex
    from RAG.types import EmbeddingModel


@dataclass
class MemoryContext:
    """Snapshot of all memory layers captured before a pipeline run."""
    ltm_hit: Optional[MemoryEntry]
    working_context: str
    history_messages: List[dict]

    @property
    def has_ltm_hit(self) -> bool:
        return self.ltm_hit is not None


class MemoryManager:
    """
    Unified entry point for Short-term / Working / Long-term memory.

    Usage:
        mem = MemoryManager(db_path="storage/ltm.db", index=faiss_idx, embedding_model=emb)
        mem.working.new_session()

        ctx = mem.before_query(query)
        if ctx.has_ltm_hit:
            answer = ctx.ltm_hit.answer      # pipeline bypass
        else:
            answer = pipeline_result
            mem.after_query(query, context, answer)
            # Later, user can call mem.save_last_turn() to persist to LTM
            # or mem.save_note("some insight") to store free-form text
    """

    def __init__(
        self,
        db_path: str = "storage/ltm.db",
        index: Optional["VectorIndex"] = None,
        embedding_model: Optional["EmbeddingModel"] = None,
        short_term_max_turns: int = 5,
        short_term_max_tokens: int = 1500,
        ltm_similarity_threshold: float = 0.92,
    ) -> None:
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

        if index is None:
            from RAG.index.faiss_index import FaissVectorIndex
            index = FaissVectorIndex(dimension=768)

        self.short_term = ShortTermMemory(
            max_turns=short_term_max_turns,
            max_tokens=short_term_max_tokens,
        )
        self.working = WorkingMemory()
        self.long_term = LongTermMemory(
            db_path=db_path,
            index=index,
            embedding_model=embedding_model,
            similarity_threshold=ltm_similarity_threshold,
        )

    # ── Query hooks ────────────────────────────────────────────────────

    def before_query(self, query: str) -> MemoryContext:
        return MemoryContext(
            ltm_hit=self.long_term.search(query, verified_only=True),
            working_context=self.working.to_context_str() if self.working.active else "",
            history_messages=self.short_term.to_messages(),
        )

    def after_query(self, query: str, context: str, answer: str) -> None:
        """Update short-term and working memory (LTM requires explicit save)."""
        self.short_term.add(query, answer, context=context)
        if self.working.active:
            self.working.retrieval.increment()

    def save_last_turn(self) -> Optional[str]:
        """
        Persist the most recent Q&A turn to long-term memory.
        Skips if similar content already exists (dedup via embedding search).
        Returns entry_id or None.
        """
        turn = self.short_term.last_turn
        if turn is None:
            return None
        existing = self.long_term.search(turn.query, k=1, verified_only=False)
        if existing is not None:
            return None
        return self.long_term.add(turn.query, turn.answer, turn.context, verified=True)

    def save_note(self, content: str) -> Optional[str]:
        """
        Persist a free-form note/insight to long-term memory.
        Skips if similar content already exists.
        Returns entry_id or None.
        """
        existing = self.long_term.search(content, k=1, verified_only=False)
        if existing is not None:
            return None
        return self.long_term.add_note(content)

    # ── Pass-throughs ──────────────────────────────────────────────────

    def verify(self, entry_id: str) -> bool:
        """Promote an LTM entry to trusted (persists immediately via SQLite)."""
        return self.long_term.verify(entry_id)

    def close(self) -> None:
        self.long_term.close()

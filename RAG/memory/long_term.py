#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LongTermMemory — SQLite for structured metadata + VectorIndex for embeddings.

Design:
  - SQLite stores every MemoryEntry field (query, answer, context, verified, …)
  - VectorIndex stores query embeddings, keyed by an integer faiss_id
  - faiss_id is the SQLite rowid, making the two stores trivially linkable
  - Only verified entries are used for pipeline bypass (search verified_only=True)
  - Lifecycle: expire_old() removes entries older than N days
"""

from __future__ import annotations

import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from RAG.index.base import VectorIndex
    from RAG.types import EmbeddingModel


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS memory_entries (
    id          TEXT PRIMARY KEY,
    faiss_id    INTEGER,
    query       TEXT NOT NULL,
    answer      TEXT NOT NULL,
    context     TEXT NOT NULL,
    source      TEXT DEFAULT '',
    verified    INTEGER DEFAULT 0,
    created_at  TEXT NOT NULL,
    verified_at TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_faiss_id
    ON memory_entries (faiss_id)
    WHERE faiss_id IS NOT NULL;
"""


@dataclass
class MemoryEntry:
    id: str
    query: str
    answer: str
    context: str
    source: str
    verified: bool
    created_at: str
    verified_at: Optional[str]
    embedding_ref: int          # faiss_id / SQLite rowid


class LongTermMemory:
    """
    Production-grade long-term memory backed by SQLite + VectorIndex.

    Thread safety: SQLite connections are not thread-safe by default;
    use check_same_thread=False only in single-writer scenarios.
    """

    def __init__(
        self,
        db_path: str,
        index: "VectorIndex",
        embedding_model: Optional["EmbeddingModel"] = None,
        similarity_threshold: float = 0.92,
    ) -> None:
        self._db_path = db_path
        self._index = index
        self._embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold

        # Persist FAISS index alongside the DB so faiss_ids survive restarts
        self._index_path = os.path.splitext(db_path)[0] + "_index" if db_path != ":memory:" else None
        if self._index_path and os.path.exists(self._index_path):
            self._index.load(self._index_path)

        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_CREATE_TABLE)
        self._conn.commit()

    # ── Write ──────────────────────────────────────────────────────────

    def add(
        self,
        query: str,
        answer: str,
        context: str,
        source: str = "",
        verified: bool = False,
    ) -> str:
        """Store a Q&A pair. Returns entry id. verified=True for user-saved entries."""
        entry_id = uuid.uuid4().hex[:12]
        created_at = datetime.now(timezone.utc).isoformat()

        # Insert with faiss_id=NULL; update after embedding
        cur = self._conn.execute(
            "INSERT INTO memory_entries (id, query, answer, context, source, verified, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (entry_id, query, answer, context, source, int(verified), created_at),
        )
        self._conn.commit()

        # Add embedding to VectorIndex; leave faiss_id NULL when no model
        vector = self._embed(query)
        if vector is not None:
            self._index.add(np.array([vector], dtype=np.float32))
            faiss_id = self._index.size - 1
            self._conn.execute(
                "UPDATE memory_entries SET faiss_id=? WHERE id=?",
                (faiss_id, entry_id),
            )
            self._conn.commit()
            if self._index_path:
                self._index.save(self._index_path)
        return entry_id

    def add_note(self, content: str) -> str:
        """Store free-form user note. Always verified=1."""
        entry_id = uuid.uuid4().hex[:12]
        created_at = datetime.now(timezone.utc).isoformat()

        cur = self._conn.execute(
            "INSERT INTO memory_entries (id, query, answer, context, source, verified, created_at) "
            "VALUES (?, ?, ?, ?, ?, 1, ?)",
            (entry_id, content, content, "", "user_note", created_at),
        )
        self._conn.commit()

        vector = self._embed(content)
        if vector is not None:
            self._index.add(np.array([vector], dtype=np.float32))
            faiss_id = self._index.size - 1
            self._conn.execute(
                "UPDATE memory_entries SET faiss_id=? WHERE id=?",
                (faiss_id, entry_id),
            )
            self._conn.commit()
            if self._index_path:
                self._index.save(self._index_path)
        return entry_id

    def verify(self, entry_id: str) -> bool:
        """Manually promote an entry to trusted. Returns False if not found."""
        now = datetime.now(timezone.utc).isoformat()
        cur = self._conn.execute(
            "UPDATE memory_entries SET verified=1, verified_at=? WHERE id=?",
            (now, entry_id),
        )
        self._conn.commit()
        return cur.rowcount > 0

    # ── Read ───────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        k: int = 5,
        verified_only: bool = True,
    ) -> Optional[MemoryEntry]:
        """
        Return the single best-matching entry above similarity_threshold, or None.
        verified_only=True (default) ensures only trusted answers bypass the pipeline.
        """
        vector = self._embed(query)
        if vector is None or self._index.size == 0:
            return None

        results = self._index.search(np.array(vector, dtype=np.float32), k=k)
        for sr in results:
            if sr.score < self.similarity_threshold:
                break
            entry = self._fetch_by_faiss_id(sr.index, verified_only=verified_only)
            if entry is not None:
                return entry
        return None

    def list_unverified(self) -> List[MemoryEntry]:
        rows = self._conn.execute(
            "SELECT * FROM memory_entries WHERE verified=0 ORDER BY created_at DESC"
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def list_verified(self) -> List[MemoryEntry]:
        rows = self._conn.execute(
            "SELECT * FROM memory_entries WHERE verified=1 ORDER BY verified_at DESC"
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    # ── Lifecycle ──────────────────────────────────────────────────────

    def expire_old(self, days: int = 90) -> int:
        """Delete unverified entries older than `days`. Returns count removed."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        cur = self._conn.execute(
            "DELETE FROM memory_entries WHERE verified=0 AND created_at < ?",
            (cutoff,),
        )
        self._conn.commit()
        return cur.rowcount

    def close(self) -> None:
        self._conn.close()

    # ── Internal ───────────────────────────────────────────────────────

    def _embed(self, text: str) -> Optional[List[float]]:
        if self._embedding_model is None:
            return None
        return self._embedding_model.get_embedding(text)

    def _fetch_by_faiss_id(
        self, faiss_id: int, verified_only: bool
    ) -> Optional[MemoryEntry]:
        q = "SELECT * FROM memory_entries WHERE faiss_id=?"
        if verified_only:
            q += " AND verified=1"
        row = self._conn.execute(q, (faiss_id,)).fetchone()
        return self._row_to_entry(row) if row else None

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> MemoryEntry:
        return MemoryEntry(
            id=row["id"],
            query=row["query"],
            answer=row["answer"],
            context=row["context"],
            source=row["source"],
            verified=bool(row["verified"]),
            created_at=row["created_at"],
            verified_at=row["verified_at"],
            embedding_ref=row["faiss_id"],
        )

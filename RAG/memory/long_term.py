#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LongTermMemory — Chroma-backed persistent memory.

Stores notes and user preferences in a Chroma collection.
Native support for insert, upsert, delete, and metadata filtering.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional, TYPE_CHECKING

import chromadb

if TYPE_CHECKING:
    from RAG.types import EmbeddingModel


@dataclass
class MemoryEntry:
    id: str
    content: str
    entry_type: str          # 'note' | 'preference'
    created_at: str
    updated_at: str


class LongTermMemory:

    def __init__(
        self,
        db_path: str = "storage/ltm",
        embedding_model: Optional["EmbeddingModel"] = None,
        similarity_threshold: float = 0.85,
    ) -> None:
        self._similarity_threshold = similarity_threshold
        self._embedding_model = embedding_model
        self._client = chromadb.PersistentClient(path=db_path)
        self._collection = self._client.get_or_create_collection(
            name="ltm_entries",
            metadata={"hnsw:space": "cosine"},
        )

    # ── Write ──────────────────────────────────────────────────────────

    def save(self, content: str, entry_type: str = "note") -> str:
        """Insert or overwrite. Returns entry id."""
        now = datetime.now(timezone.utc).isoformat()
        existing = self._find_similar(content, entry_type)
        if existing is not None:
            entry_id = existing.id
            embedding = self._embed(content)
            self._collection.update(
                ids=[entry_id],
                documents=[content],
                embeddings=[embedding] if embedding else None,
                metadatas=[{"entry_type": entry_type, "created_at": existing.created_at, "updated_at": now}],
            )
            return entry_id

        entry_id = uuid.uuid4().hex[:12]
        embedding = self._embed(content)
        self._collection.add(
            ids=[entry_id],
            documents=[content],
            embeddings=[embedding] if embedding else None,
            metadatas=[{"entry_type": entry_type, "created_at": now, "updated_at": now}],
        )
        return entry_id

    # ── Read ───────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        k: int = 5,
        entry_type: Optional[str] = None,
    ) -> List[MemoryEntry]:
        """Semantic search. Optionally filter by entry_type."""
        where = {"entry_type": entry_type} if entry_type else None
        embedding = self._embed(query)
        if embedding is None:
            return []
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        entries: List[MemoryEntry] = []
        if not results["ids"] or not results["ids"][0]:
            return entries
        for i, eid in enumerate(results["ids"][0]):
            distance = results["distances"][0][i] if results.get("distances") else 0.0
            similarity = 1.0 - float(distance)  # Chroma 返回余弦距离 → 转相似度
            if similarity < self._similarity_threshold:
                continue
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            entries.append(MemoryEntry(
                id=eid,
                content=results["documents"][0][i],
                entry_type=meta.get("entry_type", ""),
                created_at=meta.get("created_at", ""),
                updated_at=meta.get("updated_at", ""),
            ))
        return entries

    def list_all(self, entry_type: Optional[str] = None) -> List[MemoryEntry]:
        """Return all entries, optionally filtered by type."""
        where = {"entry_type": entry_type} if entry_type else None
        try:
            results = self._collection.get(
                where=where,
                include=["documents", "metadatas"],
            )
        except Exception:
            return []
        if not results["ids"]:
            return []
        entries: List[MemoryEntry] = []
        for i, eid in enumerate(results["ids"]):
            meta = results["metadatas"][i] if results["metadatas"] else {}
            entries.append(MemoryEntry(
                id=eid,
                content=results["documents"][i] if results["documents"] else "",
                entry_type=meta.get("entry_type", ""),
                created_at=meta.get("created_at", ""),
                updated_at=meta.get("updated_at", ""),
            ))
        return entries

    # ── Delete ─────────────────────────────────────────────────────────

    def forget(self, entry_id: str) -> None:
        self._collection.delete(ids=[entry_id])

    def expire_old(self, days: int = 90) -> int:
        """Delete entries not updated for `days`. Returns count removed."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        old_ids: List[str] = []
        for entry in self.list_all():
            if entry.updated_at < cutoff:
                old_ids.append(entry.id)
        if old_ids:
            self._collection.delete(ids=old_ids)
        return len(old_ids)

    def close(self) -> None:
        pass

    # ── Internal ───────────────────────────────────────────────────────

    def _embed(self, text: str) -> Optional[List[float]]:
        if self._embedding_model is None:
            return None
        return self._embedding_model.get_embedding(text)

    def _find_similar(self, content: str, entry_type: Optional[str] = None) -> Optional[MemoryEntry]:
        results = self.search(content, k=1, entry_type=entry_type)
        return results[0] if results else None

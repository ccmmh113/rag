#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WorkingMemory — session-scoped facade over three typed sub-states.

  retrieval  → RetrievalState  (what was retrieved, strategies tried)
  session    → SessionState    (user KV context)
  planning   → PlanningState   (goal / step tracking)

Must call new_session() before any operation.
Any access on an inactive memory raises RuntimeError.
"""

from __future__ import annotations

import uuid
from typing import Optional

from RAG.memory.working.planning_state import PlanningState
from RAG.memory.working.retrieval_state import RetrievalState
from RAG.memory.working.session_state import SessionState


class WorkingMemory:
    """
    Session-scoped working memory with strict state separation.
    No catch-all dict — every concern has its own typed state object.
    """

    def __init__(self) -> None:
        self._retrieval: Optional[RetrievalState] = None
        self._session: Optional[SessionState] = None
        self._planning: Optional[PlanningState] = None

    # ── Session lifecycle ──────────────────────────────────────────────

    def new_session(self, session_id: Optional[str] = None) -> str:
        sid = session_id or uuid.uuid4().hex[:8]
        self._retrieval = RetrievalState()
        self._session = SessionState(session_id=sid)
        self._planning = PlanningState()
        return sid

    def end_session(self) -> None:
        self._retrieval = None
        self._session = None
        self._planning = None

    @property
    def active(self) -> bool:
        return self._session is not None

    @property
    def session_id(self) -> Optional[str]:
        return self._session.session_id if self._session else None

    # ── Sub-state accessors (raise if inactive) ────────────────────────

    @property
    def retrieval(self) -> RetrievalState:
        self._require_session()
        return self._retrieval  # type: ignore[return-value]

    @property
    def session(self) -> SessionState:
        self._require_session()
        return self._session  # type: ignore[return-value]

    @property
    def planning(self) -> PlanningState:
        self._require_session()
        return self._planning  # type: ignore[return-value]

    # ── Prompt serialisation ──────────────────────────────────────────

    def to_context_str(self) -> str:
        if not self.active:
            return ""
        parts = [self._session.to_str()]  # type: ignore[union-attr]
        plan_str = self._planning.to_str()  # type: ignore[union-attr]
        if plan_str:
            parts.append(plan_str)
        parts.append(
            f"已检索文档数: {len(self._retrieval.retrieved_doc_ids)}"  # type: ignore[union-attr]
        )
        tried = self._retrieval.tried_strategies  # type: ignore[union-attr]
        if tried:
            parts.append(f"已尝试策略: {', '.join(tried)}")
        return "\n".join(parts)

    # ── Internal ───────────────────────────────────────────────────────

    def _require_session(self) -> None:
        if self._session is None:
            raise RuntimeError("No active session. Call new_session() first.")

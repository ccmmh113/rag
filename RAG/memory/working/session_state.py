#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SessionState — user-defined key-value context for the current session.
Replaces the ad-hoc dict that accumulated in the old WorkingMemory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict


@dataclass
class SessionState:
    """
    Structured session metadata.
    Values are user-set (e.g. current_project, user_role).
    Do NOT store retrieval bookkeeping here; that belongs in RetrievalState.
    """
    session_id: str
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    context: Dict[str, Any] = field(default_factory=dict)

    def set(self, key: str, value: Any) -> None:
        self.context[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.context.get(key, default)

    def to_str(self) -> str:
        if not self.context:
            return f"Session: {self.session_id}"
        lines = [f"Session: {self.session_id}"]
        lines += [f"  {k}: {v}" for k, v in self.context.items()]
        return "\n".join(lines)

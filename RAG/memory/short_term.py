#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
except ModuleNotFoundError:
    _enc = None


def _count_tokens(text: str) -> int:
    if _enc is None:
        return len(text) // 4
    return len(_enc.encode(text))


@dataclass
class QATurn:
    query: str
    answer: str
    context: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ShortTermMemory:
    """
    Recent-turn conversation buffer with a token budget.
    Oldest turns are evicted automatically when budget is exceeded.
    Feeds directly into the LLM prompt as message history.
    """

    def __init__(self, max_turns: int = 5, max_tokens: int = 1500) -> None:
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self._turns: List[QATurn] = []

    def add(self, query: str, answer: str, context: str = "") -> None:
        self._turns.append(QATurn(query=query, answer=answer, context=context))
        self._evict()

    def to_messages(self) -> List[dict]:
        """Return as LLM-compatible message list."""
        messages = []
        for turn in self._turns:
            messages.append({"role": "user", "content": turn.query})
            messages.append({"role": "assistant", "content": turn.answer})
        return messages

    @property
    def token_count(self) -> int:
        return sum(_count_tokens(t.query) + _count_tokens(t.answer) for t in self._turns)

    @property
    def last_turn(self) -> Optional[QATurn]:
        return self._turns[-1] if self._turns else None

    def clear(self) -> None:
        self._turns.clear()

    def __len__(self) -> int:
        return len(self._turns)

    def _evict(self) -> None:
        while len(self._turns) > self.max_turns:
            self._turns.pop(0)
        while self._turns and self.token_count > self.max_tokens:
            self._turns.pop(0)

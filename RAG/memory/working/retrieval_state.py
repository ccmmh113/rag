#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RetrievalState — tracks what was retrieved and which strategies were tried
within the current session. Fed directly to PolicyRouter to avoid repeating
exhausted strategies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Set


@dataclass
class RetrievalState:
    """
    Session-scoped retrieval bookkeeping.
    Reset on every new_session() call.
    """
    retrieved_doc_ids: Set[str] = field(default_factory=set)
    tried_strategies: List[str] = field(default_factory=list)
    query_count: int = 0

    def add_retrieved(self, identity: str) -> None:
        self.retrieved_doc_ids.add(identity)

    def add_tried_strategy(self, strategy: str) -> None:
        if strategy not in self.tried_strategies:
            self.tried_strategies.append(strategy)

    def has_tried(self, strategy: str) -> bool:
        return strategy in self.tried_strategies

    def has_retrieved(self, identity: str) -> bool:
        return identity in self.retrieved_doc_ids

    def increment(self) -> None:
        self.query_count += 1

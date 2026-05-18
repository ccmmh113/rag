#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Base classes for the policy-based retrieval router.
Each policy is responsible for scoring how suitable it is for a given query.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class PolicyScore:
    """Score output from a single policy."""
    policy_name: str
    score: float            # 0.0 – 1.0, higher = more suitable
    dense_weight: float
    sparse_weight: float
    reason: str


@dataclass
class RouteDecision:
    """Final routing decision selected by PolicyRouter."""
    strategy: str           # winning policy name
    dense_weight: float
    sparse_weight: float
    reason: str
    all_scores: list        # List[PolicyScore] — kept for trace


class BaseRetrievalPolicy(ABC):
    """A retrieval policy scores how well it suits the current query."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique policy identifier."""

    @abstractmethod
    def score(self, query: str) -> PolicyScore:
        """Return a PolicyScore for this query. Must NOT raise; return score=0.0 on any error."""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Base classes for the policy-based retrieval router.
Each policy is responsible for scoring how suitable it is for a given query.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from RAG.memory.working import WorkingMemory


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
    strategy: str           # winning policy name, recorded in WorkingMemory
    dense_weight: float
    sparse_weight: float
    reason: str
    all_scores: list        # List[PolicyScore] — kept for trace


class BaseRetrievalPolicy(ABC):
    """
    A retrieval policy scores how well it suits the current query.
    Policies are stateless; session state is passed in explicitly.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique policy identifier (also used as strategy key in WorkingMemory)."""

    @abstractmethod
    def score(self, query: str, working_memory: "WorkingMemory") -> PolicyScore:
        """
        Return a PolicyScore for this query.
        Must NOT raise; return score=0.0 on any error.
        """

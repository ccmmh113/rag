#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Policy-based retrieval router.
Aggregates scores from all registered policies and picks the best
untried strategy. Supports dynamic policy registration and fallback.
"""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

from RAG.router.base import BaseRetrievalPolicy, PolicyScore, RouteDecision
from RAG.router.policies import HybridPolicy

if TYPE_CHECKING:
    from RAG.memory.working import WorkingMemory


class PolicyRouter:
    """
    Scoring + voting retrieval router.

    Decision process:
      1. Ask every registered policy for a score.
      2. Filter out policies already tried in this session.
      3. Pick the highest-scoring remaining policy.
      4. Fall back to HybridPolicy if all others are exhausted.

    Policies can be added at runtime via register(), allowing
    project-specific strategies without touching core code.
    """

    def __init__(
        self,
        policies: Optional[List[BaseRetrievalPolicy]] = None,
        fallback: Optional[BaseRetrievalPolicy] = None,
    ) -> None:
        self._policies: List[BaseRetrievalPolicy] = list(policies or [])
        self._fallback = fallback or HybridPolicy()

    # ── Public API ─────────────────────────────────────────────────────

    def route(self, query: str, working_memory: "WorkingMemory") -> RouteDecision:
        """
        Select a retrieval strategy for this query.
        Records the chosen strategy into working_memory.retrieval.
        """
        tried = set(working_memory.retrieval.tried_strategies) if working_memory.active else set()

        all_scores: List[PolicyScore] = [
            p.score(query, working_memory) for p in self._policies
        ]

        # Filter already-tried, then pick highest score
        candidates = [s for s in all_scores if s.policy_name not in tried]
        chosen = max(candidates, key=lambda s: s.score) if candidates else None

        if chosen is None:
            # All policies exhausted — reset via fallback
            fb = self._fallback.score(query, working_memory)
            chosen = fb
            all_scores.append(fb)

        if working_memory.active:
            working_memory.retrieval.add_tried_strategy(chosen.policy_name)

        return RouteDecision(
            strategy=chosen.policy_name,
            dense_weight=chosen.dense_weight,
            sparse_weight=chosen.sparse_weight,
            reason=chosen.reason,
            all_scores=all_scores,
        )

    def register(self, policy: BaseRetrievalPolicy) -> None:
        """Dynamically add a new retrieval policy."""
        self._policies.append(policy)

    def policy_names(self) -> List[str]:
        return [p.name for p in self._policies]

    # ── Factory ────────────────────────────────────────────────────────

    @classmethod
    def default(cls) -> "PolicyRouter":
        """Pre-configured router with all built-in policies."""
        from RAG.router.policies import CodePolicy, DensePolicy, HybridPolicy, SparsePolicy
        return cls(
            policies=[CodePolicy(), DensePolicy(), SparsePolicy(), HybridPolicy()],
            fallback=HybridPolicy(),
        )

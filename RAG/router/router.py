#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Policy-based retrieval router.
Aggregates scores from all registered policies and picks the best one.
"""

from __future__ import annotations

from typing import List, Optional

from RAG.router.base import BaseRetrievalPolicy, PolicyScore, RouteDecision
from RAG.router.policies import HybridPolicy


class PolicyRouter:

    def __init__(
        self,
        policies: Optional[List[BaseRetrievalPolicy]] = None,
        fallback: Optional[BaseRetrievalPolicy] = None,
    ) -> None:
        self._policies: List[BaseRetrievalPolicy] = list(policies or [])
        self._fallback = fallback or HybridPolicy()

    def route(self, query: str) -> RouteDecision:
        all_scores: List[PolicyScore] = [
            p.score(query) for p in self._policies
        ]
        chosen = max(all_scores, key=lambda s: s.score) if all_scores else None

        if chosen is None:
            fb = self._fallback.score(query)
            chosen = fb
            all_scores.append(fb)

        return RouteDecision(
            strategy=chosen.policy_name,
            dense_weight=chosen.dense_weight,
            sparse_weight=chosen.sparse_weight,
            reason=chosen.reason,
            all_scores=all_scores,
        )

    def register(self, policy: BaseRetrievalPolicy) -> None:
        self._policies.append(policy)

    def policy_names(self) -> List[str]:
        return [p.name for p in self._policies]

    @classmethod
    def default(cls) -> "PolicyRouter":
        from RAG.router.policies import CodePolicy, DensePolicy, HybridPolicy, SparsePolicy
        return cls(
            policies=[CodePolicy(), DensePolicy(), SparsePolicy(), HybridPolicy()],
            fallback=HybridPolicy(),
        )

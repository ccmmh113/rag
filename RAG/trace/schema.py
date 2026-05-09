#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified QueryTrace schema.
Every field is optional beyond the identity fields so partial traces
(e.g. LTM cache hits that skip retrieval) remain valid.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class QueryTrace:
    """
    One trace per user query. Append-only; never mutated after save().

    Identity
    --------
    trace_id, timestamp, query

    Routing
    -------
    route               winning policy name
    route_policy_scores {policy_name: score} for all evaluated policies

    Retrieval
    ---------
    retrieval_latency   ms from query to merged result list
    recalled_count      docs returned by HybridRetriever
    recalled_docs       compact summaries (capped, not full text)

    Reranking
    ---------
    rerank_latency      ms
    reranked_count      docs after reranking
    rerank_gain         MRR delta vs pre-rerank order (optional)

    Context
    -------
    prompt_tokens       tokens in the final context string
    citations           source attribution list

    Cache
    -----
    cache_hit           True if result came from LTM without retrieval

    Generation
    ----------
    generation_latency  ms
    answer_preview      first 200 chars of LLM response

    Reflection
    ----------
    reflection_retries  number of reflection-triggered retries
    quality_score       final ReflectionEngine quality estimate

    Extensions
    ----------
    metadata            arbitrary key-value for future fields
    """

    # Identity
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: str = field(default_factory=_now)
    query: str = ""

    # Routing
    route: Optional[str] = None
    route_policy_scores: Dict[str, float] = field(default_factory=dict)

    # Retrieval
    retrieval_latency: float = 0.0
    recalled_count: int = 0
    recalled_docs: List[Dict[str, Any]] = field(default_factory=list)

    # Reranking
    rerank_latency: float = 0.0
    reranked_count: int = 0
    rerank_gain: Optional[float] = None

    # Context
    prompt_tokens: int = 0
    prompt_messages: List[Dict[str, Any]] = field(default_factory=list)
    context_preview: Optional[str] = None
    citations: List[Dict[str, Any]] = field(default_factory=list)

    # Cache
    cache_hit: bool = False

    # Generation (filled by caller after LLM responds)
    generation_latency: float = 0.0
    answer_preview: Optional[str] = None

    # Reflection
    reflection_retries: int = 0
    quality_score: Optional[float] = None

    # Extensions
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ── Helpers ────────────────────────────────────────────────────────

    def complete_generation(
        self,
        answer: str,
        latency_ms: float,
        quality_score: Optional[float] = None,
    ) -> "QueryTrace":
        """Call after LLM responds to fill generation fields."""
        self.answer_preview = answer[:200]
        self.generation_latency = round(latency_ms, 1)
        if quality_score is not None:
            self.quality_score = quality_score
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

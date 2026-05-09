#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAGAS-aligned metric dataclasses.

A. Retrieval:  Context Precision, Context Recall
B. Generation: Faithfulness, Answer Relevancy
C. End-to-End: Answer Correctness
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class RetrievalMetrics:
    context_precision: float       # avg precision: relevant chunks ranked high?
    context_recall: float          # fraction of all relevant chunks retrieved
    hit_at_k: float                # fraction of queries with ≥1 relevant doc in top-K
    mrr: float                     # mean reciprocal rank of first relevant doc
    avg_latency_ms: float
    k: int
    total: int
    details: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationMetrics:
    faithfulness: float             # fraction of claims in answer supported by context
    hallucination_rate: float       # 1 - faithfulness
    answer_relevancy: float         # semantic similarity of question to reverse-generated questions
    total: int
    details: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EndToEndMetrics:
    answer_correctness: float       # factual + semantic agreement with ground_truth
    total: int
    details: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SystemMetrics:
    avg_retrieval_latency_ms: float
    avg_generation_latency_ms: float
    cache_hit_rate: float
    total_traces: int

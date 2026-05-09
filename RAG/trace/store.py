#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TraceStore — append-only JSONL trace persistence.
One JSON object per line; safe for concurrent readers, single-writer append.
Supports replay (load_all), recent-N queries, and filter helpers.
"""

from __future__ import annotations

import json
import os
from typing import Callable, List, Optional

from RAG.trace.schema import QueryTrace


class TraceStore:
    """
    Append-only JSONL store.

    Write:  store.append(trace)                    — one line per call
    Read:   store.load_all()                       — full history
            store.load_recent(n)                   — last N traces
            store.filter(lambda t: t.cache_hit)    — predicate filter

    Replay / debug: each line is self-contained JSON, grep-friendly.
    """

    def __init__(self, path: str = "storage/traces.jsonl") -> None:
        self.path = path

    def append(self, trace: QueryTrace) -> None:
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")

    def load_all(self) -> List[QueryTrace]:
        if not os.path.exists(self.path):
            return []
        traces = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        traces.append(QueryTrace(**json.loads(line)))
                    except (TypeError, KeyError):
                        pass    # skip malformed lines
        return traces

    def load_recent(self, n: int = 20) -> List[QueryTrace]:
        return self.load_all()[-n:]

    def filter(self, predicate: Callable[[QueryTrace], bool]) -> List[QueryTrace]:
        return [t for t in self.load_all() if predicate(t)]

    # ── Analysis helpers ───────────────────────────────────────────────

    def cache_hit_rate(self) -> float:
        traces = self.load_all()
        if not traces:
            return 0.0
        return sum(1 for t in traces if t.cache_hit) / len(traces)

    def avg_latency(self) -> dict:
        traces = [t for t in self.load_all() if not t.cache_hit]
        if not traces:
            return {}
        return {
            "retrieval_ms": sum(t.retrieval_latency for t in traces) / len(traces),
            "rerank_ms": sum(t.rerank_latency for t in traces) / len(traces),
            "generation_ms": sum(t.generation_latency for t in traces) / len(traces),
        }

    def strategy_distribution(self) -> dict:
        from collections import Counter
        return dict(Counter(t.route for t in self.load_all() if t.route))

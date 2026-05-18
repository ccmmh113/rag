#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Concrete retrieval policies.
Each policy independently scores query suitability (0–1).
PolicyRouter picks the highest-scoring policy.
"""

from __future__ import annotations

import re

from RAG.router.base import BaseRetrievalPolicy, PolicyScore


_QUESTION_RE = re.compile(
    r"什么|怎么|为什么|如何|是否|哪些?|when|what|why|how|which|is\b|are\b|does\b",
    re.IGNORECASE,
)
_CODE_RE = re.compile(r"[A-Z_]{2,}|def |class |import |function\b|->|::|#include")
_CHINESE_RE = re.compile(r"[一-鿿]")


def _token_count(query: str) -> int:
    return len(re.findall(r"[a-z0-9_]+|[一-鿿]", query.lower()))


def _has_question_word(query: str) -> bool:
    return bool(_QUESTION_RE.search(query))


def _has_code_term(query: str) -> bool:
    return bool(_CODE_RE.search(query))


def _is_short(query: str, threshold: int = 5) -> bool:
    return _token_count(query) <= threshold


class DensePolicy(BaseRetrievalPolicy):

    @property
    def name(self) -> str:
        return "dense_heavy"

    def score(self, query: str) -> PolicyScore:
        s = 0.0
        if _has_question_word(query):
            s += 0.5
        if not _is_short(query):
            s += 0.3
        if _CHINESE_RE.search(query):
            s += 0.2
        return PolicyScore(
            policy_name=self.name,
            score=min(s, 1.0),
            dense_weight=0.7,
            sparse_weight=0.3,
            reason="语义型问句，Dense 权重加重",
        )


class SparsePolicy(BaseRetrievalPolicy):

    @property
    def name(self) -> str:
        return "sparse_heavy"

    def score(self, query: str) -> PolicyScore:
        s = 0.0
        if _is_short(query):
            s += 0.5
        if not _has_question_word(query):
            s += 0.3
        if not _CHINESE_RE.search(query):
            s += 0.2
        return PolicyScore(
            policy_name=self.name,
            score=min(s, 1.0),
            dense_weight=0.3,
            sparse_weight=0.7,
            reason="短关键词查询，BM25 权重加重",
        )


class CodePolicy(BaseRetrievalPolicy):

    @property
    def name(self) -> str:
        return "code_sparse"

    def score(self, query: str) -> PolicyScore:
        s = 0.8 if _has_code_term(query) else 0.0
        return PolicyScore(
            policy_name=self.name,
            score=s,
            dense_weight=0.2,
            sparse_weight=0.8,
            reason="代码标识符查询，BM25 精确匹配",
        )


class HybridPolicy(BaseRetrievalPolicy):
    """Balanced fallback — always applicable, moderate score."""

    @property
    def name(self) -> str:
        return "balanced"

    def score(self, query: str) -> PolicyScore:
        return PolicyScore(
            policy_name=self.name,
            score=0.35,
            dense_weight=0.5,
            sparse_weight=0.5,
            reason="均衡混合检索（兜底）",
        )

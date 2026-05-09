#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ReflectionEngine — post-generation quality check with optional retry.

Loop:
  answer = llm.chat(prompt)
  result = engine.reflect(query, context, answer)
  if result.should_retry:
      answer = llm.chat(improved_query_prompt)   ← max_retries times

Decoupled: ReflectionEngine only needs an LLM; it has no dependency
on the retrieval or memory subsystems.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from RAG.LLM import BaseModel


@dataclass
class ReflectionResult:
    should_retry: bool
    reason: str
    improved_query: Optional[str]   # None when should_retry is False
    quality_score: float            # 0–1, used in trace


_REFLECTION_PROMPT = """\
你是一个RAG质量评估员。请评估以下回答是否充分回答了问题，只输出JSON，不要任何解释。

问题: {query}
检索上下文（摘要）: {context_preview}
模型回答: {answer}

评估标准:
- quality_score: 0到1，1表示完整回答了问题
- should_retry: 若回答缺失关键信息或与上下文明显矛盾则为true
- improved_query: 若should_retry为true，给出一个更好的查询语句；否则为null
- reason: 评估理由

输出格式:
{{"quality_score": 0.8, "should_retry": false, "improved_query": null, "reason": "..."}}"""


class ReflectionEngine:
    """
    Self-reflection loop for RAG generation quality.
    Stateless — call reflect() on any (query, context, answer) triple.
    """

    def __init__(
        self,
        llm: "BaseModel",
        max_retries: int = 2,
        quality_threshold: float = 0.6,
        context_preview_chars: int = 400,
    ) -> None:
        self._llm = llm
        self.max_retries = max_retries
        self.quality_threshold = quality_threshold
        self._preview_chars = context_preview_chars

    def reflect(
        self,
        query: str,
        context: str,
        answer: str,
    ) -> ReflectionResult:
        """
        Ask the LLM to judge the answer quality.
        Returns should_retry=False on any parse error to avoid infinite loops.
        """
        prompt = _REFLECTION_PROMPT.format(
            query=query,
            context_preview=context[: self._preview_chars],
            answer=answer,
        )
        try:
            raw = self._llm.chat([{"role": "user", "content": prompt}])
            data = json.loads(raw)
            score = float(data.get("quality_score", 0.5))
            should_retry = bool(data.get("should_retry", False)) and score < self.quality_threshold
            return ReflectionResult(
                should_retry=should_retry,
                reason=str(data.get("reason", "")),
                improved_query=data.get("improved_query") if should_retry else None,
                quality_score=score,
            )
        except Exception as exc:
            return ReflectionResult(
                should_retry=False,
                reason=f"Reflection parse error: {exc}",
                improved_query=None,
                quality_score=0.5,
            )

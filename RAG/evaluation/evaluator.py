#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAGAS-aligned evaluator.

Five metrics across three evaluation layers:
  A. Retrieval  — context_precision, context_recall
  B. Generation — faithfulness, answer_relevancy
  C. End-to-End — answer_correctness

LLM-judge powered by LLMFactory. All judge prompts are self-contained.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, TYPE_CHECKING

import numpy as np

from RAG.evaluation.metrics import (
    EndToEndMetrics,
    GenerationMetrics,
    RetrievalMetrics,
    SystemMetrics,
)

if TYPE_CHECKING:
    from RAG.LLM import BaseModel
    from RAG.schema import Document
    from RAG.trace.schema import QueryTrace

# ─────────────────────────────────────────────────────────────────────────
# Retriever callable signatures
#   retrieve_fn(query: str) -> List[Document]
#   generate_fn(question: str, context: str) -> str
# ─────────────────────────────────────────────────────────────────────────


class RAGEvaluator:
    def __init__(self, judge_llm: Optional["BaseModel"] = None) -> None:
        self._judge = judge_llm

    # ═════════════════════════════════════════════════════════════════════
    # A. Retrieval
    # ═════════════════════════════════════════════════════════════════════

    def evaluate_retrieval(
        self,
        qa_items: Sequence[Dict[str, Any]],
        retriever_fn: Callable[[str], List["Document"]],
        k: int = 5,
    ) -> RetrievalMetrics:
        precisions = []
        recalls = []
        hits = []
        rr_scores = []
        latencies = []
        details = []

        for item in qa_items:
            question = item["question"]
            relevant_ids = set(item.get("relevant_ids", []))

            t0 = time.perf_counter()
            retrieved_docs = retriever_fn(question)
            elapsed = (time.perf_counter() - t0) * 1000

            retrieved_ids = [self._doc_id(d) for d in retrieved_docs[:k]]

            cp = self._context_precision_at_k(retrieved_ids, relevant_ids, k)
            cr = self._context_recall(retrieved_ids, relevant_ids)
            hit = self._hit_at_k(retrieved_ids, relevant_ids)
            rr = self._reciprocal_rank(retrieved_ids, relevant_ids)

            precisions.append(cp)
            recalls.append(cr)
            hits.append(hit)
            rr_scores.append(rr)
            latencies.append(elapsed)

            hit_ranks = [i + 1 for i, rid in enumerate(retrieved_ids) if rid in relevant_ids]
            details.append({
                "question": question[:100],
                "context_precision": round(cp, 4),
                "context_recall": round(cr, 4),
                "hit_at_k": int(hit),
                "mrr": round(rr, 4),
                "latency_ms": round(elapsed, 1),
                "matched_ranks": hit_ranks,
                "retrieved": len(retrieved_ids),
                "relevant": len(relevant_ids),
            })

        n = len(qa_items)
        if n > 0 and sum(precisions) == 0 and sum(recalls) == 0:
            sample_retrieved = details[0].get("retrieved", 0) if details else 0
            sample_relevant = details[0].get("relevant", 0) if details else 0
            sample_ranks = details[0].get("matched_ranks", []) if details else []
            print(f"  [DIAG] 所有检索评分为0 — 抽样: retrieved={sample_retrieved}, "
                  f"relevant={sample_relevant}, matched_ranks={sample_ranks}")
            if sample_retrieved > 0 and sample_relevant > 0:
                print(f"  [DIAG] 有检索结果但 identity 未匹配，可能是 doc_id 不一致")
        return RetrievalMetrics(
            context_precision=sum(precisions) / n if n else 0,
            context_recall=sum(recalls) / n if n else 0,
            hit_at_k=sum(hits) / n if n else 0,
            mrr=sum(rr_scores) / n if n else 0,
            avg_latency_ms=sum(latencies) / n if n else 0,
            k=k,
            total=n,
            details=details,
        )

    def _context_precision_at_k(self, retrieved: List[str], relevant: set, k: int) -> float:
        """Average Precision @ k. Rewards relevant chunks appearing early."""
        if not relevant:
            return 0.0
        score = 0.0
        hits = 0
        for i, rid in enumerate(retrieved[:k]):
            if rid in relevant:
                hits += 1
                score += hits / (i + 1)
        return score / min(len(relevant), k) if relevant else 0.0

    def _context_recall(self, retrieved: List[str], relevant: set) -> float:
        """Fraction of relevant chunks that appear in retrieved results."""
        if not relevant:
            return 0.0
        return len(set(retrieved) & relevant) / len(relevant)

    @staticmethod
    def _hit_at_k(retrieved: List[str], relevant: set) -> float:
        """1 if at least one relevant doc in top-K, else 0."""
        if not relevant:
            return 0.0
        return 1.0 if any(rid in relevant for rid in retrieved) else 0.0

    @staticmethod
    def _reciprocal_rank(retrieved: List[str], relevant: set) -> float:
        """1 / rank of first relevant doc; 0 if none found."""
        for i, rid in enumerate(retrieved):
            if rid in relevant:
                return 1.0 / (i + 1)
        return 0.0

    # ═════════════════════════════════════════════════════════════════════
    # B. Generation
    # ═════════════════════════════════════════════════════════════════════

    def evaluate_generation(
        self,
        qa_items: Sequence[Dict[str, Any]],
        retrieve_fn: Callable[[str], List["Document"]],
        generate_fn: Callable[[str, str], str],
        k: int = 5,
    ) -> GenerationMetrics:
        faithfulness_scores = []
        relevancy_scores = []
        details = []

        for item in qa_items:
            question = item["question"]
            docs = retrieve_fn(question)[:k]
            context = "\n\n".join(d.text for d in docs)
            answer = generate_fn(question, context)

            faith = (
                self._judge_faithfulness(question, context, answer)
                if self._judge else self._heuristic_faithfulness(answer, context)
            )
            relev = (
                self._judge_answer_relevancy(question, answer)
                if self._judge else self._heuristic_relevancy(question, answer)
            )

            faithfulness_scores.append(faith)
            relevancy_scores.append(relev)
            details.append({
                "question": question[:100],
                "answer_preview": answer[:200],
                "faithfulness": round(faith, 4),
                "answer_relevancy": round(relev, 4),
            })

        n = len(qa_items)
        avg_faith = sum(faithfulness_scores) / n if n else 0
        return GenerationMetrics(
            faithfulness=avg_faith,
            hallucination_rate=1.0 - avg_faith,
            answer_relevancy=sum(relevancy_scores) / n if n else 0,
            total=n,
            details=details,
        )

    # ═════════════════════════════════════════════════════════════════════
    # C. End-to-End
    # ═════════════════════════════════════════════════════════════════════

    def evaluate_end_to_end(
        self,
        qa_items: Sequence[Dict[str, Any]],
        generate_fn: Callable[[str, str], str],
        retrieve_fn: Callable[[str], List["Document"]],
        k: int = 5,
    ) -> EndToEndMetrics:
        correctness_scores = []
        details = []

        for item in qa_items:
            question = item["question"]
            ground_truth = item.get("ground_truth") or item.get("gold_answer", "")
            docs = retrieve_fn(question)[:k]
            context = "\n\n".join(d.text for d in docs)
            answer = generate_fn(question, context)

            score = (
                self._judge_answer_correctness(question, answer, ground_truth)
                if self._judge and ground_truth
                else self._heuristic_correctness(answer, ground_truth)
            )

            correctness_scores.append(score)
            details.append({
                "question": question[:100],
                "answer_preview": answer[:200],
                "ground_truth_preview": ground_truth[:200],
                "answer_correctness": round(score, 4),
            })

        n = len(qa_items)
        return EndToEndMetrics(
            answer_correctness=sum(correctness_scores) / n if n else 0,
            total=n,
            details=details,
        )

    # ═════════════════════════════════════════════════════════════════════
    # D. System
    # ═════════════════════════════════════════════════════════════════════

    def evaluate_system(self, traces: List["QueryTrace"]) -> SystemMetrics:
        if not traces:
            return SystemMetrics(0, 0, 0, 0, 0, 0)

        def avg(seq):
            return sum(seq) / len(seq) if seq else 0.0

        total_prompt = sum(t.total_prompt_tokens for t in traces)
        total_cached = sum(t.cached_prompt_tokens for t in traces)
        cache_rate = total_cached / total_prompt if total_prompt > 0 else 0.0

        return SystemMetrics(
            avg_retrieval_latency_ms=avg([t.retrieval_latency for t in traces]),
            avg_generation_latency_ms=avg([t.generation_latency for t in traces]),
            llm_cache_rate=cache_rate,
            avg_cached_prompt_tokens=avg([t.cached_prompt_tokens for t in traces]),
            avg_total_prompt_tokens=avg([t.total_prompt_tokens for t in traces]),
            total_traces=len(traces),
        )

    # ═════════════════════════════════════════════════════════════════════
    # LLM Judge prompts
    # ═════════════════════════════════════════════════════════════════════

    def _judge_faithfulness(self, question: str, context: str, answer: str) -> float:
        prompt = (
            "你的任务是评估生成的答案是否完全基于给定的上下文。\n\n"
            "步骤：\n"
            "1. 将答案拆解为独立的事实陈述（claims）\n"
            "2. 逐一判断每个陈述是否能在上下文中找到支撑\n"
            "3. 计算得到支撑的陈述占比\n\n"
            f"问题: {question}\n\n"
            f"上下文:\n{context[:2000]}\n\n"
            f"答案:\n{answer}\n\n"
            '只输出JSON: {"claims": ["陈述1","陈述2"], "supported": [true,false], '
            '"faithfulness": <0到1的小数>}'
        )
        return self._call_judge(prompt, "faithfulness", default=0.5)

    def _judge_answer_relevancy(self, question: str, answer: str) -> float:
        prompt = (
            "评估答案是否直接回答了用户的问题。\n\n"
            f"用户问题: {question}\n\n"
            f"生成的答案: {answer}\n\n"
            "请对答案的相关性打分（0-1）：\n"
            "- 1.0: 答案完整且直接地回应了问题，没有无关内容\n"
            "- 0.7: 答案大体相关，但有少量冗余\n"
            "- 0.3: 答案部分相关，但偏离了问题核心\n"
            "- 0.0: 答案完全无关或跑题\n\n"
            '只输出JSON: {"relevancy": <0到1>, "reason": "<一句话理由>"}'
        )
        return self._call_judge(prompt, "relevancy", default=0.5)

    def _judge_answer_correctness(
        self, question: str, answer: str, ground_truth: str
    ) -> float:
        prompt = (
            "对比生成的答案与标准答案，从事实准确性和语义完整性两个维度打分。\n\n"
            f"问题: {question}\n\n"
            f"标准答案: {ground_truth}\n\n"
            f"生成答案: {answer}\n\n"
            "打分规则（0-1）：\n"
            "- 事实准确性(50%): 生成答案中的事实是否与标准答案一致\n"
            "- 语义完整性(50%): 生成答案是否覆盖了标准答案的关键要点\n\n"
            '只输出JSON: {"factual_accuracy": <0到1>, "completeness": <0到1>, '
            '"correctness": <0到1>, "reason": "<一句话>"}'
        )
        return self._call_judge(prompt, "correctness", default=0.5)

    def _call_judge(self, prompt: str, field: str, default: float) -> float:
        if self._judge is None:
            return default
        try:
            raw = self._judge.chat([{"role": "user", "content": prompt}])
            data = self._parse_json_response(raw)
            # _parse_json_response may wrap a single dict in a list
            if isinstance(data, list):
                data = data[0] if data else {}
            if not isinstance(data, dict):
                print(f"  [JUDGE] unexpected type: {type(data).__name__}, raw[:200]={raw[:200]}")
                return default
            val = float(data.get(field, default))
            if val == default and field not in data:
                print(f"  [JUDGE] {field} missing, raw[:200]={raw[:200]}")
            return val
        except (ValueError, TypeError) as exc:
            print(f"  [JUDGE] {field} parse error: {exc}, raw[:200]={raw[:200] if 'raw' in dir() else 'N/A'}")
            return default

    # ═════════════════════════════════════════════════════════════════════
    # Heuristic fallbacks (no LLM)
    # ═════════════════════════════════════════════════════════════════════

    @staticmethod
    def _heuristic_faithfulness(answer: str, context: str) -> float:
        sentences = [s.strip() for s in re.split(r"[。！？.!?]", answer) if s.strip()]
        if not sentences:
            return 0.5
        ctx_norm = _normalise(context)
        supported = sum(1 for s in sentences if _normalise(s) in ctx_norm)
        return supported / len(sentences)

    @staticmethod
    def _heuristic_relevancy(question: str, answer: str) -> float:
        q_terms = set(re.findall(r"[a-z0-9_]+|[一-鿿]", question.lower()))
        a_terms = set(re.findall(r"[a-z0-9_]+|[一-鿿]", answer.lower()))
        if not q_terms:
            return 0.0
        return len(q_terms & a_terms) / len(q_terms)

    @staticmethod
    def _heuristic_correctness(answer: str, ground_truth: str) -> float:
        if not ground_truth:
            return 0.5
        # Jaccard token overlap as weak proxy
        a_set = set(re.findall(r"[a-z0-9_]+|[一-鿿]", answer.lower()))
        g_set = set(re.findall(r"[a-z0-9_]+|[一-鿿]", ground_truth.lower()))
        if not g_set:
            return 0.5
        return len(a_set & g_set) / len(g_set)

    # ═════════════════════════════════════════════════════════════════════
    # Helpers
    # ═════════════════════════════════════════════════════════════════════

    @staticmethod
    def _doc_id(doc: "Document") -> str:
        if hasattr(doc, "identity"):
            return doc.identity
        if isinstance(doc, str):
            return doc
        return ""

    @staticmethod
    def _parse_json_response(raw: str) -> Dict[str, Any]:
        text = raw.strip()
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if m:
            text = m.group(1).strip()
        for bracket in [("{", "}"), ("[", "]")]:
            start = text.find(bracket[0])
            if start == -1:
                continue
            depth = 0
            end = -1
            for i in range(start, len(text)):
                if text[i] == bracket[0]:
                    depth += 1
                elif text[i] == bracket[1]:
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    continue
        return {}


def _normalise(text: str) -> str:
    return "".join(str(text).split()).lower()

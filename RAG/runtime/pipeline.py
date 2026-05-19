#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAGRuntime — top-level orchestrator.

Coordinates: Router → Retriever → Reranker → ContextBuilder → Compressor
             → PromptManager → LLM → MemoryManager → TraceStore

Single entry point: runtime.query(query) → RAGResponse
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from collections import defaultdict
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from RAG.chunking import count_tokens
from RAG.context.builder import BuiltContext, ContextBuilder
from RAG.context.compressor import ContextCompressor
from RAG.core.config import RAGConfig
from RAG.memory.manager import MemoryContext
from RAG.trace.schema import QueryTrace
from RAG.trace.store import TraceStore

if TYPE_CHECKING:
    from RAG.LLM import BaseModel, PromptManager
    from RAG.memory.manager import MemoryContext, MemoryManager
    from RAG.Reranker import BaseReranker
    from RAG.retrievers import HybridRetriever
    from RAG.router.router import PolicyRouter
    from RAG.schema import Document


@dataclass
class RAGResponse:
    answer: str
    context: str
    citations: List[Dict[str, Any]]
    trace: QueryTrace


def _doc_summary(doc: "Document", chars: int = 120) -> Dict[str, Any]:
    return {
        "text_preview": doc.text[:chars],
        "score": round(float(doc.score), 4),
        "rerank_score": round(float(doc.rerank_score), 4) if doc.rerank_score is not None else None,
        "source": doc.metadata.get("source", ""),
        "chunk_id": doc.metadata.get("chunk_id"),
        "section": doc.metadata.get("section"),
    }


class RAGRuntime:

    def __init__(
        self,
        retriever: "HybridRetriever",
        context_builder: ContextBuilder,
        llm: "BaseModel",
        prompt_manager: "PromptManager",
        memory: Optional["MemoryManager"] = None,
        router: Optional["PolicyRouter"] = None,
        reranker: Optional["BaseReranker"] = None,
        compressor: Optional[ContextCompressor] = None,
        trace_store: Optional[TraceStore] = None,
        config: Optional[RAGConfig] = None,
    ) -> None:
        self._retriever = retriever
        self._context_builder = context_builder
        self._llm = llm
        self._prompt_manager = prompt_manager
        self._memory = memory
        self._router = router
        self._reranker = reranker
        self._compressor = compressor
        self._trace_store = trace_store
        self._cfg = config or RAGConfig()

    def query(
        self,
        query: str,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> RAGResponse:
        trace = QueryTrace(query=query)
        t_total = time.perf_counter()

        # ── 1. Memory context ─────────────────────────────────────────
        mem_ctx = self._memory.before_query(query) if self._memory else MemoryContext([], [])

        # ── 2. Routing ─────────────────────────────────────────────────
        route_decision = None
        if self._router:
            route_decision = self._router.route(query)
            trace.route = route_decision.strategy
            trace.route_policy_scores = {
                s.policy_name: round(s.score, 3)
                for s in route_decision.all_scores
            }
            self._apply_route(route_decision)

        # ── 3. Retrieval ───────────────────────────────────────────────
        t0 = time.perf_counter()
        recalled = self._retriever.retrieve(
            query,
            top_k=self._cfg.retrieval.final_top_k,
            metadata_filter=metadata_filter,
        )
        trace.retrieval_latency = round((time.perf_counter() - t0) * 1000, 1)
        trace.recalled_count = len(recalled)
        trace.recalled_docs = [_doc_summary(d) for d in recalled[:self._cfg.trace.recalled_docs_cap]]

        # ── 4. Candidate cleanup ───────────────────────────────────────
        candidates = self._prepare_rerank_candidates(recalled)
        trace.metadata["rerank_candidate_count"] = len(candidates)
        trace.metadata["rerank_candidate_dropped"] = max(0, len(recalled) - len(candidates))

        # ── 5. Rerank ──────────────────────────────────────────────────
        t0 = time.perf_counter()
        if self._reranker:
            reranked = self._reranker.rerank_documents(
                query, candidates, k=self._cfg.reranker.top_k
            )
        else:
            reranked = candidates[: self._cfg.reranker.top_k]
        trace.rerank_latency = round((time.perf_counter() - t0) * 1000, 1)
        trace.reranked_count = len(reranked)

        # ── 6. Parent expansion and compression ────────────────────────
        context_docs = self._context_builder.prepare_documents(reranked)
        if self._compressor:
            context_docs = self._compressor.compress(
                context_docs, query, self._cfg.context.max_tokens
            )

        # ── 7. Build context ───────────────────────────────────────────
        built = self._context_builder.build(context_docs, resolve_parents=False)
        trace.prompt_tokens = built.token_count
        trace.citations = built.citations

        # ── 8. Generate ────────────────────────────────────────────────
        session_ctx = self._memory.short_term.get_context_str() if self._memory else ""
        history_messages = mem_ctx.history_messages
        preferences = mem_ctx.preferences
        prompt_budget = self._cfg.compression.prompt_max_tokens
        if prompt_budget:
            history_messages, preferences, session_ctx, context_docs, built = (
                self._fit_prompt_budget(
                    history_messages=history_messages,
                    preferences=preferences,
                    session_context=session_ctx,
                    context_docs=context_docs,
                    query=query,
                    prompt_budget=prompt_budget,
                )
            )
        messages = self._prompt_manager.build_messages(
            history=history_messages,
            context=built.context,
            question=query,
            preferences=preferences,
            session_context=session_ctx,
        )
        trace.prompt_tokens = built.token_count
        trace.citations = built.citations

        # Append full prompt to prompts.jsonl for offline review
        try:
            os.makedirs("storage", exist_ok=True)
            with open("storage/prompts.jsonl", "a", encoding="utf-8") as _pf:
                _pf.write(json.dumps({
                    "trace_id": trace.trace_id,
                    "timestamp": trace.timestamp,
                    "query": query,
                    "messages": messages,
                }, ensure_ascii=False) + "\n")
        except OSError:
            pass

        t0 = time.perf_counter()
        answer = self._llm.chat(messages)
        trace.generation_latency = round((time.perf_counter() - t0) * 1000, 1)
        trace.cached_prompt_tokens = self._llm.cached_prompt_tokens
        trace.total_prompt_tokens = self._llm.total_prompt_tokens
        trace.complete_generation(answer, latency_ms=trace.generation_latency)

        # ── 9. Update memory ───────────────────────────────────────────
        if self._memory:
            self._memory.after_query(query, built.context, answer)

        # ── 10. Save trace ─────────────────────────────────────────────
        trace.metadata["total_latency_ms"] = round(
            (time.perf_counter() - t_total) * 1000, 1
        )
        if self._trace_store:
            self._trace_store.append(trace)

        return RAGResponse(
            answer=answer,
            context=built.context,
            citations=built.citations,
            trace=trace,
        )

    def _fit_prompt_budget(
        self,
        history_messages: List[Dict[str, str]],
        preferences: List[str],
        session_context: str,
        context_docs: List["Document"],
        query: str,
        prompt_budget: int,
    ) -> tuple[List[Dict[str, str]], List[str], str, List["Document"], BuiltContext]:
        history = list(history_messages)
        prefs = list(preferences)
        sess_ctx = session_context
        docs = list(context_docs)
        built = self._context_builder.build(docs, resolve_parents=False)

        def build_messages() -> List[Dict[str, str]]:
            return self._prompt_manager.build_messages(
                history=history,
                context=built.context,
                question=query,
                preferences=prefs,
                session_context=sess_ctx,
            )

        while history and self._messages_token_count(build_messages()) > prompt_budget:
            drop_count = 2 if len(history) >= 2 else 1
            history = history[drop_count:]

        while prefs and self._messages_token_count(build_messages()) > prompt_budget:
            prefs = prefs[:-1]

        if sess_ctx and self._messages_token_count(build_messages()) > prompt_budget:
            sess_ctx = ""

        while len(docs) > 1 and self._messages_token_count(build_messages()) > prompt_budget:
            docs = docs[:-1]
            built = self._context_builder.build(docs, resolve_parents=False)

        if docs and self._messages_token_count(build_messages()) > prompt_budget:
            fixed_prompt_tokens = self._messages_token_count(
                self._prompt_manager.build_messages(
                    history=history,
                    context="",
                    question=query,
                    preferences=prefs,
                    session_context=sess_ctx,
                )
            )
            remaining_context_tokens = max(1, prompt_budget - fixed_prompt_tokens)
            built = self._context_builder.build(
                docs,
                resolve_parents=False,
                max_tokens=remaining_context_tokens,
            )

        return history, prefs, sess_ctx, docs, built

    def _messages_token_count(self, messages: List[Dict[str, str]]) -> int:
        # Add a small per-message allowance for chat role/format tokens.
        return sum(count_tokens(message.get("content", "")) + 4 for message in messages)

    def _apply_route(self, route_decision) -> None:
        """Apply route-selected weights to the hybrid retriever config."""
        self._retriever.config.dense_weight = route_decision.dense_weight
        self._retriever.config.sparse_weight = route_decision.sparse_weight

    def _prepare_rerank_candidates(self, documents: List["Document"]) -> List["Document"]:
        """
        Lightweight cleanup before cross-encoder reranking.

        Keep this stage conservative: remove exact duplicate chunks and cap repeated
        children from the same parent, but leave semantic relevance decisions to the
        reranker/compressor where query-document interaction is available.
        """
        candidate_top_k = self._cfg.reranker.candidate_top_k
        max_per_parent = max(1, self._cfg.reranker.max_candidates_per_parent)

        unique_by_id: Dict[str, "Document"] = {}
        for doc in documents:
            if not doc.text.strip():
                continue
            current = unique_by_id.get(doc.identity)
            if current is None or doc.score > current.score:
                unique_by_id[doc.identity] = doc

        ordered = sorted(unique_by_id.values(), key=lambda d: d.score, reverse=True)
        parent_counts: Dict[str, int] = defaultdict(int)
        cleaned: List["Document"] = []
        for doc in ordered:
            parent_key = doc.metadata.get("parent_id") or doc.identity
            if parent_counts[parent_key] >= max_per_parent:
                continue
            parent_counts[parent_key] += 1
            cleaned.append(doc)
            if candidate_top_k > 0 and len(cleaned) >= candidate_top_k:
                break
        return cleaned

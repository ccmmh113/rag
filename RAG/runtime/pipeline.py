#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAGRuntime — top-level orchestrator.

Coordinates: Router → Retriever → Reranker → Compressor → ContextBuilder
             → PromptManager → LLM → MemoryManager → TraceStore

Single entry point: runtime.query(query) → RAGResponse
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from RAG.context.builder import ContextBuilder
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

        # ── 4. Rerank ──────────────────────────────────────────────────
        t0 = time.perf_counter()
        if self._reranker:
            reranked = self._reranker.rerank_documents(
                query, recalled, k=self._cfg.reranker.top_k
            )
        else:
            reranked = recalled[: self._cfg.reranker.top_k]
        trace.rerank_latency = round((time.perf_counter() - t0) * 1000, 1)
        trace.reranked_count = len(reranked)

        # ── 5. Compress ────────────────────────────────────────────────
        if self._compressor:
            reranked = self._compressor.compress(
                reranked, query, self._cfg.context.max_tokens
            )

        # ── 6. Build context ───────────────────────────────────────────
        built = self._context_builder.build(reranked)
        trace.prompt_tokens = built.token_count
        trace.citations = built.citations

        # ── 7. Generate ────────────────────────────────────────────────
        session_ctx = self._memory.short_term.get_context_str() if self._memory else ""
        messages = self._prompt_manager.build_messages(
            history=mem_ctx.history_messages,
            context=built.context,
            question=query,
            preferences=mem_ctx.preferences,
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

        # ── 8. Update memory ───────────────────────────────────────────
        if self._memory:
            self._memory.after_query(query, built.context, answer)

        # ── 9. Save trace ──────────────────────────────────────────────
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

    def _apply_route(self, route_decision) -> None:
        """Apply route-selected weights to the hybrid retriever config."""
        self._retriever.config.dense_weight = route_decision.dense_weight
        self._retriever.config.sparse_weight = route_decision.sparse_weight

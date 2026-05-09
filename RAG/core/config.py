#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Centralised configuration for all RAG subsystems.
Every subsystem reads its own dataclass from here — no magic numbers elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional


# ── Index ──────────────────────────────────────────────────────────────────────

@dataclass
class IndexConfig:
    """Vector index construction and search parameters."""
    index_type: Literal["flat_ip", "hnsw"] = "flat_ip"
    hnsw_m: int = 32                  # HNSW graph connectivity
    hnsw_ef_construction: int = 200   # HNSW build-time search width
    hnsw_ef_search: int = 64          # HNSW query-time search width
    dimension: int = 768              # embedding dimension (set at runtime)


# ── Chunking ───────────────────────────────────────────────────────────────────

@dataclass
class ChunkingConfig:
    max_tokens: int = 600
    overlap_tokens: int = 120
    min_chunk_tokens: int = 20


@dataclass
class ParentChildConfig:
    child_max_tokens: int = 150
    child_overlap_tokens: int = 30
    parent_max_tokens: int = 2000


# ── Retrieval ──────────────────────────────────────────────────────────────────

@dataclass
class RetrievalConfig:
    dense_top_k: int = 50
    sparse_top_k: int = 50
    final_top_k: int = 50
    fusion: Literal["rrf", "weighted"] = "rrf"
    rrf_k: int = 60
    dense_weight: float = 0.5
    sparse_weight: float = 0.5
    parallel: bool = True


# ── Reranker ───────────────────────────────────────────────────────────────────

@dataclass
class RerankerConfig:
    model_path: str = "BAAI/bge-reranker-base"
    top_k: int = 8
    max_length: int = 512


# ── Router ─────────────────────────────────────────────────────────────────────

@dataclass
class RouterConfig:
    """Weights used by the policy scorer."""
    short_query_threshold: int = 5    # token count below which query is "short"
    code_score_boost: float = 0.3     # extra score for CodePolicy on code queries


# ── Context ────────────────────────────────────────────────────────────────────

@dataclass
class ContextConfig:
    max_tokens: int = 3000
    min_score: float = 0.0
    overlap_chars: int = 160
    semantic_dedup_threshold: float = 0.92
    enable_semantic_dedup: bool = True


@dataclass
class CompressionConfig:
    strategy: Literal["relevance_filter", "summary", "hybrid"] = "relevance_filter"
    relevance_threshold: float = 0.3   # min rerank_score to keep
    summary_max_tokens: int = 200      # per-chunk summary budget


# ── Memory ─────────────────────────────────────────────────────────────────────

@dataclass
class ShortTermConfig:
    max_turns: int = 10
    max_tokens: int = 2000


@dataclass
class LongTermConfig:
    db_path: str = "storage/ltm.db"
    index_path: str = "storage/ltm_index"
    similarity_threshold: float = 0.92
    default_expire_days: int = 90


# ── Evaluation ─────────────────────────────────────────────────────────────────

@dataclass
class EvalConfig:
    retrieval_k: int = 3
    faithfulness_model: str = "none"   # "none" | llm provider key
    hallucination_threshold: float = 0.5


# ── Trace ──────────────────────────────────────────────────────────────────────

@dataclass
class TraceConfig:
    store_path: str = "storage/traces.jsonl"
    recalled_docs_cap: int = 10        # max recalled doc summaries stored per trace
    answer_preview_chars: int = 200


# ── Reflection ─────────────────────────────────────────────────────────────────

@dataclass
class ReflectionConfig:
    enabled: bool = False
    max_retries: int = 2
    quality_threshold: float = 0.6     # below this → retry


# ── Debug ───────────────────────────────────────────────────────────────────────

@dataclass
class DebugConfig:
    print_prompt: bool = False          # print full prompt before LLM call


# ── Top-level pipeline ─────────────────────────────────────────────────────────

@dataclass
class RAGConfig:
    """One object to configure the entire RAGRuntime."""
    index: IndexConfig = field(default_factory=IndexConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    parent_child: ParentChildConfig = field(default_factory=ParentChildConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    short_term: ShortTermConfig = field(default_factory=ShortTermConfig)
    long_term: LongTermConfig = field(default_factory=LongTermConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    trace: TraceConfig = field(default_factory=TraceConfig)
    reflection: ReflectionConfig = field(default_factory=ReflectionConfig)

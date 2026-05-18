#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build a larger strategy benchmark for retrieval experiments.

This benchmark is intentionally controlled: every answer is tied to an anchor
inside data/eval_*.md, and each item declares whether it is mainly semantic,
exact/code-token, or multi-hop. The script resolves anchors to current chunk
identities so it stays compatible after re-chunking.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).parent.parent))

from RAG.core.config import RAGConfig
from RAG.utils import ReadFiles


def _basename(path: str) -> str:
    return os.path.basename(path.replace("\\", "/"))


def _chunk_data(data_dir: str, config: RAGConfig):
    pc = config.parent_child
    reader = ReadFiles(data_dir)
    return reader.get_parent_child_documents(
        child_max_tokens=pc.child_max_tokens,
        child_overlap_tokens=pc.child_overlap_tokens,
        parent_max_tokens=pc.parent_max_tokens,
    )


def _resolve_relevant_ids(
    spec: dict[str, Any],
    docs,
    parent_map: dict[str, str],
    max_ids: int,
) -> tuple[list[str], list[str]]:
    relevant_ids: list[str] = []
    contexts: list[str] = []
    for doc in docs:
        if _basename(doc.metadata.get("source", "")) != spec["source"]:
            continue
        parent_text = parent_map.get(doc.metadata.get("parent_id"), "")
        haystack = f"{doc.text}\n{parent_text}"
        if spec["anchor"] not in haystack:
            continue
        relevant_ids.append(doc.identity)
        context = parent_text or doc.text
        if context and context not in contexts:
            contexts.append(context[:1200])
        if len(relevant_ids) >= max_ids:
            break
    return relevant_ids, contexts


def _spec(
    question: str,
    answer: str,
    qa_type: str,
    source: str,
    anchor: str,
    focus: str,
) -> dict[str, Any]:
    return {
        "question": question,
        "ground_truth": answer,
        "qa_type": qa_type,
        "source": source,
        "anchor": anchor,
        "retrieval_focus": focus,
    }


SPECS: list[dict[str, Any]] = [
    # Pipeline and chunking
    _spec("TinyRAG 的完整 RAG pipeline 有哪些关键阶段？", "TinyRAG 包含文档读取、父子分块、embedding、FAISS 稠密检索、BM25 稀疏检索、混合召回、候选清洗、reranker、压缩、ContextBuilder、PromptManager、LLM 和 trace。", "list", "eval_01_rag_pipeline_and_chunking.md", "EVAL-RAG-001", "semantic"),
    _spec("召回之后为什么不能直接把 top-k chunk 丢给 LLM？", "因为召回结果可能重复、弱相关或上下文不完整，需要候选清洗、reranker、压缩和 ContextBuilder 处理后再进入 prompt。", "why", "eval_01_rag_pipeline_and_chunking.md", "EVAL-RAG-001", "semantic"),
    _spec("TinyRAG 如何在召回率、精度和 token 成本之间平衡？", "它通过混合召回保证不漏证据，通过 reranker 提升精度，通过 ContextCompressor 和 ContextBuilder 控制上下文噪声与 token 成本。", "howto", "eval_01_rag_pipeline_and_chunking.md", "EVAL-RAG-001", "semantic"),
    _spec("PromptManager 在 TinyRAG 链路中的位置是什么？", "PromptManager 位于 ContextBuilder 之后、OpenAIChat 之前，负责把上下文、问题和记忆组织成最终消息。", "definition", "eval_01_rag_pipeline_and_chunking.md", "EVAL-RAG-001", "exact"),
    _spec("ReadFiles 到 TraceStore 之间的组件顺序是什么？", "顺序是 ReadFiles、ParentChildChunker、EmbeddingModel、FaissVectorIndex、DenseRetriever/BM25Retriever、HybridRetriever、候选清洗、BgeReranker、ContextCompressor、ContextBuilder、PromptManager、OpenAIChat、TraceStore。", "list", "eval_01_rag_pipeline_and_chunking.md", "EVAL-RAG-001", "exact"),
    _spec("父子分块为什么能缓解语义被切割的问题？", "child chunk 负责精确命中局部证据，parent chunk 负责补充完整上下文，避免只用小块生成时语义不完整。", "why", "eval_01_rag_pipeline_and_chunking.md", "EVAL-RAG-002", "semantic"),
    _spec("parent_id 在父子分块里起什么作用？", "parent_id 是 child 到 parent 的映射键，ContextBuilder 根据它从 parent_map 中找到完整 parent text。", "definition", "eval_01_rag_pipeline_and_chunking.md", "EVAL-RAG-002", "exact"),
    _spec("只用大 chunk 检索和只用小 chunk 生成分别有什么问题？", "只用大 chunk 检索可能不够精准，只用小 chunk 生成可能缺少完整语义，父子分块用不同粒度分别服务召回和生成。", "compare", "eval_01_rag_pipeline_and_chunking.md", "EVAL-RAG-002", "semantic"),
    _spec("多个 child 命中同一个 parent 时应该怎么处理？", "ContextBuilder 应按 parent_id 去重，同一个 parent 只放一次，避免重复浪费 token。", "howto", "eval_01_rag_pipeline_and_chunking.md", "EVAL-RAG-002", "semantic"),
    _spec("child chunk 和 parent chunk 的职责分别是什么？", "child chunk 用于精准命中问题相关局部内容，parent chunk 用于补充同一主题下的完整解释。", "compare", "eval_01_rag_pipeline_and_chunking.md", "EVAL-RAG-002", "semantic"),
    _spec("稳定 chunk identity 为什么对评估很重要？", "稳定 identity 能保证评测集 relevant_ids 不失效，也能让引用定位和旧向量复用可靠。", "why", "eval_01_rag_pipeline_and_chunking.md", "EVAL-RAG-003", "semantic"),
    _spec("chunk_uid 由哪些信息组成会比较稳定？", "推荐包含 doc_id、source_hash、page、parent_chunk_id、child_chunk_id 和 text_hash 等信息。", "list", "eval_01_rag_pipeline_and_chunking.md", "EVAL-RAG-003", "exact"),
    _spec("为什么 doc_id:source:chunk_id 在父子分块下不够稳？", "因为不同 parent 下的 child chunk 可能都从 0 开始编号，仅靠 chunk_id 容易冲突。", "why", "eval_01_rag_pipeline_and_chunking.md", "EVAL-RAG-003", "semantic"),
    _spec("build_chunk_uid 函数返回的字符串格式是什么？", "格式是 doc_id、source_key、page、parent_idx 和 child_idx 拼成的稳定 chunk uid，例如 doc::source::p0::1::c2。", "definition", "eval_01_rag_pipeline_and_chunking.md", "EVAL-RAG-003", "exact"),

    # Hybrid retrieval
    _spec("BM25 的优势查询场景有哪些？", "BM25 适合函数名、配置项、错误码、类名、字段名、英文缩写和专有名词等精确 token 查询。", "list", "eval_02_hybrid_retrieval_bm25_dense.md", "EVAL-RET-001", "exact"),
    _spec("为什么 RAG_INDEX_STALE 这类问题 BM25 更容易命中？", "因为错误码是精确字符串，BM25 基于词项匹配，可以直接命中包含该 token 的文档。", "why", "eval_02_hybrid_retrieval_bm25_dense.md", "EVAL-RET-001", "exact"),
    _spec("BM25 在代码检索里为什么重要？", "代码检索经常包含函数名、类名和配置 key，这些精确 token 的语义向量不一定稳定，BM25 更可靠。", "why", "eval_02_hybrid_retrieval_bm25_dense.md", "EVAL-RET-001", "semantic"),
    _spec("HybridRetriever 这个类名适合用哪类检索命中？", "HybridRetriever 是精确类名，BM25 应该能稳定命中包含该 token 的文档。", "definition", "eval_02_hybrid_retrieval_bm25_dense.md", "EVAL-RET-001", "exact"),
    _spec("candidate_top_k 和 chunk_uid 这类字段为什么能体现 BM25 优势？", "它们是配置项或字段名，属于精确 token，BM25 基于词项匹配通常比纯向量检索更稳定。", "why", "eval_02_hybrid_retrieval_bm25_dense.md", "EVAL-RET-001", "exact"),
    _spec("Dense retrieval 适合处理哪类问题？", "Dense retrieval 适合同义改写、概念解释和语义相关问题，即使 query 与文档字面不完全一致也可能召回相关内容。", "list", "eval_02_hybrid_retrieval_bm25_dense.md", "EVAL-RET-002", "semantic"),
    _spec("为什么“不能只用很小分块回答”属于语义检索友好问题？", "因为它和文档中的上下文不完整、语义切割、小块生成信息不足等表达语义相关，但字面不一定完全相同。", "why", "eval_02_hybrid_retrieval_bm25_dense.md", "EVAL-RET-002", "semantic"),
    _spec("向量检索对函数名和错误码有什么弱点？", "稠密检索对精确字符串不够稳定，函数名、版本号、错误码和配置 key 可能被当成普通 token。", "compare", "eval_02_hybrid_retrieval_bm25_dense.md", "EVAL-RET-002", "semantic"),
    _spec("cosine similarity 在 dense retrieval 中有什么作用？", "它用于衡量 query 向量和 chunk 向量在语义空间中的接近程度。", "definition", "eval_02_hybrid_retrieval_bm25_dense.md", "EVAL-RET-002", "semantic"),
    _spec("RRF 融合的核心公式是什么？", "RRF 根据排名给分，score 等于 dense_weight/(rrf_k+dense_rank) 加 sparse_weight/(rrf_k+sparse_rank)。", "definition", "eval_02_hybrid_retrieval_bm25_dense.md", "EVAL-RET-003", "exact"),
    _spec("RRF 为什么比直接加原始分数更稳定？", "因为 BM25 分数和向量相似度尺度不同，RRF 使用排名融合，不强制比较原始分数。", "why", "eval_02_hybrid_retrieval_bm25_dense.md", "EVAL-RET-003", "semantic"),
    _spec("HybridRetriever 使用 RRF 时融合了哪两个检索器？", "它融合 DenseRetriever 和 BM25Retriever 的排序结果。", "definition", "eval_02_hybrid_retrieval_bm25_dense.md", "EVAL-RET-003", "exact"),
    _spec("weighted fusion 相比 RRF 的风险是什么？", "Weighted fusion 对分数分布更敏感，如果某个检索器分数跨度异常，结果可能偏向一侧。", "compare", "eval_02_hybrid_retrieval_bm25_dense.md", "EVAL-RET-004", "semantic"),
    _spec("什么时候可以提高 sparse_weight？", "短关键词或代码标识符查询可以提高 sparse_weight。", "howto", "eval_02_hybrid_retrieval_bm25_dense.md", "EVAL-RET-004", "semantic"),
    _spec("长语义问题应该更偏向 dense_weight 还是 sparse_weight？", "长语义问题更适合提高 dense_weight，因为 dense retrieval 更擅长语义相关召回。", "definition", "eval_02_hybrid_retrieval_bm25_dense.md", "EVAL-RET-004", "semantic"),

    # Index and FAISS
    _spec("增量 embedding 具体减少了什么成本？", "它减少 embedding 计算成本，未变化 chunk 直接复用旧向量，只对新增或修改 chunk 重新计算向量。", "definition", "eval_03_vector_index_incremental_faiss.md", "EVAL-IDX-001", "semantic"),
    _spec("增量 index update 和增量 embedding 的区别是什么？", "增量 embedding 是少算向量，增量 index update 是少改索引，支持新增、修改、删除向量直接 upsert/delete。", "compare", "eval_03_vector_index_incremental_faiss.md", "EVAL-IDX-001", "semantic"),
    _spec("TinyRAG 当前的 FAISS 更新策略是什么？", "当前策略是增量 embedding 复用，FAISS index 基于完整向量集合重建。", "definition", "eval_03_vector_index_incremental_faiss.md", "EVAL-IDX-001", "semantic"),
    _spec("为什么说当前方案不是向量数据库级 upsert/delete？", "因为它没有在索引层对单个向量做原生 upsert/delete，而是复用向量后重建完整 FAISS index。", "why", "eval_03_vector_index_incremental_faiss.md", "EVAL-IDX-001", "semantic"),
    _spec("index manifest 的作用是什么？", "manifest 用于判断知识库是否变化，记录文件路径、大小、SHA1、分块参数和 schema version。", "definition", "eval_03_vector_index_incremental_faiss.md", "EVAL-IDX-002", "semantic"),
    _spec("manifest 变化后系统应该做什么？", "系统应该重新读取文档、切 chunk，并复用未变化 chunk 的历史向量，然后刷新索引。", "howto", "eval_03_vector_index_incremental_faiss.md", "EVAL-IDX-002", "semantic"),
    _spec("manifest 中 parent_child 记录了哪些参数？", "parent_child 记录 child_max_tokens、child_overlap_tokens 和 parent_max_tokens。", "list", "eval_03_vector_index_incremental_faiss.md", "EVAL-IDX-002", "exact"),
    _spec("FAISS 适合什么样的场景？", "FAISS 适合本地轻量、高性能相似度搜索，适合 demo、单机部署和中小规模知识库。", "definition", "eval_03_vector_index_incremental_faiss.md", "EVAL-IDX-003", "semantic"),
    _spec("FAISS 相比 Qdrant 的弱点是什么？", "FAISS 缺少原生 payload filter、服务化 API、按 metadata 删除和标准化 upsert 能力。", "compare", "eval_03_vector_index_incremental_faiss.md", "EVAL-IDX-003", "semantic"),
    _spec("什么时候应该从 FAISS 演进到 Qdrant？", "当知识库需要频繁新增删除、按租户隔离、metadata 过滤或服务化部署时，适合抽象 VectorStore 并切换 Qdrant。", "howto", "eval_03_vector_index_incremental_faiss.md", "EVAL-IDX-003", "semantic"),
    _spec("RAG_INDEX_STALE 表示什么？", "RAG_INDEX_STALE 表示 manifest 与磁盘索引不一致，旧索引不能直接使用，需要触发索引更新。", "definition", "eval_03_vector_index_incremental_faiss.md", "EVAL-IDX-004", "exact"),
    _spec("ensure_manifest_compatible 什么时候抛出 IndexStateError？", "当 saved_manifest 和 current_manifest 不一致时，它抛出 IndexStateError，并提示 RAG_INDEX_STALE。", "definition", "eval_03_vector_index_incremental_faiss.md", "EVAL-IDX-004", "exact"),

    # Reranker/context
    _spec("Reranker 前适合做哪些轻量候选清洗？", "适合过滤空文本、按 identity 去重、限制同 parent_id 的 child 数量，并限制送入 reranker 的候选总数。", "list", "eval_04_reranker_context_budget.md", "EVAL-CTX-001", "semantic"),
    _spec("为什么 reranker 前不适合做强语义过滤？", "因为 reranker 前还没有 query-doc 交互分数，强过滤可能误删有用证据。", "why", "eval_04_reranker_context_budget.md", "EVAL-CTX-001", "semantic"),
    _spec("candidate_top_k 控制什么？", "candidate_top_k 控制候选清洗后最多送入 reranker 的候选数量。", "definition", "eval_04_reranker_context_budget.md", "EVAL-CTX-001", "exact"),
    _spec("max_candidates_per_parent 控制什么？", "它限制同一个 parent_id 下保留的 child 数量，避免一个父块占满候选集。", "definition", "eval_04_reranker_context_budget.md", "EVAL-CTX-001", "exact"),
    _spec("BGE reranker 的输入和输出是什么？", "输入是 query 和 document 组成的 pair，输出是相关性分数。", "definition", "eval_04_reranker_context_budget.md", "EVAL-CTX-002", "semantic"),
    _spec("召回 top 50、候选 top 30、reranker top 8 分别追求什么？", "召回 top 50 追求 recall，候选 top 30 减少 reranker 成本，reranker top 8 追求 precision。", "compare", "eval_04_reranker_context_budget.md", "EVAL-CTX-002", "semantic"),
    _spec("候选清洗为什么不能替代 reranker？", "候选清洗只看结构信息，reranker 会结合 query 判断内容相关性。", "why", "eval_04_reranker_context_budget.md", "EVAL-CTX-002", "semantic"),
    _spec("ContextCompressor 默认使用什么策略？", "默认使用 relevance_filter 策略。", "definition", "eval_04_reranker_context_budget.md", "EVAL-CTX-003", "exact"),
    _spec("relevance_threshold 应该优先作用在哪种分数上？", "应该优先作用在 rerank_score 上，而不是直接作用在 RRF 小分值上。", "definition", "eval_04_reranker_context_budget.md", "EVAL-CTX-003", "semantic"),
    _spec("把 RRF 分数当 reranker 分数有什么风险？", "RRF 分数通常很小，直接和 0.3 等阈值比较可能误删所有文档。", "why", "eval_04_reranker_context_budget.md", "EVAL-CTX-003", "semantic"),
    _spec("ContextBuilder 如何做近重复去重？", "有 embedding_model 时使用 embedding cosine similarity，没有时用 Jaccard 词集合相似度兜底。", "howto", "eval_04_reranker_context_budget.md", "EVAL-CTX-004", "semantic"),
    _spec("semantic_dedup_threshold 的默认值是多少？", "默认阈值是 0.92，超过阈值的候选会被认为和已有上下文近重复。", "definition", "eval_04_reranker_context_budget.md", "EVAL-CTX-004", "exact"),
    _spec("没有 embedding_model 时为什么还能去重？", "因为系统会退化为 Jaccard 词集合相似度，作为轻量兜底方案。", "why", "eval_04_reranker_context_budget.md", "EVAL-CTX-004", "semantic"),
    _spec("ContextBuilder 的 budget 控制流程是什么？", "先父块去重、exact dedup、semantic dedup，再计算 header token 和 available tokens，trim_to_tokens 后追加 citation。", "list", "eval_04_reranker_context_budget.md", "EVAL-CTX-005", "semantic"),
    _spec("为什么要在计算 budget 前先去重？", "因为重复上下文会浪费有限 prompt budget，先去重能让高相关且不同的信息优先进入 prompt。", "why", "eval_04_reranker_context_budget.md", "EVAL-CTX-005", "semantic"),

    # Web/code
    _spec("TinyRAG Web 层暴露了哪些核心 API？", "核心 API 包括 /api/status、/api/index/build、/api/config、/api/chat、/api/traces、/api/memories 等。", "list", "eval_05_fastapi_vue_code.md", "EVAL-WEB-001", "exact"),
    _spec("RuntimeManager 负责什么？", "RuntimeManager 负责懒加载 embedding、索引、reranker 和 RAGRuntime。", "definition", "eval_05_fastapi_vue_code.md", "EVAL-WEB-001", "semantic"),
    _spec("POST /api/chat 返回哪些内容？", "POST /api/chat 调用 runtime.query，并返回 answer、context、citations 和 trace。", "definition", "eval_05_fastapi_vue_code.md", "EVAL-WEB-001", "exact"),
    _spec("/api/files/upload 上传文件后应该做哪些事情？", "应该先保存文件、计算 hash、更新 manifest，再触发索引更新，不应盲目覆盖旧索引。", "howto", "eval_05_fastapi_vue_code.md", "EVAL-WEB-002", "semantic"),
    _spec("UploadFile 代码片段支持哪些文件后缀？", "支持 .md、.txt 和 .pdf，其他后缀返回 UNSUPPORTED_FILE_TYPE。", "definition", "eval_05_fastapi_vue_code.md", "EVAL-WEB-002", "exact"),
    _spec("UNSUPPORTED_FILE_TYPE 适合用什么检索命中？", "它是精确错误码，BM25 通常更容易命中。", "definition", "eval_05_fastapi_vue_code.md", "EVAL-WEB-002", "exact"),
    _spec("Vue 配置面板可以暴露哪些检索参数？", "可以暴露 top_k、fusion、dense_weight、sparse_weight、use_reranker 和 compression strategy。", "list", "eval_05_fastapi_vue_code.md", "EVAL-WEB-003", "semantic"),
    _spec("saveConfig 方法调用哪个 API？", "saveConfig 调用 PATCH /api/config，并提交 dense_weight、sparse_weight、use_reranker、fusion 等配置。", "definition", "eval_05_fastapi_vue_code.md", "EVAL-WEB-003", "exact"),

    # Evaluation/trace
    _spec("检索评估应该看哪些指标？", "应该看 Context Precision、Context Recall、Hit@K、MRR 和 latency。", "list", "eval_06_trace_evaluation_report.md", "EVAL-EVAL-001", "semantic"),
    _spec("Context Precision 和 Context Recall 分别关注什么？", "Precision 关注相关文档是否排在前面，Recall 关注所有相关文档被召回的比例。", "compare", "eval_06_trace_evaluation_report.md", "EVAL-EVAL-001", "semantic"),
    _spec("正式消融实验应该比较哪些检索组？", "应该比较 Dense only、BM25 only、Hybrid RRF、Hybrid Weighted、Hybrid + Reranker 和 Full Pipeline。", "list", "eval_06_trace_evaluation_report.md", "EVAL-EVAL-001", "semantic"),
    _spec("Hybrid Recall 高于 Dense 和 BM25 说明什么？", "说明 dense 和 sparse 召回结果互补，融合能召回更多相关证据。", "why", "eval_06_trace_evaluation_report.md", "EVAL-EVAL-001", "semantic"),
    _spec("生成评估关注哪些指标？", "生成评估关注 Faithfulness、Answer Relevancy、Answer Correctness 和 Hallucination Rate。", "list", "eval_06_trace_evaluation_report.md", "EVAL-EVAL-002", "semantic"),
    _spec("为什么检索评估要先于生成评估？", "因为生成错误可能来自召回、排序、压缩或 LLM 幻觉，先稳定检索有助于定位问题。", "why", "eval_06_trace_evaluation_report.md", "EVAL-EVAL-002", "semantic"),
    _spec("Faithfulness 衡量什么？", "Faithfulness 判断答案是否被上下文支持。", "definition", "eval_06_trace_evaluation_report.md", "EVAL-EVAL-002", "semantic"),
    _spec("trace 中的 rerank_candidate_dropped 表示什么？", "它表示候选清洗阶段从 recalled_docs 中丢弃的候选数量。", "definition", "eval_06_trace_evaluation_report.md", "EVAL-EVAL-003", "exact"),
    _spec("一次问答 trace 应该记录哪些延迟字段？", "应该记录 retrieval_latency、rerank_latency 和 generation_latency。", "list", "eval_06_trace_evaluation_report.md", "EVAL-EVAL-003", "semantic"),
    _spec("如果 recalled_docs 没有相关内容，失败原因是什么？", "这是 retrieval_miss，即召回阶段没有取到证据。", "definition", "eval_06_trace_evaluation_report.md", "EVAL-EVAL-003", "semantic"),
    _spec("failure taxonomy 包括哪几类？", "包括 retrieval_miss、ranking_error、compression_drop、context_budget_overflow 和 generation_hallucination。", "list", "eval_06_trace_evaluation_report.md", "EVAL-EVAL-004", "exact"),
    _spec("compression_drop 和 context_budget_overflow 有什么区别？", "compression_drop 是压缩阶段误删证据，context_budget_overflow 是证据存在但因 token budget 没进入 prompt。", "compare", "eval_06_trace_evaluation_report.md", "EVAL-EVAL-004", "semantic"),
    _spec("为什么失败样本归因比只看平均指标更有说服力？", "因为归因能说明系统瓶颈来自召回、排序、压缩、预算还是生成，能指导下一步优化。", "why", "eval_06_trace_evaluation_report.md", "EVAL-EVAL-004", "semantic"),

    # Distractors
    _spec("Agent memory 和 RAG 向量索引有什么区别？", "Agent memory 保存偏好、历史任务、失败经验和流程；RAG 向量索引用于按 query 检索外部知识片段。", "compare", "eval_07_agent_memory_distractors.md", "EVAL-DIST-001", "semantic"),
    _spec("为什么 Agent memory 文档是干扰语料？", "因为它也包含 memory、embedding、retrieval、context 等词，但主题是 Agent 行为记忆而不是 FAISS 知识库索引。", "why", "eval_07_agent_memory_distractors.md", "EVAL-DIST-001", "semantic"),
    _spec("长期记忆如何影响 prompt？", "长期记忆可保存跨会话偏好和笔记，在 before_query 中作为 preferences 或历史上下文进入 prompt。", "howto", "eval_07_agent_memory_distractors.md", "EVAL-DIST-002", "semantic"),
    _spec("MemoryManager 的 save 方法返回什么？", "save 方法保存 note 或 preference，并返回 memory id。", "definition", "eval_07_agent_memory_distractors.md", "EVAL-DIST-002", "exact"),
    _spec("Workflow、Agent、Tools 的区别是什么？", "Workflow 是预定义流程，Agent 根据目标动态规划动作，Tools 是可调用的外部能力。", "compare", "eval_07_agent_memory_distractors.md", "EVAL-DIST-003", "semantic"),
    _spec("为什么 Workflow 文档可能干扰 RAG 检索？", "因为它也包含 route、context、trace、evaluation 等相似词，但并不直接回答 BM25、FAISS 或 chunk_uid 问题。", "why", "eval_07_agent_memory_distractors.md", "EVAL-DIST-003", "semantic"),

    # Code tokens
    _spec("_vector_cache_from_state 的作用是什么？", "它从旧索引状态和 vectors.npy 中构造向量缓存，用于增量 embedding 复用。", "definition", "eval_08_code_error_playbook.md", "EVAL-CODE-001", "exact"),
    _spec("_vector_cache_from_state 使用什么作为缓存 key？", "它使用 doc.identity 和 text_hash 组成的 tuple 作为缓存 key。", "definition", "eval_08_code_error_playbook.md", "EVAL-CODE-001", "exact"),
    _spec("vectors.npy 在增量 embedding 中有什么作用？", "vectors.npy 保存历史向量，系统可以从中读取未变化 chunk 的旧 embedding。", "definition", "eval_08_code_error_playbook.md", "EVAL-CODE-001", "exact"),
    _spec("_prepare_rerank_candidates 为什么要保留 score 更高的重复 identity？", "同一 identity 可能从不同检索器或排序路径出现，保留更高 score 的候选更合理。", "why", "eval_08_code_error_playbook.md", "EVAL-CODE-002", "exact"),
    _spec("_prepare_rerank_candidates 是否替代 reranker？", "不能。它只做保守清洗，减少 cross-encoder 输入成本，相关性判断仍交给 reranker。", "definition", "eval_08_code_error_playbook.md", "EVAL-CODE-002", "exact"),
    _spec("parent_counts 在候选清洗代码中用于什么？", "parent_counts 用于统计每个 parent_key 已保留多少 child，超过 max_candidates_per_parent 就跳过。", "definition", "eval_08_code_error_playbook.md", "EVAL-CODE-002", "exact"),
    _spec("RAG_VECTOR_DIM_MISMATCH 表示什么？", "它表示向量维度和 FAISS 索引维度不匹配。", "definition", "eval_08_code_error_playbook.md", "EVAL-CODE-003", "exact"),
    _spec("RAG_EMPTY_CORPUS 表示什么？", "它表示 data 目录没有可支持的文档。", "definition", "eval_08_code_error_playbook.md", "EVAL-CODE-003", "exact"),
    _spec("RAG_RERANKER_UNAVAILABLE 表示什么？", "它表示 BGE reranker 加载失败或不可用。", "definition", "eval_08_code_error_playbook.md", "EVAL-CODE-003", "exact"),
    _spec("JSON 配置里 retrieval.fusion 的默认示例是什么？", "示例配置中 retrieval.fusion 是 rrf。", "definition", "eval_08_code_error_playbook.md", "EVAL-CODE-004", "exact"),
    _spec("JSON 配置里 reranker.top_k 是多少？", "示例配置中 reranker.top_k 是 8。", "definition", "eval_08_code_error_playbook.md", "EVAL-CODE-004", "exact"),
    _spec("compression.strategy 在配置样例中是什么？", "配置样例中 compression.strategy 是 relevance_filter。", "definition", "eval_08_code_error_playbook.md", "EVAL-CODE-004", "exact"),

    # Cross-cutting / multi-hop checks
    _spec("新增文件后，系统如何同时避免重复 embedding 并保证 FAISS 索引一致？", "系统通过 manifest 判断变化，通过 chunk identity 和 text_hash 复用未变化 chunk 的 embedding，然后基于完整向量集合重建 FAISS index 保证一致性。", "howto", "eval_03_vector_index_incremental_faiss.md", "EVAL-IDX-001", "multi_hop"),
    _spec("为什么稳定 chunk_uid 会影响增量 embedding 和评估 relevant_ids？", "稳定 chunk_uid 让未变化 chunk 能匹配旧向量缓存，也让评估集的 relevant_ids 不会因为重新分块或 parent child 编号冲突而失效。", "why", "eval_01_rag_pipeline_and_chunking.md", "EVAL-RAG-003", "multi_hop"),
    _spec("Hybrid RRF 为什么能同时改善语义改写问题和代码 token 问题？", "语义改写问题由 dense retrieval 补强，代码 token 和错误码由 BM25 补强，RRF 根据排名融合两路结果，因此整体更稳。", "why", "eval_02_hybrid_retrieval_bm25_dense.md", "EVAL-RET-003", "multi_hop"),
    _spec("如果 query 是 RAG_VECTOR_DIM_MISMATCH，为什么 BM25 和 Hybrid 应该优于纯 Dense？", "RAG_VECTOR_DIM_MISMATCH 是精确错误码，BM25 能直接匹配；Hybrid 融合 BM25 与 Dense，所以在精确 token 查询上通常比纯 Dense 更可靠。", "why", "eval_08_code_error_playbook.md", "EVAL-CODE-003", "multi_hop"),
    _spec("候选清洗、reranker 和 compressor 分别解决什么问题？", "候选清洗减少重复和无效候选，reranker 用 query-doc 交互提升排序精度，compressor 过滤弱相关或过长上下文。", "compare", "eval_04_reranker_context_budget.md", "EVAL-CTX-002", "multi_hop"),
    _spec("为什么不能只用 ContextBuilder 的语义去重来替代 pre-rerank cleanup？", "pre-rerank cleanup 用于减少 reranker 输入成本，ContextBuilder 发生在精排和压缩之后，主要控制最终 prompt 的重复和预算，两者位置和目标不同。", "why", "eval_04_reranker_context_budget.md", "EVAL-CTX-005", "multi_hop"),
    _spec("评估失败时如何区分召回失败、排序失败和生成幻觉？", "如果 recalled_docs 无证据是 retrieval_miss；如果召回有证据但没进最终上下文是 ranking_error 或 budget/compression 问题；如果上下文正确但答案错是 generation_hallucination。", "howto", "eval_06_trace_evaluation_report.md", "EVAL-EVAL-004", "multi_hop"),
    _spec("为什么 Agent memory 干扰文档可以检验检索策略鲁棒性？", "它包含 memory、retrieval、context 等相似词，但主题不同；如果 RAG 索引问题被误召回到 Agent memory，说明检索或重排区分能力不足。", "why", "eval_07_agent_memory_distractors.md", "EVAL-DIST-001", "multi_hop"),
    _spec("FastAPI 上传接口和增量索引之间应该如何衔接？", "上传接口保存文件并校验类型后，应计算 hash、更新 manifest，再触发索引更新，使新增文件进入增量 embedding 和索引刷新流程。", "howto", "eval_05_fastapi_vue_code.md", "EVAL-WEB-002", "multi_hop"),
    _spec("为什么 README 里应该同时报告 Precision、Recall、MRR 和 latency？", "Precision/Recall 说明检索质量，MRR 说明首个相关结果排序，latency 说明成本；一起报告能展示效果和性能取舍。", "why", "eval_06_trace_evaluation_report.md", "EVAL-EVAL-001", "multi_hop"),
]


def build(args) -> dict[str, Any]:
    config = RAGConfig()
    docs, parent_map = _chunk_data(args.data_dir, config)
    items: list[dict[str, Any]] = []
    unresolved: list[str] = []

    for idx, spec in enumerate(SPECS, start=1):
        relevant_ids, contexts = _resolve_relevant_ids(
            spec,
            docs,
            parent_map,
            max_ids=args.max_relevant_ids,
        )
        if not relevant_ids:
            unresolved.append(f"Q{idx:03d}: {spec['question']}")
            if args.strict:
                continue
        items.append({
            "id": f"QB{idx:03d}",
            "question": spec["question"],
            "ground_truth": spec["ground_truth"],
            "qa_type": spec["qa_type"],
            "source": os.path.join(args.data_dir, spec["source"]),
            "section": spec["anchor"],
            "retrieval_focus": spec["retrieval_focus"],
            "relevant_contexts": contexts,
            "relevant_ids": relevant_ids,
        })

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    report = {
        "spec_count": len(SPECS),
        "output_count": len(items),
        "unresolved": unresolved,
        "avg_relevant_ids": (
            round(sum(len(item["relevant_ids"]) for item in items) / len(items), 3)
            if items else 0
        ),
        "qa_type_distribution": dict(Counter(item["qa_type"] for item in items)),
        "retrieval_focus_distribution": dict(Counter(item["retrieval_focus"] for item in items)),
        "source_distribution": dict(Counter(_basename(item["source"]) for item in items)),
        "output": args.output,
    }
    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 100+ TinyRAG strategy benchmark")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="benchmark/qa_strategy_benchmark.json")
    parser.add_argument("--report", default="benchmark/qa_strategy_benchmark_report.json")
    parser.add_argument("--max-relevant-ids", type=int, default=4)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    print(json.dumps(build(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Integration tests for RAGRuntime.

All tests are self-contained: no API keys, no file I/O.
FakeEmbedding and FakeLLM replace real providers.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

import run as rag_cli
from RAG.LLM import BaseModel, PromptManager
from RAG.chunking import ParentChildChunker
from RAG.context.builder import ContextBuilder, ContextBuilderConfig
from RAG.context.compressor import RelevanceFilterCompressor
from RAG.core.config import RAGConfig
from RAG.index.faiss_index import FaissVectorIndex
from RAG.memory.manager import MemoryManager
from RAG.retrievers import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    HybridRetrievalConfig,
)
from RAG.runtime.pipeline import RAGResponse, RAGRuntime
from RAG.schema import Document


# ── Fakes ─────────────────────────────────────────────────────────────────────

class FakeEmbedding:
    """Deterministic 3-dim embedding based on keyword counts."""

    def __init__(self) -> None:
        self.calls = 0

    def get_embedding(self, text: str) -> list:
        self.calls += 1
        return [
            float(text.count("git")),
            float(text.count("branch") + text.count("分支")),
            float(len(text) % 7),
        ]


class FakeLLM(BaseModel):
    def chat(self, messages: list) -> str:
        return "这是一个测试回答。"


# ── Shared fixture ─────────────────────────────────────────────────────────────

DOCS = [
    Document(
        "git branch 分支管理是版本控制的核心功能",
        metadata={"source": "git.md", "chunk_id": 0, "section": "分支", "page": None},
    ),
    Document(
        "git commit 提交代码到本地仓库",
        metadata={"source": "git.md", "chunk_id": 1, "section": "提交", "page": None},
    ),
    Document(
        "python list 和 tuple 的区别在于可变性",
        metadata={"source": "python.md", "chunk_id": 0, "section": "数据类型", "page": None},
    ),
    Document(
        "git remote 远程仓库的协作与贡献",
        metadata={"source": "git.md", "chunk_id": 2, "section": "远程", "page": None},
    ),
]


def _make_runtime():
    embedding = FakeEmbedding()
    vectors = [embedding.get_embedding(d.text) for d in DOCS]
    index = FaissVectorIndex(dimension=len(vectors[0]))
    index.add(np.array(vectors, dtype=np.float32))
    retriever = HybridRetriever(
        DenseRetriever(DOCS, index, embedding),
        BM25Retriever(DOCS),
        HybridRetrievalConfig(dense_top_k=4, sparse_top_k=4, final_top_k=4),
    )
    tmpdir = tempfile.mkdtemp()
    memory = MemoryManager(db_path=tmpdir, embedding_model=embedding)
    runtime = RAGRuntime(
        retriever=retriever,
        context_builder=ContextBuilder(ContextBuilderConfig(max_tokens=2000)),
        llm=FakeLLM(),
        prompt_manager=PromptManager(),
        memory=memory,
    )
    return runtime


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestRAGRuntimeBasic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = _make_runtime()

    def test_response_type(self):
        resp = self.runtime.query("git 分支")
        self.assertIsInstance(resp, RAGResponse)

    def test_answer_is_string(self):
        resp = self.runtime.query("git commit")
        self.assertIsInstance(resp.answer, str)
        self.assertTrue(len(resp.answer) > 0)

    def test_context_contains_source_header(self):
        resp = self.runtime.query("git branch 分支")
        self.assertIn("[source=", resp.context)

    def test_citations_have_required_fields(self):
        resp = self.runtime.query("git remote 远程仓库")
        for c in resp.citations:
            self.assertIn("source", c)
            self.assertIn("chunk_id", c)
            self.assertIn("score", c)


class TestContextOptimization(unittest.TestCase):
    def test_parent_child_chunk_ids_are_unique(self):
        text = (
            "# 第一节\n"
            "git branch 分支管理需要理解主干、特性分支和合并策略。\n\n"
            "# 第二节\n"
            "git commit 提交记录需要关注 message、diff 和回滚操作。\n"
        )
        chunker = ParentChildChunker(
            child_max_tokens=12,
            child_overlap_tokens=0,
            parent_max_tokens=30,
        )
        docs, _ = chunker.chunk(text, source="git.md", doc_id="doc-git")
        identities = [doc.identity for doc in docs]

        self.assertEqual(len(identities), len(set(identities)))
        self.assertTrue(all("chunk_uid" in doc.metadata for doc in docs))

    def test_parent_chunk_is_expanded_once(self):
        parent_map = {
            "doc-a::0": "TinyRAG 使用 parent-child chunking。小块负责召回，父块负责补充上下文。"
        }
        docs = [
            Document(
                "小块负责召回",
                score=0.9,
                metadata={"source": "tiny.md", "chunk_id": 0, "parent_id": "doc-a::0"},
            ),
            Document(
                "父块负责补充上下文",
                score=0.8,
                metadata={"source": "tiny.md", "chunk_id": 1, "parent_id": "doc-a::0"},
            ),
        ]
        builder = ContextBuilder(ContextBuilderConfig(max_tokens=1000), parent_map=parent_map)
        built = builder.build(docs)

        self.assertEqual(len(built.citations), 1)
        self.assertEqual(built.context.count("TinyRAG 使用 parent-child chunking"), 1)
        self.assertEqual(built.citations[0]["parent_id"], "doc-a::0")

    def test_context_builder_uses_embedding_for_semantic_dedup(self):
        docs = [
            Document("git branch one", score=0.9, metadata={"source": "git.md", "chunk_id": 0}),
            Document("git branch two", score=0.8, metadata={"source": "git.md", "chunk_id": 1}),
        ]
        builder = ContextBuilder(
            ContextBuilderConfig(
                max_tokens=1000,
                semantic_dedup_threshold=0.99,
                enable_semantic_dedup=True,
            ),
            embedding_model=FakeEmbedding(),
        )
        built = builder.build(docs)

        self.assertEqual(len(built.documents), 1)

    def test_relevance_filter_uses_rerank_threshold(self):
        docs = [
            Document("强相关内容", score=0.8, rerank_score=0.91),
            Document("弱相关内容", score=0.7, rerank_score=0.12),
        ]
        compressor = RelevanceFilterCompressor(threshold=0.3)
        compressed = compressor.compress(docs, "TinyRAG", token_budget=1000)

        self.assertEqual([doc.text for doc in compressed], ["强相关内容"])


class TestIndexIncrementalBuild(unittest.TestCase):
    def test_unchanged_chunks_reuse_cached_vectors_when_file_added(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "data"
            storage_dir = tmp_path / "storage"
            data_dir.mkdir()
            (data_dir / "git.md").write_text(
                "# Git\n"
                "git branch 分支管理需要理解主干、特性分支、合并策略和冲突处理。\n",
                encoding="utf-8",
            )

            old_paths = (
                rag_cli.STORAGE_DIR,
                rag_cli.INDEX_FILE,
                rag_cli.VECTORS_FILE,
            )
            old_index_dir = rag_cli._index_dir
            try:
                rag_cli.STORAGE_DIR = str(storage_dir)
                rag_cli.INDEX_FILE = str(storage_dir / "index.pkl")
                rag_cli.VECTORS_FILE = str(storage_dir / "vectors.npy")
                rag_cli._index_dir = lambda index_type: str(storage_dir / f"dense_index_{index_type}")

                embedding = FakeEmbedding()
                config = RAGConfig()
                config.parent_child.child_max_tokens = 80
                config.parent_child.parent_max_tokens = 200

                docs, index, parent_map = rag_cli.load_or_build_index(
                    embedding,
                    config,
                    data_dir=str(data_dir),
                )
                first_build_calls = embedding.calls
                self.assertEqual(index.size, len(docs))
                self.assertGreater(first_build_calls, 0)

                docs, index, parent_map = rag_cli.load_or_build_index(
                    embedding,
                    config,
                    data_dir=str(data_dir),
                )
                self.assertEqual(embedding.calls, first_build_calls)

                (data_dir / "python.md").write_text(
                    "# Python\n"
                    "python list 和 tuple 的区别主要体现在可变性、内存开销和使用场景。\n",
                    encoding="utf-8",
                )
                docs, index, parent_map = rag_cli.load_or_build_index(
                    embedding,
                    config,
                    data_dir=str(data_dir),
                )

                self.assertEqual(index.size, len(docs))
                self.assertGreater(embedding.calls, first_build_calls)
                self.assertLess(embedding.calls, first_build_calls + len(docs))
            finally:
                rag_cli.STORAGE_DIR, rag_cli.INDEX_FILE, rag_cli.VECTORS_FILE = old_paths
                rag_cli._index_dir = old_index_dir


class TestRAGRuntimeRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = _make_runtime()

    def test_git_query_retrieves_git_docs(self):
        resp = self.runtime.query("git branch 分支管理")
        sources = [c["source"] for c in resp.citations]
        self.assertIn("git.md", sources)

    def test_python_query_retrieves_python_docs(self):
        resp = self.runtime.query("python list tuple 可变性")
        sources = [c["source"] for c in resp.citations]
        self.assertIn("python.md", sources)

    def test_at_least_one_citation(self):
        resp = self.runtime.query("git commit 提交")
        self.assertGreaterEqual(len(resp.citations), 1)

    def test_pre_rerank_cleanup_deduplicates_and_caps_parent_children(self):
        rt = _make_runtime()
        rt._cfg.reranker.candidate_top_k = 10
        rt._cfg.reranker.max_candidates_per_parent = 2
        docs = [
            Document("same", score=0.1, metadata={"chunk_uid": "a", "parent_id": "p1"}),
            Document("same", score=0.9, metadata={"chunk_uid": "a", "parent_id": "p1"}),
            Document("p1 child b", score=0.8, metadata={"chunk_uid": "b", "parent_id": "p1"}),
            Document("p1 child c", score=0.7, metadata={"chunk_uid": "c", "parent_id": "p1"}),
            Document("p2 child", score=0.6, metadata={"chunk_uid": "d", "parent_id": "p2"}),
            Document("", score=1.0, metadata={"chunk_uid": "empty"}),
        ]

        cleaned = rt._prepare_rerank_candidates(docs)

        self.assertEqual([doc.identity for doc in cleaned], ["a", "b", "d"])
        self.assertEqual(cleaned[0].score, 0.9)


class TestRAGRuntimeTrace(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = _make_runtime()
        cls.resp = cls.runtime.query("git 版本控制")

    def test_trace_recalled_count(self):
        self.assertGreater(self.resp.trace.recalled_count, 0)

    def test_trace_latencies_are_positive(self):
        t = self.resp.trace
        self.assertGreaterEqual(t.retrieval_latency, 0)
        self.assertGreaterEqual(t.generation_latency, 0)

    def test_trace_prompt_tokens(self):
        self.assertGreater(self.resp.trace.prompt_tokens, 0)


class TestRAGRuntimeMemory(unittest.TestCase):
    def test_short_term_accumulates_turns(self):
        rt = _make_runtime()
        rt.query("git branch")
        rt.query("git commit")
        self.assertEqual(len(rt._memory.short_term), 2)

    def test_session_context_set_get(self):
        rt = _make_runtime()
        rt._memory.short_term.set_context("project", "test_project")
        ctx_str = rt._memory.short_term.get_context_str()
        self.assertIn("project: test_project", ctx_str)

    def test_session_context_empty(self):
        rt = _make_runtime()
        self.assertEqual(rt._memory.short_term.get_context_str(), "")


class TestRAGRuntimeLTM(unittest.TestCase):
    """Long-term memory with Chroma backend."""

    def test_save_note_returns_id(self):
        rt = _make_runtime()
        entry_id = rt._memory.save("RAG 系统使用 Chroma 做向量存储")
        self.assertIsNotNone(entry_id)
        self.assertTrue(len(entry_id) > 0)

    def test_save_note_overwrite_similar(self):
        rt = _make_runtime()
        first = rt._memory.save("长期记忆支持覆盖更新")
        second = rt._memory.save("长期记忆支持覆盖更新")
        self.assertEqual(first, second)

    def test_save_preference(self):
        rt = _make_runtime()
        entry_id = rt._memory.save("我是后端开发者", "preference")
        self.assertIsNotNone(entry_id)

    def test_list_all(self):
        rt = _make_runtime()
        rt._memory.save("git")
        entries = rt._memory.list_all()
        self.assertEqual(len(entries), 1)
        rt._memory.save("git")  # same content → overwrite
        entries = rt._memory.list_all()
        self.assertEqual(len(entries), 1)

    def test_save_two_different_notes(self):
        rt = _make_runtime()
        rt._memory.save("git git")       # [2, 0, 0]
        rt._memory.save("分支 分支")      # [0, 2, 5], cosine ≈ 0 with [2,0,0]
        entries = rt._memory.list_all()
        self.assertEqual(len(entries), 2)

    def test_list_filter_by_type(self):
        rt = _make_runtime()
        rt._memory.save("一条笔记")
        rt._memory.save("偏好内容", "preference")
        notes = rt._memory.list_all("note")
        prefs = rt._memory.list_all("preference")
        self.assertGreaterEqual(len(notes), 1)
        self.assertGreaterEqual(len(prefs), 1)

    def test_forget(self):
        rt = _make_runtime()
        entry_id = rt._memory.save("待删除内容")
        self.assertEqual(len(rt._memory.list_all()), 1)
        rt._memory.forget(entry_id)
        self.assertEqual(len(rt._memory.list_all()), 0)

    def test_preferences_flow_to_prompt(self):
        rt = _make_runtime()
        rt._memory.save("我使用 Python", "preference")
        mem_ctx = rt._memory.before_query("git 分支")
        self.assertIn("我使用 Python", mem_ctx.preferences)


if __name__ == "__main__":
    unittest.main(verbosity=2)

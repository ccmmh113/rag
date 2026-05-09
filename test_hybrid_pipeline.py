#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Integration tests for RAGRuntime.

All tests are self-contained: no API keys, no file I/O.
FakeEmbedding and FakeLLM replace real providers.
"""

import unittest

import numpy as np

from RAG.LLM import BaseModel, PromptManager
from RAG.context.builder import ContextBuilder, ContextBuilderConfig
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

    def get_embedding(self, text: str) -> list:
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


def _make_runtime() -> RAGRuntime:
    embedding = FakeEmbedding()
    vectors = [embedding.get_embedding(d.text) for d in DOCS]
    index = FaissVectorIndex(dimension=len(vectors[0]))
    index.add(np.array(vectors, dtype=np.float32))
    retriever = HybridRetriever(
        DenseRetriever(DOCS, index, embedding),
        BM25Retriever(DOCS),
        HybridRetrievalConfig(dense_top_k=4, sparse_top_k=4, final_top_k=4),
    )
    memory = MemoryManager(db_path=":memory:", index=FaissVectorIndex(dimension=3), embedding_model=embedding)
    runtime = RAGRuntime(
        retriever=retriever,
        context_builder=ContextBuilder(ContextBuilderConfig(max_tokens=2000)),
        llm=FakeLLM(),
        prompt_manager=PromptManager(),
        memory=memory,
    )
    runtime._memory.working.new_session()
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

    def test_not_from_cache_on_first_query(self):
        rt = _make_runtime()
        resp = rt.query("python list")
        self.assertFalse(resp.from_cache)

    def test_context_contains_source_header(self):
        resp = self.runtime.query("git branch 分支")
        self.assertIn("[source=", resp.context)

    def test_citations_have_required_fields(self):
        resp = self.runtime.query("git remote 远程仓库")
        for c in resp.citations:
            self.assertIn("source", c)
            self.assertIn("chunk_id", c)
            self.assertIn("score", c)


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

    def test_trace_has_ltm_entry_id(self):
        self.assertIn("ltm_entry_id", self.resp.trace.metadata)

    def test_ltm_entry_id_is_empty_on_first_query(self):
        """LTM is no longer auto-populated — entry_id is empty string."""
        rt = _make_runtime()
        resp = rt.query("git 版本控制")
        self.assertEqual(resp.trace.metadata["ltm_entry_id"], "")


class TestRAGRuntimeMemory(unittest.TestCase):
    def test_short_term_accumulates_turns(self):
        rt = _make_runtime()
        rt.query("git branch")
        rt.query("git commit")
        self.assertEqual(len(rt._memory.short_term), 2)

    def test_working_memory_increments_query_count(self):
        rt = _make_runtime()
        rt.query("git 分支")
        rt.query("git 提交")
        self.assertEqual(rt._memory.working.retrieval.query_count, 2)

    def test_new_session_resets_working_state(self):
        rt = _make_runtime()
        rt.query("git branch")
        rt._memory.working.new_session()
        self.assertEqual(rt._memory.working.retrieval.query_count, 0)


class TestRAGRuntimeLTMManual(unittest.TestCase):
    """Long-term memory requires explicit user action to save."""

    def test_save_last_turn_returns_id(self):
        rt = _make_runtime()
        rt.query("git branch 分支管理")
        entry_id = rt._memory.save_last_turn()
        self.assertIsNotNone(entry_id)
        self.assertTrue(len(entry_id) > 0)

    def test_save_last_turn_without_turns_returns_none(self):
        rt = _make_runtime()
        entry_id = rt._memory.save_last_turn()
        self.assertIsNone(entry_id)

    def test_ltm_cache_hit_after_save(self):
        rt = _make_runtime()
        rt.query("git branch 分支管理")
        rt._memory.save_last_turn()
        resp = rt.query("git branch 分支管理")
        self.assertTrue(resp.from_cache)

    def test_save_last_turn_dedup(self):
        rt = _make_runtime()
        rt.query("git branch 分支管理")
        first = rt._memory.save_last_turn()
        self.assertIsNotNone(first)
        # Same query again should be dedup'd
        rt.query("git branch 分支管理")
        second = rt._memory.save_last_turn()
        self.assertIsNone(second)

    def test_save_note_returns_id(self):
        rt = _make_runtime()
        entry_id = rt._memory.save_note("RAG 系统使用 FAISS 做向量检索")
        self.assertIsNotNone(entry_id)
        self.assertTrue(len(entry_id) > 0)

    def test_save_note_dedup(self):
        rt = _make_runtime()
        note = "长期记忆需要用户手动保存"
        first = rt._memory.save_note(note)
        self.assertIsNotNone(first)
        second = rt._memory.save_note(note)
        self.assertIsNone(second)


if __name__ == "__main__":
    unittest.main(verbosity=2)

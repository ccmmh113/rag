#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run.py — TinyRAG 端到端交互入口

使用步骤：
  1. 把文档（.md / .txt / .pdf）放入 ./data/ 目录
  2. 在 .env 或环境变量中设置 OPENAI_API_KEY（以及可选的 OPENAI_BASE_URL）
python run.py

首次运行：自动读取 data/ 并嵌入，索引保存到 storage/
后续运行：直接加载 storage/，跳过重新嵌入

交互命令：
  exit        退出
  trace       打印上一次查询的完整 trace 元数据
  verify <id> 将某条长期记忆标记为可信（入参为 entry_id）
  new         开启新的 working memory session
"""

from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()
import json
import numpy as np
import os
import pickle
import sys
from textwrap import dedent

from RAG.Embeddings import  EmbeddingFactory
from RAG.LLM import OpenAIChat, PromptManager
from RAG.context.builder import ContextBuilder, ContextBuilderConfig
from RAG.index.faiss_index import FaissVectorIndex
from RAG.memory.manager import MemoryManager
from RAG.retrievers import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    HybridRetrievalConfig,
)
from RAG.runtime.pipeline import RAGRuntime
from RAG.trace.store import TraceStore
from RAG.utils import ReadFiles

STORAGE_DIR = "storage"
INDEX_FILE = os.path.join(STORAGE_DIR, "index.pkl")
TRACE_FILE = os.path.join(STORAGE_DIR, "traces.jsonl")
LTM_DB = os.path.join(STORAGE_DIR, "ltm.db")
DEFAULT_CHAT_MODEL = "gpt-5.2"

COMMANDS = {
    "help": "显示内置命令",
    "trace": "打印上一次查询的 trace 元数据",
    "save": "保存最近一轮问答到长期记忆",
    "note <文本>": "将一段感悟/总结存入长期记忆",
    "verify <id>": "将某条长期记忆标记为可信",
    "new": "开启新的 working memory session，并清空短期记忆",
    "exit": "退出程序",
}


# ── 索引构建 / 加载 ────────────────────────────────────────────────────────────

def _build_index(embedding_model):
    print("正在从 ./data/ 构建索引...")
    docs = ReadFiles("./data").get_documents(max_token_len=600, cover_content=120)
    if not docs:
        raise RuntimeError("data/ 目录下没有找到文档，请先放入 .md/.txt/.pdf 文件。")
    print(f"  已加载 {len(docs)} 个文本块")

    vectors = []
    for i, doc in enumerate(docs, 1):
        vectors.append(embedding_model.get_embedding(doc.text))
        if i % 20 == 0:
            print(f"嵌入进度 {i}/{len(docs)}")

    dim = len(vectors[0])
    index = FaissVectorIndex(dimension=dim)
    index.add(np.array(vectors, dtype=np.float32))

    os.makedirs(STORAGE_DIR, exist_ok=True)
    with open(INDEX_FILE, "wb") as f:
        pickle.dump({"docs": docs, "dim": dim}, f)
    index.save(os.path.join(STORAGE_DIR, "dense_index"))
    print(f"  索引已保存到 {STORAGE_DIR}/\n")
    return docs, index


def _load_index():
    print(f"从 {STORAGE_DIR}/ 加载索引...")
    with open(INDEX_FILE, "rb") as f:
        saved = pickle.load(f)
    docs = saved["docs"]
    index = FaissVectorIndex(dimension=saved["dim"])
    index.load(os.path.join(STORAGE_DIR, "dense_index"))
    print(f"  已加载 {len(docs)} 个文本块\n")
    return docs, index


def load_or_build_index(embedding_model):
    if os.path.exists(INDEX_FILE):
        return _load_index()
    return _build_index(embedding_model)


def _mask_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        return "未设置"
    if len(value) <= 8:
        return "已设置"
    return f"{value[:4]}...{value[-4:]}"


def _print_commands() -> None:
    print("内置命令:")
    for command, description in COMMANDS.items():
        print(f"  {command:<12} {description}")
    print()


def _print_banner(docs_count: int) -> None:
    base_url = os.getenv("OPENAI_BASE_URL") or "默认 OpenAI API"
    chat_model = os.getenv("OPENAI_MODEL") or DEFAULT_CHAT_MODEL
    index_status = "已加载" if os.path.exists(INDEX_FILE) else "本次新建"
    trace_status = TRACE_FILE if os.path.exists(TRACE_FILE) else "暂无 trace 文件"
    print(dedent(f"""
    ============================================================
                             (\__/)
                             ( •_•)
                                / >🥕

                         RAG 引擎启动完成
    ============================================================
      文档块数量 : {docs_count}
      索引文件   : {INDEX_FILE} ({index_status})
      Trace 文件 : {trace_status}
      长期记忆库 : {LTM_DB}

      Embedding  : bge (本地)
      Chat LLM   : OpenAIChat / {chat_model}
    ============================================================
    """).strip())
    print()
    _print_commands()


# ── Runtime 组装 ───────────────────────────────────────────────────────────────

def build_runtime(docs, index, embedding_model) -> RAGRuntime:
    chat_model = os.getenv("OPENAI_MODEL") or DEFAULT_CHAT_MODEL
    retriever = HybridRetriever(
        DenseRetriever(docs, index, embedding_model),
        BM25Retriever(docs),
        HybridRetrievalConfig(dense_top_k=20, sparse_top_k=20, final_top_k=10),
    )
    context_builder = ContextBuilder(
        ContextBuilderConfig(max_tokens=3000, min_score=0.0)
    )
    memory = MemoryManager(
        db_path=LTM_DB,
        index=FaissVectorIndex(dimension=768),
        embedding_model=embedding_model,
    )
    return RAGRuntime(
        retriever=retriever,
        context_builder=context_builder,
        llm=OpenAIChat(model=chat_model),
        prompt_manager=PromptManager(),
        memory=memory,
        trace_store=TraceStore(path=TRACE_FILE),
    )


# ── 交互 REPL ─────────────────────────────────────────────────────────────────

def main():
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    # embedding = OpenAIEmbedding()
    embedding = EmbeddingFactory.create("bge")
    docs, index = load_or_build_index(embedding)
    runtime = build_runtime(docs, index, embedding)
    runtime._memory.working.new_session()
    _print_banner(len(docs))
    print("输入问题开始查询，输入 help 查看命令。\n")

    last_trace = None

    while True:
        try:
            line = input("->Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not line:
            continue

        if line.lower() == "exit":
            break

        if line.lower() == "help":
            _print_commands()
            continue

        if line.lower() == "trace":
            if last_trace:
                trace_dict = last_trace.to_dict()
                # Remove verbose fields unless --verbose
                if not verbose:
                    for key in ["recalled_docs"]:
                        trace_dict.pop(key, None)
                print(json.dumps(trace_dict, ensure_ascii=False, indent=2))
            else:
                print("  (暂无 trace，先提问一次)")
            continue

        if line.lower() == "save":
            entry_id = runtime._memory.save_last_turn()
            if entry_id:
                print(f"  ✓ 已保存到长期记忆: {entry_id}")
            else:
                print(f"  ✗ 未保存（无对话记录或已有相似内容）")
            continue

        if line.lower().startswith("note "):
            content = line.split(" ", 1)[1].strip()
            if content:
                entry_id = runtime._memory.save_note(content)
                if entry_id:
                    print(f"  ✓ 已保存笔记到长期记忆: {entry_id}")
                else:
                    print(f"  ✗ 未保存（已有相似内容）")
            else:
                print("  用法: note <文本内容>")
            continue

        if line.lower().startswith("verify "):
            entry_id = line.split(" ", 1)[1].strip()
            ok = runtime._memory.verify(entry_id)
            print(f"  {'✓ 已标记为可信' if ok else '✗ 未找到该 entry_id'}: {entry_id}")
            continue

        if line.lower() == "new":
            runtime._memory.working.new_session()
            runtime._memory.short_term.clear()
            print("  已开启新 session\n")
            continue

        try:
            response = runtime.query(line)
        except Exception as exc:
            if exc.__class__.__name__ == "PermissionDeniedError":
                continue
            raise
        print(f"\nA: {response.answer}\n")

        if response.from_cache:
            print("  [长期记忆缓存命中]\n")
        else:
            t = response.trace
            ltm_id = t.metadata.get("ltm_entry_id", "")
            print(
                f"  [检索 {t.recalled_count} 篇 → {t.reranked_count} 篇 | "
                f"{t.prompt_tokens} tokens | 生成 {t.generation_latency:.0f}ms | "
                f"ltm_id={ltm_id}]"
            )
            # if response.citations:
            #     sources = list({c["source"] for c in response.citations})
            #     print(f"  [引用来源: {', '.join(sources[:4])}]")
            # print()

        last_trace = response.trace


if __name__ == "__main__":
    main()

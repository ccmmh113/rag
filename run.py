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
  note <文本>   保存一条笔记
  pref <文本>   保存用户偏好/背景
  forget <id>   删除一条记忆
  list          列出所有记忆
  context       查看当前会话上下文
  context set <key> <value>  设置会话上下文
  trace         打印上一次查询的完整 trace 元数据
  exit          退出
"""

from __future__ import annotations
from dotenv import load_dotenv

from RAG.context import ContextCompressor

load_dotenv()
import hashlib
import json
import numpy as np
import os
import pickle
import sys
from textwrap import dedent

from RAG.Embeddings import EmbeddingFactory
from RAG.LLM import OpenAIChat, PromptManager
from RAG.Reranker import BgeReranker
from RAG.context.builder import ContextBuilder, ContextBuilderConfig
from RAG.core.config import RAGConfig
from RAG.index.faiss_index import FaissVectorIndex
from RAG.memory.manager import MemoryManager
from RAG.retrievers import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    HybridRetrievalConfig,
)
from RAG.router.router import PolicyRouter
from RAG.runtime.pipeline import RAGRuntime
from RAG.trace.store import TraceStore
from RAG.utils import ReadFiles

STORAGE_DIR = "storage"
INDEX_FILE = os.path.join(STORAGE_DIR, "index.pkl")
VECTORS_FILE = os.path.join(STORAGE_DIR, "vectors.npy")
TRACE_FILE = os.path.join(STORAGE_DIR, "traces.jsonl")
LTM_DIR = os.path.join(STORAGE_DIR, "ltm")
INDEX_SCHEMA_VERSION = 2


def _index_dir(index_type: str) -> str:
    return os.path.join(STORAGE_DIR, f"dense_index_{index_type}")
DEFAULT_CHAT_MODEL = "gpt-5.2"

COMMANDS = {
    "help": "显示内置命令",
    "note <文本>":"保存一条笔记到长期记忆",
    "pref <文本>":"保存用户偏好/背景到长期记忆",
    "forget <id>": "删除一条记忆",
    "list": "列出所有记忆",
    "context": "查看当前会话上下文",
    "context set <key> <value>": "设置会话上下文",
    "filter [key=value ...]": "设置元数据过滤 (source=xxx section=xxx)，filter off 关闭",
    "trace": "打印上一次查询的 trace 元数据",
    "exit": "退出程序",
}


# ── 索引构建 / 加载 ────────────────────────────────────────────────────────────

def _parse_filter(args: str) -> dict:
    """Parse 'key1=value1 key2=value2' into a metadata_filter dict."""
    result = {}
    for part in args.split():
        if "=" in part:
            key, value = part.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def _normalise_path(path: str) -> str:
    return os.path.normpath(path).replace("\\", "/")


def _sha1_file(path: str) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def _data_manifest(data_dir: str, config: RAGConfig) -> dict:
    files = []
    reader = ReadFiles(data_dir)
    for path in sorted(reader.file_list):
        stat = os.stat(path)
        files.append({
            "path": _normalise_path(os.path.abspath(path)),
            "size": stat.st_size,
            "sha1": _sha1_file(path),
        })
    pc = config.parent_child
    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "data_dir": _normalise_path(os.path.abspath(data_dir)),
        "files": files,
        "parent_child": {
            "child_max_tokens": pc.child_max_tokens,
            "child_overlap_tokens": pc.child_overlap_tokens,
            "parent_max_tokens": pc.parent_max_tokens,
        },
    }


def _load_saved_state() -> dict | None:
    if not os.path.exists(INDEX_FILE):
        return None
    with open(INDEX_FILE, "rb") as f:
        return pickle.load(f)


def _vector_cache_from_state(saved: dict | None) -> dict[tuple[str, str], np.ndarray]:
    if not saved or not os.path.exists(VECTORS_FILE):
        return {}
    try:
        old_docs = saved.get("docs", [])
        old_vectors = np.load(VECTORS_FILE)
    except Exception:
        return {}
    cache: dict[tuple[str, str], np.ndarray] = {}
    for doc, vector in zip(old_docs, old_vectors):
        text_hash = doc.metadata.get("text_hash") or _sha1_text(doc.text)
        cache[(doc.identity, text_hash)] = vector.astype(np.float32)
    return cache


def _stamp_chunk_hashes(docs):
    for doc in docs:
        doc.metadata["text_hash"] = _sha1_text(doc.text)


def _build_index(embedding_model, config: RAGConfig, data_dir: str = "./data"):
    pc = config.parent_child
    print(f"正在从 {data_dir} 构建/更新索引... (index_type={config.index.index_type})")
    manifest = _data_manifest(data_dir, config)
    previous_state = _load_saved_state()
    vector_cache = _vector_cache_from_state(previous_state)

    reader = ReadFiles(data_dir)
    child_docs, parent_map = reader.get_parent_child_documents(
        child_max_tokens=pc.child_max_tokens,
        child_overlap_tokens=pc.child_overlap_tokens,
        parent_max_tokens=pc.parent_max_tokens,
    )
    if not child_docs:
        raise RuntimeError(f"{data_dir} 目录下没有找到文档，请先放入 .md/.txt/.pdf 文件。")
    _stamp_chunk_hashes(child_docs)
    print(f"  子块数量: {len(child_docs)}, 父块数量: {len(parent_map)}")

    vectors = []
    reused = 0
    embedded = 0
    for i, doc in enumerate(child_docs, 1):
        cache_key = (doc.identity, doc.metadata["text_hash"])
        cached = vector_cache.get(cache_key)
        if cached is not None:
            vectors.append(cached)
            reused += 1
        else:
            vectors.append(embedding_model.get_embedding(doc.text))
            embedded += 1
        if embedded and embedded % 20 == 0:
            print(f"新增嵌入进度 {embedded} 个 (扫描 {i}/{len(child_docs)})")

    print(f"  向量复用: {reused}, 新增嵌入: {embedded}")

    vectors_arr = np.array(vectors, dtype=np.float32)
    dim = vectors_arr.shape[1]

    os.makedirs(STORAGE_DIR, exist_ok=True)
    np.save(VECTORS_FILE, vectors_arr)
    with open(INDEX_FILE, "wb") as f:
        pickle.dump({
            "docs": child_docs,
            "dim": dim,
            "parent_map": parent_map,
            "manifest": manifest,
            "schema_version": INDEX_SCHEMA_VERSION,
            "identity_version": 2,
        }, f)

    index = FaissVectorIndex(dimension=dim, index_type=config.index.index_type)
    index.add(vectors_arr)
    index.save(_index_dir(config.index.index_type))
    print(f"  索引已保存到 {_index_dir(config.index.index_type)}\n")
    return child_docs, index, parent_map


def _load_index(config: RAGConfig):
    index_dir = _index_dir(config.index.index_type)
    print(f"从 {index_dir} 加载索引...")
    saved = _load_saved_state()
    if saved is None:
        raise FileNotFoundError(INDEX_FILE)
    docs = saved["docs"]
    parent_map = saved.get("parent_map", {})
    index = FaissVectorIndex(dimension=saved["dim"], index_type=config.index.index_type)
    index.load(index_dir)
    print(f"  已加载 {len(docs)} 个子块, {len(parent_map)} 个父块 (index_type={config.index.index_type})\n")
    return docs, index, parent_map


def load_or_build_index(embedding_model, config: RAGConfig, data_dir: str = "./data"):
    index_dir = _index_dir(config.index.index_type)
    current_manifest = _data_manifest(data_dir, config)
    saved = _load_saved_state()
    if (
        os.path.exists(index_dir)
        and saved
        and saved.get("schema_version") == INDEX_SCHEMA_VERSION
        and saved.get("manifest") == current_manifest
    ):
        return _load_index(config)

    if saved and os.path.exists(index_dir):
        print("检测到 data/ 文档或分块配置变化，开始增量更新索引...")

    # 该类型索引不存在，尝试从缓存向量重建（无需重新 embed）
    if (
        saved
        and os.path.exists(VECTORS_FILE)
        and saved.get("schema_version") == INDEX_SCHEMA_VERSION
        and saved.get("manifest") == current_manifest
    ):
        print(f"从缓存向量重建 {config.index.index_type} 索引 (无需重新嵌入)...")
        docs = saved["docs"]
        parent_map = saved.get("parent_map", {})
        vectors_arr = np.load(VECTORS_FILE)
        index = FaissVectorIndex(dimension=saved["dim"], index_type=config.index.index_type)
        index.add(vectors_arr)
        index.save(index_dir)
        print(f"  {config.index.index_type} 索引已保存到 {index_dir}\n")
        return docs, index, parent_map

    return _build_index(embedding_model, config, data_dir=data_dir)


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
        print(f"  {command:<24} {description}")
    print()


def _print_banner(docs_count: int) -> None:
    chat_model = os.getenv("OPENAI_MODEL") or DEFAULT_CHAT_MODEL
    print(dedent(f"""
    ============================================================
                             (\\__/)
                             ( •_•)
                                / >🥕

                         RAG 引擎启动完成
    ============================================================
      文档块数量 : {docs_count}
      索引文件   : {INDEX_FILE}
      Trace 文件 : {TRACE_FILE}
      长期记忆库 : {LTM_DIR}

      Embedding  : bge (本地)
      Chat LLM   : OpenAIChat / {chat_model}
    ============================================================
    """).strip())
    print()
    _print_commands()


# ── Runtime 组装 ───────────────────────────────────────────────────────────────

def build_runtime(docs, index, embedding_model, parent_map, config: RAGConfig, reranker=None) -> RAGRuntime:
    chat_model = os.getenv("OPENAI_MODEL") or DEFAULT_CHAT_MODEL
    ret_cfg = config.retrieval
    ctx_cfg = config.context
    compress_cfg=config.compression
    retriever = HybridRetriever(
        DenseRetriever(docs, index, embedding_model),
        BM25Retriever(docs),
        HybridRetrievalConfig(
            dense_top_k=ret_cfg.dense_top_k,
            sparse_top_k=ret_cfg.sparse_top_k,
            final_top_k=ret_cfg.final_top_k,
            fusion=ret_cfg.fusion,
            rrf_k=ret_cfg.rrf_k,
            dense_weight=ret_cfg.dense_weight,
            sparse_weight=ret_cfg.sparse_weight,
            parallel=ret_cfg.parallel,
        ),
    )
    context_builder = ContextBuilder(
        ContextBuilderConfig(
            max_tokens=ctx_cfg.max_tokens,
            min_score=ctx_cfg.min_score,
            overlap_chars=ctx_cfg.overlap_chars,
            semantic_dedup_threshold=ctx_cfg.semantic_dedup_threshold,
            enable_semantic_dedup=ctx_cfg.enable_semantic_dedup,
        ),
        embedding_model=embedding_model,
        parent_map=parent_map,
    )
    memory = MemoryManager(
        db_path=LTM_DIR,
        embedding_model=embedding_model,
    )
    compress= ContextCompressor(compress_cfg,OpenAIChat(model=chat_model))
    return RAGRuntime(
        retriever=retriever,
        context_builder=context_builder,
        llm=OpenAIChat(model=chat_model),
        prompt_manager=PromptManager(),
        memory=memory,
        compressor=compress,
        router=PolicyRouter.default(),
        reranker=reranker,
        trace_store=TraceStore(path=TRACE_FILE),
        config=config,
    )


# ── 交互 REPL ─────────────────────────────────────────────────────────────────

def main():
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    # ── 统一配置 ────────────────────────────────────────────────────────
    config = RAGConfig()

    # CLI 覆盖
    if "--index-type" in sys.argv:
        idx = sys.argv.index("--index-type") + 1
        if idx < len(sys.argv):
            config.index.index_type = sys.argv[idx]

    use_reranker = "--no-reranker" not in sys.argv

    # ── Reranker ────────────────────────────────────────────────────────
    reranker = None
    if use_reranker:
        try:
            reranker = BgeReranker()
            print("BGE Reranker 已加载")
        except Exception as exc:
            print(f"BGE Reranker 不可用: {exc}")

    embedding = EmbeddingFactory.create("bge")
    docs, index, parent_map = load_or_build_index(embedding, config)
    runtime = build_runtime(docs, index, embedding, parent_map, config, reranker)
    _print_banner(len(docs))
    print("输入问题开始查询，输入 help 查看命令。\n")

    last_trace = None
    current_filter = None

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
                if not verbose:
                    for key in ["recalled_docs"]:
                        trace_dict.pop(key, None)
                print(json.dumps(trace_dict, ensure_ascii=False, indent=2))
            else:
                print("  (暂无 trace，先提问一次)")
            continue

        if line.lower() == "list":
            entries = runtime._memory.list_all()
            if not entries:
                print("  (暂无记忆)")
            else:
                for e in entries:
                    type_tag = "📝" if e.entry_type == "note" else "👤"
                    print(f"  [{e.id}] {type_tag} {e.content[:80]}")
            continue

        if line.lower().startswith("note "):
            content = line.split(" ", 1)[1].strip()
            if content:
                entry_id = runtime._memory.save(content, "note")
                print(f"  ✓ 笔记已保存: {entry_id}")
            else:
                print("  用法: note <文本内容>")
            continue

        if line.lower().startswith("pref "):
            content = line.split(" ", 1)[1].strip()
            if content:
                entry_id = runtime._memory.save(content, "preference")
                print(f"  ✓ 偏好已保存: {entry_id}")
            else:
                print("  用法: pref <文本内容>")
            continue

        if line.lower().startswith("forget "):
            entry_id = line.split(" ", 1)[1].strip()
            runtime._memory.forget(entry_id)
            print(f"  ✓ 已删除: {entry_id}")
            continue

        if line.lower() == "context":
            ctx = runtime._memory.short_term.get_context_str()
            if ctx:
                print(f"  当前会话上下文:\n{ctx}")
            else:
                print("  (未设置会话上下文)")
            continue

        if line.lower().startswith("context set "):
            parts = line.split(" ", 2)
            if len(parts) >= 4:
                key = parts[2]
                value = parts[3] if len(parts) > 3 else ""
                runtime._memory.short_term.set_context(key, value)
                print(f"  ✓ 已设置: {key} = {value}")
            else:
                print("  用法: context set <key> <value>")
            continue

        if line.lower().startswith("filter"):
            args = line.split(" ", 1)[1] if " " in line else ""
            if not args or args.lower() in ("off", "clear", "none"):
                current_filter = None
                print("  ✓ 过滤已关闭")
            elif args.lower() in ("show",):
                if current_filter:
                    print(f"  当前过滤: {current_filter}")
                else:
                    print("  (未设置过滤)")
            else:
                current_filter = _parse_filter(args)
                if current_filter:
                    print(f"  ✓ 过滤已设置: {current_filter}")
                else:
                    print("  用法: filter source=xxx [section=xxx ...]，filter off 关闭")
            continue

        try:
            response = runtime.query(line, metadata_filter=current_filter)
        except Exception as exc:
            if exc.__class__.__name__ == "PermissionDeniedError":
                continue
            raise
        print(f"\nA: {response.answer}\n")

        # t = response.trace
        # print(
        #     f"  [检索 {t.recalled_count} 篇 → {t.reranked_count} 篇 | "
        #     f"{t.prompt_tokens} tokens | 生成 {t.generation_latency:.0f}ms]"
        # )

        last_trace = response.trace


if __name__ == "__main__":
    main()

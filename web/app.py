#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import threading
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import run as rag_cli
from RAG.Embeddings import EmbeddingFactory
from RAG.Reranker import BgeReranker
from RAG.core.config import RAGConfig
from RAG.trace.store import TraceStore


ROOT_DIR = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"


def _to_dict(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, list):
        return [_to_dict(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_dict(item) for key, item in value.items()}
    return value


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    metadata_filter: Optional[Dict[str, Any]] = None


class MemoryRequest(BaseModel):
    content: str = Field(..., min_length=1)
    entry_type: str = Field(default="note", pattern="^(note|preference)$")


class ConfigPatch(BaseModel):
    dense_top_k: Optional[int] = Field(default=None, ge=1, le=200)
    sparse_top_k: Optional[int] = Field(default=None, ge=1, le=200)
    final_top_k: Optional[int] = Field(default=None, ge=1, le=200)
    reranker_top_k: Optional[int] = Field(default=None, ge=1, le=50)
    context_max_tokens: Optional[int] = Field(default=None, ge=256, le=16000)
    dense_weight: Optional[float] = Field(default=None, ge=0, le=1)
    sparse_weight: Optional[float] = Field(default=None, ge=0, le=1)
    fusion: Optional[str] = Field(default=None, pattern="^(rrf|weighted)$")
    index_type: Optional[str] = Field(default=None, pattern="^(flat_ip|hnsw)$")
    use_reranker: Optional[bool] = None


class RuntimeManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.config = RAGConfig()
        self.runtime = None
        self.embedding = None
        self.docs = []
        self.parent_map = {}
        self.use_reranker = True
        self.last_error: Optional[str] = None

    def status(self) -> Dict[str, Any]:
        index_dir = ROOT_DIR / rag_cli._index_dir(self.config.index.index_type)
        data_dir = ROOT_DIR / "data"
        data_files = []
        if data_dir.exists():
            data_files = [
                str(path.relative_to(ROOT_DIR))
                for path in data_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in {".md", ".txt", ".pdf"}
            ]
        return {
            "ready": self.runtime is not None,
            "docs_count": len(self.docs),
            "parent_count": len(self.parent_map),
            "index_type": self.config.index.index_type,
            "index_exists": index_dir.exists(),
            "data_files": data_files,
            "openai_key": bool(os.getenv("OPENAI_API_KEY")),
            "chat_model": os.getenv("OPENAI_MODEL") or rag_cli.DEFAULT_CHAT_MODEL,
            "use_reranker": self.use_reranker,
            "last_error": self.last_error,
        }

    def ensure_runtime(self):
        with self._lock:
            if self.runtime is not None:
                return self.runtime
            try:
                if self.embedding is None:
                    self.embedding = EmbeddingFactory.create("bge")
                docs, index, parent_map = rag_cli.load_or_build_index(self.embedding, self.config)
                reranker = self._load_reranker() if self.use_reranker else None
                self.runtime = rag_cli.build_runtime(
                    docs=docs,
                    index=index,
                    embedding_model=self.embedding,
                    parent_map=parent_map,
                    config=self.config,
                    reranker=reranker,
                )
                self.docs = docs
                self.parent_map = parent_map
                self.last_error = None
                return self.runtime
            except Exception as exc:
                self.last_error = str(exc)
                raise

    def build_index(self) -> Dict[str, Any]:
        with self._lock:
            try:
                if self.embedding is None:
                    self.embedding = EmbeddingFactory.create("bge")
                docs, index, parent_map = rag_cli._build_index(self.embedding, self.config)
                reranker = self._load_reranker() if self.use_reranker else None
                self.runtime = rag_cli.build_runtime(
                    docs=docs,
                    index=index,
                    embedding_model=self.embedding,
                    parent_map=parent_map,
                    config=self.config,
                    reranker=reranker,
                )
                self.docs = docs
                self.parent_map = parent_map
                self.last_error = None
                return self.status()
            except Exception as exc:
                self.last_error = str(exc)
                raise

    def patch_config(self, patch: ConfigPatch) -> Dict[str, Any]:
        data = patch.model_dump(exclude_none=True)
        with self._lock:
            for key in ("dense_top_k", "sparse_top_k", "final_top_k", "dense_weight", "sparse_weight", "fusion"):
                if key in data:
                    setattr(self.config.retrieval, key, data[key])
            if "reranker_top_k" in data:
                self.config.reranker.top_k = data["reranker_top_k"]
            if "context_max_tokens" in data:
                self.config.context.max_tokens = data["context_max_tokens"]
            if "index_type" in data:
                self.config.index.index_type = data["index_type"]
            if "use_reranker" in data:
                self.use_reranker = data["use_reranker"]
            self.runtime = None
            return self.config_snapshot()

    def config_snapshot(self) -> Dict[str, Any]:
        return {
            "retrieval": _to_dict(self.config.retrieval),
            "reranker": _to_dict(self.config.reranker),
            "context": _to_dict(self.config.context),
            "index": _to_dict(self.config.index),
            "use_reranker": self.use_reranker,
        }

    def _load_reranker(self):
        try:
            return BgeReranker()
        except Exception as exc:
            self.last_error = f"Reranker unavailable: {exc}"
            return None


manager = RuntimeManager()
app = FastAPI(title="TinyRAG Web Console", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
def status() -> Dict[str, Any]:
    return manager.status()


@app.post("/api/index/build")
def build_index() -> Dict[str, Any]:
    try:
        return manager.build_index()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/config")
def get_config() -> Dict[str, Any]:
    return manager.config_snapshot()


@app.patch("/api/config")
def patch_config(patch: ConfigPatch) -> Dict[str, Any]:
    return manager.patch_config(patch)


@app.post("/api/chat")
def chat(payload: ChatRequest) -> Dict[str, Any]:
    try:
        runtime = manager.ensure_runtime()
        response = runtime.query(payload.question, metadata_filter=payload.metadata_filter)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "answer": response.answer,
        "context": response.context,
        "citations": response.citations,
        "trace": response.trace.to_dict(),
    }


@app.get("/api/traces")
def traces(limit: int = 20) -> Dict[str, Any]:
    store = TraceStore(path=rag_cli.TRACE_FILE)
    recent = [trace.to_dict() for trace in store.load_recent(limit)]
    return {
        "items": list(reversed(recent)),
        "latency": store.avg_latency(),
        "strategy_distribution": store.strategy_distribution(),
        "llm_cache_rate": store.llm_cache_rate(),
    }


@app.get("/api/memories")
def memories(entry_type: Optional[str] = None) -> Dict[str, Any]:
    try:
        runtime = manager.ensure_runtime()
        entries = runtime._memory.list_all(entry_type)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"items": [_to_dict(entry) for entry in entries]}


@app.post("/api/memories")
def create_memory(payload: MemoryRequest) -> Dict[str, Any]:
    try:
        runtime = manager.ensure_runtime()
        entry_id = runtime._memory.save(payload.content, payload.entry_type)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"id": entry_id}


@app.delete("/api/memories/{entry_id}")
def delete_memory(entry_id: str) -> Dict[str, Any]:
    try:
        runtime = manager.ensure_runtime()
        runtime._memory.forget(entry_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"ok": True}

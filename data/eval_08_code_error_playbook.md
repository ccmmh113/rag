# Eval Corpus 08: Code Tokens, Error Codes, and Debug Playbook

## EVAL-CODE-001: exact code token retrieval

以下代码片段用于测试 BM25 对精确 token 的检索能力。问题中如果包含 `_vector_cache_from_state`、`chunk_uid`、`RAG_INDEX_STALE`、`candidate_top_k`，BM25 应该能稳定命中。

```python
def _vector_cache_from_state(saved: dict | None) -> dict[tuple[str, str], np.ndarray]:
    if not saved or not os.path.exists(VECTORS_FILE):
        return {}
    old_docs = saved.get("docs", [])
    old_vectors = np.load(VECTORS_FILE)
    cache = {}
    for doc, vector in zip(old_docs, old_vectors):
        text_hash = doc.metadata.get("text_hash")
        cache[(doc.identity, text_hash)] = vector.astype(np.float32)
    return cache
```

这个函数的作用是从旧索引状态和 vectors.npy 中构造向量缓存，用于增量 embedding 复用。

## EVAL-CODE-002: candidate cleanup code

```python
def _prepare_rerank_candidates(documents: list[Document]) -> list[Document]:
    unique_by_id = {}
    for doc in documents:
        if not doc.text.strip():
            continue
        current = unique_by_id.get(doc.identity)
        if current is None or doc.score > current.score:
            unique_by_id[doc.identity] = doc

    parent_counts = defaultdict(int)
    cleaned = []
    for doc in sorted(unique_by_id.values(), key=lambda d: d.score, reverse=True):
        parent_key = doc.metadata.get("parent_id") or doc.identity
        if parent_counts[parent_key] >= max_candidates_per_parent:
            continue
        parent_counts[parent_key] += 1
        cleaned.append(doc)
        if len(cleaned) >= candidate_top_k:
            break
    return cleaned
```

这个函数不会替代 reranker。它只做保守清洗：去除空文本、重复 identity 和过多同 parent child，减少 cross-encoder 的输入成本。

## EVAL-CODE-003: error code glossary

内部错误码说明：

```text
RAG_INDEX_STALE: manifest changed and dense index must be refreshed
RAG_EMPTY_CORPUS: data directory has no supported documents
RAG_VECTOR_DIM_MISMATCH: vector dimension does not match FAISS index dimension
RAG_UNSUPPORTED_FILE_TYPE: upload file suffix is not .md, .txt, or .pdf
RAG_RERANKER_UNAVAILABLE: BGE reranker failed to load
```

这些错误码适合测试精确检索。如果 query 是 `RAG_VECTOR_DIM_MISMATCH 怎么处理`，BM25 通常比纯语义检索更可靠。

## EVAL-CODE-004: JSON config sample

```json
{
  "retrieval": {
    "fusion": "rrf",
    "dense_top_k": 50,
    "sparse_top_k": 50,
    "final_top_k": 50,
    "dense_weight": 0.9,
    "sparse_weight": 0.1
  },
  "reranker": {
    "candidate_top_k": 30,
    "max_candidates_per_parent": 2,
    "top_k": 8
  },
  "compression": {
    "strategy": "relevance_filter",
    "relevance_threshold": 0.3
  }
}
```

如果用户问 `max_candidates_per_parent` 的作用，答案是限制同一个 parent_id 下送入 reranker 的 child 数量，避免一个父块占满候选集。


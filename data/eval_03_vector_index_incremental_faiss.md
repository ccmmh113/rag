# Eval Corpus 03: FAISS Index and Incremental Embedding

## EVAL-IDX-001: incremental embedding versus index update

增量 embedding 和增量 index update 是两层不同的优化。

增量 embedding 解决的是“少算向量”：如果 chunk 的 identity、text_hash 和 embedding model 没变，就直接复用旧向量，只给新增或修改的 chunk 计算 embedding。

增量 index update 解决的是“少改索引”：新增向量时直接 add，修改时 delete old vector 再 upsert new vector，删除文件时按 id 删除对应向量。

TinyRAG 当前采用的策略是：

```text
增量 embedding 复用
FAISS index 基于完整向量集合重建
```

这是一个本地轻量场景下的一致性优先方案。它减少了最昂贵的 embedding 计算，但没有声称自己实现了向量数据库级别的 upsert/delete。

## EVAL-IDX-002: index manifest

索引 manifest 用来判断知识库是否发生变化。它记录文件路径、文件大小、文件 SHA1、分块参数和 schema version。

示例 manifest：

```json
{
  "schema_version": 2,
  "data_dir": "F:/TinyRAG-master/data",
  "files": [
    {
      "path": "F:/TinyRAG-master/data/eval_03_vector_index_incremental_faiss.md",
      "size": 2048,
      "sha1": "example-sha1"
    }
  ],
  "parent_child": {
    "child_max_tokens": 250,
    "child_overlap_tokens": 30,
    "parent_max_tokens": 2000
  }
}
```

如果 manifest 没变，系统可以直接加载旧索引。如果 manifest 变化，系统重新读取文档、切 chunk，并复用未变化 chunk 的历史向量。

## EVAL-IDX-003: FAISS tradeoff

FAISS 适合本地轻量、高性能相似度搜索。它对 demo、单机部署和中小规模知识库很友好。与 Qdrant 或 Milvus 相比，FAISS 的优势是简单、快、依赖少；劣势是缺少原生 payload filter、服务化 API、按 metadata 删除和标准化 upsert 能力。

面试回答中可以明确边界：

```text
当前项目为了轻量性保留 FAISS。
如果知识库需要频繁新增、删除、按租户隔离或服务化部署，会抽象 VectorStore 并切换 Qdrant。
```

## EVAL-IDX-004: stale index error

当 manifest 与磁盘索引不一致时，可以定义内部错误码 `RAG_INDEX_STALE`。这个错误表示文档或分块配置已经变化，旧索引不能直接使用，需要触发索引更新。

示例处理：

```python
class IndexStateError(RuntimeError):
    pass


def ensure_manifest_compatible(saved_manifest: dict, current_manifest: dict) -> None:
    if saved_manifest != current_manifest:
        raise IndexStateError("RAG_INDEX_STALE: manifest changed, rebuild index required")
```

这个片段适合测试 BM25 对精确错误码和函数名的召回能力。


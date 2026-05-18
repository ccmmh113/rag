# Eval Corpus 01: RAG Pipeline and Parent-Child Chunking

## EVAL-RAG-001: TinyRAG end-to-end pipeline

TinyRAG 的主链路由多个阶段组成：文档读取、父子分块、embedding、稠密检索、BM25 稀疏检索、混合召回、候选清洗、reranker 精排、上下文压缩、ContextBuilder 构造 prompt、LLM 生成答案、trace 记录。这个链路的目标不是单纯把 top-k chunk 拼给模型，而是在召回率、精度、上下文完整性和 token 成本之间做平衡。

典型执行顺序如下：

```text
ReadFiles
  -> ParentChildChunker
  -> EmbeddingModel
  -> FaissVectorIndex
  -> DenseRetriever + BM25Retriever
  -> HybridRetriever
  -> _prepare_rerank_candidates
  -> BgeReranker
  -> ContextCompressor
  -> ContextBuilder
  -> PromptManager
  -> OpenAIChat
  -> TraceStore
```

面试回答要点：召回阶段尽量不要漏掉相关证据，reranker 阶段负责把真正能回答问题的片段排到前面，ContextBuilder 负责把检索命中的小块扩展成更完整的父块，并控制最终 prompt 长度。

## EVAL-RAG-002: parent-child chunking

Parent-child chunking 用两种粒度服务不同目标。child chunk 更短，用于向量检索和关键词检索；parent chunk 更长，用于给 LLM 提供完整上下文。这样可以缓解“召回粒度”和“生成上下文完整性”的矛盾。

如果只用大块检索，召回可能不够精准；如果只用小块生成，LLM 可能看不到完整语义。父子分块的折中是：

```text
child chunk: 精准命中问题相关的局部内容
parent chunk: 补充同一主题下的完整解释
parent_id: child 到 parent 的映射键
```

在 TinyRAG 中，child 文档包含 `parent_id`，ContextBuilder 会根据 `parent_map[parent_id]` 找到 parent text。多个 child 命中同一个 parent 时，只保留一次 parent，避免重复浪费 token。

## EVAL-RAG-003: stable chunk identity

稳定 chunk identity 是增量入库和评估匹配的基础。一个不稳定的 chunk id 会导致三类问题：评测集中的 relevant_ids 失效、引用定位不准确、旧向量无法可靠复用。

推荐的 identity 组成包括：

```text
doc_id
source_hash
page
parent_chunk_id
child_chunk_id
text_hash
```

TinyRAG 的新版实现使用 `chunk_uid` 作为优先身份标识。它比只使用 `doc_id:source:chunk_id` 更稳，因为父子分块下不同 parent 的 child chunk 都可能从 0 开始编号。

```python
def build_chunk_uid(doc_id: str, source_key: str, page: int, parent_idx: int, child_idx: int) -> str:
    return f"{doc_id}::{source_key}::p{page}::{parent_idx}::c{child_idx}"
```


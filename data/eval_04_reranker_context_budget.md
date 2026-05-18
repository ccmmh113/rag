# Eval Corpus 04: Reranker, Compressor, Deduplication, and Budget

## EVAL-CTX-001: pre-rerank candidate cleanup

Reranker 前适合做轻量候选清洗，而不适合做强语义过滤。原因是 reranker 前还没有 query-doc 交互分数，强过滤可能误删有用证据。

TinyRAG 的候选清洗包括：

```text
过滤空文本
按 identity 去重，保留 score 更高的候选
同一个 parent_id 下限制 child 数量
最多保留 candidate_top_k 个候选送入 reranker
```

对应函数名是 `_prepare_rerank_candidates`，核心配置项是 `candidate_top_k` 和 `max_candidates_per_parent`。

## EVAL-CTX-002: reranker

BGE reranker 是 cross-encoder 精排模型。它输入 `(query, document)` 对，输出相关性分数。相比第一阶段召回，reranker 计算更慢，但能更准确判断某个 chunk 是否真正回答当前问题。

典型策略：

```text
召回阶段 top 50，追求 recall
候选清洗后 top 30，减少 reranker 成本
reranker 输出 top 8，追求 precision
```

如果用户问“为什么候选清洗不能代替 reranker”，答案是：候选清洗只看去重、父块数量和空文本等结构信息；reranker 会结合 query 判断内容相关性。

## EVAL-CTX-003: context compressor

ContextCompressor 用来减少弱相关上下文。TinyRAG 默认使用 `relevance_filter`。如果有 rerank_score，就根据 `relevance_threshold` 过滤低分文档；如果没有 rerank_score，则避免用 RRF 的小分值直接做绝对阈值过滤。

配置示例：

```python
CompressionConfig(
    strategy="relevance_filter",
    relevance_threshold=0.3,
    summary_max_tokens=200,
)
```

这个设计避免了一个常见错误：把 RRF 分数当成 reranker 分数使用。RRF 分数通常很小，不能直接和 0.3 这样的阈值比较。

## EVAL-CTX-004: semantic deduplication

ContextBuilder 在构造 prompt 前会做近重复去重。如果传入 embedding_model，则使用 embedding cosine similarity；如果没有 embedding_model，则退化为 Jaccard 词集合相似度。

简化逻辑：

```python
def similarity(left: str, right: str) -> float:
    if embedding_model is not None:
        lv = np.array(embedding_model.get_embedding(left))
        rv = np.array(embedding_model.get_embedding(right))
        return cosine(lv, rv)
    return jaccard(tokenize(left), tokenize(right))
```

默认阈值 `semantic_dedup_threshold=0.92`。如果当前候选和已选上下文相似度超过阈值，就跳过当前候选，避免重复占用 prompt budget。

## EVAL-CTX-005: token budget

ContextBuilder 用 `max_tokens` 控制 prompt 上下文预算。它按照 reranker 排序依次加入片段，每加入一段前先计算剩余 token。如果剩余预算不足，则截断当前片段或停止加入。

预算控制顺序：

```text
父块去重
exact dedup
semantic dedup
计算 header token
计算 available tokens
trim_to_tokens
追加 citation
```

这个策略保证高相关片段优先进入 prompt，低排序或重复片段不会挤占上下文空间。


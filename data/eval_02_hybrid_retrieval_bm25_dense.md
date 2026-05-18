# Eval Corpus 02: Hybrid Retrieval, BM25, and Dense Search

## EVAL-RET-001: BM25 sparse retrieval

BM25 是基于词项匹配的稀疏检索算法。它关注 query 词是否出现在文档中、词在当前文档中的出现频率、词在全局语料中的稀有程度，以及文档长度归一化。BM25 对代码标识符、配置项、错误码、专有名词和英文缩写特别敏感。

BM25 的优势场景：

```text
函数名: _prepare_rerank_candidates
配置项: candidate_top_k
错误码: RAG_INDEX_STALE
类名: HybridRetriever
字段名: chunk_uid
```

如果用户直接搜索 `RAG_INDEX_STALE`，向量检索可能把它当成普通 token，而 BM25 可以精确命中包含该错误码的文档。

## EVAL-RET-002: dense semantic retrieval

稠密向量检索把 query 和 chunk 映射到向量空间，通过 cosine similarity 或 inner product 找语义接近的内容。它适合处理同义改写和概念类问题。

例如用户问：

```text
为什么不能只用很小的分块给模型回答？
```

即使文档里没有完全相同的句子，dense retrieval 也可能召回 parent-child chunking 相关内容，因为它能捕捉“上下文不完整”“语义被切割”“小块生成信息不足”等语义关系。

稠密检索的弱点是对精确字符串不够稳定，例如函数名、版本号、错误码和配置 key。

## EVAL-RET-003: RRF fusion

RRF，全称 Reciprocal Rank Fusion，用于融合多个检索器的排序结果。它不直接比较 BM25 分数和向量相似度，因为两者分数尺度不同，而是根据排名给分。

简化公式：

```text
score(doc) = dense_weight / (rrf_k + dense_rank)
           + sparse_weight / (rrf_k + sparse_rank)
```

在 TinyRAG 中，HybridRetriever 可以使用 RRF 融合 DenseRetriever 和 BM25Retriever。RRF 的优点是稳定，不需要强行把不同检索器的原始分数归一到同一个尺度。

## EVAL-RET-004: weighted fusion

Weighted fusion 会先把 dense 和 sparse 的分数归一化，再按权重相加。它适合做消融实验，但对分数分布更敏感。如果某个检索器的分数跨度异常，weighted fusion 可能偏向一侧。

TinyRAG 中的建议：

```text
默认使用 RRF
短关键词或代码标识符查询提高 sparse_weight
长语义问题提高 dense_weight
```


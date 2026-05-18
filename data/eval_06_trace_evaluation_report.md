# Eval Corpus 06: Trace, Evaluation, and Error Analysis

## EVAL-EVAL-001: retrieval metrics

RAG 检索评估常用指标包括 Context Precision、Context Recall、Hit@K、MRR 和 latency。Context Precision 关注相关文档是否排在前面，Context Recall 关注所有相关文档被召回的比例，Hit@K 关注 top-k 中是否至少命中一个相关文档，MRR 关注第一个相关结果的位置。

推荐比较组：

```text
Dense only
BM25 only
Hybrid RRF
Hybrid Weighted
Hybrid + Reranker
Full Pipeline
```

如果 Hybrid 的 Context Recall 高于 Dense only 和 BM25 only，说明两类召回互补。如果 Hybrid + Reranker 的 Context Precision 更高，说明精排有效。

## EVAL-EVAL-002: generation metrics

生成评估关注 Faithfulness、Answer Relevancy、Answer Correctness 和 Hallucination Rate。Faithfulness 判断答案是否被上下文支持；Answer Relevancy 判断答案是否直接回应问题；Answer Correctness 判断答案和标准答案是否一致。

对于本项目，检索评估应先跑稳定，再跑生成评估。否则生成质量差可能来自召回失败、reranker 排序失败、context compressor 误删证据或 LLM 自身幻觉，难以定位。

## EVAL-EVAL-003: trace fields

Trace 用来解释一次问答为什么成功或失败。建议记录：

```text
query
route
route_policy_scores
recalled_docs
rerank_candidate_count
rerank_candidate_dropped
reranked_count
retrieval_latency
rerank_latency
generation_latency
prompt_tokens
citations
answer_preview
```

当某个问题回答错误时，可以根据 trace 判断失败位置。如果 recalled_docs 没有相关内容，是召回失败；如果 recalled_docs 有但 reranked 没保留，是精排失败；如果 citations 正确但答案错误，是生成或 prompt 约束问题。

## EVAL-EVAL-004: failure taxonomy

错误分析可以分为五类：

```text
retrieval_miss: 召回阶段没有取到证据
ranking_error: 召回有证据，但排序太靠后
compression_drop: 压缩阶段误删证据
context_budget_overflow: 证据存在，但因为 token budget 没进入 prompt
generation_hallucination: 上下文正确，但 LLM 生成错误
```

面试中最有说服力的不是只展示平均指标，而是展示失败样本和归因。


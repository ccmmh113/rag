  TinyRAG 完整运行链路分析

  一、总览：9 阶段 Pipeline

  用户 Query
    │
    ├─[1] Memory Context ── 拉取对话历史 + 用户偏好
    ├─[2] Query Routing ── 判断查询类型，决定 dense/sparse 权重
    ├─[3] Retrieval ────── 稠密检索 (FAISS) + 稀疏检索 (BM25) 并行，融合排序
    ├─[4] Rerank ───────── 交叉编码器对 Top-50 精排，截断到 Top-8
    ├─[5] Compress ─────── 可选：相关性过滤 / LLM 摘要 / 混合
    ├─[6] Build Context ── 父块展开、去重、去重叠、token 预算截断
    ├─[7] Generate ─────── 拼装 Prompt → LLM 生成答案
    ├─[8] Update Memory ── 保存本轮 Q&A 到短期记忆
    └─[9] Save Trace ───── 全链路指标持久化到 JSONL

  ---
  二、前置阶段：索引构建（启动时一次性）

  入口：run.py → load_or_build_index()

  1. 读取文档 (RAG/utils.py:ReadFiles)：扫描 data/ 目录下的 .md、.txt、.pdf 文件
    - PDF：PyPDF2 逐页提取文本
    - Markdown：保留原始格式（用于后续标题感知分块）
    - TXT：直接读取
  2. Parent-Child 双层分块 (RAG/chunking.py:ParentChildChunker)：
    - Parent chunker：max_tokens=2000，把文档切成大块（完整章节级）
    - Child chunker：max_tokens=250, overlap=30，在每个 parent 内部切细块
    - 每个 child 的 metadata 中记录 parent_id，建立映射 parent_map
    - 设计意图：小 child 用于精确向量检索（匹配更准），大 parent 用于给 LLM 提供完整上下文
  3. MarkdownAwareChunker 分块细节：
    - 按空行分段（paragraph-level blocks）
    - 遇到 # heading 强制切分，并记录章节路径（如 "第一章 > 1.1 概述"）
    - 代码块 (```) 内部不切分，保持完整性
    - 超长块按 max_tokens - overlap_tokens 步长滑动窗口切分
    - 过小 chunk 合并：token 不足 min_chunk_tokens(20) 的合并到前一个 chunk
    - Overlap：每个 chunk 尾部保留 120 token 的文本，作为下一个 chunk 的开头
  4. 向量化 & 索引 (run.py:_build_index)：
    - 用 BGE 模型 (BAAI/bge-base-zh-v1.5) 逐个 child 编码为 768 维向量
    - 存入 FAISS IndexFlatIP（内积索引，向量已 L2 归一化，等价于余弦相似度）
    - 支持 HNSW 近似索引作为替代方案
    - 索引、向量、文档元数据持久化到 storage/

  ---
  三、在线查询：9 阶段详解

  阶段 1：Memory Context（记忆上下文）

  代码位置：RAG/runtime/pipeline.py:90

  mem_ctx = self._memory.before_query(query)

  - 短期记忆 (ShortTermMemory)：保存最近 N 轮（默认 5 轮）的 Q&A 对话
    - 转换为 [{"role":"user", ...}, {"role":"assistant", ...}] 消息列表
    - 自动驱逐：超过 max_turns 或 max_tokens(1500) 时从头部弹出最旧轮次
  - 长期记忆 (LongTermMemory)：ChromaDB 持久化存储
    - 对当前 query 做语义搜索，召回相关 preference 类型的记忆（如用户背景、偏好）
    - 有自动去重机制（保存时发现相似条目则更新而非新建）
  - 会话上下文：用户可通过 context set key value 设置的键值对，直接注入 prompt

  阶段 2：Query Routing（查询路由）

  代码位置：RAG/runtime/pipeline.py:92-101 → RAG/router/router.py:PolicyRouter.route()

  4 个策略对 query 独立打分（0-1），选最高分者，动态决定 dense/sparse 融合权重：

  策略: DensePolicy
  触发条件: 包含疑问词（什么/怎么/why/how）、长句、含中文
  评分规则: 含疑问词 +0.5，非短句 +0.3，含中文 +0.2
  权重 (dense/sparse): 0.7 / 0.3
  ────────────────────────────────────────
  策略: SparsePolicy
  触发条件: 短查询（≤5 token）、无疑问词、纯英文/代码
  评分规则: 短查询 +0.5，无疑问词 +0.3，无中文 +0.2
  权重 (dense/sparse): 0.3 / 0.7
  ────────────────────────────────────────
  策略: CodePolicy
  触发条件: 包含代码特征（def/class/import/#include/大写标识符）
  评分规则: 命中直接 0.8，否则 0
  权重 (dense/sparse): 0.2 / 0.8
  ────────────────────────────────────────
  策略: HybridPolicy
  触发条件: 兜底
  评分规则: 恒定 0.35
  权重 (dense/sparse): 0.5 / 0.5

  选中的权重会实时写入 HybridRetriever.config.dense_weight / sparse_weight。

  阶段 3：Retrieval（检索召回）

  代码位置：RAG/runtime/pipeline.py:103-112 → RAG/retrievers.py:HybridRetriever.retrieve()

  并行双路召回（ThreadPoolExecutor）：

  - Dense 路 (DenseRetriever)：
    a. 将 query 用 BGE 模型编码为 768 维向量
    b. FAISS 向量索引搜索 top-50（如有 metadata filter 则 over-fetch 4x 即 200，再过滤）
    c. 分数为余弦相似度
  - Sparse 路 (BM25Retriever)：
    a. 使用 bm25s 库的 BM25 算法
    b. 分词器：英文单词 + 中文字符 + 中文二元组（bigram）
    c. 同样搜索 top-50，支持 metadata filter 的 over-fetch

  融合排序（两种模式，默认 RRF）：

  - RRF (Reciprocal Rank Fusion)：
  score(doc) = dense_weight / (60 + rank_dense) + sparse_weight / (60 + rank_sparse)
  - 按融合分数降序排列，取 top-50
  - Weighted Fusion：
    a. 分别对两路分数做 min-max 归一化到 [0, 1]
    b. score = dense_weight * norm_dense_score + sparse_weight * norm_sparse_score
  - 两路同一文档的去重：使用 Document.identity（基于 text hash）合并

  阶段 4：Rerank（重排序）

  代码位置：RAG/runtime/pipeline.py:114-123 → RAG/Reranker.py:BgeReranker

  - 使用 BGE-Reranker 交叉编码器 (BAAI/bge-reranker-base)
  - 对每个 (query, document_text) 对做拼接编码，通过一个分类头输出相关性 logit
  - 对 top-50 篇文档重新打分，按新分数降序排列，截断到 top-8（reranker.top_k=8）
  - 每篇文档此时带有两个分数：原始 score（检索分数）和 rerank_score（重排序分数）

  阶段 5：Compress（上下文压缩）

  代码位置：RAG/runtime/pipeline.py:125-129 → RAG/context/compressor.py

  三种策略（通过 CompressionConfig.strategy 配置，默认 relevance_filter）：

  - relevance_filter (RelevanceFilterCompressor)：
    - 检查 rerank 后的文档总 token 数是否超出 max_tokens(3000)
    - 若超出：按 rerank_score 降序排列，贪心选取直到装满 token 预算
    - O(n)，无需 LLM 调用
  - summary (SummaryCompressor)：
    - 对每篇文档调用 LLM 做摘要压缩到 200 token
    - prompt："请用不超过200个token对以下内容进行摘要，保留与问题「{query}」最相关的信息"
    - LLM 调用失败时 fallback 回原文
  - hybrid：先 filter（预算 ×2），再对幸存者做 summary

  注意：此阶段不是默认开启的。run.py 构建 RAGRuntime 时 compressor 参数为 None，所以实际运行中此阶段被跳过。需要在构建
  Runtime 时显式传入 ContextCompressor 才会执行。

  阶段 6：Build Context（构建上下文）

  代码位置：RAG/runtime/pipeline.py:131-134 → RAG/context/builder.py:ContextBuilder.build()

  这是最关键的阶段，将 rerank 后的文档组装为最终注入 LLM 的上下文字符串：

  1. 分数过滤：丢弃 rerank_score（或 score）低于 min_score(0.0) 的文档
  2. Parent 展开（_resolve_parent）：
    - 查找 child chunk 的 parent_id
    - 若在 parent_map 中找到，用父块完整文本替换子块文本
    - 这是关键设计：检索用精确的小 chunk，给 LLM 看的是完整的大段落
  3. 精确去重（_normalise_text）：
    - 文本做空白符归一化后 hash，相同文本只保留一份
  4. 语义去重（_is_semantic_duplicate）：
    - 若提供了 embedding_model：计算两段文本的余弦相似度，≥0.92 视为重复
    - 否则：Jaccard 相似度（词级别的交并比）
    - 可配置关闭
  5. 重叠去除（_remove_overlap）：
    - 由于 chunking 时有 overlap，相邻 chunk 的开头可能重复
    - 取前一个 chunk 尾部 160 字符，与当前 chunk 开头做最长前缀匹配
    - 匹配到则从当前 chunk 开头切除重叠部分
  6. Token 预算硬截断：
    - 每个 chunk 附带头部标记 [source=文件名 chunk=序号]
    - 累加 token 计数，超出 max_tokens(3000) 时停止添加
    - 最后一个能放下的 chunk 也会被 trim_to_tokens 截断到剩余可用空间
  7. 输出：BuiltContext — 包含最终的上下文字符串、入选文档列表、引用信息

  阶段 7：Generate（生成答案）

  代码位置：RAG/runtime/pipeline.py:136-166 → RAG/LLM.py:PromptManager.build_messages()

  Prompt 组装 (PromptManager.build_rag_prompt())：

  [System Prompt]
  你是一个严谨的问答助手。请基于给定上下文回答问题。

  [Messages History from ShortTermMemory]
  user: 上一轮的query
  assistant: 上一轮的answer
  ...

  [Current User Message — 由 build_rag_prompt 构建]
  使用以下上下文来回答用户的问题。如果你不知道答案，可以根据现有的知识，如果不确定就说不知道。总是使用中文回答。

  用户背景:                          ← 来自长期记忆的 preferences
    - 我是后端开发工程师
    - 熟悉 Python 和 Go

  会话上下文:                        ← 来自 session_context
    - topic: RAG系统架构

  问题: 什么是RAG的检索增强?

  可参考的上下文:
  [source=doc1.md chunk=3]
  RAG (Retrieval-Augmented Generation) 是一种结合信息检索与文本生成的技术架构...

  [source=doc2.pdf chunk=7 page=3]
  在实际应用中，RAG系统通常包含索引构建和在线查询两个阶段...

  如果给定的上下文无法让你做出回答，请回答数据库中没有这个内容，不知道。
  回答:

  LLM 调用：
  - 使用 OpenAI 兼容 API（OpenAIChat），默认模型为 gpt-5.4
  - 参数：temperature=0.1, max_tokens=512
  - 同时记录 prompt_tokens 和 cached_prompt_tokens（用于追踪 prompt cache 命中率）
  - 完整 prompt 追加写入 storage/prompts.jsonl

  阶段 8：Update Memory（更新记忆）

  代码位置：RAG/runtime/pipeline.py:168-170

  self._memory.after_query(query, built.context, answer)

  将本轮 (query, answer, context) 追加到 ShortTermMemory 的 turn buffer。超出限制时自动驱逐旧轮次。

  阶段 9：Save Trace（保存追踪）

  代码位置：RAG/runtime/pipeline.py:172-178

  QueryTrace 记录每轮查询的完整指标：
  - 路由策略、各策略得分
  - 检索延迟、召回数量
  - 重排序延迟、重排后数量
  - prompt token 数、LLM cache 命中情况
  - 生成延迟、答案预览
  - 总延迟

  追加写入 storage/traces.jsonl，TraceStore 提供分析查询（avg 延迟、cache rate、策略分布等）。

  ---
  四、关键数据流

  data/*.pdf,.md,.txt
    │  [ParentChildChunker]
    ▼
  child_docs (small, ~250 tokens each)  +  parent_map
    │  [BgeEmbedding]
    ▼
  vectors (768-dim)  ──→  FAISS Index (FlatIP)
    │
    │  query 进来
    ▼
  Dense: FAISS search top-50   ╲
                                 ╲  [RRF / Weighted Fusion]
  Sparse: BM25 search top-50   ╱
                                 ╱
    ▼
  Top-50 fused documents
    │  [BgeReranker cross-encoder]
    ▼
  Top-8 reranked documents (带 rerank_score)
    │  [ContextBuilder: parent expand → dedup → overlap remove → token cap]
    ▼
  BuiltContext (≤ 3000 tokens 的上下文字符串 + citations)
    │  [PromptManager: system prompt + history + preferences + context + question]
    ▼
  messages (OpenAI chat format)
    │  [OpenAIChat.chat()]
    ▼
  RAGResponse { answer, context, citations, trace }

  五、核心设计要点总结

  环节: 分块
  关键技术: Parent-Child 双层分块
  亮点: 小块精检索 + 大块富上下文，解决 chunk 太小丢失语义、太大检索不准的矛盾
  ────────────────────────────────────────
  环节: 索引
  关键技术: FAISS FlatIP / HNSW
  亮点: L2 归一化后内积=余弦相似度，HNSW 支持大规模近似检索
  ────────────────────────────────────────
  环节: 检索
  关键技术: Dense + BM25 并行 + RRF 融合
  亮点: 语义匹配 + 关键词精确匹配互补，RRF 无需调参
  ────────────────────────────────────────
  环节: 路由
  关键技术: 多策略打分竞争
  亮点: 根据查询类型自适应调整 dense/sparse 权重
  ────────────────────────────────────────
  环节: 重排序
  关键技术: BGE Cross-Encoder
  亮点: 交互式编码比向量相似度更精确，但计算量大，所以只对 top-50 做
  ────────────────────────────────────────
  环节: 压缩
  关键技术: 三种策略可选
  亮点: 默认不开启，仅在 token 预算紧张时使用
  ────────────────────────────────────────
  环节: 上下文
  关键技术: Parent 展开 + 去重 + 去重叠
  亮点: 保证 LLM 拿到完整、无冗余的上下文
  ────────────────────────────────────────
  环节: 记忆
  关键技术: 短期 turn buffer + 长期 ChromaDB
  亮点: 实现多轮对话和用户偏好持久化
  ────────────────────────────────────────
  环节: 追踪
  关键技术: JSONL 全链路指标
  亮点: 可分析延迟、cache 命中率、策略分布

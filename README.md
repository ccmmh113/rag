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

  入口：run.py 

  1. 读取文档 (RAG/utils.py:ReadFiles)：扫描 data/ 目录下的 .md、.txt、.pdf 文件
    - PDF：PyPDF2 逐页提取文本
    - Markdown：保留原始格式（用于后续标题感知分块）
    - TXT：直接读取
  2. Parent-Child 双层分块 
    - Parent chunker：max_tokens=2000，把文档切成大块（完整章节级）
    - Child chunker：max_tokens=250, overlap=30，在每个 parent 内部切细块
    - 每个 child 的 metadata 中记录 parent_id，建立映射 parent_map
小 child 用于精确向量检索（匹配更准），大 parent 用于给 LLM 提供完整上下文
  3. MarkdownAwareChunker 分块细节：
    - 按空行分段（paragraph-level blocks）
    - 遇到 # heading 强制切分，并记录章节路径（如 "第一章 > 1.1 概述"）
    - 代码块 (```) 内部不切分，保持完整性
    - 超长块按 max_tokens - overlap_tokens 步长滑动窗口切分
    - 过小 chunk 合并：token 不足 min_chunk_tokens(20) 的合并到前一个 chunk
    - Overlap：每个 chunk 尾部保留 120 token 的文本，作为下一个 chunk 的开头
  ---
  记忆模块：
    - 短期记忆 (ShortTermMemory)：保存最近 N 轮（默认 5 轮）的 Q&A 对话
    - 自动驱逐：超过 max_turns 或 max_tokens(1500) 时从头部弹出最旧轮次
  - 长期记忆 (LongTermMemory)：持久化存储
    - 对当前 query 做语义搜索，召回相关 preference 类型的记忆（如用户背景、偏好）
    - 有自动去重机制（保存时发现相似条目则更新而非新建）
   4 个策略对 query 独立打分（0-1），选最高分者，动态决定 dense/sparse 融合权重：
查询路由：
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
并行双路召回：
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

Rerank（重排序）
  - 使用 BGE-Reranker 交叉编码器 (BAAI/bge-reranker-base)
  - 对每个 (query, document_text) 对做拼接编码，通过一个分类头输出相关性 logit
  - 对 top-50 篇文档重新打分，按新分数降序排列，截断到 top-8（reranker.top_k=8）
  - 每篇文档此时带有两个分数：原始 score（检索分数）和 rerank_score（重排序分数）
Compress（上下文压缩）




  

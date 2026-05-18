# Eval Corpus 07: Agent Memory and Workflow Distractors

## EVAL-DIST-001: agent memory is not vector index memory

Agent memory 和 RAG 向量索引都可能使用“记忆”这个词，但它们不是同一个概念。Agent memory 通常指用户偏好、历史任务、失败经验和可复用流程；RAG 向量索引主要用于根据 query 检索外部知识片段。

这个文档故意作为干扰语料使用。它包含 memory、embedding、retrieval、context 等词，但主题是 Agent 行为记忆，不是 TinyRAG 的 FAISS 知识库索引。

## EVAL-DIST-002: short-term and long-term memory

短期记忆保存当前会话上下文，长期记忆保存跨会话偏好和笔记。典型接口：

```python
class MemoryManager:
    def before_query(self, query: str):
        return MemoryContext(history_messages=[], preferences=[])

    def after_query(self, query: str, context: str, answer: str):
        pass

    def save(self, content: str, entry_type: str = "note") -> str:
        return "memory-id"
```

如果用户问“长期记忆如何影响 prompt”，应该召回 Agent memory 相关内容；如果用户问“新增文档如何复用旧向量”，不应该召回这个干扰段落。

## EVAL-DIST-003: workflow agent tools

Workflow、Agent、Tools 的区别：

```text
Workflow: 预定义流程，路径相对固定
Agent: 根据目标动态规划和选择动作
Tools: Agent 或 Workflow 可调用的外部能力
```

这个主题与 RAG pipeline 有相似词，例如 route、context、trace、evaluation，但它不直接回答 BM25、FAISS、chunk_uid 或 parent-child chunking 的问题。


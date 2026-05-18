# Eval Corpus 05: FastAPI and Vue Interface

## EVAL-WEB-001: FastAPI runtime manager

TinyRAG 的 Web 层使用 FastAPI 暴露接口，前端使用 Vue CDN 构建轻量页面。RuntimeManager 负责懒加载 embedding、索引、reranker 和 RAGRuntime。

核心接口：

```text
GET  /api/status
POST /api/index/build
GET  /api/config
PATCH /api/config
POST /api/chat
GET  /api/traces
GET  /api/memories
POST /api/memories
```

`POST /api/index/build` 会触发索引构建或更新。`POST /api/chat` 会调用 runtime.query，并返回 answer、context、citations 和 trace。

## EVAL-WEB-002: upload and ingestion design

如果要支持上传新文件，FastAPI 可以增加 `/api/files/upload`。上传后不要立刻覆盖旧索引，而是先保存文件、计算 hash、更新 manifest，再触发索引更新。

示例代码：

```python
from fastapi import APIRouter, UploadFile, File
from pathlib import Path

router = APIRouter()
DATA_DIR = Path("data")


@router.post("/api/files/upload")
async def upload_file(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".md", ".txt", ".pdf"}:
        return {"ok": False, "error": "UNSUPPORTED_FILE_TYPE"}

    target = DATA_DIR / file.filename
    content = await file.read()
    target.write_bytes(content)
    return {"ok": True, "filename": file.filename, "bytes": len(content)}
```

这个片段适合测试检索器对 `UNSUPPORTED_FILE_TYPE`、`UploadFile`、`/api/files/upload` 等精确代码 token 的召回能力。

## EVAL-WEB-003: Vue config panel

前端配置面板可以暴露 top_k、fusion、dense_weight、sparse_weight、use_reranker、compression strategy 等参数。用户改完配置后调用 `PATCH /api/config`，再决定是否重建索引。

示例 Vue 方法：

```javascript
async saveConfig() {
  const payload = {
    dense_weight: this.config.dense_weight,
    sparse_weight: this.config.sparse_weight,
    use_reranker: this.config.use_reranker,
    fusion: this.config.fusion
  };
  this.status = await this.api('/api/config', {
    method: 'PATCH',
    body: JSON.stringify(payload)
  });
}
```

这类代码检索更依赖 BM25，因为函数名和 API path 是精确字符串。


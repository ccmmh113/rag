# TinyRAG Web Console

FastAPI + Vue single-page console for the TinyRAG runtime.

## Run

Install the small web layer dependency:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-web.txt
```

Start the server:

```powershell
.venv\Scripts\python.exe -m uvicorn web.app:app --host 127.0.0.1 --port 8000 --reload
```

Open:

```text
http://127.0.0.1:8000
```

## Notes

- Put `.md`, `.txt`, or `.pdf` files under `data/`.
- Set `OPENAI_API_KEY` before asking questions.
- The first query lazily loads the embedding model and index, so it may take longer.
- The frontend is intentionally build-free: FastAPI serves `web/static/index.html`, and Vue is loaded from a CDN.

# LocalGPT Backend

The backend is the API gateway for the current LocalGPT application. It runs on port `8000` and sits between the Next.js frontend, the RAG API server, Ollama, and the SQLite metadata database.

For the full architecture, start with:
- [System overview](../Documentation/system_overview.md)
- [API reference](../Documentation/api_reference.md)
- [Quick start](../Documentation/quick_start.md)

## Role

`backend/server.py` is responsible for:
- Chat session and message persistence
- Index metadata, document uploads, and session/index linking
- Background index build jobs and job progress endpoints
- Maintenance endpoints for index repair, cleanup, and diagnostics
- Routing chat requests to direct Ollama responses or the RAG API

The backend does not perform the heavy RAG indexing work itself. It delegates document processing and retrieval to the RAG API on port `8001`.

## Main Services

| Service | Port | Purpose |
|---------|------|---------|
| Frontend | `3000` | Next.js UI |
| Backend | `8000` | API gateway, sessions, jobs, metadata |
| RAG API | `8001` | Indexing, retrieval, generation pipeline |
| Ollama | `11434` | Local model serving |

## Start

From the project root:

```bash
./start-localgpt
```

Or start only the backend after activating the Python environment:

```bash
python backend/server.py
```

## Health Check

```bash
curl http://localhost:8000/health
```

## Current Model Defaults

The current docs and launch scripts use Qwen/Ollama defaults such as:
- Generation: `qwen3:8b`
- Fast enrichment/routing: `qwen3:0.6b`
- Embeddings: `Qwen/Qwen3-Embedding-0.6B`

Do not use this README as the model registry; check [system_overview.md](../Documentation/system_overview.md) and runtime configuration for the authoritative values.

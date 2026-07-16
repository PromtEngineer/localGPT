# LocalGPT Backend

The backend on port 8000 is the browser-facing API and the sole owner of SQLite chat persistence. It provides health/model discovery, sessions, messages, SSE streaming, uploads, index lifecycle, and session-to-index links. It delegates retrieval/indexing to the internal RAG API configured by `RAG_API_URL`.

Start it from the repository root:

```bash
pip install -r requirements.txt
python -m backend.server
```

The default bind address is `127.0.0.1:8000`. Configure it with `LOCALGPT_BACKEND_HOST` and `LOCALGPT_BACKEND_PORT`.

The Next.js frontend calls its same-origin `/api/backend/*` proxy. In Docker, that proxy uses `BACKEND_INTERNAL_URL=http://backend:8000`. If `LOCALGPT_API_TOKEN` is set, the proxy injects the bearer token server-side.

Uploads are stored under `LOCALGPT_UPLOAD_DIR`, restricted to supported document extensions, and limited by `LOCALGPT_MAX_UPLOAD_BYTES`. Session and index uploads both pass through the same validator.

See [`../Documentation/api_reference.md`](../Documentation/api_reference.md) for routes and payloads, and [`../Documentation/system_overview.md`](../Documentation/system_overview.md) for ownership and storage boundaries.

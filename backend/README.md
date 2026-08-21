# localGPT Backend

The gateway between the Next.js frontend and the rest of the system. It owns chat sessions,
uploaded documents and index bookkeeping in SQLite, answers simple queries directly with
Ollama, and forwards document-grounded queries and indexing jobs to the RAG API.

```
frontend :3000  ──►  backend/server.py :8000  ──►  rag_system/api_server.py :8001  ──►  Ollama :11434
                              │
                              └──► Ollama :11434 (direct, non-RAG answers)
```

Built on the standard library's `http.server` (`ThreadingTCPServer`, so requests are
handled concurrently). No web framework.

## Prerequisites

1. **Python 3.10+** (3.11 recommended).

2. **Ollama** running locally:
   ```bash
   # https://ollama.com/download, or:
   curl -fsSL https://ollama.ai/install.sh | sh
   ollama serve
   ```

3. **Models** — the defaults the backend resolves to:
   ```bash
   ollama pull qwen3.5:9b   # generation
   ollama pull qwen3.5:4b   # enrichment / utility (used by the RAG API, not by the gateway)
   ```

4. **The RAG API** (`python -m rag_system.api_server`) if you want document-grounded
   answers or indexing. Without it, `/sessions/<id>/messages` still works for non-document
   queries, and RAG queries return a "could not connect" message.

## Setup

```bash
# From the repository root
pip install -r backend/requirements.txt   # requests, python-dotenv

# Smoke-test the Ollama connection (note: this pulls GENERATION_MODEL if it is missing)
python backend/ollama_client.py

# Start the server
python backend/server.py
```

The server listens on `http://localhost:8000`. Run it from the repository root so the
relative paths it uses (`shared_uploads/`, `index_store/`, `lancedb/`, `backend/chat_data.db`)
resolve the same way they do in the Docker image.

Usually you do not start it by hand — `python run_system.py` launches Ollama, the RAG API,
the backend and the frontend together.

## Configuration

All environment variables are optional; the value shown is the code default.

| Variable | Default | Used for |
| --- | --- | --- |
| `OLLAMA_HOST` | `http://localhost:11434` | Direct Ollama calls (`backend/ollama_client.py`). |
| `RAG_API_URL` | `http://localhost:8001` | Base URL for the RAG API; `/chat` and `/index` are built from it. |
| `RAG_API_TIMEOUT` | `600` | Seconds to wait for a RAG chat response before returning 504. |
| `RAG_API_INDEX_TIMEOUT` | `3600` | Seconds to wait for a RAG indexing run before returning 504. |
| `GENERATION_MODEL` | `qwen3.5:9b` | Default answer model. Resolved as env var → `rag_system.main.OLLAMA_CONFIG` → literal default. |
| `ENRICHMENT_MODEL` | `qwen3.5:4b` | Recorded in index metadata as the enrich/overview model. The gateway no longer calls it (routing is local since Phase 2.3). |
| `EMBEDDING_MODEL` | `microsoft/harrier-oss-v1-0.6b` | Recorded in the default index metadata. |
| `DB_PATH` | `backend/chat_data.db` (`/app/backend/chat_data.db` in Docker) | SQLite file for sessions, messages, indexes and documents. |
| `LANCEDB_PATH` | `storage.lancedb_uri` from the default pipeline profile (`./lancedb`) | Vector store the backend drops tables from when an index is deleted. |

The backend imports `PIPELINE_CONFIGS` and `OLLAMA_CONFIG` from `rag_system.main` when the
package is importable, and degrades gracefully (printing a warning) when it is not.

## Request routing

`POST /sessions/<id>/messages` decides per message whether to use RAG. The decision is
made by `should_use_rag(message, idx_ids, force_rag)` — a module-level function in
`server.py` with no LLM call, no file reads and no network I/O:

1. `force_rag` (or `forceRag`) in the body → RAG, unconditionally.
2. No indexes linked to the session → direct Ollama answer.
3. The whole message is smalltalk (`hello`, `thanks!`, `bye`, `ok` — a whole-message
   allowlist regex, max six words) or a question about the assistant itself
   (`who are you`, `what model are you`) → direct Ollama answer.
4. Everything else → RAG.

RAG queries are forwarded to `POST {RAG_API_URL}/chat`; the answer and `source_documents`
come back from there. Direct answers call Ollama with thinking disabled.

The gate is intentionally biased toward RAG ("escalate, don't pre-decide"). The RAG API's
agent triage runs on every forwarded request and can still answer directly, so the gateway
only has to keep greetings out of the retrieval pipeline — it does not have to be right
about which questions the documents can answer. The pre-retrieval enrichment-model router
and its keyword/length fallback (which misrouted any question containing the word "test")
were removed in Phase 2.3; `Documentation/research/` has the evidence.

`backend/test_gateway_routing.py` covers the gate: `.venv/bin/python backend/test_gateway_routing.py`.

**Option casing.** Chat and index options are accepted in both `camelCase` and
`snake_case` and normalised to one canonical `snake_case` key before being forwarded
(`rerankerTopK` → `reranker_top_k`, `enableLatechunk` → `enable_latechunk`, and so on).
Options the caller omits are left out of the payload entirely so the RAG pipeline's own
defaults apply.

**Chat history.** This server is the only writer of chat message rows. The frontend's
streaming path talks to `:8001/chat/stream` directly and then persists the completed
turn here via `POST /sessions/<id>/messages/save`.

## API Endpoints

`Documentation/api_reference.md` has the request/response detail; this is the route table.

### GET

| Route | Returns |
| --- | --- |
| `/health` | `{status, ollama_running, available_models, database_stats}` |
| `/sessions` | `{sessions, total}` |
| `/sessions/cleanup` | `{message, cleanup_count}` — deletes sessions with no messages |
| `/sessions/<session_id>` | `{session, messages}` |
| `/sessions/<session_id>/documents` | `{session, files, file_count}` |
| `/sessions/<session_id>/indexes` | `{indexes, total}` |
| `/models` | `{generation_models, embedding_models}` |
| `/indexes` | `{indexes, total}` |
| `/indexes/<index_id>` | the index record, or 404 |

### POST

| Route | Body | Returns |
| --- | --- | --- |
| `/chat` | `{message, model?, conversation_history?}` | `{response, model, message_count}` — legacy sessionless path, always direct Ollama |
| `/sessions` | `{title?, model?}` | `{session, session_id}` (201) |
| `/sessions/<id>/messages` | `{message, model?, force_rag?, …chat options}` | `{response, session, source_documents, used_rag}` |
| `/sessions/<id>/upload` | `multipart/form-data`, field `files` | `{message, uploaded_files}` |
| `/sessions/<id>/index` | optional JSON index options | the RAG API's `/index` response, or `{message: "No documents to index for this session."}` |
| `/sessions/<id>/rename` | `{title}` | `{message, session}` |
| `/sessions/<sid>/indexes/<iid>` | — | `{message}` — links an index to a session |
| `/indexes` | `{name, description?, metadata?}` | `{index_id}` (201) |
| `/indexes/<id>/upload` | `multipart/form-data`, field `files` | `{message, uploaded_files}` |
| `/indexes/<id>/build` | optional JSON index options | `{response, …applied options}` |

### DELETE

| Route | Returns |
| --- | --- |
| `/sessions/<session_id>` | `{deleted: true}`, or 404 |
| `/indexes/<index_id>` | `{message, index_id}`, or 404 — also drops the index's LanceDB table |

`OPTIONS` on any path returns the CORS preflight headers (`GET, POST, DELETE, OPTIONS`).

## Testing

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!"}'
```

For an end-to-end check of the whole stack use `python run_system.py --health` or
`python system_health_check.py`. There is no automated test suite in this directory.

## Frontend integration

The frontend reads `NEXT_PUBLIC_API_URL` (default `http://localhost:8000`) for this
server and `NEXT_PUBLIC_RAG_API_URL` (default `http://localhost:8001`) for the streaming
endpoint. Both are inlined at `next build` time, so they must be set before the frontend
is built.

## Implemented today

- Session-scoped chat with SQLite-persisted history and auto-generated titles.
- Document upload (per session and per index) into `shared_uploads/`.
- Indexing delegated to the RAG API, with the resulting configuration stored as index
  metadata.
- Vector-store bookkeeping: index records point at their LanceDB table, and deleting an
  index drops that table (the late-chunk `_lc` sibling is left behind).
- Retrieval-augmented answers with source documents, plus a direct-LLM fast path for
  general questions.

Streaming is **not** served by this process — the frontend streams from the RAG API's
`/chat/stream` endpoint directly.

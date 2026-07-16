# LocalGPT API Reference

This reference matches the FastAPI/OpenAPI implementation. The browser calls the same-origin Next.js `/api/backend/*` proxy, which forwards to the backend on `http://127.0.0.1:8000`. Port `8001` is an internal RAG worker. Interactive docs are at `/docs`; the checked-in contract is `Documentation/openapi.json`.

## Security and runtime boundaries

- Both services bind to `127.0.0.1` by default. Docker sets the container bind address to `0.0.0.0` but publishes ports only on the host loopback interface.
- CORS accepts the comma-separated `LOCALGPT_ALLOWED_ORIGINS` list. The default allows the frontend on `localhost:3000` and `127.0.0.1:3000`.
- If `LOCALGPT_API_TOKEN` is set, direct API clients send `Authorization: Bearer <token>`. The Next.js proxy injects this server-side, so the token is never bundled into browser JavaScript.
- The token-injecting proxy rejects cross-origin mutation requests by comparing the request `Origin` with the application host.
- Uploads support PDF, DOCX, PPTX, XLSX, HTML, Markdown, text, CSV, JSON, and EML. They default to 50 MiB maximum, are checked for obvious signature/type mismatches and pathological Office archives, and are stored beneath `LOCALGPT_UPLOAD_DIR`.
- The RAG `/index` endpoint accepts only existing paths beneath that upload directory. It is not a general server-side file reader.

## Backend API

Base URL: `http://127.0.0.1:8000`

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Service, Ollama, model, and database health |
| GET/POST | `/sessions` | List or create chat sessions |
| GET/DELETE | `/sessions/{id}` | Read a session with messages or delete it |
| POST | `/sessions/{id}/rename` | Rename with `{ "title": "..." }` |
| POST | `/sessions/{id}/messages` | Non-streaming session chat |
| POST | `/sessions/{id}/messages/stream` | Streaming session chat over SSE |
| GET | `/sessions/{id}/documents` | List session uploads |
| POST | `/sessions/{id}/upload` | Multipart upload using the `files` field |
| POST | `/sessions/{id}/index` | Build a dedicated index from temporary chat attachments and link it |
| GET | `/sessions/{id}/indexes` | List linked indexes |
| POST | `/sessions/{sid}/indexes/{iid}` | Link a compatible index |
| GET/POST | `/indexes` | List or create indexes |
| GET/DELETE | `/indexes/{id}` | Read or delete an index |
| POST | `/indexes/{id}/upload` | Multipart upload using the `files` field |
| POST | `/indexes/{id}/build` | Build or replace an index |
| GET | `/models` | List generation and embedding models |

## Durable v1 API

| Method | Path | Purpose |
|---|---|---|
| POST/GET | `/v1/runs` | Submit or list durable message runs |
| GET/DELETE | `/v1/runs/{id}` | Read or delete a terminal run |
| GET | `/v1/runs/{id}/events` | Replay/follow SSE events; resume with `Last-Event-ID` |
| POST | `/v1/runs/{id}/cancel` | Request cooperative cancellation |
| POST | `/v1/runs/{id}/retry` | Create a new run from a failed/cancelled/complete request |
| POST | `/v1/index-jobs` | Submit a durable index build without holding the request open |
| GET | `/v1/models` | Provider-neutral capability discovery |
| POST | `/v1/embeddings` | Ollama-compatible batch embedding |
| GET | `/v1/tools` | List registered JSON-schema tool contracts |
| POST | `/v1/tools/{name}/execute` | Execute a tool under explicit permissions/approval |
| GET/POST | `/v1/skills` | List or create immutable skill definitions |
| POST | `/v1/skills/{id}/versions` | Add an immutable version |
| GET | `/v1/artifacts` | List session- or index-scoped artifacts |
| GET | `/v1/artifacts/{id}` | Download with matching `session_id` or `index_id` scope |
| GET | `/v1/connectors` | List configured connector metadata (never secrets) |

`POST /v1/runs` accepts `message` or a provider-neutral `messages` array, model settings, bounded iteration/tool/time budgets, an allowlist of tools, skill IDs, requested permissions, explicitly approved tools, and retrieval overrides. Server permissions always cap request permissions.

Events are append-only and use normal SSE fields:

```text
id: 41
event: tool.completed
data: {"tool":"search_knowledge","result":{...}}

id: 42
event: run.completed
data: {"status":"completed"}
```

### Session chat

```json
{
  "message": "What are the contract termination terms?",
  "model": "qwen3:8b",
  "search_type": "hybrid",
  "dense_weight": 0.7,
  "retrieval_k": 20,
  "context_window_size": 1,
  "reranker_top_k": 10,
  "query_decompose": true,
  "compose_sub_answers": true,
  "ai_rerank": true,
  "context_expand": true,
  "verify": true,
  "force_rag": false,
  "provence_prune": false
}
```

The non-streaming response contains `response`, `source_documents`, `route`, `used_rag`, `user_message_id`, `ai_message_id`, and the updated `session`.

The legacy streaming endpoint accepts either `query` or `message`. Events have the form:

```text
data: {"type":"token","data":{"text":"..."}}

data: {"type":"complete","data":{"answer":"...","source_documents":[],"user_message_id":"...","assistant_message_id":"...","session":{...}}}
```

The backend writes the user message once before delegation and the assistant message once when the upstream stream completes. The RAG service does not own chat persistence.

### Index build

The browser-facing build endpoint accepts camelCase options for compatibility with the UI:

```json
{
  "latechunk": false,
  "doclingChunk": false,
  "chunkSize": 512,
  "chunkOverlap": 64,
  "retrievalMode": "hybrid",
  "windowSize": 2,
  "enableEnrich": true,
  "embeddingModel": "Qwen/Qwen3-Embedding-0.6B",
  "enrichModel": "qwen3:0.6b",
  "overviewModel": "qwen3:0.6b",
  "batchSizeEmbed": 50,
  "batchSizeEnrich": 25
}
```

Rebuilding replaces the main and late-chunk vector tables, rewrites the overview manifest, and invalidates the in-process semantic query cache; it does not append duplicate chunks. Deleting an index first asks the RAG artifact owner to remove both tables and its overview, then removes SQLite metadata and uploaded source files.

Chat attachments use `/sessions/{id}/upload` followed by `/sessions/{id}/index`. The second call creates a normal dedicated index, builds it, links it to the session, and transfers ownership of the temporary uploads to that index. Its response contains `index_id`, the new `index`, and build details.

## Internal RAG API

Base URL: `http://127.0.0.1:8001`

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Lightweight health probe |
| GET | `/models` | Available models |
| POST | `/chat` | Internal non-streaming agent call |
| POST | `/chat/stream` | Internal SSE agent call |
| POST | `/index` | Validated internal index build |
| DELETE | `/indexes/{id}` | Delete RAG-owned tables and overview before backend metadata deletion |

`/chat` and `/chat/stream` accept `table_names` so a session can retrieve across all linked indexes. `search_type` accepts `hybrid`, `dense`/`vector`, or `lexical`/`fts`/`bm25` aliases.

The canonical `/index` body uses snake_case:

```json
{
  "file_paths": ["/absolute/path/beneath/LOCALGPT_UPLOAD_DIR/document.pdf"],
  "index_id": "index-id",
  "table_name": "text_pages_index-id",
  "chunk_size": 512,
  "chunk_overlap": 64,
  "enable_latechunk": false,
  "enable_docling_chunk": false,
  "retrieval_mode": "hybrid",
  "window_size": 2,
  "enable_enrich": true,
  "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
  "enrich_model": "qwen3:0.6b",
  "overview_model": "qwen3:0.6b",
  "batch_size_embed": 50,
  "batch_size_enrich": 25
}
```

Legacy camelCase model keys are accepted, but new clients should use the canonical names above.

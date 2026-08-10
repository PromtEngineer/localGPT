# 📚 API Reference (Backend Gateway & RAG API)

_Last updated: 2026-08-08_

Two HTTP services, both reachable from the browser:

| Service | Base URL | Source |
|---------|----------|--------|
| Backend gateway | `http://localhost:8000` | `backend/server.py` |
| RAG API | `http://localhost:8001` | `rag_system/api_server.py` |

Both send `Access-Control-Allow-Origin: *` on every response and answer `OPTIONS` preflights. Neither implements authentication.

**Wire format** is snake_case. Both services additionally accept camelCase and normalise it to the same canonical key at parse time, so `rerankerTopK` and `reranker_top_k` are interchangeable. When both spellings are present, the explicit snake_case value wins. The gateway uses an explicit alias table that covers every option it accepts (plus a few legacy names such as `latechunk` and `decompose`); the RAG API converts any camelCase key generically.

---

## 1. Backend Gateway — `http://localhost:8000`

### 1.1 Route table

| Endpoint | Method | Description | Request body | Success response |
|----------|--------|-------------|--------------|------------------|
| `/health` | GET | Health probe with Ollama status and DB stats | – | `{ status, ollama_running, available_models, database_stats }` |
| `/chat` | POST | Stateless chat, no session, no retrieval | `{ message, model?, conversation_history? }` | `{ response, model, message_count }` |
| `/models` | GET | Available generation / embedding models | – | `{ generation_models, embedding_models }` |
| `/sessions` | GET | List sessions | – | `{ sessions, total }` |
| `/sessions` | POST | Create a session | `{ title?, model? }` | 201 `{ session, session_id }` |
| `/sessions/cleanup` | GET | Delete sessions that have no messages | – | `{ message, cleanup_count }` |
| `/sessions/<id>` | GET | Session plus its messages | – | `{ session, messages }` |
| `/sessions/<id>` | DELETE | Delete a session and its messages | – | `{ deleted: true }` |
| `/sessions/<id>/rename` | POST | Rename a session | `{ title }` | `{ message, session }` |
| `/sessions/<id>/messages` | POST | Session chat (persisted) | [Session chat request](#12-session-chat-request) | `{ response, session, source_documents, used_rag }` |
| `/sessions/<id>/messages/save` | POST | Persist a completed streamed turn (the browser calls this after `POST :8001/chat/stream` finishes) | `{ user_message, assistant_message, source_documents?, steps? }` | `{ session, user_message_id, ai_message_id }` — sources and steps are stored in the assistant message's `metadata.source_documents` / `metadata.steps` |
| `/sessions/<id>/documents` | GET | Files uploaded to a session | – | `{ session, files, file_count }` |
| `/sessions/<id>/upload` | POST | Upload files to a session | multipart, field `files` | `{ message, uploaded_files }` |
| `/sessions/<id>/index` | POST | Index the session's documents | [Index options](#14-index-options) (optional) | the RAG API `/index` response |
| `/sessions/<id>/indexes` | GET | Indexes linked to a session | – | `{ indexes, total }` |
| `/sessions/<sid>/indexes/<idx>` | POST | Link an index to a session | – | `{ message }` |
| `/indexes` | GET | List all indexes | – | `{ indexes, total }` |
| `/indexes` | POST | Create a named index | `{ name, description?, metadata? }` | 201 `{ index_id }` |
| `/indexes/<id>` | GET | One index | – | the index object (see below) |
| `/indexes/<id>` | DELETE | Delete an index, its links and its LanceDB table | – | `{ message, index_id }` |
| `/indexes/<id>/upload` | POST | Upload files to an index | multipart, field `files` | `{ message, uploaded_files }` |
| `/indexes/<id>/build` | POST | Build / rebuild the index | [Index options](#14-index-options) (optional) | `{ response, ...echoed options }` |

`uploaded_files` entries are `{ filename, stored_path }`. The index object returned by `GET /indexes/<id>` is `{ id, name, description, created_at, updated_at, vector_table_name, metadata, documents[] }` — it is **not** wrapped in `{ index: … }`.

> **Body required.** `POST /chat`, `POST /sessions`, `POST /sessions/<id>/messages` and `POST /indexes` read `Content-Length` unconditionally; send a JSON body (at minimum `{}`) or the request fails with a 500. `POST /sessions/<id>/rename` returns a clean `400 { "error": "Request body required" }`. `POST /sessions/<id>/index` and `POST /indexes/<id>/build` treat the body as optional.

### 1.2 Session chat request

`POST /sessions/<id>/messages`

```jsonc
{
  "message": "string",            // required
  "model": "qwen3.5:9b",          // optional – generation model for this request
  "force_rag": false,             // optional – skip gateway routing, always call the RAG API

  // Retrieval options, forwarded to the RAG API when the RAG route is taken.
  // Omitted options are not forwarded, so the pipeline profile default applies.
  "compose_sub_answers": true,
  "query_decompose": true,        // alias: "decompose"
  "ai_rerank": true,              // profile default when omitted is OFF (eval/DECISIONS.md)
  "context_expand": true,
  "verify": true,
  "retrieval_k": 20,
  "context_window_size": 1,
  "reranker_top_k": 10,
  "retrieval_mode": "hybrid",     // "hybrid" | "vector_only" | "fts_only" (alias: "search_type")
  "provence_prune": false,
  "provence_threshold": 0.1,
  "filters": { }                  // optional metadata filter object, forwarded verbatim and
                                  // validated by the RAG API (§2.1). A present filter also
                                  // forces the RAG route, like force_rag.
}
```

camelCase aliases are accepted for all of them (`composeSubAnswers`, `queryDecompose`, `aiRerank`, `contextExpand`, `retrievalK`, `contextWindowSize`, `rerankerTopK`, `retrievalMode`, `searchType`, `provencePrune`, `provenceThreshold`, `forceRag`).

Response:

```jsonc
{
  "response": "string",           // assistant answer, <think> tags stripped
  "session": { /* ChatSession */ },
  "source_documents": [],         // empty on the direct-LLM route
  "used_rag": true
}
```

This endpoint persists the turn itself: it writes the user message, derives a session title from the first message, then writes the assistant message with its sources in `metadata.source_documents`. (Streamed turns are persisted separately via `/messages/save`.)

`force_rag` decides gateway routing **and** is forwarded to the RAG API, so the agent's own triage is skipped too.

### 1.3 Gateway routing

`POST /sessions/<id>/messages` picks its path with a deterministic gate — no model call, no retrieval, sub-millisecond:

1. `force_rag: true` (or `forceRag`) → RAG, unconditionally.
2. Session has no linked indexes → answered directly by Ollama, no retrieval.
3. The whole message is smalltalk (`hello`, `thanks!`, `bye`, `ok` — an allowlist regex capped at six words) or a question about the assistant itself (`who are you`, `what model are you`) → direct.
4. Anything else → RAG.

`used_rag` in the response tells you which way it went. The gate errs toward RAG on purpose: the RAG API's own agent triage still runs on every forwarded request and can answer directly without retrieving, so a false "use RAG" costs one triage call, not a wrong answer. If you need the answer grounded in documents regardless, send `force_rag`.

The streaming path (`:8001/chat/stream`, the UI default) never touches this gate — the browser calls the RAG API directly and only the agent triage applies.

### 1.4 Index options

Accepted by `POST /sessions/<id>/index` and `POST /indexes/<id>/build`, normalised and forwarded to the RAG API `/index`. **Options you omit are not sent**, so the RAG API's own defaults apply (§2.5).

```jsonc
{
  "chunk_size": 512,
  "window_size": 2,
  "retrieval_mode": "hybrid",       // "hybrid" | "vector_only" | "fts_only"
  "enable_enrich": true,
  "enable_latechunk": false,        // aliases: "latechunk", "enableLatechunk"
  "enable_docling_chunk": true,     // aliases: "doclingChunk", "enableDoclingChunk"; false = legacy chunker — see §2.5
  "embedding_model": "microsoft/harrier-oss-v1-0.6b",
  "enrich_model": "qwen3.5:4b",
  "overview_model_name": "qwen3.5:4b",   // aliases: "overviewModel", "overview_model"
  "batch_size_embed": 50,
  "batch_size_enrich": 25
}
```

`POST /sessions/<id>/index` returns the RAG API's `/index` response unchanged, or `200 { "message": "No documents to index for this session." }` when the session has no uploaded files.

`POST /indexes/<id>/build` echoes the canonical options back alongside the RAG API response, renaming `enable_latechunk` → `latechunk`, `enable_docling_chunk` → `docling_chunk` and `overview_model_name` → `overview_model`, and stores the same values in the index's metadata. If the RAG API reports that the table already exists, the build is treated as idempotent and returns `{ message: "Index already built – skipping rebuild.", note }`.

### 1.5 Error responses

| Status | When |
|--------|------|
| 400 | Missing required field, invalid JSON, no files in a multipart upload |
| 404 | Unknown route, unknown session, unknown index |
| 500 | Unhandled server error, or the RAG API returned a non-200 |
| 502 | Could not connect to the RAG API (`RAG_API_URL`) |
| 503 | Ollama is not reachable (`POST /chat` only) |
| 504 | The RAG API did not answer within `RAG_API_TIMEOUT` (chat, default 600 s) or `RAG_API_INDEX_TIMEOUT` (indexing, default 3600 s) |

Error bodies are `{ "error": "..." }`.

---

## 2. RAG API — `http://localhost:8001`

| Endpoint | Method | Description | Request body | Success response |
|----------|--------|-------------|--------------|------------------|
| `/health` | GET | Liveness probe | – | `{ "status": "ok" }` |
| `/models` | GET | Models available to the active LLM backend | – | `{ generation_models, embedding_models }` |
| `/chat` | POST | Run the full agent pipeline | [Chat request](#21-chat-request) | `{ answer, source_documents }` |
| `/chat/stream` | POST | Same pipeline, streamed as SSE | [Chat request](#21-chat-request) | `text/event-stream` |
| `/index` | POST | Index documents | [Index request](#25-index-request) | see §2.6 |

Unknown routes return 404 `{ "error": "Not Found" }`.

### 2.1 Chat request

```jsonc
{
  "query": "string",              // required
  "session_id": "string",         // optional – loads the session's overviews and index metadata
  "table_name": "string",         // optional – LanceDB table; otherwise resolved from session_id
  "model": "qwen3.5:9b",          // optional – generation model for this request only

  "compose_sub_answers": true,    // optional – profile default when omitted
  "query_decompose": true,        // optional – profile default when omitted
  "ai_rerank": true,              // optional – profile default when omitted, which is OFF (eval/DECISIONS.md)
  "context_expand": true,         // optional – false forces a context window of 0
  "verify": true,                 // optional – profile default when omitted
  "force_rag": false,             // optional – skip triage and go straight to retrieval

  "retrieval_k": 20,              // optional – profile default when omitted
  "context_window_size": 1,       // optional – profile default when omitted
  "reranker_top_k": 10,           // optional – profile default when omitted

  "retrieval_mode": "hybrid",     // "hybrid" | "vector_only" | "fts_only"; alias "search_type"
  "provence_prune": false,        // optional – sentence-level pruning
  "provence_threshold": 0.1,      // optional – pruning threshold

  "filters": {                    // optional – metadata filter (roadmap 4.4); prefilters BOTH search legs
    "document_id": { "eq": "07_nda.pdf" },        // also: {"in": [...]}, {"contains": "..."}
    "chunk_index": { "gte": 0, "lt": 10 }         // also on document_name (contains), chunk_id (eq/in)
  }
}
```

Notes:

* Every option, including `retrieval_k`, `context_window_size` and `reranker_top_k`, falls back to the pipeline profile when omitted; only options you actually send override the profile.
* An unsupported `retrieval_mode` is rejected with `400 { "error": "Unsupported retrieval mode '…'. Supported: hybrid, vector_only, fts_only." }`.
* `model` is applied for the duration of the request and then restored. A model id that does not match the active `LLM_BACKEND` (for example an Ollama tag while `LLM_BACKEND=watsonx`) is ignored with a warning.
* If the table's index metadata records an `embedding_model`, the retrieval pipeline switches to it before searching.
* `filters` is validated by `rag_system/retrieval/filters.py`: unknown fields/operators, wrong types, empty IN-lists, an empty object, and values containing quoting characters (`'`, `"`, `\`, `;`, backtick, control chars — refused, never escaped) all return `400` with the validator's message. Sending `filters` also skips triage, like `force_rag`. Page/date filtering is not supported (the values live inside the metadata JSON column).

### 2.2 Chat response

```jsonc
{
  "answer": "string",
  "source_documents": [
    {
      "chunk_id": "string",
      "text": "string",
      "score": 0.0164,          // higher is better; RRF score in hybrid mode
      "document_id": "report.pdf",
      "chunk_index": 12,
      "metadata": { },
      "rerank_score": 0.87,     // present only when reranking ran
      "bm25": 4.21              // present only when the full-text leg matched this chunk
    }
  ],
  "token_usage": {              // per-query token accounting (roadmap 4.5, always on)
    "by_stage": { "synthesis": { "prompt_tokens": 1192, "output_tokens": 861, "calls": 1 } },
    "total": { "prompt_tokens": 1192, "output_tokens": 861, "calls": 1, "total_tokens": 2053 }
  },
  "document_escalation": [ ]    // present only when full-document escalation fired (flag-gated, off by default)
}
```

An absent key in `by_stage` means that stage made no LLM call, not that it cost
zero tokens. Only Ollama reports real counts; watsonx reports zeros.

There is no top-level `confidence` or `reasoning` field. When verification is enabled the confidence is appended to `answer` as `" [Confidence: N%]"`, plus `" [Warning: Low confidence. Groundedness: <bool>]"` when the answer is judged ungrounded or scores below 50. Nothing is appended when the verifier's score parses as 0.

When nothing is retrieved, `answer` is `"I could not find an answer in the documents."` and `source_documents` is empty.

### 2.3 `POST /chat/stream` (SSE)

Same request body. The response is `text/event-stream`; each event is a single line:

```
data: {"type": "<event>", "data": <payload>}
```

| Event | Payload | Emitted when |
|-------|---------|--------------|
| `analyze` | `{query}` | Start of the run |
| `direct_answer` | `{}` | Triage chose a direct answer |
| `decomposition` | `{sub_queries}` | Query decomposition produced sub-queries |
| `retrieval_started` | `{mode}` or `{count}` | `{mode}` from the retrieval pipeline, `{count}` from the decomposition branch |
| `retrieval_done` | `{count}` | Retrieval finished |
| `retrieval_retry` | `{score_before, score_after, kept, …}` | The evidence-sufficiency retry ran (design_rationale §5) |
| `crossref_hop` | `{targets, chunks_added}` | The cross-reference hop pulled chunks (flag-gated, off by default) |
| `document_escalation` | `{document_id, document_name, chunks_used, chunks_total, approx_tokens, truncated, signal, score, threshold, token_budget}` | Full-document escalation fired (flag-gated, off by default); never contains the document text |
| `rerank_started` / `rerank_done` | `{count}` | Reranking |
| `context_expand_started` / `context_expand_done` | `{count}` | Context expansion |
| `prune_started` / `prune_done` | `{count}` | Provence pruning (only when enabled) |
| `token` | `{text}` | Streamed answer tokens |
| `sub_query_token` | `{index, text, question}` | Tokens from a parallel sub-query |
| `sub_query_result` | `{index, query, answer, source_documents}` | A sub-query finished |
| `single_query_result` | the pipeline result | Decomposition produced exactly one sub-query |
| `final_answer` | `{answer, source_documents}` | Composed answer ready |
| `complete` | `{answer, source_documents, token_usage}` | Final event; clients may close here. `token_usage` has the shape shown in §2.2 |
| `error` | `{error}` | Failure after the stream opened |

> This endpoint does **not** write to SQLite. The browser's default chat path calls it directly and, once the `complete` event arrives, persists the finished turn via `POST :8000/sessions/<id>/messages/save`. Clients that consume the stream directly must do the same if they want the turn in the session history.

### 2.4 `GET /models`

```jsonc
{
  "generation_models": ["qwen3.5:4b", "qwen3.5:9b"],
  "embedding_models": ["Qwen/Qwen3-Embedding-0.6B", "Qwen/Qwen3-Embedding-4B", "Qwen/Qwen3-Embedding-8B", "microsoft/harrier-oss-v1-0.6b"]
}
```

With `LLM_BACKEND=ollama` the generation list comes from `GET {OLLAMA_HOST}/api/tags` (5 s timeout; on failure the list is simply shorter), split by a substring match on `embed` / `bge` / `embedding`. With `LLM_BACKEND=watsonx` it lists the configured WatsonX generation and enrichment models. The embedding list always includes the pipeline's configured embedding model, the default `microsoft/harrier-oss-v1-0.6b` and the three Qwen3-Embedding sizes; it is returned sorted and de-duplicated.

### 2.5 Index request

```jsonc
{
  "file_paths": ["/abs/path/a.pdf", "/abs/path/b.docx"],  // required
  "session_id": "string",              // optional – overview file + metadata target
  "table_name": "string",              // optional – otherwise resolved from session_id

  "chunk_size": 512,                   // default 512  – token budget per chunk
  "window_size": 2,                    // default 2    – contextual-enrichment window
  "enable_enrich": true,               // default true
  "enable_latechunk": false,           // default false
  "enable_docling_chunk": true,        // default true – false selects the legacy fixed-size chunker
  "retrieval_mode": "hybrid",          // optional – validated, recorded on the index config
  "embedding_model": "microsoft/harrier-oss-v1-0.6b",
  "enrich_model": "qwen3.5:4b",
  "overview_model_name": "qwen3.5:4b",
  "batch_size_embed": 50,              // default 50
  "batch_size_enrich": 25              // default 25
}
```

* `session_id` is **optional**. Without it the default LanceDB table is used and overviews go to the global `index_store/overviews/overviews.jsonl` instead of a per-index file.
* `retrieval_mode` cannot change the artifacts written at index time; it is validated (400 on an unsupported value) and stored in the index config as `retrieval.search_type`. The mode that matters is the one you send at query time.
* `embedding_model` is applied to this build **and** recorded in the index metadata (when `session_id` is present), which is what makes queries against that index use the same embedder.
* `enable_latechunk` defaults to `false` here, so an HTTP build without the flag writes no `<table>_lc` table even though the `default` profile enables late chunking.
* `enable_docling_chunk` defaults to `true`; sending `false` selects the legacy fixed-size chunker instead of Docling's structure-aware chunker.
* Unknown fields are ignored silently.

### 2.6 Index response

```jsonc
{
  "message": "Indexing process for 2 file(s) completed successfully.",
  "table_name": "text_pages_<index_id>",
  "latechunk": false,
  "docling_chunk": true,
  "indexing_config": {
    "chunk_size": 512,
    "retrieval_mode": "hybrid",
    "window_size": 2,
    "enable_enrich": true,
    "embedding_model": "microsoft/harrier-oss-v1-0.6b",
    "enrich_model": "qwen3.5:4b",
    "overview_model_name": "qwen3.5:4b",
    "batch_size_embed": 50,
    "batch_size_enrich": 25
  }
}
```

`indexing_config.embedding_model` reports the model actually used for the build, not the raw request value. There is no `indexed_files` field.

Errors: `400 { "error": "A 'file_paths' list is required." }`, `400 { "error": "Invalid JSON" }`, `400` for an unsupported `retrieval_mode`, and `500 { "error": "Failed to start indexing: …" }`.

---

## 3. Frontend Wrapper (`src/lib/api.ts`)

The typed client exported as `chatAPI`. Base URLs come from `NEXT_PUBLIC_API_URL` (default `http://localhost:8000`) and `NEXT_PUBLIC_RAG_API_URL` (default `http://localhost:8001`), both inlined at build time. **The browser talks to both origins**, so a deployment must expose both.

| Method | Target |
|--------|--------|
| `checkHealth()` | `GET :8000/health` |
| `sendMessage({message, model?, conversation_history?})` | `POST :8000/chat` |
| `getSessions()` | `GET :8000/sessions` |
| `createSession(title?, model?)` | `POST :8000/sessions` |
| `getSession(sessionId)` | `GET :8000/sessions/<id>` |
| `sendSessionMessage(sessionId, message, opts)` | `POST :8000/sessions/<id>/messages` |
| `saveStreamedTurn(sessionId, userMessage, assistantMessage, sourceDocuments?)` | `POST :8000/sessions/<id>/messages/save` |
| `deleteSession(sessionId)` | `DELETE :8000/sessions/<id>` |
| `renameSession(sessionId, title)` | `POST :8000/sessions/<id>/rename` |
| `uploadFiles(sessionId, files)` | `POST :8000/sessions/<id>/upload` |
| `indexDocuments(sessionId)` | `POST :8000/sessions/<id>/index` (no options) |
| `getModels()` | `GET :8000/models` |
| `createIndex(name, description?, metadata?)` | `POST :8000/indexes` |
| `uploadFilesToIndex(indexId, files)` | `POST :8000/indexes/<id>/upload` |
| `buildIndex(indexId, opts)` | `POST :8000/indexes/<id>/build` |
| `listIndexes()` | `GET :8000/indexes` |
| `getSessionIndexes(sessionId)` | `GET :8000/sessions/<id>/indexes` |
| `deleteIndex(indexId)` | `DELETE :8000/indexes/<id>` |
| `linkIndexToSession(sessionId, indexId)` | `POST :8000/sessions/<sid>/indexes/<idx>` |
| **`streamSessionMessage(params, onEvent)`** | **`POST :8001/chat/stream`** — the default chat path |

The camelCase argument names on these methods (`retrievalK`, `rerankerTopK`, `doclingChunk`, …) are TypeScript parameter names. `sendSessionMessage` and `streamSessionMessage` serialise them to snake_case JSON; `buildIndex` sends camelCase, which both servers normalise.

Exported model defaults, kept in step with `rag_system/main.py`:

```ts
export const DEFAULT_GENERATION_MODEL = 'qwen3.5:9b';
export const DEFAULT_ENRICHMENT_MODEL = 'qwen3.5:4b';
export const DEFAULT_EMBEDDING_MODEL  = 'microsoft/harrier-oss-v1-0.6b';
```

---

_Derived by reading `backend/server.py`, `rag_system/api_server.py` and `src/lib/api.ts`. Keep it in sync with route, option and response-shape changes._

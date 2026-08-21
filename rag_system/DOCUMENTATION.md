# RAG System Documentation

Reference for the `rag_system` package: pipelines, configuration keys, the HTTP API and
the CLI. For the shorter tour of the package, see [`README.md`](README.md).

Everything below is described as the code behaves today. Where a knob exists but nothing
reads it, that is called out explicitly rather than glossed over.

## 1. Where this runs

`rag_system` is one process in a four-process system:

```
Next.js frontend :3000  ──►  backend gateway :8000  ──►  RAG API :8001  ──►  Ollama :11434
        └────────────────── /chat/stream (SSE) ─────────────►
```

The RAG API (`rag_system/api_server.py`) is a standard-library `http.server` running on a
plain `socketserver.TCPServer`, so **requests are handled one at a time**. A long chat or
indexing call blocks every other request to port 8001 until it finishes. Plan for a single
concurrent user per RAG API process.

The API opens the same SQLite file as the backend gateway (`backend/chat_data.db`, or
`DB_PATH`), but only to read session→index links and to read/write index metadata.
**Chat message rows are written exclusively by `backend/server.py`.** The frontend's
streaming path posts straight to `:8001/chat/stream`; the stream itself persists nothing,
and the browser saves the completed turn through the gateway
(`POST :8000/sessions/<id>/messages/save`) once the `complete` event arrives.

## 2. Indexing pipeline

`rag_system/pipelines/indexing_pipeline.py`. Public entry point:

```python
IndexingPipeline(config, llm_client, ollama_config).run(file_paths: list[str])
```

`run()` also accepts the legacy keyword alias `documents=`. There is no
`process_documents()` method.

### Steps

1.  **Conversion** — `ingestion/document_converter.py`. Docling converts the file to a
    single Markdown string plus the `DoclingDocument` object.
    -   `.pdf`, `.docx`, `.html`, `.htm`, `.md` go through Docling; `.txt` is read
        directly and wrapped in a fenced block.
    -   PDFs are first probed with PyMuPDF (`fitz`). If any page yields text, the no-OCR
        converter is used; otherwise the OCR converter is used.
    -   OCR engine selection is dynamic: `OcrMacOptions` (macOS only), then `EasyOcrOptions`,
        `RapidOcrOptions`, `TesseractOcrOptions`, `TesseractCliOcrOptions` — the first one
        whose backend module or binary is actually installed wins. If none is available,
        Docling's default OCR settings are used and a message is printed.
    -   The three converters (no-OCR, OCR, general) are built independently, so a failing
        OCR engine does not disable the other paths.

2.  **Chunking** — `chunker_mode` selects the chunker.
    -   `docling` (default): `ingestion/docling_chunker.py`. Walks the DoclingDocument
        element tree in reading order, emits tables as atomic Markdown chunks, tracks the
        heading path, and token-packs paragraphs up to `chunk_size` using the embedding
        model's tokenizer. Each chunk records `heading_path`, `heading_level`,
        `block_type` and (when available) `page`. If the tree walk fails it falls back to
        splitting the exported Markdown with `overlap_sentences` sentences of overlap
        (default 1).
    -   `legacy`: `ingestion/chunking.py::MarkdownRecursiveChunker`. Recursively splits on
        `\n## `, `\n### `, `\n#### `, code fences and blank lines, measuring size with the
        embedding model's tokenizer, with `max_chunk_size = chunk_size` and
        `min_chunk_size = max(1, chunk_size // 4)`.
    -   If the Docling chunker fails to initialise, the pipeline falls back to the legacy
        chunker automatically.
    -   Each chunk gets a sequential `metadata.chunk_index` within its document.

3.  **Document overview** — `indexing/overview_builder.py`, on unless
    `overview.enabled` is `false`. The enrichment model summarises the first
    `overview_first_n_chunks` (or `overview.max_chunks`, default 5) chunks into one
    paragraph, appended as JSONL to `overview_path`
    (default `index_store/overviews/overviews.jsonl`). These overviews are what the query
    router reads later. Failures here are logged and do not abort indexing.

4.  **Contextual enrichment** — `indexing/contextualizer.py`, when
    `contextual_enricher.enabled`. For each chunk the enrichment model writes a 2–5
    sentence summary of the surrounding `window_size` chunks; the summary is prepended to
    the chunk text (`"Context: …\n\n---\n\n<original>"`) and the untouched text is stored
    in `metadata.original_text`. The enriched text is what gets embedded.

5.  **Embedding and indexing** — `indexing/representations.py` +
    `indexing/embedders.py`. Chunks are embedded in batches of
    `indexing.embedding_batch_size` and written to LanceDB. The Arrow schema is
    `vector` (fixed-size float32 list), `text`, `chunk_id`, `document_id`, `chunk_index`,
    `metadata` (JSON string of the whole chunk).
    -   The vector width comes from the embeddings the model produced. Appending to a table
        whose stored width differs raises
        `Table '<name>' stores N-dim vectors but the current embedding model produced M-dim
        vectors` — re-index instead.
    -   Chunks whose vector contains NaN/Inf are skipped with a warning.
    -   After the append, a **native LanceDB FTS index** is created on `text` (index name
        `text_idx`, `use_tantivy=False`) unless `text_idx` or the historical `fts_text`
        already exists. There is no separate BM25 library or sidecar index.

6.  **Late chunking** (optional) — `indexing/latechunk.py`. The whole document is fed to
    the embedding model once, per-token hidden states are mean-pooled inside each chunk's
    character span, and the resulting vectors are written to a sibling table:
    `latechunk.lancedb_table_name` if set, otherwise
    `f"{table_name}{latechunk.table_suffix}"` with `table_suffix` defaulting to `_lc`.
    Documents whose vector count does not match their chunk count are skipped.

There is no step 7. Knowledge-graph extraction (`indexing/graph_extractor.py`, the
`retrieval.graph.*` keys, the NetworkX `.gml` writer) was **removed on 2026-08-09**
(roadmap item 2.5): unreachable in every shipped profile, and evidence-negative —
GraphRAG loses on single-hop retrieval, its multi-hop gains are contested, and it costs
41–57x at indexing and up to ~377x in query tokens
(`Documentation/research/academic-evidence-2026.md` §6).

## 3. Retrieval: agent and pipeline

### 3.1 Agent (`rag_system/agent/loop.py`)

```python
Agent.run(
    query,
    table_name=None, session_id=None,
    compose_sub_answers=None, query_decompose=None, ai_rerank=None,
    context_expand=None, verify=None,
    retrieval_k=None, context_window_size=None, reranker_top_k=None,
    retrieval_mode=None, force_rag=False, event_callback=None,
) -> {"answer": str, "source_documents": list[dict]}
```

`run()` is a synchronous wrapper around `_run_async()`. Every `None` argument leaves the
profile value in place; anything else is written into the live retrieval-pipeline config.

**Routing order.**

1.  If `force_rag` is true, triage is skipped and the query type is pinned to `rag_query`.
2.  Otherwise the overview router runs first: the enrichment model is shown up to 40
    document overviews and returns `{"category": "direct_answer" | "rag_query"}`. If no
    overviews are loaded it returns nothing and routing continues.
3.  If the session already has history, the query is treated as a follow-up →
    `rag_query`.
4.  Otherwise an LLM triage prompt picks `rag_query` or `direct_answer`.
    A JSON parse failure defaults to `rag_query`.

`Agent._normalize_triage()` is applied to every verdict: anything that is not an explicit
`direct_answer` becomes `rag_query`. That is what catches a small utility model still
emitting the retired `graph_query` label (the graph module was removed on 2026-08-09).

**Semantic cache.** For non-`direct_answer` queries the raw query is embedded and compared
against a `TTLCache(maxsize=100, ttl=300)` of previous results. A cosine similarity ≥
`semantic_cache_threshold` (0.98) is a hit. With `cache_scope` at its default `session`,
entries from other sessions are skipped; setting it to `global` shares answers — including
document-derived ones — across sessions.

**Query decomposition.** When enabled, `retrieval/query_transformer.py::QueryDecomposer`
splits the raw query (plus the last 5 turns for pronoun resolution) into at most
`query_decomposition.max_sub_queries` (default 10) sub-queries. What happens next depends
on `compose_from_sub_answers` (default `false`) and `pooled_first_stage` (default `true`
since arm H, 2026-08-15):

* **`compose_from_sub_answers: false` + `pooled_first_stage: true`** (shipped default) —
  the first stage runs **per sub-query**, the candidates are pooled and de-duplicated,
  then get ONE rerank pass and ONE synthesis over the union context
  (`_pooled_first_stage` in `pipelines/retrieval_pipeline.py`). This replaced N rerank
  passes, N synthesis calls and the compose step, where multi-hop facts were measurably
  lost (arm E).
* **`compose_from_sub_answers: false` + `pooled_first_stage: false`** — the first stage
  runs **once, on the full original query**, and the sub-queries are applied at the
  **rerank** stage instead: every candidate is scored against every sub-query and the
  scores combined with `query_decomposition.rerank_aggregate` (`"mean"` default, or
  `"max"`). With reranking off there is no rerank stage, so the sub-queries go unused.
* **`compose_from_sub_answers: true`** — one full `RetrievalPipeline.run()` per sub-query,
  in parallel on up to 3 worker threads, then the generation model composes one answer from
  the sub-answers. Available as an option; no longer the shipped default.
* One sub-query after decomposition takes the direct path with the resolved query.

**Verification.** When `verification.enabled` (or the per-request `verify` flag) is true
*and* the result has source documents, `agent/verifier.py` grades groundedness with the
enrichment model and appends ` [Confidence: N%]` to the answer string, plus
` [Warning: Low confidence. Groundedness: <bool>]` when the answer is not grounded or the
score is under 50. A score of 0 (parse failure) appends nothing. There is no separate
`confidence` field in the response.

Setting `verification.model` (or the `VERIFIER_MODEL` env var) to a HuggingFace model name
swaps the LLM prompt for a local NLI/verifier model scored per answer sentence against the
evidence (roadmap 2.4, off by default). `Documentation/verifier.md` has the availability
findings and the reasons `[Confidence: N%]` is UX rather than a calibrated measurement.

**History.** The agent keeps an in-process `LRUCache(maxsize=100)` of per-session
`{query, answer}` turns. This is not persisted and is lost on restart.

### 3.2 Retrieval pipeline (`rag_system/pipelines/retrieval_pipeline.py`)

`run(query, table_name=None, window_size_override=None, event_callback=None)`:

1.  **Retrieve** — `retrieval/retrievers.py::MultiVectorRetriever.retrieve(text_query,
    table_name, k, search_type)`:
    -   `hybrid` (default): the FTS leg and the vector leg each fetch `k` rows in parallel
        and are fused with **reciprocal rank fusion** (`1/(60 + rank)` per leg). There are
        no tunable leg weights.
    -   `vector_only` / `fts_only`: a single leg; the ordering is LanceDB's own.
    -   Single-word FTS queries are expanded to `word* OR word~` for prefix/fuzzy recall.
    -   Every returned document carries a finite, higher-is-better `score`: the BM25 score
        for `fts_only`, `1/(1+distance)` for `vector_only`, the RRF score for `hybrid`.
        `bm25` and `_distance` are only present when that leg actually matched.
    -   An unknown mode logs a warning and degrades to `hybrid`.
2.  **Late-chunk leg** — if late chunking is enabled, the same query also runs against the
    `_lc` table and those hits are appended to the candidate list. Then "late-chunk
    merging" runs over every candidate: each chunk's text is replaced by itself plus its
    ±1 neighbours from the main table, joined in `chunk_index` order.
2b. **Evidence-sufficiency retry** — if `retrieval.retry.enabled` (true in `default`, false
    in `fast`) and the first pass found weak evidence, the query is reformulated once on the
    enrichment model and steps 1–3 run again; the better of the two result sets is kept. The
    signal is the *contrast* between the top candidate's cosine similarity and the
    background of the rest, not the raw top similarity (which measured anti-correlated with
    success), and it is only available on L2-normalized v4+ tables. See
    `Documentation/retrieval_pipeline.md` §2b.
3.  **Rerank** — if `reranker.enabled` and a reranker loaded. `strategy: "rerankers-lib"`
    (the default) loads the model through the `rerankers` library with
    `model_type` (default `cross-encoder`); any other strategy uses the local
    `rerankers/reranker.py::CrossEncoderReranker`. Results are trimmed to
    `reranker.top_k` (or `reranker.top_percent` of the candidate count). **If the model
    cannot be loaded the pipeline prints a warning and skips reranking** rather than
    failing.
4.  **Context expansion** — when the effective window is > 0, each surviving chunk pulls
    `chunk_index ± window_size` siblings from the same `document_id` via a LanceDB metadata
    filter. Results are re-sorted by `rerank_score`, then `_distance`, then `score`. If any
    chunk carries a `rerank_score`, non-reranked chunks are dropped.
5.  **Sentence pruning** — when `provence.enabled`, `rerankers/sentence_pruner.py` runs
    `naver/provence-reranker-debertav3-v1` over each chunk at `provence.threshold`
    (default 0.1) and chunks pruned to empty are dropped. If the Provence weights cannot be
    loaded, pruning is a no-op.
6.  **Synthesis** — the generation model streams the final answer from the surviving
    chunks. `_distance` and `vector` are stripped from the returned documents and
    NaN/Inf numerics are nulled so the payload is JSON-safe.

Returned source documents have the shape
`{chunk_id, text, score, document_id, chunk_index, metadata}` plus `bm25` and/or
`rerank_score` when those stages ran. `metadata` is the chunk record that was serialised
into the LanceDB `metadata` column at index time, with `document_id` and `chunk_index`
filled in from the row.

## 4. Configuration reference

Defined in `rag_system/main.py`. `factory.get_pipeline_config(mode)` hands out a deep copy,
so per-request overrides never mutate the master dictionaries.

### 4.1 Model configuration

| Object | Keys | Env overrides |
| --- | --- | --- |
| `LLM_BACKEND` | `ollama` (default) or `watsonx` | `LLM_BACKEND` |
| `OLLAMA_CONFIG` | `host`, `generation_model`, `enrichment_model` | `OLLAMA_HOST`, `GENERATION_MODEL`, `ENRICHMENT_MODEL` |
| `WATSONX_CONFIG` | `api_key`, `project_id`, `url`, `generation_model`, `enrichment_model` | `WATSONX_API_KEY`, `WATSONX_PROJECT_ID`, `WATSONX_URL`, `WATSONX_GENERATION_MODEL`, `WATSONX_ENRICHMENT_MODEL` |
| `EXTERNAL_MODELS` | `embedding_model`, `reranker_model` | `EMBEDDING_MODEL`, `RERANKER_MODEL` |

### 4.2 Pipeline profile keys

`PIPELINE_CONFIGS` contains exactly two profiles: `default` and `fast`.

| Key | Read by | Notes |
| --- | --- | --- |
| `storage.lancedb_uri` | both pipelines | LanceDB directory (`./lancedb`). Also accepted as `storage.db_path` / `storage.lancedb_path`. |
| `storage.text_table_name` | both pipelines | Default table (`text_pages_v4`). |
| `retrieval.search_type` | `RetrievalPipeline._retrieval_mode` | `hybrid`, `vector_only`, `fts_only`. |
| `retrieval.dense.enabled` | both pipelines | `false` skips vector indexing and disables retrieval entirely — `MultiVectorRetriever` owns both the FTS and the vector leg, so no retriever is built at all. |
| `retrieval.dense.lancedb_table_name` | indexing | Fallback table name when `storage.text_table_name` is unset. |
| `retrieval.latechunk.enabled` | both pipelines | Also accepted under `retrievers.latechunk` / `retrieval.late_chunking`. |
| `retrieval.latechunk.table_suffix` / `.lancedb_table_name` | both pipelines | Late-chunk table name; suffix defaults to `_lc` on both sides. |
| `embedding_model_name` | both pipelines | Required — the retrieval pipeline raises if it is missing rather than guessing a dimension. |
| `reranker.enabled` | retrieval | |
| `reranker.strategy` | retrieval | `rerankers-lib` (default) or anything else → local `CrossEncoderReranker`. |
| `reranker.model_type` | retrieval | `rerankers` library model type, default `cross-encoder`. Use `colbert` for late-interaction models. |
| `reranker.model_name` | retrieval | Missing value logs a warning and skips reranking. |
| `reranker.top_k` / `reranker.top_percent` | retrieval | `top_percent` (0–1) wins when set. |
| `query_decomposition.enabled` | agent | |
| `query_decomposition.compose_from_sub_answers` | agent | Default `false`; with `pooled_first_stage: true` (also the default) the first stage runs per sub-query and candidates are pooled for one rerank + one synthesis. |
| `query_decomposition.max_sub_queries` | agent | Default 10; not present in the shipped profiles. |
| `query_decomposition.rerank_aggregate` | retrieval | `mean` (default) or `max`; how per-sub-query rerank scores combine. Only read when reranking runs with sub-queries. |
| `retrieval.retry.enabled` / `.min_top_score` / `.max_attempts` | retrieval | Evidence-sufficiency retry. `true` / `0.12` / `1` in `default`; disabled in `fast`. Also accepted under `retrievers.retry`. |
| `retrieval.retry.min_rerank_score` | retrieval | Threshold used instead of `min_top_score` when the reranker produced a 0–1 probability. Defaults to `min_top_score`. |
| `verification.enabled` | agent | |
| `verification.model` | agent | HuggingFace model name for the local verifier; unset ⇒ the LLM-prompt verifier. Also settable as `VERIFIER_MODEL`. |
| `verification.threshold` | agent | Default `0.5`. Local verifier only. |
| `retrieval_k` | retrieval | Rows fetched per leg; the fused list is truncated to the same value. |
| `context_window_size` | retrieval | `0` disables context expansion (the value in both shipped profiles). |
| `semantic_cache_threshold` | agent | Cosine similarity for a cache hit (0.98). |
| `cache_scope` | agent | `session` (default) or `global`. |
| `contextual_enricher.enabled` / `.window_size` | indexing | |
| `enrich_model` / `enrichment_model_name` | indexing | Per-index override of `OLLAMA_CONFIG.enrichment_model`. |
| `overview.enabled` / `.model` / `.max_chunks` | indexing | Not present in the shipped profiles; defaults are enabled, enrichment model, 5 chunks. |
| `overview_model_name` / `overview_first_n_chunks` / `overview_path` | indexing | Top-level equivalents, set per request by the API. |
| `chunker_mode` | indexing | `docling` (default) or `legacy`. |
| `chunking.chunk_size` | indexing | Token budget. Also accepted as top-level `chunk_size` or `max_tokens`. **Default when unset: 1500**; the HTTP `/index` endpoint sends 512. |
| `overlap_sentences` | indexing | Docling chunker sentence overlap, default 1. |
| `indexing.embedding_batch_size` / `.enrichment_batch_size` | indexing | |
| `provence.enabled` / `.threshold` | retrieval | Not in the profiles; set per request by the API. Threshold default 0.1. |

Keys that exist in the shipped profiles but that **nothing reads**: `description`,
`reranker.type`, and `indexing.enable_progress_tracking`.

`retrieval.graph.*` and `graph_strategy.*` no longer exist — they are ignored if present
in a hand-written config (see the removal note in §2).

There is no `dense.weight`/`denseWeight`, no `bm25_path` or other BM25 index path, no
`fallback_reranker`, no `vision_model_name`, and no `chunk_overlap` — the sparse leg is
LanceDB's native FTS, hybrid fusion is weight-free, and chunk overlap was never applied by
either chunker.

## 5. HTTP API

`rag_system/api_server.py`, default port 8001. Start it with
`python -m rag_system.api_server` or `python -m rag_system.main api --port 8001`. The
profile it loads comes from `RAG_CONFIG_MODE` (default `default`).

All responses are JSON with `Access-Control-Allow-Origin: *`. Errors are
`{"error": "<message>"}` with a 4xx/5xx status. `OPTIONS` on any path returns the CORS
preflight headers (`GET, POST, OPTIONS`); any unmatched path returns
`{"error": "Not Found"}` with 404.

**Key casing.** Every request body is normalised once at parse time: `camelCase` keys are
converted to `snake_case`, so `rerankerTopK` and `reranker_top_k` land in the same place.
An explicit `snake_case` value always wins over its camelCase twin. `overview_model` /
`overviewModel` are additionally aliased to `overview_model_name`. Unknown fields are
ignored.

### `GET /health`

```json
{"status": "ok"}
```

### `GET /models`

```json
{"generation_models": ["..."], "embedding_models": ["..."]}
```

With `LLM_BACKEND=ollama` the list comes from `GET {OLLAMA_HOST}/api/tags` (5 s timeout);
tags whose name contains `embed`, `bge` or `embedding` are classified as embedding models
and the rest as generation models. With `LLM_BACKEND=watsonx` the generation list is the
two configured granite ids. `embedding_models` always also contains the currently
configured embedding model plus `microsoft/harrier-oss-v1-0.6b` and
`Qwen/Qwen3-Embedding-4B`, `-0.6B` and `-8B`.

### `POST /chat`

| Field | Type | Default | Behaviour |
| --- | --- | --- | --- |
| `query` | string | — | **Required**; 400 if missing. |
| `session_id` | string | – | Loads that session's linked index (table name, embedding model, overviews) and its in-memory history. |
| `table_name` | string | – | Explicit LanceDB table; otherwise resolved from `session_id`, otherwise the profile default. |
| `model` | string | – | Per-request generation model, applied for this request only and restored afterwards. Ignored with a warning when the id is not valid for the active backend (watsonx ids contain `/`, Ollama tags do not). |
| `retrieval_mode` (alias `search_type`) | string | profile | `hybrid`, `vector_only`, `fts_only`. **Any other value is a 400.** |
| `force_rag` | bool | `false` | Skips triage and forces the RAG path; all other toggles still apply. |
| `query_decompose` | bool | profile | |
| `compose_sub_answers` | bool | profile | |
| `ai_rerank` | bool | profile | Toggles `reranker.enabled`. |
| `context_expand` | bool | profile | `false` forces the expansion window to 0. |
| `verify` | bool | profile | |
| `retrieval_k` | int | `20` | |
| `context_window_size` | int | `1` | Note: the profiles ship `0`, so an HTTP request expands context by default while the CLI does not. |
| `reranker_top_k` | int | `10` | |
| `provence_prune` | bool | off | Enables Provence sentence pruning. |
| `provence_threshold` | float | `0.1` | |

Response:

```json
{
  "answer": "…",
  "source_documents": [
    {"chunk_id": "…", "text": "…", "score": 0.031, "document_id": "…",
     "chunk_index": 4, "metadata": {"…": "…"}, "rerank_score": 6.1}
  ]
}
```

### `POST /chat/stream`

Same request body as `/chat`. Responds with `Content-Type: text/event-stream` and one
`data: {"type": "<event>", "data": {...}}\n\n` frame per event.

Event types: `analyze`, `direct_answer`, `decomposition`, `retrieval_started`,
`retrieval_done`, `rerank_started`, `rerank_done`, `context_expand_started`,
`context_expand_done`, `prune_started`, `prune_done`, `token`, `sub_query_token`,
`sub_query_result`, `single_query_result`, `final_answer`, `complete`, `error`.
`complete` carries the same object `/chat` would have returned and is the last frame.

### `POST /index`

| Field | Type | Default | Behaviour |
| --- | --- | --- | --- |
| `file_paths` | string[] | — | **Required** list of absolute paths; 400 otherwise. |
| `session_id` | string | – | Resolves the target table and sets `overview_path` to `index_store/overviews/<session_id>.jsonl`. |
| `table_name` | string | – | Explicit LanceDB table. |
| `embedding_model` | string | profile | Overrides `embedding_model_name` for this build. Also written into the index metadata when `session_id` is supplied, so later queries reuse the same embedder. |
| `enrich_model` | string | profile | Model used for contextual enrichment. |
| `overview_model_name` | string | profile | Model used for document overviews. |
| `enable_latechunk` | bool | `false` | **Note:** the HTTP default is off even though the `default` profile enables late chunking. |
| `enable_enrich` | bool | `true` | |
| `window_size` | int | `2` | Contextual-enrichment window. |
| `chunk_size` | int | `512` | Token budget per chunk. |
| `enable_docling_chunk` | bool | `true` | `true` pins `chunker_mode` to `docling`. Passing `false` selects the `legacy` chunker. |
| `retrieval_mode` (alias `search_type`) | string | profile | Validated against the same three values (400 otherwise) and recorded on the index config. The mode only changes behaviour at query time. |
| `batch_size_embed` | int | `50` | |
| `batch_size_enrich` | int | `25` | |

Response:

```json
{
  "message": "Indexing process for 3 file(s) completed successfully.",
  "table_name": "text_pages_<id>",
  "latechunk": false,
  "docling_chunk": true,
  "indexing_config": {
    "chunk_size": 512, "retrieval_mode": "hybrid", "window_size": 2,
    "enable_enrich": true, "embedding_model": "microsoft/harrier-oss-v1-0.6b",
    "enrich_model": null, "overview_model_name": null,
    "batch_size_embed": 50, "batch_size_enrich": 25
  }
}
```

Indexing is synchronous: the response is sent after the pipeline finishes. There is no
per-file result list and no progress endpoint.

## 6. Command line

Always run as a module from the repository root:

```bash
python -m rag_system.main index <file-or-directory> [--mode default|fast]
python -m rag_system.main chat "<query>" [--mode default|fast]
python -m rag_system.main api [--port 8001]
```

`index` accepts a single file or walks a directory for `.pdf`, `.docx`, `.html`, `.htm`,
`.md` and `.txt`. `chat` prints the JSON result of one `Agent.run()` call. `api` is
equivalent to `python -m rag_system.api_server`.

`python rag_system/main.py …` does not work (the package would not be importable).

## 7. Storage layout

| Path | Contents |
| --- | --- |
| `./lancedb/` | LanceDB tables. `text_pages_v4` is the profile default; the UI creates `text_pages_<index_id>` per index, and late chunking adds `<table>_lc`. Override with `LANCEDB_PATH` or `storage.lancedb_uri`. |
| `./index_store/overviews/*.jsonl` | Per-document overviews, one JSON object (`doc_id`, `overview`) per line. `overviews.jsonl` is the global fallback; per-session/index files are named `<id>.jsonl`. |
| `backend/chat_data.db` | SQLite: sessions, messages, indexes, documents and index metadata. Override with `DB_PATH`. |
| `./shared_uploads/` | Files uploaded through the web UI. |

## 8. Operational notes

-   **Logging** — `rag_system/__init__.py` configures the root logger; set `RAG_LOG_LEVEL`
    (`DEBUG`/`INFO`/`WARNING`/`ERROR`, default `INFO`). Much of the pipeline still prints
    progress directly to stdout.
-   **Hugging Face auth** — `HF_TOKEN` (or `HUGGINGFACE_HUB_TOKEN`) is picked up on import
    and used to log in to the Hub.
-   **Model loading** — the embedder is cached per model name in-process, and the reranker
    and Provence loads are guarded by locks so parallel sub-queries do not load the same
    weights twice.
-   **Changing the embedding model requires re-indexing.** Vector width is part of the
    LanceDB table schema and a mismatch is a hard error.
-   **Thread safety** — `rerankers` backends are not thread-safe, so `.rank()` calls are
    serialised behind a lock. The API server itself is single-threaded.

# RAG System

This directory contains the retrieval-augmented generation engine behind localGPT: the
master configuration, the indexing and retrieval pipelines, the agent loop, and the HTTP
API that the backend gateway and the frontend talk to.

For a deeper, key-by-key reference (config tables, HTTP request/response shapes, SSE
events) see [`DOCUMENTATION.md`](DOCUMENTATION.md) next to this file.

## 1. Overview

The system is **text-only**. Documents are converted to Markdown with
[Docling](https://github.com/docling-project/docling), chunked, optionally enriched with
LLM-generated context, embedded with a local Hugging Face model, and written to
**LanceDB**. Queries are answered by combining LanceDB's native full-text search with
vector search, reranking the merged candidates, and synthesising an answer with a local
Ollama model.

Core capabilities:

-   **Docling ingestion** — PDF, DOCX, HTML/HTM and MD go through Docling; TXT is read
    directly and wrapped in a fenced block. Scanned PDFs (no text layer) take an OCR
    pipeline when an OCR backend is installed.
-   **Hybrid retrieval** — LanceDB full-text (BM25-scored, native to LanceDB) and vector
    search run in parallel and are fused with reciprocal rank fusion. `vector_only` and
    `fts_only` run a single leg.
-   **Cross-encoder reranking** — the merged candidates are reordered by a reranker loaded
    through the [`rerankers`](https://github.com/AnswerDotAI/rerankers) library.
-   **Agentic query handling** — triage (documents vs. general knowledge), optional query
    decomposition with parallel sub-query retrieval, optional answer verification.
-   **Late chunking** — an optional second embedding pass that encodes the whole document
    and mean-pools per-chunk spans, stored in a sibling `<table>_lc` table.

There is **no multimodal path**. Nothing produces image embeddings and no
vision-language model is invoked; PDF understanding is Docling's layout parsing plus
OCR. If you want page-image understanding you have to build it; GLM-OCR or Qwen3-VL
would be reasonable starting points, but neither is integrated today.

## 2. Architecture

### Key modules

-   `main.py` — the MASTER configuration (`OLLAMA_CONFIG`, `WATSONX_CONFIG`,
    `EXTERNAL_MODELS`, `PIPELINE_CONFIGS`) plus a thin argparse CLI. It builds nothing
    itself; the CLI delegates to the factory.
-   `factory.py` — the single factory: `get_agent(mode)`, `get_indexing_pipeline(mode)`
    and `get_pipeline_config(mode)` (which returns a deep copy so per-request overrides
    cannot mutate the master config).
-   `api_server.py` — the HTTP API on port 8001 (`/health`, `/models`, `/chat`,
    `/chat/stream`, `/index`), built on the standard library's `http.server`.
-   `ingestion/` — everything that turns a file into chunks.
    -   `document_converter.py`: Docling converters (no-OCR, OCR, general) and OCR-engine
        selection.
    -   `docling_chunker.py`: token-budgeted chunker that walks the DoclingDocument tree,
        keeping tables atomic and recording the heading path on every chunk.
    -   `chunking.py`: `MarkdownRecursiveChunker`, the heading/paragraph recursive
        splitter used as the `legacy` chunker and as the Docling chunker's fallback.
-   `indexing/` — everything that turns chunks into an index.
    -   `representations.py`: `QwenEmbedder` (local Hugging Face), `OllamaEmbedder`,
        `EmbeddingGenerator` and the `select_embedder()` dispatcher.
    -   `embedders.py`: `LanceDBManager` and `VectorIndexer` (schema, incremental append,
        vector-width guard).
    -   `contextualizer.py`: `ContextualEnricher`, which prepends an LLM summary of the
        surrounding window to each chunk before embedding.
    -   `latechunk.py`: `LateChunkEncoder`.
    -   `overview_builder.py`: per-document overviews used by the triage router.
-   `retrieval/` — `retrievers.py` (`MultiVectorRetriever`) and
    `query_transformer.py` (`QueryDecomposer`).
-   `rerankers/` — `reranker.py` (`CrossEncoderReranker`, the non-`rerankers`-library
    fallback) and `sentence_pruner.py` (Provence sentence-level pruning).
-   `pipelines/` — `indexing_pipeline.py` and `retrieval_pipeline.py`.
-   `agent/` — `loop.py` (`Agent`: triage, decomposition, orchestration, semantic cache,
    per-session history) and `verifier.py` (`Verifier`).
-   `utils/` — `ollama_client.py`, `watsonx_client.py`, `batch_processor.py`,
    `logging_utils.py`.

### Indexing data flow

1.  `DocumentConverter.convert_to_markdown()` converts the file to Markdown. PDFs are
    probed with PyMuPDF for an existing text layer; only text-layer-less PDFs take the
    OCR converter.
2.  `DoclingChunker` (default) or `MarkdownRecursiveChunker` splits the document into
    chunks with a token budget of `chunking.chunk_size`.
3.  `OverviewBuilder` writes a one-paragraph overview per document to
    `index_store/overviews/*.jsonl` (used later for query routing).
4.  If `contextual_enricher.enabled`, `ContextualEnricher` asks the enrichment model for a
    short summary of each chunk's neighbourhood and prepends it to the chunk text. The
    untouched text is kept in `metadata.original_text`.
5.  `EmbeddingGenerator` embeds the chunks in batches and `VectorIndexer` writes them to
    the LanceDB table, then creates the native FTS index on the `text` column.
6.  If late chunking is enabled, `LateChunkEncoder` re-encodes each document as a whole and
    writes per-chunk vectors to `<table>_lc`.

There is no knowledge-graph step: `graph_extractor.py`, `GraphRetriever`,
`GraphQueryTranslator` and the `retrieval.graph` / `graph_strategy` config blocks were
removed on 2026-08-09 (roadmap item 2.5). The path was unreachable, and the evidence is
against it — GraphRAG loses on single-hop retrieval, its multi-hop gains are contested,
and it costs 41–57x at indexing and up to ~377x in query tokens
(`Documentation/research/academic-evidence-2026.md` §6).

### Retrieval data flow

1.  `Agent.run()` routes the query: the overview router first, then a
    "history exists → treat as follow-up" short circuit, then an LLM triage fallback that
    picks `rag_query` or `direct_answer`. `force_rag=True` skips triage
    entirely and pins `rag_query`.
2.  Non-`direct_answer` queries are checked against the in-process semantic cache
    (cosine similarity ≥ `semantic_cache_threshold`, scoped per session by default).
3.  If query decomposition is on, `QueryDecomposer` splits the query. With the shipped
    default (`compose_from_sub_answers: false`, `pooled_first_stage: true`) the first
    stage runs per sub-query, the candidates are pooled and de-duplicated, then get one
    rerank pass and one synthesis; with `compose_from_sub_answers: true` each sub-query
    gets its own full retrieval in parallel, up to 3 workers, and the answers are composed.
4.  `RetrievalPipeline.run()` retrieves via `MultiVectorRetriever.retrieve()` in the
    configured mode, optionally also querying the late-chunk table and merging neighbours.
4b. If `retrieval.retry.enabled` and the first pass found weak evidence — measured as the
    *contrast* between the top candidate and the background of the rest, not the raw top
    similarity — the query is reformulated once on the enrichment model and steps 4-5 run
    again, keeping whichever result set scores better (roadmap 2.1).
5.  The reranker (if enabled) reorders the candidates and keeps `reranker.top_k`. When
    sub-queries are present each candidate is scored against all of them and the scores
    combined with `query_decomposition.rerank_aggregate`.
6.  Context expansion pulls ±`context_window_size` neighbouring chunks from LanceDB.
7.  Provence pruning (off unless requested) drops irrelevant sentences from each chunk.
8.  The generation model synthesises the answer from the surviving chunks, streaming
    tokens to the caller when an event callback is supplied.
9.  If verification is enabled and there are source documents, `Verifier` grades the
    answer and appends ` [Confidence: N%]` (plus a low-confidence warning) to the answer
    string.

## 3. Models

The defaults live in `main.py` and every one of them except the Provence pruner is
overridable with an environment variable.

| Role | Default | Runtime | Env override |
| ---- | ------- | ------- | ------------ |
| **Generation** (answers, sub-answer composition) | `qwen3.5:9b` | Ollama | `GENERATION_MODEL` |
| **Enrichment / utility** (routing, triage, decomposition, contextual enrichment, overviews, verification) | `qwen3.5:4b` | Ollama | `ENRICHMENT_MODEL` |
| **Embedding** | `microsoft/harrier-oss-v1-0.6b` (MIT, 1024-dim) | `transformers`, in-process | `EMBEDDING_MODEL` |
| **Reranker** (on by default) | `Qwen/Qwen3-Reranker-4B` | own yes/no-logit scorer, loaded lazily | `RERANKER_MODEL` |
| **Sentence pruning** (optional) | `naver/provence-reranker-debertav3-v1` | `transformers`, in-process | — (hardcoded in `rerankers/sentence_pruner.py`) |

Documented alternatives:

-   Generation: `qwen3.6:27b` (high-end, ~17 GB) or `qwen3.5:4b` (light).
-   Enrichment: `qwen3.5:2b` (light).
-   Embedding: `Qwen/Qwen3-Embedding-4B` (2560-dim, 32K context; for multilingual or
    long-context corpora), `Qwen/Qwen3-Embedding-0.6B` (1024-dim).
-   Reranker: `BAAI/bge-reranker-v2-m3` (low latency; only pays off with a weaker
    embedder than the default — see [`../eval/DECISIONS.md`](../eval/DECISIONS.md)),
    `answerdotai/answerai-colbert-small-v1` (late interaction — also set
    `reranker.model_type` to `colbert`), `Qwen/Qwen3-Reranker-0.6B`.

Notes:

-   **Embedding dimensions are never hardcoded.** `VectorIndexer.index()` reads the width
    from the vectors the loaded model actually produced and builds the LanceDB schema from
    it. Appending vectors of a different width to an existing table raises an explicit
    error — **changing the embedding model requires re-indexing every existing index.**
-   **The width check is not enough on its own**, because two different models can share
    it (harrier-oss-v1-0.6b and Qwen3-Embedding-0.6B are both 1024-dim). Every table
    therefore records the embedding model that wrote it and whether its vectors are
    L2-normalized; indexing into or querying a table with a different embedder raises
    `EmbedderMismatchError` and names the model to rebuild with.
-   **Vectors are L2-normalized at write and query time**, so LanceDB's default L2
    ordering is the cosine ordering both model cards specify. Tables written before this
    existed carry no marker: they keep working with unnormalized vectors and log a
    warning recommending a rebuild.
-   `QwenEmbedder` truncates inputs at `min(tokenizer.model_max_length, 8192)` tokens.
-   `select_embedder()` treats a name containing `/` as a Hugging Face repo id and anything
    else as an Ollama tag served through `/api/embeddings`.
-   If the reranker cannot be loaded, the pipeline logs a warning and continues **without**
    reranking rather than failing the query.

## 4. Configuration

All configuration lives in `main.py`.

-   **`LLM_BACKEND`** — `ollama` (default) or `watsonx`. See
    [`../WATSONX_README.md`](../WATSONX_README.md).
-   **`OLLAMA_CONFIG`** — `host`, `generation_model`, `enrichment_model`. The generation
    model writes user-facing answers; the enrichment model does all the utility work
    (routing, triage, decomposition, contextual enrichment, document overviews,
    verification).
-   **`WATSONX_CONFIG`** — credentials, URL and the two granite model ids. It has the same
    `generation_model` / `enrichment_model` keys so it is a drop-in replacement.
-   **`EXTERNAL_MODELS`** — `embedding_model` and `reranker_model`, the two Hugging Face
    models loaded in-process.
-   **`PIPELINE_CONFIGS`** — exactly two profiles, `default` and `fast`. Selected with
    `--mode` on the CLI or the `RAG_CONFIG_MODE` environment variable for the API server.

| | `default` | `fast` |
| --- | --- | --- |
| `retrieval.search_type` | `hybrid` | `vector_only` |
| `retrieval.latechunk.enabled` | `true` | `false` |
| `reranker.enabled` | `false` (top 10 when switched on) | `false` |
| `query_decomposition.enabled` | `true` | `false` |
| `verification.enabled` | `true` | `false` |
| `contextual_enricher.enabled` | `true` (window 1) | `false` |
| `retrieval_k` | 20 | 10 |
| `indexing` batch sizes | 50 embed / 10 enrich | 100 embed / 50 enrich |

Both profiles store vectors under `./lancedb` in the table `text_pages_v4`, use
`semantic_cache_threshold: 0.98` and `cache_scope: "session"`.

A per-key table (including which keys the API can override per request) is in
[`DOCUMENTATION.md`](DOCUMENTATION.md#4-configuration-reference).

## 5. Usage

### Prerequisites

1.  **Python 3.10+** (3.11 recommended) and dependencies, installed from the repository
    root:
    ```bash
    pip install -r requirements.txt
    ```
    The root `requirements.txt` is the complete local-run set. `rag_system/requirements.txt`
    is a partial list of the heavy ML dependencies that additionally pins
    `ibm-watsonx-ai` and the macOS-only `ocrmac`. The Docker images install
    `requirements-docker.txt` instead.

2.  **Ollama models**:
    ```bash
    ollama pull qwen3.5:9b
    ollama pull qwen3.5:4b
    ```

3.  **Hugging Face models** — the embedder, reranker and (if used) Provence weights are
    downloaded on first use by `transformers`. Set `HF_TOKEN` if you point at a gated
    repository.

### Command line

Run as a module from the repository root — `python rag_system/main.py` will not work
because the package needs to be importable:

```bash
# Index one file or a whole directory (walks *.pdf, *.docx, *.html, *.htm, *.md, *.txt)
python -m rag_system.main index ./shared_uploads --mode default

# Ask a single question and print the JSON result
python -m rag_system.main chat "What was the revenue growth in Q3?"

# Start the HTTP API (equivalent to python -m rag_system.api_server)
python -m rag_system.main api --port 8001
```

`--mode` accepts `default` or `fast`. There is no interactive REPL.

### Programmatic

```python
from rag_system.factory import get_agent, get_indexing_pipeline

get_indexing_pipeline("default").run(["/abs/path/to/document.pdf"])

agent = get_agent("default")
result = agent.run("What does the contract say about termination?")
print(result["answer"])
print(len(result["source_documents"]))
```

`Agent.run()` returns `{"answer": str, "source_documents": list[dict]}` — there is no
top-level confidence field; the verifier appends its score to the answer string.

### HTTP

```bash
python -m rag_system.api_server            # port 8001
curl http://localhost:8001/health          # {"status": "ok"}
curl -X POST http://localhost:8001/chat \
  -H 'Content-Type: application/json' \
  -d '{"query": "What is in these documents?"}'
```

Endpoints, accepted fields and defaults are documented in
[`DOCUMENTATION.md`](DOCUMENTATION.md#5-http-api).

## 6. Where this fits

The RAG API is one of four processes. The frontend (`:3000`) talks to the backend gateway
(`:8000`), which forwards chat and indexing to this API (`:8001`), which in turn calls
Ollama (`:11434`). The frontend also streams directly from `:8001/chat/stream`. See the
repository [`README.md`](../README.md) and `Documentation/` for the system-level view.

# Indexing Pipeline

The active implementation is `rag_system/pipelines/indexing_pipeline.py`.
For the public API lifecycle, service ownership, storage contracts, and the
handoff into retrieval, see the canonical
[ingestion and retrieval pipeline](ingestion_and_retrieval_pipeline.md).

## End-to-end flow

1. The backend accepts multipart uploads, rejects traversal paths, enforces supported extensions and a size limit, assigns a unique stored name, and records the absolute shared-upload path.
   Multi-file requests are processed one file at a time; if a later file is rejected, earlier accepted files remain stored and registered.
2. `/indexes/{id}/build` sends those paths and build options to the internal RAG `/index` endpoint.
3. The RAG boundary verifies that every path exists beneath `LOCALGPT_UPLOAD_DIR`.
4. `DocumentConverter` handles PDF, DOCX, PPTX, XLSX, HTML/HTM, Markdown, text, CSV, JSON, and EML. PDFs use a no-OCR fast path when a text layer exists and an OCR fallback otherwise.
5. The selected chunker is explicit:
   - `enable_docling_chunk=false`: recursive Markdown chunking.
   - `enable_docling_chunk=true`: Docling structure-aware chunking.
6. `chunk_size` and `chunk_overlap` are applied as token-window settings. Explicit false values are preserved.
7. Optional contextual enrichment runs separately for each document, so adjacent documents never contaminate one another's enrichment window.
8. Each document gets an overview for routing. The overview file is rewritten on every rebuild.
9. Embeddings and normalized metadata are stored in the index's dedicated LanceDB table. `metadata` contains only the metadata object; top-level `document_id` and `chunk_index` support neighbor queries.
10. The main table is replaced on rebuild. If late chunking is enabled, its `<table>_lc` table is also replaced, then appended per document during that one build.
11. LanceDB full-text search is created over the same text rows used for vector search.
12. The API returns build status, requested/processed file counts, chunk count, late-chunk/enrichment/overview success counters, table name, and the effective configuration; SQLite stores the same effective metadata and the semantic query cache is invalidated.

## Canonical options

| Option | Default | Notes |
|---|---:|---|
| `chunk_size` | 512 | Must be positive |
| `chunk_overlap` | 64 | Must be between 0 and `chunk_size - 1` |
| `retrieval_mode` | `hybrid` | `hybrid`, `dense`, or `lexical` |
| `enable_docling_chunk` | false | Selects structural Docling chunking |
| `enable_latechunk` | false | Builds the companion late-chunk table |
| `enable_enrich` | true | Enables document-local contextual enrichment |
| `window_size` | 2 | Enrichment/context window |
| `embedding_model` | pipeline default | Persisted and enforced when linking indexes |
| `enrich_model` | pipeline default | Contextual enrichment model |
| `overview_model` | pipeline default | Routing overview model |
| `batch_size_embed` | 50 | Embedding batch size |
| `batch_size_enrich` | 25 | Enrichment batch size |

## Failure behavior

- Invalid public uploads return HTTP 413 or 422; invalid internal indexing paths return HTTP 400 before model work begins.
- Conversion and per-document optional-stage errors are counted; status is `partial` when only part of the requested conversion, enrichment, overview, or late-chunk work succeeds. If no chunks are produced, metadata records `empty` and the build request returns HTTP 422 rather than presenting an unusable index as successful.
- Invalid chunk settings are rejected before indexing; the public create path returns HTTP 422 and the internal RAG boundary returns HTTP 400.
- Rebuilds are idempotent with respect to vectors and overview rows.
- A requested Docling or late-chunk component that cannot initialize fails the build instead of silently falling back while claiming the requested feature.

Graph extraction remains an optional disabled configuration path. Page-image embedding and VLM synthesis are not part of the active indexing flow; PDF OCR and table preservation should not be described as image-vector retrieval.

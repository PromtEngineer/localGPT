# LocalGPT RAG Core

This directory contains the active document indexing, retrieval, routing, synthesis, and verification implementation.

## Implemented capabilities

- PDF, DOCX, HTML, Markdown, and text conversion through Docling, with PDF OCR fallback.
- Recursive Markdown or Docling structure-aware chunking with configurable token overlap.
- Optional document-local contextual enrichment.
- Qwen/Hugging Face or Ollama-compatible embeddings stored in LanceDB.
- LanceDB full-text, vector, and weighted hybrid retrieval.
- Retrieval across every index linked to a session.
- Optional late-chunk embeddings, ColBERT reranking, Provence pruning, query decomposition, context expansion, and answer verification.
- Document-overview routing between private-document RAG and direct model answers.
- Ollama and optional WatsonX generation clients behind a compatible interface.

## Active services

`python -m rag_system.api_server` starts the internal API. The browser does not call this service directly; `backend/server.py` proxies non-streaming and SSE chat so SQLite has one persistence owner.

The API exposes `/health`, `/models`, `/chat`, `/chat/stream`, `/index`, and internal index-artifact deletion. See `Documentation/api_reference.md` for the current request contracts.

## Multimodal status

The active pipeline preserves tables as text/Markdown and applies OCR to image-only PDFs. It does not currently create page-image embeddings or send retrieved page images to a vision-language model during answer synthesis.

`indexing/multimodal.py` is experimental scaffolding and is not invoked by the pipeline. Do not describe LocalGPT as full image-vector/VLM RAG until that path is integrated, configurable, and covered by end-to-end tests.

## Graph status

Knowledge-graph extraction and graph retrieval are optional and disabled in the default configuration. The normal product flow uses text/vector retrieval.

## Storage invariants

- One main LanceDB table per index: the SQLite `vector_table_name`.
- Optional late-chunk table: `<vector_table_name>_lc`.
- Rebuilds replace both vector tables and the index overview JSONL and invalidate the semantic query cache.
- Deletion removes both vector tables and the overview before backend metadata/source-file deletion.
- Chunk metadata stores the metadata object, with `document_id` and `chunk_index` duplicated as top-level columns for efficient neighbor lookups.
- Linked indexes must use the same embedding model.
- Semantic cache entries are isolated by session, table selection, and retrieval settings.

For the detailed active behavior, see:

- `Documentation/indexing_pipeline.md`
- `Documentation/retrieval_pipeline.md`
- `Documentation/api_reference.md`

# LocalGPT RAG System

This package contains the document processing, retrieval, and answer-generation core used by LocalGPT. It runs **in-process** inside the unified FastAPI backend on port `8000` — chat is dispatched through `rag_system/chat_runtime.py` and indexing through `rag_system/indexing_runtime.py`. (The standalone RAG API that used to listen on port `8001` has been retired.)

For canonical docs, see:
- [System overview](../Documentation/system_overview.md)
- [Indexing pipeline](../Documentation/indexing_pipeline.md)
- [Retrieval pipeline](../Documentation/retrieval_pipeline.md)
- [Persistent indexing jobs](../Documentation/persistent_indexing_jobs.md)

## Current Scope

The active pipeline is primarily text-first RAG:
- Converts uploaded documents to text/markdown
- Chunks documents with standard or DocLing-aware chunking
- Optionally builds document overviews and contextual enrichment
- Embeds chunks into LanceDB
- Retrieves with dense, BM25/FTS, or hybrid search
- Optionally reranks and verifies answers against retrieved context

Multimodal and graph features appear in parts of the codebase and roadmap, but they should be treated as experimental or future-facing unless the relevant pipeline is explicitly enabled and validated.

## Main Modules

| Path | Purpose |
|------|---------|
| `chat_runtime.py` | Transport-neutral chat dispatch (`execute_chat`) used in-process by the backend |
| `indexing_runtime.py` | Transport-neutral index build/config used in-process by the backend |
| `main.py` | Runtime model and pipeline configuration entry point; CLI for `index`/`chat`/`show_graph` |
| `pipelines/indexing_pipeline.py` | Document conversion, chunking, enrichment, embedding, storage |
| `pipelines/retrieval_pipeline.py` | Query processing, retrieval, reranking, synthesis |
| `agent/` | Routing, orchestration, verification helpers |
| `indexing/` | Embedders, enrichment, overview generation, graph helpers |
| `retrieval/` | Dense, BM25/FTS, hybrid retrieval utilities |
| `utils/` | Ollama client, batch processing, logging, maintenance support |

## Service Usage

Start the whole stack from the project root:

```bash
./start-localgpt
```

The RAG runtime is embedded in the backend; there is no separate service to launch. Check it through the unified backend:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/models
```

For ad-hoc local use, `main.py` still exposes a CLI:

```bash
python -m rag_system.main index <files...>
python -m rag_system.main chat "your question"
```

## Current Model Defaults

The current default model family is Qwen-based:
- Embedding: `Qwen/Qwen3-Embedding-0.6B`
- Generation: `qwen3:8b`
- Routing/enrichment: `qwen3:8b`

Older references to `llama3`, `qwen2.5vl`, `Qwen2-7B-instruct`, or mandatory image embeddings are legacy documentation and should not be treated as the current default system state.

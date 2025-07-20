# RAG System – Improvement Road-map

_Revision: 2025-07-05_

This document captures high-impact enhancements identified during the July 2025 code-review.  Items are grouped by theme and include a short rationale plus suggested implementation notes.  **No code has been changed – this file is planning only.**

---

## 1. Retrieval Accuracy & Speed

| ID | Item | Rationale | Notes |
|----|------|-----------|-------|
| 1.1 | Late-chunk result merging | Returned snippets can be single late-chunks → fragmented. | After retrieval, gather sibling chunks (±1) and concatenate before reranking / display. |
| 1.2 | Tiered retrieval (ANN pre-filter) | Large indexes → LanceDB full scan can be slow. | Use in-memory FAISS/HNSW to narrow to top-N, then exact LanceDB search. |
| 1.3 | Dynamic fusion weights | Different corpora favour dense vs BM25 differently. | Learn weight on small validation set; store in index `metadata`. |
| 1.4 | Query expansion via KG | Use extracted entities to enrich queries. | Requires Graph-RAG path clean-up first. |

## 2. Routing / Triage

| ID | Item | Rationale |
|----|------|-----------|
| 2.1 | Embed + cache document overviews | LLM router costs tokens; cosine-similarity pre-check is cheaper. |
| 2.2 | Session-level routing memo | Avoid repeated LLM triage for follow-up queries. |
| 2.3 | Remove legacy pattern rules | Simplifies maintenance once overview & ML routing mature. |

## 3. Indexing Pipeline

| ID | Item | Rationale |
|----|------|-----------|
| 3.1 | Parallel document conversion | PDF→MD + chunking is serial today; speed gains possible. |
| 3.2 | Incremental indexing | Re-embedding whole corpus wastes time. |
| 3.3 | Auto GPU dtype selection | Use FP16 on CUDA / MPS for memory and speed. |
| 3.4 | Post-build health check | Catch broken indexes (dim mismatch etc.) early. |

## 4. Embedding Model Management

* **Registry file** mapping tag → dims/source/license.  UI & backend validate against it.
* **Embedder pool** caches loaded HF/Ollama weights per model to save RAM.

## 5. Database & Storage

* LanceDB table GC for orphaned tables.
* Scheduled SQLite `VACUUM` when fragmentation > X %.

## 6. Observability & Ops

* JSON structured logging.
* `/metrics` endpoint for Prometheus.
* Deep health-probe (`/health/deep`) exercising end-to-end query.

## 7. Front-end UX

* SSE-driven progress bar for indexing.
* Matched-term highlighting in retrieved snippets.
* Preset buttons (Fast / Balanced / High-Recall) for retrieval settings.

## 8. Testing & CI

* Replace deleted BM25 tests with LanceDB hybrid tests.
* Integration test: build → query → assert ≥1 doc.
* GitHub Action that spins up Ollama, pulls small embedding model, runs smoke test.

## 9. Codebase Hygiene

* Graph-RAG integration (currently disabled, can be implemented if needed).
* Consolidate duplicate config keys (`embedding_model_name`, etc.).
* Run `mypy --strict`, pylint, and black in CI.

---

### 🧹 System Cleanup (Priority: **HIGH**)
Reduce complexity and improve maintainability.

* **✅ COMPLETED**: Remove experimental DSPy integration and unused modules (35+ files removed)  
* **✅ COMPLETED**: Clean up duplicate or obsolete documentation files
* **✅ COMPLETED**: Remove unused import statements and dependencies  
* **✅ COMPLETED**: Consolidate similar configuration files
* **✅ COMPLETED**: Remove broken or non-functional ReAct agent implementation

### Priority Matrix (suggested order)

1.  **Critical reliability**: 3.4, 5.1, 9.2
2.  **User-visible wins**: 1.1, 7.1, 7.2
3.  **Performance**: 1.2, 3.1, 3.3
4.  **Long-term maintainability**: 2.3, 9.1, 9.3

Feel free to rearrange based on team objectives and resource availability. 
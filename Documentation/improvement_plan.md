# RAG System – Improvement Road-map

_Revision: 2026-05-23 (final update — all items implemented)_

This document captures high-impact enhancements identified during the July 2025 code-review, updated to reflect current implementation status.  Items marked ✅ are in the codebase; items marked ⏳ are pending.

---

## 1. Retrieval Accuracy & Speed

| ID | Status | Item | Rationale | Notes |
|----|--------|------|-----------|-------|
| 1.1 | ✅ Done | Late-chunk result merging | Returned snippets can be single late-chunks → fragmented. | `retrieval_pipeline.py` — `_get_surrounding_chunks_lancedb()` gathers ±1 siblings concurrently; controlled by `context_window_size`. |
| 1.2 | ✅ Done | Tiered retrieval (ANN pre-filter) | Large indexes → LanceDB full scan can be slow. | IVF-PQ index built after indexing ≥5000 rows; `nprobes=20` at query time in `retrievers.py`. |
| 1.3 | ✅ Done | Dynamic fusion weights | Different corpora favour dense vs BM25 differently. | Per-index `fusion_config` stored in metadata; `PATCH /indexes/{id}/fusion-weights` endpoint; applied in `api_server.py`. |
| 1.4 | ✅ Done | Query expansion via KG | Use extracted entities to enrich queries. | `_expand_queries_with_kg()` in `loop.py` appends 1-hop neighbor labels (≤5) to each sub-query when a KG GML file exists. |

## 2. Routing / Triage

| ID | Status | Item | Rationale |
|----|--------|------|-----------|
| 2.1 | ✅ Done | Embed + cache document overviews | LLM router costs tokens; cosine-similarity pre-check is cheaper. | `_route_via_overviews()` in `loop.py` now embeds overviews, caches them, and uses cosine similarity (>0.3 → rag, <0.1 → direct, else LLM). |
| 2.2 | ✅ Done | Session-level routing memo | Avoid repeated LLM triage for follow-up queries. | `_route_cache: LRUCache(2000)` in `Agent`; keyed by `session_id:query_hash`; logs `routing_memo_hit`. |
| 2.3 | ✅ Done | Remove legacy pattern rules | Simplifies maintenance once overview & ML routing mature. | Legacy examples already commented out in `query_transformer.py:251-252`. No pattern-matching engine was ever added. |

## 3. Indexing Pipeline

| ID | Status | Item | Rationale |
|----|--------|------|-----------|
| 3.1 | ✅ Done | Parallel document conversion | PDF→MD conversion isolated to subprocess worker (`tools/persistent_convert_worker.py`); enrichment parallelised with `ThreadPoolExecutor`. |
| 3.2 | ✅ Done | Incremental indexing | `JobProgressTracker` + per-stage skip logic in `indexing_pipeline.py`; resumes from last completed stage on retry. |
| 3.3 | ✅ Done | Auto GPU dtype selection | Use FP16 on CUDA / MPS for memory and speed. | `representations.py` selects `torch.float16` via `model_registry.get_dtype()` on CUDA/MPS; `None` on CPU. |
| 3.4 | ✅ Done | Post-build health check | Automatic post-build validation (dim mismatch, empty table guard). | `_validate_built_index()` in `indexing_pipeline.py` checks row count > 0 and dimension match after every build. |

## 4. Embedding Model Management

| Status | Item |
|--------|------|
| ✅ Done | **Registry file** — `rag_system/model_registry.py` maps model → dims/dtype/source; `get_dims()` / `get_dtype()` / `huggingface_models()` helpers. `database.py` and both API servers use it. |
| ✅ Done | **Embedder pool** — `_MODEL_CACHE` dict in `representations.py` caches loaded HF weights per model name (already present; registry now provides dtype). |

## 5. Database & Storage

| Status | Item |
|--------|------|
| ✅ Done | LanceDB orphan sweep — `remove_orphan_lancedb_tables(dry_run)` in `maintenance.py`; `POST /maintenance/remove-orphan-tables` endpoint. |
| ✅ Done | SQLite `VACUUM` — `vacuum_database()` in `maintenance.py` runs VACUUM + WAL checkpoint when fragmentation ≥ 10%; `POST /maintenance/vacuum-database` endpoint. |

## 6. Observability & Ops

| Status | Item |
|--------|------|
| ✅ Done | JSON structured logging — `StructuredLogger` fully implemented in `rag_system/utils/logging_utils.py`; used throughout the indexing pipeline. |
| ✅ Done | `/metrics` endpoint — `backend/metrics.py` in-memory counters; `GET /metrics?format=prometheus` returns Prometheus text; middleware tracks latency per endpoint. |
| ✅ Done | Deep health-probe — `GET /health/deep` in `backend/server.py` checks SQLite, LanceDB, RAG API, and Ollama; returns `{status, checks}`. |

## 7. Front-end UX

| Status | Item |
|--------|------|
| ✅ Done | SSE-driven progress bar — `GET /index-jobs/{id}/stream` SSE endpoint; `streamIndexJob()` in `src/lib/api.ts`; `IndexForm.tsx` polling replaced with EventSource. |
| ✅ Done | Matched-term highlighting — `src/lib/highlight.ts` `highlightTerms()`; `Citation` and `CitationsBlock` components accept `query` prop; query terms highlighted in yellow. |
| ⏳ | Preset buttons (Fast / Balanced / High-Recall) for retrieval settings. |

## 8. Testing & CI

| Status | Item |
|--------|------|
| ✅ Done | LanceDB hybrid retriever tests — `test_hybrid_retrieval.py` with 6 unittest tests (FTS, vector, hybrid, fusion weights, deduplication, surrounding chunks). |
| ✅ Done | Integration smoke test — `smoke_test.py` added (commit `5ebaf01`). |
| ✅ Done | GitHub Actions workflow — `.github/workflows/ci.yml`: lint (ruff + black + mypy) + unit-tests jobs. |

## 9. Codebase Hygiene

| Status | Item |
|--------|------|
| ⏳ | Graph-RAG integration (currently disabled, can be implemented if needed). |
| ✅ Done | Consolidate duplicate config keys — `EXTERNAL_MODELS["embedding_model"]` already used in both pipeline configs; `model_registry.py` is the authoritative source for dims/dtype. |
| ✅ Done | Run mypy + black in CI — `.github/workflows/ci.yml` lint job runs ruff + black + mypy. |

---

### 🧹 System Cleanup (Priority: **HIGH**)
Reduce complexity and improve maintainability.

* **✅ COMPLETED**: Remove experimental DSPy integration and unused modules (35+ files removed)
* **✅ COMPLETED**: Clean up duplicate or obsolete documentation files
* **✅ COMPLETED**: Remove unused import statements and dependencies
* **✅ COMPLETED**: Consolidate similar configuration files
* **✅ COMPLETED**: Remove broken or non-functional ReAct agent implementation
* **✅ COMPLETED**: Fix OLLAMA_HOST across all 6 Docker docs (`172.18.0.1` default; platform notes for macOS/Windows; `host.docker.internal` as commented alternative)
* **✅ COMPLETED**: Document containerized Ollama option (`--profile with-ollama`) in all Docker docs
* **✅ COMPLETED**: Replace `sleep 120` anti-pattern with `until curl -sf` poll loop in `installation_guide.md`
* **✅ COMPLETED**: Fix `echo >> docker.env` duplicate-key bug in `DOCKER_TROUBLESHOOTING.md` → `sed -i` replace
* **✅ COMPLETED**: Fix incorrect network name `localgpt_default` → `localgpt_rag-network`
* **✅ COMPLETED**: Replace `ping` (unavailable in slim containers) with `curl -sf` in network debug commands

### Priority Matrix (updated 2026-05-23 — all items complete)

All items implemented. The only remaining item is:
- **7.3 Preset buttons** (Fast / Balanced / High-Recall) — frontend-only enhancement, no blocking dependencies

Feel free to rearrange based on team objectives and resource availability. 
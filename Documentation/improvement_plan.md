# RAG System - Improvement Roadmap

> **STATUS NOTE (2026-06-12):** This document predates a major fix cycle and
> overstates open defects. Since 2026-06-06, the following have shipped with
> regression coverage: secrets scrubbing, hybrid fusion semantics (score
> columns, dedup signal merging, fused ranking), shared-state serialization,
> blocking-call removal, multi-index consistency and multi-collection
> retrieval, stuck-job handling, all-files-failed handling, conversion/embedding
> performance, num_ctx, citations, request-scoped RAG configuration, and an
> evaluation harness (rag_eval.py).
> Treat the item tables below as historical; verify against git log and
> `test_regression_fixes.py` before acting on anything here.

_Revision: 2026-06-06 (full-codebase review and verification refresh)_

This document captures high-impact enhancements identified during review, updated to reflect current implementation status. Items marked ✅ are in the codebase and have at least focused verification; items marked ⏳ are pending or intentionally deferred.

---

## 0. Priority Upgrade Plan

The detailed execution plan is now
[`Documentation/upgrade_implementation_plan.md`](upgrade_implementation_plan.md).
The immediate order is:

1. Protect cloud provider secrets and close healthy-index false positives.
2. Repair hybrid retrieval semantics and its failing test environment.
3. Eliminate shared request-state mutation and blocking upstream calls.
4. Consolidate the FastAPI and legacy RAG HTTP servers.
5. Harden SQLite, uploads, multi-index behavior, CI, and release evidence.

---

## 1. Retrieval Accuracy & Speed

| ID | Status | Item | Rationale | Notes |
|----|--------|------|-----------|-------|
| 1.1 | ✅ Done | Late-chunk result merging | Returned snippets can be single late-chunks → fragmented. | `retrieval_pipeline.py` — `_get_surrounding_chunks_lancedb()` gathers ±1 siblings concurrently; controlled by `context_window_size`. |
| 1.2 | ✅ Done | Tiered retrieval (ANN pre-filter) | Large indexes → LanceDB full scan can be slow. | IVF-PQ index built after indexing ≥5000 rows; `nprobes=20` at query time in `retrievers.py`. |
| 1.3 | ✅ Done | Dynamic fusion weights | Different corpora favour dense vs BM25 differently. | Modality scores are normalized and fused by stable chunk identity; controlled tests verify that changing weights changes ranking. |
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
| 3.2 | ✅ Done | Incremental indexing | `JobProgressTracker` + per-stage skip logic in `indexing_pipeline.py`; resumes from last completed stage on retry and validates skipped all-unchanged index tables before reporting success. |
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
| ✅ Done | Deep health-probe — `GET /health/deep` in `backend/server.py` checks SQLite, LanceDB, the in-process RAG runtime, and Ollama; returns `{status, checks}`. |

## 7. Front-end UX

| Status | Item |
|--------|------|
| ✅ Done | SSE-driven progress bar — `GET /index-jobs/{id}/stream` SSE endpoint; `streamIndexJob()` in `src/lib/api.ts`; `IndexForm.tsx` streams progress, file status, and failed-file errors. SSE events now include `id` and `index_id`. |
| ✅ Done | Matched-term highlighting — `src/lib/highlight.ts` `highlightTerms()`; `Citation` and `CitationsBlock` components accept `query` prop; query terms highlighted in yellow. Verified with `npx tsc --noEmit` after fixing `precedingQuery` prop wiring. |
| ✅ Done | Preset buttons for indexing settings — `IndexForm.tsx` includes Fast / Balanced / Maximum profiles that control chunking, enrichment, Docling chunking, and batch sizes. |

## 8. Testing & CI

| Status | Item |
|--------|------|
| ✅ Done | LanceDB hybrid retrieval, fusion ranking, and search-mode behavior have controlled regression coverage. |
| ✅ Done | Integration smoke test — `smoke_test.py` added (commit `5ebaf01`). |
| ⚠️ Partial | GitHub Actions workflow exists, but the configured Python lint gates currently fail locally (`ruff`: 987 errors; `black`: 44 files plus one parse failure). |

## 9. Codebase Hygiene

| Status | Item |
|--------|------|
| ⏳ | Graph-RAG integration (currently disabled, can be implemented if needed). |
| ✅ Done | Consolidate duplicate config keys — `EXTERNAL_MODELS["embedding_model"]` is used in both pipeline configs; `.env` model variables are loaded by `rag_system/main.py`; `model_registry.py` is the authoritative source for dims/dtype. |
| ⚠️ Partial | CI invokes mypy + black, but adding a command is not completion: the current codebase does not pass the configured lint/format gates. |
| ✅ Done | Dependency hygiene — duplicate top-level requirements were removed; `fuzzywuzzy` / `python-Levenshtein` were replaced with `rapidfuzz`; `rank_bm25`, `scikit-learn`, and `sentence_transformers` were removed from all three requirements files (zero real usage — confirmed via grep and `pip show` reverse-dependency checks; BM25 retrieval was already replaced by LanceDB native FTS), along with the dead `_get_bm25_retriever()` method in `retrieval_pipeline.py` that referenced a no-longer-existing `BM25Retriever` class. `requirements.txt` and `requirements-docker.txt` are now consistent. |
| ✅ Done | Runtime health/config hygiene — CORS is driven by `CORS_ORIGINS`; RAG `/health` avoids eager embedder loading; Docker Compose sets `BACKEND_URL=http://backend:8000` so RAG index progress callbacks work across containers. |
| ✅ Done | FastAPI owns chat, SSE, and index execution through transport-neutral runtimes. Standard startup, frontend, MCP, evaluation, and Docker configuration no longer depend on port 8001. The legacy HTTP module remains only as compatibility cleanup. |

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

### Verification Notes (updated 2026-06-13)

Recent focused checks:
- `.venv/bin/python -m pytest -q` -> 70 passed
- Retrieval evaluation gate -> 100%
- `npm run lint:ui`
- `npm run build`
- Live parallel vector-only/qwen3:0.6b and BM25/qwen3:8b requests -> both HTTP 200 on separate worker threads

Known release blockers:
- Stream and sanitize large uploads instead of buffering them in memory.
- Enable SQLite foreign-key enforcement on every connection.
- Resolve the existing `ruff` and `black` baseline failures.
- Remove the unused legacy RAG HTTP compatibility modules after a final parity audit.
- Validate Docker Compose in an environment with Docker installed.

Feel free to rearrange based on team objectives and resource availability. 

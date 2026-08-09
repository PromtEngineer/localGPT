# localGPT — Improvement Road-map

_Revision: 2026-08-09_

Planned work only. Nothing in the **Open** sections is implemented. The **Landed** section records changes that were verified in the current working tree at the revision date — every entry there names the file that proves it, so the list can be re-checked rather than trusted.

> An evidence-based, phased extension of this plan lives in
> [research_roadmap.md](research_roadmap.md), grounded in the August 2026
> research sweeps under [research/](research/). Items graduate from there into
> this file's Landed table as they ship.

---

## 0. Landed (verified in-tree, 2026-08-08; Phase 1 rows added 2026-08-09)

| Area | Change | Verify at |
|------|--------|-----------|
| Architecture | One RAG API server; the parallel `api_server_with_progress.py` is gone | `rag_system/` has a single `api_server.py` |
| Architecture | `factory.py` is the only agent/pipeline factory; `main.py` is config + a thin `index` / `chat` / `api` CLI | `rag_system/factory.py`, `rag_system/main.py` |
| Architecture | Backend gateway is threaded and every RAG API call has a timeout (`RAG_API_TIMEOUT`, `RAG_API_INDEX_TIMEOUT`) | `backend/server.py` |
| Ops | RAG API exposes `GET /health`; `run_system.py --health`, `--stop`, `--logs-only` and `--mode prod` all work | `rag_system/api_server.py`, `run_system.py` |
| Config | Service URLs and model ids come from environment variables, documented in `.env.example` | `.env.example`, `rag_system/main.py`, `backend/server.py`, `src/lib/api.ts` |
| Retrieval | Hybrid search fuses the full-text and vector legs with reciprocal rank fusion instead of a broken weighted blend; `retrieval_mode` (`hybrid` / `vector_only` / `fts_only`) is honoured | `rag_system/retrieval/retrievers.py` |
| Retrieval | **1.1 Late-chunk result merging** — retrieved late-chunks are merged with their ±1 siblings before reranking | `rag_system/pipelines/retrieval_pipeline.py` |
| Retrieval | A reranker that fails to load logs a warning and is skipped instead of throwing | `rag_system/pipelines/retrieval_pipeline.py::_get_ai_reranker` |
| Indexing | **3.3 Auto GPU dtype selection** — CUDA > MPS > CPU with fp16 off CPU, for both the embedder and the late-chunk encoder | `rag_system/indexing/representations.py`, `rag_system/indexing/latechunk.py` |
| Indexing | Vector width is derived from the loaded model, and appending mismatched vectors to an existing table raises with a "rebuild the index" message (part of **3.4**) | `rag_system/indexing/embedders.py::VectorIndexer` |
| Indexing | OCR engine is chosen by probing which backend is installed, so a missing macOS-only engine no longer breaks every conversion | `rag_system/ingestion/document_converter.py` |
| Storage | The LanceDB path is resolved from `LANCEDB_PATH` / the pipeline config instead of three hard-coded literals, so index deletion targets the store indexing writes to | `backend/database.py::resolve_lancedb_path` |
| Privacy | The semantic cache defaults to `cache_scope: "session"`, closing the cross-session answer leak | `rag_system/main.py`, `rag_system/agent/loop.py` |
| Hygiene | DSPy modules and the non-functional ReAct agent are gone from the code; `rank_bm25`, `scikit-learn`, `nltk`, `sentence_transformers`, `colpali-engine` and `matplotlib` were dropped from `requirements.txt` | `grep` returns no code hits for any of them |
| Retrieval | **Phase 1.2 embedder** (`research_roadmap.md` §1.2) — default embedder is `microsoft/harrier-oss-v1-0.6b` (MIT, 1024-dim) with the query-side `Instruct: … Query: …` prefix on for instruction-tuned families; documents stay unprefixed. Measured mixed-corpus first-stage nDCG@10 0.915 vs 0.875 for the previous `Qwen3-Embedding-4B` default, at ~3× lower latency (0.911 for the shipped stack once vectors are normalized — see the row below). `EMBEDDING_MODEL` still overrides | `rag_system/main.py::EXTERNAL_MODELS`, `rag_system/indexing/representations.py::default_query_instruction`, `rag_system/pipelines/retrieval_pipeline.py::_query_instruction`, `eval/DECISIONS.md` |
| Retrieval | **Phase 1.1 reranker** (`research_roadmap.md` §1.1) — the `default` profile ships `reranker.enabled = False`: with this first stage the cheap cross-encoder measures net-negative (0.915 → 0.892 mixed) and the reranker that wins costs ~12.7 s/query. The toggle (UI "AI reranker" / `reranker.enabled`) now loads `Qwen/Qwen3-Reranker-4B` lazily through the in-repo `QwenRerankerScorer` | `rag_system/main.py::PIPELINE_CONFIGS`, `rag_system/pipelines/retrieval_pipeline.py::_get_ai_reranker`, `src/components/ui/session-chat.tsx`, `eval/DECISIONS.md` |
| Indexing | **Embedder-identity guard** — the width-only check cannot catch a swap between two 1024-dim models, so every table records the embedding model that wrote it (Arrow schema metadata, sidecar fallback). Indexing into or querying a table with a different embedder raises `EmbedderMismatchError` instead of returning nonsense | `rag_system/indexing/embedders.py::read_table_marker`/`assert_embedder_matches`, `rag_system/retrieval/retrievers.py::MultiVectorRetriever._check_table_identity` |
| Indexing | **Cosine normalization** — vectors are L2-normalized at write and query time, so LanceDB's default L2 ordering is the cosine ordering both model cards specify. Gated per table on the `normalized` marker, so legacy tables keep working unnormalized with a warning; the default table moved to `text_pages_v4`. Measured a wash on the gold set (mixed nDCG@10 0.915 → 0.911, recall@5 0.917 → 0.931): adopted for card-conformance, not for a number | `rag_system/indexing/embedders.py::l2_normalize`, `rag_system/retrieval/retrievers.py`, `rag_system/main.py` |
| Evaluation | **Phase 0 harness** (`research_roadmap.md` §0) — three corpora (two planted-fact PDFs + `Documentation/*.md`), a 72-query gold set labelled by answer-bearing text rather than chunk id, an in-process recall@5/10/20 + nDCG@10 runner, a binary groundedness judge validated at TPR 1.00 / TNR 1.00 on 20 hand-labelled cases, and a scripted end-to-end smoke that passes 25/25 assertions. Baseline recorded; every Phase 1/2 item now has something to A/B against | `eval/run_eval.py`, `eval/judge.py`, `eval/smoke_e2e.py`, `eval/goldset/*.jsonl`, `eval/BASELINE.md` |
| Models | **Roadmap 1.1/1.2 — evidence-gated defaults**: embedder `microsoft/harrier-oss-v1-0.6b` + query instruction prefix (mixed nDCG@10 0.915 vs 0.875 for the 8 GB Qwen3-4B); default profile reranker **off** (bge measured net-negative on this first stage; Qwen3-Reranker-4B is the lazy opt-in at +0.06 nDCG for ~12.7 s/query); per-table embedder-identity markers + L2 normalization, `text_pages_v4` | `rag_system/main.py`, `rag_system/indexing/embedders.py`, `eval/DECISIONS.md` |
| Routing | **Roadmap 2.3 — gateway routing is a deterministic gate.** The per-message enrichment-model router and the `_simple_pattern_routing` keyword/length fallback are deleted; `should_use_rag()` routes on `force_rag` → linked indexes → a whole-message smalltalk/assistant-meta allowlist → RAG. ~750 ms/message saved; agent triage is now the only LLM routing layer | `backend/server.py::should_use_rag`, `backend/test_gateway_routing.py` (155/155), `eval/decisions/phase2-gateway.md` |
| Retrieval | **Roadmap 2.5 — graph module removed**: `GraphExtractor`, `GraphRetriever`, `GraphQueryTranslator`, the `graph_query` triage outcome and the `retrieval.graph` / `graph_strategy` config keys are gone; `networkx`, `fuzzywuzzy`, `python-Levenshtein` dropped from all requirements files. Contested gains, 41–57× indexing and up to ~377× query-token cost (`research/academic-evidence-2026.md` §6) | `rag_system/indexing/` has no `graph_extractor.py`; `eval/decisions/phase2-pipeline.md` §1 |
| Retrieval | **Roadmap 2.1 — evidence-sufficiency retry**: one conditional second retrieval on weak evidence (candidate-set contrast signal — raw top similarity measured anti-correlated), on in `default`, off in `fast`. Fires on ~10% of queries, +0.008–0.017 nDCG@10, zero per-query regressions in four runs | `rag_system/pipelines/retrieval_pipeline.py::retrieve_candidates`, `eval/decisions/phase2-pipeline.md` §2 |
| Retrieval | **Roadmap 2.2 — decomposition at rerank**: the first stage always runs once on the full query; sub-queries score candidates at rerank (`query_decomposition.rerank_aggregate`), and first-stage fan-out survives only behind `compose_from_sub_answers`. Measured negative on truly-decomposing queries — no shipped profile enables rerank-decomposition | `rag_system/pipelines/retrieval_pipeline.py`, `eval/decisions/phase2-pipeline.md` §3 |
| Verification | **Roadmap 2.4 — verifier model seam**: `VERIFIER_MODEL` / `verification.model` swaps the LLM-prompt verifier for a local NLI model (MiniCheck-DeBERTa 19/20, DeBERTa-MNLI 18/20 on the judge set); default unchanged. ThinknCheck has no public weights | `rag_system/agent/verifier.py::LocalNLIVerifier`, `eval/decisions/phase2-pipeline.md` §4 |
| Eval | `run_eval.py` drives `RetrievalPipeline.retrieve_candidates()` — first stage, retry and rerank are the shipped code path; `--retry`, `--decompose`, `--aggregate` flags added; gold rows `docs_d10` + the triage-model row re-anchored after 2.5/2.3 deleted their source text (recorded per-row) | `eval/run_eval.py`, `eval/goldset/docs.jsonl` |

The previous revision of this file marked five cleanup items ✅ COMPLETED. Two of those claims (removing unused imports/dependencies, consolidating configuration files) were not true when written and are only partly true now — the surviving work is tracked in §9 below.

---

## 1. Retrieval accuracy & speed

| ID | Item | Rationale | Notes |
|----|------|-----------|-------|
| 1.2 | Tiered retrieval (ANN pre-filter) | Large tables make LanceDB scans slow. | Narrow to top-N with an in-memory index, then exact search. |
| 1.3 | Corpus-tuned fusion | RRF is weight-free and safe, but a tuned fusion could beat it per corpus. | Would need a validation set and a place to store the setting per index. |
| 1.4 | Query expansion via extracted entities | Richer queries for entity-heavy corpora. | Depends on the Graph-RAG path (§9) being finished or removed. |
| 1.5 | Deduplicate the late-chunk leg | Late-chunk hits are appended to the base hits, so the same passage can occupy two slots before reranking. | Dedupe on `chunk_id` across both legs, or rank the legs jointly. |

## 2. Routing / triage

| ID | Item | Rationale |
|----|------|-----------|
| 2.1 | Embed and cache document overviews | The agent router (now the only LLM routing layer) makes an LLM call per query; a cosine pre-check would be far cheaper. |
| 2.2 | Session-level routing memo | Today the only shortcut is "history exists → `rag_query`". Cache the decision instead. |

## 3. Indexing pipeline

| ID | Item | Rationale |
|----|------|-----------|
| 3.1 | Parallel document conversion | Conversion and chunking are serial per file. |
| 3.2 | Incremental indexing | Re-embedding the whole corpus to add one document is wasteful. |
| 3.4 | Post-build health check | The dimension guard exists; a build should also assert the FTS index, the row count and the late-chunk table when requested. |
| 3.5 | Align chunk-size defaults | No profile sets `chunking.chunk_size`, so CLI indexing chunks at 1500 tokens while `POST /index` defaults to 512 (`api_server.py:414`) — same class of split as 3.6. Documented in `system_overview.md` §5.3. |
| 3.6 | Align late-chunk defaults | `POST /index` defaults `enable_latechunk` to `false` while the `default` profile enables late-chunk retrieval, so HTTP builds and CLI builds silently differ (retrieval degrades gracefully when the `_lc` table is absent). |

## 4. Model management

* Registry mapping model tag → dimensions, source and license, validated by the UI and both servers.
* An embedder pool that keeps one copy of each loaded model in memory (a module-level cache exists in `representations.py`; the reranker and pruner have their own ad-hoc singletons).
* Warn — or refuse — when the embedding model configured for a query differs from the one recorded in the index metadata.

## 5. Database & storage

* Garbage-collect orphaned LanceDB tables (an index row deleted outside the API leaves its table behind).
* Delete files from `shared_uploads/` when their session or index is deleted.
* Scheduled SQLite `VACUUM` when fragmentation is high.

## 6. Observability & ops

* JSON structured logging and log rotation in `run_system.setup_logging` (today: plain text, no rotation).
* Move the agent's and pipelines' `print()` progress output onto the `logging` module.
* A `/metrics` endpoint for Prometheus.
* A deep health probe (`/health/deep`) that runs a real end-to-end query.
* Per-request pipeline configuration on the RAG API, so options stop leaking between requests, followed by switching the RAG API to a threading server.

## 7. Front-end UX

* SSE-driven progress for indexing (chat already streams phases; indexing is a blocking POST).
* Matched-term highlighting in retrieved snippets.
* Preset buttons (Fast / Balanced / High-Recall) over the retrieval settings.
* Surface the verifier's confidence as a field rather than parsing it out of the answer string.

## 8. Testing & CI

There is no automated test suite. The only checks are `python system_health_check.py`, `python run_system.py --health` and `./test_docker_build.sh`.

* Unit tests for `MultiVectorRetriever.retrieve` across all three modes, including the RRF ordering.
* Integration test: build an index → query → assert at least one source document.
* A GitHub Action that starts Ollama, pulls a small embedding model and runs the smoke test.
* Re-enable the type/lint gates that `next.config.ts` currently disables (`eslint.ignoreDuringBuilds`, `typescript.ignoreBuildErrors`).

## 9. Codebase hygiene

* `docker-compose.local-ollama.yml` now duplicates `docker-compose.yml` (which already defaults to host Ollama). Pick one.
* Run `mypy`, `pylint` and `black` in CI.

---

### Priority matrix (suggested order)

1. **Correctness / data loss**: 3.6
2. **User-visible wins**: 7.1, 7.2, 2.4
3. **Reliability**: 3.4, 5.1, 8.2
4. **Performance**: 1.2, 1.5, 3.1
5. **Long-term maintainability**: 2.1, 4, 9

Rearrange to suit team objectives and available time.

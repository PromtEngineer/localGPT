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
| Observability | **Roadmap 4.5 — per-query token accounting** (on by default): Ollama `prompt_eval_count`/`eval_count` aggregated per user query, bucketed by stage, returned as `token_usage` on `/chat` and the SSE `complete` event. ContextVar-based; watsonx reports zeros; embedding calls uncounted | `rag_system/utils/ollama_client.py::TokenUsageTracker`, `eval/decisions/phase4-escalation-tokens.md` |
| Retrieval | **Roadmap 4.4 — metadata filter DSL** (no flag; inert without a `filters` argument, md5-verified byte-identical when unused): JSON filters compiled to LanceDB where-clauses prefiltering both search legs; quoting characters refused, malformed filters are the RAG API's 400; gateway forwards `filters` and treats a present filter as force-RAG. Page/date filtering NOT shipped (needs real columns + re-index → §3.7) | `rag_system/retrieval/filters.py`, `rag_system/api_server.py`, `backend/server.py`, `eval/decisions/phase4-filters-askfolder.md` |
| CLI | **Roadmap 4.6 — ask-a-folder**: `python -m rag_system.main ask <folder> "<q>"` builds an ephemeral index (fast profile), answers with the standard pipeline, cleans up everything including on SIGTERM | `rag_system/ask_folder.py`, `eval/decisions/phase4-filters-askfolder.md` |
| Indexing | **Roadmap 4.2 (index half) — cross-reference extraction** (on by default): regex-only extraction of exhibit/section/document-name references into `metadata.crossrefs`, filename resolution incl. numeric-prefix-stripped aliases; verified bit-identical text/vector columns. The **query-time hop is REJECTED as a default** (flag kept, off): 0/11 fires at the shipped k=20, 0/11 target precision where forced, beaten by raising `k` at equal budget | `rag_system/indexing/crossref.py`, `eval/decisions/phase4-retrieval-benchmarks.md`, `eval/decisions/phase4-answer-quality.md` |
| Retrieval | **Roadmap 4.1 escalation — REJECTED as default (2026-08-12 re-run); 4.3 overview prefilter — HELD off**: the HOLD's re-run condition was executed after the num_ctx fix and the lift did not survive — the escalation-off baseline on the identical fire subset went 0/9→7/9 (the 2026-08-09 "lift" was front-truncation favoring the tail-appended document) and both product-default fires were regressions (`eval/decisions/phase4-escalation-rerun.md`); prefilter `boost` wins only on heterogeneous corpora, `restrict` is rejected outright (kills recall@20 on 4 queries/corpus). Both stay flag-gated off | `rag_system/agent/escalation.py`, `rag_system/pipelines/retrieval_pipeline.py`, `eval/decisions/phase4-*.md`, `design_rationale.md` §13a |

The previous revision of this file marked five cleanup items ✅ COMPLETED. Two of those claims (removing unused imports/dependencies, consolidating configuration files) were not true when written and are only partly true now — the surviving work is tracked in §9 below.

---

> **Chunker data loss — FIXED 2026-08-12, REBUILD INDEXES.** `MarkdownRecursiveChunker._split_text`'s separator-reassembly loop dropped the first and every other body segment of any document large enough to split, so long documents entered the index at ~50% of their content (RFC 9000: 49% retained; `design_rationale.md`: 49%). Found by the unseen-corpus RFC shakedown — the authored eval corpora were too small to trigger the path. Fixed and property-tested lossless (30 randomized structured docs, zero non-whitespace loss; post-fix retention 94–99.8%, remainder is whitespace normalization). **Every index built before this fix under-contains its long documents** — product indexes in `index_store/` and the tracked eval indexes should be rebuilt, and pre-fix retrieval baselines on the `docs` corpus are not comparable to post-fix numbers.

## 1. Retrieval accuracy & speed

| ID | Item | Rationale | Notes |
|----|------|-----------|-------|
| 1.2 | Tiered retrieval (ANN pre-filter) | Large tables make LanceDB scans slow. | Narrow to top-N with an in-memory index, then exact search. |
| 1.3 | Corpus-tuned fusion | RRF is weight-free and safe, but a tuned fusion could beat it per corpus. | Would need a validation set and a place to store the setting per index. |
| 1.4 | Query expansion via extracted entities | Richer queries for entity-heavy corpora. | Depends on the Graph-RAG path (§9) being finished or removed. |
| 1.5 | ~~Deduplicate the late-chunk leg~~ **FIXED 2026-08-14** | Late-chunk hits were appended to the base hits with no dedupe, so the same passage occupied two slots — and with the ±1 sibling merge tripling every entry, synthesis context reached ~94k tokens/call while Ollama's slot-split window served only ~16k and silently front-truncated (the model saw the *tail* of the ranking). | `retrieval_pipeline.py` now dedupes across both legs on `(document_id, chunk_index)` and packs rank-ordered docs into an explicit synthesis token budget (default 12k, `synthesis_context_tokens` config) with sibling-span overlap suppression. Measured (arm F, `eval/decisions/synthesis-grounding-ab-2026-08-13.md`): context 335k → ~40k chars, truncation warnings 32 → 0, E2E wall time −32%, judged pass **7/24 → 16/24**. |
| 1.6 | Query-aware crossref-hop targets | The 4.2 hop was rejected as a default partly because target selection is query-blind: `max_hops=1` takes the first unrepresented reference in scan order, which lands on hub documents (21/24 hops → 2 docs, 0/11 expected-source precision). | Score candidate targets against the query (overview embeddings exist) before spending the hop; then re-run `eval/decisions/phase4-retrieval-benchmarks.md`'s A/B. |
| 1.8 | Crossref extraction for real-world naming conventions | The unseen-corpus RFC shakedown measured the extractor at **0/1403 resolved** on 23 real RFCs: filenames never occur in the prose, 355 explicit `[QUIC-TRANSPORT] Section N` cross-document references are discarded by `_SECTION_RE`, and bare `RFC NNNN` mentions have no pattern (`eval/decisions/rfc-shakedown-2026-08-13.md`). The extractor's patterns encode the authored acq corpus's conventions. | Add bracketed-citation and `RFC NNNN`-style patterns + title-based (not just filename-based) resolution; re-measure on the rfc corpus (a rename experiment showed 91 resolutions are reachable). Index-inert either way; the hop stays off. |
| 1.9 | Synthesis grounding on dense unseen technical text | With retrieval fixed (recall@20 0.958 on the rfc corpus), end-to-end answer quality is the measured bottleneck: **5/24** judged pass (Sonnet panel; single-doc 1/14, crossref 4/10). The 9b answers from its prior instead of the supplied snippets and fabricates citations (e.g. inventing a quote from "RFC 9002 §13.4"). The verifier flags these low-confidence but the wrong claim still leads the answer. | A/B run 2026-08-13 (`eval/decisions/synthesis-grounding-ab-2026-08-13.md`): strict snippet-only prompt + temperature-0 decode ADOPTED as default (7/24 vs 5/24 — at the noise floor, adopted for escape-hatch removal + determinism, not claimed as a quality win); **qwen3.6:35b-a3b REJECTED** (equal-or-worse at 4-5/24 under both prompts — grounding is not parameter count at this scale). Strict compose prompt TESTED AND REJECTED same day (arm E: 4/24 vs C's 7/24; all three pass→fail flips were composed rows — strict copy-rules against already-synthesized sub-answer prose drop facts the sub-answers carried; reverted, decision file appendix). **2026-08-14 (arm F): the dominant cause was context overflow, not the model** — synthesis prompts were ~94k tokens against a slot-split ~16k served window, so front-truncation deleted the top-ranked evidence on every call. Cross-leg dedupe + 12k context budget (item 1.5) lifted judged pass to **16/24 (single-doc 10/14, crossref 6/10)** with a near-unanimous panel (1 split/72 votes) and −32% wall time. Item narrows to the crossref/multi-hop residue: next levers are deterministic decomposition (temp 0 — also stabilizes A/B row sets); abstain-on-low-verifier-confidence; passage-level citation forcing. |
| 1.7 | ~~Sanitize FTS sub-query input~~ **FIXED 2026-08-12** | A decomposer-emitted sub-query containing double quotes made the LanceDB FTS parser raise ("position is not found but required for phrase queries"), killing hybrid retrieval for that query outright (1/24 gold queries during the Phase-4 A/B). | `retrievers.py` now strips double quotes before the FTS leg, and a hybrid search whose FTS leg fails for any other reason degrades to dense-only with a warning instead of returning nothing (`fts_only` mode still propagates). Verified against the exact incident query: 0 docs → 5 docs, answer document ranked first; degradation tested with a simulated FTS failure. |

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
| 3.7 | Real `page`/`date` columns | Roadmap 4.4 promised page/date filters; they did not ship because both live inside the metadata JSON string column (`metadata LIKE` would false-match `"page": 31` for page 3). Needs first-class columns + a re-index window. |

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
* **Context-window budgeting — FIXED 2026-08-12 (found by the Phase-4 A/B, 2026-08-09):** `ollama_client` never set `num_ctx`, and the served ceiling on this host was **8194 prompt tokens** with silent *front*-truncation — a 373-chunk corpus's synthesis prompt (20–25k tokens) lost its top-ranked evidence without any error. Now both clients (`rag_system/utils/ollama_client.py`, `backend/ollama_client.py`) size `options.num_ctx` per request from the prompt length, bucketed 8k/16k/32k and per-model **monotonic** — the window only ratchets upward, because changing `num_ctx` between calls forces a KV-cache reallocation (measured on the smoke suite: naive per-call bucketing 924s; fixed 32k window 259s; the shipped monotonic ratchet 208s; pre-fix baseline 363s). `OLLAMA_NUM_CTX` pins a value, `OLLAMA_NUM_CTX_MAX` caps the bucket (default 32768), and a filled window (`prompt_eval_count ≥ num_ctx − 16`) logs a loud truncation warning. Verified 2026-08-12: the 100k-char probe that returned `prompt_eval_count: 8194` now evaluates the full prompt (17,351 tokens) and recovers a fact planted at position 0. The 4.1 escalation A/B re-run this unblocked was executed 2026-08-12 (`eval/decisions/phase4-escalation-rerun.md`; verdict: reject). Note the fix also **voids** the 2026-08-09 `acqdocs` baseline in `phase4-answer-quality.md` §4 — that 0/9 was a truncated-context artifact and reads 7/9 on identical inputs post-fix. Still open: prompts >~90k chars hit the 32k default cap — client-side facts-string budgeting remains worthwhile. Related, **FIXED 2026-08-12**: the synthesis stream paths never set `think:false`, so the 9b could burn its window on chain-of-thought and return an empty answer (measured: prompt 9,351 + thinking 7,033 = window exactly, empty `response`). All three streaming synthesis call sites (`retrieval_pipeline._synthesize_final_answer` and both agent-loop compose paths) now pass `enable_thinking=False`; verified at the wire that `think: false` reaches the Ollama payload. The eval harness's `patch_no_think` monkeypatch is now redundant for these paths but harmless.
* Render `token_usage` in the UI (it reaches the browser on the SSE `complete` event and is persisted in the turn's steps snapshot; `conversation-page.tsx` doesn't display it yet), and forward it on the gateway's non-streaming path (`backend/server.py::_query_rag_api` extracts only `answer` + `source_documents`).
* Judge noise is larger than the Phase-0 validation suggested: 5–7 verdict flips per 24 rows between identical runs were observed in the Phase-4 A/Bs. Direction-deciding cells need k≥5 judge votes (`eval/decisions/phase4-answer-quality.md`). The smoke test shares the underlying cause — its substring assertions run against temperature-1.0 generations, and 1 of 6 runs on 2026-08-09 flaked on one question (a rerun passed 25/25). Consider `temperature 0` for smoke/eval generation calls. Worse, the 2026-08-12 re-run caught the judge returning verdicts its own reasons contradict (an answer containing the gold fact verbatim voted 0/5), and being perturbed by the verifier's `[Confidence: N%] [Warning: …]` suffix, which one judge reason cited as grounds for rejection. Two mitigations landed 2026-08-12: `eval/judge.py` now strips the verifier suffix from both judge slots before scoring, and the judge gained an Anthropic-API backend (eval-only; `JUDGE_MODEL=claude-sonnet-5` routes any `claude-*` model name through the `anthropic` SDK with a server-enforced JSON schema — the product stays fully local). The 18 hand-adjudicated rows from the re-run are preserved as a permanent hard-case benchmark (`eval/judge_hard_cases.jsonl` + `eval/validate_judge_hard.py`; the 4b's k=5 majority scores 13/18 on it — any candidate judge must beat that). **A Sonnet-class judge is now validated (2026-08-13)**: three independent Claude Sonnet subagent voters, each running the exact v1 prompt (suffix-stripped) over all 38 cases, scored **20/20 on the Phase-0 validation set and 18/18 on the hard set, with zero split votes across 114 judgments** — including the fact-present-but-prefaced-with-a-denial row both 4b arms failed (`eval/decisions/judge-sonnet-validation-2026-08-13.json`). Protocol for direction-deciding cells: generate the per-case v1 prompts (as `eval/judge.py` builds them) and fan them out to 3 Sonnet subagent voters, majority decides; the 4b remains the free bulk-pass judge. The `JUDGE_MODEL=claude-*` API backend in `eval/judge.py` runs the identical prompt and is available when API credentials exist.
* **Whole-document context can send the 9b model into verbatim transcription**: with escalation forced on and an untruncated 32k window, one query produced a 35,290-char answer (vs 1,400 baseline) that answered correctly in its first paragraph and then regurgitated `design_rationale.md` text wholesale (`eval/decisions/phase4-escalation-rerun.md` §8.1). Any future feature that injects large verbatim blocks needs an output cap or a "do not transcribe the context" instruction in the synthesis prompt.

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

# Evidence-Based Roadmap

_Created 2026-08-08. Status of every item here is **PLANNED, not implemented**,
except where a row says otherwise — when an item ships, move its row to
`improvement_plan.md`'s completed table and update the component docs. Do not
describe any of this as current behavior._

**Phase 1 is complete (2026-08-09).** Items 1.1, 1.2 and 1.3 were measured and
decided; the adopted defaults, the joint matrix behind them and what stays
opt-in are in [`../eval/DECISIONS.md`](../eval/DECISIONS.md).

This roadmap turns the findings in [`research/`](research/) into staged,
testable changes. Ordering principle, straight from the evidence: **retriever
and reranker quality dominate everything else** (+14.2 pts from an embedder
swap on BrowseComp-Plus, ACL 2026; +17.2 pp MRR@3 from a cross-encoder), every
clever layer on top is conditional, and no upgrade counts until it is measured
on our own eval set (public-leaderboard deltas under ~2 points do not transfer;
ranking inversions on production corpora are documented).

---

## Phase 0 — Evaluation harness (prerequisite; nothing else lands without it)

The single most consistent practitioner finding: generic metrics create false
confidence; teams need a small, owned eval set and binary judgments
(Hamel/Husain–Shankar FAQ; Anthropic "20–50 tasks from real failures is a
great start"; Chroma's generative-benchmarking ranking inversions).

| # | Item | Detail | Acceptance |
|---|------|--------|------------|
| 0.1 | Gold retrieval set | Reverse-generate ~50–100 queries from indexed chunks across 2–3 real corpora + the planted-fact test PDF, using structured dimension tuples (topic × question-type × difficulty), not freeform "give me questions". Store under `eval/goldset/` with the chunk IDs that must be retrieved. | Set committed; every query hand-checked once |
| 0.2 | Retrieval metrics runner | `eval/run_eval.py` hits the RAG API: **recall@k** at first stage, **nDCG@10 after rerank** — the two metrics the evidence says matter. Runs in minutes, no LLM judge needed. | Baseline numbers for the current stack recorded in `eval/BASELINE.md` |
| 0.3 | Groundedness judge | Binary pass/fail LLM judge for answer faithfulness, validated against ~20 hand-labeled answers (report TPR/TNR, not raw agreement). Binary, not Likert, per the near-unanimous practitioner canon. | Judge agrees with hand labels ≥90% before it gates anything |
| 0.4 | End-to-end smoke | Scripted: index the test PDF → 5 planted-fact questions → assert grounding + citation presence + persistence round-trip. Extends the E2E script already exercised on 2026-08-08. | Runs green on the current tree |

**Effort:** ~1 day. **Risk:** low. **Everything below cites Phase 0 numbers in
its acceptance criteria.**

## Phase 1 — Component upgrades (A/B against Phase 0, adopt only on wins)

| # | Item | Evidence | Plan | Risk |
|---|------|----------|------|------|
| 1.1 ✅ **DONE 2026-08-09** — reranking now ships **off**, with `Qwen/Qwen3-Reranker-4B` as the model the toggle loads. Decision + numbers: [`../eval/DECISIONS.md`](../eval/DECISIONS.md), evidence: [`../eval/decisions/reranker.md`](../eval/decisions/reranker.md) | **Reranker A/B: Qwen3-Reranker-4B (and 0.6B) vs bge-reranker-v2-m3** | bge-v2 lineage static for 2 years; near-zero on instruction-following (FollowIR −0.01) and weak on code (41.38 MTEB-Code); Qwen3-Reranker-4B leads permissive options (69.76 MTEB-R, Apache 2.0). Reranker is the highest-ROI slot in the stack. | Config-swap via `RERANKER_MODEL`; verify the `rerankers` library loads Qwen3's causal yes/no-logit scoring (it is NOT a SequenceClassification model — may need a small custom scorer). Measure nDCG@10 delta and per-query latency on MPS. | Integration: medium. Throughput regression is expected; accept only if quality delta justifies it, else keep 0.6B or bge |
| 1.2 ✅ **DONE 2026-08-09** — `microsoft/harrier-oss-v1-0.6b` adopted as the default with the query prefix on, in the same re-index window as 1.1 and the two index-format fixes. Decision + numbers: [`../eval/DECISIONS.md`](../eval/DECISIONS.md), evidence: [`../eval/decisions/embedder.md`](../eval/decisions/embedder.md) | **Embedder audit + A/B: instruction prefix, then harrier-oss-v1 vs Qwen3-Embedding** | All top-2026 embedders are instruction-tuned (+1–5% from the query-side prefix alone); microsoft/harrier-oss-v1 (MIT, Mar 2026) beats Qwen3-Embedding at equal size (0.6B: 69.0 vs 64.3 MMTEB). | First verify our query path sends the `Instruct: … Query: …` prefix (free win if missing). Then A/B harrier-0.6b and Qwen3-4B on the gold set. **Changing the default forces re-indexing** — decide once, alongside 1.1, in a single index-format window. | Low code risk; migration cost is the re-index. harrier is decoder/last-token-pooling like Qwen3, so `embedders.py` should need at most a config entry |
| 1.3 | **GLM-OCR behind Docling for scanned/complex PDFs** — _spike complete 2026-08-09: **GO-LATER**, see `eval/decisions/glm-ocr-spike.md`_ | 0.9B MIT specialist VLM; **95.22 OmniDocBench v1.6_full (third, behind PaddleOCR-VL-1.6 96.34 and MinerU2.5-Pro 95.75)** — an earlier "#1 / beats GPT-5.2 by ~10 pts" framing from the research sweep did not survive source verification and is retracted. Spike proved the serving path on Apple Silicon: official `ollama pull glm-ocr` (2.2GB), and our pinned docling 2.118.1 already ships a `glm_ocr` preset with an Ollama override. Parse quality on a degraded scanned invoice: 30/30 table cells vs the current chain losing every price. | Blocked on three items before adoption: (a) Ollama's Modelfile ignores prompts, so GLM-OCR's table/formula modes are unreachable there; (b) deterministic double-transcription on some pages; (c) docling flattens its pipe tables (`tables: 0`). Prerequisite: add a scanned/tabular corpus with ground truth to `eval/` — nothing in Phase 0 exercises the OCR branch — and A/B GLM-OCR against docling's other 2026 presets (`lightonocr`, `dots_ocr`, `nanonets_ocr2`…), not as a foregone winner. | Cheaper wins shipped first: the RapidOCR probe bug is fixed (stale module name sent every scan to Tesseract CLI), and `pip install ocrmac` gives macOS the best classic chain |

**Effort:** ~2–3 days including eval runs. **Decision gate:** adopt each only
on a measured win; record adopted/rejected + numbers in `eval/DECISIONS.md`.
**Gate cleared 2026-08-09** — see [`../eval/DECISIONS.md`](../eval/DECISIONS.md).
1.1 and 1.2 could not be decided separately (the reranker's value depends on the
first stage), so they were re-measured jointly and shipped in one window
together with the embedder-identity guard and cosine normalization that the 1.2
audit flagged. 1.3 remains GO-LATER: no code was written.

## Phase 2 — Pipeline-shape fixes — _COMPLETE 2026-08-09; measurements in `eval/decisions/phase2-pipeline.md` and `phase2-gateway.md`; Landed rows graduated to improvement_plan.md. Notable: 2.2 measured negative on truly-decomposing queries and ships disabled; 2.1's suggested signal was replaced after measuring anti-correlation._

| # | Item | Evidence | Plan |
|---|------|----------|------|
| 2.1 | **Evidence-sufficiency retry** | One conditional second retrieval iteration captures ~95% of deep-loop gains (+7.1 EM for iteration 1→2 on a local 7B; iterations ≥3 are noise). Terminate on evidence sufficiency, not query count. | If top-k rerank scores fall below a threshold, reformulate once (enrichment model) and retrieve again; hard cap 2 iterations; off in `fast` profile. Surface as a step in the SSE cascade so the UI shows it |
| 2.2 | **Decomposition at rerank, not first-stage** | Decomposition at initial retrieval "frequently harms via semantic dilution"; it helps applied at reranking (2026 finding, MultiConIR/SSRB). | Keep the full query for hybrid retrieval; score candidates against sub-queries during rerank. Touches `retrieval_pipeline.py` only |
| 2.3 | **Cheapen gateway routing** | Pre-retrieval LLM routing is the weakest pattern in the 2026 evidence (four ML approaches failed; TF-IDF+SVM ≥ LLM routers at ~zero cost; fixed-hybrid beat adaptive routing). | Replace the gateway's LLM routing call with a cheap gate (heuristics + optional logit-margin TARG-style check), keep `force_rag`, keep agent-side triage as the single LLM routing layer. Net: one less LLM call per message, simpler failure surface |
| 2.4 | **Dedicated verifier model (optional)** | Verification helps only as an external check; a 4-bit 1B verifier (ThinknCheck, CC0) now beats the 7B 2024 SOTA; Granite Guardian (Apache) is the permissive alternative. Current LLM-prompt verifier works but is slower and uncalibrated. | Add `VERIFIER_MODEL` option: local NLI/verifier model scoring answer-vs-evidence per sentence; keep the LLM prompt as fallback. Present `[Confidence: N%]` as UX, never as a calibrated measurement (document this) |
| 2.5 | **Delete the graph module** | GraphRAG loses single-hop, contested multi-hop gains, 41–57× indexing / up to ~377× query-token cost; unreachable in this repo already (no profile sets `graph_strategy`). | Remove `graph_extractor.py`, `GraphRetriever`, `GraphQueryTranslator` + config remnants; note the decision and evidence in `design_rationale.md` (resolves improvement_plan §9) |

**Effort:** ~2 days. Each item is independently shippable and independently
revertible; each lands with a Phase-0 eval delta.

## Phase 3 — Documentation: make the evidence part of the repo's argument

**3.1 is complete (2026-08-09):** [`design_rationale.md`](design_rationale.md)
ships one section per component with its evidence and its eval number, plus the
"deliberately not implemented" list. 3.2's README/roadmap/docs-index cross-links
landed with it.

| # | Item | Plan |
|---|------|------|
| 3.1 ✅ **DONE 2026-08-09** → [`design_rationale.md`](design_rationale.md) | One section per component: what localGPT does, and the evidence for why (with citations into `research/`). Includes a **"deliberately not implemented"** list — HyDE-by-default, multi-query expansion, weighted fusion knobs, GraphRAG, vendor memory systems, deep subagent fan-out — each with its citation, so future contributors don't re-add deprecated patterns |
| 3.2 | Cross-link | README gets one line pointing at the rationale; `improvement_plan.md` links each open item to its evidence; this roadmap's rows move to improvement_plan as they complete |
| 3.3 | Honesty rule (already in force) | Every phase lands code + docs + eval delta in the same change. Nothing in `design_rationale.md` may describe unshipped behavior — that is what this roadmap file is for |

**Effort:** ~half a day, mostly distillation from the synthesis already written.

## Explicitly out of scope (evidence-negative — revisit only with new evidence)

- **Deep agent loops / parallel subagent fan-out** — pays on open-web breadth-first research at ~15× token budgets; fails silently elsewhere (41.8% of failures at the hand-off).
- **RL-trained searchers** — no out-of-distribution transfer (Search-R1 = its base model, ACL 2026); also mis-calibrates confidence, which would break 2.3's logit gate.
- **Always-on HyDE / multi-query** — measured negative on entity/numeric corpora; multi-query scored below plain BM25.
- **Vendor memory layers** — the null baseline (RAG over the transcript) wins; our session store already is the null baseline.
- **Token-level context compression (LLMLingua-style)** — ≤18% best-case speedup, net-negative outside a narrow window; we already ship the surviving alternative (Provence pruning).

## Sequencing summary

```
Phase 0 (eval harness) ──► Phase 1 (reranker / embedder / parser A/Bs) ──► Phase 2 (pipeline shape) ──► Phase 3 (rationale docs)
     ~1 day                      ~2–3 days                                     ~2 days                      ~0.5 day
     ✅ done                     ✅ done 2026-08-09 (1.3 = GO-LATER)
```

Phase 0 blocks everything. Phases 1 and 2 can interleave per-item, but 1.1/1.2
should conclude before any re-index-requiring release. Phase 3.1 can be drafted
in parallel at any time.

## Phase 4 — Ideas adopted from agentic-file-search (planned 2026-08-09, not implemented)

Source: [PromtEngineer/agentic-file-search](https://github.com/PromtEngineer/agentic-file-search)
(FsExplorer lineage — an agentic filesystem QA agent: three-phase scan/dive/backtrack,
Docling parsing, grep/glob tools, DuckDB+VSS indexed search with a metadata-filter DSL,
exploration traces, per-query token/cost tracking). The evidence lens for every item:
escalate-don't-pre-decide (PEA-CAE), retriever quality dominates agency
(BrowseComp-Plus), and filesystem agents win small corpora but lose to ranked
retrieval at scale with ~39x token cost (BM25-wins-at-scale) — so we adopt its
*escalation mechanisms*, not its loop.

| # | Item | From their design | Our incorporation | Evidence fit |
|---|------|-------------------|-------------------|--------------|
| 4.1 | **Full-document escalation** | `parse_file` / `get_document` deep-read tools | When the evidence-sufficiency retry (2.1) still lands weak, reassemble the top-cited document IN ORDER from its chunks (chunk_index exists in metadata) and hand it to synthesis, token-capped, one document max. New RAG API helper + agent step; SSE event `document_escalation`. | PEA-CAE escalation; DOS-RAG document-order finding; bounded, not a loop |
| 4.2 | **Cross-reference hop** | Phase-3 backtracking on "See Exhibit B" | Index-time regex extraction of intra-corpus references (exhibit/section/filename mentions) into chunk metadata; query-time one-hop pull of the referenced doc's overview + top chunks when a top-ranked chunk carries one. Capped at one hop, no LLM in the hop. | Fixes the real "cross-references are invisible to embeddings" gap without unbounded agency |
| 4.3 | **Overview prefilter ("peripheral vision")** | Phase-1 parallel scan + RELEVANT/MAYBE/SKIP triage | We already build per-doc overviews; embed them once and use overview-vs-query similarity to boost/restrict chunk retrieval to top documents on multi-document indexes. No per-query LLM cost (their scan phase is an LLM call per document — the measured 39x failure mode). | Jason Liu facets/peripheral-vision; avoids their linear-cost scan |
| 4.4 | **Metadata filter DSL / self-query** | `semantic_search(filters="field=value, field in (a,b)")` on DuckDB | Same surface on LanceDB `where` clauses: accept `filters` on /chat + /chat/stream, wire to chunk metadata (document name, page, date). LLM filter-extraction later — small local models measured ~0.999 F1 on easy/medium filter translation. | The one query-planning technique local models already nail (component-map §6.4) |
| 4.5 | **Per-query token/cost tracking** | TokenTracker + cost summary per query | Surface Ollama's prompt_eval_count/eval_count per stage in the SSE `complete` event and UI (local cost = time + watts, still worth showing). | Their nicest UX idea; zero risk |
| 4.6 | **Ephemeral "ask a folder" mode** | Index-free filesystem QA | `python -m rag_system.main ask <folder> "<q>"`: build a temp in-memory/throwaway index (fast profile, no enrichment), answer, delete. Same pipeline, no agent loop. | FS-agents win small corpora — but an ephemeral *index* beats an ephemeral *agent* on our own retriever-dominance evidence |

**Deliberately NOT adopted** (goes in design_rationale §13 when implemented): the
LLM-per-document scan phase (linear cost in corpus size — the exact pattern
BM25-wins-at-scale measured at ~39x tokens), the free-form ReAct exploration loop
(search volume correlates weakly with quality; silent hand-off failures), cloud
Gemini (violates the local/privacy premise), llama-index-workflows + DuckDB
(duplicate our framework-free stack and LanceDB).

Sequencing: 4.5 and 4.4 are independent quick wins; 4.1 depends on 2.1's signal
(shipped); 4.2/4.3 are index-format-adjacent and should share a re-index window;
4.6 is CLI-only. All gated on the Phase 0 harness like everything else — 4.2 and
4.3 need multi-document gold queries with cross-references added to eval/goldset.

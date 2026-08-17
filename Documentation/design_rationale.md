# Design Rationale

_Revision: 2026-08-09. Roadmap item [3.1](research_roadmap.md#phase-3--documentation-make-the-evidence-part-of-the-repos-argument)._

This document answers one question per component: **why does localGPT do it this
way?** It describes only what ships in the current tree. Anything planned,
proposed or merely promising lives in [`research_roadmap.md`](research_roadmap.md)
and [`improvement_plan.md`](improvement_plan.md), not here.

## The method

Three artefacts, in a fixed order:

1. **Evidence** — [`research/`](research/) holds three August-2026 sweeps of
   primary sources (papers, model cards, first-party engineering blogs), each
   claim graded *established* / *emerging* / *contested* and each carrying its
   own "could not verify" appendix. These documents describe the field, **not
   this repo**.
2. **Our own eval** — [`eval/`](../eval/) holds a 72-query gold set over three
   corpora, a recall/nDCG runner, a binary groundedness judge validated against
   hand labels, and an end-to-end smoke test. See [`eval/README.md`](../eval/README.md).
3. **A decision** — nothing changes a default without a measured delta on (2).
   The decisions and their numbers are in [`eval/DECISIONS.md`](../eval/DECISIONS.md)
   and [`eval/decisions/`](../eval/decisions/).

The order matters because **it repeatedly produced the opposite answer from the
literature.** Three times in one week:

| The evidence said | Our eval measured | What shipped |
|---|---|---|
| "Cross-encoder rerank: **YES, unconditionally**, +17.2 pp MRR@3" ([`component-map-2026.md` §5.1](research/component-map-2026.md), §6.5) | `bge-reranker-v2-m3` on top of the new first stage: **−0.022 nDCG@10** on `mixed`, **−0.058** on `docs`, for ~1.6 s/query | Reranking initially **off** by default ([`DECISIONS.md` §1](../eval/DECISIONS.md)); turned **on** with score-threshold selection in arm G, 2026-08-14 |
| Decomposition helps when applied at *reranking* rather than first-stage (2026 MultiConIR/SSRB, [`component-map-2026.md` §6.2](research/component-map-2026.md)) | On the 6 queries that genuinely decompose: **−0.046** (`max`) / **−0.012** (`mean`) nDCG@10 | Shape change shipped; sub-query scoring at rerank now applies whenever the reranker is on (arm G), and the pooled first stage is the default decomposition path (arm H, 2026-08-15) ([`phase2-pipeline.md` §3](../eval/decisions/phase2-pipeline.md)) |
| Bigger instruction-tuned embedders lead the boards; the shipped default was the 8 GB `Qwen3-Embedding-4B` | A 1.2 GB MIT model **dominated** it: mixed nDCG@10 **0.915 vs 0.875**, ~3× lower latency, ~7× less memory | `microsoft/harrier-oss-v1-0.6b` as the default ([`embedder.md` gate section](../eval/decisions/embedder.md)) |

That is the whole method: the literature nominates candidates, our gold set
decides. Where the two disagree, the gold set wins and the disagreement is
written down rather than smoothed over.

Every number below is traceable to a file under `eval/`. Where a claim has no
number, it says so.

---

## 1. Parsing / OCR

**What ships.** Every document goes through [docling](https://github.com/docling-project/docling)
(`rag_system/ingestion/document_converter.py`). PDFs take one of two converters:
a text-layer probe (`_pdf_has_text`, PyMuPDF) decides whether OCR runs at all —
if *any* page has extractable text, the whole document is converted without OCR.
DOCX / HTML / MD go through a third general converter; `.txt` is read directly
and fenced. Conversion returns `(markdown, metadata, DoclingDocument)` so the
chunker can use the element tree rather than re-parsing markdown.

The OCR engine is **probed, not configured**: `build_ocr_options()` walks
`OcrMac → EasyOCR → RapidOCR → tesserocr → tesseract-cli` and picks the first
whose backend is actually importable on this host. There is no VLM parser.

**Why.** The 2026 evidence ([`component-map-2026.md` §1.5](research/component-map-2026.md))
says the local pipeline is "Docling as orchestration + a 0.9–1.2B specialist VLM
as the parsing engine", and that traditional parsers "survive as the fast path,
not the quality path". We ship the orchestration half and not the VLM half, on
purpose:

* The VLM spike ([`glm-ocr-spike.md`](../eval/decisions/glm-ocr-spike.md)) is
  **GO-LATER**, not NO-GO. It demonstrated a real, large win on the one document
  class that matters — a degraded scanned invoice where GLM-OCR read **30/30
  table cells** and the current chain lost every price and 4 of 5 part numbers.
* Three defects block adoption: Ollama's Modelfile ignores prompts (so GLM-OCR's
  table/formula modes are unreachable), some pages are deterministically
  transcribed twice, and docling flattens the model's pipe tables to
  `tables: 0`. None of them is code we should write blind.
* **There is no OCR eval.** `eval/corpora/` contains only digital-born PDFs with
  clean text layers, so nothing in the harness exercises the OCR branch. Adopting
  a parser on leaderboard position alone would violate the gate this repo runs on
  — and the spike showed exactly why: the roadmap's original "#1 on OmniDocBench,
  beats GPT-5.2 by ~10 points" line did not survive source verification (GLM-OCR
  is **third** on v1.6_full, behind PaddleOCR-VL-1.6 and MinerU2.5-Pro).

The probe order was also the cheap win: the RapidOCR probe used to test for the
stale module name `rapidocr_onnxruntime`, so a host with `rapidocr` 3.x installed
silently fell all the way through to the Tesseract CLI. It now accepts either
name (`OCR_BACKENDS`, `document_converter.py`), and on this machine RapidOCR is
what resolves.

**What would change the decision.** Build a scanned/tabular corpus with ground
truth under `eval/corpora/`, then A/B GLM-OCR against the *fixed* classic chain
**and** against docling's other 2026 presets (`lightonocr`, `dots_ocr`,
`nanonets_ocr2`) — one column in a table, not a foregone winner. Independently of
any VLM: `pip install ocrmac` on macOS is a free upgrade over the current chain,
and the text-layer probe should become per-page (today a scanned insert inside a
digital PDF gets no OCR at all).

## 2. Chunking + index-time enrichment

**What ships.** `DoclingChunker` (`rag_system/ingestion/docling_chunker.py`) is
the default (`chunker_mode: "docling"`). It walks the `DoclingDocument` element
tree, emits **tables, code and figures as atomic chunks**, sentence-packs
paragraph nodes up to a token budget, and attaches the heading path and block
type to every chunk's metadata. Token counting uses the *embedding model's own*
tokenizer. Budget is 512 tokens for HTTP index builds
(`rag_system/api_server.py`) and falls back to 1500 for the CLI path, because no
profile sets `chunking.chunk_size`. Overlap is one sentence.

**Index-time contextual enrichment is on** in the `default` profile
(`contextual_enricher: {enabled: True, window_size: 1}`, `rag_system/main.py`):
`ContextualEnricher` asks the enrichment model for a 2–5 sentence situating
summary per chunk from its ±1 neighbours and prepends it to the indexed text,
keeping the original text in metadata (`rag_system/indexing/contextualizer.py`).

**Late chunking** is implemented (`rag_system/indexing/latechunk.py`: embed the
whole document, mean-pool inside chunk spans) and enabled in the `default`
profile, but `POST /index` defaults `enable_latechunk` to `false`, so HTTP builds
and CLI builds differ — tracked as [`improvement_plan.md`](improvement_plan.md)
§3.6, not defended here. When a late-chunk table exists, its hits are merged with
their ±1 siblings before reranking (`retrieval_pipeline.py::_first_stage`).

**Why.**

* **Boring chunking is the right default.** Every controlled study since 2024
  puts the *total* spread across all chunking methods at ~9 points of recall —
  Chroma's 2024 report and a 2026 eight-method / nine-dataset replication both
  land on recursive/fixed splitting as the cost-effective winner, with LLM-driven
  chunkers (DenseX 69.1 Acc@5, 15+ hours) far behind
  ([`component-map-2026.md` §2.1, §2.5](research/component-map-2026.md)). The
  same source's summary is blunt: "chunking is not where your quality is."
* **Structure-awareness is nearly free** when the parser already hands you an
  element tree, which is why tables and code are atomic
  ([`component-map-2026.md` §2.5](research/component-map-2026.md)).
* **Contextual enrichment is the one chunking-adjacent intervention with a large
  measured effect** — 35% top-20 failure-rate reduction from contextual
  embeddings alone, 67% stacked with BM25 and reranking (Anthropic, via
  [`component-map-2026.md` §2.3](research/component-map-2026.md)). It is an
  offline cost, which is the only reason it is affordable here.
* **Late chunking is deliberately conditional.** The evidence grades it
  *contested* — efficient, but losing relevance to contextual retrieval, with the
  sign flipping by corpus ([`component-map-2026.md` §2.2](research/component-map-2026.md)).
  It is a fix for cross-reference breakage, not a general upgrade, which matches
  shipping it as a per-build flag rather than a mandate.

**What would change the decision.** Chunking has never been A/B'd on our gold
set — no `eval/` number defends the 512/1500 budget or the one-sentence overlap.
If chunking is ever revisited, re-run it *after* any local-reader upgrade: the
same literature that says compression benefit shrinks as the reader gets stronger
says the identical thing about chunking ([`component-map-2026.md` §2.4](research/component-map-2026.md)).

## 3. Embeddings

**What ships.** `microsoft/harrier-oss-v1-0.6b` (MIT, 1024-dim, 1.2 GB) is the
default (`rag_system/main.py::EXTERNAL_MODELS`), overridable with
`EMBEDDING_MODEL`. Queries get the `Instruct: {task}\nQuery: {text}` prefix;
**documents do not** (`representations.py::default_query_instruction` +
`apply_query_instruction`, resolved per-pipeline by
`retrieval_pipeline.py::_query_instruction`, which honours
`config["embedding_instruction"]` then `EMBEDDING_INSTRUCTION` then the model
family's default). The prefix applies only to families that were trained with one
(`qwen3-embedding`, `harrier`); everything else gets `""`.

Two index-format guarantees ship alongside it, both in
`rag_system/indexing/embedders.py` and `rag_system/retrieval/retrievers.py`:

* **Per-table embedder identity.** Every table records the model that wrote it
  plus a `normalized` flag in the Arrow schema metadata (with a JSON sidecar
  fallback). Indexing into or querying a table written by a different embedder
  raises `EmbedderMismatchError`.
* **L2 normalization** at write and query time, gated on that marker, so
  LanceDB's default L2 ordering *is* the cosine ordering both model cards
  specify. Legacy unmarked tables keep working unnormalized, with a warning. The
  default table name moved to `text_pages_v4` so the shipped default starts clean.

**Why.**

* **The prefix**: every top-2026 embedder is instruction-tuned, all use the same
  format, all specify no instruction on the document side, and Qwen3-Embedding
  reports 1–5% from the prefix alone
  ([`component-map-2026.md` §3.5](research/component-map-2026.md)). The audit
  found the repo was sending **no prefix at all**
  ([`embedder.md` §1](../eval/decisions/embedder.md)). Our measurement is more
  nuanced than the card: on harrier the prefix helps both metrics
  (mixed nDCG@10 0.908 → 0.915), on Qwen3-0.6B it is a ranking-vs-recall trade
  that washes out after a cross-encoder. Because only the query vector changes,
  turning it on invalidated no index.
* **The model**: not chosen from a leaderboard. Measured head-to-head on our gold
  set, harrier-0.6b beat the then-shipped 8 GB `Qwen3-Embedding-4B` on mixed
  nDCG@10 (**0.915 vs 0.875**), docs nDCG@10 (**0.759 vs 0.638**), recall@5 and
  recall@10, at ~3× lower latency and ~7× less memory
  ([`embedder.md`, gate validation](../eval/decisions/embedder.md)). MIT is also
  strictly more permissive than Apache-2.0.
* **The identity marker exists because the *previous* guard could not have caught
  this swap.** It compared vector *width* only, and harrier-0.6b and
  Qwen3-Embedding-0.6B are both 1024-dim — exactly the swap this adoption makes
  people likely to perform. It would have appended mutually unintelligible
  vectors to a live table, silently ([`DECISIONS.md` §2](../eval/DECISIONS.md)).
* **Normalization was adopted for card-conformance, not for a number, and the
  page says so.** Measured on `mixed`: nDCG@10 0.915 → **0.911**, recall@5 0.917 →
  0.931. A wash. It ships because the previous behaviour was neither cosine nor
  intended, not because it won.

**Post-adoption verification.** `mixed`, 317 chunks, 72 queries, first stage only:
recall@5 **0.944**, recall@10 0.958, recall@20 1.000, nDCG@10 **0.913**
([`DECISIONS.md` §4](../eval/DECISIONS.md)). Note that the `docs` and `mixed`
corpora index live `Documentation/*.md` — **editing this file moves these
numbers.** Only compare runs made against the same tree.

**What would change the decision.** `Qwen/Qwen3-Embedding-4B` stays a documented
option for multilingual or long-context (32K) corpora, which this English,
digital-born gold set does not exercise — keep the prefix on for it, worth +0.059
nDCG@10. Any embedder change forces a re-index **and** re-opens §6: the reranker's
value is a function of first-stage quality, and these two cannot be decided
independently.

## 4. Hybrid retrieval + RRF

**What ships.** `MultiVectorRetriever.retrieve` (`rag_system/retrieval/retrievers.py`)
runs LanceDB's native full-text leg and the dense leg **in parallel** and fuses
them with reciprocal rank fusion at `_RRF_K = 60`. `retrieval_mode` selects
`hybrid` (default), `vector_only` or `fts_only`; an unknown value falls back to
hybrid with a warning. Single-word FTS queries are rewritten to
`"<word>* OR <word>~"` for prefix and fuzzy matching. Rows are de-duplicated on
`chunk_id` (then `_rowid`, then text) across the two legs, and each mode exposes
exactly one "higher is better" `score` field.

**There are no fusion weights, and there is no knob to add them.**

**Why.** BM25 scores are unbounded positives while cosine is bounded, so naive
weighted addition is meaningless; RRF sidesteps normalization entirely by using
ranks ([`component-map-2026.md` §4.3](research/component-map-2026.md)). Hybrid+RRF
measurably beats both legs (T2-RAGBench: Recall@5 0.695 vs BM25 0.644 vs dense
0.587), and — the useful corrective — **BM25 alone beat dense alone there by 5.7
points**, so dropping the sparse leg is not the simplification it looks like.
Qdrant's own published guidance is the most honest vendor position available:
weighted RRF only when you have a tuned eval set with a train/val split, "neither
method dominates universally", and retune whenever retrievers, embeddings or
corpus change. §4.4's bottom line: "Do not spend time tuning fusion before you
have a reranker." This repo's history includes a *broken* weighted blend, which
RRF replaced ([`improvement_plan.md` §0](improvement_plan.md)).

**What would change the decision.** A per-index place to store tuned weights plus
a per-corpus validation split — that is [`improvement_plan.md`](improvement_plan.md)
§1.3, and it is correctly still open. The prerequisite is not the code; it is
having something to tune against that is not the same 72 queries used to evaluate.

## 5. Evidence-sufficiency retry

**What ships.** One conditional second retrieval
(`retrieval_pipeline.py::retrieve_candidates`), on in `default`
(`retrieval.retry: {enabled: True, min_top_score: 0.12, max_attempts: 1}`), off in
`fast`. When the first pass's evidence score falls below the threshold, the
enrichment model rewrites the query once (JSON-formatted so a small model's
"thinking" preamble cannot leak in), retrieval re-runs, and **the better of the
two result sets is kept** — a retry that did not improve the evidence is
discarded, not merged. It is inert in `fts_only` mode and on legacy unnormalized
tables, by design: no signal, no retry. The retry surfaces as a
`retrieval_retry` SSE event.

The signal is **not** the raw top similarity:

```
evidence = (cos_top − cos_background) / (1 − cos_background)
```

with `cos_background` the mean cosine from rank 6 down
(`_dense_evidence_score`). When a reranker returns a calibrated 0–1 probability,
its top score is preferred (`_rerank_evidence_score`); arbitrary logits are
rejected rather than compared to a probability threshold.

**Why.** The evidence says one conditional second iteration captures nearly all
of the deep-loop gain and that stopping criteria should be based on **cumulative
evidence sufficiency, not query count** — across six agents on BrowseComp-Plus,
search volume correlates only weakly with answer quality, and redundant queries
characterise *underperforming* agents ([`component-map-2026.md` §8.6](research/component-map-2026.md)).
The broader 2026 rule is "escalate, don't pre-decide"
([`component-map-2026.md` §6.5, §7.4](research/component-map-2026.md)).

Our own measurement changed the *signal*, which is the part worth recording. The
roadmap said to trigger on the top score. On the gold set, raw top cosine is
**anti-correlated with success**: all three `mixed` first-stage misses scored
*above* the median successful query, and any threshold catching all three fires
on 94% of the successes. Contrast works; absolute similarity mostly encodes how
close a query's phrasing sits to the corpus register
([`phase2-pipeline.md` §2.1](../eval/decisions/phase2-pipeline.md)).

Effect, four runs: fires on **9.7–11.1%** of `mixed`, **+0.008 to +0.017**
nDCG@10 and +0.014 recall@10, **zero per-query regressions**. It repaired one
genuine recall@10 = 0 miss. Mean latency over all queries 96 ms → 186 ms
([`phase2-pipeline.md` §2.3](../eval/decisions/phase2-pipeline.md)).

**Honest limits, carried forward rather than buried.** The threshold was
calibrated on the same 72 queries it is evaluated on, against three first-stage
misses. `0.12` is a starting value, not a tuned constant; it catches **one** of
the three real misses, because catching all three costs a 26% false-fire rate. On
`mixed` the +0.008 is *inside* the one-query noise floor — the reason to ship is
that it is positive on both corpora across four runs with nothing getting worse.

**What would change the decision.** A larger gold set with more first-stage
misses to calibrate against. The firing rate drifted 9.7% → 11.1% as the corpus
grew, so it is a property of the corpus, not a constant — re-check it after any
embedder change.

## 5a. Per-query token accounting

**What ships.** Every Ollama completion the agent makes — streaming or not —
reports `prompt_eval_count` and `eval_count` on its final object. Those are
aggregated per user query, bucketed by pipeline stage (`triage`,
`decomposition`, `synthesis`, `verification`), and returned as `token_usage` on
the `/chat` response body and in the SSE `complete` event. On by default: it
costs one dict update per LLM call and adds no request.

The aggregation point is a `ContextVar` in `rag_system/utils/ollama_client.py`
rather than an argument threaded through every call site, because one
`OllamaClient` is shared by the agent, the retrieval pipeline, the verifier and
the decomposer. `await` and `asyncio.to_thread` propagate it; the agent's
parallel sub-query `ThreadPoolExecutor` copies it explicitly.

Three honest gaps: the retry's reformulation call is billed to `synthesis`,
because it happens inside `RetrievalPipeline.run()` which the agent labels as
one stage; watsonx reports zeros, because the SDK path in use surfaces no
per-call counts; and an absent stage key means "no LLM call in that stage",
not "zero tokens". Evidence: `eval/decisions/phase4-escalation-tokens.md`.

## 5b. Metadata filters and ask-a-folder

**What ships.** `/chat` and `/chat/stream` accept an optional `filters` JSON
object (document id/name, chunk id, chunk-index ranges) compiled to LanceDB
where-clauses that prefilter **both** the vector and FTS legs. There is no
flag: with no `filters` argument the path is byte-identical to not having the
feature (md5-verified against a pre-change tree). Values containing quoting
characters are refused, never escaped; a malformed filter is a 400 from the one
validator in `rag_system/retrieval/filters.py`. Page and date filters are NOT
shipped — they live inside the metadata JSON string column and need real
columns plus a re-index.

`python -m rag_system.main ask <folder> "<question>"` builds an ephemeral index
under a temp directory (fast profile, no enrichment), answers with the standard
pipeline, and removes everything afterwards — including on SIGTERM.
Evidence: `eval/decisions/phase4-filters-askfolder.md`.

## 6. Reranking posture

**What ships.** Since arm G (2026-08-14) the `default` profile ships
`reranker.enabled = True` with threshold selection — `top_k: 10`,
`min_score: 0.5`, `min_keep: 3` (`rag_system/main.py::PIPELINE_CONFIGS`), and
the UI "AI reranker" toggle defaults on to match
(`src/components/ui/session-chat.tsx`). `Qwen/Qwen3-Reranker-4B` is loaded
lazily on the first reranked query through the in-repo
`QwenRerankerScorer` (`rag_system/rerankers/reranker.py`), routed either by
explicit `reranker.model_type: "qwen3"` or by model name
(`retrieval_pipeline.py::_get_ai_reranker`). Any other model still goes through
the `rerankers` library. A reranker that fails to load logs a warning and is
skipped — there is no fallback reranker.

**Why this is the most counter-intuitive decision in the repo.** The evidence is
about as strong as evidence gets: reranking is "the single highest-ROI component
in the stack", +17.2 pp MRR@3 on T2-RAGBench, −1.7 EM when removed from a local
7B ablation, and the 2026 recommendation is a flat "**YES, unconditionally**"
([`component-map-2026.md` §5.1, §6.5](research/component-map-2026.md)).

Our gold set says otherwise, for a specific and explicable reason. The joint
matrix ([`DECISIONS.md` §1](../eval/DECISIONS.md)), `mixed` corpus, same 20
first-stage candidates reordered by each:

| Stack | mixed nDCG@10 | docs nDCG@10 | added latency |
|---|---|---|---|
| **harrier-0.6b, first stage only** | 0.915 | 0.759 | — |
| + `BAAI/bge-reranker-v2-m3` | 0.892 | 0.701 | **+1.6 s — net negative** |
| + `Qwen/Qwen3-Reranker-4B` ← shipped since arm G (2026-08-14) | 0.977 | 0.932 | +12.7 s |

Two findings, both load-bearing:

1. **The cheap cross-encoder now hurts.** bge-reranker-v2-m3's famous +0.232 on
   `docs` was largely a *repair job on a weak first stage*. Improve the first
   stage and the repair becomes damage. This is exactly why 1.1 and 1.2 could not
   be decided independently and were re-measured jointly in one re-index window.
2. **The good reranker is a real win and was initially judged too slow to
   default on.** +0.062
   nDCG@10 on `mixed`, +0.173 on `docs` — the largest single quality win in this
   repo's eval history — for ~12.7 s per query and 7.5 GB of resident weights on
   a single-user, single-threaded server. Arm G (2026-08-14) reversed the "off by
   default" call: that call predated the synthesis context budget, when rank
   order barely mattered because front-truncation fed synthesis the tail of the
   list anyway. Now the budget keeps exactly the top-ranked documents, so
   ordering and selection decide everything the model reads — and `min_score`
   selection sends a small, clean context on easy questions instead of a fixed
   ten.

`Qwen/Qwen3-Reranker-0.6B` was rejected outright: +0.021 nDCG@10 over bge (one to
two queries out of 72, inside the noise band) bought with 1.5–2.8× the latency,
while *losing* recall@10 on both corpora ([`reranker.md` §7](../eval/decisions/reranker.md)).

**One integration finding worth keeping.** `rerankers` 0.10.0 has no
Qwen3-Reranker backend. Loading one through the shipped cross-encoder path builds
a `Qwen3ForSequenceClassification` with a **randomly initialised score head** —
had the batching not thrown, it would have returned untrained noise while
printing "AI reranker initialized successfully". `QwenRerankerScorer` implements
the model card's actual scheme (causal LM, left padding, chat template,
softmax over the `yes`/`no` logits at the final position), and the name-based
route in the loader exists specifically so no configuration can reach the random
head ([`reranker.md` §1](../eval/decisions/reranker.md)).

**What would change the decision.** Three concrete triggers:

* **Re-run the A/B if the embedder changes.** The reranker's headroom is largest
  exactly where the first stage is weakest, so this decision is a function of §3
  and expires with it.
* **Tune the latency knobs.** Batch size (8) and the 2048-token truncation cap
  are both untuned, and reranking only the top 10 candidates instead of 20 is
  unmeasured. Either could move the 12.7 s materially — that is unmeasured work,
  not a promise.
* **A later `rerankers` release** may add a Qwen3 backend, at which point the
  name-based route should be revisited.

Note also what no reranker fixed: `docs_d09` and `docs_d17` degrade under every
model tested. They are a query-understanding problem, and this section is not
where they get solved.

## 7. Query decomposition

**What ships.** Since arm H (2026-08-15) the `default` profile ships
`compose_from_sub_answers: false` with `pooled_first_stage: true`: each
sub-query runs first-stage retrieval, the candidates are pooled and
de-duplicated, and there is ONE rerank pass and ONE synthesis over the union
context (`retrieval_pipeline.py::_pooled_first_stage`). Sub-queries are also
used at the *rerank* stage, where each candidate is scored against every
sub-query and the scores are aggregated by
`query_decomposition.rerank_aggregate` (`"mean"` default, `"max"` available) —
`retrieval_pipeline.py::_rerank_stage`. The compose path — a separate *answer*
per sub-question, composed into the final answer — remains available behind
`compose_from_sub_answers: true` (`rag_system/agent/loop.py`).

Consequence, stated plainly: the earlier posture — **no shipped profile enables
sub-query scoring at rerank** — held while the `default` profile kept the
reranker off and `compose_from_sub_answers: true`. Arm G turned the reranker on
(2026-08-14), so the aggregation path now runs by default, and arm H
(2026-08-15) made the pooled first stage the default decomposition path.

**Why.** The evidence puts decomposition at "conditional — multi-hop, applied at
*rerank*", noting that decomposition at initial retrieval dilutes the query
semantically ([`component-map-2026.md` §6.2, §6.5](research/component-map-2026.md)).

We shipped the shape change and **measured the payload negative.** On `docs` with
`Qwen3-Reranker-4B`, only 6 of 24 queries decompose into more than one sub-query.
On exactly those 6, scoring against sub-queries at rerank is worse under both
aggregates: **0.8862 → 0.8406 (`max`, −0.046)** and **0.8740 (`mean`, −0.012)**.
The whole-corpus "gain" comes entirely from the 18 single-sub-query rows, where
the win is query *rewriting*, not decomposition
([`phase2-pipeline.md` §3](../eval/decisions/phase2-pipeline.md)).

The shape change ships anyway because it is a strict reduction in work — the
aggregate path used to issue N first-stage retrievals and now issues one — and
because the structural check passed: the first-stage number is byte-identical
across all three arms, proving decomposition can no longer touch it. (Arm H's
pooled first stage later re-introduced per-sub-query retrieval; the reduction it
keeps is one rerank pass and one synthesis instead of N.)

**What would change the decision.** n_effective = **6 queries on one corpus**.
That is too small to call the 2026 MultiConIR/SSRB finding wrong; it is big
enough to say it did not reproduce here, which is why nothing was switched on. A
multi-hop corpus with enough genuinely-decomposing queries to move a metric would
re-open it. `mean` stays the default aggregate on the "less bad" argument, not a
positive one.

## 8. Routing / triage

**What ships — two layers, exactly one of which calls an LLM.**

*Layer 1, the gateway* (`backend/server.py::should_use_rag`) is deterministic and
makes no network call: `force_rag` → RAG; no linked indexes → direct LLM;
whole-message smalltalk (≤ 6 words, anchored allowlist, must contain a core
phrase) or assistant-meta ("who are you", "what model are you") → direct LLM;
**everything else → RAG.** Unit-tested at 155/155 in
`backend/test_gateway_routing.py`.

*Layer 2, the agent* (`rag_system/agent/loop.py::_triage_query_async`) is the
system's single LLM routing layer: document overviews + the utility model decide
`rag_query` vs `direct_answer`, with "history exists → `rag_query`" as a shortcut
and an LLM fallback when no overviews are loaded. `_normalize_triage` collapses
anything that is not an explicit `direct_answer` to `rag_query`, so a small model
still emitting the retired `graph_query` label lands on the RAG path.

**Why.** Pre-retrieval LLM routing is the weakest measured pattern of 2026, and
three independent sources agree: four ML approaches to pre-retrieval routing all
failed because "the need for augmentation cannot be determined from the query
alone"; rule-based retriever routing *lost* to fixed hybrid by 1.8 EM; and
TF-IDF+SVM matches or beats neural and LLM routers at ~zero cost
([`component-map-2026.md` §7.1, §7.3, §7.4](research/component-map-2026.md)). The
four-year pattern in §7.3 is "a small discriminative classifier is the right
tool; an LLM router is rarely justified."

The bias toward over-sending to RAG is deliberate and cheap: agent triage runs on
every forwarded request and can still answer directly, so a false "use RAG" costs
one call on a model that would have been called anyway, while a false "answer
directly" costs an unanswerable question. The gateway is a smalltalk filter in
front of the decision-maker, not the decision-maker.

**Measured** ([`phase2-gateway.md` §3](../eval/decisions/phase2-gateway.md)): mean
routing decision **750.613 ms → 0.002 ms** over the same 20 messages, with
**20/20 decision agreement** with the LLM router it replaced. The deleted
keyword fallback was worse than the old docs claimed — it matched greetings by
*substring*, so `'hi'` matched *w**hi**ch*, *t**hi**s* and *mac**hi**ne*, routing
**7 of 8** real Atlas-7 questions to the direct LLM. That is why it was deleted
rather than patched, and why "messages containing test/check route RAG" is now a
regression test.

**What would change the decision.** [`improvement_plan.md`](improvement_plan.md)
§2.1 (embed and cache document overviews for a cosine pre-check) and §2.2
(session-level routing memo) both still stand for the *agent* layer, which is now
the only per-query LLM routing call. The evidence's own carve-out is that routing
still pays for *pipeline depth* selection, implemented as a post-retrieval
cascade — which is what §5's retry is.

## 9. Verification

**What ships.** Verification is on in `default` (`verification: {enabled: True}`).
The shipped backend is an LLM prompt on the utility model
(`rag_system/agent/verifier.py::Verifier.verify_async`), returning a JSON verdict
and a confidence score; a low-confidence or ungrounded verdict appends a warning
to the answer (`agent/loop.py`).

A **seam** exists for a local model: `VERIFIER_MODEL` / `verification.model`
swaps in `LocalNLIVerifier`, which sentence-splits the answer, scores each
sentence against the retrieved evidence as premise, and takes the **minimum** —
one unsupported sentence makes the answer ungrounded, matching the binary
semantics `eval/judge.py` already uses. A model that cannot be loaded **raises**,
printing the availability table, rather than falling back: a verifier that
silently is not the verifier you configured is worse than an error. **The default
is unchanged.**

**Why.** Verification helps as an *external* check, and the 2026 result is that a
4-bit 1B verifier (ThinknCheck, 78.1 BAcc) now beats the 7B 2024 SOTA, which
would make per-answer grounding cheap enough to always run
([`component-map-2026.md` §9.2](research/component-map-2026.md)).

Both of the roadmap's named candidates failed availability checks against the
HuggingFace Hub API: **ThinknCheck has no public weights** (zero models returned;
the paper links no release), and Granite Guardian is either 8B/~16 GB or — in its
38M form — a hate/abuse classifier, the wrong task entirely. Two substitutes were
wired and exercised rather than left as a stub, and both were run against all 20
hand-labelled cases in `eval/judge_validation.jsonl`
([`phase2-pipeline.md` §4](../eval/decisions/phase2-pipeline.md)):

| Verifier | agreement | TPR | TNR |
|---|---|---|---|
| `lytang/MiniCheck-DeBERTa-v3-Large` (1.74 GB) | 19/20 | 10/10 | 9/10 |
| `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` (369 MB) | 18/20 | 8/10 | 10/10 |

They fail in **opposite directions** — MiniCheck missed a swapped-entity error,
the generic NLI model rejected two correct answers — so neither is a drop-in
improvement without its own validation run. Nothing was made the default.

**`[Confidence: N%]` is UX, not a calibrated measurement**, and
[`verifier.md`](verifier.md) says so in a callout. Changing the backend changes
where the number comes from; it does not calibrate it. The evidence supports the
caution independently: faithfulness metrics measure precision and ignore
coverage, so a system gated only on "is every claim supported" learns to abstain
([`component-map-2026.md` §9.4](research/component-map-2026.md)).

**What would change the decision.** Run `eval/judge.py --validate`-style
TPR/TNR discipline on a candidate verifier over more than 20 cases, and beat the
LLM prompt on both directions rather than one. If ThinknCheck ever publishes
weights, it is the first thing to try. Also worth knowing: verifying against the
retrieved chunk alone systematically *under*-detects, because supporting evidence
is frequently outside the truncated passage
([`component-map-2026.md` §9.3](research/component-map-2026.md)).

## 10. Sentence pruning

**What ships.** Provence (`naver/provence-reranker-debertav3-v1`) via
`rag_system/rerankers/sentence_pruner.py`, applied after context expansion and
before synthesis (`retrieval_pipeline.py::run`), with fully-pruned chunks
dropped. It is **opt-in**: `provence.enabled` defaults to `False`, exposed as the
UI toggle "Prune irrelevant sentences" and as `provence_prune` on the API. The
model is loaded lazily, once, behind a lock, and a load failure skips pruning
rather than failing the query.

**Why.** Provence is the one context-reduction variant whose economics survive
2026 scrutiny: it formulates pruning as sequence labelling **unified with
reranking**, so it is folded into a stage you already run, at "negligible to no
drop in performance… at almost no cost"
([`component-map-2026.md` §10.1](research/component-map-2026.md)). §10.5's verdict
is exactly the posture here: "**prune, don't compress.**"

It was kept off by default because localGPT did *not* run a cross-encoder by
default at the time (§6), so Provence's "zero marginal cost" argument — the
entire reason it beats token-level compression — did not hold in the shipped
configuration: an extra DeBERTa forward pass, not a free rider. Since arm G
(2026-08-14) the reranker **is** on by default, so that premise no longer holds;
folding pruning into the rerank stage is now a live candidate, but no `eval/`
number yet measures its effect on our gold set.

**What would change the decision.** Measure it. Reranking returned to the
default profile in arm G (2026-08-14), so re-evaluate folding pruning in with
it. `OpenProvence`
(30M–310M, MIT, ModernBERT) is the lighter candidate the evidence points at.

## 11. Caching and memory

**What ships.** An in-process semantic cache in the agent
(`rag_system/agent/loop.py`): the raw query is embedded, compared by cosine
against cached entries, and a hit at or above `semantic_cache_threshold`
(**0.98**) returns the stored result. `cache_scope` defaults to **`"session"`**,
so an entry from another session is skipped before the similarity is even
computed. Conversation memory is a session transcript — an in-process
`chat_histories` map plus SQLite rows in `backend/chat_data.db` — formatted into
the query as history. There is **no vendor memory layer**, no entity graph, no
consolidation pass.

**Why.**

* **The null baseline wins.** The 2026 guidance is explicit: "plain RAG over a
  session transcript store as the baseline you must beat" — Cloud-RAG beat Mem0
  on LongMemEval-S, a filesystem agent beat Mem0 on LoCoMo, and the benchmark the
  whole category is scored on (LOCOMO) is broadly discredited on structural
  grounds ([`component-map-2026.md` §11.1, §11.4](research/component-map-2026.md)).
  Our session store *is* that null baseline. The same section notes a bare
  embedding-model swap moves accuracy 6.2 pp — **larger than most claimed
  memory-system deltas** — which is where the effort went instead (§3).
* **Session scoping is a privacy fix, not a performance one.** A global cache
  returns one session's answer to another session's user.
  [`improvement_plan.md` §0](improvement_plan.md) records it as closing a
  cross-session answer leak.
* **The 0.98 threshold is deliberately extreme.** Semantic caches are attackable
  by embedding-similarity collision (86% hijacking hit rate), and the consistent
  2026 theme is that hit rate is the wrong headline metric — calibration,
  freshness and collision-resistance decide whether a cache is *safe*
  ([`component-map-2026.md` §11.3](research/component-map-2026.md)). At 0.98
  nothing but a near-identical query hits, which was also verified as a side
  effect of the prefix work: with the query prefix on, mean pairwise cosine over
  the 72 gold queries *falls* (0.361 → 0.216) and max reaches 0.899, so nothing
  crosses the bar either way ([`embedder.md` §8](../eval/decisions/embedder.md)).

**What would change the decision.** Instrument the cache for calibration and
staleness rather than hit rate before loosening the threshold. Graphiti +
FalkorDB Lite is the evidence's pick *if* temporal/entity memory with provenance
is ever actually needed — but the null baseline has to lose on our own eval first,
and there is no memory eval in `eval/` today.

## 12. Streaming and persistence

**What ships.** `POST :8001/chat/stream` emits SSE
(`rag_system/api_server.py::handle_chat_stream`) and every pipeline phase is a
typed event: `retrieval_started`, `retrieval_done`, `retrieval_retry`,
`rerank_started`, `rerank_done`, `context_expand_*`, `prune_*`, `token`,
`sub_query_token`, `complete`. Synthesis streams token-by-token from
`_synthesize_final_answer`. A client disconnect (`BrokenPipeError`) is logged,
not raised.

**The stream writes nothing to SQLite.** When the stream finishes, the UI posts
the completed turn to `POST :8000/sessions/{id}/messages/save`
(`src/lib/api.ts::saveStreamedTurn`, `backend/server.py`), which persists the
user message, the assistant message, the source documents and the pipeline
cascade in `metadata.steps`. Direct stream consumers must do the same to get
history — documented as known limitation #1 in
[`system_overview.md`](system_overview.md#11-known-limitations).

**Why.** This is an architecture choice, not an evidence-backed one, and it is
recorded here so it is not mistaken for the latter. The rationale is
single-writer discipline: the RAG API owns retrieval and the vector store, the
gateway owns SQLite, and no request writes to a store it does not own. The cost
is the extra round-trip and the fact that an interrupted stream persists nothing.

The one thing the evidence *does* motivate is that the retry and the rerank are
surfaced as SSE steps rather than hidden — the 2026 escalation-architecture
literature treats visible, staged escalation as the point of the design
([`component-map-2026.md` §8.6](research/component-map-2026.md), PEA-CAE).

**Verified by** `eval/smoke_e2e.py`: **25/25 assertions**, including planted-fact
answers with non-empty `source_documents`, the `[Confidence: N%]` tag, and the
streamed-turn save/round-trip ([`phase2-gateway.md` §3.4](../eval/decisions/phase2-gateway.md)).
Note the honest gap recorded there: smoke sends `force_rag: True`, so it
exercises the gateway's `force_rag` branch, not the discriminative one.

**What would change the decision.** Making the stream itself durable would need
the RAG API to write session state, which crosses the ownership boundary. The
lower-cost fix is a server-side save on stream completion in the gateway's
proxy path.

---

## 13. Deliberately not implemented

Each of these is **absent on purpose**. If you are about to add one, read the row
first — and bring a gold-set number, not a paper.

| Not implemented | Why (evidence) | Revisit when |
|---|---|---|
| **HyDE** — in any form. `grep -i hyde` over `rag_system/`, `backend/`, `src/` and `eval/` returns **zero hits**; there is no flag, no dead code, no "coming soon". | HyDE underperforms plain dense retrieval on entity/numeric corpora (T2-RAGBench: Recall@5 0.544 vs 0.587, nDCG@10 0.433 vs 0.466) because pseudo-documents fabricate figures; and in a production system only **27.8%** of real queries needed LLM augmentation while synthetic evals implied >90% — "the Coverage Illusion". Verdict: "alive but no longer a default. **Never always-on.**" ([`component-map-2026.md` §6.1](research/component-map-2026.md)) | Only as a *post-retrieval* escalation (retrieval returned nothing → then HyDE), or moved to index time (HyPE). The escalation slot in this codebase is already occupied by §5's retry; a HyDE variant would have to beat it on the gold set. |
| **Multi-query expansion** | The weakest of the three query transformations: on T2-RAGBench multi-query scored Recall@5 0.640 — **worse than plain BM25** (0.644). Prompt-only LLM rewriting measured **−9.0% nDCG@10 (p<0.001)** on FiQA, and an attempt to *gate* the rewriter reached only AUC 0.593 ([`component-map-2026.md` §6.3](research/component-map-2026.md)). | A trained lexical expander with BM25-level cost (the STORM line) that wins on our gold set. Note §5's retry is a *single conditional* rewrite kept only when it scores better — that is the sanctioned shape. |
| **Weighted / tunable fusion knobs** | BM25 and cosine are not on a common scale; "a fixed alpha over raw scores tends to be dominated by whichever retriever has larger raw magnitudes", and no 2026 evidence shows learned fusion beating RRF in the general case ([`component-map-2026.md` §4.3, §4.4](research/component-map-2026.md)). RRF is the scale-free safe default. | A per-index validation split exists to tune against and a place to store per-corpus weights — [`improvement_plan.md`](improvement_plan.md) §1.3. Tuning fusion before a reranker is explicitly the wrong order. |
| **GraphRAG** — removed 2026-08-09 | Loses on single-hop retrieval (64.78 vs 63.01 F1 on NQ); multi-hop gains span +3 to +27 depending entirely on how well the vector baseline is tuned; costs **41–57× at indexing** and up to **~377× in query tokens** ([`academic-evidence-2026.md` §6](research/academic-evidence-2026.md)). It was also unreachable — no shipped profile ever set `graph_strategy`. | A corpus where entity-linking is the task and a tuned vector baseline demonstrably fails on our gold set. Deleted: `graph_extractor.py`, `GraphRetriever`, `GraphQueryTranslator`, the `graph_query` triage outcome, and `networkx`/`fuzzywuzzy`/`python-Levenshtein` ([`phase2-pipeline.md` §1](../eval/decisions/phase2-pipeline.md)). |
| **Vendor memory systems** (Mem0, Zep/Graphiti, Letta, LangMem, GPTCache) | "Plain RAG over a session transcript store [is] the baseline you must beat" — Cloud-RAG beat Mem0 on LongMemEval-S, a filesystem agent beat it on LoCoMo, and LOCOMO itself is discredited (conversations of 16k–26k tokens do not stress memory). GPTCache is dormant since Aug 2024 ([`component-map-2026.md` §11.1, §11.2, §11.4](research/component-map-2026.md)). | The null baseline (§11) loses on a memory eval we actually own. Fix the embedding model before comparing memory systems — a bare swap moves accuracy 6.2 pp, more than most claimed memory deltas. |
| **Deep subagent / parallel fan-out loops** | The pro case (Anthropic's orchestrator-worker, +90.2%) runs at **~15× chat tokens** on frontier models. The 2026 counter-evidence: on repository-level code QA, plain semantic search scored **65.2%** vs deep agentic search **46.2% at >2× cost**, with **41.8% of failures at the planner→subagent hand-off** — "usually silent, ending in a fluent and confident answer that was wrong" ([`component-map-2026.md` §8.6](research/component-map-2026.md)). | Never, for a read-only indexable corpus on a single-user local box. The bounded version of this idea is already shipped: one conditional retry (§5), capped at `max_attempts: 1`. |
| **RL-trained searchers** (Search-R1 lineage) | On BrowseComp-Plus — a fixed, human-verified corpus that disentangles retriever from agent — **Search-R1 + BM25 scores 3.86%** while GPT-5 + BM25 scores 55.9% and GPT-5 + Qwen3-Embedding-8B scores 70.1%, with *fewer* search calls (ACL 2026 Main, [`component-map-2026.md` §8.3](research/component-map-2026.md)). No out-of-distribution transfer; also mis-calibrates confidence, which would corrupt §5's evidence signal. | A local-training story with demonstrated OOD transfer. §8.2 documents the 2026 lineage if that changes; nothing in it is currently a better use of a laptop GPU than a better embedder. |
| **Token-level context compression** (LLMLingua-style) | Across thousands of runs on 30,000 queries and three GPU classes: **at most ~18% end-to-end speedup**, and only when prompt length, ratio and hardware align — outside that window "compression overhead dominates and cancels the decoding gains entirely" (ECIR 2026). Fixed ratios reversed **31%** of pairwise model rankings on LongMemEval-S and obscured **80%** of the gain from a reader upgrade; hard compressors leave the answer path incomplete in 34–60% of multi-hop bridge examples. Upstream quiet since Dec 2024 ([`component-map-2026.md` §10.2–10.5](research/component-map-2026.md)). | Not in this shape. The surviving alternative — extractive sentence pruning — is already shipped as §10. If compression is ever profiled here, do it per model/hardware pair and re-run the ablation on every reader upgrade. |

### 13a. Implemented but measured out of the defaults (Phase 4, 2026-08-09)

These three shipped as flag-gated code, were benchmarked on/off, and stay OFF on
the numbers. The flags exist so the A/Bs can be re-run; the code is not dead, it
is *disabled by measurement*.

| Flag (both profiles) | Verdict | The number that decided it |
|---|---|---|
| `retrieval.document_escalation` (4.1) | **REJECTED as default** (was HOLD) | The 2026-08-09 lift (0/7→2/7 judged on fired queries) was the truncation bug, not document reassembly: the escalated block was appended at the tail and *survived* front-truncation while top-ranked chunks were discarded. The HOLD's condition — re-run after the context-window fix — was executed 2026-08-12 (`eval/decisions/phase4-escalation-rerun.md`): with prompts that fit (249 calls, zero truncation), the escalation-OFF baseline on the identical fire subset went 0/9 → 7/9 and escalation added nothing on top (5/7 → 2/7 mechanically, 5/7 → 5/7 by hand adjudication); both fires under the true product default were regressions on both dates. Flag and code kept — unfalsified for a corpus where document ordering matters within a fitting prompt; this one never presents that case. |
| `retrieval.crossref_hop` (4.2) | **REJECTED as a default** | Fires 0/11 at the shipped k=20; where forced (k=5), 0/11 hopped chunks hit an expected source — target selection is query-blind and lands on hub documents (21/24 hops to 2 documents) — and raising `k` beats the hop at equal context budget in 3 of 4 cells. Index-time `extract_crossrefs` stays ON (free, regex-only, bit-identical text/vectors). `eval/decisions/phase4-retrieval-benchmarks.md`, `phase4-answer-quality.md`. |
| `retrieval.overview_prefilter` (4.3) | **boost: HOLD · restrict: REJECTED** | boost: +0.106 nDCG@10 on the heterogeneous acquisition slice, −0.021 on `mixed` — a per-index opt-in candidate, not a default. restrict: removed the answer document entirely (recall@20 1→0) for 4 queries per corpus. `eval/decisions/phase4-retrieval-benchmarks.md`. |

## 14. How to re-litigate a decision

None of the above is permanent. The process for changing one:

1. **Reproduce the recorded number first.** The `docs` and `mixed` corpora index
   live `Documentation/*.md` content, so **editing documentation moves the
   metric.** Snapshot the pre-change tree and record the chunk count; only
   compare runs made against the same tree.

   ```bash
   .venv/bin/python eval/run_eval.py --corpus all \
     --json-out eval/results/before.json
   ```

2. **Run your change against the same corpus snapshot**, and run it twice if any
   arm makes an LLM call (the retry does), so the spread between runs is visible
   rather than mistaken for signal.

   ```bash
   .venv/bin/python eval/run_eval.py --corpus all \
     --json-out eval/results/after.json
   .venv/bin/python eval/smoke_e2e.py          # 25/25 assertions, exit 0
   ```

3. **Beat the recorded number by more than the noise floor.** On `mixed`, one
   query ≈ **0.014 nDCG@10**. Deltas under that are not findings. A delta that is
   positive on both corpora across multiple runs with zero per-query regressions
   is (that is the standard §5 was held to). Public-leaderboard deltas under ~2
   points are known not to transfer
   ([`component-map-2026.md` §3.6](research/component-map-2026.md)).

4. **Write it down.** Add the outcome and the numbers to
   [`eval/DECISIONS.md`](../eval/DECISIONS.md), with a page under
   [`eval/decisions/`](../eval/decisions/) if the investigation is substantial.
   Record the *rejections* too — the negative results in
   [`reranker.md`](../eval/decisions/reranker.md) and
   [`phase2-pipeline.md`](../eval/decisions/phase2-pipeline.md) are the most
   valuable pages in that directory.

5. **Land code, docs and the eval delta in the same change**, and update the
   relevant section here. Nothing in this file may describe unshipped behaviour —
   that is what [`research_roadmap.md`](research_roadmap.md) is for.

Two standing rules from the harness itself: never quietly repair a gold row to
make your own change look better (`docs_d10` was left failing and reported until
the gate re-anchored it, on the record), and never quote a leaderboard position in
this repo's docs — cite our own eval or nothing.

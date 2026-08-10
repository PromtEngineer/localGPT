# Phase 0 baseline — measured 2026-08-08

> **Historical.** This page describes the stack as it stood on 2026-08-08.
> The defaults it calls "shipped" (`Qwen/Qwen3-Embedding-4B`,
> `BAAI/bge-reranker-v2-m3`, reranking on) were replaced at the Phase 1
> adoption gate on 2026-08-09 — see [`DECISIONS.md`](DECISIONS.md). The
> measurements below are unchanged and still valid *for that configuration*;
> they are the baseline Phase 1 was measured against, not current behavior.

Every number on this page came from a run executed on this machine on
2026-08-08 (UTC 2026-08-09). Nothing here is estimated, extrapolated or copied
from a leaderboard. Where something could not be measured, it says so.

Raw outputs: `eval/results/baseline_rerank.json`, `eval/results/baseline_norerank.json`,
`eval/results/judge_v{1,2,3}_*.json` (all git-ignored — re-run the commands below
to regenerate them).

## Configuration under test

| | |
|---|---|
| Embedder | **`Qwen/Qwen3-Embedding-0.6B`** (1024-dim). The **shipped default is `Qwen/Qwen3-Embedding-4B`** (2560-dim); the 0.6B was used here because the 4B weights are not in the local HF cache (checked: `~/.cache/huggingface/hub/` holds `models--Qwen--Qwen3-Embedding-0.6B` and no 4B) and Phase 0 explicitly does not download new models. **These numbers are therefore not the shipped default's numbers.** |
| Reranker | `BAAI/bge-reranker-v2-m3`, loaded through the `rerankers` library as a cross-encoder — the shipped default |
| Generation model | `qwen3.5:9b` (smoke test only) |
| Utility model | `qwen3.5:4b` (gold-query generation, groundedness judge, verifier in the smoke test) |
| Profile | `PIPELINE_CONFIGS["default"]`, with **contextual enrichment OFF, document overviews OFF, late chunking OFF, context expansion OFF, decomposition OFF, verification OFF, synthesis skipped** (rationale in `README.md`) |
| Chunking | docling chunker, `chunk_size = 512` tokens (what the HTTP path sends) |
| Retrieval | `hybrid` (LanceDB FTS + vector, RRF-fused), `k = 20` first-stage candidates |
| Seeds | `random`, `numpy`, `torch` all seeded with `20260808`; corpora and queries iterated in sorted order |
| Machine | Apple M2 Max, 96 GB, macOS 15.5, torch 2.4.1 on **MPS**, Python 3.12.12 |
| Libraries | transformers 4.51.0, rerankers 0.10.0, lancedb 0.36.0, docling 2.118.1, Ollama 0.32.6 |

## Corpora and gold set

| Corpus | Source | Files | Chunks indexed | Gold queries |
|--------|--------|-------|----------------|--------------|
| `atlas7` | `eval/corpora/atlas7_service_manual.pdf` (planted facts, fictional) | 1 | **1** | 24 |
| `hr` | `eval/corpora/northwind_leave_policy.pdf` (synthetic, fictional) | 1 | **2** | 24 |
| `docs` | `Documentation/*.md` minus `improvement_plan.md` / `research_roadmap.md` | 12 | **313** | 24 |
| `mixed` | all of the above in one table | 14 | **316** | 72 |

**Gold set: 72 rows, all 72 verified, 0 discarded.** 72 (query, anchor) pairs were
generated from hand-authored dimension tuples, then checked one by one:

| Verification verdict | Count |
|---|---|
| accepted verbatim | 39 |
| rescued (model emitted `{"question": …}` or a truncated payload; question hand-written from the anchor) | 6 |
| rewritten (wrong premise, too vague to be answerable, or the query restated the expected string) | 27 |
| discarded as unanswerable | **0** |

Both automated gates passed with no exclusions:

* Gate 1 — `eval/corpora/verify_facts.py`: **76/76** planted-fact strings present in their source document.
* Gate 2 — reachability after conversion + chunking: **24/24, 24/24, 24/24, 72/72** rows reachable; `coverage_failures` is empty in both results files.

## Retrieval — first-stage recall and nDCG@10

The `--no-rerank` run and the reranked run produce **identical first-stage
numbers on all 144 query evaluations** (checked per query, not just in
aggregate): same seeds, same sorted iteration, same ranking twice. That is the
determinism check.

**First stage (recall is the headline metric here; nDCG@10 is over the
first-stage ordering):**

| Corpus | n | chunks | recall@5 | recall@10 | recall@20 | nDCG@10 |
|--------|---|--------|----------|-----------|-----------|---------|
| `atlas7` | 24 | 1 | 1.000 | 1.000 | 1.000 | 1.000 |
| `hr` | 24 | 2 | 1.000 | 1.000 | 1.000 | 0.954 |
| `docs` | 24 | 313 | 0.750 | 0.917 | 0.958 | 0.515 |
| **`mixed`** | **72** | **316** | **0.917** | **0.972** | **0.986** | **0.805** |

**After the `bge-reranker-v2-m3` cross-encoder** (same 20 candidates, reordered):

| Corpus | recall@5 | recall@10 | recall@20 | nDCG@10 | Δ nDCG@10 |
|--------|----------|-----------|-----------|---------|-----------|
| `atlas7` | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| `hr` | 1.000 | 1.000 | 1.000 | 1.000 | +0.046 |
| `docs` | 0.750 | 0.958 | 0.958 | 0.731 | **+0.216** |
| **`mixed`** | **0.917** | **0.986** | **0.986** | **0.903** | **+0.098** |

**Read `mixed` as the baseline.** `atlas7` and `hr` are 1 and 2 chunks, so `k=20`
sweeps the entire document and their recall is 1.0 by construction — those two
rows test the plumbing, not the retriever.

The reranker is where the measurable quality is: **+0.216 nDCG@10 on the only
corpus with real distractors**, and +0.098 on the mixed table. It also moves
recall, because recall@5 and recall@10 are prefixes of a list it reordered:
16 of the 144 evaluations change their recall vector after reranking (docs
recall@10 0.917 → 0.958, mixed 0.972 → 0.986). recall@20 is unchanged by
construction — reranking cannot add candidates, only reorder the 20 it was
given.

### By dimension (mixed + per-corpus runs pooled, n = 144 query evaluations)

| Slice | n | recall@10 | nDCG@10 after rerank |
|-------|---|-----------|----------------------|
| difficulty = easy | 88 | 0.955 | 0.916 |
| difficulty = hard | 56 | 1.000 | 0.891 |
| type = factoid | 72 | 0.972 | 0.936 |
| type = comparative | 18 | 0.889 | 0.955 |
| type = negative | 28 | 1.000 | 0.886 |
| type = procedural | 26 | 1.000 | 0.815 |

`hard` scoring above `easy` on recall is not a paradox to celebrate — with 144
evaluations the difference is 4 queries. Treat these slices as diagnostics, not
findings.

### Latency (per query, wall clock, MPS)

| Corpus | first stage mean | first stage p90 | rerank mean | rerank p90 |
|--------|-----------------|-----------------|-------------|------------|
| `atlas7` (1 chunk) | 158 ms | 210 ms | 176 ms | 107 ms |
| `hr` (2 chunks) | 125 ms | 132 ms | 293 ms | 216 ms |
| `docs` (313 chunks) | 142 ms | 150 ms | 1788 ms | 1783 ms |
| `mixed` (316 chunks) | 147 ms | 173 ms | 1741 ms | 1817 ms |

The `atlas7` and `hr` rerank means exceed their p90 because the very first
rerank call in the process pays the cross-encoder's `from_pretrained()` load
(~2.5 s) and drags the mean up. **The cross-encoder is ~12x the cost of the
whole first stage** on a 20-candidate list — that is the number Phase 1.1's
reranker A/B has to beat or justify.

### Where retrieval actually fails today

On the `docs` corpus, 8 of 24 gold queries scored below 0.5 nDCG@10 after
reranking and 1 more missed entirely. The same 9 reappear in `mixed`, joined by
one `atlas7` query the reranker demotes.

| Query | Symptom |
|-------|---------|
| `docs_d08` "What does the enricher do when the model returns an almost-empty summary?" | first-stage recall@5 = 0, @10 = 0, @20 = 1 — the answer-bearing chunk sits at rank 11–20. The reranker rescues it to rank 1–5, which is the single clearest case for keeping a cross-encoder |
| `docs_d16` "Does this project build an ANN index…" | recall = 0 at every k. It is a `match: "all"` comparative and only one of its two anchors is ever retrieved; nDCG@10 is 1.000 because the anchor it did find was ranked first |
| `docs_d07`, `d09`, `d13`, `d14`, `d15`, `d19`, `d21` | the answer-bearing chunk is retrieved but ranked 4th–10th, so nDCG@10 lands at 0.32–0.39 |

Three queries are **worse** after reranking than before it: `docs_d15`
(1.000 → 0.316), `docs_d09` (0.431 → 0.333), `docs_d21` (0.500 → 0.387), plus
`atlas7_a16` in the mixed table (1.000 → 0.431). On `docs` recall@5 the
cross-encoder promotes four queries into the top 5 (`d08`, `d10`, `d19`, `d22`)
and demotes four out of it (`d09`, `d14`, `d15`, `d17`) — a net zero at @5, and a
clear win at @10. It is a net win in aggregate and a loss on some paraphrased
queries: exactly the situation Phase 1.1's reranker A/B exists to arbitrate.

## Groundedness judge

`qwen3.5:4b`, `format="json"`, `think: false`, binary verdict. 20 hand-built
cases in `eval/judge_validation.jsonl`: 10 grounded, 10 subtly ungrounded (wrong
number, transposed part code, swapped entities, unsupported addition).

Three prompts were run. **v1 passed the gate on its first run**, so no iteration
was forced; v2 and v3 were then run as an ablation.

| Prompt | TP | FN | TN | FP | unparseable | TPR | TNR | agreement |
|--------|----|----|----|----|-------------|-----|-----|-----------|
| **v1 (default)** | **10** | **0** | **10** | **0** | **0** | **1.00** | **1.00** | **1.00** |
| v2 (stricter, claim-by-claim) | 5 | 5 | 10 | 0 | 0 | 0.50 | 1.00 | 0.75 |
| v3 (stricter + explicit procedure) | 10 | 0 | 10 | 0 | 0 | 1.00 | 1.00 | 1.00 |

**Gate (≥90% agreement): PASSED by v1 and v3. v2 fails at 75%.** v1 ships because
it ties v3 and is the shorter prompt. v1 was run twice (as `--prompt-version v1`
and again as the default after being promoted) and produced the identical
confusion matrix both times — the judge is nondeterministic in principle, but it
did not wobble on this set.

The honest caveat: 20/20 on 20 cases has a 95% Wilson lower bound of **0.839**
overall and **0.723** for TPR and TNR individually. This says the judge is not
badly broken; it does not say it is 100% accurate. v2's result is the useful
finding — telling a 4B model to be *more* rigorous made it reject five correct
answers (it faulted `g08_sabbatical` for quoting the approver's title, and
`g05_warranty_length` for paraphrasing "above 8 percent" as "more than 8
percent"). Before this judge gates anything in Phase 2, the validation set
should grow past 20.

## End-to-end smoke

`eval/smoke_e2e.py`, both services started as child processes against a temp
SQLite DB and a temp LanceDB directory, driven over HTTP, torn down afterwards.

**Result: 25/25 assertions passed, exit code 0, 243.7 s wall clock.**

| # | Assertion | Result |
|---|-----------|--------|
| 1 | both services became healthy (`:8001/health`, `:8000/health`) | PASS |
| 2 | `POST /indexes/<id>/upload` accepted the PDF | PASS |
| 3 | `POST /indexes/<id>/build` returned 200 with no `error` key | PASS |
| 4 | `POST /sessions/<sid>/indexes/<iid>` linked them | PASS |
| 5–8 | **q1** "What pressure does the brew boiler operate at during extraction?" → answer contains `9.2` · `source_documents` non-empty (1) · `[Confidence: 100%]` present · `message_count == 2` | PASS ×4 |
| 9–12 | **q2** "Which sensor part should be replaced when error code E11 appears?" → `TS-71` · 1 source · `[Confidence: 100%]` · `message_count == 4` | PASS ×4 |
| 13–16 | **q3** "How long is the Atlas-7 parts warranty?" → `36` · 1 source · `[Confidence: 100%]` · `message_count == 6` | PASS ×4 |
| 17–20 | **q4** "Where is the serial number engraved?" → `drip tray` · 1 source · `[Confidence: 100%]` · `message_count == 8` | PASS ×4 |
| 21 | `POST /sessions/<sid>/messages/save` returned 200 | PASS |
| 22 | the saved assistant message reads back out of SQLite | PASS |
| 23 | `source_documents` round-trip in `metadata.source_documents` | PASS |
| 24 | `steps` round-trip in `metadata.steps`, in order | PASS |
| 25 | `message_count == 10` after the saved turn | PASS |

Teardown removed both child processes (SIGTERM, exit `-15`), the uploaded file
the gateway wrote into `shared_uploads/`, and the temp directory.

All four verifier confidence tags came back at 100%. Do not read that as
calibration — the roadmap (2.4) already flags `[Confidence: N%]` as UX, not a
measurement, and 4 identical maxima on 4 easy questions is exactly what an
uncalibrated self-report looks like.

## Reproducing every number above

```bash
cd /path/to/localGPT

# gold-set gate 1
.venv/bin/python eval/corpora/verify_facts.py

# retrieval, with the cross-encoder (this also runs gate 2)
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B \
  .venv/bin/python eval/run_eval.py --corpus all \
  --json-out eval/results/baseline_rerank.json

# retrieval, first stage only
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B \
  .venv/bin/python eval/run_eval.py --corpus all --no-rerank \
  --json-out eval/results/baseline_norerank.json

# judge
.venv/bin/python eval/judge.py --validate --prompt-version v1
.venv/bin/python eval/judge.py --validate --prompt-version v2
.venv/bin/python eval/judge.py --validate --prompt-version v3

# end to end
.venv/bin/python eval/smoke_e2e.py
```

### Wall clock, as measured

| Step | Time |
|------|------|
| Full retrieval eval, cold (rebuilds the `docs` and `mixed` indexes: 36.3 s + 40.9 s) | **281.0 s** |
| Full retrieval eval, warm (cached indexes, first stage only) | **19.3 s** |
| Judge validation, one prompt version, 20 cases | **21 s** |
| End-to-end smoke, including both service starts and 4 generations on `qwen3.5:9b` | **243.7 s** |

Re-indexing is only triggered when a corpus file's size or mtime changes — the
fingerprint lives in `eval/.eval_indexes/<embedder>/<corpus>/eval_<corpus>.built.json`.

## What this baseline does not tell you

* Nothing about the **shipped 4B embedder**. Phase 1.2 must re-baseline before it
  can claim a delta, because these indexes are 1024-dim.
* Nothing about **answer quality end to end** beyond the 4 smoke questions. The
  retrieval metrics stop at the ranked chunk list; synthesis and verification are
  deliberately excluded.
* Nothing about **contextual enrichment**, which is on in the shipped `default`
  profile and off here. Turning it on changes the indexed text and would change
  every number in the retrieval table.
* Nothing about **latency under load**. The RAG API is a single-threaded
  `TCPServer`; everything above is single-user, single-request.

---

# Phase 4 baseline (pre-implementation) — measured 2026-08-09

The gate for [`Documentation/research_roadmap.md`](../Documentation/research_roadmap.md)
§ Phase 4 (mechanisms adopted from *agentic-file-search*). **Nothing in Phase 4
is implemented yet** — this section measures the stack *as it ships today* on a
corpus and gold set built specifically to expose what 4.1/4.2/4.3 are supposed
to fix, so the post-implementation run has something to beat.

Every number below came from a run executed on this machine on 2026-08-09.
Raw outputs (git-ignored, re-runnable — commands at the end of this section):
`eval/results/phase4_baseline_acq.json`,
`eval/results/phase4_baseline_acq_docs.json`,
`eval/results/phase4_baseline_acq_docs_k{5,3}.json`,
`eval/results/phase4_regression_mixed.json`.

## Configuration under test — the current shipped defaults

| | |
|---|---|
| Embedder | `microsoft/harrier-oss-v1-0.6b` (1024-dim, the shipped default since the Phase 1 adoption gate — [`DECISIONS.md`](DECISIONS.md)) |
| Reranker | **off**, matching the shipped `default` profile |
| Evidence-sufficiency retry (2.1) | **on** (`--retry profile`, `min_top_score 0.12`) |
| Profile | `PIPELINE_CONFIGS["default"]` with enrichment / overviews / late chunking / context expansion / decomposition / verification off, as everywhere else in this harness |
| Chunking | docling chunker, `chunk_size = 512` |
| Retrieval | `hybrid` (LanceDB FTS + vector, RRF-fused), `k = 20` unless a row says otherwise |
| Machine | Apple M2 Max, macOS (Darwin arm64), torch 2.4.1 on **MPS**, Python 3.12.12, lancedb 0.36.0, docling 2.118.1, transformers 4.51.0 |

## The new corpus: `acq`

`eval/corpora/acquisition/` — 10 interlinked synthetic M&A documents (TechCorp
acquires StartupXYZ), reused verbatim from the user's own
[PromtEngineer/agentic-file-search](https://github.com/PromtEngineer/agentic-file-search)
at `data/test_acquisition/`. They matter because they are the only corpus here
whose documents **reference each other**: 54 catalogued pointers of the form
`Document: <Title>`, `Exhibit A - Financial Terms`, `Schedule 1 - IP Assets`.
That link graph is what roadmap 4.2 (cross-reference hop) and 4.3 (overview
prefilter) act on, and neither planted-fact PDF nor `Documentation/*.md` has one.

| Corpus | Files | Chunks | Gold queries |
|--------|-------|--------|--------------|
| `acq` | 10 PDFs (20 pages) | **13** | 24 |
| `acq+docs` | those 10 + `Documentation/*.md` minus the two excluded files | **373** | 48 (24 `acq` + 24 `docs`) |

`eval/corpora/acquisition.facts.json` catalogues **100 planted facts** and the
**54 cross-references** (52 resolving inside the corpus, 2 deliberately dangling
— both files point at a "Document: Integration Plan" that does not exist).

**Gold set: 24 rows in `eval/goldset/acquisition.jsonl`, all 24 verified, 0 discarded.**
Hand-authored, not model-generated — 8 rows adapt questions from the source
repo's `TEST_QUESTIONS.md`, the other 16 are new. The rows carry the usual
`{topic, question_type, difficulty}` plus a new boolean **`requires_crossref`**:
true when the query's premise points at document A while the answer text lives
in document B, reachable from A only through an explicit reference.

| Composition | Count |
|---|---|
| `requires_crossref = true` | **11** |
| `requires_crossref = false` (control) | 13 |
| multi-document (`expected` spans ≥2 documents, `match: "all"`) | 4 |
| question type factoid / comparative / negative / procedural | 14 / 5 / 3 / 2 |
| difficulty easy / hard | 11 / 13 |

Per-row verification tally, all mechanical (re-run with
`.venv/bin/python eval/verify_crossref_goldset.py`), over the 31 `expected`
strings in the 24 rows:

| Check | Result |
|---|---|
| `expected` present in the document named in `expected_sources` | **31/31** |
| query does not contain its `expected` string verbatim (no leak) | **31/31** |
| `expected` occurs in **exactly one** document of the ten (so "answer lives in a different document" is a real claim) | **31/31** |
| `fact_ids` resolve to the sidecar, with matching text and source | **31/31** |
| `requires_crossref` / `multi_document` consistent with `anchor_doc` | **24/24** rows |
| Gate 1 (`verify_facts.py`), whole repo | **176/176** facts, **54/54** cross-reference cues |
| Gate 2 (reachability after conversion + chunking) | **24/24** on `acq`, **48/48** on `acq+docs`, `coverage_failures` empty |

## Results — first stage, shipped defaults

| Corpus | slice | n | chunks | recall@5 | recall@10 | recall@20 | nDCG@10 (1st) | 1st ms mean | 1st ms p90 |
|---|---|---|---|---|---|---|---|---|---|
| `acq` | all | 24 | 13 | 0.958 | 1.000 | 1.000 | 0.810 | 292 | 207 |
| `acq` | **`requires_crossref=true`** | 11 | 13 | **1.000** | **1.000** | **1.000** | **0.748** | | |
| `acq` | control (`=false`) | 13 | 13 | 0.923 | 1.000 | 1.000 | 0.863 | | |
| `acq+docs` | all | 48 | 373 | 0.917 | 0.958 | 1.000 | 0.738 | 309 | 997 |
| `acq+docs` | **`requires_crossref=true`** | 11 | 373 | **1.000** | **1.000** | **1.000** | **0.748** | | |
| `acq+docs` | control (`=false`) | 13 | 373 | 0.923 | 0.923 | 1.000 | 0.796 | | |
| `acq+docs` | *(the 24 `acq` rows alone)* | 24 | 373 | 0.958 | 0.958 | 1.000 | 0.774 | | |
| `acq+docs` | *(the 24 `docs` rows alone)* | 24 | 373 | 0.875 | 0.958 | 1.000 | 0.703 | | |
| `acq+docs` | multi-document rows | 4 | 373 | 1.000 | 1.000 | 1.000 | 0.748 | | |

By dimension, `acq+docs`, pooled (recall@10 / nDCG@10 first stage):

| Slice | n | recall@10 | nDCG@10 (1st) |
|---|---|---|---|
| `requires_crossref = true` | 11 | 1.000 | 0.748 |
| `requires_crossref = false` | 13 | 0.923 | 0.796 |
| difficulty = easy | 25 | 1.000 | 0.820 |
| difficulty = hard | 23 | 0.913 | 0.650 |
| type = factoid | 25 | 0.960 | 0.773 |
| type = comparative | 8 | 1.000 | 0.706 |
| type = negative | 9 | 1.000 | 0.739 |
| type = procedural | 6 | 0.833 | 0.637 |

The evidence-sufficiency retry fired on 1/24 `acq` queries (`acq_q12`, rewrite
not kept) and 8/48 `acq+docs` queries (5 rewrites kept).

## The crossref slice is **not** weak — and that is the finding

The expectation going in was that `requires_crossref=true` would be the visibly
broken slice. **It is not.** On both corpora the crossref rows hit
recall@5/@10/@20 = 1.000 — *better* than their own control (0.923). The only
gap is in ranking, and it is small: nDCG@10 0.748 vs 0.863 (`acq`) / 0.796
(`acq+docs`), i.e. the answer-bearing chunk typically lands at rank 2–3 instead
of rank 1. Seven of the eleven crossref rows score 0.500–0.631 (`q13`, `q23`,
`q14`, `q17`, `q18`, `q19`, `q22`); the other four score 1.000.

Tightening `k` does not reverse it either. Same corpus, same gold set, first
stage only:

| `acq+docs`, k = | crossref recall@k | control recall@k | crossref nDCG@10 | control nDCG@10 |
|---|---|---|---|---|
| 20 | 1.000 | 0.923 | 0.748 | 0.796 |
| 5 | 0.909 | 0.923 | 0.723 | 0.841 |
| 3 | **1.000** | **0.692** | **0.791** | 0.741 |

Three honest reasons this corpus does not reproduce the "cross-references are
invisible to embeddings" failure at the first stage, all of them properties of
the measurement rather than of the retriever:

1. **The deal room is 13 chunks.** With `k = 20` the first stage sweeps every
   chunk of every document, exactly the saturation caveat that already applies
   to `atlas7` and `hr`. `acq+docs` adds 360 distractor chunks, but they are
   localGPT documentation — topically disjoint from an M&A deal room, so they
   compete weakly.
2. **The references are echoed at both ends.** "Exhibit A - Financial Terms"
   appears in the Acquisition Agreement *and* in the Financial Adjustments Memo;
   "Document: Risk Assessment Memo" appears in seven files. The hybrid FTS leg
   therefore resolves most pointers lexically, without needing a hop.
3. **A crossref query still names its subject.** These queries state document
   A's premise *and* ask about B's subject matter, which is what an honest
   user question looks like — but it hands the retriever lexical signal from
   both ends of the reference.

What this means for Phase 4, stated as a limitation rather than a conclusion:
**first-stage recall on `acq` cannot by itself decide item 4.2.** 11 rows on a
13-chunk corpus is a small measurement, and the slice is already at ceiling.
When 4.2 lands, the comparison worth making is (a) nDCG@10 on the crossref
slice, which has 0.25 of headroom and is the number in the table above, and
(b) an end-to-end measurement of *which document gets cited*, which this
retrieval-only harness does not perform. A stricter reference-only gold set —
queries that name the pointer and nothing about the target's content — would be
the way to make the first-stage metric discriminative, and does not exist yet.

## Where retrieval actually fails on this corpus today

| Query | Symptom |
|---|---|
| `acq_q04` "What proportion of the target company's turnover comes from its single biggest client?" | recall = 0 at @5 and @10, 1 at @20 on `acq+docs`; nDCG@10 = 0.000. The only full miss. Fully paraphrased away from the document's vocabulary ("turnover"/"biggest client" vs "revenue"/"largest customer") — a query-understanding failure, not a cross-reference one |
| `acq_q12` "How large is the target's workforce…" | nDCG@10 = 0.316 (`acq`) / 0.431 (`acq+docs`); the one `acq` query the evidence-sufficiency retry fires on, and the rewrite was not kept |
| `acq_q01` "What is the total purchase price…" | nDCG@10 = 0.500 — the Financial Adjustments Memo's restated price outranks the Agreement's own definition |
| `acq_q13`, `q23`, `q14`, `q17`, `q18`, `q19`, `q22` | crossref rows whose answer chunk is retrieved but ranked 2nd–3rd (nDCG@10 0.500–0.631) |

## Regression check: the pre-existing corpora are untouched

Adding `acq` changed no shared code path — `corpus_files()` gained list-valued
globs (existing corpora pass a string), corpus keys are slugged for the
filesystem (no existing key contains `+`), and `by_dimension` skips rows without
a `requires_crossref` key, which is all of them outside `acq`. `mixed`
re-measured on this tree:

| | n | chunks | recall@5 | recall@10 | recall@20 | nDCG@10 (1st) |
|---|---|---|---|---|---|---|
| `mixed`, this run | 72 | **363** | 0.958 | 0.986 | 1.000 | **0.898** |
| `mixed`, `DECISIONS.md` §4 | 72 | 317 | 0.944 | 0.958 | 1.000 | 0.913 |

The two are not comparable and the difference is not a regression: `mixed`
contains live `Documentation/*.md`, which grew from 317 to 363 chunks between
the two runs. Only compare `mixed` runs made against the same tree.

## Reproducing every number in this section

```bash
cd /path/to/localGPT

# gate 1 — planted facts and cross-reference cues really are in the PDFs
.venv/bin/python eval/corpora/verify_facts.py

# row-level gate for the hand-authored gold set (the 31/31 tallies above)
.venv/bin/python eval/verify_crossref_goldset.py

# gate 2 only
.venv/bin/python eval/run_eval.py --corpus acq --coverage-only

# the two headline runs (shipped defaults: harrier, reranker off, retry on)
.venv/bin/python eval/run_eval.py --corpus acq \
  --json-out eval/results/phase4_baseline_acq.json
.venv/bin/python eval/run_eval.py --corpus acq+docs \
  --json-out eval/results/phase4_baseline_acq_docs.json

# the k sweep behind the crossref-vs-k table
.venv/bin/python eval/run_eval.py --corpus acq+docs --k 5 \
  --json-out eval/results/phase4_baseline_acq_docs_k5.json
.venv/bin/python eval/run_eval.py --corpus acq+docs --k 3 \
  --json-out eval/results/phase4_baseline_acq_docs_k3.json

# regression check on the tracked corpus
.venv/bin/python eval/run_eval.py --corpus mixed \
  --json-out eval/results/phase4_regression_mixed.json
```

| Step | Wall clock, as measured |
|---|---|
| `acq` index build, cold (10 PDFs → 13 chunks) | 9.9 s |
| `acq` eval, warm | 8.8 s |
| `acq+docs` eval, cold (373-chunk index build included) | 59.9 s |
| `mixed` eval, warm | 43.0 s |

## What this Phase 4 baseline does not tell you

* **Nothing about 4.1, 4.4, 4.5 or 4.6.** Document escalation is a synthesis-time
  behaviour, the filter DSL needs a `filters` argument that does not exist yet,
  token accounting is an SSE field, and `ask` mode is a CLI entry point. None of
  them is a first-stage retrieval metric, and none is measured here.
* **Nothing about 4.3 with overviews on.** This harness disables document
  overviews (one LLM call per document, and it changes indexed text). The
  overview-prefilter A/B will need that switched back on, which makes it the one
  Phase 4 item that cannot reuse these exact indexes.
* **Nothing about answer quality.** Same boundary as the rest of this page: the
  metrics stop at the ranked chunk list.
* **24 queries, one machine, one corpus of 13 chunks.** One query is 0.042 of
  any recall figure on `acq`. Treat every slice here as a diagnostic.

---

# Final-candidate-list metrics (added 2026-08-09)

Everything above this line scores the **first stage** —
`retrieve_candidates()["first_stage"]`. That is the right number for the
retriever, and it is the wrong number for roadmap item 4.2: the cross-reference
hop *appends* to `retrieve_candidates()["documents"]` and never mutates
`first_stage`, so a first-stage-only harness reports a flat line for the hop no
matter how well it works.

`eval/run_eval.py` now scores **both** lists on every query.

| metric family | source | meaning |
|---|---|---|
| `recall@k`, `ndcg@10_first_stage` | `out["first_stage"]` | the retriever's ordering. **Unchanged** — every number above this line still means what it meant. |
| `recall@k_final`, `ndcg@10_final` | `out["documents"]` | post-rerank **and** post-cross-reference-hop: the list the answer stage would see |
| `ndcg@10_reranked`, `recall_reranked` | `out["documents"]` minus the `via_crossref` rows | still post-rerank / **pre**-hop, so it stays comparable with every earlier decision file |

The results table prints the final family to the right of a `|` bar, plus a
`hop q` column (how many queries hopped). New CLI toggles, same shape as
`--retry`: `--crossref-hop {profile,on,off}` and
`--overview-prefilter {profile,off,boost,restrict}`; `profile` means whatever
`main.py` says, which is OFF for both today.

## The invariant

With reranking off and the hop off, `documents` **is** `first_stage`. Every run
checks this per query (chunk-id sequence, plus both metric families) and records
the verdict in the results JSON as `final_vs_first_stage_invariant`.

```
.venv/bin/python eval/run_eval.py --corpus mixed --retry off \
  --crossref-hop off --overview-prefilter off \
  --json-out eval/results/phase4_finalmetric_regression_mixed.json
```

```
corpus          n  chunks     R@5    R@10    R@20   nDCG@10   nDCG@10 |     R@5    R@10    R@20   nDCG@10  hop q   1st ms
                                                      (1st)  (rerank) |   (fin)   (fin)   (fin)   (final)
-------------------------------------------------------------------------------------------------------------------------
mixed          72     363   0.944   0.972   1.000     0.887       n/a |   0.944   0.972   1.000     0.887      0      124

invariant  ✅ final == first_stage on all 72 queries (rerank OFF, crossref hop OFF) — chunk-id order and both metrics
```

First stage identical to the tracked retry-off baseline (0.944 / 0.972 / 1.000,
nDCG@10 first-stage 0.887); final equals first-stage on all 72 queries.

## First 4.2 A/B — measured, and negative

> **Superseded 2026-08-09 by the rebuilt-index runs at the bottom of this page.**
> Every arm in this subsection ran against indexes built *before*
> `rag_system/indexing/crossref.py` learned numeric-prefix-stripped filename
> aliases, so **none of the acquisition corpus's references resolved** and the hop
> could not fire on it. The subsection is kept because it is the measurement that
> found the resolver bug; it is not current evidence about the hop.

All arms `--retry off`, reranker off, cached indexes (`acq` 13 chunks,
`acq+docs` 373 chunks), so the two arms of each pair differ only in the flag.

| corpus | k | arm | slice | n | nDCG@10 (1st) | **nDCG@10 (final)** | queries that hopped |
|---|---|---|---|---|---|---|---|
| `acq` | 20 | hop off | all | 24 | 0.8101 | **0.8101** | 0 |
| `acq` | 20 | hop **on** | all | 24 | 0.8101 | **0.8101** | 0 |
| `acq` | 20 | hop off | `requires_crossref=true` | 11 | 0.7477 | **0.7477** | 0 |
| `acq` | 20 | hop **on** | `requires_crossref=true` | 11 | 0.7477 | **0.7477** | 0 |
| `acq+docs` | 20 | hop off | all | 48 | 0.7194 | **0.7194** | 0 |
| `acq+docs` | 20 | hop **on** | all | 48 | 0.7194 | **0.7194** | **7** |
| `acq+docs` | 20 | hop off | `requires_crossref=true` | 11 | 0.7477 | **0.7477** | 0 |
| `acq+docs` | 20 | hop **on** | `requires_crossref=true` | 11 | 0.7477 | **0.7477** | 0 |

Recall is identical across every pair as well (`acq` 0.958 / 1.000 / 1.000;
`acq+docs` 0.854 / 0.896 / 0.958), first-stage and final alike. `--k 5` and
`--k 3` were also run on `acq`, both arms, and still fired zero hops.

Why: **none of the acquisition corpus's 34 extracted cross-references resolve to
a target document** (`exhibit a`, `schedule 1`, `section 4.1` — the Exhibits and
Schedules are sections *inside* `01_acquisition_agreement.pdf`, and resolution is
filename-based). The 7 hops on `acq+docs` all come from `Documentation/*.md`
title matches, and `hit_expected_source = 0`, `hopped_chunk_relevant = 0` — the
hop pulled nothing gold. Full analysis and the raw index dump:
[`decisions/phase4-eval-final-metric.md`](decisions/phase4-eval-final-metric.md).

## Reproducing this subsection

```bash
cd /path/to/localGPT

# regression + invariant on the tracked corpus
.venv/bin/python eval/run_eval.py --corpus mixed --retry off \
  --crossref-hop off --overview-prefilter off \
  --json-out eval/results/phase4_finalmetric_regression_mixed.json

# the 4.2 A/B (each pair differs only in --crossref-hop)
.venv/bin/python eval/run_eval.py --corpus acq --retry off --crossref-hop off \
  --overview-prefilter off --json-out eval/results/phase4_42_acq_hop_off.json
.venv/bin/python eval/run_eval.py --corpus acq --retry off --crossref-hop on \
  --overview-prefilter off --json-out eval/results/phase4_42_acq_hop_on.json
.venv/bin/python eval/run_eval.py --corpus acq+docs --retry off --crossref-hop off \
  --overview-prefilter off --json-out eval/results/phase4_42_acqdocs_hop_off.json
.venv/bin/python eval/run_eval.py --corpus acq+docs --retry off --crossref-hop on \
  --overview-prefilter off --json-out eval/results/phase4_42_acqdocs_hop_on.json

# the k sweep that shows low k does not rescue the hop on `acq`
for k in 5 3; do for arm in off on; do
  .venv/bin/python eval/run_eval.py --corpus acq --retry off --crossref-hop $arm \
    --overview-prefilter off --k $k \
    --json-out eval/results/phase4_42_acq_k${k}_hop_${arm}.json
done; done
```

**Determinism protocol** for any comparison run: pass `--retry off` (the retry is
an LLM reformulation and is nondeterministic), never pass an empty-string env var
(`EMBEDDING_MODEL=` breaks the run), and do not edit anything under
`Documentation/` between arms — the docs corpus is live in `mixed`, `docs` and
`acq+docs`, and a doc diff moves the numbers.

---

# Rebuilt-index baselines — measured 2026-08-09 (supersede the two rows above)

## Why the rebuild

Cross-references are stamped into chunk metadata **at index time**. The
`acq` and `acq_plus_docs` eval indexes were built before
`rag_system/indexing/crossref.py` existed, and certainly before the gate's
**resolver fix** — each known document is now additionally registered under its
numeric-prefix-stripped name, so `08_regulatory_approval.pdf` also answers to
`"regulatory approval"`, which is how the acquisition PDFs actually refer to each
other (`phase4-crossref-prefilter.md` § *Gate correction (2026-08-09)*). Both
index directories were deleted and rebuilt on 2026-08-09.

## What the rebuild produced

Read out of the built LanceDB tables (`metadata` → `["metadata"]["crossrefs"]`),
not from the build log:

| index | chunks | chunks with crossrefs | refs | **resolved** | documents linked | self-edges |
|---|---|---|---|---|---|---|
| `acq` | 13 | 11 | 68 | **34** | **9 of 10** | none |
| `acq+docs` | 373 | 70 | 215 | **93** | 21 | none |

Previously: **0 resolved on `acq`**. The 34 that still do not resolve are the
Exhibits and Schedules — sections *inside* `01_acquisition_agreement.pdf`, not
separate files — plus bare `section N` forms. A filename-based resolver cannot
reach them and correctly leaves them `target_doc: null`.

## Re-baseline — zero drift

`--retry off`, reranker off, hop off, prefilter off, k = 20. Gold coverage
24/24 and 48/48, `coverage_failures` empty, chunk counts unchanged (13 / 373),
`final == first_stage` invariant ✅ on all 72 queries across the two runs.

| corpus | slice | n | chunks | recall@5 | recall@10 | recall@20 | nDCG@10 (1st) | vs. previous retry-off figure |
|---|---|---|---|---|---|---|---|---|
| `acq` | all | 24 | 13 | 0.958 | 1.000 | 1.000 | **0.8101** | identical |
| `acq` | `requires_crossref=true` | 11 | 13 | 1.000 | 1.000 | 1.000 | **0.7477** | identical |
| `acq` | control (`=false`) | 13 | 13 | 0.923 | 1.000 | 1.000 | **0.8628** | identical |
| `acq+docs` | all | 48 | 373 | 0.854 | 0.896 | 0.958 | **0.7194** | identical |
| `acq+docs` | `requires_crossref=true` | 11 | 373 | 1.000 | 1.000 | 1.000 | **0.7477** | identical |
| `acq+docs` | control (`=false`) | 13 | 373 | 0.769 | 0.769 | 0.846 | **0.7731** | identical |

Identical to four decimals on every cell, which is the expected result and the
point of running it: extraction writes chunk *metadata* only — `text` and
`vector` are untouched — so adding 34 resolved references cannot move a
first-stage ranking. These are the numbers any future 4.2/4.3 arm must be
compared against.

## Headline of the re-measured A/B (full matrix in the decision file)

| | |
|---|---|
| Hop fires now | 0 → **24/24** queries on `acq` at k=3 and k=5; **33/48** on `acq+docs` |
| At the shipped **k = 20** | **inert**: 0 hops on `acq` (13-chunk corpus < candidate budget), 14 hops on `acq+docs` with every metric bit-identical |
| On the `requires_crossref` slice | **0/11 hops hit a gold source document at any k on either corpus; no metric moved** |
| Where it gains | only the `requires_crossref=false` control slice at k=3/k=5 (e.g. `acq` k=3 recall@10 final 0.692 → 0.846) |
| Budget-matched vs. simply raising `k` | wins **1 of 4** cells |
| Harm | none — the hop only appends, recall never fell |

## 4.3 is now measurable — `--overviews on`

The overview prefilter reads an `.npz` sidecar that only an overview-enabled
build writes, and this harness had overviews hard-off. `eval/run_eval.py` gained
**`--overviews {off,on}`** (default `off`, unchanged behaviour): `on` enables
document overviews + the embedded sidecar and redirects `overview_path` into the
corpus's own index directory (`<corpus>_ov`), so the sidecar is owned by the
index and the repo's shared `index_store/overviews/` is never written. Verified
chunk-for-chunk identical to a normal build (373/373 chunk ids, text identical,
max abs vector delta **0.0**).

| corpus | arm | recall@5 | recall@10 | recall@20 | nDCG@10 (1st) |
|---|---|---|---|---|---|
| `acq+docs` | off | 0.854 | 0.896 | 0.958 | **0.7194** |
| `acq+docs` | boost | 0.812 | 0.917 | 0.958 | **0.7017** |
| `acq+docs` | restrict | 0.812 | 0.854 | **0.896** | **0.6951** |
| `mixed` | off | 0.944 | 0.972 | 1.000 | **0.8873** |
| `mixed` | boost | 0.889 | 0.972 | 1.000 | **0.8662** |
| `mixed` | restrict | 0.889 | 0.931 | **0.944** | **0.8740** |

`boost` is a large gain on the heterogeneous slice (`acq` control rows of
`acq+docs`: nDCG@10 0.7731 → 0.8790) and a loss on `mixed`, where twelve of
fifteen documents are localGPT documentation and the overviews carry no
discriminating signal. **`restrict` loses four queries their answer document
entirely on each corpus** (recall@20 1 → 0) — the harm check the previous
decision file asked for.

Caveat specific to 4.3: overview text is LLM-generated. All three arms of each
comparison read the same sidecar, so each comparison is exact, but a rebuild of
`<corpus>_ov` will produce different overviews and can move these numbers with no
code change.

Full matrix, hop-precision columns, budget-matched controls, per-query harm
traces and the proposed adopt/reject/hold calls:
[`decisions/phase4-retrieval-benchmarks.md`](decisions/phase4-retrieval-benchmarks.md).

## Reproducing this section

```bash
cd /path/to/localGPT

rm -rf eval/.eval_indexes/microsoft__harrier-oss-v1-0.6b/acq \
       eval/.eval_indexes/microsoft__harrier-oss-v1-0.6b/acq_plus_docs

# rebuild + re-baseline (these are the two hop-off k=20 arms)
.venv/bin/python eval/run_eval.py --corpus acq --retry off --crossref-hop off \
  --overview-prefilter off \
  --json-out eval/results/phase4_w3_rebuild_acq_hop_off.json
.venv/bin/python eval/run_eval.py --corpus acq+docs --retry off --crossref-hop off \
  --overview-prefilter off \
  --json-out eval/results/phase4_w3_rebuild_acqdocs_hop_off.json

# 4.3 arms (the first run per corpus builds the _ov index: 23 / 15 LLM calls)
for m in off boost restrict; do
  .venv/bin/python eval/run_eval.py --corpus acq+docs --overviews on --retry off \
    --crossref-hop off --overview-prefilter $m \
    --json-out eval/results/phase4_w3_43_acqdocs_ov_${m}.json
  .venv/bin/python eval/run_eval.py --corpus mixed --overviews on --retry off \
    --crossref-hop off --overview-prefilter $m \
    --json-out eval/results/phase4_w3_43_mixed_ov_${m}.json
done
```

The full 4.2 k-sweep and the budget-matched controls are in
[`decisions/phase4-retrieval-benchmarks.md`](decisions/phase4-retrieval-benchmarks.md) § 7.

**Latency**: not reported for any run in this section. A concurrent agent shared
the Ollama instance throughout, so every wall-clock figure is contended.

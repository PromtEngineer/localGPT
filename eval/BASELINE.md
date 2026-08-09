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

# Phase 2 pipeline-shape items — 2.5, 2.1, 2.2, 2.4

_Investigated and shipped 2026-08-09._

The four items this page covers, from
[`Documentation/research_roadmap.md`](../../Documentation/research_roadmap.md) §Phase 2:

| # | Item | Outcome |
|---|------|---------|
| 2.5 | Delete the graph module | **Removed.** Code, config keys, two dependencies, and every doc section |
| 2.1 | Evidence-sufficiency retry | **Shipped ON** in `default`, off in `fast`. Fires on 9.7–11.1% of `mixed`, +0.008 to +0.017 nDCG@10 and +0.014 recall@10, zero per-query regressions across four runs |
| 2.2 | Decomposition at rerank | **Shape change shipped; sub-query scoring measured NEGATIVE (−0.046 `max`, −0.012 `mean`) on the 6 affected queries and is enabled by no shipped profile.** First stage no longer fans out over sub-queries except behind `compose_from_sub_answers` |
| 2.4 | Verifier model seam | **Seam shipped, default unchanged.** ThinknCheck has no public weights; two suitable substitutes were found, wired and smoke-tested |

Every number below was produced by running `eval/run_eval.py` on this tree. Where
a measurement is missing or was not affordable, this page says so rather than
estimating.

---

## 0. The baseline, and an honest note about corpus drift

The `docs` and `mixed` corpora index live `Documentation/*.md`, so **this work's
own documentation edits changed the corpus underneath the metric.** Every run
below therefore names its chunk count, and only same-chunk-count runs are
compared to each other.

| Run | chunks (`mixed`) | mixed nDCG@10 | mixed r@10 | docs nDCG@10 | docs r@10 |
|---|---|---|---|---|---|
| Pre-change tree (snapshotted first, `phase2_before.json`) | 317 | 0.913 | 0.958 | 0.753 | 0.875 |
| Settled tree, retry **off** (`phase2_final_retry_off.json`) | 331 | 0.888 | 0.958 | 0.680 | 0.875 |
| Settled tree, shipped defaults, run 1 | 331 | 0.896 | 0.958 | 0.735 | 0.917 |
| **Settled tree, shipped defaults, run 2** (`phase2_after.json`) | 331 | **0.901** | **0.972** | **0.735** | **0.917** |

Two shipped-defaults runs are listed because the retry makes one LLM call when it
fires, so the arm is not bit-reproducible. The spread (0.896–0.901 on `mixed`) is
the size of that nondeterminism; the firing *set* is identical between them.

The pre-change 0.913 reproduces `DECISIONS.md` §4 exactly, so the snapshot is
sound. The 0.913 → 0.888 gap is **entirely corpus-side**, and that is checkable
rather than asserted: with the retry off, the first-stage code path is unchanged,
and of the 72 `mixed` queries **only 11 moved at all — every one of them
docs-anchored, zero PDF-anchored**. The graph-removal rewrites plus the new
retry/decomposition/verifier sections added 14 chunks of fresh distractor prose to
a 317-chunk corpus.

### One gold row was orphaned by item 2.5

`docs_d10` asks *"How many model calls does knowledge-graph extraction spend on
each chunk?"* and anchors on the string `"makes two LLM calls per chunk"`, which
lived **only** in the `indexing_pipeline.md` knowledge-graph section that item 2.5
deleted. The row is now structurally unanswerable. The harness reports it as a
coverage failure on both corpora rather than hiding it, which is Gate 2 working as
designed — but it also scores 0 by construction and drags every future run down by
~0.014 on `mixed` and ~0.042 on `docs`.

**It is left in place, not quietly edited** — `eval/goldset/` is not this work's to
change, and silently repairing a gold row to make one's own change look better is
exactly the failure mode the honesty rule exists to prevent. Excluding it:

| Run (mixed, n=71, docs_d10 excluded) | nDCG@10 | recall@10 |
|---|---|---|
| Pre-change tree | 0.9114 | 0.9577 |
| Settled tree, retry off | 0.9006 | 0.9718 |
| Settled tree, shipped defaults, run 1 | 0.9086 | 0.9718 |
| **Settled tree, shipped defaults, run 2** | **0.9141** | **0.9859** |

| Run (docs, n=23, docs_d10 excluded) | nDCG@10 | recall@10 |
|---|---|---|
| Pre-change tree | 0.7427 | 0.8696 |
| Settled tree, retry off | 0.7092 | 0.9130 |
| **Settled tree, shipped defaults** | **0.7668** | **0.9565** |

So the shipped stack scores **0.9086–0.9141 on `mixed` against a pre-change 0.9114**
— i.e. it straddles the baseline, well inside a noise floor of one query ≈ 0.014,
while carrying 14 extra distractor chunks. Recall@10 is up in both runs
(0.9577 → 0.9718/0.9859), and `docs` **beats** the pre-change baseline outright
(+0.024 nDCG@10, +0.087 recall@10). **Nothing regressed.**

---

## 1. Item 2.5 — the graph module is gone

Deleted: `rag_system/indexing/graph_extractor.py`; `GraphRetriever`
(`retrieval/retrievers.py`); `GraphQueryTranslator` (`retrieval/query_transformer.py`);
the extraction block and `networkx` import in `pipelines/indexing_pipeline.py`; the
`graph_strategy` constructor branch, `_run_graph_query()` and the `graph_query`
routing branch in `agent/loop.py`; and the `retrieval.graph` keys in `main.py`.

`networkx`, `fuzzywuzzy` and `python-Levenshtein` had no other consumer once
`GraphRetriever` went (verified by grep across `*.py`) and were removed from
`requirements.txt`, `requirements-docker.txt` and `rag_system/requirements.txt`.

**Triage is now two-way.** `Agent._normalize_triage()` runs on every verdict from
both routers and maps anything that is not an explicit `direct_answer` to
`rag_query`, so a small utility model that still emits the retired `graph_query`
label lands on the RAG path instead of a `hasattr` check that no longer exists.

**Evidence** (`Documentation/research/academic-evidence-2026.md` §6): GraphRAG
*loses* on single-hop retrieval (64.78 vs 63.01 F1 on NQ; 60.92% vs 60.14% on
GraphRAG-Bench); its multi-hop gains span +3 to +27 points depending entirely on
how well the vector baseline is tuned; and it costs **41–57× at indexing**
(135s → 5,560–7,702s) and up to **~377× in query tokens** (879 → 331,375 prompt
tokens/query for MS-GraphRAG global). It was also unreachable — no shipped profile
ever set `graph_strategy`.

Docs updated in the same change: `retrieval_pipeline.md`, `indexing_pipeline.md`,
`system_overview.md`, `architecture_overview.md`, `triage_system.md`,
`prompt_inventory.md`, `verifier.md`, `docker_usage.md`, `rag_system/README.md`,
`rag_system/DOCUMENTATION.md`, `README.md`, `DOCKER_README.md`.

`improvement_plan.md` §9's graph bullet resolves to **removed** (proposed row in §5).

---

## 2. Item 2.1 — evidence-sufficiency retry

### 2.1 The signal, and the one that did not work

The roadmap says to trigger on the top score. Measured on the gold set, **the raw
top cosine similarity is anti-correlated with success and is unusable**:

| `mixed`, top cosine | value |
|---|---|
| The 3 first-stage misses | 0.609, 0.629, 0.642 |
| Successful queries | min 0.441, p25 0.544, **median 0.576**, max 0.753 |

All three failures score *above* the median success. Any threshold catching all
three fires on **94% of successful queries**. Absolute similarity mostly encodes how
close a query's phrasing sits to the corpus register, not whether the answer was
found. RRF scores are worse still — every query's top RRF is 0.031–0.033.

What does carry signal is **contrast** — how far the best candidate stands above the
background of everything else the query pulled in:

```
evidence = (cos_top − cos_background) / (1 − cos_background)
```

with `cos_background` the mean cosine of candidates from rank 6 down, and
`cos = 1 − _distance/2` on the L2-normalized v4 tables. The denominator rescales
against this query's reachable headroom, keeping the score in 0–1 and comparable
across queries.

### 2.2 Calibration

| threshold | `mixed` fails caught | `mixed` successes fired | `mixed` fire rate |
|---|---|---|---|
| 0.10 | 1/3 | 3/69 (4.3%) | 5.6% |
| **0.12 ← shipped** | **1/3** | **5/69 (7.2%)** | **8.3%** |
| 0.14 | 1/3 | 10/69 (14.5%) | 15.3% |
| 0.20 | 3/3 | 18/69 (26.1%) | 29.2% |

**The brief's target — fire on the genuine failures without firing on >10% of
successes — is not achievable on this gold set, and that is a finding, not a tuning
failure.** Catching all three misses costs 26% false-fire on `mixed` and 67% on
`docs`. `0.12` was chosen as the largest threshold that respects the ≤10% budget;
it catches one of the three real misses.

### 2.3 Measured effect

Retry off vs on, same tree, same corpus snapshot. Two on-runs, because the
reformulation is an LLM call:

| Arm | mixed nDCG@10 | mixed r@10 | docs nDCG@10 | docs r@10 | mixed fired | docs fired |
|---|---|---|---|---|---|---|
| off (mid-work tree, 313 ch) | 0.8889 | 0.9583 | 0.6781 | 0.8750 | — | — |
| on, run 1 (same tree) | 0.9011 | 0.9722 | 0.7096 | 0.8750 | 7/72 (9.7%) | 5/24 (20.8%) |
| on, run 2 (same tree) | 0.9063 | 0.9722 | 0.7304 | 0.9167 | 7/72 (9.7%) | 5/24 (20.8%) |
| off (settled tree, 331 ch) | 0.8881 | 0.9583 | 0.6796 | 0.8750 | — | — |
| on, run 1 (settled tree) | 0.8960 | 0.9583 | 0.7348 | 0.9167 | 7/72 (9.7%) | 5/24 (20.8%) |
| on, run 2 (settled tree) | 0.9014 | 0.9722 | 0.7348 | 0.9167 | 8/72 (11.1%) | 6/24 (25.0%) |

Compare only within a chunk count. Both tree snapshots give the same verdict: the
retry is positive on both corpora and on both metrics, four runs out of four.

* **Firing rate: 9.7% on `mixed`** on the tree the threshold was calibrated
  against — inside the ≤10% budget — drifting to 11.1% on the settled tree as the
  corpus grew. That drift is the honest caveat on the threshold: it is a property
  of the corpus, not a constant.
* The firing *set* is deterministic for a given corpus; only the rewrites vary,
  which is why the two runs on each tree differ.
* **Zero per-query regressions on `mixed`** in any run. Three queries improved.
* It repaired `docs_d16` (*"Does this project build an approximate-nearest-neighbour
  index…"*), a genuine recall@10 = 0 miss, by rewriting it to *"approximate nearest
  neighbor (ANN) data structure implementation and vector search execution…"*.
* Cost: one enrichment-model call plus one extra retrieval, on ~10% of queries.
  `mixed` mean latency 96 ms → 186 ms averaged over all queries.

**Verdict: ship enabled in `default`, disabled in `fast`.** The delta is positive on
both corpora and no query got worse. It is small — +0.008 to +0.017 nDCG@10 on
`mixed` — and it is bought with an LLM call, so it belongs in the quality profile
and not in the speed one.

Caveats worth carrying: `atlas7` and `hr` are 1- and 2-chunk tables, so the
background term is degenerate there and `hr` fires on 62.5% of queries. It is
harmless (the better result set is kept either way) but those rows measure nothing.
The retry is inert on legacy unnormalized tables and in `fts_only` mode, by design —
no signal, no retry.

---

## 3. Item 2.2 — decomposition at rerank, not first stage

### 3.1 What was happening before

Sub-query fan-out **did** happen at the first stage. `Agent._run_async` submitted
one full `RetrievalPipeline.run()` per sub-query to a 3-worker pool, for both the
`compose_from_sub_answers` path and the aggregate path.

### 3.2 What runs now

| `query_decomposition` | reranker | First stage | Rerank scored against |
|---|---|---|---|
| off | either | once, full query | full query |
| on, `compose_from_sub_answers: true` (profile default) | either | **once per sub-query**, parallel | that sub-query |
| on, `compose_from_sub_answers: false` | on | once, full query | **all sub-queries, aggregated** |
| on, `compose_from_sub_answers: false` | off | once, full query | — (no rerank stage; sub-queries unused) |
| on, one sub-query after decomposition | either | once, the resolved query | the resolved query |

The `compose_from_sub_answers` path keeps first-stage fan-out **behind its existing
flag**, as the brief allows: it needs a separate *answer* per sub-question to
compose from, which one shared candidate set cannot produce. Everything else now
retrieves once on the full original query.

### 3.3 Measurement

`docs` (24 queries), `Qwen/Qwen3-Reranker-4B`, retry off so it does not confound
the arms. `mixed` with the reranker on was **not run**: at ~12.5 s/query it is
~15 minutes per arm × 3 arms, and the brief explicitly allows skipping it.

| Arm | first-stage nDCG@10 | **post-rerank nDCG@10** | rerank ms/query |
|---|---|---|---|
| decompose off | 0.6796 | **0.8377** | 12,471 |
| decompose on, `max` | 0.6796 | 0.8415 | 16,984 |
| decompose on, `mean` | 0.6796 | **0.8499** | 16,755 |

**The first-stage number is byte-identical across all three arms**, which is the
structural check that matters: decomposition provably never touches the first
stage any more. A fourth arm — decompose on with the **reranker off** — reproduced
the off arm exactly (0.680, no rerank stage), confirming the documented no-op.

### The headline number is confounded; the honest split is worse

Only **6 of 24** queries decompose into more than one sub-query. The other 18
return a single (pronoun-resolved) sub-query, and those are *not* testing
decomposition at all — they are testing "rerank against the decomposer's rewrite
of the query". Splitting them:

| Subset | off | `max` | `mean` |
|---|---|---|---|
| All 24 | 0.8377 | 0.8415 | 0.8499 |
| **The 6 genuinely decomposed** | **0.8862** | **0.8406 (−0.046)** | **0.8740 (−0.012)** |
| The 18 single-sub-query (rewrite only) | 0.8215 | 0.8418 (+0.020) | 0.8418 (+0.020) |

**On the queries decomposition actually affects, scoring against sub-queries at
rerank is negative under both aggregates.** Two queries carry it: `docs_d14`
(0.500 → 0.431 under `max`) and `docs_d17` (0.818 → 0.613 under both). The whole-
corpus "gain" comes entirely from the single-sub-query rows, where the win is
query *rewriting*, not decomposition — and even there only 2 of 18 rows moved.

### Verdict

* **The shape change ships.** First-stage retrieval always uses the full original
  query. This is the part the evidence supports, and it is a strict reduction in
  work: the aggregate path used to issue N first-stage retrievals and now issues
  one.
* **Sub-query scoring at rerank is not switched on anywhere by default.** The
  `default` profile ships `compose_from_sub_answers: true`, which never reaches
  the aggregation path. Nothing in a shipped profile enables it.
* **`mean` is the default aggregate**, because it beat `max` on every subset
  measured (−0.012 vs −0.046 where it matters). Less bad, not good.
* n_effective = 6 queries on one corpus. This measurement is too small to call
  the 2026 MultiConIR/SSRB finding wrong; it is big enough to say **it did not
  reproduce here**, so nothing was turned on because of it.

---

## 4. Item 2.4 — verifier model seam

### 4.1 Availability, checked 2026-08-09 against the HuggingFace Hub API

| Candidate | Verdict |
|---|---|
| **ThinknCheck** (arXiv 2604.01652, UPenn; 1B, 78.1 BAcc on LLMAggreFact) | **No public weights.** The paper is real and checks out, but a Hub search for `thinkncheck` returns **zero** models and the paper links no release. **Cannot be wired.** |
| `ibm-granite/granite-guardian-3.3-8b` | Exists, Apache-2.0. **8B / ~16 GB** — an order of magnitude over the "small local verifier" budget. |
| `ibm-granite/granite-guardian-hap-38m` | Exists, 38M, Apache-2.0 — but it is a **hate/abuse/profanity RoBERTa classifier**. Wrong task entirely: it does not score answer-vs-evidence. |
| `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` | ✅ MIT, **369 MB**, no custom code. Generic NLI. |
| `lytang/MiniCheck-DeBERTa-v3-Large` | ✅ MIT, **1.74 GB**, no custom code. Purpose-built grounded claim verification — the baseline ThinknCheck itself benchmarks against. |
| `vectara/hallucination_evaluation_model` (HHEM-2.1-open) | Apache-2.0, 438 MB, but ships custom modelling code — gated behind `VERIFIER_TRUST_REMOTE_CODE=1`. |

So the roadmap's two named candidates both fail — one has no weights, the other is
either too big or the wrong task — but **two suitable substitutes do exist**, so the
seam was wired *and* exercised rather than left as a stub.

### 4.2 What shipped

`VERIFIER_MODEL` env var / `verification.model` config key, plus
`verification.threshold` (default 0.5). Unset ⇒ the LLM-prompt verifier, unchanged.
Set ⇒ `LocalNLIVerifier` loads the model lazily on first use, splits the answer into
sentences, scores each against the retrieved evidence as premise, and takes the
**minimum** — one unsupported sentence makes the answer ungrounded, matching the
binary semantics `eval/judge.py` already uses.

A model that cannot be loaded **raises**, printing the table above, rather than
falling back to the LLM prompt. A verifier that silently is not the verifier you
configured is worse than an error.

### 4.3 Smoke test on `judge_validation.jsonl`

The brief asked for a 5-case smoke test. Both models passed 5/5 on a balanced
5-case sample, so all 20 hand-labelled cases were run — it costs a minute once the
weights are cached and 5 cases cannot distinguish the two:

| Verifier | agreement | TPR (grounded, n=10) | TNR (ungrounded, n=10) | notes |
|---|---|---|---|---|
| `lytang/MiniCheck-DeBERTa-v3-Large` (1.74 GB) | **19/20** | **10/10** | 9/10 | one false *positive*: `u03_boilers_swapped` scored 63% — it did not notice the two boilers' pressures had been swapped |
| `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` (369 MB) | 18/20 | 8/10 | **10/10** | two false *negatives*: `g07_sick_pay` (22%) and `g10_contractors` (2%) — both grounded answers it called unsupported |

The two fail in opposite directions, which is the useful result: MiniCheck is the
more permissive of the pair and misses a swapped-entity error, the generic NLI model
is stricter and rejects two correct answers. Neither is a drop-in improvement on the
LLM-prompt verifier without its own validation run — `eval/judge.py --validate`
reports TPR/TNR for the judge, and the same discipline should apply before any of
these becomes a default. **Nothing was made the default here.**

One practical note: `MiniCheck-DeBERTa-v3-Large` ships only `pytorch_model.bin` (no
safetensors) and its 1.74 GB blob stalled twice on first download in this
environment before completing; the 369 MB model is the faster thing to try first.

Reproduce:

```bash
VERIFIER_MODEL=MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli \
  .venv/bin/python -c "
from rag_system.agent.verifier import LocalNLIVerifier; import json, os
v = LocalNLIVerifier(os.environ['VERIFIER_MODEL'])
for r in (json.loads(l) for l in open('eval/judge_validation.jsonl')):
    print(r['id'], r['label_grounded'], v.verify(r['question'], chr(10).join(r['evidence']), r['answer']).is_grounded)
"
```

`[Confidence: N%]` remains **UX, not a calibrated measurement**, and
`Documentation/verifier.md` now says so in a callout. Changing the backend changes
where the number comes from; it does not calibrate it.

---

## 5. Proposed `improvement_plan.md` Landed rows

For the gate to graduate — this page does not edit `improvement_plan.md` or
`research_roadmap.md`.

| Area | Change | Verify at |
|------|--------|-----------|
| Retrieval | **2.5 Graph module removed** — `GraphExtractor`, `GraphRetriever`, `GraphQueryTranslator`, the `graph_query` triage outcome and the `retrieval.graph` / `graph_strategy` config keys are gone; `networkx`, `fuzzywuzzy` and `python-Levenshtein` dropped from all three requirements files. Contested gains, 41–57× indexing and up to ~377× query-token cost (`Documentation/research/academic-evidence-2026.md` §6) | `rag_system/indexing/` has no `graph_extractor.py`; `requirements.txt`; `eval/decisions/phase2-pipeline.md` §1 |
| Retrieval | **2.1 Evidence-sufficiency retry** — one conditional second retrieval on weak evidence, on in `default` and off in `fast`, triggered by candidate-set *contrast* rather than raw top similarity (which measured anti-correlated with success). Fires on 9.7–11.1% of `mixed`, +0.008–0.017 nDCG@10 and +0.014 recall@10, zero per-query regressions across four runs | `rag_system/pipelines/retrieval_pipeline.py::retrieve_candidates`, `rag_system/main.py` `retrieval.retry`; numbers in `eval/decisions/phase2-pipeline.md` §2 |
| Retrieval | **2.2 Decomposition applied at rerank** — the first stage always runs once on the full original query; sub-queries score candidates at the rerank stage, aggregated by `query_decomposition.rerank_aggregate`. First-stage fan-out survives only behind the pre-existing `compose_from_sub_answers` flag | `rag_system/pipelines/retrieval_pipeline.py::_rerank_stage`, `rag_system/agent/loop.py`; numbers in `eval/decisions/phase2-pipeline.md` §3 |
| Verification | **2.4 Verifier model seam** — `VERIFIER_MODEL` / `verification.model` swaps the LLM-prompt verifier for a local NLI/verifier model; default unchanged. ThinknCheck has no public weights and Granite Guardian is either 8B or the wrong task, so two verified substitutes were wired and smoke-tested instead | `rag_system/agent/verifier.py::LocalNLIVerifier`, `Documentation/verifier.md`; numbers in `eval/decisions/phase2-pipeline.md` §4 |
| Eval | `eval/run_eval.py` now drives `RetrievalPipeline.retrieve_candidates()` instead of calling the retriever directly, so first stage, rerank and retry are all the shipped code path; `--retry`, `--decompose` and `--aggregate` added | `eval/run_eval.py`, `eval/README.md` |
| Hygiene | §9's Graph-RAG bullet ("finish it or delete it") resolves to **deleted** | `eval/decisions/phase2-pipeline.md` §1 |

### Not resolved here

`docs_d10` in `eval/goldset/docs.jsonl` is orphaned by item 2.5 and needs retiring or
re-anchoring by whoever owns the gold set (§0).

---

## 6. Limits of this evidence

* **72 English queries on one laptop.** One query is ~0.014 nDCG@10 on `mixed`. The
  retry's +0.008 is *inside* that noise floor on `mixed`; the reason to ship it is
  that it is positive on both corpora across three runs with zero regressions, not
  that any single delta is significant.
* **The retry's calibration set is the same 72 queries it is evaluated on.** With
  three first-stage misses to calibrate against, the threshold is fitted to a
  handful of points. Treat 0.12 as a starting value, not a tuned constant.
* **`atlas7` and `hr` are 1- and 2-chunk tables.** Their contrast scores are
  meaningless and their rows are plumbing checks, not measurements.
* **Latency numbers are from a shared GPU** on an M2 Max and are indicative only.
* **Nothing here measures answer quality.** These are retrieval metrics; the
  verifier smoke test is 20 hand-labelled cases, which is a small sample.


---

**Gate resolution (2026-08-09):** the orphaned gold row `docs_d10` flagged above was
re-anchored at the validation gate (embedder-identity-guard prose, topic
`graph` → `index_safety`, recorded in the row's `verification` field). The eval
numbers in this file predate that repair. Also fixed at the gate: the OCR probe's
stale `rapidocr_onnxruntime` module name (Q4 of the GLM-OCR spike), an explicit
`lang=['english']` (+ `OCR_LANG` env) for RapidOCR replacing docling's
`['chinese']` default, and `ocrmac` installed so macOS resolves `OcrMacOptions`.

# Phase 4 item 4.2 — making the cross-reference hop measurable, and the first A/B

Date: 2026-08-09
Status: **measurement infrastructure landed in `eval/run_eval.py`; the 4.2 A/B was
run and is a NEGATIVE result — the hop fires zero times on the corpus built to
exercise it.** No adopt/reject call is made here; that is the gate's.
Scope of this wave: `eval/run_eval.py`, `eval/BASELINE.md`, this file. No
`Documentation/` diffs, no `rag_system/` edits.

---

## 1. The problem this wave was given

`eval/run_eval.py` called `pipeline.retrieve_candidates(...)` and scored
`out["first_stage"]` (plus `ndcg10_reranked` off the reranked list). The
cross-reference hop
(`rag_system/pipelines/retrieval_pipeline.py::_crossref_hop`) appends its hopped
chunks to `result["documents"]` and **deliberately never mutates
`first_stage`** — the previous decision file
([`phase4-crossref-prefilter.md`](phase4-crossref-prefilter.md) §5) says so in
as many words: *"The harness will therefore report a flat line for 4.2 no matter
how well it works."*

## 2. What changed in `eval/run_eval.py`

### 2.1 The final candidate list is now scored

Every query is scored twice, against two lists:

| metric family | source | meaning |
|---|---|---|
| `recall` / `ndcg10_first_stage` | `out["first_stage"]` | the retriever's own ordering (unchanged; every historical number in `BASELINE.md` still means the same thing) |
| `recall_final` / `ndcg10_final` | `out["documents"]` | **post-rerank AND post-hop** — the list the answer stage would actually see |

Summary keys: `recall@{5,10,20}_final`, `ndcg@10_final`, present on the whole-corpus
summary, on the `crossref` / `crossref_control` slices, and in `by_dimension`
(`recall@10_final`, `ndcg@10_final`, `queries_with_crossref_hop`).

`ndcg10_reranked` / `recall_reranked` were **kept meaning post-rerank, pre-hop**.
The hop only ever appends, so dropping the `via_crossref`-tagged rows
reconstructs the pre-hop list exactly; every earlier decision file's
`ndcg@10_reranked` stays comparable.

### 2.2 The invariant, checked rather than argued

With reranking off and the hop off, `documents` **is** `first_stage` (the same
list object). Every run now records `final_equals_first_stage` per query
(chunk-id sequence equality) and prints a run-level verdict, also written to the
results JSON as `final_vs_first_stage_invariant`. If the final metrics ever drift
from the first-stage metrics on a run where nothing may reorder or append, the
metric is measuring its own bug.

### 2.3 Hop instrumentation — precision, not just rank movement

Per query, when anything was hopped:

* `crossref_chunks_in_final` — how many `via_crossref` chunks are in `documents`
* `crossref_documents` — which documents they came from
* `crossref_hit_expected_source` — did a hop land in a document named in the gold
  row's `expected_sources`? (document-level precision)
* `crossref_chunk_relevant` — does a hopped chunk actually contain the gold
  `expected` text? (text-level precision)
* `first_relevant_rank_final`, and the raw `crossref_hop` record from the pipeline

Aggregated per corpus and per slice as `crossref_hop: {queries_with_hop,
fire_rate, chunks_added_total, chunks_added_mean_when_fired, hit_expected_source,
hopped_chunk_relevant}`.

### 2.4 CLI toggles for the Phase-4 flags

Following the existing `--retry {profile,on,off}` pattern, via a new
`apply_phase4_settings()` that writes `retrieval.crossref_hop` /
`retrieval.overview_prefilter` the same way `apply_retry_setting` writes
`retrieval.retry`:

```
--crossref-hop {profile,on,off}
--overview-prefilter {profile,off,boost,restrict}
```

`profile` = whatever `main.py` says (both are OFF there today). Both states are
echoed in the run header and recorded in the results JSON (`run.crossref_hop`,
`run.overview_prefilter` and their config blocks).

---

## 3. Verification

Determinism protocol for everything below: `--retry off`, no empty-string env
vars, no `Documentation/` edits between runs. All indexes were reused from cache
(`acq` 13 chunks, `acq+docs` 373 chunks, `mixed` 363 chunks), so every arm below
is a code-to-code comparison on identical bytes.

### 3.1 Regression — `mixed`, all Phase-4 flags off

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

First stage is **identical** to the tracked baseline (mixed, 363 chunks: 0.944 /
0.972 / 1.000, nDCG@10 first-stage 0.887 — `phase4-crossref-prefilter.md` §2, the
retry-off arm). Final metrics equal first-stage metrics on all 72 queries.

### 3.2 The 4.2 A/B — `acq`, hop off vs hop on

```
.venv/bin/python eval/run_eval.py --corpus acq --retry off --crossref-hop off \
  --overview-prefilter off --json-out eval/results/phase4_42_acq_hop_off.json
.venv/bin/python eval/run_eval.py --corpus acq --retry off --crossref-hop on \
  --overview-prefilter off --json-out eval/results/phase4_42_acq_hop_on.json
```

| arm | slice | n | R@5 | R@10 | R@20 | nDCG@10 (1st) | R@5 (fin) | R@10 (fin) | R@20 (fin) | **nDCG@10 (final)** | queries that hopped |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hop **off** | all | 24 | 0.958 | 1.000 | 1.000 | 0.8101 | 0.958 | 1.000 | 1.000 | **0.8101** | 0 |
| hop **off** | `requires_crossref=true` | 11 | 1.000 | 1.000 | 1.000 | 0.7477 | 1.000 | 1.000 | 1.000 | **0.7477** | 0 |
| hop **off** | control (`=false`) | 13 | 0.923 | 1.000 | 1.000 | 0.8628 | 0.923 | 1.000 | 1.000 | **0.8628** | 0 |
| hop **on** | all | 24 | 0.958 | 1.000 | 1.000 | 0.8101 | 0.958 | 1.000 | 1.000 | **0.8101** | **0** |
| hop **on** | `requires_crossref=true` | 11 | 1.000 | 1.000 | 1.000 | 0.7477 | 1.000 | 1.000 | 1.000 | **0.7477** | **0** |
| hop **on** | control (`=false`) | 13 | 0.923 | 1.000 | 1.000 | 0.8628 | 0.923 | 1.000 | 1.000 | **0.8628** | **0** |

First stage is bit-identical across arms, as designed. **Final metrics are also
identical, because the hop fired on 0 of 24 queries.** Two independent reasons,
both verified:

1. **`acq` is 13 chunks.** At `k = 20` every query retrieves all 13, so all ten
   documents are already `represented` and the hop's "not already a candidate"
   guard suppresses every target. Confirmed: `candidates` = 13 and
   `final_candidates` = 13 on all 24 queries.
2. **More fundamental — none of `acq`'s cross-references resolve.** Read straight
   out of the built index:

   ```
   acq index: 13 chunks, 11 chunks carrying crossrefs, 34 refs
   total refs 34 resolved 0
   top unresolved on acq: [('exhibit','exhibit c') x6, ('exhibit','schedule 3') x5,
     ('exhibit','schedule 1') x5, ('exhibit','exhibit a') x4, ('exhibit','exhibit b') x4,
     ('exhibit','schedule 2') x4, ('section','section 2.2') x2, ('section','section 1.5'),
     ('section','section 4.1'), ('section','section 1.1'), ('section','section 4')]
   ```

   Reason 2 is why lowering `k` does not rescue it. Both `--k 5` and `--k 3` were
   run on both arms (`phase4_42_acq_k{5,3}_hop_{off,on}.json`) — at `k = 3` only
   3 of 13 chunks are candidates, so most documents are unrepresented, and the hop
   *still* fired 0 times. Every number is identical across arms:

   | arm | slice | n | R@5 | nDCG@10 (1st) | R@5 (fin) | nDCG@10 (final) | hops |
   |---|---|---|---|---|---|---|---|
   | k=5 hop off | all | 24 | 0.917 | 0.7875 | 0.917 | 0.7875 | 0 |
   | k=5 hop on | all | 24 | 0.917 | 0.7875 | 0.917 | 0.7875 | 0 |
   | k=5 hop off | xref | 11 | 1.000 | 0.7358 | 1.000 | 0.7358 | 0 |
   | k=5 hop on | xref | 11 | 1.000 | 0.7358 | 1.000 | 0.7358 | 0 |
   | k=3 hop off | all | 24 | 0.792 | 0.8036 | 0.792 | 0.8036 | 0 |
   | k=3 hop on | all | 24 | 0.792 | 0.8036 | 0.792 | 0.8036 | 0 |
   | k=3 hop off | xref | 11 | 0.909 | 0.7868 | 0.909 | 0.7868 | 0 |
   | k=3 hop on | xref | 11 | 0.909 | 0.7868 | 0.909 | 0.7868 | 0 |

### 3.3 `acq+docs`, hop off vs hop on

```
.venv/bin/python eval/run_eval.py --corpus acq+docs --retry off --crossref-hop off \
  --overview-prefilter off --json-out eval/results/phase4_42_acqdocs_hop_off.json
.venv/bin/python eval/run_eval.py --corpus acq+docs --retry off --crossref-hop on \
  --overview-prefilter off --json-out eval/results/phase4_42_acqdocs_hop_on.json
```

| arm | slice | n | R@5 | R@10 | R@20 | nDCG@10 (1st) | R@5 (fin) | R@10 (fin) | R@20 (fin) | **nDCG@10 (final)** | queries that hopped |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hop **off** | all | 48 | 0.854 | 0.896 | 0.958 | 0.7194 | 0.854 | 0.896 | 0.958 | **0.7194** | 0 |
| hop **off** | `requires_crossref=true` | 11 | 1.000 | 1.000 | 1.000 | 0.7477 | 1.000 | 1.000 | 1.000 | **0.7477** | 0 |
| hop **off** | control (`=false`) | 13 | 0.769 | 0.769 | 0.846 | 0.7731 | 0.769 | 0.769 | 0.846 | **0.7731** | 0 |
| hop **on** | all | 48 | 0.854 | 0.896 | 0.958 | 0.7194 | 0.854 | 0.896 | 0.958 | **0.7194** | **7** |
| hop **on** | `requires_crossref=true` | 11 | 1.000 | 1.000 | 1.000 | 0.7477 | 1.000 | 1.000 | 1.000 | **0.7477** | **0** |
| hop **on** | control (`=false`) | 13 | 0.769 | 0.769 | 0.846 | 0.7731 | 0.769 | 0.769 | 0.846 | **0.7731** | **1** |

Here the hop *does* fire — 7 of 48 queries, 21 chunks added — and the final
metrics still do not move. Hop precision, verbatim from the results JSON:

```
summary crossref_hop: {
  "queries_with_hop": 7,
  "fire_rate": 0.1458,
  "chunks_added_total": 21,
  "chunks_added_mean_when_fired": 3.0,
  "hit_expected_source": 0,
  "hopped_chunk_relevant": 0
}
```

**0 of 7 hops landed in a gold source document, and 0 of 21 hopped chunks carried
gold text.** All 7 hops are `kind: "document"` title matches *between
`Documentation/*.md` files* (`triage_system.md → retrieval_pipeline.md`,
`prompt_inventory.md → verifier.md`, `architecture_overview.md →
indexing_pipeline.md`, …) — i.e. they come from the distractor corpus, not from
the acquisition deal room. Not one hop originated on an `acq` PDF, consistent
with §3.2's "0 of 34 acq references resolve".

Both `requires_crossref` rows and the 4.2 slice therefore hopped **zero** times
on `acq+docs` too: those queries already retrieve their answer document inside
the top 20 (`recall@10 = 1.000` on the slice), so the "not already represented"
guard is correct to suppress the hop — there is nothing to fetch.

`nDCG@10 (final)` is unchanged to 4 decimals in every arm above. The one query
whose final list grew and whose score could have moved, `acq_q12`, was already at
`nDCG@10 = 0.000` and the hop pulled `verifier.md`, which is unrelated: 0.000 →
0.000. **The hop neither helped nor hurt any measured number.**

### 3.4 Sanity: hopped chunks really are in `documents`, tagged

Direct pipeline call, `acq+docs`, `--crossref-hop on`, query `docs_d03`:

```
--- Performing hybrid retrieval for query: 'How many overviews does the overview router use?' on table 'eval_acq_plus_docs' ---
Retrieved 20 documents.
🔗 Cross-reference hop: pulled 3 chunk(s) from 1 referenced document(s) (retrieval_pipeline.md).
first_stage: 20  documents: 23
via_crossref chunks: 3
{
  "chunk_id": "retrieval_pipeline.md_7",
  "document_id": "retrieval_pipeline.md",
  "via_crossref": true,
  "crossref": {
    "kind": "document",
    "ref": "retrieval pipeline",
    "from_chunk_id": "triage_system.md_2",
    "from_document_id": "triage_system.md"
  },
  "text_head": "7. Semantic cache ( agent/loop.py:130-154, 305-324, 587-594 ) Owned by Agent , not by the pipeline:  TTLCache(maxsize=100, ttl=300) ( loop.py:33 ) keyed by raw "
}
first_stage carries via_crossref?: False
```

20 first-stage candidates → 23 final, 3 tagged `via_crossref: true`, and
`first_stage` untouched. Exactly the contract the pipeline documents.

### 3.5 The rerank path still works, and `reranked` vs `final` line up

The `rerank_error` detection had to change from list identity to element
identity (the hop rebuilds `documents` as a new list), so the rerank path was
re-run rather than assumed:

```
.venv/bin/python eval/run_eval.py --corpus acq --retry off --crossref-hop on \
  --overview-prefilter off --reranker BAAI/bge-reranker-v2-m3 \
  --json-out eval/results/phase4_42_acq_hop_on_rerank.json
```

```
  acq — roadmap 4.2 slice:
    requires_crossref=false    n=13  recall@10=1.000 nDCG@10(1st)=0.863 | recall@10(fin)=1.000 nDCG@10(fin)=0.879 hops=0
    requires_crossref=true     n=11  recall@10=1.000 nDCG@10(1st)=0.748 | recall@10(fin)=1.000 nDCG@10(fin)=0.822 hops=0

rerank_error: 0
reranked==final on all: True
summary: ndcg@10_first_stage 0.8101   ndcg@10_reranked 0.8531   ndcg@10_final 0.8531
```

`ndcg@10_reranked == ndcg@10_final` on every query, which is the correct
relationship when zero hops fired, and no query reported `rerank_error`.
(This arm is *not* part of the 4.2 A/B — reranking is off in the shipped profile;
it is here only to prove the final metric composes with the rerank stage.)

`--coverage-only` was also re-run (`acq`: `gold coverage 24/24 rows reachable`),
since the invariant check had to be skipped on that path.

---

## 4. What the numbers say, and what they do not

**The measurement infrastructure works** — §3.4 shows the hop's output reaching
the scored list, §3.1 shows the metric is inert when nothing may change it, and
§3.3 shows the instrumentation catching a firing hop and correctly scoring it as
useless.

**The 4.2 A/B is a negative result: no lift, no harm, because the hop cannot
reach the cases it was built for.** The blocker is not the hop, it is
index-time resolution. `rag_system/indexing/crossref.py::normalize_name` reduces
`08_regulatory_approval.pdf` to the literal string `08 regulatory approval`, and
the `document` family only matches when that whole string appears as whole words
in a chunk. The acquisition PDFs reference each other as *titles* ("Regulatory
Approval Documentation"), never with the numeric filename prefix, so the
`document` family matches nothing in that corpus. What the extractor *does* find
there — `exhibit a`, `schedule 1`, `section 4.1` — cannot resolve either, because
the Exhibits and Schedules are **sections inside** `01_acquisition_agreement.pdf`,
not separate files, and resolution is filename-based.

Fixing that is an `rag_system/indexing/crossref.py` change (title-aware
resolution, or a numeric-prefix strip in `normalize_name`), which this wave does
not own and does not touch. Recording it as the finding is the deliverable.

## 5. Caveats, stated plainly

* **n = 11** on the `requires_crossref` slice, and it is at ceiling on recall
  (1.000 at @5 on both `acq` and `acq+docs`). One query is 0.09 of any slice
  figure. It cannot support a fine-grained adopt/reject call even if the hop
  had fired.
* **`acq` is 13 chunks.** At the shipped `k = 20` the corpus is smaller than the
  candidate budget, so *no* candidate-selection change can move a metric on it.
  Any future 4.2 measurement needs either a bigger corpus or a smaller `k`, and
  `--k 3` was tried here and still could not fire the hop.
* **The `acq+docs` full-set figures in this file (0.854 / 0.896 / 0.958,
  nDCG@10 1st 0.7194) are lower than `BASELINE.md`'s Phase 4 baseline row
  (0.917 / 0.958 / 1.000, 0.738).** That is the retry, not a regression: the
  baseline row was run at `--retry profile` (on), everything here is `--retry
  off` for determinism. Compare arms within this file, and compare the `mixed`
  regression against the retry-off baseline only.
* **Eval nondeterminism**: with `--retry off`, no reranker and no decomposition,
  this harness makes no LLM call on the query path, and all arms reused cached
  indexes — so the arms here are deterministic. Runs at `--retry profile` are
  not (LLM query reformulation); the 0.887–0.903 spread documented in
  `phase4-crossref-prefilter.md` §2 is that effect.
* **Retrieval only.** No answer synthesis, no citation check. A hop that puts the
  right document in the context but does not change `nDCG@10` would still be
  invisible here — the honest form of the "which document gets cited" gap
  `BASELINE.md` already names.
* **`--overview-prefilter` is wired but unmeasured.** 4.3 needs document
  overviews switched on at index time (an LLM call per document, which this
  harness disables), so no 4.3 arm was run. The toggle exists so Wave 3 can run
  one without a code edit.

## 6. Files touched

* `eval/run_eval.py` — final-list metrics, invariant check, hop instrumentation,
  `--crossref-hop` / `--overview-prefilter`.
* `eval/BASELINE.md` — new subsection *Final-candidate-list metrics*.
* `eval/decisions/phase4-eval-final-metric.md` — this file.

Nothing under `rag_system/` or `Documentation/` was modified.

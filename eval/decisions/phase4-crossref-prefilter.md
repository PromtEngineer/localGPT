# Phase 4 items 4.2 and 4.3 — cross-reference hop and overview prefilter

Date: 2026-08-09
Status: **implemented, flag-gated, both defaults OFF, not yet benchmarked.**
Owner of this wave: retrieval/indexing pipelines only. `main.py` and
`Documentation/*.md` were deliberately **not** edited — the config keys and doc
diffs this change needs are *proposed* at the bottom of this file and belong to
the adoption gate.

There is **no gold-set evidence for or against either feature yet**. The mixed
gold set has no cross-reference queries and no multi-document overview queries
(roadmap Phase 4: "4.2 and 4.3 need multi-document gold queries with
cross-references added to eval/goldset"). Everything below is a mechanism proof
plus a regression proof that the mechanisms are inert while switched off. Do not
turn either flag on in a shipped profile on the strength of this document.

---

## 1. What shipped

### 4.2 Cross-reference hop

**Index time** — `rag_system/indexing/crossref.py` (new),
`rag_system/pipelines/indexing_pipeline.py`.

Deterministic regexes over each chunk's *original* text (the pass runs after
chunking and **before** contextual enrichment, so an LLM-written preamble can
neither invent nor swallow a reference). Three reference families:

| kind | matched | example `ref` |
|------|---------|---------------|
| `exhibit` | `exhibit / appendix / schedule / annex / attachment / addendum` + a single capital letter or a dotted number, optional `No.` / `#` | `exhibit b`, `schedule 2.1`, `appendix a` |
| `section` | `section / clause / article` + a dotted number, and the `§` symbol form | `section 4.3`, `section 7` |
| `document` | the normalized filename/title of *another* document in the resolution set, appearing as whole words in the chunk | `northwind leave policy` |

Stored as `metadata.crossrefs = [{"kind", "ref", "target_doc"}]`. Capped at 8
distinct references per chunk (more than that means a table of contents).
Chunks with no references get **no key at all**, so metadata does not grow for
corpora that have none.

Resolution is name-based only: a reference resolves when some document's
normalized filename/title contains it as whole words (`"Exhibit B"` →
`exhibit_b.pdf`). The resolution set is the current indexing batch **plus** the
document ids already in the target LanceDB table, so an incremental add can
still point at a document indexed last week (best effort; any failure reading
the table silently falls back to batch-only with one log line). Unresolvable
references are still recorded with `target_doc: null` — they are true, a UI can
show them, they just have nowhere to hop to.

**Never resolves to the chunk's own document.** Otherwise `exhibit_b.pdf` saying
"this Exhibit B" self-resolves on every chunk, and a document that repeats its
own title mints a useless reference per chunk.

Config: `indexing.extract_crossrefs`, default **TRUE**. Extraction is a handful
of regexes over text already in memory — no LLM, no second pass — and it only
writes chunk metadata. The `text` and `vector` columns are unchanged, which is
why it is safe on by default while the query-time hop is not.

**Query time** — `rag_system/pipelines/retrieval_pipeline.py`.

`retrieval.crossref_hop = {"enabled": false, "max_hops": 1, "chunks_per_hop": 3}`.

After the candidate set is final (first stage + rerank + the evidence-sufficiency
retry), each of the **top 3** candidates is inspected for a `crossrefs` entry
whose `target_doc` is a document **not already represented** in the candidates.
Up to `max_hops` such documents are expanded: a dense, document-filtered LanceDB
search pulls that document's `chunks_per_hop` most on-topic chunks, and they are
appended to `documents` tagged `via_crossref: true` (both top-level and inside
`metadata`, plus a `crossref` record naming the source chunk and the reference).

Bounds, all hard: only the top 3 candidates can trigger a hop; hopped chunks can
never trigger another one (no recursion); no LLM anywhere in the path; the
worst case is `max_hops` extra filtered vector searches per query.

`result["crossref_hop"]` carries the hop record, and an event `crossref_hop` is
emitted so the UI/citations can show it. `first_stage` is **never** mutated —
with reranking off, `documents` and `first_stage` are the same list object, so
the hop rebuilds `documents` as a new list.

Two downstream interactions were fixed in `run()`:

* Context expansion re-reads the row from LanceDB, which knows nothing about how
  the chunk arrived — the `via_crossref` marker is now carried over onto the
  central chunk the same way `rerank_score` already was.
* The "hide non-reranked chunks" filter would otherwise delete every hopped
  chunk whenever the reranker is on, since hops are appended *after* reranking
  by design. Hopped chunks are now exempt from that filter. (Scoring them with
  the reranker would defeat the point: the referenced document is precisely the
  one whose text does not look like the query.)

### 4.3 Overview prefilter

**Index time** — `rag_system/indexing/overview_builder.py`,
`rag_system/pipelines/indexing_pipeline.py`.

At the end of an index build, every overview in
`index_store/overviews/<index_id>.jsonl` is embedded with the **document-side**
embedder (no instruction prefix — same asymmetry as the chunk index) and written
to a sidecar:

```
index_store/overviews/<index_id>.jsonl          the overviews (unchanged)
index_store/overviews/<index_id>.vectors.npz    doc_ids + L2-normalized vectors + {embedding_model, normalized}
```

Rebuilt wholesale rather than appended, because the JSONL is append-only and a
re-indexed document has several lines of which only the last is current. Cost is
one embedding per *document*. Failures print a warning and never fail the build.
Config: `overview.embed`, default TRUE (only reachable when `overview.enabled`).

It is a sidecar and not a LanceDB table because it is one row per document (tens,
not thousands), it is rebuilt wholesale rather than queried, and a missing
sidecar has to be a graceful no-op rather than a schema problem.

**Query time** —
`retrieval.overview_prefilter = {"enabled": false, "top_documents": 5, "mode": "boost"}`.

The query vector is scored against the overview vectors (cosine; both sides
L2-normalized) and the top `top_documents` documents are selected. Then:

* `mode: "boost"` (the default when enabled — the safer of the two): the
  candidate ordering is fused with the document-overview ordering by **RRF at the
  same `_RRF_K = 60` the retriever uses**. A rank bonus, not a score bonus, and
  no weight knob — for the reason design_rationale §4 gives for the BM25/dense
  fusion: the two orderings are not on a common scale and there is no validation
  split here to tune a weight against. Nothing is dropped; documents outside the
  top-N simply contribute no second leg.
* `mode: "restrict"`: the first stage itself runs with a LanceDB
  `document_id IN (…)` prefilter. If the restricted search returns nothing, it
  logs and falls back to unrestricted retrieval for that query.

Both are computed **inside `_first_stage`**, not around it, so the
evidence-sufficiency retry re-scores documents against its reformulated query
too.

Sidecar path resolution, most explicit first:
`retrieval.overview_prefilter.vectors_path` → `config["overview_path"]` with
`.jsonl` swapped for `.vectors.npz` → `index_store/overviews/<index_id>.vectors.npz`.
Nothing beyond that is guessed: silently prefiltering against *some other
index's* overviews would be worse than not prefiltering. `api_server.py` already
sets `rp_cfg["overview_path"]` per session (`:367`), so the HTTP path needs no
new plumbing for resolution — only the flag.

**Graceful degradation** (all one log line, then normal retrieval): no overview
path configured; sidecar file absent; sidecar unreadable; sidecar written by a
different embedding model than the pipeline is configured for.

### Shared primitive

`RetrievalPipeline._search_within_documents()` — a document-filtered search that
mirrors `MultiVectorRetriever.retrieve` exactly (same prefiltered FTS and vector
legs, same RRF at `_RRF_K = 60`, same output row shape). It lives in the
retrieval pipeline rather than in `retrievers.py` because `retrieve()` has no
filter parameter and that module is owned elsewhere this wave. **If a filtered
variant is ever added to `MultiVectorRetriever`, this helper should be deleted
in favour of it** — it is duplication, and it is only justified by the ownership
boundary.

`retrieve_candidates` returns from five places (the retry has four early outs).
All five now route through `_post_candidates()`, a tail hook that runs whatever
must see a *final* candidate set. If another change needs the same, add it there
rather than to the five return sites.

---

## 2. Verification

### `py_compile`

Clean on `rag_system/indexing/crossref.py`,
`rag_system/indexing/overview_builder.py`,
`rag_system/pipelines/indexing_pipeline.py`,
`rag_system/pipelines/retrieval_pipeline.py`.

### Regression, both flags OFF — `--corpus mixed`

Same LanceDB index in every row below (built 2026-08-09T20:45:43Z, 363 chunks,
15 files), so these are code-to-code comparisons.

| run | retry | R@5 | R@10 | R@20 | nDCG@10 (1st) |
|-----|-------|-----|------|------|---------------|
| **before** this change (`phase4_i2_before.json`) | on (profile) | 0.944 | 0.972 | 1.000 | **0.898** |
| after, run 1 (`phase4_i2_regression.json`) | on (profile) | 0.944 | 0.972 | 1.000 | **0.903** |
| after, run 2 (`phase4_i2_regression_run2.json`) | on (profile) | 0.958 | 0.986 | 1.000 | **0.887** |
| after, deterministic arm (`phase4_i2_regression_retryoff.json`) | off (forced) | 0.944 | 0.972 | 1.000 | **0.887** |
| Phase 2 gate, for reference (`gate_phase2_all.json`, 331 chunks / 14 files) | on (profile) | 0.944 | 0.972 | 1.000 | 0.9063 |

In the 0.90–0.91 band, and recall is bit-identical to the pre-change run. The
0.887–0.903 spread across the retry-on runs is the evidence-sufficiency retry's
LLM reformulation, which is nondeterministic; the retry-off arm is deterministic
and lands at 0.887, against 0.8881 for the last recorded retry-off run
(`phase2_final_retry_off.json`) on a corpus that has since grown from 331 to 363
chunks. **Honest caveat: I do not have a pre-change retry-off number on this
exact index, so the deterministic arm is compared against a different corpus
revision.** The code-level argument is the stronger one: with both flags absent,
`_overview_prefilter_documents` returns at the `enabled` check before any I/O and
`_crossref_hop` returns at the `enabled` check before touching the result, so
nothing on the flags-off path executes a new line.

### Index-side inertness (`indexing.extract_crossrefs` defaults ON)

The riskiest default in this change, so it was checked rather than argued. The
`mixed` corpus was rebuilt into a throwaway directory with the current pipeline
(extraction on) and compared chunk-by-chunk against the shared eval index, which
was built *before* this change:

```
old (pre-change build): 363 chunks
new (extraction ON)   : 363 chunks
chunk_id sets identical: True
text column identical  : True
vector shapes          : (363, 1024) (363, 1024)
vectors bit-identical  : True
max abs vector delta   : 0.0

new index: 147 crossrefs across 59 chunks, 59 resolved
old index: 0 chunks carrying crossrefs (expected 0)
```

Extraction adds 147 references across 59 of 363 chunks on the real
`Documentation/*.md` corpus, 59 of them resolving to 12 documents, and moves
neither a byte of `text` nor a bit of any vector.

### End-to-end smoke

`.venv/bin/python eval/smoke_e2e.py` — **25/25 assertions passed** (676.9s,
exit 0). Worth noting: the smoke's teardown reported removing a leaked
`index_store/overviews/<id>.vectors.npz` alongside the `.jsonl`, which
independently confirms the sidecar is produced on the real HTTP index-build path
and that the smoke's cleanup already globs it — no change to `eval/` was needed.

### Scratch functional test

Not in `eval/` — a throwaway 3-document index in a temp directory
(`master_agreement.md` references "Exhibit B" and "Schedule 2.1" and contains no
prices; `exhibit_b.md` is the rate card; `hr_handbook.md` is an unrelated
distractor). Full verbatim output is in the wave report. Deviation from the
brief: the documents are `.md`, not `.pdf`, because no PDF writer is installed in
this venv. The extension is stripped by name normalization, so the mechanism is
identical.

1. **Extraction.** `master_agreement.md#0` and `#1` both carry
   `{"kind": "exhibit", "ref": "exhibit b", "target_doc": "exhibit_b.md"}`;
   `schedule 2.1` and the five `section N` references are recorded with
   `target_doc: null`; `exhibit_b.md#0`'s own "Exhibit B" resolves to `null`
   (self-reference suppressed).
2. **Hop, negative case** (`retrieval_k=3`, exhibit_b already a candidate): no
   hop fires. The "not already represented" guard works.
3. **Hop, positive case** (`retrieval_k=1`, exhibit_b outside the candidates):
   the single candidate is `master_agreement.md#0`; the hop pulls both
   `exhibit_b.md` chunks tagged `via_crossref`, and the answer string `4,250`
   enters the context. `first_stage` is unchanged.
   *Caveat: `k=1` is induced. The whole corpus is 5 chunks, so at any k ≥ 2 the
   referenced document is already a candidate. On a real corpus the situation is
   the normal one; here it had to be forced.*
4. **Overview sidecar.** Written with 3 doc_ids, `(3, 1024)` vectors,
   `meta={'embedding_model': 'microsoft/harrier-oss-v1-0.6b', 'normalized': True}`.
5. **Boost.** For *"How much does the Client have to pay to onboard a new
   location?"* the first stage ranks `master_agreement.md` first and
   `exhibit_b.md` second; the prefilter selects `exhibit_b.md` from its overview
   ("a **rate card** outlining service pricing…") and boost flips the top-1 to
   `exhibit_b.md`, which is the document holding the answer.
6. **Restrict.** Confines retrieval to `['exhibit_b.md']`.
7. **Degradation.** A missing sidecar prints one line and returns the identical
   unfiltered result set.

---

## 3. Proposed config keys (for `rag_system/main.py`, at the adoption gate)

`PIPELINE_CONFIGS["default"]["retrieval"]`:

```python
            # Cross-reference hop (roadmap 4.2). OFF: index-time extraction is
            # free and additive, but the query-time hop appends chunks the
            # retriever never scored, and no gold query exercises it yet.
            "crossref_hop": {
                "enabled": False,
                "max_hops": 1,          # referenced documents expanded, no recursion
                "chunks_per_hop": 3
            },
            # Overview prefilter (roadmap 4.3). OFF until benchmarked. "boost"
            # is the safe mode — it reorders; "restrict" can hide a document.
            "overview_prefilter": {
                "enabled": False,
                "top_documents": 5,
                "mode": "boost"         # "boost" | "restrict"
            }
```

`PIPELINE_CONFIGS["default"]["indexing"]`:

```python
            "extract_crossrefs": True   # regex-only, no LLM; writes chunk metadata
```

`PIPELINE_CONFIGS["fast"]`: same two `retrieval` blocks with
`"enabled": False`, and `"extract_crossrefs": True` in `indexing` (it costs
nothing). `overview.embed` defaults to `True` in code and needs no profile entry
unless someone wants it discoverable.

Both blocks are read through the same `retrieval`/`retrievers` merge the retry
and late-chunk blocks use, so an API runtime override written under
`retrievers.crossref_hop` / `retrievers.overview_prefilter` already wins over the
profile with no extra code. Wiring UI toggles in `api_server.py` is therefore a
two-line mapping per flag — also a gate task, not this wave.

## 4. Proposed documentation diffs (none applied this wave)

* **`Documentation/indexing_pipeline.md`**
  * New section *Cross-reference extraction*, between *Chunking* and *Document
    overviews*: what the three regex families match, the `metadata.crossrefs`
    shape, the "resolves against batch + existing table, never against itself"
    rule, and the placement before contextual enrichment.
  * *Document overviews*: add the `.vectors.npz` sidecar — written at the end of
    the build with the document-side embedder, rebuilt wholesale, one row per
    document.
  * *Pipeline config keys* table: add `indexing.extract_crossrefs` (default
    `true`) and `overview.embed` (default `true`).
* **`Documentation/retrieval_pipeline.md`**
  * New stages for the cross-reference hop (after the retry, before context
    expansion) and the overview prefilter (inside the first stage), both marked
    default-off; the `crossref_hop` SSE event; the `via_crossref` /`crossref`
    fields on a source document.
* **`Documentation/design_rationale.md`**
  * §2 (chunking + index-time enrichment): one paragraph on why cross-reference
    extraction is index-time regex rather than an LLM pass, and why it precedes
    enrichment.
  * §4 (hybrid retrieval + RRF): note that the overview-prefilter boost reuses
    RRF at the same `_RRF_K` and deliberately introduces no weight, consistent
    with "there are no fusion weights, and there is no knob to add them".
  * A new short section (or §13 entry) recording that both features are shipped
    dark pending gold queries — the roadmap's own precondition.
* **`Documentation/research_roadmap.md`**
  * Phase 4 table rows 4.2 and 4.3: mark implemented-but-gated, pointing here.

## 5. What would settle it

Gold queries. Specifically:

* **4.2** — at least 4–6 `mixed` rows whose answer text lives in a document that
  is *only* reachable through a reference in another document, with `expected`
  anchored on the referenced document's text. Today's gold set cannot move at
  all when the hop is on, because the hop only ever *appends*, and the harness
  measures `first_stage`, which the hop never touches. **The harness will
  therefore report a flat line for 4.2 no matter how well it works** — measuring
  it needs either a post-hop metric or gold rows scored on `documents`.
* **4.3** — multi-document rows where the answer-bearing document is not the
  lexically closest one. `boost` and `restrict` should be run as separate arms;
  `restrict` also needs a *harm* check, because it can remove the correct
  document from consideration entirely, and recall@20 is where that would show.
* Both need the `mixed` corpus to actually contain a cross-referenced document
  pair; it does not today.

---

## Gate correction (2026-08-09) — resolver could not fire on the target corpus

The first 4.2 A/B (eval/decisions/phase4-eval-final-metric.md) measured **zero
hops on `acq`** and 7 non-gold hops on `acq+docs`. Root cause, verified at the
gate directly against the built indexes: all 34 references extracted from the
acquisition corpus had `target_doc: null`. Two reasons:

1. Exhibits/Schedules in this corpus are *sections inside*
   `01_acquisition_agreement.pdf`, not separate files — unresolvable by design,
   and correctly left null.
2. Document-name mentions could never match, because `normalize_name` keeps the
   numeric filename prefix (`08_regulatory_approval.pdf` → `"08 regulatory
   approval"`), and prose says "the Regulatory Approval documentation", never
   "08 regulatory approval".

**Fix applied at the gate** (`rag_system/indexing/crossref.py`,
`CrossRefExtractor.__init__`): each known document is additionally registered
under its numeric-prefix-stripped name (`"regulatory approval"`), subject to the
same `_MIN_NAME_CHARS`/`_MIN_NAME_TOKENS` guards. The full-name entry wins ties;
self-suppression is unchanged (it compares resolved doc ids, not names).

Verified at the gate on the real corpus text: `acq` extraction went from
0/34 resolved to **34/68 resolved, 9 of 10 documents linked**, with sensible
edges (due_diligence_report → regulatory_approval, risk_assessment →
closing_checklist, …) and no self-resolution. Original behaviors regression-
tested (exhibit_b resolution, self-suppression, label-pass dedup).

Consequences for the Wave-3 re-measurement:

* The `acq` and `acq_plus_docs` eval indexes must be **rebuilt** before any hop
  arm runs — crossrefs are stamped at index time and the existing indexes
  predate both the extractor and this fix.
* Known laxness accepted: a stripped single-word alias ("01_overview.pdf" →
  "overview") can over-match; the guards limit but do not eliminate this. The
  hop A/B's `crossref_hit_expected_source` precision column is the check.
* At the product's `retrieval_k=20`, appended hop chunks sit beyond rank 10 and
  cannot move nDCG@10 by construction. The meaningful mechanism test is small-k
  (`--k 3`, `--k 5`) recall/nDCG on the final list; the meaningful product test
  is judged end-to-end answers on the `requires_crossref` rows, hop on vs off.

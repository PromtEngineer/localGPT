> **Erratum 2026-08-16.** This page's "reranking OFF by default" decision no
> longer describes the shipped `default` profile: arm G (2026-08-14) turned
> reranking ON with `min_score: 0.5` / `min_keep: 3` / `top_k: 10` threshold
> selection, and arm H (2026-08-15) made pooled first-stage decomposition the
> default (`compose_from_sub_answers: False, pooled_first_stage: True`). Repro
> commands that assume a reranker-off stack must now pass `--no-rerank`
> explicitly. Note also that the eval harness's `build_config` replaced the
> whole reranker block with `top_k: None` until this date, so `final` metrics
> from runs before this change described a reorder-without-selection stack that
> never shipped; the harness now keeps the profile's selection and overrides
> only the model name. The measurements below are unchanged — they are records
> of their runs.

# Phase 1 component decisions — adopted 2026-08-09

The decision gate the roadmap asks for
([`Documentation/research_roadmap.md`](../Documentation/research_roadmap.md) §1):
*"adopt each only on a measured win; record adopted/rejected + numbers in
`eval/DECISIONS.md`."* This is that record.

Three A/Bs fed it, each with its own evidence page:

| Roadmap item | Investigation | Outcome |
|---|---|---|
| 1.1 reranker | [`decisions/reranker.md`](decisions/reranker.md) | **Reranking OFF by default**; `Qwen/Qwen3-Reranker-4B` is what the toggle loads |
| 1.2 embedder | [`decisions/embedder.md`](decisions/embedder.md) | **`microsoft/harrier-oss-v1-0.6b` adopted as the default embedder**, query-side instruction prefix on |
| 1.3 GLM-OCR | [`decisions/glm-ocr-spike.md`](decisions/glm-ocr-spike.md) | **GO-LATER — no code change**; three defects and a missing eval corpus block adoption |

1.1 and 1.2 could not be decided independently: the reranker's value depends
entirely on how good the first stage is. They were therefore re-measured
*jointly* and decided in one re-index window, together with the two index-format
hazards the embedder audit surfaced.

---

## 1. The joint matrix that decided it

`mixed` corpus (72 queries, 316 chunks — the only corpus with real distractors),
`docs` corpus (24 queries). First stage is hybrid RRF at k=20; every reranker
reorders the same 20 candidates. Latency is per query on an M2 Max (MPS) with a
shared GPU, so treat it as an order of magnitude, not a benchmark.

| Stack | mixed nDCG@10 | docs nDCG@10 | added latency |
|---|---|---|---|
| **harrier-0.6b, first stage only** ← **shipped** | **0.915** | **0.759** | — (~140 ms total) |
| harrier-0.6b + `BAAI/bge-reranker-v2-m3` | 0.892 | 0.701 | **+1.6 s — net negative** |
| harrier-0.6b + `Qwen/Qwen3-Reranker-4B` | 0.977 | 0.932 | +12.7 s |
| *(the previously-measured stack: `BASELINE.md`)* Qwen3-Embedding-0.6B + bge | 0.908 | 0.747 | +1.6 s |

Three things follow, and all three are counter-intuitive enough to be worth
stating plainly:

1. **harrier's first stage alone beats the whole previously-measured stack**
   (0.915 vs 0.908 on `mixed`) at a fraction of the latency.
2. **The cheap cross-encoder now *hurts*.** −0.022 nDCG@10 on `mixed` and −0.058
   on `docs`, for ~1.6 s per query. bge-reranker-v2-m3's famous +0.232 on `docs`
   was largely a repair job on a weak first stage; improve the first stage and
   the repair becomes damage.
3. **The good reranker is still a real win — and still too slow to default on.**
   +0.062 nDCG@10 on `mixed` and +0.173 on `docs`, for ~12.7 s per query and
   7.5 GB of resident weights, on a single-user server. That is worth paying on
   demand, not on every message.

One caveat on the table itself: every cell in it was measured **before** the
cosine-normalization fix in §2 (item 7), on unnormalized vectors. Normalization moves
the shipped first-stage cell by −0.004 nDCG@10 — smaller than the gaps the
decision rests on, but it means the shipped number is 0.911/0.913, not 0.915.
§2 and §4 carry the shipped measurement.

Supporting detail lives in the two decision pages: the embedder comparison
(harrier 0.915 vs Qwen3-Embedding-4B 0.875 vs Qwen3-Embedding-0.6B 0.805 on
`mixed`, first stage) in `decisions/embedder.md` §2 and its gate section, and the
three-way reranker A/B (bge 0.9077, Qwen3-Reranker-0.6B 0.9289, Qwen3-Reranker-4B
0.9825 on the older 0.6B-embedder first stage) in `decisions/reranker.md` §3.

---

## 2. What shipped

| # | Change | Where |
|---|---|---|
| 1 | Default embedder → `microsoft/harrier-oss-v1-0.6b` (MIT, 1024-dim, 1.2 GB) | `rag_system/main.py::EXTERNAL_MODELS` |
| 2 | Query-side instruction prefix stays on for harrier and the Qwen3-Embedding family | `rag_system/indexing/representations.py`, `rag_system/pipelines/retrieval_pipeline.py::_query_instruction` |
| 3 | `default` profile ships `reranker.enabled = False` | `rag_system/main.py::PIPELINE_CONFIGS` |
| 4 | Default reranker model → `Qwen/Qwen3-Reranker-4B`, loaded lazily only when the toggle is on | `rag_system/main.py::EXTERNAL_MODELS`, `rag_system/pipelines/retrieval_pipeline.py::_get_ai_reranker` |
| 5 | UI "AI reranker" toggle now defaults off, matching the profile | `src/components/ui/session-chat.tsx` |
| 6 | Per-table embedder identity marker + guard (index time and query time) | `rag_system/indexing/embedders.py`, `rag_system/retrieval/retrievers.py` |
| 7 | Vectors L2-normalized at write and query time, gated on that marker | same two files |
| 8 | Default table name `text_pages_v3` → `text_pages_v4` | `rag_system/main.py` |
| 9 | GLM-OCR | *nothing* — GO-LATER, by design |

### Why the identity marker (6)

The pre-existing guard compared **vector width only**. `harrier-oss-v1-0.6b` and
`Qwen3-Embedding-0.6B` are **both 1024-dim**, so swapping between exactly those
two — the swap this adoption makes people likely to perform — passed the guard
and appended mutually unintelligible vectors to a live table, silently. Each
table now records the embedding model that wrote it plus a `normalized` flag, in
the table's Arrow schema metadata (lancedb 0.36.0 round-trips it; a sidecar
`<db_path>/table_meta/<table>.json` is written if a future version does not).
Indexing into, or querying, a table whose recorded model differs from the
configured one now raises with a rebuild instruction instead of returning
nonsense. This is the pipeline-level guard and it covers **every** table,
including CLI-built ones; the backend's per-named-index `embedding_model`
metadata is unchanged and complementary.

### Why normalization (7) — and what it did *not* buy

Both model cards specify cosine similarity, but vectors were stored unnormalized
and searched with LanceDB's default L2 metric, so the ranking was neither cosine
nor intended. L2 ordering equals cosine ordering exactly when every vector is
unit length, so vectors are now L2-normalized on both sides.

Measured on `mixed`, this is **a wash, not a win** — it was adopted for
conformance with the model cards, not for a number. Both arms below were run
back to back against the same 316-chunk corpus snapshot, with normalization
neutered at *both* write and query time for the control:

| mixed, harrier-0.6b, first stage | recall@5 | recall@10 | recall@20 | nDCG@10 |
|---|---|---|---|---|
| unnormalized (control — reproduces `decisions/embedder.md` exactly) | 0.917 | 0.958 | 0.986 | 0.915 |
| **L2-normalized (shipped)** | 0.931 | 0.944 | 0.986 | **0.911** |

(The control is not a repo flag: it was produced by monkey-patching
`l2_normalize` to the identity in both `rag_system/indexing/embedders.py` and
`rag_system/retrieval/retrievers.py` and pointing the harness at a throwaway
index directory. There is no supported way to turn normalization off, and there
should not be.)

One query gained at recall@5, one lost at recall@10, nDCG@10 moved −0.004.
On 72 queries that is inside the noise floor `decisions/embedder.md` §8 sets for
itself. **The 0.915 headline in `decisions/embedder.md` was measured on
unnormalized vectors and does not transfer unchanged to the shipped stack** —
§4 below has the number that does.

Normalization is a property of the *table*, not of the config: new tables are
normalized, and a table without the marker is queried the old way with a warning
recommending a rebuild, so no index ever mixes the two conventions. The default
table name moved to `text_pages_v4` so the shipped default starts clean.

---

## 3. What stays opt-in (and how to switch it on)

| Option | When it is the right choice | Switch |
|---|---|---|
| `Qwen/Qwen3-Embedding-4B` | multilingual or long-context (32K) corpora — capabilities this English, digital-born gold set does not exercise. Keep the query prefix on: it measured **+0.059 nDCG@10** for this model. | `EMBEDDING_MODEL=Qwen/Qwen3-Embedding-4B` + rebuild the index |
| `Qwen/Qwen3-Reranker-4B` | quality-first sessions where ~12.7 s per query is acceptable | UI "AI reranker" toggle, or `reranker.enabled: true` |
| `BAAI/bge-reranker-v2-m3` | the low-latency legacy option; only pays off with a **weaker** embedder than the current default — on top of harrier it is net negative | `RERANKER_MODEL=BAAI/bge-reranker-v2-m3` + enable reranking |
| GLM-OCR for scanned PDFs | not yet — see `decisions/glm-ocr-spike.md` | no code exists; nothing to switch |

Rejected outright: **`Qwen/Qwen3-Reranker-0.6B`** — +0.021 nDCG@10 over bge
(one to two queries out of 72) bought with 1.5–2.8× the latency, while *losing*
recall@10 on both corpora. `decisions/reranker.md` §7 has the full argument.

---

## 4. Post-adoption verification

Re-run any time; the reranker follows the shipped profile, so a bare run
measures the shipped stack:

```bash
.venv/bin/python eval/run_eval.py --corpus mixed \
  --json-out eval/results/post_adoption_mixed.json
```

Latest result, on this tree after the documentation updates that shipped with
this decision — `mixed`, 317 chunks, 72 queries, first stage only, embedder
reported as `microsoft/harrier-oss-v1-0.6b` and reranker as `(disabled)`:

| | recall@5 | recall@10 | recall@20 | nDCG@10 (1st stage) | mean ms | p90 ms |
|---|---|---|---|---|---|---|
| `mixed` | **0.944** | **0.958** | **1.000** | **0.913** | 120.4 | 200.1 |

Gold coverage: 72/72 rows reachable, zero coverage failures.
Raw output: `eval/results/post_adoption_mixed.json`.

Note that the `docs` and `mixed` corpora are live `Documentation/*.md` content,
so editing the documentation moves these numbers — the run above is 317 chunks
because this decision's own doc updates grew the corpus by one chunk, which is
why it differs slightly from the 316-chunk numbers in §2. Only compare runs made
against the same tree.

---

## 5. Limits of this evidence

Carried forward verbatim in spirit from the two decision pages, because they
apply to the adopted defaults just as much as to the experiments:

* **72 English queries on one machine, over digital-born corpora.** A 0.014
  recall delta is one query. Only the large gaps (+0.110 embedder, +0.173
  reranker on `docs`) are outside the noise.
* **Latency was measured on a shared GPU** and is indicative, not a benchmark.
* **Nothing here measures answer quality** — these are retrieval metrics.
  Groundedness is a separate harness (`eval/judge.py`).
* **`docs_d09` and `docs_d17`** degrade under every reranker tested. They are a
  query-understanding problem and no default here fixes them.
* **Qwen3-Reranker latency was never tuned** (batch size 8, 2048-token cap,
  20 candidates). The 12.7 s figure is untuned, and tuning it is unmeasured
  work — not a promise.

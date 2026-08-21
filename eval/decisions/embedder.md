# Phase 1.2 — embedder audit + A/B — measured 2026-08-09

> **Superseded as a recommendation, retained as evidence.** The defaults this
> page says it did *not* change were changed at the Phase 1 adoption gate on
> 2026-08-09; what actually shipped, and the joint matrix behind it, is in
> [`../DECISIONS.md`](../DECISIONS.md). Every measurement below stands.

Every number on this page came from a run executed on this machine on
2026-08-09. Nothing is estimated, extrapolated, or copied from a leaderboard.
Where something could not be measured, it says so.

Raw outputs (git-ignored, re-runnable — commands at the bottom):
`eval/results/p12_{qwen06b,harrier06b}_{prefix,noprefix}[_rr].json`.

**No default was changed in `rag_system/main.py` and no user-facing doc was
updated.** This file is a recommendation, not a shipped change.

---

## 1. Audit finding: the instruction prefix was **absent**

The repo did **not** send an instruction prefix on the query side. Verified by
reading every query-embedding call site, not by grepping alone:

| Call site | What it embedded | Prefix before this change |
|---|---|---|
| `rag_system/retrieval/retrievers.py:79` (`MultiVectorRetriever._embed_single`) | the user query, for the dense leg of hybrid search | none |
| `rag_system/agent/loop.py:310` | the raw query, for the semantic cache | none |
| `rag_system/pipelines/indexing_pipeline.py:88` | chunk text (document side) | none — correct, and unchanged |

`QwenEmbedder.create_embeddings` tokenized whatever it was handed. Queries and
documents went through the identical code path, so the query side was missing
the `Instruct: {task}\nQuery: {q}` block that both model families are trained
with. Both cards are explicit that this costs accuracy: Qwen3-Embedding claims
1–5%, and harrier's card answers "Do I need to add instructions to the query?"
with "Yes, this is how the model is trained, otherwise you will see a
performance degradation."

### What was implemented

Query side only, in three owned files:

* `rag_system/indexing/representations.py` — `QUERY_PROMPT_TEMPLATE`,
  `DEFAULT_RETRIEVAL_INSTRUCTION`, `default_query_instruction(model_name)`,
  `apply_query_instruction(texts, instruction)`. `QwenEmbedder` and
  `OllamaEmbedder` take a `query_instruction`; when it is non-empty that
  *instance* prefixes every text it is given. `select_embedder` passes it
  through as a keyword argument, defaulting to `None`.
* `rag_system/pipelines/retrieval_pipeline.py` — `_query_instruction()` resolves
  `config["embedding_instruction"]` → `EMBEDDING_INSTRUCTION` env var → the
  model family's default; `_get_text_embedder()` passes the result.
  (Edit confined to `_get_text_embedder` and the helper directly above it; the
  reranker-loading section is owned by Phase 1.1 and was not touched.)
* `rag_system/indexing/embedders.py` — **not modified.** It turned out to be the
  LanceDB indexer, not the embedding model; nothing there needed to change.

Family default: the official retrieval instruction for any model name matching
`qwen3-embedding` or `harrier`; empty string for everything else (bge, Ollama
tags, etc.), so non-instruction-tuned models are unaffected.

### The document side stays unprefixed — stated explicitly

**Documents are embedded with no instruction, before and after this change, and
that is deliberate.** `IndexingPipeline` calls `select_embedder` without a
`query_instruction`, so it constructs a plain embedder; only `RetrievalPipeline`
constructs an instructed one. The asymmetry is what both model cards specify
("there is no need to add instructions to the document side") and it is also what
makes this change **index-compatible: every existing LanceDB index remains valid,
because no stored vector moves.** Only the query vector changes.

This was verified, not assumed: turning the prefix on and off for the same
embedder reused the same cached index (the `harrier` prefix-off run completed in
18.4 s against the index the prefix-on run had built, same 313/316 chunks).

---

## 2. Results

Gold set: the committed 72-row set, unchanged. Config: `k=20`, `chunk_size=512`,
hybrid retrieval, enrichment/overviews/late-chunking/context-expansion off —
i.e. exactly `BASELINE.md`'s configuration. Reranker, where used, is the shipped
`BAAI/bge-reranker-v2-m3`.

**Control: the harness reproduced `BASELINE.md`'s first-stage numbers to the
digit** with the prefix off (docs 0.750 / 0.917 / 0.958 / 0.515; mixed 0.917 /
0.972 / 0.986 / 0.805). The instrumentation is a no-op when disabled.

### First stage (the metric an embedder actually controls)

| Embedder | Prefix | Corpus | R@5 | R@10 | R@20 | nDCG@10 |
|---|---|---|---|---|---|---|
| Qwen3-Embedding-0.6B | off *(= BASELINE)* | docs | 0.750 | 0.917 | 0.958 | 0.515 |
| Qwen3-Embedding-0.6B | **ON** | docs | 0.708 | 0.875 | 0.958 | 0.577 |
| harrier-oss-v1-0.6b | off | docs | 0.708 | 0.875 | 0.958 | 0.741 |
| **harrier-oss-v1-0.6b** | **ON** | docs | **0.750** | 0.875 | 0.958 | **0.759** |
| Qwen3-Embedding-0.6B | off *(= BASELINE)* | **mixed** | 0.917 | 0.972 | 0.986 | 0.805 |
| Qwen3-Embedding-0.6B | **ON** | **mixed** | 0.903 | 0.958 | 0.986 | 0.847 |
| harrier-oss-v1-0.6b | off | **mixed** | 0.903 | 0.958 | 0.986 | 0.908 |
| **harrier-oss-v1-0.6b** | **ON** | **mixed** | **0.917** | 0.958 | 0.986 | **0.915** |

### After the `bge-reranker-v2-m3` cross-encoder

| Embedder | Prefix | Corpus | nDCG@10 first | nDCG@10 reranked | Δ from reranking |
|---|---|---|---|---|---|
| Qwen3-0.6B | off | docs | 0.515 | 0.747 | **+0.232** |
| Qwen3-0.6B | ON | docs | 0.577 | 0.734 | +0.156 |
| harrier-0.6b | ON | docs | 0.759 | 0.701 | **−0.058** |
| Qwen3-0.6B | off | mixed | 0.805 | 0.908 | **+0.103** |
| Qwen3-0.6B | ON | mixed | 0.847 | 0.903 | +0.056 |
| harrier-0.6b | ON | mixed | 0.915 | 0.892 | **−0.022** |

### Reading the tables

**a) The prefix is not the free win the roadmap hoped for — on Qwen3 it is a
trade.** On Qwen3-0.6B it buys ranking and pays in coverage: mixed nDCG@10
+0.042, but recall@5 −0.014 and recall@10 −0.014 (one query out of 72; on docs
it is one query out of 24). Per query on docs, nDCG improved on 27 and worsened
on 14. And **the gain does not survive the cross-encoder**: post-rerank mixed
goes 0.908 → 0.903 and docs 0.747 → 0.734, i.e. slightly *worse* with the prefix
on. End to end, on the model the repo ships today, the prefix is a wash.

**b) On harrier the prefix helps both metrics**, as its card demands: docs R@5
0.708 → 0.750 and nDCG 0.741 → 0.759; mixed R@5 0.903 → 0.917 and nDCG 0.908 →
0.915. For harrier the prefix is not optional.

**c) The embedder swap is the large effect.** harrier-0.6b + prefix vs the
Qwen3-0.6B baseline, first stage: **mixed nDCG@10 0.805 → 0.915 (+0.110)** and
**docs 0.515 → 0.759 (+0.244)**, at identical recall@5 (0.917 / 0.750) and
identical recall@20. On docs, 16 of 24 queries improve by >0.02 nDCG and 3
regress.

**d) The headline: harrier's *first stage alone* (mixed nDCG@10 0.915) beats
every reranked configuration measured here, including the shipped stack's 0.908.**
And bge-reranker-v2-m3 applied on top of harrier is **net negative** (−0.022
mixed, −0.058 docs) — it reorders a list that was already better than it is. This
is a direct input to Phase 1.1: the reranker's +0.232 on docs is largely a
first-stage-quality repair, and it shrinks or inverts as the first stage improves.
**The two decisions must be made together, not independently.**

**e) The recall@10 regression is real and should not be waved away.** harrier
loses one query at recall@10 on mixed (0.972 → 0.958) and one on docs (0.917 →
0.875) versus the Qwen3 baseline. The specific casualty on docs is `docs_d09`,
which goes from nDCG 0.431 to a complete miss (recall@5 and @10 both 1 → 0);
`docs_d14` and `docs_d17` drop out of the top 5 but stay in the top 10.
recall@20 is identical for every arm (0.958 docs / 0.986 mixed), so nothing is
lost from the candidate set — it is ordering within the top 10.

---

## 3. Does harrier need code changes? No.

`microsoft/harrier-oss-v1-0.6b` is a `Qwen3Model` (`config.json`
`model_type: qwen3`, 28 layers, hidden size 1024) with last-token pooling
(`1_Pooling/config.json`: `pooling_mode_lasttoken: true`) and L2 normalization —
architecturally the same shape the existing `QwenEmbedder` path already handles.
**It loaded through that path with no model-name branch and no config entry**, on
MPS, under the repo's transformers 4.51.0 despite the card's weights being saved
by 4.57.6.

Correctness was verified against the model card's own reference implementation
rather than assumed from "it ran": embedding the card's MS-MARCO example through
`QwenEmbedder` and through the card's `last_token_pool` + `F.normalize` snippet
gives **cosine 1.0000 between the two vectors for all four texts**, and the
score matrix matches to two decimals (65.19/25.47/30.66/70.13 vs
65.21/25.48/30.66/70.14 — fp16 vs fp32). Diagonal dominance is correct.
No `NaN`/`inf`.

Weights: 1,192,133,232 bytes, sha256 verified against the LFS blob id
`6bb124…8b66c`. (`huggingface_hub`'s downloader stalled repeatedly at 0 bytes on
this machine and once truncated a completed file; the weights were fetched with a
resumable `curl` and placed in the cache manually. A tooling problem, not a model
problem — worth knowing before anyone else tries this download.)

---

## 4. Index-rebuild implications

| Change | Re-index required? | Measured cost |
|---|---|---|
| Turning the **query prefix** on or off | **No.** Document vectors are untouched by construction. | zero |
| Swapping **Qwen3-0.6B → harrier-0.6b** | **Yes** — different vector space. | 32.2 s for the 313-chunk docs corpus, 34.9 s for 316-chunk mixed (vs 35.6 s / 37.1 s for Qwen3-0.6B). harrier is marginally *faster* to index. |
| Swapping to **Qwen3-4B** (today's shipped default) | Yes | not measured — see caveats |

**A silent-corruption hazard worth fixing in the same window.**
`VectorIndexer.index` guards an embedder swap by comparing *vector width only*
(`embedders.py`, `_table_vector_dim`). harrier-0.6b and Qwen3-Embedding-0.6B are
**both 1024-dim**, so swapping between those two would pass the guard and append
mutually-unintelligible vectors to an existing table with no error. The shipped
4B default is 2560-dim, so a 4B → harrier swap *would* be caught — but the
0.6B → harrier path would not. Recommendation: stamp the embedder model name into
the table (or a sidecar) and compare it, alongside the dim check, during the
re-index window. Not done here: it is an index-format change, and this file's
mandate was to measure, not to alter the index format.

The eval harness itself is already safe on two independent axes, verified:
`eval/run_eval.py` writes each embedder's index to
`eval/.eval_indexes/<slug(embedder)>/<corpus>/` **and** carries `embedder` inside
the fingerprint dict, so two embedders can neither share a directory nor accept
each other's cached marker. **No fingerprint fix was needed and `run_eval.py` was
left unmodified.**

---

## 5. Licences

| Model | Licence | Verdict |
|---|---|---|
| `Qwen/Qwen3-Embedding-0.6B` / `-4B` | Apache 2.0 | fine |
| `microsoft/harrier-oss-v1-0.6b` | **MIT** (confirmed in the model card's front-matter: `license: mit`) | fine — MIT is strictly more permissive |

Neither licence blocks anything. Both allow commercial use and redistribution.

---

## 6. Latency

The GPU was shared with another agent's reranker A/B throughout, so **latency
here is noisy and the quality numbers are not.** First-stage means per query
across all arms: 100–168 ms, with harrier (103–120 ms) indistinguishable from
Qwen3-0.6B (100–109 ms) — same architecture, same 1024-dim output, one extra
short prefix to tokenize. Rerank means ranged 1884–3308 ms for the identical
20-candidate cross-encoder workload, against `BASELINE.md`'s 1741–1788 ms; that
spread is contention, not signal. **Do not quote the rerank column as a
measurement.**

Determinism was checked rather than assumed: re-running one identical
configuration end to end produced bit-identical metrics
(docs R@5 0.7500, R@10 0.9167, nDCG first 0.5150, nDCG reranked 0.7468 twice).

---

## 7. Recommendation

1. **Adopt `microsoft/harrier-oss-v1-0.6b` as the embedder**, with the query-side
   instruction prefix on. On the only corpus with real distractors it is
   +0.110 nDCG@10 first-stage over the Qwen3-0.6B baseline at equal recall@5 and
   equal recall@20, it needs no code beyond what is already merged, it indexes
   slightly faster, and MIT is the more permissive licence. **Gate: this must land
   in the same re-index window as the Phase 1.1 reranker decision**, because
   finding (d) says bge-reranker-v2-m3 is net-negative on top of harrier — adopting
   the embedder without revisiting the reranker would ship a stack whose last
   stage costs ~2 s per query to make the ranking *worse*.
2. **Keep the prefix on for harrier; treat it as undecided for Qwen3.** If the
   repo stays on a Qwen3 embedder, the honest reading of the measurement is that
   the prefix is a wash-to-slightly-negative end to end, and
   `embedding_instruction: ""` should be set explicitly rather than inheriting the
   family default.
3. **Do not change `main.py` yet.** Both of the above are recommendations
   pending the Phase 1.1 outcome and the 4B measurement below.

---

## 8. Caveats — what this does not tell you

* **Nothing about the shipped default.** The comparison is against
  Qwen3-Embedding-**0.6B**; the repo ships Qwen3-Embedding-**4B**. The 4B was not
  measured: it is ~8 GB and measured throughput to the HF CDN on this machine was
  **630–670 KB/s**, i.e. roughly 3.5 hours of download, on a machine already
  shared with another agent's GPU work. A 4B-vs-harrier-0.6b number is the single
  most valuable missing data point and should be the first thing run when
  bandwidth allows. It is entirely possible the 4B closes some of the +0.110 gap.
* **The prefix now defaults ON for Qwen3-Embedding-4B, untested.** The
  family-default rule matches `qwen3-embedding`, so the shipped 4B default now
  receives a prefix nobody has measured. Given that the prefix measured as a wash
  on the 0.6B, this is a live risk, not a certainty of improvement. One-line
  revert: set `embedding_instruction: ""` in the profile.
* **Post-rerank numbers are internally comparable only.** All arms here ran within
  the same hour against the same tree, and a repeat was bit-identical — but they
  do **not** reproduce `BASELINE.md`'s post-rerank column (docs 0.747 here vs
  0.731 there). Another agent is actively editing `rag_system/rerankers/reranker.py`
  and `retrieval_pipeline.py`'s reranker section for Phase 1.1, so the tree changed
  between the two measurements. First-stage numbers **do** reproduce BASELINE
  exactly, which is why the recommendation rests on them.
* **72 queries.** A 0.014 recall delta is one query. Deltas below ~0.03 on recall
  should not be treated as findings; the nDCG gaps driving the recommendation
  (+0.110, +0.244) are an order of magnitude larger than that floor.
* **Vectors are never L2-normalized and the vector search uses LanceDB's default
  L2 metric** (`retrievers.py` calls `tbl.search(vector).limit(k)` with no
  `.metric()`), while both model cards specify cosine similarity over normalized
  embeddings. Every arm above is affected identically, so the comparison is fair,
  but all of them may be leaving accuracy on the table. Not changed here:
  normalizing is a document-side change that invalidates existing indexes, and
  `retrievers.py` is outside this task's ownership. **Worth testing in the
  re-index window** — it is the cheapest remaining lever.
* **The semantic cache is affected but harmlessly.** It compares query-to-query,
  so both sides carry the prefix. Measured on the 72 gold queries, the prefix
  *lowers* mean pairwise cosine (0.361 → 0.216) and max (0.928 → 0.899); with the
  shipped `semantic_cache_threshold` of 0.98 nothing crosses the bar either way,
  so the cache only ever fires on near-identical queries, as before.
* **Nothing about answer quality.** These are retrieval metrics; synthesis,
  verification and groundedness are untouched by this file.
* `atlas7` and `hr` are 1 and 2 chunks and saturate at recall 1.0 by construction.
  They are omitted above for that reason — read `mixed`.

---

## 9. Reproducing every number above

```bash
cd /path/to/localGPT

# control: reproduces BASELINE.md's first-stage numbers exactly
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B EMBEDDING_INSTRUCTION="" \
  .venv/bin/python eval/run_eval.py --corpus all --no-rerank \
  --json-out eval/results/p12_qwen06b_noprefix.json

# Qwen3 + query prefix (family default; omit EMBEDDING_INSTRUCTION)
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B \
  .venv/bin/python eval/run_eval.py --corpus all --no-rerank \
  --json-out eval/results/p12_qwen06b_prefix.json

# harrier, prefix off / on
EMBEDDING_MODEL=microsoft/harrier-oss-v1-0.6b EMBEDDING_INSTRUCTION="" \
  .venv/bin/python eval/run_eval.py --corpus all --no-rerank \
  --json-out eval/results/p12_harrier06b_noprefix.json
EMBEDDING_MODEL=microsoft/harrier-oss-v1-0.6b \
  .venv/bin/python eval/run_eval.py --corpus all --no-rerank \
  --json-out eval/results/p12_harrier06b_prefix.json

# add --reranker BAAI/bge-reranker-v2-m3 and drop --no-rerank for the
# post-rerank column, e.g.
EMBEDDING_MODEL=microsoft/harrier-oss-v1-0.6b \
  .venv/bin/python eval/run_eval.py --corpus all \
  --reranker BAAI/bge-reranker-v2-m3 \
  --json-out eval/results/p12_harrier06b_prefix_rr.json
```

`EMBEDDING_INSTRUCTION=""` switches the prefix off; unset inherits the model
family's default; any other string overrides the task description. The same
knob exists as `config["embedding_instruction"]`, which takes precedence.

---

## Gate validation (2026-08-09, run by the validating orchestrator, not the Phase-1.2 agent)

Independent reproduction: harrier-0.6b + prefix on `mixed` reproduced the agent's
headline exactly (R@5 0.917 / R@10 0.958 / nDCG@10 first-stage **0.915**;
`eval/results/` + `/tmp/val_harrier.json`).

The declared 4B gap is now closed. Qwen/Qwen3-Embedding-4B (the shipped default),
first stage, both prefix modes — commands as in §9 with the 4B model id:

| Config | mixed R@5 | mixed R@10 | mixed R@20 | mixed nDCG@10 | docs nDCG@10 | 1st-stage ms |
|---|---|---|---|---|---|---|
| 4B + prefix (family default) | 0.889 | 0.931 | 0.972 | 0.875 | 0.638 | ~334 |
| 4B, prefix off | 0.889 | 0.931 | 0.986 | 0.816 | 0.518 | ~366 |
| harrier-0.6b + prefix (ref) | 0.917 | 0.958 | 0.986 | **0.915** | **0.759** | ~133 |

**Finding: the 8 GB shipped default is dominated by the 1.2 GB harrier on this
gold set in both prefix configurations — on ranking quality, recall@5/@10,
latency (~3x), and memory (~7x).** The prefix helps 4B's ranking (+0.059 nDCG)
while leaving recall unchanged, so if 4B is retained anywhere, keep the prefix.

Honest scope limits: 72 English queries, one machine, digital-born corpora.
Qwen3-4B's documented advantages (multilingual breadth, MRL dimension
flexibility, 32K context) are not exercised by this gold set, so the right
docs framing is "harrier-0.6b default, Qwen3-Embedding-4B documented option
for multilingual/long-context corpora" — not "4B is bad".

Adoption remains gated on the joint reranker decision (Phase 1.1) plus the two
hazards above (width-only index guard; L2-vs-cosine normalization), all to land
in one re-index window.


---

**Gate correction (2026-08-09, post-adoption):** §8's statement "Vectors are never
L2-normalized and the vector search uses LanceDB's default L2 metric" described the
tree at measurement time and is now false: v4 tables L2-normalize at write and query
(`rag_system/indexing/embedders.py`), making L2 ordering the cosine ordering. The §8
fairness argument (all arms measured under identical metric handling) still holds.

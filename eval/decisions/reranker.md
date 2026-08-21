# Phase 1.1 decision — reranker A/B, measured 2026-08-09

> **Superseded as a recommendation, retained as evidence.** The defaults this
> page says it did *not* change were changed at the Phase 1 adoption gate on
> 2026-08-09; what actually shipped, and the joint matrix behind it, is in
> [`../DECISIONS.md`](../DECISIONS.md). Every measurement below stands.

Roadmap item: [`Documentation/research_roadmap.md`](../../Documentation/research_roadmap.md) §1.1.
Baseline this A/B is measured against: [`eval/BASELINE.md`](../BASELINE.md).

**Every number on this page came from a run executed on this machine on
2026-08-09 (UTC).** Nothing is copied from a leaderboard, estimated or
extrapolated. Failures are reported verbatim. Where a number could not be
measured, it says so.

Raw outputs (git-ignored, re-run the commands in "Reproducing" to regenerate):
`eval/results/ab_bge_run{1,2}.json`, `eval/results/ab_qwen06_run{1,2}.json`,
`eval/results/ab_qwen4b_run{1,2}.json`.

**This document changes no default.** `rag_system/main.py` still ships
`BAAI/bge-reranker-v2-m3`. This is a recommendation with its evidence attached.

---

## 1. Integration finding — the `rerankers` library cannot score Qwen3-Reranker

The roadmap flagged this as an open question ("may need a small custom
scorer"). It was tested, not assumed.

`rerankers` 0.10.0, the version installed in `.venv`, has **no Qwen3-Reranker
backend**. Its `models/` directory holds ColBERT, FlashRank, LLM-layerwise,
MonoVLM, mxbai-v2, PyLate, RankGPT, RankLLM, T5, UPR and a generic
transformer cross-encoder; the only `qwen` matches in the whole package are
`lightonai/MonoQwen2-VL-v0.1` and a Qwen chat-template string inside
`mxbai_v2.py`.

Loading `Qwen/Qwen3-Reranker-0.6B` through the shipped path
(`Reranker(model_name, model_type="cross-encoder")` — exactly what
`retrieval_pipeline.py` and `eval/run_eval.py` do) produced, verbatim:

```
Some weights of Qwen3ForSequenceClassification were not initialized from the model
checkpoint at Qwen/Qwen3-Reranker-0.6B and are newly initialized: ['score.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for
predictions and inference.
Loading TransformerRanker model Qwen/Qwen3-Reranker-0.6B (this message can be suppressed by setting verbose=0)
No device set
Using device mps
No dtype set
Using dtype torch.float32
Loaded model Qwen/Qwen3-Reranker-0.6B
Using device mps.
Using dtype torch.float32.
FAIL cross-encoder ValueError Cannot handle batch sizes > 1 if no padding token is defined.
```

Two separate defects in one call:

1. **Silent correctness failure.** The library builds a
   `Qwen3ForSequenceClassification` with a **randomly initialised `score`
   head**. Had the batching not thrown, this configuration would have returned
   untrained noise while printing "Loaded model" and "AI reranker initialized
   successfully" — a wrong-answer failure, not a crash.
2. **Hard failure.** `ValueError: Cannot handle batch sizes > 1 if no padding
   token is defined.` — the model's tokenizer has no `pad_token` configured for
   the sequence-classification path, so it cannot batch at all.

Conclusion: a custom scorer was required. It was written.

### What was implemented

`QwenRerankerScorer` in
[`rag_system/rerankers/reranker.py`](../../rag_system/rerankers/reranker.py),
implementing the scoring scheme published on the Qwen3-Reranker model card:

* `AutoModelForCausalLM`, not `AutoModelForSequenceClassification`.
* Tokenizer loaded with `padding_side="left"` (required — the score is read off
  the last position of a causal LM).
* Each (query, document) pair wrapped in the model's chat template:
  a system turn instructing a yes/no judgment, a user turn
  `<Instruct>: … \n<Query>: … \n<Document>: …`, and an assistant turn opened
  with an empty `<think></think>` block.
* Score = `softmax` over the `yes`/`no` token logits at the final position,
  reported as `P(yes)` in `[0, 1]`.
* fp16 on MPS/CUDA, fp32 on CPU. Batch size 8. Truncation cap 2048 tokens
  (padding is to the longest item in the batch, not to the cap).

Interface: `rank(query, docs, top_k=None) -> [(score, original_index), …]`
sorted descending, plus a dict-in/dict-out `rerank()` mirroring
`CrossEncoderReranker`. Both `RetrievalPipeline.run()` (lines 374–383) and
`eval/run_eval.py`'s `rerank()` already fall back to treating a plain list as
`(score, idx)` pairs when the returned object has no `.results`, so **no change
to either call site was needed** — only to the loader.

3-pair sanity check on the implemented scorer (executed, not assumed):

| query | document | 0.6B `P(yes)` | 4B `P(yes)` |
|---|---|---|---|
| "What is the capital of France?" | "Paris is the capital of France." | 0.9974 | 0.9294 |
| " | "The Eiffel Tower is a tower in Paris." | 0.0054 | 0.0633 |
| " | "Bananas are yellow fruit." | 0.0000161 | 0.0000597 |

### Loader wiring

`RetrievalPipeline._get_ai_reranker()` routes to `QwenRerankerScorer` when
**either** condition holds:

* `reranker.model_type == "qwen3"` (the new explicit config value), **or**
* the model name matches `is_qwen3_reranker()`, i.e. contains
  `qwen3-reranker` (case-insensitive).

The name-based leg is not cosmetic — it is load-bearing for the eval harness.
`eval/run_eval.py:191-197` hard-codes `"model_type": "cross-encoder"` in the
config it builds, and `run_eval.py` is out of this task's ownership scope, so
an explicit-config-only route could not have been A/B-tested at all. It also
closes the silent-noise failure above: any Qwen3-Reranker name now gets a
trained scorer instead of a random head, however it is configured.

The cross-encoder path is untouched and remains the default. `bge-reranker-v2-m3`
still goes `strategy == "rerankers-lib"` → `Reranker(model_name,
model_type="cross-encoder")`, byte-for-byte the previous behaviour; the bge
re-run below reproduces the baseline first stage exactly, which is the evidence.

---

## 2. Conditions under test

| | |
|---|---|
| Embedder | `Qwen/Qwen3-Embedding-0.6B` (1024-dim), same as `BASELINE.md`. **Not** the shipped 4B default. |
| Query-side instruction | **Pinned off** via `EMBEDDING_INSTRUCTION=` on every run. A concurrent Phase 1.2 change added a default Qwen3 query prefix (`Given a web search query, retrieve relevant passages that answer the query`) to `RetrievalPipeline._query_instruction()` while this A/B was running; pinning it empty reproduces the `BASELINE.md` first stage and holds the candidate lists identical across all six runs. |
| Index | Rebuilt once at the start, `2026-08-09T06:29:53Z` (docs, 313 chunks) and `06:30:30Z` (mixed, 316 chunks). All six runs reuse that same cached index — verified by the `built_at` timestamps and by md5-hashing the 12 `Documentation/*.md` corpus files before the first run and after the last (**unchanged**). |
| Profile | `PIPELINE_CONFIGS["default"]`, enrichment / overviews / late-chunking / context-expansion / decomposition / verification all OFF, `k = 20`, `chunk_size = 512` — i.e. `run_eval.py` defaults, identical to Phase 0. |
| Hardware | Apple M2 Max, 96 GB, MPS, torch 2.4.1, transformers 4.51.0, rerankers 0.10.0. |
| Load | **Shared.** Other agents were running their own MPS evals against the same GPU throughout. Latency is noisy by construction; every latency-sensitive comparison was run twice and both runs are reported. |

Model weights on disk: bge-reranker-v2-m3 2.1 GB · Qwen3-Reranker-0.6B 1.1 GB ·
Qwen3-Reranker-4B 7.5 GB (downloaded for this A/B).

### Deviation from `BASELINE.md`, stated plainly

The `docs` corpus is live repo content and `Documentation/indexing_pipeline.md`
was edited after the Phase 0 index was built, so the index had to be rebuilt.
Chunk counts came out identical (313 / 316) and the **first stage reproduced
exactly** — mixed nDCG@10 0.805, recall@5 0.917, recall@10 0.972, recall@20
0.986, matching `BASELINE.md` to three decimals. The bge post-rerank number
moved slightly (mixed 0.903 → 0.9077, docs 0.731 → 0.7468) because the reranker
sees slightly different chunk text. **That is why bge was re-run rather than
quoted**; the bge column below, not `BASELINE.md`, is the comparison point.

---

## 3. Quality — `mixed` corpus (n = 72, 316 chunks) — the headline table

All three rerankers reorder the **same** 20 first-stage candidates per query.

| Reranker | nDCG@10 first stage | **nDCG@10 post-rerank** | Δ vs first stage | recall@5 post-rerank | recall@10 post-rerank |
|---|---|---|---|---|---|
| *(no rerank — first stage)* | 0.805 | — | — | 0.917 | 0.972 |
| `BAAI/bge-reranker-v2-m3` (current default) | 0.805 | **0.9077** | +0.103 | 0.9167 | 0.9861 |
| `Qwen/Qwen3-Reranker-0.6B` | 0.805 | **0.9289** | +0.124 | 0.9583 | 0.9722 |
| `Qwen/Qwen3-Reranker-4B` | 0.805 | **0.9825** | +0.178 | 0.9861 | 0.9861 |

Deltas against the re-run bge column: **0.6B +0.021 nDCG@10, 4B +0.075
nDCG@10.** recall@5 post-rerank: 0.6B +0.042, 4B +0.069. recall@10
post-rerank: 4B unchanged at 0.9861, **0.6B is 0.014 worse** (0.9722 vs
0.9861 — one query out of 72).

Every quality figure above reproduced **identically** on the second run of each
model (all six runs are deterministic to four decimals). Only latency moved.

## 4. Quality — all corpora

`docs` (n = 24, 313 chunks) is the only corpus with real distractors; `atlas7`
(1 chunk) and `hr` (2 chunks) are saturated by construction at k = 20 and test
the plumbing, not the retriever.

| Corpus | metric | first stage | bge-v2-m3 | Qwen3-0.6B | Qwen3-4B |
|---|---|---|---|---|---|
| `docs` | nDCG@10 | 0.515 | 0.7468 | 0.7868 | **0.9474** |
| `docs` | recall@5 | 0.750 | 0.750 | 0.875 | **0.9583** |
| `docs` | recall@10 | 0.9167 | 0.9583 | 0.9167 | 0.9583 |
| `atlas7` | nDCG@10 | 1.000 | 1.000 | 1.000 | 1.000 |
| `atlas7` | recall@5 / @10 | 1.000 | 1.000 | 1.000 | 1.000 |
| `hr` | nDCG@10 | 0.954 | 1.000 | 1.000 | 1.000 |
| `hr` | recall@5 / @10 | 1.000 | 1.000 | 1.000 | 1.000 |
| `mixed` | nDCG@10 | 0.805 | 0.9077 | 0.9289 | **0.9825** |

On `docs`, the corpus that actually discriminates: **4B is +0.201 nDCG@10 over
bge** and +0.208 recall@5. The 0.6B is +0.040 nDCG@10 over bge but **loses
recall@10** (0.9167 vs 0.9583) — it demotes one answer-bearing chunk out of the
top 10 that bge keeps in.

### By dimension (all corpora pooled, n = 144 query evaluations, nDCG@10 post-rerank)

| Slice | n | bge | Qwen3-0.6B | Qwen3-4B |
|---|---|---|---|---|
| difficulty = easy | 88 | 0.916 | 0.935 | **1.000** |
| difficulty = hard | 56 | 0.904 | 0.920 | **0.955** |
| type = factoid | 72 | 0.936 | 0.941 | **0.983** |
| type = comparative | 18 | 0.955 | 0.957 | **0.983** |
| type = negative | 28 | 0.912 | **0.866** | **0.964** |
| type = procedural | 26 | 0.815 | 0.943 | **1.000** |

The one slice where a Qwen3 model loses to bge: **0.6B on `negative` questions,
0.866 vs bge's 0.912.** With n = 28 that is roughly one query; treat it as a
diagnostic, not a finding. The 4B wins every slice.

---

## 5. The four known bge regressions — do the Qwen3 models fix them?

`BASELINE.md` names four queries the bge cross-encoder ranks **worse** than the
first stage did. Per-query nDCG@10 on `mixed`, same candidate lists:

| Query | first stage | bge | Qwen3-0.6B | Qwen3-4B | verdict |
|---|---|---|---|---|---|
| `atlas7_a16` | 1.000 | 0.431 | **1.000** | **1.000** | **fixed by both** |
| `docs_d15` | 1.000 | 0.316 | 0.631 | **1.000** | fixed by 4B; 0.6B halves the damage but still regresses |
| `docs_d21` | 0.500 | 0.387 | **0.631** | **1.000** | **fixed by both** (both now beat the first stage) |
| `docs_d09` | 0.431 | 0.333 | 0.387 | 0.387 | **shared by all three.** Both Qwen3 models improve on bge but neither reaches the first-stage score |

**Three of four fixed by the 4B; two of four by the 0.6B. `docs_d09` is a
shared regression — no candidate reranker recovers it.**

Counting every query on `mixed` whose post-rerank nDCG@10 falls below its
first-stage value:

| Reranker | queries degraded (of 72) | which | queries improved |
|---|---|---|---|
| bge-v2-m3 | **7** | `atlas7_a16`, `docs_d02`, `d09`, `d14`, `d15`, `d17`, `d21` | 20 |
| Qwen3-0.6B | **5** | `docs_d02`, `d09`, `d15`, `d17`, `d20` | 23 |
| Qwen3-4B | **2** | `docs_d09`, `docs_d17` | 24 |

The 4B cuts reranker-induced damage from 7 queries to 2. `docs_d09` and
`docs_d17` are the residue every model shares or nearly shares — the honest
statement is that reranker choice does not solve them and they need a different
intervention.

The `docs` queries `BASELINE.md` lists as retrieved-but-badly-ranked also move,
sharply:

| Query | first stage | bge | Qwen3-0.6B | Qwen3-4B |
|---|---|---|---|---|
| `docs_d08` (the "clearest case for keeping a cross-encoder") | 0.000 | 0.387 | 0.631 | **1.000** |
| `docs_d07` | 0.387 | 0.387 | 1.000 | **1.000** |
| `docs_d13` | 0.316 | 0.316 | 0.631 | **1.000** |
| `docs_d14` | 0.387 | 0.356 | 1.000 | **1.000** |
| `docs_d19` | 0.333 | 0.387 | 0.631 | **1.000** |
| `docs_d16` | 1.000 | 1.000 | 1.000 | 1.000 |

`docs_d16` stays at nDCG 1.000 for all three and at recall 0 for all three: it
is a `match: "all"` comparative whose second anchor never enters the candidate
set. **No reranker can fix a first-stage coverage miss** — that one belongs to
Phase 1.2.

---

## 6. Latency — per query, `mixed`, wall clock, MPS, shared GPU

Every model was run twice on `mixed`. **Both runs are reported; neither is
discarded.** The GPU was shared with other agents' evals throughout, and the
spread between the two runs is the honest measure of how much that matters.

| Reranker | run | mean | median | **p95** | max |
|---|---|---|---|---|---|
| bge-v2-m3 | 1 | 1883 ms | 1905 ms | 2181 ms | 3675 ms |
| bge-v2-m3 | 2 | 2189 ms | 1625 ms | 3559 ms | 4067 ms |
| Qwen3-0.6B | 1 | 5296 ms | 5578 ms | 6849 ms | 9867 ms |
| Qwen3-0.6B | 2 | 3010 ms | 2836 ms | 4179 ms | 6108 ms |
| Qwen3-4B | 1 | 19520 ms | 14255 ms | 43883 ms | 48064 ms |
| Qwen3-4B | 2 | 11957 ms | 12507 ms | 13238 ms | 14683 ms |

Reading the two runs together, per 20-candidate query:

* **bge ≈ 1.6–2.2 s mean.** Consistent with `BASELINE.md`'s 1.55–1.74 s; the
  excess is contention.
* **Qwen3-0.6B ≈ 3.0–5.3 s mean, ~1.4–2.8× bge.** Run 2 is the cleaner
  measurement (its p95/median ratio is 1.47 vs run 1's 1.23 on a much higher
  base); call it **~1.5–2× bge** under light load.
* **Qwen3-4B ≈ 12–19.5 s mean, ~5.5–10× bge.** Run 1's p95 of 43.9 s was taken
  while another agent ran a Qwen3-Embedding-4B eval on the same GPU. **Run 2's
  12.0 s mean / 13.2 s p95 is the number to plan against, and it is still
  ~5.5× bge and ~80× the 134–322 ms first stage.**

Whole-run wall clock, for scale: `--corpus all` (144 query evaluations) took
**210 s** with bge, **575 s** with Qwen3-0.6B, **1916 s** with Qwen3-4B.
The 4B's `mixed`-only run 2 took **875 s** for 72 queries.

Latency is not a property of the scorer alone: the Qwen3 path pays a full
causal-LM forward pass over `prompt + query + document` per pair, where bge
pays one 512-token encoder pass. Batch size 8 and the 2048-token truncation cap
in `QwenRerankerScorer` are the two knobs; neither was tuned for this A/B, and
tuning them is unmeasured work, not a promise.

---

## 7. Recommendation

**Adopt Qwen3-Reranker-4B for a quality-first profile; keep
`bge-reranker-v2-m3` as the shipped default until the latency is addressed.
Do not adopt Qwen3-Reranker-0.6B.**

That is an *adopt-with-size-choice*, and the size choice is 4B-or-nothing.

**Why 4B is the only Qwen3 worth taking.** It is the largest single measured
quality win in this repo's eval history: **+0.075 nDCG@10 on `mixed` and +0.201
on `docs`** over a re-run bge baseline, with recall@5 up 0.069 and recall@10
not regressing. It fixes three of the four known bge regressions, cuts
reranker-induced damage from 7 queries to 2, wins every dimension slice, and
lifts `docs_d08` — the query `BASELINE.md` singles out as the strongest
argument for having a cross-encoder at all — from bge's 0.387 to 1.000. The
roadmap's own bar is "adopt only on a measured win"; this clears it by a wide
margin and is far above the ~2-point threshold below which leaderboard deltas
are known not to transfer.

**Why it should not become the default today.** ~12 s per query at p95 13.2 s,
against bge's ~2 s, on a single-user, single-threaded `TCPServer`. That is a
user-visible pause on every question and it multiplies under decomposition
(which fans out into multiple rerank calls). It also costs 7.5 GB of weights
resident alongside the embedder and the generation model. The quality is worth
paying for on demand; it is not obviously worth paying for on every message.
Concretely: expose it as an opt-in profile (`reranker.model_name:
"Qwen/Qwen3-Reranker-4B"` in a quality/deep profile), leave `default` and
`fast` on bge, and revisit the default only after the latency work below.

**Why 0.6B is rejected.** +0.021 nDCG@10 on `mixed` over bge — one to two
queries out of 72, inside the noise band the roadmap itself says not to trust —
bought with a 1.5–2.8× latency increase. It also *loses* recall@10 on both
`mixed` (0.9722 vs 0.9861) and `docs` (0.9167 vs 0.9583), and loses to bge on
the `negative` question slice (0.866 vs 0.912). It fixes only two of the four
known regressions. A small ranking gain paid for with a recall loss and 2× the
latency is not a win; there is no configuration in which the 0.6B is the right
answer when 4B and bge both exist.

**What this does not settle.** These numbers are on the 1024-dim
`Qwen3-Embedding-0.6B` index with the query-side instruction pinned off, not on
the shipped 4B embedder and not with the Phase 1.2 instruction prefix on. If
Phase 1.2 changes the embedder default, the first stage changes and this A/B
must be re-run before the reranker choice is final — the 4B's headroom is
largest exactly where the first stage is weakest (`docs`, nDCG 0.515), so a
better first stage will shrink, not grow, its margin.

### Suggested follow-up before any default flips

1. **Tune the Qwen3 latency knobs** — batch size (currently 8) and the 2048-token
   truncation cap (currently generous: chunks are 512 tokens). Both are
   untested; either could move the 12 s materially.
2. **Try `top_k` truncation of the candidate list before reranking.** All the
   quality above is on 20 candidates. If reranking the top 10 keeps the nDCG,
   it halves the cost.
3. **Re-A/B after Phase 1.2** settles the embedder and the instruction prefix.
4. **`docs_d09` and `docs_d17`** are reranker-proof — every model degrades them.
   They are a query-understanding problem, not a reranker problem.

---

## 8. Reproducing every number above

```bash
cd /path/to/localGPT

# one-time: fetch the 4B weights (7.5 GB). Disable xet — with it enabled the
# download stalled at 0 bytes twice on this machine (see Caveats).
HF_HUB_DISABLE_XET=1 .venv/bin/python -c \
  "from huggingface_hub import snapshot_download; \
   print(snapshot_download('Qwen/Qwen3-Reranker-4B', max_workers=2))"

# build/refresh the shared index once, so all three models see identical candidates
EMBEDDING_INSTRUCTION= EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B \
  .venv/bin/python eval/run_eval.py --corpus all --coverage-only

# the A/B — run 1 (all corpora)
for M in BAAI/bge-reranker-v2-m3 Qwen/Qwen3-Reranker-0.6B Qwen/Qwen3-Reranker-4B; do
  EMBEDDING_INSTRUCTION= EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B \
    .venv/bin/python eval/run_eval.py --corpus all --reranker "$M" \
    --json-out "eval/results/ab_$(basename $M)_run1.json"
done

# run 2 — mixed only, for the second latency measurement
for M in BAAI/bge-reranker-v2-m3 Qwen/Qwen3-Reranker-0.6B Qwen/Qwen3-Reranker-4B; do
  EMBEDDING_INSTRUCTION= EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B \
    .venv/bin/python eval/run_eval.py --corpus mixed --reranker "$M" \
    --json-out "eval/results/ab_$(basename $M)_run2.json"
done
```

`EMBEDDING_INSTRUCTION=` (empty) is not optional if you want these exact
numbers — without it the Phase 1.2 query prefix changes the first stage.

`--reranker Qwen/Qwen3-Reranker-*` reaches `QwenRerankerScorer` through the
name-based leg of the loader, because `run_eval.py` hard-codes
`model_type: "cross-encoder"`. In application config, prefer the explicit form:

```yaml
reranker:
  enabled: true
  model_type: "qwen3"
  model_name: "Qwen/Qwen3-Reranker-4B"
  top_k: 10
```

---

## 9. Caveats, verbatim

* **Shared GPU.** Other agents ran MPS evals concurrently for the whole
  session. Quality is unaffected (all six runs are bit-identical on every
  metric), latency is not. Both runs are published; where they disagree, run 2
  is the lighter-load measurement.
* **The `docs` corpus is live repo content.** It was rebuilt at
  `2026-08-09T06:29:53Z` / `06:30:30Z` and md5-verified unchanged across all
  six runs, but it is *not* byte-identical to the corpus behind `BASELINE.md`
  (`Documentation/indexing_pipeline.md` changed in between). This is why bge was
  re-run.
* **`atlas7` and `hr` are saturated** at 1 and 2 chunks. Their 1.000s are
  plumbing checks, not retrieval measurements.
* **n = 72 on `mixed`.** A 0.021 nDCG@10 difference is one to two queries. The
  0.6B-vs-bge gap is inside that band; the 4B-vs-bge gap (0.075, and 0.201 on
  `docs`) is not.
* **No throughput or memory measurement.** Nothing here says what the 4B does to
  concurrent requests or to peak RSS alongside the generation model.
* **The `rerankers` 0.10.0 finding is version-specific.** A later release may add
  a Qwen3 backend; the name-based route in the loader would then still win, and
  should be revisited at that point.
* **`HF_HUB_DISABLE_XET=1` was required to download the 4B.** With the default
  xet transport the two safetensors shards sat at 0 bytes and made no progress
  across two attempts; with xet disabled the 7.5 GB fetched in 51 minutes at
  ~2.6 MB/s. Recorded because it will cost the next person an hour otherwise.


---

**Gate correction (2026-08-09, post-adoption):** the header's "`rag_system/main.py`
still ships `BAAI/bge-reranker-v2-m3`" described the tree at measurement time. Shipped
now: default profile reranker disabled; `RERANKER_MODEL` defaults to
`Qwen/Qwen3-Reranker-4B` for the opt-in path. See `eval/DECISIONS.md`.

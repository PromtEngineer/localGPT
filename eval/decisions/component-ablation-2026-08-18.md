# Component ablation study — 2026-08-18

**Question (user-directed, prompted by the HF multi-vector-encoder post): does
each pipeline component actually help?** Method: six arms, each the full
pipeline minus ONE component, on all five benches (120 rows/arm, v2 indexes,
temp-0 deterministic 4b judge; 3-voter Sonnet panels on every flipped cell of
the actionable arms — 72 cells, 4 split votes). Baseline = arm M/"fixed"
(92/120 4b-det).

## Results

| arm | component removed | 4b-det | panel-corrected net | verdict |
|---|---|---|---|---|
| norerank | Qwen3-Reranker-4B selection | 85 (−7) | not panelled (beyond noise) | **KEEP** — largest single contributor; losses concentrate on the chunk-rich corpora (acq −4, docs −2) and it re-loses hr_h05 |
| nodecomp | query decomposition | 95 (+3) | **+3 real** (4 gains, 1 loss, 0 noise) | **HURTING on single-turn** — splitting adds retrieval noise; docs +3, acq 30% faster |
| dense | hybrid FTS leg | 92 (±0) | — | quality-neutral; hybrid is 40–85% FASTER (lexical hits give the reranker cleaner candidates) → keep for latency |
| nolc | late-chunk leg | 90 (−2) | **−1 real** (4 losses, 3 gains, 9 noise) | ~1 row per 120 for 2x vectors + a second table per index → **defaulted OFF** (user decision) |
| noverify | verifier | 92 (0 flips) | — | annotates, never changes answers; verdicts byte-identical on all 120 → **defaulted OFF** (user decision); latency-only cost |
| noenrich | contextual enrichment | 93 (+1) | **−1 real** (6 losses, 5 gains, 4 noise) | ~1 row per 120 for ~5x index build time (12 min vs ~55 min); kept ON for now, prime simplification candidate |

Wall-time notes: rerank saves are modest (~10–14s/query on big benches, one
batched pass). Later arms' timings are polluted by Ollama contention (the
noverify arm measured SLOWER, which is impossible) — treat quality columns as
the reliable signal; timing conclusions only where the direction is
mechanical (dense-only doubling rfc wall time).

## Actions taken (user-directed)

- `retrieval.latechunk.enabled` → **False** in the default profile (one flag
  governs both index-time build and query-time leg).
- `verification.enabled` → **False** in the default profile.
- Both remain opt-in per config/request.

## Recommendations (not yet acted on)

1. **Decomposition**: the +3 comes from not SPLITTING single-turn queries;
   the same component also does multi-turn pronoun resolution (measured
   essential — multiturn.jsonl mt_07). Candidate: resolve-only mode — keep the
   decomposer call for context resolution, cap sub-queries at 1 unless the
   query is genuinely multi-part. Needs its own A/B + multiturn gate.
2. **Reranker latency**: its function is worth 7/120. A LateOn-149M
   MaxSim rescorer (Sentence Transformers v6 MultiVectorEncoder,
   hf.co/blog/multi-vector-encoder) could preserve most of that at ~25x less
   compute than the 4B cross-encoder. Worth an arm.
3. **Enrichment**: −1 real net for ~5x indexing cost. If indexing speed ever
   matters (large corpora), turning it off is nearly free in quality.

## Artifacts

Session scratchpad `ablation/`: answers_{arm}.jsonl, judged4bdet_{arm}.jsonl,
panel_{A,B}_prompts.jsonl, votes_{A,B}_{1..3}.jsonl, run/judge logs,
index_noenrich_{authored,rfc}/. Baseline: arm M/fixed artifacts.

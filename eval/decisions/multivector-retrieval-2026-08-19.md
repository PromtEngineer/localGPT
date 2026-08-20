# Multi-vector (late-interaction) first-stage retrieval — NOT adopted

**Date:** 2026-08-19 · **Status:** measured, not adopted · **Code:** env-gated hook kept (`MV_RETRIEVAL_ENDPOINT`)

## Question

The earlier experiment (experiments-resolveonly-maxsim-2026-08-19.md) tested MaxSim as a
*reranker* and lost. This one tests the other use from the HF multi-vector-encoder blog:
replacing the **dense leg of hybrid retrieval** with token-level MaxSim search — does
multi-vector *candidate generation* beat our single-vector dense leg?

## Model selection (from the blog's NanoBEIR NDCG@10 table)

- `lightonai/LateOn-regularized` — 149M, 128d, 0.6897 (top text-retrieval score) → arm mvA
- `LiquidAI/LFM2.5-ColBERT-350M` — 353M, 128d, 0.6864 (near-tied, different family) → arm mvB

Not run: GTE-ModernColBERT (dominated by LateOn, same lab), mxbai-edge-colbert-32m /
answerai-colbert-small (speed plays; irrelevant at our corpus sizes), mLateOn (multilingual).

## Setup

- Sidecar (`scratchpad/mvretrieval/server.py`) in the isolated ST6 venv (ST 6.0.0 needs
  transformers 5.x / torch ≥2.5; repo stays pinned to transformers 4.51.0 / torch 2.4.1 MPS).
  Pre-encodes each bench table's `text` column (the same enriched text the dense vectors
  were built from), serves brute-force MaxSim top-k over 5–683 chunks per corpus.
- Repo hook: `MultiVectorRetriever._mv_sidecar_search`, env-gated on `MV_RETRIEVAL_ENDPOINT`.
  Replaces only the dense leg; FTS leg, RRF fusion, Qwen3-Reranker-4B, generation all
  unchanged. Raises (never silently falls back) when the env is set, so an arm cannot
  quietly revert to the control. Zero extra calls when the env is unset.
- Verified active: a trap embedder in the dense path was never invoked; top hit for the
  RFC 2119 smoke query was the correct definition chunk.
- Control: current defaults (latechunk off, verification off) = **90/120** 4b-det
  (the nolc arm, re-confirmed by the resolve-only control).

## Results (120-row single-turn suite, deterministic 4b judge + blind 3× Sonnet panel on every flip)

| arm | model | 4b-det | flips vs control | panel-corrected net |
|-----|-------|--------|------------------|---------------------|
| mvA | LateOn-regularized | 85/120 | ↑1 ↓6 | **−2** (+1 real / −3 real; 4b wrong on 3 downs) |
| mvB | LFM2.5-ColBERT-350M | 88/120 | ↑3 ↓5 | **−1** (+2 real / −3 real; 4b wrong on 2 downs + 1 up) |

14 of 15 panel cells unanimous (single split: mvA rfc_q06, resolved grounded 2-of-1).

Real losses: mvA acq_q08, docs_d17, docs_d22; mvB docs_d08, docs_d17, rfc_q06.
Real gains: docs_d12 (both arms); mvB rfc_q24 — a long-standing crossref-residue loss
that multi-vector retrieval genuinely fixed.

## Verdict

Neither model beats the single-vector dense leg. mvB's −1 sits at the noise floor
(1–2 rows on n=24), mvA's −2 just below it — but the direction is negative in both arms
and adoption is not free: token-level vectors (~30–40x storage), a second model process
on a dependency stack the repo cannot host in-venv, and no latency win (the dense leg
was never the bottleneck; the reranker is). **Not adopted.** The env-gated hook stays
(zero-cost when unset) for future experiments.

## Confounds / scope notes

- Embedder sizes differ (dense harrier-oss-v1-0.6b ≈ 600M vs 149M/350M MV). The blog's
  leaderboard has no larger maintained text MV encoder to remove this.
- Document-scoped crossref-hop searches and the overview prefilter still use the dense
  path in the MV arms (shared with the control everywhere except the first stage).
- Possible ColBERT document-length truncation on long chunks was not instrumented.
- Metadata prefilters are unimplemented on the MV path (raises; bench uses none).

## Lead worth keeping

mvB fixing rfc_q24 while the control misses it shows MV retrieval surfaces genuinely
different candidates. If anything comes of this, it is a **third RRF leg**
(FTS + dense + MV) rather than a replacement — untested, and only worth trying with
storage/process costs solved.

## Addendum (same day, user-directed): the third-RRF-leg experiment — also NOT adopted

Arm `rrf3`: FTS + dense + LFM2.5-ColBERT MaxSim fused as THREE RRF legs
(`MV_RRF_LEG=1` alongside `MV_RETRIEVAL_ENDPOINT`; dense leg verified still firing
plus exactly one sidecar call per retrieval). Judged per user direction with
**Sonnet, not the 4b model** — both arms bulk-judged by 5 blind Sonnet agents
(one per corpus), flips panel-verified by 3 more.

| arm | Sonnet bulk | panel-corrected |
|-----|-------------|-----------------|
| control (current defaults) | 100/120 | — |
| rrf3 | 100/120 | **net −1** (+2 real: acq_q01, acq_q11; −3 real: docs_d05, docs_d07, docs_d22; 36/36 panel votes unanimous; docs_d20 flip dissolved — both arms ungrounded) |

The third leg trades rows ~1:1 instead of adding recall: RRF dilution shifts fused
rankings everywhere at once. The rfc_q24 gain from replacement mode did NOT survive
dilution to a third leg (rfc identical 20/24 both arms). Verdict: not adopted; both
env-gated modes stay as documented experimental hooks.

Calibration note: the same 120 control answers score 90/120 under the deterministic
4b judge and 100/120 under Sonnet — the 4b bulk judge under-credits ~10 rows/120,
consistent with every panel correction to date. Ordering conclusions survive; exact
4b totals should not be quoted as absolute quality.

Ops: the sidecar now persists document token-embeddings to disk
(`mvretrieval/emb_cache/`, keyed by model + exact corpus content), so encodings are
computed once per model+table and restarts reload from disk.

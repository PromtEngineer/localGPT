# Candidate-pool union fusion (FTS ∪ dense ∪ MV → reranker) — workload-dependent, not a default

**Date:** 2026-08-20 · **Judge:** Sonnet throughout · **Follow-up to:** paraphrase-robustness-2026-08-20.md

## What was tested

The fusion recommendation from the paraphrase study, implemented as `MV_UNION=1`
(retrievers.py): run all three legs (FTS, dense, LFM2.5-ColBERT MaxSim) at k=20 each,
skip the RRF top-k cut, and hand the FULL union (~47 unique candidates measured on rfc)
to the Qwen3-Reranker-4B to arbitrate. A disagreeing leg can then only ADD candidates —
never push another leg's find out of the pool. Rerank pool ~2.3x → per-query latency
roughly +30–40%.

## Results (Sonnet bulk + 3-voter panels on every flip)

| question set | control (2-leg hybrid) | 3-leg RRF | 3-leg UNION |
|---|---|---|---|
| paraphrased | 95 | 97 (net +2, split votes) | **99 — panel net +4 REAL** (6 gains / 2 losses, 24/24 votes unanimous) |
| original | 100 | 100 (net −1) | 95 — panel net **−3 REAL** (docs_d05/d07/d22, unanimous; 4 other flips dissolved as judge noise) |

## Findings

1. **The fusion diagnosis was correct.** Same legs as 3-leg RRF, only the fusion
   changed, and the paraphrase-set result went from +2 (with split votes) to +4
   (unanimous) — the largest verified gain of the whole multi-vector investigation.
   Gains include four rows paraphrasing had broken (docs_d08/d13/d19, rfc_q19).
2. **Union amplifies in both directions.** On original questions the wider pool admits
   distractors: all three real losses are docs — 608 chunks of near-duplicate
   documentation text — where the reranker, shown 47 candidates instead of 20,
   sometimes prefers a plausible-but-wrong chunk the RRF cut used to hide from it.
3. Net across both sets: +1. Not a default.

## Decision

Defaults unchanged (2-leg hybrid, RRF). The measured configuration guide:

- **Document-phrased queries** (users quote the docs' vocabulary): shipped 2-leg hybrid. Best cell: 100.
- **Paraphrase-heavy / conversational queries**: `MV_RETRIEVAL_ENDPOINT` + `MV_UNION=1`. Best cell: 99 (+4 real over 2-leg), at ~+30–40% latency and the MV sidecar/storage cost.

## Untested refinements (recorded, not run)

- Cap the union's per-leg contribution (top-10 per leg instead of top-20) to shrink the
  distractor surface on dense corpora.
- Gate the MV leg on FTS-confidence (add MV candidates only when the BM25 top score is
  weak — a proxy for "the query doesn't match document wording").

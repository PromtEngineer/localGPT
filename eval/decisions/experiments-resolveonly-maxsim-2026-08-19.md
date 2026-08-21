# Experiments: resolve-only decomposition + LateOn MaxSim reranker — 2026-08-19

Both user-approved follow-ups from the component ablation
(component-ablation-2026-08-18.md). **Verdicts: neither is adopted.** The
runs also surfaced a multi-turn cost of the latechunk-off default and a
2-row multi-turn regression from the fix-set window — both documented below.

## 1. Resolve-only decomposition — NOT ADOPTED

Hypothesis: keep the decomposer's context resolution (multi-turn needs it),
stop splitting. Implementation: `query_decomposition.resolve_only` flag —
same LLM call, same frozen prompts (single-turn dump byte-identity
unaffected), pipeline uses `resolved_query` instead of the splits.

- Single-turn (120 rows, 4b-det): resolveonly 90 vs its proper control
  (nolc, same latechunk-off defaults) 90 — **neutral**. The nodecomp arm's
  +3 came from using the RAW query; the decomposer's rewriting of
  self-contained questions costs the gains back.
- Multi-turn: 9/12, but the control (same defaults, resolve_only off) also
  scored 9/12 with identical misses — resolve-only exonerated there.
- Verdict: no benefit anywhere; flag stays in the code, default False, as a
  documented negative result. The single-turn win the ablation pointed at
  requires bypassing the decomposer entirely, which multi-turn cannot afford
  as a global default. A triage-style "history-empty → skip decomposer"
  fast path remains the plausible shape; not attempted here.

## 2. LateOn-149M MaxSim rescorer — NOT ADOPTED as default; kept as opt-in

Sentence Transformers v6 `MultiVectorEncoder` (hf.co/blog/multi-vector-
encoder). ST6 requires transformers 5.x / torch ≥2.5 — incompatible with the
repo's pinned MPS stack (an in-place install broke the venv and was rolled
back to transformers==4.51.0) — so the model runs OUT OF PROCESS in its own
venv behind a localhost scoring endpoint; `reranker.strategy: "maxsim"`
selects the thin client (`MaxSimRerankerScorer`).

- Quality vs the Qwen-4B control (nolc, same defaults): 90 → 87 (4b-det).
  Sonnet panels on all 13 down-flips: **11 REAL losses, 0 split votes** —
  including the rfc attribution/crossref rows (q06, q18, q20, q21). Not churn.
- Latency: the rerank stage drops from ~25–30s to ~1–2s; end-to-end docs
  62→18s, hr 22→9s, atlas7 14→8s, rfc 58→40s per query.
- Known confound: maxsim ran fixed top-10 selection (MaxSim scores are not
  calibrated probabilities, so the min_score threshold — gated on
  isinstance(QwenRerankerScorer) — does not apply). Some flips may be
  selection-policy, not scorer quality; untangling would need a
  qwen-top10-no-threshold control arm.
- Verdict: real quality cost (~3+ net real rows/120) for a large speed win.
  Not the default. Kept as an experimental opt-in for latency-sensitive
  setups; sidecar setup documented in scratch `maxsim/server.py` (not
  production-packaged).

## 3. Incidental finding: multi-turn attribution matrix

The multiturn set (12 conversations) was re-run under four configs on
current code:

| config | score | misses |
|---|---|---|
| defaults (lc off, verify off) | 9/12 | mt_03, mt_11, mt_12 |
| + resolve_only | 9/12 | same three |
| lc ON, verify off | 10/12 ×2 runs | mt_11, mt_12 |
| lc ON, verify ON (m1e-era config) | 10/12 | mt_11, mt_12 |

- **mt_03 is a real latechunk casualty**: the latechunk-off default costs
  this conversation (follow-up phrasing drifts from document wording; the
  document-context vectors were finding it). Single-turn ablation showed
  −1/120; multi-turn adds this cost the ablation gate could not see.
- **mt_11/mt_12 (both rfc) fail under EVERY current config** including the
  exact config that scored 12/12 three times on 2026-08-16 — the regression
  entered with the code changes between m1e and now (fix-set 3a8ebd9 and
  after). Prime suspect: the retry-judge temperature-0 pin changing
  keep/reject decisions on borderline rfc retrievals. Both rows are
  answer-entity conversations whose turn-2 retrieval is marginal. Queued
  for diagnosis; n=12 diagnostic set, 2 rows ≈ its noise floor, but the
  consistency across 4 runs makes it real.

## Standing recommendation for the user

Re-enabling latechunk recovers mt_03 (+1/12 multi-turn) and the −1/120
single-turn panel cost of removing it — at the price of 2x vectors and a
second table per index. The verifier flip remains cost-free for quality.
The mt_11/12 diagnosis is queued follow-up work either way.

## Artifacts

Scratch `ablation/`: answers/judged4bdet for resolveonly + maxsim,
panel_ms_prompts.jsonl, votes_ms_{1..3}.jsonl. Scratch `multiturn/`:
mt_answers_{mt_resolveonly,mt_ctrl_newdefaults,mt_lcon,mt_lcon2,mt_fullcfg}.jsonl.
Scratch `maxsim/`: sidecar venv + server.py + server.log.

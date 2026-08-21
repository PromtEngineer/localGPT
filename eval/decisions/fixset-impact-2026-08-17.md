# Code-review fix-set: measured impact (arm M / "fixed") — 2026-08-17

**Verdict: fix-set stays.** Net quality across the five benches is
flat-to-positive with one confirmed trade on rfc. All changes are
correctness fixes, not tuning; the indexes every prior number was measured
on are now known to have been structurally degraded.

## What was measured

Commits `3a8ebd9`+`3f78be4` changed index *content* (docling walk restored
heading paths + reading order; latechunk got real vectors past 8192 tokens
instead of identical CLS-garbage; split_markdown loop fixed; enrichment
windows stopped crossing document boundaries; FTS on `_lc` from build). So
all 5 bench indexes were REBUILT with the new code (v2 dirs; v1 kept), and
the full 120-row E2E suite + 12-conversation multiturn set re-ran on them.
Baseline: arm K answers (pre-fix code, v1 indexes) and the arm-I Sonnet
panel record.

Index deltas alone tell the story of bug 2.6: acq 13→82 chunks, hr 2→10,
atlas7 1→5, docs 366→608 (the old walk collapsed structure); rfc 683→683
(txt path — its change is enrichment + latechunk vectors, not chunking).

## Results

| bench | pre-fix | post-fix | judge |
|---|---|---|---|
| acq | 19/24 | 19/24 | deterministic 4b (temp 0) |
| atlas7 | 22/24 | 23/24 | deterministic 4b |
| hr | 19/24 | 21/24 | deterministic 4b |
| docs | 14/24 | 15/24 | deterministic 4b |
| authored total | 74/96 | **78/96** | deterministic 4b |
| rfc (full Sonnet panel) | 21/24 (arm I record) | **19/24** | 3-voter Sonnet, 1 split/72 |
| multiturn | 12/12 | **12/12** | mechanical |
| rfc mean wall | 59s | 51s (−14%) | — |

Sonnet arbitration of every 4b down-flip (12 rows × both arms, 3 voters):
7 of 12 were 4b noise; **5 real losses** — acq_q15, docs_d18, docs_d20,
rfc_q06, rfc_q24. Since the authored total still rose +4 *including* its 3
real losses, the authored gains are real and larger.

## The notable individual outcomes

- **hr_h05 RECOVERED (3/3)** — the "2 days extended family" row that the
  completeness-prompt clause could not fix (arm J, reverted). With hr now
  chunked 2→10, retrieval surfaces the bereavement section as its own
  chunk and the model states both durations. Confirms the arm-J diagnosis:
  it was an attention/structure problem, never an instruction problem.
  hr_h08/h13 remain failed (0/3) — same qualifier-omission style issue.
- **rfc q10, q15, q20 now PASS** — q10/q15 were the long-standing
  single-doc residue of arm I. The real latechunk vectors changed which
  candidates the lc leg contributes on long RFCs.
- **rfc q06, q24 now FAIL (real, panel-confirmed)** — previously-passing
  crossref rows lost by the same candidate redistribution. q04/q11/q17
  also fail at M, but q04 was already failing at arm K (pre-fix — panel
  batch 1) and q17 was arm I residue; they are not fix-set losses.

## Interpretation

rfc 21→19 is a −2 net from a changed retrieval surface: two real losses,
two-to-three real gains elsewhere in the same bench, on n=24 where the
documented noise floor is 1–2 rows. The losses were bought by removing
objectively-wrong behavior (identical garbage vectors indexed as real
data). Reverting correctness fixes to protect two bench rows would be
tuning-to-the-bench in reverse. The fix-set stays; q06/q24 join the
residue list as diagnosable candidates (both crossref rows — the crossref
residue remains the top rfc lever, item 1.8).

## Artifacts

`fiximpact/` in the session scratchpad: build_v2.log, e2e logs,
`authored_e2e_answers_fixed.jsonl`, `rfc_e2e_answers_m.jsonl`,
`mt_answers_m1e.jsonl`, `judged4bdet_{fixed,rfc_m}.jsonl`,
`panel{,2}_prompts.jsonl`, `votes_{1..3}.jsonl`, `votes2_{1..3}.jsonl`.
v2 indexes: `{authored_bench,rfc_shakedown}/product_index_v2/`.
Note: the fix-set session deleted the scratch rfc E2E runner; it was
reconstructed as `fiximpact/run_rfc_m2.py` — committing a repo-adapted rfc
runner (mirroring eval/multiturn/) remains open.

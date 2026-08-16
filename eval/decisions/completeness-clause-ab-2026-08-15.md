# Completeness-clause A/B (arm J) — 2026-08-15

**Verdict: REVERTED.** The clause does not fix the failure mode it was written
for, and its only aggregate movement is inside the documented noise band.

## What was tested

Cross-bench validation (eval/decisions/cross-bench-validation-2026-08-15.md)
found hr regressed 24→21 under the strict synthesis prompt because the model
answers the literal question and omits gold clauses attached to the value
("substitute day off", "effective 1 February 2026"). The candidate fix was a
new strict-prompt rule 8:

> When you state a value, rate, entitlement or rule, also include any
> condition, exception, alternative or effective date that the snippets attach
> directly to it. Answering the literal question while omitting an attached
> qualifier counts as an incomplete answer.

Per the anti-overfit rule established in the cross-bench record, the change was
gated on the full 5-bench suite (120 rows): rfc gold (arm J,
`rfc_shakedown/run_e2e_arm_j.py`) + the 4 authored corpora (ARM=clause,
`authored_bench/run_e2e_authored.py`). Identical indexes; both arms clean, 0
errors.

## Primary evidence: the three target rows

| row | gold qualifier missing at HEAD | arm J behaviour |
|---|---|---|
| hr_h05 | "2 days extended family" | **Answer byte-identical to HEAD.** Clause had no effect (the missing text is a parallel fact, not a qualifier "attached to the value" — the clause wording never reaches it). |
| hr_h08 | "effective 1 February 2026" | Still missing; answer also *dropped* the source-document mention HEAD included. |
| hr_h13 | "substitute day off within the same quarter" | Still missing. Instead the clause induced padding with irrelevant qualifiers (policy id, owner, Gothenburg, "9 public holidays"). |

Since hr_h05's answer is byte-identical to one the Sonnet panel already judged
wrong, arm J cannot recover the hr regression regardless of judging. The
clause fails its purpose by construction — no Sonnet panel spend needed for a
non-adoption.

## Secondary evidence: 4b coarse screen (not the deciding measurement)

| bench | HEAD (4b) | arm J (4b) |
|---|---|---|
| acq | 18/24 | 16/24 |
| atlas7 | 19/24 | 22/24 |
| hr | 19/24 | 20/24 |
| docs | 13/24 | 14/24 |
| authored total | 69/96 | 72/96 |
| rfc (same-judge I vs J) | 14/24 | 14/24 |

+3/96 with 16 downward flips and 19 upward flips is churn, not signal
(1–2 rows on n=24 is the established noise floor; this is that, four times).
New losses include hr_h04 — the parental-leave row now omits a qualifier, the
exact failure mode the clause was meant to prevent. Mechanical substring:
docs 17→14, rfc 10→9. Mean answer length ~unchanged (576→555 chars), so the
clause did not even make answers systematically more complete.

## Interpretation

The hr −3 is a *selective-attention* failure, not an instruction-shortage
failure: the model already has the qualifier in context and a rule telling it
to be thorough (rule 6). Adding a more specific instruction moved which rows
it pays attention to, not how completely it answers. Prompt-wording churn that
does not achieve its stated purpose is risk without benefit — and adopting it
anyway on the strength of noise-band movement elsewhere (atlas7 +3 under a
judge documented to contradict itself) would be exactly the
tuned-to-the-bench drift this eval program exists to prevent.

## Status

- Rule 8 reverted; strict prompt back to the arm-I form (7 rules).
- hr 24→21 stands as a known, documented cost of the strict prompt
  (style, not grounding). Future candidates for it should change *what the
  synthesizer attends to* (e.g. qualifier-aware snippet formatting), not add
  more prompt rules; any such change re-runs this same 5-bench gate.
- Artifacts: `rfc_e2e_answers_j.jsonl`, `authored_e2e_answers_clause.jsonl`,
  `judged4b_clause.jsonl`, `judged4b_rfc_{i,j}.jsonl` in the session scratchpad.

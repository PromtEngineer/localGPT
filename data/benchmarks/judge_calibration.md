# LLM-judge calibration — qwen3.5:9b answer grader

**Question addressed:** the 9B answer judge (`qwen3.5:9b`, `JUDGE_SYSTEM` in
`src/marag/eval/answer_eval.py`) has never been validated against reference labels. This report
supplies that validation: a careful manual re-grade of already-stored answers against each
benchmark's *own* gold acceptance criteria, compared to the stored 9B verdict.

**Method (CPU-only, no models, no server, no evals):** read stored `runs/answers_*.json`; for each
item produce an independent reference label (`correct: yes/no`) by reading the model answer against
the gold answer and its "Accept also"/rubric/do-not-accept notes, applying the *gold's own*
acceptance bar (not stricter); then compare to the stored `correct` field. The judge model is
`qwen3.5:9b` (pinned as the fixed measuring instrument in every config, e.g. `configs/default.yaml`,
`configs/gemma4_only.yaml`; the `provenance` block is `null` in these particular run files, but the
judge identity is fixed by config, not by the run). Reference labels for all 104 graded items are in
`judge_calibration_labels.json`.

## Sample

104 unique `(question, answer, gold)` gradings, deduplicated, most-recent run per `(dataset, mode)`:

| Slice | Source run | n | judge PASS / FAIL |
|---|---|---|---|
| **Aggregate/rubric** | `answers_financial_agg_agentic_20260717_211217` (7) + `answers_research_agg_agentic_20260717_212006` (5) | 12 | 7 / 5 |
| **Standard — agentic** | most-recent agentic per domain: financial_docs `…193609`, legal_docs `…073748`, health_docs `…221617`, research_papers `…064519` | 60 | 52 / 8 |
| **Standard — single-shot** | most-recent single-shot per domain (`…rejudged` where present); all of legal + every judge=PASS item elsewhere, selected to hunt false-PASSes and cover legal FAILs | 32 | 19 / 13 |

The single-shot set was added deliberately: agentic runs are high-accuracy, so genuinely-wrong
answers (the place a *false-PASS* would hide) are scarce there. Single-shot RAG is far weaker
(accuracy 0.13–0.47), giving 13 genuinely-wrong legal answers plus 19 judge=PASS items to audit.

## Headline results

- **Overall agreement: 102/104 = 98.1%** (Cohen's κ = 0.947).
- **Aggregate/rubric slice: 10/12 = 83.3%.**
- **Standard slice: 92/92 = 100.0%** (agentic 60/60, single-shot 32/32).
- **Judge false-FAILs: 2. Judge false-PASSes: 0.**

Overall confusion matrix (rows = judge, cols = reference):

|              | ref PASS | ref FAIL |
|--------------|:--------:|:--------:|
| **judge PASS** | 78 | **0** (false-PASS) |
| **judge FAIL** | **2** (false-FAIL) | 24 |

Per-slice confusion:

**Aggregate (n=12, 83.3%)**

|              | ref PASS | ref FAIL |
|--------------|:--------:|:--------:|
| judge PASS | 7 | 0 |
| judge FAIL | **2** | 3 |

**Standard (n=92, 100%)**

|              | ref PASS | ref FAIL |
|--------------|:--------:|:--------:|
| judge PASS | 71 | 0 |
| judge FAIL | 0 | 21 |

**The hypothesis is confirmed.** The judge is *perfectly* reliable on the standard, single-fact /
exact-match golds (92/92, including 21 correct FAILs on wrong-number and missing-info answers and
71 correct PASSes). Every disagreement is in the aggregate/rubric slice, and every disagreement is a
**false-FAIL** — the judge under-credits multi-part / rubric answers. It never once passed a wrong
answer (0 false-PASSes across 104 items, including 13 genuinely-wrong single-shot legal answers and
19 audited judge=PASS single-shot answers).

## The two disagreements (both false-FAILs, both aggregate)

### fin_agg_q07 — CONFIRMS the review claim

- **Question:** which risk themes recur across the five macro reports; name the major ones.
- **Gold rubric (verbatim):** *"the answer must name at least 3 of the following 5 recurring
  themes … Grade PASS if at least 3 of the 5 numbered themes are clearly named; FAIL otherwise …
  the two 'supporting' themes are optional extras, not required and not penalized."*
- **Model answer** clearly names: (1) tariffs / trade-policy tensions, (4) inflation, (3) fiscal
  deficits / public debt — and also (2) elevated policy uncertainty in passing. That is ≥3 of the 5
  numbered themes → **PASS by the rubric.** It then added a 4th, non-listed theme, NBFI /
  non-bank-financial-intermediary vulnerabilities, before being cut off.
- **Stored 9B verdict: FAIL.** Judge reason: *"The model answer includes 'Financial System
  Vulnerabilities in Non-Bank Financial Intermediaries' which is not among the 5 required themes …"*
- **Verdict: MISGRADED (false-FAIL).** The judge failed the answer **for including an extra theme**,
  exactly as the review alleged. The rubric requires only ≥3 of 5 and explicitly does not penalize
  extras; `JUDGE_SYSTEM` itself says "Extra correct detail is fine." The judge ignored both. The
  ≥3 required themes were all present well within the visible text, so this is not a truncation
  artifact — it is a grading-logic error.

### fin_agg_q06 — also misgraded (re-checked per task)

- **Question:** which of the 10 annual filers paid cash dividends in the latest FY; name every payer.
- **Gold's own fail-criteria (verbatim):** *"Grading fails if Intel is omitted or if
  AMD/Amazon/Tesla/Pinterest are included."* Plus: *"Accept also: answers that flag Intel's Q4-2024
  suspension — Intel still counts as a payer …"* (i.e. flagging the suspension is optional).
- **Model answer** names exactly the correct set — NVIDIA, Alphabet, Apple, Meta, Intel, Qualcomm —
  and correctly lists AMD/Amazon/Tesla/Pinterest as non-payers. Intel is present; no wrong inclusions
  → **PASS by the gold's own criteria.**
- **Stored 9B verdict: FAIL.** Judge reason: *"incorrectly states Alphabet paid $3.5 billion instead
  of the gold's $7,363M and omits Intel's critical Q4 2024 dividend suspension detail required by the
  grading criteria."*
- **Verdict: MISGRADED (false-FAIL).** Both stated reasons are invalid:
  1. The question asks *who paid*, not *how much*. Alphabet's `$3.5B` is a **volunteered extra
     number the question never requested**; the gold's `$7,363M` is supporting evidence, not a graded
     value. The judge applied its literal "numbers must match the gold values" rule to an incidental
     figure that is not part of the required answer.
  2. The Intel Q4-suspension flag is marked **"Accept also"** in the gold — explicitly *optional*.
     The judge invented it as a requirement ("required by the grading criteria"), which the gold text
     contradicts. Intel is correctly named as a payer, which is all the gold requires.

Both false-FAILs share one root cause: **on set / rubric / enumeration golds the judge grades against
incidental details (an extra theme, an un-requested number, an optional flag) instead of against the
gold's stated acceptance bar.** Applying these two fixes would raise financial_agg accuracy from
**3/7 (42.9%) to 5/7 (71.4%)** — material, because financial_agg is the graph-RAG decision gate.

## The judge is working correctly everywhere else

The 24 correct FAILs and 78 correct PASSes show the instrument is sound on its core job. Representative
correct FAILs (true-negatives) — none of these should flip:

- **Wrong numbers:** `fin_q02` (reports Compute&Networking $116,193M/89% where the question asks the
  Data Center end-market, gold $115,186M/88%); `fin_q05` (MAPCs read as 2,572M vs gold 171M);
  `hlt_q10` (infant RSV peak read as ~0.9/1,000 vs gold ~0.75, outside the ±10% figure tolerance);
  `hlt_q15` (60%/35%/1.09M vs gold ~67%/~32%/1.25M); `leg_q04-SS` (94.5pp vs a wrong 81.0pp);
  `rp_agg_q04` (MS MARCO named but count 10 vs gold's load-bearing 12).
- **Missing / refused required content:** `fin_q01`, `fin_q09` (question explicitly asks both years'
  Google-Cloud figures; model says "not provided"), `fin_q15` (World-Bank half "missing"), `rp_q13`
  (figure-read half declared unreadable), and 11 of 13 single-shot legal FAILs where the model
  answered one hop and declared the other "not in the provided context."

Two design behaviors specifically checked and **not** found to misfire in this sample:
- **Punishing a design-mandated hedge:** no benchmark gold in the sample *mandates* an "uncertain"
  answer, so the blanket rule *"An answer that says the information is missing is incorrect"* only
  ever fired on genuinely-incomplete answers (correct). No false-FAIL of this type.
- **3000-char answer truncation cutting graded content:** in every enumeration/rubric item the
  required content appeared within the first ~1,500 chars, so truncation changed **0** verdicts
  here. It remains a latent risk for longer enumerations (see fix 3).

## Proposed `JUDGE_SYSTEM` changes (targeted, non-overfitting)

Ranked by number of stored verdicts corrected in this 104-item sample. Each rule is scoped so it
**cannot** create a false-PASS: it only relaxes grading for items the gold marks as extra/optional or
counts by a "≥N" rule; it never loosens a required value or set-membership check (verified: none of
the 21 correct standard FAILs, nor `fin_agg_q01`/`q03`/`rp_agg_q04`, would flip under any rule below).

**Fix 1 — Rubric "name ≥N of these" golds: grade by counting, extras never fail. → flips `fin_agg_q07` (FAIL→PASS).**
Add to `JUDGE_SYSTEM`: *"If the gold states a rubric such as 'name at least N of the following', grade
PASS iff at least N of those listed items are clearly named. Items the answer adds that are NOT on the
list — extra themes, extra entities, extra detail — are never grounds for FAIL."* This is the single
highest-value change for the aggregate class and makes the existing "Extra correct detail is fine"
line enforceable rather than advisory. Flips exactly 1 stored verdict; `rp_agg_q05` (already PASS)
stays PASS.

**Fix 2 — Grade the required answer only; honor the gold's own acceptance clauses. → flips `fin_agg_q06` (FAIL→PASS).**
Add: *"Grade only the facts the QUESTION asks for. For set/enumeration questions ('name every …',
'which companies …'), grade PASS/FAIL on set membership against the gold set; do NOT fail because an
incidental number the answer volunteered (one the question did not request) disagrees with the gold's
supporting figures. Treat any 'Accept also', 'do not penalize', 'optional', or 'not required' clause
in the gold as authoritative — never require something the gold marks optional."* Flips exactly 1
stored verdict. The "numbers must match" rule stays fully in force for values the question actually
asks for (so `fin_q02`, `fin_q05`, `rp_agg_q04`, etc. still correctly FAIL).

**Fix 3 — Make accept-bands authoritative in the gold, and lift the 3000-char truncation for enumerations. → 0 flips here (preventive).**
(a) Bake numeric accept-bands / tolerances into the gold answer text (e.g. `hlt_*` "±10%",
`fin_agg_q04` "$248B–$249B") so the judge reads them as data rather than re-deriving a stricter bar.
(b) In `eval_answers`, raise or remove `result['answer'][:3000]` (and the 2000-char stored cap) for
`answer_type in {rubric, entity_set, doc_set, paper_list, count_and_set}`, so a long enumeration's
graded items are never cut before the judge sees them. Corrects no verdict in this sample but removes
a latent false-FAIL source for future longer answers.

### Not recommended (would overfit / risk false-PASSes)

- Do **not** add blanket numeric leniency — the judge's strict number matching is exactly why it
  scored 0 false-PASSes; keep it for values the question asks for.
- Do **not** weaken the "missing information = incorrect" rule for standard questions; it produced
  only correct FAILs here. Scope any exception strictly to golds whose own answer is "not available."

## Bottom line

The 9B judge is a trustworthy instrument for standard single-fact benchmarks (100% agreement, κ high,
zero false-PASSes) but systematically **under-credits aggregate/rubric answers** (2/2 errors are
false-FAILs caused by penalizing extra/optional/incidental content). Two small, tightly-scoped
`JUDGE_SYSTEM` edits (Fixes 1 and 2) correct both known misgradings and lift the graph-RAG-gate
financial_agg score from 42.9% to 71.4% without introducing any false-PASS.

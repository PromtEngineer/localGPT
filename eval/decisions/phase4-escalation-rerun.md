# Roadmap 4.1 (full-document escalation) — re-run after the context-window fix

Date: **2026-08-12**. Branch `rearchitect/evidence-gated-aug-2026`, HEAD **007d0b6**,
interpreter `.venv/bin/python`, Ollama `localhost:11434`, generation `qwen3.5:9b`,
judge/enrichment `qwen3.5:4b`.

This re-runs the exact A/B that `eval/decisions/phase4-answer-quality.md` §9 put on
HOLD, under the condition that file set: *"Fix the context-window bug in §8 first,
then re-run this exact A/B; if the lift survives a prompt that actually fits, adopt."*

**Answer: the lift does not survive. It was the truncation artifact §6 predicted it
might be.** Proposed call in §7 below.

No file under the repo was edited by this wave. Everything below lives in
`SCRATCH = /private/tmp/claude-501/-Users-prompt-videos-localgpt-08082026/4d62420b-7ab2-4be1-90f2-708d7bae9146/scratchpad`.

---

## 1. Setup verification

### 1.1 The fix is active

`SCRATCH/verify_num_ctx.py`, the 99k-char probe that previously returned
`prompt_eval_count = 8194` and lost a fact planted at character 0:

```
sizing helper: OK
prompt chars: 98763
prompt_eval_count = 17351  (was 8194 before the fix)
eval_count        = 9
answer            = 'ZEBRA-7741'

context window fixed: PASS
front fact recovered: PASS
```

`/api/ps` during the runs reports `qwen3.5:9b` loaded at `context_length 32768`.

### 1.2 Indexes — opened, not rebuilt

```
w3b_acq       13 chunks     w3b_acq_lc       13
w3b_acqdocs  373 chunks     w3b_acqdocs_lc  373
```

Matches the 13 / 373 recorded on 2026-08-09. The 2026-08-09 result files in
`w3b_indexes/results/*.jsonl` were read only; every new file carries the `_fix`
suffix.

### 1.3 The only code change since the 2026-08-09 runs is the fix itself

```
007d0b6 2026-08-12 13:05  fix: size Ollama num_ctx per request
c2d9f48 2026-08-09 18:02  docs: Phase-4 verdicts …   (Documentation/*.md ONLY — 4 files, no rag_system/)
e3c3d60 2026-08-09 18:02  eval: acquisition corpus …
```

The 2026-08-09 arms finished at ~17:34, before `c2d9f48`; `c2d9f48` touches no
`rag_system/` file. The `acqdocs` index was built at 16:24 that day and was **not**
rebuilt, so both dates retrieve byte-identical chunks. Attribution of any change
below to the num_ctx fix is therefore clean.

### 1.4 Harness — reused unmodified

`w3b_common.py`, `w3b_run.py`, `w3b_judge.py` used as-is, same flags as
`chain1.sh`/`chain3.sh`. Two additions, both observation-only:

* `w3b_ctxprobe.py` — wraps `_warn_if_truncated` (the one hook that already sees both
  the request payload and the final response on all three completion paths) and
  `Agent.run`, to log per call: model, stage, prompt chars, requested `num_ctx`,
  served `prompt_eval_count`, and whether the warning condition fired. Changes no
  behaviour.
* `w3b_run_fix.py` — imports `w3b_run`, installs the probe, calls `w3b_run.main()`.

`think:false` on the generation model is still monkeypatched in the scratch runner
(the synthesis-path thinking bug is unfixed); symmetric across arms, as before.

---

## 2. Prompts now fit — the direct evidence

Across all four arms: **249 completion calls, 0 truncation warnings.**

| arm | calls | warned | max single-call `prompt_eval_count` | max `num_ctx` requested |
|---|---|---|---|---|
| `ad_fire_esc_off_fix` | 27 | **0** | 20 831 | 32768 |
| `ad_fire_esc_on_fix` | 27 | **0** | 27 516 | 32768 |
| `acq_esc_off_fix` | 102 | **0** | 9 365 | 16384 |
| `acq_esc_on_fix` | 93 | **0** | 9 966 | 32768 |

9b synthesis calls that exceeded the old 8194 ceiling: **9/9** in each `acqdocs` arm,
36/47 and 33/42 in the `acq` arms. Representative rows (`acqdocs`, escalation ON):

```
docs_d20  chars=120272  num_ctx=32768  prompt_eval_count=27516  warned=False
docs_d13  chars=110102  num_ctx=32768  prompt_eval_count=26352  warned=False
docs_d05  chars=104394  num_ctx=32768  prompt_eval_count=23512  warned=False
acq_q04   chars= 95682  num_ctx=32768  prompt_eval_count=22059  warned=False
```

**No row filled its 32768 window** (highest 27 516, i.e. 84% of the window). There is
therefore **no residual-truncation subset to report separately** — the confound is
gone for these arms, not merely reduced.

### 2.1 What this changed vs 2026-08-09, per corpus

`token_usage.by_stage.synthesis.prompt_tokens` is a **sum over 2 calls**, so compare
it only to itself:

| corpus | 2026-08-09 `prompt_tokens` (esc OFF) | 2026-08-12 (esc OFF) |
|---|---|---|
| `acqdocs` (373 chunks) | 8298 – 8311 on 8 of 9 rows — pinned at the ceiling | 11 242 – 20 940 |
| `acq` (13 chunks) | 9351 – 29 700 | 9 351 – 30 298 |

This is the key asymmetry, and it explains everything downstream. The `acq` corpus
produces ~9.3k-token synthesis calls, which fit the old server window whenever the
box was uncontended — 6 of its 24 rows report **byte-identical** `prompt_tokens` on
both dates (`acq_q01` 9353, `acq_q02` 9357, `acq_q03` 9351, `acq_q04` 9465, `acq_q05`
9357, `acq_q24` 9356), which cannot happen to a prompt being clipped to a contended
slot. The `acqdocs` corpus
produces 14k–27k-token calls, which never fit, and 8 of its 9 rows sat exactly on the
8194 ceiling. **The 2026-08-09 `acqdocs` baseline was measuring a crippled system;
the `acq` baseline mostly was not.**

---

## 3. Fire sets — new vs old

| cell | 2026-08-09 fire set | 2026-08-12 fire set | delta |
|---|---|---|---|
| `acqdocs` fire subset (decomp OFF) | `acq_q04, acq_q09, acq_q12, docs_d03, docs_d05, docs_d13, docs_d20` (7/9) | `acq_q04, acq_q07, acq_q09, acq_q12, docs_d03, docs_d05, docs_d13` (7/9) | **+`acq_q07`, −`docs_d20`** |
| `acq` product default (decomp ON) | `acq_q12` (1/24) | `acq_q04, acq_q12` (2/24) | **+`acq_q04`** |

Same size on the deciding cell, one member swapped, exactly as the brief anticipated
(the evidence-sufficiency retry's reformulation is a fresh LLM sample each run). All
escalation events fired on `dense_contrast` below the 0.12 threshold. **No event in
either new arm hit the 6000-token budget cap** (`truncated: false` on all 9), whereas
2026-08-09's `docs_d20` reported `truncated: true` at 38/45 chunks.

---

## 4. CELL 1 (deciding) — `acqdocs` fire subset, decomposition OFF, n=9

`k/5` votes, majority ≥3/5. `OLD` columns are the 2026-08-09 files, recomputed from
`judge5_ad_fire_esc_*.jsonl`, not copied from prose.

| qid | fired | **OFF k/5** | **ON k/5** | dir | OLD OFF | OLD ON | old dir | pt off→on | OLD pt off→on |
|---|---|---|---|---|---|---|---|---|---|
| `acq_q04` | Y | **4/5** | **0/5** | HARM | 0/5 | 2/5 | = | 20940 → 22168 | 8303 → 14732 |
| `acq_q07` | Y | **5/5** | **5/5** | = | 0/5 | 0/5 | = | 17134 → 17535 | 8304 → 8304 |
| `acq_q09` | Y | **4/5** | **0/5** | HARM | 0/5 | 4/5 | HELP | 14586 → 21428 | 8311 → 8311 |
| `acq_q12` | Y | 0/5 | 0/5 | = | 0/5 | 0/5 | = | 11242 → 13064 | 11242 → 13064 |
| `docs_d03` | Y | **4/5** | **5/5** | = | 0/5 | 4/5 | HELP | 20384 → 22216 | 8298 → 8298 |
| `docs_d05` | Y | **4/5** | 2/5 | HARM | 0/5 | 0/5 | = | 19383 → 23616 | 8298 → 8298 |
| `docs_d13` | Y | 1/5 | 0/5 | = | 0/5 | 0/5 | = | 16941 → 26456 | 8298 → 8298 |
| `docs_d15` | . | **4/5** | **5/5** | = | 0/5 | 5/5 | HELP | 20412 → 24684 | 8302 → 8302 |
| `docs_d20` | . | **4/5** | **4/5** | = | 0/5 | 2/5 | = | 19934 → 27625 | 8303 → 8303 |

**Majority-pass tallies:**

| subset | n | OFF | ON | 2026-08-09 OFF | 2026-08-09 ON |
|---|---|---|---|---|---|
| all rows | 9 | **7/9** | 4/9 | **0/9** | 3/9 |
| new fired rows | 7 | **5/7** | **2/7** | — | — |
| old fired rows | 7 | 5/7 | 2/7 | **0/7** | **2/7** |
| union of fire sets | 8 | 6/8 | 3/8 | — | — |

**The headline number.** The escalation-**off** baseline on this subset went from
**0/9 to 7/9** with no change other than the prompt fitting. The escalation-**on**
arm went from 3/9 to 4/9. The 2026-08-09 gap that motivated 4.1 (0/7 → 2/7) is gone;
the same rows now read **5/7 → 2/7**, i.e. the sign has flipped.

`acq_q09` is the clearest single illustration, because §4 of the decision file quoted
it as the poster child. 2026-08-09: baseline 0/5 ("the documents you supplied describe
a RAG system architecture" — it never saw the acquisition PDFs), escalated 4/5.
2026-08-12: **baseline 4/5**, answering from the retrieved chunks directly —

> "**Previously Disclosed Debt**: … **$1,500,000**. **Extra Borrowing Found Later**:
> … an additional **$175,000** in previously undisclosed capital lease obligations."

— and the escalated arm 0/5. The lift was the truncation, exactly as §6.2 warned.

---

## 5. CELL 2 (product default) — `acq`, n=24, decomposition ON

| qid | fired | OFF k/5 | ON k/5 | dir | OLD OFF | OLD ON | old dir |
|---|---|---|---|---|---|---|---|
| `acq_q04` | **Y** | **4/5** | **0/5** | HARM | 0/5 | 0/5 | = |
| `acq_q12` | **Y** | **4/5** | 2/5 | HARM | **4/5** | 1/5 | HARM |

(Full 24-row table in `SCRATCH/w3b_indexes/logs/fixreport_full.txt`.)

| subset | n | OFF | ON | 2026-08-09 OFF | 2026-08-09 ON |
|---|---|---|---|---|---|
| all rows | 24 | 10/24 | 10/24 | 11/24 | 10/24 |
| fired rows | 2 | **2/2** | **0/2** | 1/1 | 0/1 |

Whole-set is a wash both dates, as expected (22 of 24 rows are identical work in both
arms). The `acq` whole-set barely moved across dates (11/24 → 10/24 OFF), consistent
with §2.1: this corpus's prompts largely fit even before the fix. **Every fire under
the true product default, on both dates, is a mechanical regression** — now 2 for 2.

---

## 6. The measurement caveat that qualifies §4 and §5 — and does not rescue 4.1

The decision file's §7 warned the judge is "strict and, in this framing, noisy." On
these arms it is worse than noisy: **on the fired rows it returns verdicts its own
stated reasons contradict.** Every ON-arm "HARM" above was hand-checked against gold.

`acq_q04`, cell 1, ON arm, judged **0/5**. Verbatim answer:

> "the single largest client is **MegaCorp**, which accounts for **28%** of
> StartupXYZ's revenue"

against gold *"The largest customer, MegaCorp, accounts for 28% of revenue."* The gold
fact is present verbatim. Two of the five judge reasons are flatly false — *"the
EVIDENCE explicitly states that the single largest client is TechCorp Industries"* and
*"contains no information about MegaCorp's name or its 28% revenue share."* A third
cites the verifier's appended `[Confidence: 90%]` suffix as grounds for rejection.

`acq_q12`, cell 2, ON arm, judged **2/5**, answer contains *"47 employees … Engineering
with 32 … Sales with 8 … Operations with 7"* — the complete gold fact — while two of
its three sampled reasons say the answer *"accurately reflects all specific workforce
numbers and functional splits."* The verdict contradicts the reason.

The verifier's `[Confidence: N%] [Warning: … Groundedness: False]` suffix appears on
7/9 answers in **both** cell-1 arms, so it is a noise source, not an arm-specific bias.

**Manual adjudication of every fired row** (is the gold fact present in the answer?):

| cell | qid | OFF | ON |
|---|---|---|---|
| 1 | `acq_q04` | yes | yes |
| 1 | `acq_q07` | yes | yes (in a 35 290-char runaway answer) |
| 1 | `acq_q09` | yes | yes, but prefaced *"there is no information regarding borrowing"* |
| 1 | `acq_q12` | no (abstains) | no (abstains) |
| 1 | `docs_d03` | yes | yes |
| 1 | `docs_d05` | yes | yes |
| 1 | `docs_d13` | no (partial) | no |
| 2 | `acq_q04` | yes | yes |
| 2 | `acq_q12` | yes | yes |

**Manual: 7/9 → 7/9 across all fired rows in both cells — an exact wash.**
**Mechanical judge: 7/9 → 2/9.**

The two instruments disagree on whether escalation *harms*. They agree completely on
the only question the HOLD asked: **there is no lift.** No fired row in either cell
gains a gold fact it did not already have without escalation.

---

## 7. Proposed verdict

### **4.1 full-document escalation — do not adopt. Convert the HOLD to a REJECT as a shipped default; keep the flag and the code.**

**The one line it rests on:** the 0/7 → 2/7 lift was produced by front-truncation, not
by document reassembly — with prompts that actually fit, the escalation-off baseline
on the identical fire subset goes from 0/9 to **7/9** and escalation adds nothing on
top of it (**5/7 → 2/7** mechanically, **5/7 → 5/7** by hand), while both fires under
the true product default remain regressions.

Supporting points:

* The HOLD was explicitly conditional on the lift surviving. It did not survive; it
  inverted. The condition resolves to "do not adopt."
* The mechanism §6.2 hypothesised is now confirmed rather than inferred: the escalated
  block survived truncation because it was appended at the tail while top-ranked
  chunks were deleted from the front. Remove the truncation and the benefit vanishes.
* The real fix for this corpus was the context-window bug, exactly as §8 predicted
  ("any context-window guard … would change the answers on this corpus more than
  either Phase-4 feature does"). It moved the deciding cell by **7 rows**; escalation
  moved it by 0 (manual) to −3 (mechanical).
* Evidence for active *harm* is weak and should not be cited: all four mechanical
  HARM rows are judge artifacts on inspection (§6). "No benefit" is the defensible
  claim; "harmful" is not.
* Keep the code behind the flag. It is unfalsified for the case it was designed for
  (a prompt that fits *and* a document whose ordering matters); this corpus at
  `chunk_size=512` with a 32k window simply never presents that case.

### Confidence and limits

* n = 7 fired rows in the deciding cell, 2 in the product-default cell. Small, as
  before. But the decisive number is not the fired-row delta — it is the **7-row move
  in the baseline**, which is far outside the ~1-row noise floor §7 established.
* The judge remains the weakest instrument here. The verdict is stated so that it
  holds under *both* the mechanical and the manual reading.
* No wall-clock number is used as evidence.
* `atlas7` / `hr` still excluded (1 and 2 chunks). 4.2 and 4.3 untouched.

---

## 8. Anomalies and new backlog

1. **`acq_q07`, cell 1, escalation ON, produced a 35 290-character answer** (vs 1 400
   in the OFF arm) that degenerates into verbatim regurgitation of
   `design_rationale.md` text about MiniCheck — unrelated to the HSR question it
   answered correctly in its first paragraph. A whole-document block in context can
   send the 9b model into transcription. New failure mode, only visible now that
   prompts are not truncated.
2. **The judge returns verdicts contradicted by its own reasons** on multi-clause gold
   facts (§6), and is measurably perturbed by the verifier's appended
   `[Confidence: …] [Warning: … Groundedness: False]` suffix — one reason cites the
   90% confidence figure as grounds for rejecting the answer. The suffix is part of
   the answer string the judge scores. Either strip it before judging or stop
   appending it to the user-visible answer.
3. **The 2026-08-09 `acqdocs` numbers in `eval/decisions/phase4-answer-quality.md` §4
   should be treated as void**, not merely confounded: that baseline was 0/9 because
   it was reading truncated context, and it is 7/9 on identical inputs today. §3
   (`acq`) is largely unaffected (§2.1).
4. The `think:false` synthesis-path bug (§8 of the decision file) is still unfixed and
   still requires the harness monkeypatch.

---

## 9. Files produced (all under SCRATCH, nothing in the repo)

```
w3b_indexes/results/ad_fire_esc_on_fix.jsonl     ad_fire_esc_off_fix.jsonl
w3b_indexes/results/acq_esc_on_fix.jsonl         acq_esc_off_fix.jsonl
w3b_indexes/results/judge5_*_fix.jsonl           (k=5 verdicts, 4 files)
w3b_indexes/results/ctx_*_fix.jsonl              (249 per-call num_ctx / prompt_eval_count records)
w3b_indexes/logs/run_*_fix.log, *_fix.stdout.log
w3b_indexes/logs/fixreport_full.txt              (full 24-row cell-2 table)
w3b_ctxprobe.py  w3b_run_fix.py  w3b_fixreport.py  chain_fix1.sh  chain_fix1b.sh
```

Original 2026-08-09 `*.jsonl` arm outputs were read but never modified.

### Execution note

The first background chain was killed by the harness partway through
`acq_esc_off_fix` (20 of 24 rows written) and a `setsid`-based relaunch failed
(`nohup: setsid: No such file or directory` — not present on macOS). It was resumed
with plain `nohup`; `w3b_run.py` skips ids already present in its `--out` file, so
rows 21–24 of that arm ran in a second process. Consequence: the per-model `num_ctx`
ratchet restarted for those 4 rows. Since sizing is per-prompt and the ratchet only
ever *grows* the window, no prompt was under-sized — 0 warnings across all 102 calls
of that arm. No query ran twice; no row was overwritten.

---

## 10. Gate validation (2026-08-12)

Every deciding number above was independently recomputed from the raw
`*_fix.jsonl` files by the gate, trusting nothing in this report's prose:

* Cell 1 per-row `k/5` table and majorities: reproduced exactly (7/9 OFF, 4/9 ON;
  fired subset 5/7 → 2/7).
* Cell 2 majorities and fired rows: reproduced exactly (10/24 both arms;
  `acq_q04` 4/5 → 0/5, `acq_q12` 4/5 → 2/5).
* Fire sets: extracted from the `document_escalation` payloads directly —
  cell 1 fired 7 (`+acq_q07`, `−docs_d20` vs 2026-08-09), cell 2 fired
  `acq_q04` + `acq_q12`, OFF arms fired zero.
* Truncation telemetry: 249 calls, max `prompt_eval_count` 27 516, zero rows at
  `num_ctx − 16`. Confirmed from `ctx_*_fix.jsonl`, not the report.
* Judge-artifact claims spot-checked against raw rows: `acq_q04` (cell 1, ON) is
  0/5 despite containing gold verbatim ("MegaCorp, which accounts for 28%");
  `acq_q12` (cell 2, ON) verdicts contradict their own sampled reasons.
* `acq_q07` runaway answer: 35 290 chars ON vs 1 400 OFF, confirmed from the raw
  arm file.

Verdict accepted as proposed: **4.1 REJECTED as a shipped default** (was HOLD);
flag and code kept. Doc tables updated in `research_roadmap.md` and
`design_rationale.md` §13a in the same commit.

# Phase 4 items 4.1 and 4.2 — end-to-end ANSWER-QUALITY A/Bs

Date: 2026-08-09/10
Status: **both A/Bs run end-to-end and judged. Neither produces a clean win.**
Scope of this wave: `eval/decisions/phase4-answer-quality.md` (this file) only.
No `rag_system/`, no `eval/*.py`, no `Documentation/`, no `main.py` edits. Every
script that produced a number below lives in a scratch directory, not in the
repo; every index was built in that scratch directory and
`eval/.eval_indexes/**` was never read or written.

Proposed calls, stated up front. **The gate decides; this file only supplies
numbers.**

| Item | Proposed call | One-line reason |
|---|---|---|
| 4.1 full-document escalation | **HOLD — do not adopt on this evidence, do not discard the code** | It moved 2 of 7 fired queries from fail to pass on the realistic corpus, but that lift is confounded with prompt truncation (see §6), the one fire under the shipped product default was a *harm* (4/5 → 1/5), and n is 7. |
| 4.2 cross-reference hop | **REJECT at the shipped `retrieval_k`; hold the code** | At `k = 20` it fires 0 times on all 11 `requires_crossref` rows, and where it is forced to fire (`k = 5`) it lands on the wrong document 11 times out of 11 and is a judged wash (5/11 → 5/11) with two individual harms. |

---

## 1. Setup — everything below ran on this machine

**Branch** `rearchitect/evidence-gated-aug-2026`. **Interpreter** `.venv/bin/python`.
**Ollama** `localhost:11434`, generation `qwen3.5:9b`, enrichment/judge
`qwen3.5:4b`. A second agent was hammering the same Ollama for the whole
session, so **no wall-clock number in this file means anything** and none is
used as evidence.

### 1.1 Indexes (throwaway, scratch-only)

Built with `IndexingPipeline` on the shipped `default` profile — contextual
enrichment ON, document overviews ON, late chunking ON, `extract_crossrefs` ON —
with `storage.lancedb_uri` and `overview_path` pointed at the scratch directory
and `chunking.chunk_size = 512` (the value `eval/run_eval.py` uses, so chunk
counts line up with the tracked eval indexes; the profile's own default of 1500
would put most acquisition PDFs in a single chunk and make "escalate to the full
document" a no-op by construction).

| index | files | chunks | crossrefs extracted | resolved |
|---|---|---|---|---|
| `acq` (`eval/corpora/acquisition/*.pdf`) | 10 | 13 | 68 | **34** |
| `acqdocs` (same 10 PDFs + `Documentation/*.md`) | 24 | 373 | 215 | 93 |
| `atlas7` | 1 | 1 | 0 | 0 |
| `hr` | 1 | 2 | 0 | 0 |

`acq` resolving **34** references matches the number the gate recorded after the
`normalize_name` prefix-strip fix (`phase4-crossref-prefilter.md`, "Gate
correction (2026-08-09)"), verified by reading
`metadata → ["metadata"]["crossrefs"]` straight out of the built LanceDB table.
9 of 10 documents are linked; edges are sensible
(`due_diligence_report → regulatory_approval`,
`risk_assessment → closing_checklist`, …) and no chunk self-resolves.

**`atlas7` and `hr` were excluded from both A/Bs.** At `chunk_size = 512` they
are 1 and 2 chunks. "The top-ranked chunk's whole document" *is* the chunk
already in the synthesis context, so the two arms cannot differ for any reason
worth reporting. Stating that as a corpus limitation is more honest than
reporting 48 near-identical runs.

### 1.2 How the agent was driven

In-process `Agent(pipeline_configs=cfg, llm_client, ollama_config)` built from a
config dict this harness controls, so `retrieval.document_escalation.enabled`
and `retrieval.crossref_hop.enabled` flip cleanly per arm. `force_rag=True`
(triage skipped — it is an LLM coin-flip that has nothing to do with either
feature), `session_id=None`, and the agent's semantic cache is cleared before
every query. No HTTP server was started, so none was left running.

Recorded per query: the answer, `source_documents` (with `via_crossref`),
`token_usage`, every `event_callback` payload (`document_escalation`,
`crossref_hop`, `retrieval_retry`, …), and the full pipeline stdout.

### 1.3 Two deliberate deviations from the shipped defaults — both stated, both symmetric across arms

**(a) `think: false` on the generation model.** With the shipped defaults the
9b generation model spends its entire context window on chain-of-thought and
returns an **empty** answer. Verbatim from the first run of this wave
(`acq_q01`, escalation on, no other change):

```
"answer": " [Confidence: 90%] [Warning: Low confidence. Groundedness: False]"
"token_usage": {"by_stage": {"synthesis": {"prompt_tokens": 9351,
                                           "output_tokens": 7033, "calls": 1}}}
```

`9351 + 7033 = 16384` exactly — the whole context window, no `response` text at
all. Every arm in this file therefore monkeypatches `OllamaClient` in the
*scratch runner* to default `enable_thinking=False` (the repo already does this
for `format="json"` calls; nothing in `rag_system/` was edited). This is a
finding in its own right and is logged as backlog in §8.

**(b) query decomposition OFF in the `acqdocs` arms and in every 4.2 arm.**
Kept ON (profile default) for the `acq` escalation A/B in §3. Turned OFF
elsewhere for one reason: with it on, the string that actually reaches the
retriever is the decomposer's rewrite, which is a fresh LLM sample every run, so
the *fire set itself* stops being reproducible (see §3.2 for the case where it
also broke retrieval outright). Both arms of every A/B share the setting.

### 1.4 The judge

`eval/judge.py`'s Phase-0.3 `GroundednessJudge`, prompt `v1`, `qwen3.5:4b` —
**reused unmodified**, no new judge was written. Re-validated this session:

```
  prompt v1 on qwen3.5:4b
  confusion  TP=10  FN=0  TN=10  FP=0  unparseable=0
  TPR 1.0   TNR 1.0   overall agreement 1.0  (gate: >= 0.90)
```

(The JSON that run wrote into `eval/results/` was deleted — this wave owns one
file.)

**Slot assignment, which matters:** `EVIDENCE` = the *system's* answer,
`ANSWER` = the *gold* answer, `QUESTION` = the gold query. So `grounded: true`
means **"the system's answer contains the gold fact"**. The opposite assignment
fails every correct-but-verbose answer, because v1 rejects any claim the
evidence does not state and the system answers run 5–20× longer than the
one-sentence gold answers. Consequences, stated plainly: extra material in the
system answer is tolerated by construction, and an answer that states the gold
fact *and* contradicts it elsewhere would still pass. It is a gold-fact-recall
measure, not a precision measure.

**Judge nondeterminism is much worse in this framing than on the validation
set.** The brief warned about one flip in twenty; on the first two-run pass over
the 24 `acq` rows the judge disagreed with itself on **5/24** and **7/24** rows
(`14/24` vs `9/24`, and `12/24` vs `9/24`). Every judged cell below was
therefore re-run **five times** and is reported as `k/5`, with the raw first two
runs shown per row so the two-run protocol is still visible. A row counts as a
pass at majority (`≥3/5`). One verdict in the whole session came back
unparseable (1 of 550 five-run verdicts, in `acq_esc_off`); it is counted as
not-a-pass. No arm produced a literally empty answer once thinking was off; the
abstention string *"I could not find that information in the provided
documents."* appears 8 times across 110 judged runs and is judged like any other
answer.

---

## 2. Fire-subset discovery for 4.1

Two screens, because they disagree and the disagreement is informative.

**Retrieval-only screen** (`EscalatingRetrievalPipeline.retrieve_candidates`,
no synthesis, raw gold query): reads the same post-retry evidence signal the
escalation planner reads.

| corpus | n gold | fires at the shipped threshold (`retry.min_top_score` = 0.12) | fire rate |
|---|---|---|---|
| `acq` | 24 | 2 (`acq_q04` 0.1072, `acq_q12` 0.1072) | 8.3% |
| `acqdocs` | 48 | 9 (`acq_q04` .113, `acq_q07` .1129, `acq_q09` .0924, `acq_q12` .0566, `docs_d03` .0926, `docs_d05` .1189, `docs_d13` .080, `docs_d15` .0916, `docs_d20` .0955) | 18.8% |

Signal was `dense_contrast` on every single query (reranking is off in the
shipped profile, so the calibrated-reranker branch never runs).

**Agent screen** — the authoritative one, because only a real
`Agent.run()` emits `document_escalation`:

| arm | corpus | decomposition | n | queries that actually escalated |
|---|---|---|---|---|
| `acq_esc_on` | `acq` | ON (profile) | 24 | **1** — `acq_q12` |
| `ad_fire_esc_on` | `acqdocs` | OFF | 9 (the screened set) | **7** — all but `acq_q07` and `docs_d15` |

The `acq` disagreement (screen 2, agent 1) is decomposition: the agent retrieves
with the decomposer's rewrite, not the gold query, and the rewrite scores
differently. On `acqdocs` with decomposition off the screen and the agent still
disagree on 2 of 9, because the evidence-sufficiency retry's reformulation is
itself an LLM sample.

**Fire subset, as executed: 1 query under the full shipped product default
(`acq`), 7 queries on the realistic corpus with decomposition off (`acqdocs`).
Both are below the n≈5 bar the brief set for the product-default case; the
7-query `acqdocs` set is right at it. This is an insufficient-n result and is
called as such in §7.**

---

## 3. 4.1 A/B — arm 1: shipped product default, `acq`, n=24

`document_escalation.enabled` false vs true. Everything else identical and at
the profile default, including query decomposition and verification.

**Whole-set numbers** (context only — 23 of 24 queries are byte-identical work
in both arms, so the whole-set figure is measuring generation nondeterminism,
not escalation):

| arm | escalation events | majority-pass | judge run 1 | judge run 2 | mean pass fraction | mean synthesis `prompt_tokens` |
|---|---|---|---|---|---|---|
| `enabled: false` | 0 | 11/24 | 12/24 | 14/24 | 0.475 | 14563 |
| `enabled: true` | **1** | 10/24 | 12/24 | 10/24 | 0.433 | 14892 |

**The fire subset, n = 1:**

| query | arm | judge (5 runs) | first two runs | synthesis `prompt_tokens` | escalation event |
|---|---|---|---|---|---|
| `acq_q12` | off | **4/5** | `[True, True]` | 19545 | — |
| `acq_q12` | on | **1/5** | `[False, True]` | 20686 | `05_financial_adjustments.pdf`, 2/2 chunks, ~470 approx_tokens, `dense_contrast=0.1073 < 0.12` |

**Harm case, individually (the only fire in this arm):** `acq_q12` — *"How many
people does StartupXYZ employ, and how are they split across functions?"*, gold
*"47 employees: 32 in engineering, 8 in sales, 7 in operations."* The baseline
answered it (4/5). The escalated arm did not (1/5). The escalated document,
`05_financial_adjustments.pdf`, does not contain the headcount — the trigger
escalates *the top-ranked chunk's* document, and on this query the top-ranked
chunk is not in the answer-bearing document. n = 1; this is one sample of a
noisy process, not a demonstration. It is reported because the brief asks for
harm cases individually and because the mechanism it illustrates
(escalate-the-top-chunk's-document ≠ escalate-the-right-document) recurs in
§5's 4.2 numbers.

### 3.2 One non-escalation failure worth recording

`acq_q04` in the escalation-on arm returned *"I could not find an answer in the
documents."* in 10.6s with **no synthesis call at all**. Cause, verbatim from
the pipeline log:

```
Decomposed into 1 sub-queries: ['"What proportion of the target company\'s turnover comes from its single biggest client?"']
Could not search table 'w3b_acq': lance error: Invalid user input: position is not found
  but required for phrase queries, try recreating the index with position, …
Could not search table 'w3b_acq_lc': lance error: Invalid user input: Cannot perform full
  text search unless an INVERTED index has been created on at least one column, …
--- Final Documents for Synthesis ---
No documents to synthesize.
```

The decomposer emitted its single sub-query wrapped in double quotes; LanceDB
read that as a phrase query, the FTS leg raised, and **the whole hybrid
retrieval returned nothing** rather than degrading to the dense leg. Nothing to
do with 4.1 — it happened to land in the escalation arm — but it is a
retrieval-path bug that silently converts a normal query into "no answer", and
it is logged in §8.

---

## 4. 4.1 A/B — arm 2: `acqdocs`, fire subset, n=9 screened / 7 fired

> **VOID — gate note, 2026-08-12.** Every number in this section was measured
> under the §6 truncation bug and should not be cited: the 0/9 baseline below
> reads **7/9 on identical inputs** once prompts fit. The re-run this file's §9
> demanded is `phase4-escalation-rerun.md`; its verdict (reject as default)
> supersedes this section. §3 (`acq`) is largely unaffected — that corpus's
> prompts mostly fit even before the fix.

Decomposition off in both arms (§1.3b) so the retrieval query is the gold query.
`escalation off` vs `escalation on`; nothing else differs.

| query | judge OFF (5 runs) | first two | judge ON (5 runs) | first two | `prompt_tokens` off → on | escalation event (ON arm) |
|---|---|---|---|---|---|---|
| `acq_q04` | 0/5 | `[F, F]` | **2/5** | `[F, T]` | 8303 → 14732 | `02_due_diligence_report.pdf` 1/1 ch, ~540 tok, score 0.0915 |
| `acq_q07` | 0/5 | `[F, F]` | 0/5 | `[F, F]` | 8304 → 8304 | *did not fire* |
| `acq_q09` | 0/5 | `[F, F]` | **4/5** | `[T, T]` | 8311 → 8311 | `05_financial_adjustments.pdf` 2/2 ch, ~470 tok, score 0.0924 |
| `acq_q12` | 0/5 | `[F, F]` | 0/5 | `[F, F]` | 11242 → 13064 | `prompt_inventory.md` 16/16 ch, ~1797 tok, score 0.0566 |
| `docs_d03` | 0/5 | `[F, F]` | **4/5** | `[T, T]` | 8298 → 8298 | `triage_system.md` 10/10 ch, ~1768 tok, score 0.0926 |
| `docs_d05` | 0/5 | `[F, F]` | 0/5 | `[F, F]` | 8298 → 8298 | `indexing_pipeline.md` 47/47 ch, ~5460 tok, score 0.0644 |
| `docs_d13` | 0/5 | `[F, F]` | 0/5 | `[F, F]` | 8298 → 8298 | `api_reference.md` 58/58 ch, ~5436 tok, score 0.080 |
| `docs_d15` | 0/5 | `[F, F]` | **5/5** | `[T, T]` | 8302 → 8302 | *did not fire* |
| `docs_d20` | 0/5 | `[F, F]` | 2/5 | `[F, T]` | 8303 → 8303 | `retrieval_pipeline.md` 38/45 ch, ~6000 tok (**truncated at the budget**), score 0.0629 |

| arm | majority-pass | judge run 1 | judge run 2 | mean pass fraction |
|---|---|---|---|---|
| escalation **off** | **0/9** | 0/9 | 0/9 | 0.000 |
| escalation **on** | **3/9** | 3/9 | 5/9 | 0.378 |

**Restricted to the 7 queries that actually escalated: 0/7 → 2/7 majority-pass
(`acq_q09`, `docs_d03`).**

**Harm cases: none.** No query passed in the baseline and failed in the
escalated arm — but the baseline scored 0/9, so this arm structurally *cannot*
observe harm. That is a property of the subset (weak-evidence queries are the
ones the baseline gets wrong), not evidence that escalation is safe.

**One of the three "help" rows is pure noise.** `docs_d15` went 0/5 → 5/5 and
**escalation did not fire on it** — the two arms ran identical configurations.
That single row is a direct, in-sample measurement of how far generation
nondeterminism alone can move a query: the entire distance from certain-fail to
certain-pass. §5.2 measures the same thing on 11 rows.

The two genuine helps are large and legible. `acq_q09`, baseline
(verbatim, truncated):

> "Based on the provided text fragments, **there is no information** to answer
> your specific question … The documents you supplied describe: A RAG
> (Retrieval-Augmented Generation) system architecture (localGPT) …"

and escalated:

> "**First Disclosed Debt:** $1,500,000 · **Extra Borrowing Found Later:**
> $175,000 (identified as capital lease obligations)"

against gold *"Due diligence disclosed $1.5 million of debt; a further $175,000
of capital lease obligations was identified afterwards."* The baseline did not
give a wrong answer from the acquisition documents — **it never saw them.** Why
that happens is §6, and it is the reason this lift cannot be read as a clean 4.1
win.

---

## 5. 4.2 A/B — the cross-reference hop on the 11 `requires_crossref` rows

Escalation off in every arm. Decomposition off in every arm.

### 5.1 Primary: `acq`, shipped `retrieval_k = 20` — the hop fires zero times

| arm | queries that hopped | chunks added | mean synthesis `prompt_tokens` | majority-pass | judge run 1 | judge run 2 |
|---|---|---|---|---|---|---|
| `crossref_hop off` | 0/11 | 0 | 9369 | 4/11 | 6/11 | 4/11 |
| `crossref_hop on` | **0/11** | 0 | 9369 | 5/11 | 6/11 | 4/11 |

Per-row `prompt_tokens` are **identical to the token** in both arms on all 11
rows, which is the proof that the two arms fed the model the same context.

**The mechanism, and it is not the one the previous wave found.** References now
resolve — 34 of them, 9 of 10 documents linked (§1.1). The blocker is the
`_crossref_hop` "not already represented" guard. Read directly off a
retrieval-only run of all 11 rows:

```
acq_q13  cands=13  hopped=0  top3_refs=21  unrepresented_targets=0
acq_q14  cands=13  hopped=0  top3_refs=23  unrepresented_targets=0
…                                (identical shape on all 11 rows)
acq_q23  cands=13  hopped=0  top3_refs=14  unrepresented_targets=0
fired 0/11
```

The corpus is 13 chunks; at `k = 20` every query retrieves all 13, so all ten
documents are candidates; the top-3 candidates carry 14–23 resolved references
and **every one of them points at a document that is already in the candidate
set**. The guard is behaving exactly as designed. There is nothing to fetch.

The same probe on `acqdocs` at `k = 20` (373 chunks, so the guard can bite):
**2 of 11 rows hop** (`acq_q17` → `10_closing_checklist.pdf`,
`acq_q18` → `01_acquisition_agreement.pdf`), and **0 of those 2 landed in the
row's `expected_sources`.**

### 5.2 That cell is also this file's noise floor

Because §5.1's two arms provably fed the model identical context, every judged
difference between them is generation + judge noise. It is large:

| query | judge OFF | judge ON | note |
|---|---|---|---|
| `acq_q13` | 0/5 | **5/5** | identical prompts, opposite verdicts |
| `acq_q15` | 2/5 | 0/5 | |
| `acq_q22` | 5/5 | 3/5 | |
| `acq_q19` | 0/5 | 2/5 | |
| aggregate | 4/11 majority, mean 0.418 | 5/11 majority, mean 0.509 | |

**A one-row (0.09 mean-pass-fraction) aggregate difference on n = 11 is
indistinguishable from noise, and an individual row can swing the full 0/5 →
5/5.** Every judged delta elsewhere in this file has to clear that bar.

### 5.3 Exploratory arm: `acq` at `retrieval_k = 5`, where the hop can fire

Clearly labelled exploration. `k = 5` is not the shipped value; it is the
smallest `k` at which the "already represented" guard stops suppressing every
target on a 13-chunk corpus. Escalation off, decomposition off, both arms.

| query | judge OFF | first two | judge ON | first two | `prompt_tokens` off → on | hop target (1 hop, `max_hops=1`) | row's `expected_sources` | hop hit expected? |
|---|---|---|---|---|---|---|---|---|
| `acq_q13` | 2/5 | `[F, T]` | **4/5** | `[T, T]` | 3981 → 4624 | `02_due_diligence_report.pdf` (1 ch) | `08_regulatory_approval.pdf` | no |
| `acq_q14` | 2/5 | `[T, F]` | **5/5** | `[T, T]` | 3966 → 4387 | `08_regulatory_approval.pdf` (1 ch) | `04_risk_assessment.pdf` | no |
| `acq_q15` | 4/5 | `[F, T]` | 5/5 | `[T, T]` | 4054 → 4853 | `10_closing_checklist.pdf` (2 ch) | `05_financial_adjustments.pdf` | no |
| `acq_q16` | 5/5 | `[T, T]` | 3/5 | `[T, T]` | 3595 → 4225 | `04_risk_assessment.pdf` (1 ch) | `03_ip_certification.pdf` | no |
| `acq_q17` | 0/5 | `[F, F]` | 0/5 | `[F, F]` | 3506 → 4347 | `01_acquisition_agreement.pdf` (2 ch) | `02_due_diligence_report.pdf` | no |
| `acq_q18` | 2/5 | `[F, F]` | 0/5 | `[F, F]` | 3670 → 4438 | `01_acquisition_agreement.pdf` (2 ch) | `03_ip_certification.pdf` | no |
| `acq_q19` | **4/5** | `[T, T]` | **0/5** | `[F, F]` | 3265 → 4106 | `01_acquisition_agreement.pdf` (2 ch) | `09_customer_consents.pdf` | no |
| `acq_q20` | 3/5 | `[F, T]` | 3/5 | `[T, T]` | 3764 → 4417 | `01_acquisition_agreement.pdf` (2 ch) | `08_regulatory_approval.pdf` | no |
| `acq_q21` | **5/5** | `[T, T]` | **2/5** | `[T, T]` | 3927 → 4936 | `02_due_diligence_report.pdf` (1 ch) | `05_financial_adjustments.pdf` | no |
| `acq_q22` | 0/5 | `[F, F]` | 1/5 | `[F, F]` | 3605 → 4446 | `01_acquisition_agreement.pdf` (2 ch) | `10_closing_checklist.pdf` | no |
| `acq_q23` | 1/5 | `[T, F]` | 0/5 | `[F, F]` | 3987 → 4580 | `01_acquisition_agreement.pdf` (2 ch) | `05_financial_adjustments.pdf`, `08_regulatory_approval.pdf` | no |

| arm | rows that hopped | chunks added | hop hit an `expected_sources` document | majority-pass | judge run 1 | judge run 2 | mean pass fraction | mean `prompt_tokens` |
|---|---|---|---|---|---|---|---|---|
| hop **off** | 0/11 | 0 | — | **5/11** | 5/11 | 6/11 | 0.509 | 3756 |
| hop **on** | **11/11** | 18 | **0/11** | **5/11** | 6/11 | 6/11 | 0.418 | 4487 |

**Harm cases, individually:**

* **`acq_q19`** (4/5 → 0/5) — *"The legal opinion excepts change-of-control
  provisions … For each affected customer, what is the current consent status?"*
  Gold: MegaCorp obtained, DataFlow obtained, CloudTech pending. The answer is in
  `09_customer_consents.pdf`. The hop pulled 2 chunks of
  `01_acquisition_agreement.pdf` instead and displaced the row's own evidence
  from a 5-candidate budget. Largest single regression in this file.
* **`acq_q21`** (5/5 → 2/5) — *"The closing checklist calls for an escrow
  agreement tied to Exhibit C. Over what period is that escrow released?"* Gold:
  the $1,300,000 escrow releases over 18 months, in
  `05_financial_adjustments.pdf`. The hop pulled `02_due_diligence_report.pdf`.
* **`acq_q16`** (5/5 → 3/5) and **`acq_q23`** (1/5 → 0/5) are smaller moves in
  the same direction, inside the §5.2 noise band.

Helps: `acq_q13` (2/5 → 4/5) and `acq_q14` (2/5 → 5/5) — both with a hop that
did **not** hit the expected source, so if the hop helped there it helped by
accident.

**Why the precision is 0/11.** `max_hops = 1`, and the target is the *first*
unrepresented resolved reference found by scanning the top-3 candidates in
order, then that candidate's references in text order. The acquisition corpus is
hub-and-spoke: `01_acquisition_agreement.pdf` and `02_due_diligence_report.pdf`
are referenced by nearly every other document and are named early in nearly
every chunk. So the one hop the budget allows is spent on the hub, essentially
every time, while the answer lives in a spoke. The gate's own prediction — that
`crossref_hit_expected_source` would be the column to watch — is confirmed, and
it reads zero.

---

## 6. The finding that shadows every 4.1 number: the generation context does not fit

`OLLAMA_CONTEXT_LENGTH=16384` on this host, and Ollama divides that across
parallel slots; with the second agent's traffic sharing the server, a standalone
probe measured the effective ceiling at **8194 prompt tokens**:

```
prompt 46072 chars  -> prompt_eval_count 8038
prompt 138072 chars -> prompt_eval_count 8194
prompt 460073 chars -> prompt_eval_count 8194   (front of the prompt is discarded)
```

`RetrievalPipeline._synthesize_final_answer` does no client-side truncation — it
formats the whole context into one prompt and posts it. On the `acqdocs` arms
the pipeline's own log reports the constructed context at **81 837 – 101 469
characters** (≈20 000 – 25 000 tokens) *before* escalation appends anything,
against a slot ceiling of ~8 300. Every one of those synthesis calls was
served a **front-truncated** prompt.

Three consequences, all of which the gate needs:

1. **The `prompt_tokens` column in §4 is a ceiling, not a cost.** On five of the
   seven fired rows the two arms report *identical* `prompt_tokens`
   (8298 / 8298, 8311 / 8311, …) even though the escalated arm appended up to
   6 000 tokens. Escalation's true prompt-cost on this corpus was not measurable
   with `token_usage`; the honest figure is the `approx_tokens` on the
   `document_escalation` event (470 – 6 000 tokens, and `docs_d20` hit the
   budget cap and reported `truncated: true`, 38 of 45 chunks).
2. **The measured 4.1 lift is confounded.** The escalated block is appended to
   the *end* of the facts string, and front-truncation keeps the tail. So on an
   over-long prompt, escalation does not merely "add a document" — it
   *guarantees the escalated document survives truncation while the top-ranked
   retrieved chunks are discarded.* `acq_q09`'s baseline answer ("the documents
   you supplied describe a RAG system architecture") is that failure in the
   open: the acquisition chunks had been truncated away and only
   `Documentation/*.md` text reached the model. The escalated arm answered
   correctly because the answer-bearing document was at the tail. That is a real
   improvement on this deployment, but it is **not** evidence for 4.1's stated
   thesis (a whole document in original order beats similarity-ranked chunks).
   The same lift would presumably come from simply reordering or trimming the
   context.
3. **"Lost in the middle" could not be measured as designed.** The intended harm
   check — the escalated arm getting wrong what the baseline got right — has one
   observation (`acq_q12`, §3) because the `acqdocs` baseline scored 0/9 and had
   nothing to lose.

---

## 7. Caveats, stated plainly

* **n is small everywhere.** 1 fired query under the shipped product default;
  7 fired queries on the realistic corpus; 11 rows for 4.2. One query is 0.14 of
  the 4.1 fire subset and 0.09 of the 4.2 set.
* **The noise floor is ~1 row on n = 11 and up to a full 0/5 → 5/5 on a single
  row** (§5.2, `acq_q13`; §4, `docs_d15`). Nothing in this file with a delta of
  one or two rows is a result.
* **The judge is strict and, in this framing, noisy.** 20/20 on its own
  validation set, but 5–7 self-disagreements per 24 rows on real system answers,
  which is why everything is `k/5`. It also produces defensible-but-harsh
  negatives: `docs_d15`'s baseline said `.txt` files are "read directly into
  fenced markdown without parsing or OCR" against gold "bypass docling and are
  wrapped in a fenced code block" and was rejected 5/5 for not naming docling.
  **Absolute pass rates in this file should not be compared to any other
  document's.** Only the within-table paired direction means anything.
* **The judge measures gold-fact recall, not answer precision** (§1.4).
* **Wall clock is meaningless here** — a second agent shared the GPU throughout,
  and the same query ranged 11s to 272s across runs.
* **Two configuration deviations** (`think:false`; decomposition off in the
  `acqdocs` and 4.2 arms) are symmetric across arms but mean these numbers do
  not describe a byte-for-byte shipped run.
* **`chunk_size = 512`, not the profile's 1500.** Chosen so documents are
  multi-chunk at all; at 1500 the acquisition PDFs are ~1 chunk each and 4.1 is a
  no-op by construction. This makes the corpus *more* favourable to 4.1 than the
  shipped chunking would be.
* **`atlas7` / `hr` were not run** (§1.1) — 1 and 2 chunks.
* **4.3 (overview prefilter) was not touched.** Out of scope for this brief.
* **The `acq` corpus is 13 chunks.** At the shipped `k = 20` no
  candidate-selection change can move anything on it; §5.1 is the clean
  demonstration of that, not a measurement of the hop.

---

## 8. Backlog this wave created

* **The 9b generation model returns an empty answer under the shipped
  defaults.** `stream_completion` does not pass `think`, so the model thinks
  until the context is exhausted and emits no `response`. Reproduced verbatim in
  §1.3a (`9351 + 7033 = 16384`, answer = `""`). The repo already defaults
  thinking off for `format="json"` calls; the synthesis path does not. This is a
  product bug, not an eval artifact, and it is the highest-value item here.
* **The synthesis prompt is not budgeted against the generation model's context
  window.** §6: 20 000 – 25 000 token prompts posted into an 8 200-token slot,
  silently front-truncated, so the highest-ranked retrieved chunks are the first
  thing thrown away. Any context-window guard (`retrieval_k` cap by token budget,
  or a client-side trim that drops from the *bottom* of the ranking) would change
  the answers on this corpus more than either Phase-4 feature does.
* **A quoted sub-query from the decomposer kills hybrid retrieval outright.**
  §3.2: LanceDB reads `"…"` as a phrase query, the FTS leg raises, and
  `retrieve()` returns nothing instead of degrading to the dense leg. One query
  in 24 in this session.
* **`crossref_hop` picks its target by scan order, not by relevance.** §5.3:
  0/11 document-level precision on a hub-and-spoke corpus because the one
  permitted hop is always spent on the hub. If 4.2 is ever revisited, ranking
  candidate targets (by the query's similarity to the target document's
  overview, for instance) is the change that would matter — `max_hops` is not.

---

## 9. Proposed calls, with the one line each rests on

* **4.1 — HOLD.** 0/7 → 2/7 on the fire subset is directionally positive and
  larger than the noise floor, but §6 shows the mechanism producing it is prompt
  truncation rather than document reassembly, and the single fire under the true
  product default was a regression. Fix the context-window bug in §8 first, then
  re-run this exact A/B; if the lift survives a prompt that actually fits, adopt.
  Do not turn the flag on now, and do not delete the code.
* **4.2 — REJECT as a shipped default; keep the flag.** At the shipped
  `retrieval_k = 20` it fires 0/11 on the rows built for it (§5.1) — so turning
  it on buys nothing — and the only configuration where it does fire lands on the
  wrong document 11/11 and is a judged wash with two clear harms (§5.3).
  Index-time extraction (`indexing.extract_crossrefs`, already ON by default) is
  unaffected by this call and should stay on: it is free, it is correct after the
  gate's resolver fix (34/68 resolved on `acq`), and it is what a future,
  relevance-ranked hop would need.

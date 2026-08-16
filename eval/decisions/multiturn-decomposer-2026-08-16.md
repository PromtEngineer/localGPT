# Multi-turn decomposer: answer-visibility fix (arms m0–m1d, L) — 2026-08-16

**Verdict: ADOPTED as a two-variant prompt.** The decomposer now sees the last
assistant answer — but only when conversation history exists. The single-turn
prompt is frozen byte-exact, because arm L measured that even benign-looking
prompt additions cost 2 Sonnet-confirmed rfc rows.

## The gap

`QueryDecomposer` received only the last 5 *user* queries. A follow-up whose
antecedent was introduced by the *assistant* ("who is the largest supplier?" →
answer names Acme → "what is their lead time?") had no antecedent visible.

## Instrument: eval/goldset/multiturn.jsonl

12 hand-authored conversations over the 5 eval corpora (8 "answer-entity"
class isolating this gap; 4 pronoun-to-user-turn controls), every expected
answer verified against chunk text. Runner
(scratchpad `multiturn/run_e2e_multiturn.py`) executes turns sequentially
through `Agent.run` with a real `session_id` — turn 2 sees whatever the system
actually answered — and grades the final turn. Companion harness
`multiturn/decomp_stability.py` dumps temp-0 decompositions of all 120
single-turn gold queries; it is **perfectly deterministic run-to-run**
(120/120 byte-identical on a repeat run), so any dump diff is a real effect
of a prompt change.

## Measurement arc

| arm | change | multi-turn E2E | decomposition-level |
|---|---|---|---|
| m0 | baseline | 12/12 | **wrong underneath**: mt_07 resolved "their" → StartupXYZ (wrong entity); mt_09 turn 3 echoed the two previous queries verbatim. Tiny corpora + pooled synthesis-vs-original-query masked both. |
| m1 | answers interleaved into chat_history | 11/12 | mt_07 fixed (MegaCorp) but the 4b decomposer **anchored on a previous turn and substituted its query** for the current one (mt_09) |
| m1b | queries-only history + separate `last_assistant_answer` field (300-char cap) | 11/12 | mt_07 fixed; mt_09 ellipsis still collapsed to the previous question — pre-existing weakness (m0 decomposed it wrong too, with luckier coverage) |
| m1c | + output rule 1b ("resolved_query must ask the SAME fact; never substitute an earlier question") + one ellipsis example (permit domain, deliberately unlike any gold row) | **12/12** | both critical rows resolve correctly |

## Arm L: the single-turn cost that forced the two-variant design

m1c's prompt additions reworded 25/120 single-turn decompositions (smart-quote
schema fix alone: 5/120). Full 5-bench gate (arm L vs arm K, 120 rows):

- authored (deterministic 4b): 74→75, **zero genuine down-flips** on 96 rows
- rfc: 2 real regressions, Sonnet-panel-confirmed — q18 (3/3 both directions:
  pre-decomposition kept two anchors "Initial-keys salt / QUIC-TLS §5.2" +
  "QUICv2 salt"; m1c collapsed both sub-queries onto QUICv2 and lost §5.2),
  q21 (2/3: dropped "HTTP/3" from the sub-query). q16 passes both arms (3/3).

Iterating the prompt against those rows would be tuning to the bench. Instead:

## Resolution: two variants, selected on `bool(chat_history)`

- `_decompose_single_turn` — the committed pre-change prompt, **byte-exact**
  (including its curly-quoted schema; cosmetic fixes measurably shift temp-0
  decompositions, so the smart-quote fix lives only in the multi-turn variant).
- `_decompose_multi_turn` — the m1c form.

Verification: single-turn dump **byte-identical to the pre baseline on all
120 queries** (the arm-L rfc regression is eliminated by construction);
multi-turn confirmation arm m1d = 12/12, mt_07 → MegaCorp, mt_09 → "When was
the early termination of the waiting period granted for the HSR filing?".

## Rules going forward

- The single-turn prompt is a frozen measured artifact. Any byte change —
  cosmetic included — re-triggers the full 5-bench gate. The stability dump
  (`decomp_stability.py`, byte-compare vs `decomp_dump_pre.jsonl`) is the
  cheap first check.
- The multi-turn prompt is gated on `multiturn.jsonl` (grow this set as
  conversational failure shapes appear) + the single-turn byte-identity check.
- Artifacts: `mt_answers_{m0,m1,m1b,m1c,m1d}.jsonl`,
  `decomp_dump_{pre,postq,postq2,postm1,postm1c,postm1d}.jsonl`,
  `rfc_e2e_answers_l.jsonl`, `authored_e2e_answers_decomp.jsonl`,
  `judged4bdet_*.jsonl`, `votes_armL_{1,2,3}.jsonl` in the session scratchpad.

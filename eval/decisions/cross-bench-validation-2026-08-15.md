# Cross-bench validation: are the RFC-tuned changes general? (2026-08-15)

**Question (user-posed):** the week's five adopted changes (strict prompt,
dedupe + 12k context budget, Qwen3-Reranker-4B threshold selection, pooled
decomposition + temp-0 decompose, source-document labels) were all tuned on
the unseen-RFC bench. Do they generalize, or did we overfit to that test?

**Method:** all 96 authored gold rows (acquisition / atlas7 / hr / docs,
24 each — none touched during tuning) answered by both configurations
against IDENTICAL freshly-built product indexes (ingestion code unchanged
all week): HEAD = a3f999a, baseline = 80d5215 (post-chunker-fix,
pre-tuning) in a git worktree. qwen3.5:4b bulk judge over all 192 answers;
blind 3-voter Sonnet panels on the two direction-deciding cells (hr, docs)
for both arms. Raw rows: scratchpad authored_bench/results/.

## Results

| corpus | baseline | HEAD | judge | verdict |
|---|---|---|---|---|
| acq | 16/24 | 18/24 | 4b | +2, within 4b noise — flat-to-positive |
| atlas7 | 19/24 | 19/24 | 4b | flat |
| hr | **24/24** | **21/24** | Sonnet panel (0 splits both) | **−3, real regression** |
| docs | 15/24 | 17/24 | Sonnet panel (0 splits both) | +2 (4 gains / 2 losses) |
| rfc (from arm history) | 5/24 | 21/24 | Sonnet panels | +16 |
| mechanical exact-substring | 24/96 | 61/96 | — | +39/−2 (verbatim-copy style) |
| wall time (96 rows) | 3,928s | 2,847s | — | −27% |

## The hr regression, diagnosed

All three lost rows (hr_h05/h08/h13) are the same shape: the HEAD answer is
CORRECT for the literal question but omits an adjacent clause the gold
includes (the 1.5x multiplier answer omits the substitute day; the
identifier/revision answer omits the effective date). Probed in-process:
the omitted text WAS in the synthesis context (the tiny hr corpus retrieves
2 chunks; both contain it; both survived selection) — so the reranker/
budget/selection stack is innocent. The strict arm-C prompt answers
narrowly; the old loose prompt padded answers with surrounding detail and
incidentally covered gold's bonus clauses. A candidate fix (a completeness
clause in the synthesis prompt: include directly-attached conditions/
riders/dates) is NOT adopted here — any prompt change must re-validate on
all five benches per this document's own lesson.

Incidental find: late-chunk tables of tiny corpora have no FTS INVERTED
index — the hybrid FTS leg fails and degrades to dense-only (the 513344f
fallback masks it). Logged as a fix-it task.

## Verdict

**Not overfit.** The +16 RFC gain came with flat-to-positive movement on
three of four authored benches and a −27% wall-time reduction; the single
regression (hr −3) is localized, mechanism-understood, and answer-STYLE
related (narrow literal answering), not a retrieval or grounding failure.
Changes stay adopted; the completeness-prompt experiment is queued as
future work with full 5-bench validation required.

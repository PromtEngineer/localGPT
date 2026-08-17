"""Score a judge against eval/judge_hard_cases.jsonl.

These are the 18 real system answers from the 2026-08-12 escalation re-run
whose ground truth ("does the answer contain the gold fact?") was hand-
adjudicated (eval/decisions/phase4-escalation-rerun.md §6). The qwen3.5:4b
judge's 5-vote majority is wrong on 5 of the 18 — all five being answers that
contain the gold fact verbatim but were voted down. Any candidate judge must
beat 13/18 here to be worth switching to.

Slot assignment matches the A/B harness: EVIDENCE = the system's answer,
ANSWER = the gold answer, so grounded=true means "the system's answer contains
the gold fact".

Voting honesty: ``--votes`` must be a positive odd number (an even count can
tie, and a silent tie-break is not a verdict). A judge call whose verdict fails
to parse (``grounded`` is None) is recorded as an ERROR, never coerced to a
vote — ``bool(None)`` would bank free "correct" votes on every label-False
row. A row whose valid votes tie (possible once errors shrink the odd total),
or that has no valid votes at all, is an error row: reported as such and
counted neither correct nor incorrect — the same way ``judge.py --validate``
keeps unparseable verdicts out of its confusion matrix.

Usage::

    .venv/bin/python eval/validate_judge_hard.py                       # local default
    JUDGE_MODEL=claude-sonnet-5 .venv/bin/python eval/validate_judge_hard.py
    .venv/bin/python eval/validate_judge_hard.py --model claude-sonnet-5 --votes 3
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(EVAL_DIR, "..")))

from judge import DEFAULT_MODEL, DEFAULT_VERSION, GroundednessJudge  # noqa: E402

CASES_PATH = os.path.join(EVAL_DIR, "judge_hard_cases.jsonl")
RESULTS_DIR = os.path.join(EVAL_DIR, "results")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt-version", default=DEFAULT_VERSION)
    parser.add_argument("--votes", type=int, default=1,
                        help="votes per row; majority decides (must be odd)")
    args = parser.parse_args()
    if args.votes < 1 or args.votes % 2 == 0:
        parser.error("--votes must be a positive odd number — an even count can "
                     "tie, and a tie is not a verdict")

    with open(CASES_PATH, "r", encoding="utf-8") as fh:
        cases = [json.loads(line) for line in fh if line.strip()]

    judge = GroundednessJudge(model=args.model, version=args.prompt_version)
    correct = 0
    error_rows = 0
    qwen_correct = 0
    rows = []
    for case in cases:
        votes = []
        errors = 0
        reasons = []
        for _ in range(args.votes):
            verdict = judge.judge(case["question"], case["gold_answer"], case["answer"])
            # A parse/shape failure is an error, never a vote: coercing
            # grounded=None to False would bank free "correct" votes on every
            # label-False row.
            if verdict.get("error") or verdict["grounded"] is None:
                errors += 1
            else:
                votes.append(verdict["grounded"])
            reasons.append(verdict.get("reason", ""))
        # Majority over the valid votes only. A tie among them (possible once
        # errors shrink the odd total) is undecided — reported, not forced.
        predicted = None if not votes or 2 * sum(votes) == len(votes) \
            else sum(votes) > len(votes) / 2
        label = case["label_grounded"]
        ok = None if predicted is None else predicted == label
        correct += (ok is True)
        error_rows += (predicted is None)
        qwen_ok = (case["qwen4b_votes"] >= 3) == label
        qwen_correct += qwen_ok
        rows.append({**{k: case[k] for k in ("id", "label_grounded", "qwen4b_votes")},
                     "votes": votes, "error_votes": errors, "predicted": predicted,
                     "correct": ok, "reasons": reasons})
        flag = " " if ok else ("!" if ok is None else "<")
        print(f"  {flag} {case['id']:<22} label={'contains-fact' if label else 'missing-fact'}"
              f"  pred={str(predicted):<5} votes={sum(votes)}/{len(votes)}"
              f"{f' +{errors} error(s)' if errors else ''}"
              f"  (4b was {case['qwen4b_votes']}/5{'' if qwen_ok else ' WRONG'})")

    n = len(cases)
    print(f"\n  {args.model} (k={args.votes}): {correct}/{n} correct"
          + (f"  (+{error_rows} error row(s), counted neither correct nor incorrect)"
             if error_rows else ""))
    print(f"  qwen3.5:4b baseline (k=5):   {qwen_correct}/{n} correct")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_model = args.model.replace(":", "_").replace("/", "_")
    out_path = os.path.join(RESULTS_DIR, f"judge_hard_{safe_model}_{stamp}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"model": args.model, "votes": args.votes, "correct": correct,
                   "error_rows": error_rows, "n": n, "rows": rows}, fh, indent=2)
    print(f"  written {out_path}")
    return 0 if correct > qwen_correct else 1


if __name__ == "__main__":
    sys.exit(main())

"""Measure judge determinism: re-grade stored answers N times and report verdict flips.

Isolates grader noise from agent noise — the stored answers are fixed, so any disagreement
between passes is the judge alone.
"""

import glob
import json
import sys
from collections import defaultdict

from marag.config import load_config
from marag.eval.answer_eval import JUDGE_SYSTEM
from marag.eval.retrieval_eval import load_benchmark
from marag.llm import LLM

PASSES = int(sys.argv[1]) if len(sys.argv) > 1 else 3
DATASETS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["financial_docs"]

cfg = load_config()
judge = LLM("utility", cfg)

for ds in DATASETS:
    f = [x for x in sorted(glob.glob(f"runs/answers_{ds}_agentic_*.json")) if "rejudged" not in x][-1]
    run = json.loads(open(f).read())
    bench = {q["id"]: q for q in load_benchmark(ds, cfg)["questions"]}
    verdicts: dict[str, list[bool]] = defaultdict(list)

    for p in range(PASSES):
        for row in run["results"]:
            q = bench[row["id"]]
            try:
                v = judge.json(
                    [
                        {"role": "system", "content": JUDGE_SYSTEM},
                        {
                            "role": "user",
                            "content": f"QUESTION: {q['question']}\n\nGOLD ANSWER: {q['answer']}\n\n"
                            f"MODEL ANSWER: {row['answer'][:3000]}",
                        },
                    ],
                    max_tokens=600,
                    temperature=0.0,
                    reasoning="none",
                )
                verdicts[row["id"]].append(bool(v.get("correct")))
            except Exception:
                verdicts[row["id"]].append(False)

    flips = {k: v for k, v in verdicts.items() if len(set(v)) > 1}
    scores = [sum(verdicts[k][p] for k in verdicts) for p in range(PASSES)]
    print(f"\n== {ds} · {PASSES} judge passes over identical answers ({f.split('/')[-1]})")
    print(f"   per-pass score: {scores}  (out of {len(verdicts)})")
    print(f"   unstable verdicts: {len(flips)}/{len(verdicts)} {list(flips.items()) if flips else ''}")

"""Closed-book contamination control: answer each benchmark question from the model's
parametric memory ALONE — no retrieval, no tools, no corpus. The gap between this and the
open-book agentic score is the honest measure of how much the corpora (famous public 10-Ks,
arXiv papers, CDC reports — all inside the models' training window) are already memorized.

A high closed-book score does NOT invalidate the system; it means those specific questions
can't distinguish retrieval quality from recall, and the open-book delta is what retrieval buys.

Usage: MARAG_CONFIG=configs/default.yaml uv run python scripts/closed_book.py financial_docs
"""

from __future__ import annotations

import json
import sys
import time

from marag.config import load_config
from marag.eval.answer_eval import JUDGE_SYSTEM, run_provenance
from marag.eval.retrieval_eval import load_benchmark
from marag.llm import LLM

CLOSED_BOOK_SYSTEM = """You are answering a question from your own knowledge. You have NO
access to any documents. Answer as precisely as you can from what you already know; if you
do not know a specific figure, say so plainly rather than guessing. Answer first, briefly."""


def main(dataset: str) -> None:
    cfg = load_config()
    bench = load_benchmark(dataset, cfg)
    gen = LLM("orchestrator", cfg)
    judge = LLM("judge" if cfg.models.judge else "utility", cfg)

    rows, n_correct = [], 0
    for q in bench["questions"]:
        t0 = time.time()
        answer = gen.text(
            [
                {"role": "system", "content": CLOSED_BOOK_SYSTEM},
                {"role": "user", "content": q["question"]},
            ],
            max_tokens=4096,
            temperature=0.0,
        )
        verdict = judge.json(
            [
                {"role": "system", "content": JUDGE_SYSTEM},
                {
                    "role": "user",
                    "content": f"QUESTION: {q['question']}\n\nGOLD ANSWER: {q['answer']}\n\n"
                    f"MODEL ANSWER: {answer[:3000]}",
                },
            ],
            max_tokens=600,
            temperature=0.0,
            reasoning="none",
        )
        correct = bool(verdict.get("correct"))
        n_correct += correct
        rows.append({
            "id": q["id"], "correct": correct,
            "reason": str(verdict.get("reason", ""))[:200],
            "answer": answer[:2000], "gold": q["answer"], "seconds": round(time.time() - t0, 1),
        })
        print(f"{'✓' if correct else '✗'} {q['id']} ({rows[-1]['seconds']}s) {rows[-1]['reason'][:90]}")

    report = {
        "dataset": dataset, "mode": "closed_book",
        "provenance": run_provenance(cfg, judge.model),
        "n": len(rows), "accuracy": round(n_correct / max(len(rows), 1), 3),
        "results": rows,
    }
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = cfg.path("runs") / f"answers_{dataset}_closedbook_{ts}.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"\n{dataset} closed-book: {report['accuracy']:.1%} ({n_correct}/{len(rows)}) → {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "financial_docs")

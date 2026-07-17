from __future__ import annotations

import json
import re
import time

from rich.console import Console

from ..agents.search_agent import answer_agentic
from ..agents.single_shot import answer_single_shot
from ..config import Config
from ..llm import LLM
from ..retrieve.hybrid import Retriever
from .retrieval_eval import load_benchmark

console = Console()

JUDGE_SYSTEM = """You grade RAG answers against a gold answer. Respond with ONLY JSON:
{"correct": true/false, "reason": "<one line>"}
Grade "correct": true only if the model answer contains the gold answer's key facts:
- numbers must match the gold values (allow rounding in the last digit and unit reformatting),
- comparisons/rankings must agree in direction,
- for lists, all gold items must be present.
An answer that says the information is missing is incorrect. Extra correct detail is fine."""


_CITE_RE = re.compile(r"\[(\w+)\s+p(\d+)\]")


def citation_grounding(answer: str, result: dict) -> float | None:
    """Fraction of [doc pN] citations that point at evidence the system actually touched (±1 page)."""
    cites = [(d, int(p)) for d, p in _CITE_RE.findall(answer)]
    if not cites:
        return None
    if result.get("mode") == "agentic":
        seen = {(d, p) for d, p in result.get("evidence_pages", [])}
    else:
        seen = {(c["doc_id"], c["page"]) for c in result.get("contexts", [])}
    ok = sum(1 for d, p in cites if any((d, p + off) in seen for off in (-1, 0, 1)))
    return round(ok / len(cites), 3)


def eval_answers(
    dataset: str, cfg: Config, mode: str = "single_shot", limit: int | None = None
) -> dict:
    bench = load_benchmark(dataset, cfg)
    questions = bench["questions"][:limit] if limit else bench["questions"]
    retriever = Retriever(cfg)
    judge = LLM("utility", cfg)
    gen = answer_single_shot if mode == "single_shot" else answer_agentic

    rows: list[dict] = []
    n_correct = 0
    for q in questions:
        t0 = time.time()
        try:
            if mode == "single_shot":
                result = gen(q["question"], dataset, cfg, retriever)
            else:
                result = gen(q["question"], dataset, cfg, retriever)
        except Exception as e:
            rows.append({"id": q["id"], "error": str(e), "correct": False})
            console.print(f"[red]✗ {q['id']} generation failed: {e}[/]")
            continue
        elapsed = round(time.time() - t0, 1)
        try:
            verdict = judge.json(
                [
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {
                        "role": "user",
                        "content": f"QUESTION: {q['question']}\n\nGOLD ANSWER: {q['answer']}\n\n"
                        f"MODEL ANSWER: {result['answer'][:3000]}",
                    },
                ],
                max_tokens=600,
                temperature=0.0,  # a stochastic judge silently adds ±1-2 questions of noise
                reasoning="none",  # thinking judges starve their own JSON output; grading is simple
            )
            correct = bool(verdict.get("correct"))
            reason = str(verdict.get("reason", ""))[:200]
        except Exception as e:
            correct, reason = False, f"judge failed: {e}"
        n_correct += correct
        rows.append(
            {
                "id": q["id"],
                "hop_type": q.get("hop_type"),
                "correct": correct,
                "reason": reason,
                "answer": result["answer"][:2000],
                "gold": q["answer"],
                "tool_calls": result.get("tool_calls", 0),
                "citation_grounding": citation_grounding(result["answer"], result),
                "seconds": elapsed,
                # keep the tool sequence: failures are usually a wrong tool path, not a wrong model
                "tools": [
                    {"tool": t["tool"], **{k: v for k, v in t.get("args", {}).items() if k != "question"}}
                    for t in result.get("transcript", [])
                ],
            }
        )
        mark = "[green]✓[/]" if correct else "[red]✗[/]"
        console.print(f"{mark} {q['id']} ({elapsed}s, {result.get('tool_calls',0)} calls) {reason}")

    grounded = [r["citation_grounding"] for r in rows if r.get("citation_grounding") is not None]
    report = {
        "dataset": dataset,
        "mode": mode,
        "n": len(rows),
        "accuracy": round(n_correct / max(len(rows), 1), 3),
        "avg_tool_calls": round(sum(r.get("tool_calls", 0) for r in rows) / max(len(rows), 1), 1),
        "avg_citation_grounding": round(sum(grounded) / len(grounded), 3) if grounded else None,
        "results": rows,
    }
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = cfg.path("runs") / f"answers_{dataset}_{mode}_{ts}.json"
    out.write_text(json.dumps(report, indent=1))
    console.print(
        f"[bold]{dataset} / {mode}: accuracy {report['accuracy']:.1%} "
        f"({n_correct}/{len(rows)}), avg {report['avg_tool_calls']} tool calls[/] → {out}"
    )
    return report

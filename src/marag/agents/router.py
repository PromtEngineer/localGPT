from __future__ import annotations

from ..config import Config
from ..llm import LLM

SYSTEM = """Classify how a document-QA query should be handled. Respond with ONLY JSON:
{"route": "single_shot" | "iterative"}

single_shot — one retrieval pass suffices: single-fact lookup, one definition, one number from one document.
iterative — needs multiple evidence pieces or tools: comparisons across documents, multi-step
reasoning, arithmetic over tables or spreadsheets, aggregations ("all X that Y"), questions
naming 2+ entities to reconcile, and ANY whole-document operation (summarize/overview of a
document or file)."""


def route(question: str, cfg: Config) -> str:
    try:
        r = LLM("utility", cfg).json(
            [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": question},
            ],
            max_tokens=400,
            reasoning="none",  # trivial classification; skip thinking
        )
        v = str(r.get("route", "")).strip()
        return v if v in ("single_shot", "iterative") else "iterative"
    except Exception:
        return "iterative"  # fail toward the more capable path


def answer_auto(question: str, dataset: str, cfg: Config, retriever) -> dict:
    from .search_agent import answer_agentic
    from .single_shot import answer_single_shot

    r = route(question, cfg)
    if r == "single_shot":
        result = answer_single_shot(question, dataset, cfg, retriever)
    else:
        result = answer_agentic(question, dataset, cfg, retriever)
    result["route"] = r
    return result

"""Agentic plan-and-execute helpers for the RAG agent.

This is the opt-in evolution of static query decomposition (NVIDIA RAG
Blueprint's agentic path). It adds two things the plain decompose→retrieve→
compose path lacks:

1. A complexity gate, so simple questions skip the (latency-heavy) planning
   and run as a single retrieval.
2. Evidence-driven retry: a planned sub-task whose retrieval came back empty
   (or whose answer was "not found") is reformulated once and retried before
   synthesis, instead of silently contributing nothing.

The functions here own only the LLM-driven *decisions* (assess, reformulate)
and a pure evidence heuristic. Orchestration (parallel retrieval, synthesis,
verification, caching) stays in the agent so this module has no dependency on
the pipeline and is unit-testable with a stub LLM client.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Sentinel the synthesizer returns when context doesn't answer the question.
_NOT_FOUND_MARKERS = (
    "could not find",
    "not find that information",
    "does not contain",
    "no relevant information",
    "could not find relevant information",
)

# Cheap pre-filter: very short questions with no conjunction/comparison
# markers are treated as simple without spending an LLM call.
_COMPLEX_MARKERS = re.compile(
    r"\b(and|or|versus|vs\.?|compare|comparison|difference|differ|both|"
    r"as well as|each|respectively|trade-?off|pros and cons)\b",
    re.IGNORECASE,
)

COMPLEXITY_PROMPT = """Decide whether answering this question well requires gathering several distinct pieces of information (multi-part, comparison, or multi-hop), or whether it is a single factual lookup.

QUESTION: {query}

Reply with JSON only: {{"complex": true}} or {{"complex": false}}.
"""

REFORMULATE_PROMPT = """A document search for this sub-question returned no useful results.

SUB-QUESTION: {task}

Rewrite it as a single, broader search query likely to retrieve relevant passages — drop rare phrasings, keep the key entities and topic. Reply with JSON only: {{"query": "..."}}.
"""


def _parse_json_field(raw: str, key: str, default: Any) -> Any:
    try:
        return json.loads(raw).get(key, default)
    except (json.JSONDecodeError, AttributeError, TypeError):
        return default


def assess_complexity(llm_client, model: str, query: str, *, timeout: int = 30) -> bool:
    """True if the question warrants multi-task planning.

    A short query with no conjunction/comparison markers short-circuits to
    False (no LLM call). Otherwise the LLM decides; on any failure we default
    to False so the cheap single-retrieval path is used.
    """
    words = query.split()
    if len(words) <= 6 and not _COMPLEX_MARKERS.search(query):
        return False
    try:
        resp = llm_client.generate_completion(
            model, COMPLEXITY_PROMPT.format(query=query),
            format="json", enable_thinking=False, timeout=timeout,
        )
        return bool(_parse_json_field(resp.get("response", ""), "complex", False))
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("complexity_assessment_failed error=%s", e)
        return False


def is_evidence_thin(sub_result: Optional[Dict[str, Any]]) -> bool:
    """True when a sub-task gathered no usable evidence.

    Thin = the retrieval returned no source documents, or the synthesized
    sub-answer is a 'not found' response. These are the tasks worth one retry.
    """
    if not sub_result:
        return True
    if not sub_result.get("source_documents"):
        return True
    answer = (sub_result.get("answer") or "").lower()
    return any(marker in answer for marker in _NOT_FOUND_MARKERS)


def reformulate_task(llm_client, model: str, task: str, *, timeout: int = 30) -> Optional[str]:
    """Broaden a thin sub-task into a new search query, or None if unchanged."""
    try:
        resp = llm_client.generate_completion(
            model, REFORMULATE_PROMPT.format(task=task),
            format="json", enable_thinking=False, timeout=timeout,
        )
        new_q = _parse_json_field(resp.get("response", ""), "query", "")
        new_q = (new_q or "").strip()
        if new_q and new_q.lower() != task.strip().lower():
            return new_q
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("task_reformulation_failed error=%s", e)
    return None

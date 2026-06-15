"""Local long-form report generation.

An opt-in (`report: true`) workflow that turns a single query into a structured,
multi-section Markdown report grounded ENTIRELY in the local indexes — no web,
no external calls, so it carries no data-egress policy dependency. The shape
mirrors the NVIDIA report-generator pattern (plan -> retrieve -> draft ->
compile) but every "retrieve" is LocalGPT's own multi-collection pipeline.

Flow:
    plan_sections()            one LLM call -> section titles (budget-clamped)
    -> per section: pipeline.run() local retrieval + grounded synthesis
    -> remap_citations()       per-section [n] -> global [N] (drops invalid)
    -> compile_report()        Markdown with one merged References list

Budgeting reuses the reflection philosophy: section count is clamped so the
report can't fan out unboundedly.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

EventCallback = Optional[Callable[[str, Any], None]]

_MAX_REPORT_SECTIONS = 8
_DEFAULT_SECTIONS = 4

_PLAN_PROMPT = """You are outlining a structured report that answers the user's request using a document collection.

USER REQUEST: {query}

Propose between 3 and {max_sections} section titles that together answer the request comprehensively, ordered logically (overview first, specifics after). Each title is a short noun phrase — no numbering, no "Introduction"/"Conclusion" filler unless genuinely needed.

Reply with JSON only: {{"sections": ["title one", "title two", ...]}}.
"""


def parse_sections(
    raw: str, max_sections: int = _DEFAULT_SECTIONS, fallback: str = "Overview"
) -> List[str]:
    """Parse the planner's JSON into a clean, de-duplicated, budget-clamped list.

    Always returns at least one section so the report can proceed even if the
    model returns garbage."""
    titles: List[str] = []
    try:
        data = json.loads(raw)
        for item in data.get("sections", []):
            title = str(item).strip()
            if title and title.lower() not in {t.lower() for t in titles}:
                titles.append(title)
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass
    if not titles:
        return [fallback]
    return titles[: max(1, min(max_sections, _MAX_REPORT_SECTIONS))]


def plan_sections(
    llm_client: Any, model: Any, query: str, max_sections: int
) -> List[str]:
    """Ask the model for an outline; fall back to a single overview on failure."""
    prompt = _PLAN_PROMPT.format(query=query, max_sections=max_sections)
    try:
        resp = llm_client.generate_completion(
            model, prompt, format="json", enable_thinking=False, timeout=120
        )
        raw = resp.get("response", "") if isinstance(resp, dict) else ""
    except Exception:
        raw = ""
    return parse_sections(raw, max_sections)


def _source_key(doc: Dict[str, Any]) -> Any:
    """Identity for cross-section dedup: a chunk is the same only within the
    same collection (chunk_id is unique per table), matching the retrieval
    pipeline's own dedup identity."""
    cid = doc.get("chunk_id")
    if cid is not None:
        return (doc.get("_source_table"), cid)
    return ("_doc", doc.get("document_id"), (doc.get("text") or "")[:64])


def remap_citations(
    text: str,
    section_sources: List[Dict[str, Any]],
    global_sources: List[Dict[str, Any]],
    global_keys: Dict[Any, int],
) -> Tuple[str, int]:
    """Rewrite a section's local ``[n]`` markers to global ``[N]`` indices,
    appending newly-seen sources to ``global_sources`` (1-based numbering).

    Citations pointing outside the section's source list are invalid (model
    hallucinated an index) and are dropped. Single-pass so remapped numbers
    can't cascade into one another. Returns (text, dropped_count)."""
    dropped = 0

    def repl(match: "re.Match[str]") -> str:
        nonlocal dropped
        n = int(match.group(1))
        if n < 1 or n > len(section_sources):
            dropped += 1
            return ""
        doc = section_sources[n - 1]
        key = _source_key(doc)
        if key not in global_keys:
            global_sources.append(doc)
            global_keys[key] = len(global_sources)  # 1-based
        return f"[{global_keys[key]}]"

    return re.sub(r"\[(\d+)\]", repl, text), dropped


def _reference_line(i: int, doc: Dict[str, Any]) -> str:
    """One References entry, labelled by origin so web-sourced facts are
    visibly distinct from local-index ones."""
    if doc.get("_origin") == "web":
        url = doc.get("url") or ""
        name = doc.get("document_id") or url or "web source"
        return f"{i}. [Web] {name}" + (f" — {url}" if url else "")
    name = doc.get("document_id") or doc.get("_source_table") or "source"
    return f"{i}. [Local] {name}"


def compile_report(
    query: str,
    sections: List[Tuple[str, str]],
    global_sources: List[Dict[str, Any]],
) -> str:
    """Assemble the final Markdown: a title, each section, then one merged
    References list keyed to the global citation numbers."""
    lines: List[str] = [f"# {query.strip()}", ""]
    for title, body in sections:
        lines.append(f"## {title}")
        lines.append("")
        lines.append(
            body.strip()
            or "_No supporting information found in the indexed documents._"
        )
        lines.append("")
    if global_sources:
        lines.append("## References")
        lines.append("")
        for i, doc in enumerate(global_sources, 1):
            lines.append(_reference_line(i, doc))
    return "\n".join(lines).rstrip() + "\n"


def _format_facts(docs: List[Dict[str, Any]]) -> str:
    """Number snippets as the synthesizer expects ([Source N: label]), tagging
    web origins so the model can attribute them correctly."""

    def label(doc: Dict[str, Any]) -> str:
        name = doc.get("document_id") or doc.get("url") or "source"
        return f"{name} (web)" if doc.get("_origin") == "web" else str(name)

    return "\n\n".join(
        f"[Source {i}: {label(d)}]\n{d.get('text', '')}"
        for i, d in enumerate(docs, start=1)
    )


def _web_docs(raw_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Adapt provider web results into the source-document shape, tagged web."""
    docs: List[Dict[str, Any]] = []
    for r in raw_results:
        docs.append(
            {
                "document_id": r.get("title") or r.get("url") or "web source",
                "url": r.get("url", ""),
                "text": r.get("text", ""),
                "chunk_id": None,
                "_origin": "web",
            }
        )
    return docs


def _augment_section_with_web(
    pipeline: Any,
    generation_model: Any,
    section_query: str,
    local_result: Dict[str, Any],
    *,
    web_max_results: int,
    web_policy: Optional[Dict[str, Any]],
    web_audit: Optional[Callable[[Dict[str, Any]], None]],
    skill_instruction: Optional[str],
    emit: Callable[[str, Any], None],
) -> Tuple[str, List[Dict[str, Any]], int]:
    """Run a policy-gated web search and, if it returns anything, re-synthesize
    the section over combined local + web evidence (one citation space).

    Returns (body, section_sources, web_count). Falls back to the local answer
    when web search is unconfigured, blocked, or empty — so enabling web never
    degrades the local report.
    """
    from rag_system.agent import web_search

    local_docs = list(local_result.get("source_documents") or [])
    for doc in local_docs:
        doc.setdefault("_origin", "local")

    if not web_search.is_configured():
        return str(local_result.get("answer") or ""), local_docs, 0

    emit("web_search_started", {})
    raw = web_search.policy_gated_search(
        section_query, policy=web_policy, audit=web_audit, max_results=web_max_results
    )
    web_docs = _web_docs(raw)
    emit("web_search_done", {"count": len(web_docs)})
    if not web_docs:
        return str(local_result.get("answer") or ""), local_docs, 0

    combined = local_docs + web_docs
    facts = _format_facts(combined)
    body = pipeline._synthesize_final_answer(
        section_query,
        facts,
        generation_model=generation_model,
        skill_instruction=skill_instruction,
    )
    return body, combined, len(web_docs)


def generate_report(
    pipeline: Any,
    generation_model: Any,
    query: str,
    *,
    run_kwargs: Dict[str, Any],
    event_callback: EventCallback = None,
    max_sections: int = _DEFAULT_SECTIONS,
    web_enabled: bool = False,
    web_max_results: int = 3,
    web_policy: Optional[Dict[str, Any]] = None,
    web_audit: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Plan sections, draft each from local retrieval, and compile a grounded
    Markdown report. Returns the standard {answer, source_documents} shape plus
    a `report` summary, so the existing chat UI renders it unchanged.

    With `web_enabled`, each section is additionally augmented by a policy-gated
    web search (off by default; no-op unless a provider is configured). Web and
    local citations share one numbering and are labelled [Web]/[Local].
    """

    def emit(event_type: str, payload: Any) -> None:
        if event_callback is not None:
            event_callback(event_type, payload)

    titles = plan_sections(
        pipeline.ollama_client, generation_model, query, max_sections
    )
    emit("report_started", {"count": len(titles), "web": bool(web_enabled)})
    skill_instruction = (run_kwargs.get("overrides") or {}).get("skill_instruction")

    global_sources: List[Dict[str, Any]] = []
    global_keys: Dict[Any, int] = {}
    compiled: List[Tuple[str, str]] = []
    dropped_total = 0
    web_total = 0

    for idx, title in enumerate(titles):
        emit("report_section", {"index": idx, "title": title})
        section_query = f"{query}\n\nFocus specifically on: {title}"
        try:
            result = pipeline.run(
                section_query, event_callback=event_callback, **run_kwargs
            )
        except Exception:
            result = {"answer": "", "source_documents": []}

        if web_enabled:
            body, section_sources, web_count = _augment_section_with_web(
                pipeline,
                generation_model,
                section_query,
                result,
                web_max_results=web_max_results,
                web_policy=web_policy,
                web_audit=web_audit,
                skill_instruction=skill_instruction,
                emit=emit,
            )
            web_total += web_count
        else:
            body = str(result.get("answer") or "")
            section_sources = list(result.get("source_documents") or [])
            for doc in section_sources:
                doc.setdefault("_origin", "local")

        remapped, dropped = remap_citations(
            body, section_sources, global_sources, global_keys
        )
        dropped_total += dropped
        compiled.append((title, remapped))

    answer = compile_report(query, compiled, global_sources)
    emit("report_done", {"count": len(titles)})
    return {
        "answer": answer,
        "source_documents": global_sources,
        "report": {
            "sections": titles,
            "section_count": len(titles),
            "dropped_citations": dropped_total,
            "web_sources": web_total,
        },
    }

"""Optional web search for the web-augmented report mode.

LocalGPT is local-first: web search is **off by default** and only ever runs
when (a) the request explicitly enables it AND (b) a provider key is configured.
Crucially, every outbound query is first screened by the data-egress policy
(`rag_system.utils.data_policy`) — the same layer that governs cloud enrichment —
so secrets/PII never leave the machine in a search query.

Provider-neutral; Tavily is the reference. No key bundled; resolved from env.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional

import requests

from rag_system.utils.data_policy import BLOCK, REDACT, evaluate


def provider() -> str:
    return (os.getenv("LOCALGPT_WEB_SEARCH_PROVIDER") or "tavily").lower()


def _api_key() -> str:
    if provider() == "tavily":
        return os.getenv("TAVILY_API_KEY", "")
    return os.getenv("WEB_SEARCH_API_KEY", "")


def is_configured() -> bool:
    """True only when a provider key is present — otherwise web search is a
    no-op and the report stays fully local."""
    return bool(_api_key())


def search(query: str, max_results: int = 3, timeout: int = 20) -> List[Dict[str, Any]]:
    """Raw provider search → [{title, url, text}]. Returns [] on any error or
    when unconfigured (fail-safe: never raises into report generation)."""
    if not is_configured() or not query.strip():
        return []
    try:
        if provider() == "tavily":
            resp = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": _api_key(),
                    "query": query,
                    "max_results": max_results,
                    "search_depth": "basic",
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            out: List[Dict[str, Any]] = []
            for r in (data.get("results") or [])[:max_results]:
                out.append(
                    {
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "text": r.get("content", ""),
                    }
                )
            return out
    except (requests.RequestException, ValueError, KeyError):
        return []
    return []


def policy_gated_search(
    query: str,
    *,
    policy: Optional[Dict[str, Any]] = None,
    audit: Optional[Callable[[Dict[str, Any]], None]] = None,
    max_results: int = 3,
) -> List[Dict[str, Any]]:
    """Screen the outbound query through the egress policy, THEN search.

    BLOCK  -> no request leaves the machine; returns [].
    REDACT -> search with the sensitive substrings masked.
    ALLOW  -> search with the query unchanged.
    """
    decision = evaluate(query, policy)
    if decision.action == BLOCK:
        if audit is not None:
            audit(
                {"stage": "web_search", "action": "block", "findings": decision.summary}
            )
        return []
    if decision.action == REDACT and decision.redacted_text is not None:
        if audit is not None:
            audit(
                {
                    "stage": "web_search",
                    "action": "redact",
                    "findings": decision.summary,
                }
            )
        return search(decision.redacted_text, max_results=max_results)
    return search(query, max_results=max_results)

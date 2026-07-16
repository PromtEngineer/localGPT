"""Conservative entity-scope filtering for retrieval candidates."""

from __future__ import annotations

import re
from typing import Any


_QUESTION_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "current",
    "give",
    "how",
    "i",
    "in",
    "is",
    "its",
    "the",
    "under",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
}

_CURRENT_QUERY_TERMS = {"active", "authoritative", "current", "latest"}
_CURRENT_MARKERS = ("current and authoritative", "status: current")
_ARCHIVED_MARKERS = (
    "archived",
    "do not use",
    "former",
    "historical",
    "retired",
    "superseded",
)


def query_entities(query: str) -> list[str]:
    """Return proper-name-like tokens while excluding sentence scaffolding."""
    tokens = re.findall(r"\b[A-Z][A-Za-z0-9_-]{2,}\b", query)
    return list(
        dict.fromkeys(
            token.lower()
            for token in tokens
            if token.lower() not in _QUESTION_WORDS
        )
    )


def _primary_instrument(document: dict[str, Any]) -> str | None:
    """Infer an explicitly declared primary instrument, if the text has one."""
    text = str(document.get("text") or "")
    patterns = (
        r"^#\s+(?:Archived\s+)?([A-Z][A-Za-z0-9_-]+)\b",
        r"^([A-Z][A-Za-z0-9_-]+)\s+"
        r"(?:maintenance|responsibility|revision|retirement)\b",
        r"\b(?:for|of)\s+(?:the\s+)?([A-Z][A-Za-z0-9_-]+)\s+instrument\b",
        r"\b([A-Z][A-Za-z0-9_-]+)\s+instrument\b",
        r'''["']?instrument["']?\s*:\s*["']([A-Z][A-Za-z0-9_-]+)["']''',
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            candidate = match.group(1).lower()
            if candidate not in _QUESTION_WORDS:
                return candidate

    return None


def _document_text(document: dict[str, Any]) -> str:
    return str(document.get("text") or "").lower()


def _prefer_current_documents(
    query: str, documents: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Drop explicitly obsolete records for questions asking for current state.

    This activates only when the query requests current state and at least one
    candidate explicitly identifies itself as current. Ambiguous corpora retain
    every candidate so that synthesis can surface the disagreement.
    """
    query_terms = set(re.findall(r"\b[a-z]+\b", query.lower()))
    if not query_terms.intersection(_CURRENT_QUERY_TERMS):
        return documents

    current = [
        document
        for document in documents
        if any(marker in _document_text(document) for marker in _CURRENT_MARKERS)
    ]
    if not current:
        return documents

    return [
        document
        for document in documents
        if not any(marker in _document_text(document) for marker in _ARCHIVED_MARKERS)
    ]


def filter_by_entity_scope(
    query: str, documents: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Exclude documents that explicitly declare a different primary entity.

    The filter activates only when the query contains a proper-name-like token.
    Neutral documents remain eligible, while a document about one named product
    cannot enter context merely because it also warns about another product.
    """
    entities = query_entities(query)
    if not entities or not documents:
        return documents
    scoped: list[dict[str, Any]] = []
    for document in documents:
        primary = _primary_instrument(document)
        if primary is not None:
            if primary in entities:
                scoped.append(document)
            continue

        # Documents without an explicit primary instrument remain neutral.
        scoped.append(document)

    return _prefer_current_documents(query, scoped or documents)

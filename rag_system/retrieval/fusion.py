"""Deterministic retrieval-result fusion without model dependencies."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


def _identity(item: Dict[str, Any]) -> str:
    return str(
        item.get("chunk_id")
        or item.get("_rowid")
        or (item.get("document_id"), item.get("chunk_index"))
    )


def fuse_ranked_results(
    lexical: Iterable[Dict[str, Any]],
    dense: Iterable[Dict[str, Any]],
    *,
    k: int,
    dense_weight: float = 0.5,
    rrf_constant: int = 60,
) -> List[Dict[str, Any]]:
    """Fuse lexical and dense rankings using weighted reciprocal-rank fusion."""
    dense_weight = min(1.0, max(0.0, float(dense_weight)))
    lexical_weight = 1.0 - dense_weight
    combined: dict[str, Dict[str, Any]] = {}

    for source, weight in ((lexical, lexical_weight), (dense, dense_weight)):
        for rank, item in enumerate(source, start=1):
            identity = _identity(item)
            if identity not in combined:
                combined[identity] = {**item, "score": 0.0}
            entry = combined[identity]
            entry["score"] += weight / (rrf_constant + rank)

    return sorted(combined.values(), key=lambda item: item["score"], reverse=True)[:k]

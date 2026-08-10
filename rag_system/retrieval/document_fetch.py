"""Reassemble a whole document from its indexed chunks (roadmap item 4.1).

This is the "deep read" half of full-document escalation: given any chunk's
``document_id``, pull every chunk of that document out of the LanceDB text table
and glue them back together **in ``chunk_index`` order**, capped at a token
budget. Chunk order is the point — DOS-RAG's finding is that handing a model the
document in its original order beats handing it the same text ranked by
similarity — so a document whose chunks cannot be ordered is not escalated at
all rather than escalated scrambled.

Nothing here calls an LLM or an embedder. It is a metadata filter and a sort.

Token counting is deliberately crude: ``len(text) // 4``. The exact number does
not need to be right, it needs to be *cheap* and never to under-count badly
enough to blow a context window; a 4-chars-per-token estimate runs slightly
conservative on English prose and needs no tokenizer load. It is reported as
``approx_tokens`` everywhere so no caller mistakes it for a real count.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Chunks are a few hundred tokens each; this is a hard stop on how many rows we
# will pull for one document, not an expected value.
_MAX_CHUNKS_SCANNED = 2000

_CHARS_PER_TOKEN = 4


def approximate_token_count(text: str) -> int:
    """A tokenizer-free token estimate: ``len(text) // 4``.

    See the module docstring for why this is not a real tokenizer count.
    """
    if not text:
        return 0
    return len(text) // _CHARS_PER_TOKEN


@dataclass
class FetchedDocument:
    """One reassembled document, already truncated to the caller's budget."""

    document_id: str
    document_name: str
    text: str
    chunks_used: int
    chunks_total: int
    approx_tokens: int
    truncated: bool
    chunk_indices: List[int] = field(default_factory=list)

    def as_event_payload(self) -> Dict[str, Any]:
        """The subset worth putting on the SSE wire (never the document text)."""
        return {
            "document_id": self.document_id,
            "document_name": self.document_name,
            "chunks_used": self.chunks_used,
            "chunks_total": self.chunks_total,
            "approx_tokens": self.approx_tokens,
            "truncated": self.truncated,
        }


def _parse_metadata(raw: Any) -> Dict[str, Any]:
    """The ``metadata`` column stores the whole chunk dict as a JSON string."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw:
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _chunk_text(row: Dict[str, Any]) -> str:
    """Prefer the clean ``original_text`` over the enriched/contextualised text.

    Indexing stores the chunk dict as JSON in the ``metadata`` column, so the
    clean text sits at ``metadata["metadata"]["original_text"]``. Older or
    hand-built rows put it one level up. The top-level ``text`` column is the
    last resort: with contextual enrichment on it carries a prepended
    "Context: …" summary, which is noise when the whole document is present.
    """
    meta = _parse_metadata(row.get("metadata"))
    inner = meta.get("metadata") if isinstance(meta.get("metadata"), dict) else {}
    for candidate in (inner.get("original_text"), meta.get("original_text"), row.get("text")):
        if isinstance(candidate, str) and candidate.strip():
            return candidate
    return ""


def _document_name(document_id: str, rows: List[Dict[str, Any]]) -> str:
    """A human-readable name: the indexed ``source`` path's basename."""
    for row in rows:
        meta = _parse_metadata(row.get("metadata"))
        inner = meta.get("metadata") if isinstance(meta.get("metadata"), dict) else {}
        source = inner.get("source") or meta.get("source")
        if isinstance(source, str) and source.strip():
            return os.path.basename(source.strip())
    # document_id is "<uuid>_<filename>" for files uploaded through the UI.
    if "_" in document_id:
        tail = document_id.split("_", 1)[1]
        if tail:
            return tail
    return document_id


def fetch_document(
    db_manager,
    table_name: str,
    document_id: str,
    token_budget: int = 6000,
) -> Optional[FetchedDocument]:
    """Reassemble ``document_id`` from ``table_name``, in chunk order.

    Returns ``None`` — never raises — when the table cannot be opened, the
    document has no rows, or the rows carry no usable ``chunk_index``. The
    caller treats that as "no escalation", which is the safe outcome.
    """
    if not document_id or not table_name:
        return None

    # The filter is interpolated into a SQL string, so refuse an id that could
    # terminate the literal instead of trying to escape it.
    if "'" in document_id or "\\" in document_id:
        print(f"⚠️  Refusing to escalate document id with quoting characters: {document_id!r}")
        return None

    try:
        table = db_manager.get_table(table_name)
    except Exception as e:
        print(f"⚠️  Full-document escalation could not open table '{table_name}': {e}")
        return None

    try:
        rows = (
            table.search()
            .where(f"document_id = '{document_id}'")
            .limit(_MAX_CHUNKS_SCANNED)
            .to_list()
        )
    except Exception as e:
        print(f"⚠️  Full-document escalation query failed for '{document_id}': {e}")
        return None

    if not rows:
        return None

    # Order is the whole point of this module: no order, no escalation.
    ordered: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        index = row.get("chunk_index")
        if index is None or index == -1:
            continue
        try:
            index = int(index)
        except (TypeError, ValueError):
            continue
        # A duplicate chunk_index means two rows claim the same slot (e.g. a
        # late-chunk table merged in). Keep the first and move on.
        ordered.setdefault(index, row)

    if not ordered:
        print(f"⚠️  Document '{document_id}' has no ordered chunks; skipping escalation.")
        return None

    chunks_total = len(ordered)
    budget_chars = max(0, int(token_budget)) * _CHARS_PER_TOKEN

    pieces: List[str] = []
    used_indices: List[int] = []
    length = 0
    truncated = False

    for index in sorted(ordered):
        text = _chunk_text(ordered[index]).strip()
        if not text:
            continue
        separator = 2 if pieces else 0  # the "\n\n" join
        remaining = budget_chars - length - separator
        if remaining <= 0:
            truncated = True
            break
        if len(text) > remaining:
            # Cut mid-chunk rather than dropping the chunk: the budget is a
            # context-window guard, and a partial final chunk still reads in order.
            pieces.append(text[:remaining])
            used_indices.append(index)
            length += remaining + separator
            truncated = True
            break
        pieces.append(text)
        used_indices.append(index)
        length += len(text) + separator

    if not pieces:
        return None

    body = "\n\n".join(pieces)
    return FetchedDocument(
        document_id=document_id,
        document_name=_document_name(document_id, rows),
        text=body,
        chunks_used=len(used_indices),
        chunks_total=chunks_total,
        approx_tokens=approximate_token_count(body),
        truncated=truncated,
        chunk_indices=used_indices,
    )


def format_escalation_block(document: FetchedDocument) -> str:
    """Wrap a reassembled document in the delimiter synthesis sees.

    Kept clearly separate from the retrieved snippets so the model can tell the
    two apart, and labelled "escalated" so the answer's sourcing story stays
    honest: the chunk citations are still the citations, this block is extra
    reading material.
    """
    header = f"FULL DOCUMENT (escalated): {document.document_name}"
    if document.truncated:
        header += (
            f" [truncated to ~{document.approx_tokens} tokens; "
            f"{document.chunks_used} of {document.chunks_total} chunks]"
        )
    return (
        "––––– " + header + " –––––\n"
        f"{document.text}\n"
        "––––– END FULL DOCUMENT –––––"
    )

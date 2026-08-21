"""Index-time extraction of intra-corpus cross-references (roadmap item 4.2).

Embeddings cannot follow a pointer. A chunk that says *"the fee schedule is set
out in Exhibit B"* is a perfect lexical and semantic match for a question about
fees, but the fee *numbers* live in a different document that shares almost no
vocabulary with the query. No amount of reranking recovers that document,
because it was never a candidate.

This module extracts those pointers with deterministic regexes at index time and
stores them on the chunk as ``metadata.crossrefs``::

    [{"kind": "exhibit" | "section" | "document",
      "ref": "<normalized reference>",
      "target_doc": "<document_id>" | None}]

Three deliberate limits:

* **No LLM.** Extraction is a handful of regexes over text we already have in
  memory, so it is free and can be on by default. The *query-time hop* that acts
  on this metadata is a separate, default-off flag.
* **Resolution is name-based only.** A reference resolves when some other
  document being indexed has a filename or title that contains it
  ("Exhibit B" -> ``exhibit_b.pdf``). Anything else stays ``target_doc: None``:
  the reference is still recorded (it is true, and a UI can show it), it just
  has nowhere to hop to.
* **Never resolves to the chunk's own document.** A contract that mentions
  "Exhibit B" fifty times inside ``exhibit_b.pdf`` produces no useful hop, and a
  document that repeats its own title would otherwise self-resolve on every
  chunk.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

# A chunk that carries more than this many distinct references is almost always
# a table of contents or an index page; keeping the first few is enough to hop.
MAX_CROSSREFS_PER_CHUNK = 8

# A document-name mention has to clear this bar before it counts, otherwise
# short generic filenames ("api.md", "v2.pdf") match half the corpus.
_MIN_NAME_CHARS = 6
_MIN_NAME_TOKENS = 2

# Label families that behave like "Exhibit B": a label word plus a letter or a
# dotted number. All of them are recorded under kind "exhibit"; the label word
# itself survives in ``ref`` ("schedule 2.1"), so nothing is conflated.
_LABEL_WORDS = ("exhibit", "appendix", "schedule", "annex", "attachment", "addendum")

# The label word is case-insensitive; the single-letter form is NOT, because a
# case-insensitive ``[A-Z]`` turns "appendix in the margin" into "appendix i".
_LABEL_RE = re.compile(
    r"\b(?i:(" + "|".join(_LABEL_WORDS) + r"))"
    r"[\s ]+(?i:no\.?[\s ]*|#[\s ]*)?"
    r"([A-Z](?![A-Za-z])|\d+(?:\.\d+)*)"
)

_SECTION_RE = re.compile(
    r"\b(?i:(section|clause|article))[\s ]+(\d+(?:\.\d+)*)\b"
)

_SECTION_SYMBOL_RE = re.compile(r"§[\s ]*(\d+(?:\.\d+)*)")

_EXTENSION_RE = re.compile(r"\.[A-Za-z0-9]{1,5}$")
_NON_ALNUM_RE = re.compile(r"[^0-9A-Za-z]+")


def normalize_name(value: str) -> str:
    """A document id, filename or title reduced to space-separated lowercase words.

    ``"Exhibit_B.pdf"`` and ``"exhibit b"`` both become ``"exhibit b"``, which is
    what makes a textual reference comparable with a filename.
    """
    if not value:
        return ""
    base = os.path.basename(str(value))
    base = _EXTENSION_RE.sub("", base)
    return _NON_ALNUM_RE.sub(" ", base).strip().lower()


def _lookup_key(ref: str) -> str:
    """``"section 4.3"`` -> ``"section 4 3"`` so it can be matched against names."""
    return _NON_ALNUM_RE.sub(" ", (ref or "").lower()).strip()


def _normalize_text(text: str) -> str:
    return _NON_ALNUM_RE.sub(" ", (text or "")).lower()


class CrossRefExtractor:
    """Extracts and resolves references against a known set of document ids.

    *known_documents* is the id set the references are resolved against: the
    documents in the current indexing batch, plus (best effort) whatever is
    already in the target table, so an incremental add can still point at an
    earlier one.
    """

    def __init__(self, known_documents: Sequence[str] = ()):
        # normalized name -> document id. Later duplicates lose, which is
        # arbitrary but stable, and duplicate normalized filenames in one corpus
        # are already ambiguous for a human.
        self._by_name: Dict[str, str] = {}
        for doc_id in known_documents or ():
            name = normalize_name(doc_id)
            if name:
                self._by_name.setdefault(name, doc_id)
            # Corpora ordered with numeric filename prefixes ("05_escrow_
            # agreement.pdf") are referenced in prose by title ("the Escrow
            # Agreement"), never by prefix, so index the stripped form too.
            # The full-name entry above still wins ties.
            stripped = re.sub(r"^\d+\s+", "", name)
            if stripped != name and (
                len(stripped) >= _MIN_NAME_CHARS or len(stripped.split()) >= _MIN_NAME_TOKENS
            ):
                self._by_name.setdefault(stripped, doc_id)
        # Longest names first so "northwind leave policy" wins over "policy".
        self._mention_names = sorted(
            (
                name for name in self._by_name
                if len(name) >= _MIN_NAME_CHARS or len(name.split()) >= _MIN_NAME_TOKENS
            ),
            key=len,
            reverse=True,
        )
        self._name_patterns = {
            name: re.compile(r"\b" + re.escape(name) + r"\b") for name in self._mention_names
        }
        # Cheap pre-computed word-boundary matchers for reference resolution.
        self._resolve_cache: Dict[str, Optional[str]] = {}

    # -- resolution --------------------------------------------------------

    def _resolve(self, ref: str, self_doc_id: Optional[str]) -> Optional[str]:
        key = _lookup_key(ref)
        if not key:
            return None
        cached = self._resolve_cache.get(key, "__miss__")
        if cached == "__miss__":
            pattern = re.compile(r"\b" + re.escape(key) + r"\b")
            hit = None
            # Prefer an exact name match, then a containing name.
            if key in self._by_name:
                hit = self._by_name[key]
            else:
                for name in sorted(self._by_name, key=len):
                    if pattern.search(name):
                        hit = self._by_name[name]
                        break
            self._resolve_cache[key] = hit
            cached = hit
        if cached is not None and self_doc_id is not None and cached == self_doc_id:
            return None
        return cached

    # -- extraction --------------------------------------------------------

    def extract(self, text: str, self_doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Every distinct reference in *text*, in order of first appearance."""
        if not text:
            return []

        found: List[Dict[str, Any]] = []
        seen = set()
        seen_refs = set()

        def add(kind: str, ref: str, target: Optional[str]) -> None:
            item = (kind, ref)
            if item in seen:
                return
            seen.add(item)
            seen_refs.add(ref)
            found.append({"kind": kind, "ref": ref, "target_doc": target})

        for match in _LABEL_RE.finditer(text):
            label = match.group(1).lower()
            value = match.group(2)
            ref = f"{label} {value.lower()}"
            add("exhibit", ref, self._resolve(ref, self_doc_id))

        for match in _SECTION_RE.finditer(text):
            ref = f"section {match.group(2)}"
            add("section", ref, self._resolve(ref, self_doc_id))

        for match in _SECTION_SYMBOL_RE.finditer(text):
            ref = f"section {match.group(1)}"
            add("section", ref, self._resolve(ref, self_doc_id))

        if self._mention_names:
            haystack = _normalize_text(text)
            self_name = normalize_name(self_doc_id) if self_doc_id else ""
            for name in self._mention_names:
                if name == self_name or name in seen_refs:
                    # Already recorded by the label pass ("Exhibit B" in a corpus
                    # that also has exhibit_b.pdf) — one record, not two.
                    continue
                target = self._by_name.get(name)
                if target is not None and target == self_doc_id:
                    continue
                if self._name_patterns[name].search(haystack):
                    add("document", name, target)

        return found[:MAX_CROSSREFS_PER_CHUNK]


def annotate_chunks(
    doc_chunks: Dict[str, List[Dict[str, Any]]],
    known_documents: Optional[Iterable[str]] = None,
) -> Dict[str, int]:
    """Stamp ``metadata.crossrefs`` on every chunk. Returns a small stats dict.

    *doc_chunks* maps document id -> that document's chunks (the shape
    ``IndexingPipeline`` already keeps for late chunking). Chunks are mutated in
    place; chunks with no references get no key at all, so the stored metadata
    does not grow for corpora that have none.
    """
    ids = list(doc_chunks.keys())
    if known_documents:
        for extra in known_documents:
            if extra not in ids:
                ids.append(extra)

    extractor = CrossRefExtractor(ids)
    stats = {"chunks_with_refs": 0, "refs": 0, "resolved": 0, "documents_linked": 0}
    linked_targets = set()

    for doc_id, chunks in doc_chunks.items():
        for chunk in chunks:
            text = chunk.get("text") or ""
            refs = extractor.extract(text, self_doc_id=doc_id)
            if not refs:
                continue
            chunk.setdefault("metadata", {})["crossrefs"] = refs
            stats["chunks_with_refs"] += 1
            stats["refs"] += len(refs)
            for ref in refs:
                if ref["target_doc"]:
                    stats["resolved"] += 1
                    linked_targets.add(ref["target_doc"])

    stats["documents_linked"] = len(linked_targets)
    return stats

from __future__ import annotations

import json
import re
from pathlib import Path

from ..config import Config

_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+)$")


def _est_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _split_blocks(md: str) -> list[str]:
    """Split page markdown into blocks; blank-line boundaries keep tables intact."""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", md) if b.strip()]
    return blocks


def chunk_doc(doc_dir: Path, doc_id: str, dataset: str, title: str, cfg: Config) -> list[dict]:
    max_tok = cfg.ingest.chunk_max_tokens
    chunks: list[dict] = []
    section = ""
    buf: list[str] = []
    buf_tok = 0
    buf_page = 1

    def flush() -> None:
        nonlocal buf, buf_tok
        if not buf:
            return
        body = "\n\n".join(buf)[: max_tok * 6]  # hard char cap: single huge lines can't be split
        header = f"{title} · {section} · p{buf_page}" if section else f"{title} · p{buf_page}"
        chunks.append(
            {
                "id": f"{doc_id}_c{len(chunks):04d}",
                "dataset": dataset,
                "doc_id": doc_id,
                "page": buf_page,
                "section": section,
                "text": f"{header}\n{body}" if cfg.ingest.contextual_headers else body,
                "raw_text": body,
                "n_tokens": _est_tokens(body),
            }
        )
        buf, buf_tok = [], 0

    with open(doc_dir / "md_pages.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            page, md = rec["page"], rec["md"]
            for block in _split_blocks(md):
                m = _HEADING_RE.match(block.split("\n", 1)[0])
                if m:
                    flush()
                    section = m.group(2).strip()[:120]
                    buf_page = page
                tok = _est_tokens(block)
                if tok > max_tok:  # oversized block (huge table): hard-split by lines
                    flush()
                    lines = block.split("\n")
                    part: list[str] = []
                    ptok = 0
                    for ln in lines:
                        ltok = _est_tokens(ln)
                        if ptok + ltok > max_tok and part:
                            buf, buf_tok, buf_page = ["\n".join(part)], ptok, page
                            flush()
                            part, ptok = [], 0
                        part.append(ln)
                        ptok += ltok
                    if part:
                        buf, buf_tok, buf_page = ["\n".join(part)], ptok, page
                        flush()
                    continue
                if buf_tok + tok > max_tok and buf:
                    flush()
                if not buf:
                    buf_page = page
                buf.append(block)
                buf_tok += tok
    flush()
    return chunks

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

import duckdb

from ..config import Config
from ..llm import LLM
from ..retrieve.hybrid import Retriever

MAX_RESULT_CHARS = 4000

_VLM_READ_PROMPT = """You are reading a rendered document page image to answer one specific question.
Rules:
- Read values EXACTLY as printed (axis labels, legends, bar/segment values, table cells, units).
- For charts: identify the relevant series/bar/segment by its legend color or label before reading its value; if a value must be estimated from axis position, say "approx" and give the tick marks you used.
- Quote the exact figure/table caption you are reading from.
- If the question cannot be answered from this page, say exactly what IS on the page instead."""


class ToolBox:
    """Multi-granularity corpus access for the search agent: search → grep → read → sql."""

    def __init__(self, cfg: Config, dataset: str, retriever: Retriever):
        self.cfg = cfg
        self.dataset = dataset
        self.retriever = retriever
        self.processed = cfg.path("processed", create=False) / dataset
        self._vlm: LLM | None = None  # lazy; shares the orchestrator's server
        # evidence tracking for marginal-utility stop
        self.evidence_seen: set[tuple[str, int]] = set()
        self.new_evidence_last_call = 0

    # ---------- tool implementations ----------

    def hybrid_search(self, query: str, top_k: int = 8) -> str:
        hits = self.retriever.search(query, self.dataset, k_final=min(int(top_k), 15))
        self._track([(h["doc_id"], h["page"]) for h in hits])
        out = []
        for h in hits:
            out.append(f"[{h['doc_id']} p{h['page']}] ({h['section']}) {h['raw_text'][:400]}")
        return _cap("\n---\n".join(out) or "no results")

    def grep(self, pattern: str, doc_id: str = "") -> str:
        try:
            rx = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return f"invalid regex: {e}"
        matches: list[str] = []
        docs = [self.processed / doc_id] if doc_id else sorted(self.processed.iterdir())
        for doc_dir in docs:
            pages = doc_dir / "pages.jsonl"
            if not pages.exists():
                continue
            with open(pages) as f:
                for line in f:
                    rec = json.loads(line)
                    for ln in rec["text"].split("\n"):
                        if rx.search(ln):
                            matches.append(f"[{doc_dir.name} p{rec['page']}] {ln.strip()[:200]}")
                            if len(matches) >= 40:
                                self._track_from_matches(matches)
                                return _cap("\n".join(matches) + "\n(truncated at 40 matches)")
        self._track_from_matches(matches)
        return _cap("\n".join(matches) or "no matches")

    def read_doc(self, doc_id: str, page_start: int, page_end: int) -> str:
        page_start, page_end = int(page_start), int(page_end)
        page_end = min(page_end, page_start + 5)  # cap window
        pages_file = self.processed / doc_id / "pages.jsonl"
        if not pages_file.exists():
            return f"unknown doc_id: {doc_id}"
        parts = []
        with open(pages_file) as f:
            for line in f:
                rec = json.loads(line)
                if page_start <= rec["page"] <= page_end:
                    parts.append(f"--- {doc_id} p{rec['page']} ---\n{rec['text']}")
        self._track([(doc_id, p) for p in range(page_start, page_end + 1)])
        return _cap("\n".join(parts) or f"no pages in range {page_start}-{page_end}")

    def list_docs(self) -> str:
        cm = self.processed / "corpus_map.md"
        return _cap(cm.read_text() if cm.exists() else "corpus map missing")

    def list_tables(self, doc_id: str = "") -> str:
        db = self.processed / "tables.duckdb"
        if not db.exists():
            return "no tables database"
        con = duckdb.connect(str(db), read_only=True)
        q = "SELECT doc_id, page, view_name, n_rows, n_cols, headers FROM _catalog"
        if doc_id:
            q += f" WHERE doc_id = '{doc_id}'"
        rows = con.execute(q + " LIMIT 80").fetchall()
        con.close()
        return _cap("\n".join(f"{r[2]} ({r[0]} p{r[1]}, {r[3]}x{r[4]}) cols={r[5]}" for r in rows) or "no tables")

    def sql(self, query: str) -> str:
        if re.search(r"\b(insert|update|delete|drop|create|alter|attach|copy)\b", query, re.I):
            return "read-only: SELECT queries only"
        db = self.processed / "tables.duckdb"
        con = duckdb.connect(str(db), read_only=True)
        try:
            df = con.execute(query).fetchdf()
            return _cap(df.head(50).to_markdown(index=False))
        except Exception as e:
            return f"sql error: {e}"
        finally:
            con.close()

    def view_page(self, doc_id: str, page: int, question: str) -> str:
        """Render-and-read: send the page image + a focused question to the VLM."""
        page = int(page)
        img = self.processed / doc_id / "pages" / f"p{page:04d}.png"
        if not img.exists():
            return f"no page image for {doc_id} p{page}"
        if self._vlm is None:
            self._vlm = LLM("orchestrator", self.cfg)
        b64 = base64.b64encode(img.read_bytes()).decode()
        try:
            reading = self._vlm.text(
                [
                    {"role": "system", "content": _VLM_READ_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Page {page} of {doc_id}. QUESTION: {question}"},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        ],
                    },
                ],
                max_tokens=2048,
                reasoning="none",  # vision+thinking starves even 3K budgets; no-think reads pages faster and fuller
            )
        except Exception as e:
            return f"view_page failed: {e}"
        self._track([(doc_id, page)])
        return _cap(f"[VLM reading of {doc_id} p{page}]\n{reading}")

    # ---------- plumbing ----------

    def _track(self, pairs: list[tuple[str, int]]) -> None:
        new = [p for p in pairs if p not in self.evidence_seen]
        self.new_evidence_last_call = len(new)
        self.evidence_seen.update(new)

    def _track_from_matches(self, matches: list[str]) -> None:
        pairs = []
        for m in matches:
            mm = re.match(r"\[(\S+) p(\d+)\]", m)
            if mm:
                pairs.append((mm.group(1), int(mm.group(2))))
        self._track(pairs)

    def dispatch(self, name: str, args: dict) -> str:
        fn = getattr(self, name, None)
        if fn is None or name.startswith("_"):
            return f"unknown tool: {name}"
        try:
            return fn(**args)
        except TypeError as e:
            return f"bad arguments for {name}: {e}"
        except Exception as e:
            return f"tool {name} failed: {e}"


def _cap(s: str) -> str:
    return s if len(s) <= MAX_RESULT_CHARS else s[:MAX_RESULT_CHARS] + "\n...(truncated)"


TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "hybrid_search",
            "description": "Semantic + keyword hybrid search over the document corpus. Returns chunks tagged [doc_id pN]. Start here.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "natural language search query"},
                    "top_k": {"type": "integer", "description": "number of results (default 8)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Regex search over raw document text. Use for exact strings, numbers, model names, section titles.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "doc_id": {"type": "string", "description": "optional: restrict to one document"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_doc",
            "description": "Read the full text of specific pages (max 6 pages per call). Use to confirm exact wording before citing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string"},
                    "page_start": {"type": "integer"},
                    "page_end": {"type": "integer"},
                },
                "required": ["doc_id", "page_start", "page_end"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_docs",
            "description": "List all documents in the corpus with titles, types, page counts.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_tables",
            "description": "List extracted tables (as SQL views) with their columns, optionally for one doc.",
            "parameters": {
                "type": "object",
                "properties": {"doc_id": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sql",
            "description": "Run a read-only DuckDB SELECT over extracted tables (see list_tables for view names). REQUIRED for any arithmetic over table data — never do math in your head.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "view_page",
            "description": "Look at the rendered page image with vision and answer a focused question about it. USE THIS when the evidence is in a chart, figure, diagram, or drawing whose values are not in the page text — text search cannot see chart bars, pie slices, or axis values.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string"},
                    "page": {"type": "integer"},
                    "question": {"type": "string", "description": "the specific thing to read off this page, e.g. 'What value does the 2019 bar show for Google in Figure 6?'"},
                },
                "required": ["doc_id", "page", "question"],
            },
        },
    },
]

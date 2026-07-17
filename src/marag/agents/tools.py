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
- If the detail is too small to read confidently, say so explicitly and name the region of the
  page it sits in (e.g. "bottom-right") so it can be re-rendered zoomed.
- If the question cannot be answered from this page, say exactly what IS on the page instead."""

# Appended only in auto-zoom LOCATE mode, so the default reading text stays unchanged.
_VLM_LOCATE_SUFFIX = """
- END with a final line `REGION: <name>` naming the SMALLEST region that fully contains the
  content relevant to the question, so it can be re-rendered zoomed. Strongly prefer a corner
  (top-left/top-right/middle-left/middle-right/bottom-left/bottom-right) — those get the highest
  resolution. Use a half (top/bottom/left/right) only when content spans the full width/height,
  `full` only when it covers the page, `none` if it is not on this page. A multi-panel figure's
  first panel is usually top-left; name that panel's region, not the whole figure's."""

_REGION_RE = re.compile(r"REGION:\s*([a-z-]+)", re.IGNORECASE)

# Corners are ~thirds tall so a zoomed quadrant clears the pixel cap at full zoom dpi.
_REGIONS: dict[str, tuple[float, float, float, float]] = {
    "full": (0.0, 0.0, 1.0, 1.0),
    "top": (0.0, 0.0, 1.0, 0.55),
    "bottom": (0.0, 0.45, 1.0, 1.0),
    "left": (0.0, 0.0, 0.55, 1.0),
    "right": (0.45, 0.0, 1.0, 1.0),
    "top-left": (0.0, 0.0, 0.58, 0.42),
    "top-right": (0.42, 0.0, 1.0, 0.42),
    "middle-left": (0.0, 0.30, 0.58, 0.72),
    "middle-right": (0.42, 0.30, 1.0, 0.72),
    "bottom-left": (0.0, 0.58, 0.58, 1.0),
    "bottom-right": (0.42, 0.58, 1.0, 1.0),
    "center": (0.15, 0.28, 0.85, 0.74),
}


class ToolBox:
    """Multi-granularity corpus access for the search agent: search → grep → read → sql."""

    def __init__(self, cfg: Config, dataset: str, retriever: Retriever):
        self.cfg = cfg
        self.dataset = dataset
        self.retriever = retriever
        self.processed = cfg.path("processed", create=False) / dataset
        self._vlm: LLM | None = None  # lazy; shares the orchestrator's server
        self._pdfs: dict[str, object] = {}  # doc_id -> open fitz doc, for high-DPI renders
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

    def summarize_doc(self, doc_id: str) -> str:
        """Whole-document summary via map-reduce over every chunk; cached to disk.

        This is the document-level primitive retrieval can't provide: top-k search assumes
        the answer lives in a few chunks, a summary needs all of them.
        """
        doc_dir = self.processed / doc_id
        cache = doc_dir / "summary.md"
        meta_f = doc_dir / "meta.json"
        if not meta_f.exists():
            return f"unknown doc_id: {doc_id}"
        meta = json.loads(meta_f.read_text())
        title = meta.get("title", doc_id)
        if cache.exists():
            self._track([(doc_id, p) for p in range(1, meta.get("n_pages", 1) + 1)])
            return _cap(f"[summary of {doc_id} · {title} (cached)]\n{cache.read_text()}")

        chunks_f = doc_dir / "chunks.jsonl"
        if not chunks_f.exists():
            return f"no parsed content for {doc_id}"
        texts = [json.loads(l)["raw_text"] for l in chunks_f.read_text().splitlines()]
        if not texts:
            return f"no parsed content for {doc_id}"

        util = LLM("utility", self.cfg)
        # map: ~10k-char batches → section notes on the small model
        batches: list[str] = []
        buf = ""
        for t in texts:
            if len(buf) + len(t) > 10_000 and buf:
                batches.append(buf)
                buf = ""
            buf += t + "\n\n"
        if buf:
            batches.append(buf)
        notes = []
        for i, b in enumerate(batches):
            notes.append(util.text(
                [{"role": "system", "content": "Summarize this document section faithfully in 5-8 bullet points. Keep exact key figures, names and dates. No preamble."},
                 {"role": "user", "content": f"Section {i + 1}/{len(batches)} of {title!r}:\n\n{b[:12000]}"}],
                max_tokens=700, reasoning="none",
            ))
        if len(notes) == 1:
            summary = notes[0]
        else:  # reduce on the orchestrator
            orch = LLM("orchestrator", self.cfg)
            summary = orch.text(
                [{"role": "system", "content": "Merge these section notes into one faithful summary of the whole document: a 2-3 sentence overview, then the key points as bullets grouped by theme. Keep exact figures. No preamble."},
                 {"role": "user", "content": f"Document: {title}\n\n" + "\n\n".join(f"[section {i + 1}]\n{n}" for i, n in enumerate(notes))}],
                max_tokens=4096,
            )
        cache.write_text(summary)
        self._track([(doc_id, p) for p in range(1, meta.get("n_pages", 1) + 1)])
        return _cap(f"[summary of {doc_id} · {title} · {len(batches)} sections mapped]\n{summary}")

    def has_tables(self) -> bool:
        db = self.processed / "tables.duckdb"
        if not db.exists():
            return False
        con = duckdb.connect(str(db), read_only=True)
        try:
            return con.execute("SELECT count(*) FROM _catalog").fetchone()[0] > 0
        except Exception:
            return False
        finally:
            con.close()

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

    def _render(self, doc_id: str, page: int, region: str) -> bytes | str:
        """Full page → the stored 150-dpi PNG; a region → re-rasterized from the source PDF.

        Measured (health_docs, 15 q): re-rendering the FULL page at 220 dpi instead of using
        the stored PNG scored WORSE (73.3% vs 80.0%), so the validated path is kept as the
        default. Zoom stays available because it costs nothing unused — but note it did not
        fix chart reading either (see RESULTS.md "high-DPI" postmortem).
        """
        import fitz

        if region == "full":
            png = self.processed / doc_id / "pages" / f"p{page:04d}.png"
            if png.exists():
                return png.read_bytes()

        meta_f = self.processed / doc_id / "meta.json"
        if not meta_f.exists():
            return f"unknown doc_id: {doc_id}"
        pdf_path = json.loads(meta_f.read_text()).get("source_pdf")
        if not pdf_path or not Path(pdf_path).exists():
            return f"source PDF unavailable for {doc_id}"
        if doc_id not in self._pdfs:
            self._pdfs[doc_id] = fitz.open(pdf_path)
        doc = self._pdfs[doc_id]
        if not 1 <= page <= len(doc):
            return f"{doc_id} has {len(doc)} pages; p{page} is out of range"

        pg = doc[page - 1]
        rect = pg.rect
        fx0, fy0, fx1, fy1 = _REGIONS.get(region, _REGIONS["full"])
        clip = fitz.Rect(
            rect.x0 + fx0 * rect.width,
            rect.y0 + fy0 * rect.height,
            rect.x0 + fx1 * rect.width,
            rect.y0 + fy1 * rect.height,
        )
        dpi = self.cfg.agent.view_page_dpi if region == "full" else self.cfg.agent.view_page_zoom_dpi
        pix = pg.get_pixmap(dpi=dpi, clip=clip)
        cap = self.cfg.agent.view_page_max_px  # bound vision latency
        if max(pix.width, pix.height) > cap:
            dpi = max(72, int(dpi * cap / max(pix.width, pix.height)))
            pix = pg.get_pixmap(dpi=dpi, clip=clip)
        return pix.tobytes("png")

    def _read(self, doc_id: str, page: int, question: str, region: str, think: bool, locate: bool) -> str:
        """One VLM read of a page/region. Returns the reading text, or an error string."""
        png = self._render(doc_id, page, region)
        if isinstance(png, str):
            return png
        if self._vlm is None:
            self._vlm = LLM("vision", self.cfg)  # falls back to orchestrator when models.vision unset
        b64 = base64.b64encode(png).decode()
        where = f"Page {page} of {doc_id}" + (f", {region} region (zoomed in)" if region != "full" else "")
        prompt = _VLM_READ_PROMPT + (_VLM_LOCATE_SUFFIX if locate else "")
        msgs = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{where}. QUESTION: {question}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            },
        ]
        if not think:
            return self._vlm.text(msgs, max_tokens=2048, reasoning="none")
        # thinking lets a dense VLM interpolate axis ticks, but starves without a big budget
        out = self._vlm.text(msgs, max_tokens=self.cfg.agent.view_page_max_tokens)
        return out if out.strip() else self._vlm.text(msgs, max_tokens=2048, reasoning="none")

    def view_page(self, doc_id: str, page: int, question: str, region: str = "") -> str:
        """Read a page with vision to answer a focused question.

        Default (validated): one full-page read, no thinking. Opt-in via agent.view_page_auto_zoom
        + view_page_thinking + models.vision: a cheap full-page locate pass reads a REGION hint,
        then that region is re-rendered at high dpi and read with thinking. See RESULTS.md.
        """
        page = int(page)
        think = self.cfg.agent.view_page_thinking
        err = ("unknown doc_id", "source PDF", f"{doc_id} has")
        try:
            if region in _REGIONS and region != "full":  # caller already knows where to look
                reading = self._read(doc_id, page, question, region, think, locate=False)
                if reading.startswith(err):
                    return reading
                self._track([(doc_id, page)])
                return _cap(f"[VLM reading of {doc_id} p{page}, region={region}]\n{reading}")

            if not self.cfg.agent.view_page_auto_zoom:  # validated single-read path
                reading = self._read(doc_id, page, question, "full", think, locate=False)
                if reading.startswith(err):
                    return reading
                self._track([(doc_id, page)])
                return _cap(f"[VLM reading of {doc_id} p{page}, region=full]\n{reading}")

            locate = self._read(doc_id, page, question, "full", think=False, locate=True)
            if locate.startswith(err):
                return locate
            self._track([(doc_id, page)])
            hint = _REGION_RE.search(locate)
            target = hint.group(1).strip().lower() if hint else ""
            if target not in _REGIONS or target == "full":
                return _cap(f"[VLM reading of {doc_id} p{page}, region=full]\n{_REGION_RE.sub('', locate).strip()}")
            zoom = self._read(doc_id, page, question, target, think=True, locate=False)
            return _cap(
                f"[VLM reading of {doc_id} p{page}] located the relevant content in the {target} "
                f"region and re-read it zoomed.\nZOOMED READING (region={target}, authoritative — "
                f"prefer these values over any full-page estimate):\n{zoom}"
            )
        except Exception as e:
            return f"view_page failed: {e}"

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
            "name": "summarize_doc",
            "description": "Faithful summary of an ENTIRE document (map-reduce over every chunk, cached after first use). USE for whole-document requests — 'summarize X', 'what is this document about', 'give me an overview' — instead of paging through it with read_doc, which cannot cover a long document within budget.",
            "parameters": {
                "type": "object",
                "properties": {"doc_id": {"type": "string"}},
                "required": ["doc_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "view_page",
            "description": "Look at the rendered page image with vision and answer a focused question about it. USE THIS when the evidence is in a chart, figure, diagram, or drawing whose values are not in the page text — text search cannot see chart bars, pie slices, or axis values. If a value is small or hard to read, call again with a region to zoom in at high resolution.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string"},
                    "page": {"type": "integer"},
                    "question": {"type": "string", "description": "the specific thing to read off this page, e.g. 'What value does the 2019 bar show for Google in Figure 6?'"},
                    "region": {
                        "type": "string",
                        "enum": ["full", "top", "bottom", "left", "right", "top-left", "top-right", "middle-left", "middle-right", "bottom-left", "bottom-right", "center"],
                        "description": "optional part of the page to zoom into, rendered at high resolution. Omit to read the whole page. Pass a region when you must read a precise value off a chart — a zoomed quadrant resolves tick marks and small labels that are unreadable in the full page.",
                    },
                },
                "required": ["doc_id", "page", "question"],
            },
        },
    },
]

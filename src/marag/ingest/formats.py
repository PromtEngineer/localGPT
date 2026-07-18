from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from ..config import Config
from .parse import _safe_columns

# ~800 tokens per pseudo-page: keeps grep/read_doc/citations meaningful for formats
# that have no fixed page layout (DOCX reflows; HTML never had pages).
PSEUDO_PAGE_CHARS = 3200

DOC_FORMATS = {".docx", ".pptx", ".html", ".htm", ".md"}
DATA_FORMATS = {".csv", ".xlsx", ".xls", ".parquet"}
IMAGE_FORMATS = {".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"}
AUDIO_FORMATS = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}

# Cap the longest edge of a normalized image page: keeps the visual late-interaction
# channel + view_page fast without throwing away legible detail.
IMAGE_MAX_EDGE = 2000

# Isolate every ML model call behind this switch. When set, ingestion produces the same
# artifact shape with a placeholder caption/transcript and NEVER imports a model — so the
# whole pipeline is unit-testable on a machine whose GPU is busy. The real caption/transcribe
# run is queued for later with MARAG_SKIP_MODELS unset.
def _skip_models() -> bool:
    return os.environ.get("MARAG_SKIP_MODELS") == "1"


def parser_for(suffix: str) -> str | None:
    """Map a file extension (any case) to its parser category, or None if unsupported.

    Factored out so the dispatch in pipeline.py is a data lookup and stays unit-testable
    (uppercase extensions like .PNG/.MP3 route the same as lowercase)."""
    s = suffix.lower()
    if s == ".pdf":
        return "pdf"
    if s in DOC_FORMATS:
        return "office"
    if s in DATA_FORMATS:
        return "data"
    if s in IMAGE_FORMATS:
        return "image"
    if s in AUDIO_FORMATS:
        return "audio"
    return None


def _write_pages(out_dir: Path, pages: list[str]) -> None:
    with open(out_dir / "pages.jsonl", "w") as f:
        for i, t in enumerate(pages):
            f.write(json.dumps({"page": i + 1, "text": t}) + "\n")
    with open(out_dir / "md_pages.jsonl", "w") as f:
        for i, t in enumerate(pages):
            f.write(json.dumps({"page": i + 1, "md": t}) + "\n")


def parse_office_doc(path: Path, out_dir: Path, cfg: Config) -> dict:
    """DOCX/PPTX/HTML/MD via Docling → markdown + tables, split into pseudo-pages.

    No page images and no visual channel (these formats have no canonical layout);
    citations are pseudo-page-level. Tables land in DuckDB like PDF tables.
    """
    from docling.document_converter import DocumentConverter

    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = out_dir / "tables_docling"
    tables_dir.mkdir(exist_ok=True)

    doc = DocumentConverter().convert(str(path)).document
    md = doc.export_to_markdown()

    catalog: list[dict] = []
    for j, t in enumerate(doc.tables):
        try:
            try:
                df = t.export_to_dataframe(doc=doc)
            except TypeError:
                df = t.export_to_dataframe()
            if df.empty:
                continue
            df.columns = _safe_columns(list(df.columns))
            pq = tables_dir / f"p0001_t{j}.parquet"
            df.to_parquet(pq)
            catalog.append(
                {"page": 1, "table_index": j, "parquet": pq.name, "n_rows": int(df.shape[0]),
                 "n_cols": int(df.shape[1]), "headers": list(df.columns)[:12]}
            )
        except Exception:
            continue
    (tables_dir / "catalog.json").write_text(json.dumps(catalog, indent=1))

    pages = [md[i : i + PSEUDO_PAGE_CHARS] for i in range(0, len(md), PSEUDO_PAGE_CHARS)] or [md]
    _write_pages(out_dir, pages)
    return {
        "n_pages": len(pages), "n_tables": len(catalog), "visual_primary": False,
        "source_pdf": None, "source_format": path.suffix.lstrip(".").lower(),
        "low_text_pages": 0,
    }


def _sheet_profile(doc_id: str, sheet: str, idx: int, df: pd.DataFrame, filename: str) -> str:
    """A searchable text card for one sheet: what it holds and the exact SQL view to query.

    The raw grid is NOT embedded — retrieval finds this profile, the agent queries DuckDB.
    """
    from ..ingest.pipeline import _slug

    view = f"t_{_slug(doc_id)}_p{idx + 1}_{idx}"  # must match _build_duckdb's naming for this catalog entry
    cols = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            cols.append(f"{c} (number, {s.min():g}–{s.max():g})" if len(s.dropna()) else f"{c} (number)")
        elif pd.api.types.is_datetime64_any_dtype(s):
            cols.append(f"{c} (date, {s.min()}–{s.max()})")
        else:
            u = s.nunique()
            sample = ", ".join(map(str, s.dropna().unique()[:4]))
            cols.append(f"{c} (text, {u} unique: {sample}{'…' if u > 4 else ''})")
    head = df.head(5).to_markdown(index=False)
    return (
        f"Data sheet {sheet!r} from {filename} — {len(df):,} rows × {df.shape[1]} columns.\n"
        f"Columns: {'; '.join(cols)}\n"
        f"Sample rows:\n{head}\n\n"
        f"Query the FULL data with sql, e.g.: SELECT ... FROM {view}"
    )


def parse_data_file(path: Path, out_dir: Path, cfg: Config, doc_id: str) -> dict:
    """CSV/XLSX/Parquet → full sheets registered in DuckDB + one profile pseudo-page per sheet."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = out_dir / "tables_docling"
    tables_dir.mkdir(exist_ok=True)

    suffix = path.suffix.lower()
    if suffix == ".csv":
        sheets = {"data": pd.read_csv(path)}
    elif suffix == ".parquet":
        sheets = {"data": pd.read_parquet(path)}
    else:
        sheets = pd.read_excel(path, sheet_name=None)

    catalog: list[dict] = []
    profiles: list[str] = []
    for name, df in sheets.items():
        if df.empty:
            continue
        # catalog page, view-name suffix and pseudo-page must all number from ONE counter
        # that skips empty sheets, or citations point at pages with no table
        idx = len(catalog)
        df.columns = _safe_columns(list(df.columns))
        pq = tables_dir / f"p{idx + 1:04d}_t{idx}.parquet"
        df.to_parquet(pq)
        catalog.append(
            {"page": idx + 1, "table_index": idx, "parquet": pq.name, "n_rows": int(df.shape[0]),
             "n_cols": int(df.shape[1]), "headers": list(df.columns)[:12]}
        )
        profiles.append(_sheet_profile(doc_id, name, idx, df, path.name))
    (tables_dir / "catalog.json").write_text(json.dumps(catalog, indent=1))
    _write_pages(out_dir, profiles or ["(empty data file)"])
    return {
        "n_pages": len(profiles), "n_tables": len(catalog), "visual_primary": False,
        "source_pdf": None, "source_format": suffix.lstrip("."), "low_text_pages": 0,
        "data_file": True,
    }


# ---------------------------------------------------------------------------
# Image ingestion — a single normalized page PNG + a VLM caption as its text.
# ---------------------------------------------------------------------------


def _normalize_image(src: Path, dst: Path, max_edge: int = IMAGE_MAX_EDGE) -> tuple[int, int]:
    """Write `src` to `dst` as an RGB PNG with the longest edge capped at `max_edge`.

    Normalizing to a page PNG lets the existing visual late-interaction channel and the
    view_page tool operate on an image doc natively, exactly like a rendered PDF page."""
    from PIL import Image

    resample = getattr(Image, "Resampling", Image).LANCZOS
    with Image.open(src) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = max_edge / max(w, h)
        if scale < 1:
            im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), resample)
        dst.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst, format="PNG")
        return im.size


def _placeholder_caption(png_path: Path) -> str:
    from PIL import Image

    with Image.open(png_path) as im:
        w, h = im.size
    return (
        f"[caption pending — MARAG_SKIP_MODELS set] Image page {png_path.name} ({w}x{h}px). "
        "Re-ingest without MARAG_SKIP_MODELS on the GPU to replace this with a real VLM caption."
    )


_IMAGE_CAPTION_PROMPT = (
    "You are describing an image so it can be found by text search and answered about later. "
    "Write a faithful, information-dense caption: name every object, person, chart, diagram, "
    "logo and any legible text or numbers exactly as printed. No preamble, no speculation."
)


def caption_image(png_path: Path, cfg: Config) -> str:
    """VLM caption for one image page. This is the ONLY model touchpoint for image ingestion;
    everything else is model-free, so ingestion is fully testable via the skip-guard / a
    caption_fn override. The real call is deferred to the queued GPU run."""
    if _skip_models():
        return _placeholder_caption(png_path)
    import base64

    from ..llm import LLM

    b64 = base64.b64encode(png_path.read_bytes()).decode()
    vlm = LLM("vision", cfg)  # falls back to orchestrator when models.vision is unset
    return vlm.text(
        [
            {"role": "system", "content": _IMAGE_CAPTION_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image for retrieval."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            },
        ],
        max_tokens=1024,
        reasoning="none",
    )


def parse_image_file(path: Path, out_dir: Path, cfg: Config, doc_id: str, caption_fn=None) -> dict:
    """Standalone image → one page PNG (pages/p0001.png) + a VLM caption as the page text.

    The caption is the retrievable/indexable text (dense + BM25 pick it up via chunk_doc);
    the normalized PNG feeds the visual channel and view_page. `caption_fn` (or the
    MARAG_SKIP_MODELS guard inside caption_image) keeps this model-free for tests."""
    path = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "pages" / "p0001.png"
    _normalize_image(path, png_path)

    fn = caption_fn or caption_image
    caption = fn(png_path, cfg) or "(no caption)"
    _write_pages(out_dir, [caption])
    return {
        "n_pages": 1,
        "n_tables": 0,
        "visual_primary": True,
        "doc_type": "image",
        "source_pdf": None,
        "source_image": str(path.resolve()),
        "source_format": path.suffix.lstrip(".").lower(),
        "low_text_pages": 0,
    }


# ---------------------------------------------------------------------------
# Audio ingestion — timestamped transcript split into [mm:ss]-marked pseudo-pages.
# ---------------------------------------------------------------------------


def _fmt_ts(seconds: float) -> str:
    s = max(0, int(seconds))
    return f"{s // 60:02d}:{s % 60:02d}"


def _segments_to_pages(segments: list[dict], max_chars: int = PSEUDO_PAGE_CHARS) -> list[str]:
    """Group timestamped segments into ~max_chars pseudo-pages (mirrors parse_office_doc's
    char cap), each segment prefixed with an [mm:ss] marker at its start time so citations
    and read_doc point at a real timestamp in the recording."""
    pages: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for seg in segments:
        text = str(seg.get("text", "")).strip()
        piece = f"[{_fmt_ts(seg.get('start', 0.0))}] {text}".rstrip()
        piece_len = len(piece) + 1  # +1 for the newline join
        if buf and buf_len + piece_len > max_chars:
            pages.append("\n".join(buf))
            buf, buf_len = [], 0
        buf.append(piece)
        buf_len += piece_len
    if buf:
        pages.append("\n".join(buf))
    return pages


def transcribe_audio(path: Path, cfg: Config) -> list[dict]:
    """Timestamped transcription via faster-whisper (CTranslate2, CPU-capable). This is the
    ONLY model touchpoint for audio ingestion; under MARAG_SKIP_MODELS it returns a single
    placeholder segment and NEVER imports whisper, so ingestion is testable without a model.
    Returns a list of {"start", "end", "text"} segments."""
    if _skip_models():
        return [
            {
                "start": 0.0,
                "end": 0.0,
                "text": (
                    f"[transcript pending — MARAG_SKIP_MODELS set] Audio {Path(path).name}. "
                    "Re-ingest without MARAG_SKIP_MODELS on the GPU for a real faster-whisper "
                    "transcript."
                ),
            }
        ]
    from faster_whisper import WhisperModel

    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, _info = model.transcribe(str(path))
    return [
        {"start": float(s.start), "end": float(s.end), "text": (s.text or "").strip()}
        for s in segments
    ]


def parse_audio_file(path: Path, out_dir: Path, cfg: Config, doc_id: str, transcript=None) -> dict:
    """Standalone audio → timestamped transcript split into [mm:ss]-marked pseudo-pages.

    No page images (audio has no visual channel). `transcript` (a list of {"start","text"}
    segments) or the MARAG_SKIP_MODELS guard inside transcribe_audio keeps this model-free
    for tests."""
    path = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    segments = transcript if transcript is not None else transcribe_audio(path, cfg)
    pages = _segments_to_pages(segments) or ["(empty transcript)"]
    _write_pages(out_dir, pages)
    return {
        "n_pages": len(pages),
        "n_tables": 0,
        "visual_primary": False,
        "doc_type": "audio",
        "source_pdf": None,
        "source_audio": str(path.resolve()),
        "source_format": path.suffix.lstrip(".").lower(),
        "low_text_pages": 0,
    }

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..config import Config
from .parse import _safe_columns

# ~800 tokens per pseudo-page: keeps grep/read_doc/citations meaningful for formats
# that have no fixed page layout (DOCX reflows; HTML never had pages).
PSEUDO_PAGE_CHARS = 3200

DOC_FORMATS = {".docx", ".pptx", ".html", ".htm", ".md"}
DATA_FORMATS = {".csv", ".xlsx", ".xls", ".parquet"}


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
    for j, (name, df) in enumerate(sheets.items()):
        if df.empty:
            continue
        df.columns = _safe_columns(list(df.columns))
        pq = tables_dir / f"p0001_t{j}.parquet"
        df.to_parquet(pq)
        catalog.append(
            {"page": j + 1, "table_index": j, "parquet": pq.name, "n_rows": int(df.shape[0]),
             "n_cols": int(df.shape[1]), "headers": list(df.columns)[:12]}
        )
        profiles.append(_sheet_profile(doc_id, name, j, df, path.name))
    (tables_dir / "catalog.json").write_text(json.dumps(catalog, indent=1))
    _write_pages(out_dir, profiles or ["(empty data file)"])
    return {
        "n_pages": len(profiles), "n_tables": len(catalog), "visual_primary": False,
        "source_pdf": None, "source_format": suffix.lstrip("."), "low_text_pages": 0,
        "data_file": True,
    }

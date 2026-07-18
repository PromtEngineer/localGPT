from __future__ import annotations

import json
import re
from pathlib import Path

import duckdb

from ..config import Config

_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _safe_doc_dir(cfg: Config, dataset: str, doc_id: str) -> Path | None:
    """Join + resolve, refusing any result outside the processed root (path traversal)."""
    root = cfg.path("processed", create=False).resolve()
    p = (root / dataset / doc_id).resolve()
    return p if p.is_relative_to(root) else None


def _table_catalog(doc_dir: Path) -> list[dict]:
    for sub in ("tables_docling", "tables"):
        f = doc_dir / sub / "catalog.json"
        if f.exists():
            return json.loads(f.read_text())
    return []


def list_sources(cfg: Config) -> list[dict]:
    """Every ingested dataset with its index/channel status — backs the sources rail."""
    processed = cfg.path("processed", create=False)
    if not processed.exists():
        return []
    out = []
    for ds_dir in sorted(processed.iterdir()):
        if not ds_dir.is_dir():
            continue
        metas = [json.loads(p.read_text()) for p in ds_dir.glob("*/meta.json")]
        if not metas:
            continue
        tables = sum(len(_table_catalog(ds_dir / m["doc_id"])) for m in metas)
        has_dense = (cfg.path("index", create=False) / "lance").exists()
        has_visual = (cfg.path("index", create=False) / f"visual_{ds_dir.name}.npz").exists()
        bench = (cfg.path("benchmarks", create=False) / f"{ds_dir.name}.json").exists()
        out.append({
            "id": ds_dir.name,
            "docs": len(metas),
            "pages": sum(m["n_pages"] for m in metas),
            "chunks": sum(m.get("n_chunks", 0) for m in metas),
            "tables": tables,
            "channels": {"dense": has_dense, "bm25": has_dense, "visual": has_visual},
            "benchmark": bench,
        })
    return out


def dataset_of(cfg: Config, doc_id: str) -> str | None:
    """Which ingested source owns this doc_id — resolves citations regardless of source."""
    processed = cfg.path("processed", create=False)
    if not processed.exists() or not _ID_RE.match(doc_id):
        return None
    for ds_dir in sorted(processed.iterdir()):
        if (ds_dir / doc_id / "meta.json").exists():
            return ds_dir.name
    return None


def list_docs(cfg: Config, dataset: str) -> list[dict]:
    root = cfg.path("processed", create=False).resolve()
    ds_dir = (root / dataset).resolve()
    if not ds_dir.is_relative_to(root):
        return []
    docs = []
    for p in sorted(ds_dir.glob("*/meta.json")):
        m = json.loads(p.read_text())
        docs.append({
            "doc_id": m["doc_id"], "title": m.get("title", ""),
            "doc_type": m.get("doc_type", ""), "pages": m["n_pages"],
            "tables": m.get("n_tables", 0), "visual_primary": m.get("visual_primary", False),
        })
    return docs


def page_evidence(cfg: Config, dataset: str, doc_id: str, page: int) -> dict:
    """Tables + text for one page — backs the evidence panel."""
    doc_dir = _safe_doc_dir(cfg, dataset, doc_id)
    if doc_dir is None:
        return {"doc_id": doc_id, "page": page, "title": "", "doc_type": "",
                "n_pages": 0, "text": "", "tables": []}
    meta = json.loads((doc_dir / "meta.json").read_text()) if (doc_dir / "meta.json").exists() else {}
    text = ""
    pages_f = doc_dir / "pages.jsonl"
    if pages_f.exists():
        for line in pages_f.read_text().splitlines():
            rec = json.loads(line)
            if rec["page"] == page:
                text = rec["text"]
                break
    tables = []
    db = doc_dir.parent / "tables.duckdb"
    if db.exists():
        con = duckdb.connect(str(db), read_only=True)
        try:
            rows = con.execute(
                "SELECT view_name, n_rows, n_cols, headers FROM _catalog WHERE doc_id=? AND page=?",
                (doc_id, page),
            ).fetchall()
            for v, nr, nc, headers in rows:
                df = con.execute(f"SELECT * FROM {v} LIMIT 12").fetchdf()
                tables.append({
                    "view": v, "n_rows": nr, "n_cols": nc,
                    "columns": list(df.columns),
                    "rows": df.astype(str).values.tolist(),
                })
        except Exception:
            pass
        finally:
            con.close()
    return {
        "doc_id": doc_id, "page": page,
        "title": meta.get("title", ""), "doc_type": meta.get("doc_type", ""),
        "n_pages": meta.get("n_pages", 0), "text": text[:4000], "tables": tables,
    }


def page_image_path(cfg: Config, dataset: str, doc_id: str, page: int) -> Path | None:
    doc_dir = _safe_doc_dir(cfg, dataset, doc_id)
    if doc_dir is None:
        return None
    p = doc_dir / "pages" / f"p{page:04d}.png"
    return p if p.exists() else None


def render_region_png(cfg: Config, dataset: str, doc_id: str, page: int, region: str) -> bytes | None:
    """Re-rasterize a page region from the source PDF at high dpi — the same crop the
    view_page tool reads. Lets the evidence grid zoom exactly like the agent does."""
    import fitz

    from ..agents.tools import _REGIONS

    if region not in _REGIONS or region == "full":
        return None
    doc_dir = _safe_doc_dir(cfg, dataset, doc_id)
    if doc_dir is None:
        return None
    meta_f = doc_dir / "meta.json"
    if not meta_f.exists():
        return None
    pdf = json.loads(meta_f.read_text()).get("source_pdf")
    if not pdf or not Path(pdf).exists():
        return None
    doc = fitz.open(pdf)
    try:
        if not 1 <= page <= len(doc):
            return None
        pg = doc[page - 1]
        r = pg.rect
        fx0, fy0, fx1, fy1 = _REGIONS[region]
        clip = fitz.Rect(r.x0 + fx0 * r.width, r.y0 + fy0 * r.height,
                         r.x0 + fx1 * r.width, r.y0 + fy1 * r.height)
        dpi = cfg.agent.view_page_zoom_dpi
        pix = pg.get_pixmap(dpi=dpi, clip=clip)
        cap = cfg.agent.view_page_max_px
        if max(pix.width, pix.height) > cap:
            dpi = max(72, int(dpi * cap / max(pix.width, pix.height)))
            pix = pg.get_pixmap(dpi=dpi, clip=clip)
        return pix.tobytes("png")
    finally:
        doc.close()

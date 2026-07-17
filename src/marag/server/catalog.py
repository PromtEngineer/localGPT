from __future__ import annotations

import json
from pathlib import Path

import duckdb

from ..config import Config


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


def list_docs(cfg: Config, dataset: str) -> list[dict]:
    ds_dir = cfg.path("processed", create=False) / dataset
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
    doc_dir = cfg.path("processed", create=False) / dataset / doc_id
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
    p = cfg.path("processed", create=False) / dataset / doc_id / "pages" / f"p{page:04d}.png"
    return p if p.exists() else None

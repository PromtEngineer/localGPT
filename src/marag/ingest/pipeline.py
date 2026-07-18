from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import duckdb
from rich.console import Console

from ..config import Config
from .chunk import chunk_doc
from .parse import parse_pdf

console = Console()


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def ingest_dataset(dataset: str, cfg: Config, limit: int | None = None, force: bool = False) -> dict:
    raw_dir = cfg.path("raw", create=False) / dataset
    manifest_path = raw_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"no manifest at {manifest_path} — dataset not downloaded yet")
    manifest = json.loads(manifest_path.read_text())
    if limit:
        manifest = manifest[:limit]

    out_root = cfg.path("processed") / dataset
    out_root.mkdir(parents=True, exist_ok=True)
    errors: list[dict] = []
    stats: list[dict] = []

    for entry in manifest:
        doc_id = entry["id"]
        pdf = raw_dir / entry["filename"]
        out_dir = out_root / doc_id
        meta_path = out_dir / "meta.json"
        if meta_path.exists() and not force:
            stats.append(json.loads(meta_path.read_text()))
            continue
        title = entry.get("title") or pdf.stem
        t0 = time.time()
        try:
            from .formats import (
                parse_audio_file,
                parse_data_file,
                parse_image_file,
                parse_office_doc,
                parser_for,
            )

            kind = parser_for(pdf.suffix)  # extension lookup, case-insensitive
            if kind == "pdf":
                meta = parse_pdf(pdf, out_dir, cfg)
            elif kind == "office":
                meta = parse_office_doc(pdf, out_dir, cfg)
            elif kind == "data":
                meta = parse_data_file(pdf, out_dir, cfg, doc_id)
            elif kind == "image":
                meta = parse_image_file(pdf, out_dir, cfg, doc_id)
            elif kind == "audio":
                meta = parse_audio_file(pdf, out_dir, cfg, doc_id)
            else:
                raise ValueError(f"unsupported format: {pdf.suffix.lower()}")
            (out_dir / "summary.md").unlink(missing_ok=True)  # stale after any re-parse
            chunks = chunk_doc(out_dir, doc_id, dataset, title, cfg)
            with open(out_dir / "chunks.jsonl", "w") as f:
                for c in chunks:
                    f.write(json.dumps(c) + "\n")
            meta.update(
                {
                    "doc_id": doc_id,
                    "dataset": dataset,
                    "title": title,
                    # manifest wins; else fall back to the parser's intrinsic type (image/audio)
                    "doc_type": entry.get("doc_type") or meta.get("doc_type", ""),
                    "n_chunks": len(chunks),
                    "parse_s": round(time.time() - t0, 1),
                }
            )
            meta_path.write_text(json.dumps(meta, indent=1))
            stats.append(meta)
            console.print(
                f"[green]✓[/] {doc_id} {title[:50]!r}: {meta['n_pages']}p "
                f"{len(chunks)}ch {meta['n_tables']}tbl {meta['parse_s']}s"
                + (" [yellow](visual-primary)[/]" if meta["visual_primary"] else "")
            )
        except Exception as e:
            errors.append({"doc_id": doc_id, "file": str(pdf), "error": str(e)})
            console.print(f"[red]✗ {doc_id}: {e}[/]")

    if errors:
        (out_root / "ingest_errors.json").write_text(json.dumps(errors, indent=1))

    _build_duckdb(dataset, cfg)
    _write_corpus_map(dataset, cfg, stats)
    return {"ok": len(stats), "failed": len(errors), "docs": stats}


def _build_duckdb(dataset: str, cfg: Config) -> Path:
    """Register every extracted table as a DuckDB view: t_<docid>_p<page>_<idx>.
    Built to a temp file then renamed, so a failed rebuild leaves the old db intact."""
    out_root = cfg.path("processed") / dataset
    db_path = out_root / "tables.duckdb"
    tmp_path = out_root / "tables.duckdb.tmp"
    tmp_path.unlink(missing_ok=True)
    try:
        con = duckdb.connect(str(tmp_path))
        catalog_rows: list[tuple] = []
        for doc_dir in sorted(out_root.iterdir()):
            # prefer the Docling extraction when it exists (higher cell recall, less garble)
            tdir = doc_dir / "tables_docling"
            if not (tdir / "catalog.json").exists():
                tdir = doc_dir / "tables"
            cat = tdir / "catalog.json"
            if not cat.exists():
                continue
            doc_id = doc_dir.name
            for t in json.loads(cat.read_text()):
                view = f"t_{_slug(doc_id)}_p{t['page']}_{t['table_index']}"
                pq = tdir / Path(t["parquet"]).name
                try:
                    con.execute(f"CREATE OR REPLACE VIEW {view} AS SELECT * FROM read_parquet('{pq}')")
                    catalog_rows.append(
                        (doc_id, t["page"], view, t["n_rows"], t["n_cols"], json.dumps(t["headers"]))
                    )
                except Exception:
                    continue
        con.execute(
            "CREATE OR REPLACE TABLE _catalog (doc_id VARCHAR, page INT, view_name VARCHAR,"
            " n_rows INT, n_cols INT, headers VARCHAR)"
        )
        if catalog_rows:
            con.executemany("INSERT INTO _catalog VALUES (?,?,?,?,?,?)", catalog_rows)
        con.close()
        os.replace(tmp_path, db_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return db_path


def _write_corpus_map(dataset: str, cfg: Config, stats: list[dict]) -> None:
    out_root = cfg.path("processed") / dataset
    lines = [f"# Corpus map — {dataset}", ""]
    for m in stats:
        lines.append(
            f"- **{m['doc_id']}** — {m['title']} ({m.get('doc_type','')}, {m['n_pages']}p, "
            f"{m['n_tables']} tables{', visual-primary' if m.get('visual_primary') else ''})"
        )
    (out_root / "corpus_map.md").write_text("\n".join(lines) + "\n")

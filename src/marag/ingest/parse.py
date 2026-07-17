from __future__ import annotations

import json
from pathlib import Path

import fitz  # pymupdf
import pymupdf4llm

from ..config import Config


def _safe_columns(cols: list) -> list[str]:
    out: list[str] = []
    seen: dict[str, int] = {}
    for i, c in enumerate(cols):
        name = str(c).strip() if c is not None and str(c).strip() else f"col{i}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        out.append(name)
    return out


def parse_pdf(pdf_path: Path, out_dir: Path, cfg: Config) -> dict:
    """Fast-path parse: page images + raw text + markdown + extracted tables.

    Returns doc metadata. Docling can be slotted in later as a quality upgrade for
    scanned/complex docs; this path keeps unattended ingestion throughput safe.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pages_dir = out_dir / "pages"
    tables_dir = out_dir / "tables"
    pages_dir.mkdir(exist_ok=True)
    tables_dir.mkdir(exist_ok=True)

    doc = fitz.open(pdf_path)
    n_pages = len(doc)

    page_texts: list[str] = []
    table_catalog: list[dict] = []
    for i, page in enumerate(doc):
        pno = i + 1
        text = page.get_text("text")
        page_texts.append(text)
        img_path = pages_dir / f"p{pno:04d}.png"
        if not img_path.exists():
            pix = page.get_pixmap(dpi=cfg.ingest.page_image_dpi)
            pix.save(img_path)
        # tables → parquet
        try:
            for j, tab in enumerate(page.find_tables().tables):
                df = tab.to_pandas()
                if df.empty or df.shape[0] < 2:
                    continue
                df.columns = _safe_columns(list(df.columns))
                pq = tables_dir / f"p{pno:04d}_t{j}.parquet"
                df.to_parquet(pq)
                table_catalog.append(
                    {
                        "page": pno,
                        "table_index": j,
                        "parquet": str(pq.relative_to(out_dir)),
                        "n_rows": int(df.shape[0]),
                        "n_cols": int(df.shape[1]),
                        "headers": list(df.columns)[:12],
                    }
                )
        except Exception:
            pass  # table detection is best-effort; text/markdown still carries the content

    # markdown, page-attributed
    try:
        md_pages = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
        md_texts = [p.get("text", "") for p in md_pages]
    except Exception:
        md_texts = page_texts  # fallback: raw text as markdown
    if len(md_texts) != n_pages:  # defensive: keep page alignment invariant
        md_texts = (md_texts + [""] * n_pages)[:n_pages]

    with open(out_dir / "pages.jsonl", "w") as f:
        for i, t in enumerate(page_texts):
            f.write(json.dumps({"page": i + 1, "text": t}) + "\n")
    with open(out_dir / "md_pages.jsonl", "w") as f:
        for i, t in enumerate(md_texts):
            f.write(json.dumps({"page": i + 1, "md": t}) + "\n")
    with open(tables_dir / "catalog.json", "w") as f:
        json.dump(table_catalog, f, indent=1)

    low_text_pages = sum(1 for t in page_texts if len(t.strip()) < 50)
    visual_primary = n_pages > 0 and (low_text_pages / n_pages) > 0.3

    meta = {
        "n_pages": n_pages,
        "n_tables": len(table_catalog),
        "visual_primary": visual_primary,
        "low_text_pages": low_text_pages,
        "source_pdf": str(pdf_path),
    }
    doc.close()
    return meta

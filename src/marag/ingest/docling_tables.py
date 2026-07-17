from __future__ import annotations

import json
import time
from pathlib import Path

from rich.console import Console

from ..config import Config
from .parse import _safe_columns

console = Console()


def _converter():
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions()
    opts.do_table_structure = True
    # Bench (runs/table_bench.json): FAST == ACCURATE on our gold pages; FAST is ~25% quicker.
    opts.table_structure_options.mode = TableFormerMode.FAST
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})


def rebuild_tables(dataset: str, cfg: Config, force: bool = False) -> dict:
    """Re-extract tables with Docling TableFormer into tables_docling/ (pymupdf layer untouched).

    Bench result that motivated this: find_tables put gold values in cells on 31% of gold
    table pages with 14% garble; Docling FAST hit 56% with 7% garble and detected tables on
    nearly every gold page.
    """
    raw_dir = cfg.path("raw", create=False) / dataset
    manifest = json.loads((raw_dir / "manifest.json").read_text())
    out_root = cfg.path("processed") / dataset
    conv = _converter()
    stats = {"docs": 0, "tables": 0, "failed": []}

    for entry in manifest:
        doc_id = entry["id"]
        out_dir = out_root / doc_id / "tables_docling"
        if (out_dir / "catalog.json").exists() and not force:
            stats["docs"] += 1
            stats["tables"] += len(json.loads((out_dir / "catalog.json").read_text()))
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        catalog: list[dict] = []
        try:
            res = conv.convert(str(raw_dir / entry["filename"]))
            doc = res.document
            per_page_idx: dict[int, int] = {}
            for t in doc.tables:
                try:
                    page = t.prov[0].page_no if t.prov else 0
                    try:
                        df = t.export_to_dataframe(doc=doc)
                    except TypeError:
                        df = t.export_to_dataframe()
                    if df.empty or df.shape[0] < 1:
                        continue
                    j = per_page_idx.get(page, 0)
                    per_page_idx[page] = j + 1
                    df.columns = _safe_columns(list(df.columns))
                    pq = out_dir / f"p{page:04d}_t{j}.parquet"
                    df.to_parquet(pq)
                    catalog.append(
                        {
                            "page": page,
                            "table_index": j,
                            "parquet": pq.name,
                            "n_rows": int(df.shape[0]),
                            "n_cols": int(df.shape[1]),
                            "headers": list(df.columns)[:12],
                        }
                    )
                except Exception:
                    continue
            (out_dir / "catalog.json").write_text(json.dumps(catalog, indent=1))
            stats["docs"] += 1
            stats["tables"] += len(catalog)
            console.print(
                f"[green]✓[/] {doc_id}: {len(catalog)} tables ({time.time()-t0:.0f}s)"
            )
        except Exception as e:
            stats["failed"].append({"doc_id": doc_id, "error": str(e)[:200]})
            console.print(f"[red]✗ {doc_id}: {e}[/]")
    return stats

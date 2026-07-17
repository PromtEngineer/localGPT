"""Table-parsing benchmark on gold table-hop pages from the QA benchmarks.

Compares: pymupdf text (current baseline) | pymupdf find_tables (current sql layer)
        | Docling FAST | Docling ACCURATE | LiteParse (spatial text).

Metrics per parser:
- value_recall: fraction of gold numbers (mined from hop evidence + answers) present in output
- adjacency: fraction of hops where the evidence label and a gold number share a line
- cell_recall (structured parsers only): gold numbers found inside table cells (sql-feedability)
- garble: fraction of extracted cells that are empty/punctuation-only
- seconds per page
"""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["research_papers", "financial_docs", "legal_docs"]

NUM_RE = re.compile(r"\d[\d,]*\.?\d+|\d{3,}")
LABEL_RE = re.compile(r"[A-Z][A-Za-z&-]+(?:\s+[A-Z&][A-Za-z&-]+)+")


def norm_num(s: str) -> str:
    return s.replace(",", "").rstrip(".")


def gold_numbers(*texts: str) -> set[str]:
    out = set()
    for t in texts:
        for m in NUM_RE.findall(t or ""):
            n = norm_num(m)
            if len(n.replace(".", "")) >= 3 and not re.fullmatch(r"(19|20)\d\d", n):
                out.add(n)
    return out


def labels_from(evidence: str) -> list[str]:
    return LABEL_RE.findall(evidence or "")[:3]


def collect_hops() -> list[dict]:
    hops, seen = [], set()
    for ds in DATASETS:
        bench = json.loads((ROOT / f"data/benchmarks/{ds}.json").read_text())
        manifest = {e["id"]: e for e in json.loads((ROOT / f"data/raw/{ds}/manifest.json").read_text())}
        for q in bench["questions"]:
            for h in q["hops"]:
                if h["modality"] != "table":
                    continue
                nums = gold_numbers(h.get("evidence", ""), q.get("answer", ""))
                if not nums:
                    continue
                for p in h["pages"]:
                    key = (ds, h["doc_id"], p)
                    entry = {
                        "dataset": ds,
                        "doc_id": h["doc_id"],
                        "page": p,
                        "pdf": str(ROOT / "data/raw" / ds / manifest[h["doc_id"]]["filename"]),
                        "numbers": sorted(nums),
                        "labels": labels_from(h.get("evidence", "")),
                        "qid": q["id"],
                    }
                    if key in seen:  # merge gold numbers for repeated pages
                        for e in hops:
                            if (e["dataset"], e["doc_id"], e["page"]) == key:
                                e["numbers"] = sorted(set(e["numbers"]) | nums)
                                e["labels"] = list(dict.fromkeys(e["labels"] + entry["labels"]))
                    else:
                        seen.add(key)
                        hops.append(entry)
    return hops


def score_text(output: str, numbers: list[str], labels: list[str]) -> dict:
    flat = output.replace(",", "")
    found = [n for n in numbers if n in flat]
    adj = False
    for line in output.split("\n"):
        lf = line.replace(",", "")
        if any(n in lf for n in numbers) and any(lb.lower() in line.lower() for lb in labels):
            adj = True
            break
    # spatial text may keep label and value on one visual row even without our labels list
    return {"found": len(found), "total": len(numbers), "adjacent": adj}


def score_cells(cells: list[str], numbers: list[str]) -> dict:
    flat_cells = [str(c).replace(",", "").strip() for c in cells]
    found = [n for n in numbers if any(n in c for c in flat_cells)]
    garble = sum(1 for c in flat_cells if not re.search(r"[A-Za-z0-9]", c)) / max(len(flat_cells), 1)
    return {"cell_found": len(found), "garble": round(garble, 3), "n_cells": len(flat_cells)}


# ---------------- parsers ----------------


def parse_pymupdf(hop: dict) -> tuple[str, list[str]]:
    pages = ROOT / "data/processed" / hop["dataset"] / hop["doc_id"] / "pages.jsonl"
    text = ""
    for line in open(pages):
        rec = json.loads(line)
        if rec["page"] == hop["page"]:
            text = rec["text"]
            break
    cells: list[str] = []
    cat_f = ROOT / "data/processed" / hop["dataset"] / hop["doc_id"] / "tables" / "catalog.json"
    if cat_f.exists():
        import pandas as pd

        for t in json.loads(cat_f.read_text()):
            if t["page"] == hop["page"]:
                df = pd.read_parquet(cat_f.parent / Path(t["parquet"]).name)
                cells += [str(x) for x in df.columns] + [str(x) for x in df.to_numpy().ravel()]
    return text, cells


class DoclingParser:
    def __init__(self, accurate: bool):
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
        from docling.document_converter import DocumentConverter, PdfFormatOption

        opts = PdfPipelineOptions()
        opts.do_table_structure = True
        opts.table_structure_options.mode = (
            TableFormerMode.ACCURATE if accurate else TableFormerMode.FAST
        )
        self.conv = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
        )

    def parse(self, hop: dict) -> tuple[str, list[str]]:
        res = self.conv.convert(hop["pdf"], page_range=(hop["page"], hop["page"]))
        doc = res.document
        md = doc.export_to_markdown()
        cells: list[str] = []
        for t in doc.tables:
            try:
                df = t.export_to_dataframe(doc=doc)
            except TypeError:
                df = t.export_to_dataframe()
            cells += [str(x) for x in df.columns] + [str(x) for x in df.to_numpy().ravel()]
        return md, cells


class LiteParser:
    def parse(self, hop: dict) -> tuple[str, list[str]]:
        from liteparse import LiteParse

        lp = LiteParse(target_pages=str(hop["page"]), quiet=True)
        res = lp.parse(hop["pdf"])
        text = "\n".join(p.text for p in res.pages)
        md = "\n".join(p.markdown for p in res.pages if p.markdown)
        return (text + "\n" + md), []  # no structured cells by design


def main() -> None:
    hops = collect_hops()
    print(f"benchmarking {len(hops)} unique gold table pages")
    parsers: dict[str, object] = {}
    results: dict[str, list[dict]] = defaultdict(list)

    for name in ["pymupdf", "docling_fast", "docling_accurate", "liteparse"]:
        t_total = 0.0
        for hop in hops:
            t0 = time.time()
            try:
                if name == "pymupdf":
                    text, cells = parse_pymupdf(hop)
                elif name.startswith("docling"):
                    if name not in parsers:
                        parsers[name] = DoclingParser(accurate=name.endswith("accurate"))
                    text, cells = parsers[name].parse(hop)
                else:
                    if name not in parsers:
                        parsers[name] = LiteParser()
                    text, cells = parsers[name].parse(hop)
                row = {"hop": f"{hop['doc_id']}:p{hop['page']}", **score_text(text, hop["numbers"], hop["labels"])}
                if cells:
                    row.update(score_cells(cells, hop["numbers"]))
                row["seconds"] = round(time.time() - t0, 2)
                results[name].append(row)
            except Exception as e:
                results[name].append({"hop": f"{hop['doc_id']}:p{hop['page']}", "error": str(e)[:120], "found": 0, "total": len(hop["numbers"]), "adjacent": False, "seconds": round(time.time() - t0, 2)})
            t_total += time.time() - t0
        print(f"  {name}: {t_total:.0f}s")

    print(f"\n{'parser':<18} {'value recall':>12} {'adjacency':>10} {'cell recall':>12} {'garble':>7} {'s/page':>7} {'errors':>7}")
    summary = {}
    for name, rows in results.items():
        tot = sum(r["total"] for r in rows)
        found = sum(r.get("found", 0) for r in rows)
        adj = sum(1 for r in rows if r.get("adjacent")) / len(rows)
        cell_rows = [r for r in rows if "cell_found" in r]
        cell_tot = sum(r["total"] for r in cell_rows)
        cell_found = sum(r["cell_found"] for r in cell_rows)
        garble = sum(r.get("garble", 0) for r in cell_rows) / max(len(cell_rows), 1)
        secs = sum(r["seconds"] for r in rows) / len(rows)
        errs = sum(1 for r in rows if "error" in r)
        cell_str = f"{cell_found}/{cell_tot}" if cell_rows else "—"
        print(f"{name:<18} {found}/{tot} ({found/max(tot,1):.0%}) {adj:>9.0%} {cell_str:>12} {garble:>7.0%} {secs:>7.1f} {errs:>7}")
        summary[name] = {"value_recall": round(found / max(tot, 1), 3), "adjacency": round(adj, 3),
                         "cell_recall": round(cell_found / max(cell_tot, 1), 3) if cell_rows else None,
                         "garble": round(garble, 3), "sec_per_page": round(secs, 2), "errors": errs,
                         "rows": rows}
    out = ROOT / "runs" / "table_bench.json"
    out.write_text(json.dumps({"n_pages": len(hops), "parsers": summary}, indent=1))
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()

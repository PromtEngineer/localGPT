"""Audit structured-table coverage against gold table hops in the QA benchmarks."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = sorted(p.stem for p in (ROOT / "data/benchmarks").glob("*.json"))


def catalog_for(ds: str, doc_id: str) -> list[dict]:
    for sub in ("tables_docling", "tables"):
        f = ROOT / "data/processed" / ds / doc_id / sub / "catalog.json"
        if f.exists():
            return json.loads(f.read_text())
    return []


def main() -> None:
    print(f"{'dataset':<18} {'sql coverage of gold table hops':>32}")
    for ds in DATASETS:
        bench = json.loads((ROOT / f"data/benchmarks/{ds}.json").read_text())
        hops = [
            (h["doc_id"], p)
            for q in bench["questions"]
            for h in q["hops"]
            if h["modality"] == "table"
            for p in h["pages"]
        ]
        hit = sum(
            1
            for doc_id, page in hops
            if any(abs(t["page"] - page) <= 1 for t in catalog_for(ds, doc_id))
        )
        n_tables = sum(len(catalog_for(ds, d.name)) for d in (ROOT / "data/processed" / ds).iterdir() if d.is_dir())
        print(f"{ds:<18} {hit}/{len(hops)} ({hit/len(hops):.0%})   total tables: {n_tables}")


if __name__ == "__main__":
    main()

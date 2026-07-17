"""Model-free unit tests: chunking, fusion, JSON extraction, eval matching."""

import json

from marag.eval.retrieval_eval import hop_hit
from marag.ingest.chunk import chunk_doc
from marag.llm import extract_json, strip_thinking
from marag.retrieve.hybrid import rrf_fuse


def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('prose before {"a": [1, 2]} after') == {"a": [1, 2]}
    assert extract_json('```json\n{"a": "b"}\n```') == {"a": "b"}
    assert extract_json('<think>reasoning {not json}</think>{"ok": true}') == {"ok": True}
    assert extract_json('{"s": "brace } in string"}') == {"s": "brace } in string"}


def test_strip_thinking():
    assert strip_thinking("<think>hmm</think>answer") == "answer"
    assert strip_thinking("no tags") == "no tags"


def test_rrf_fuse_prefers_consensus():
    fused = rrf_fuse([["a", "b", "c"], ["b", "a", "d"]], k=60)
    assert fused["a"] > fused["c"]
    assert fused["b"] > fused["d"]
    assert set(fused) == {"a", "b", "c", "d"}


def test_hop_hit_page_tolerance():
    hits = [{"doc_id": "d1", "page": 5}]
    assert hop_hit(hits, {"doc_id": "d1", "pages": [5]})
    assert hop_hit(hits, {"doc_id": "d1", "pages": [6]})  # ±1
    assert not hop_hit(hits, {"doc_id": "d1", "pages": [8]})
    assert not hop_hit(hits, {"doc_id": "d2", "pages": [5]})


def test_table_num_detection():
    from marag.agents.search_agent import _TABLE_NUM_RE

    found = _TABLE_NUM_RE.findall("Gross Bookings were $44,197M, up 18% from $37,575M")
    assert len(found) >= 3  # $44,197 / 18% / $37,575
    # prose without table figures shouldn't trip the sql nudge
    assert len(_TABLE_NUM_RE.findall("The model ranked second, ahead of three peers.")) == 0
    assert len(_TABLE_NUM_RE.findall("a rate of 6.9 points on PopQA")) == 0


def test_view_page_regions_cover_page():
    from marag.agents.tools import _REGIONS

    # every corner/half is a valid sub-rectangle inside the unit page
    for name, (x0, y0, x1, y1) in _REGIONS.items():
        assert 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1, name
    # corners are tall enough (~thirds) to clear the pixel cap at zoom dpi, not half-page
    assert _REGIONS["top-left"][3] < 0.5


def _cfg():
    from marag.config import Config, ModelsCfg

    return Config(models=ModelsCfg(orchestrator="x", utility="x", embedder="x", reranker="x"))


def test_parse_data_file_csv(tmp_path):
    import duckdb
    import pandas as pd

    from marag.ingest.formats import parse_data_file

    csv = tmp_path / "sales.csv"
    pd.DataFrame(
        {"region": ["EMEA", "APAC", "EMEA", "AMER"], "revenue": [100.5, 200.0, 50.25, 400.0]}
    ).to_csv(csv, index=False)
    out = tmp_path / "out"
    meta = parse_data_file(csv, out, _cfg(), "mix001")
    assert meta["n_tables"] == 1 and meta["n_pages"] == 1 and meta["data_file"]
    # profile page names the SQL view and carries the schema
    page = json.loads((out / "pages.jsonl").read_text().splitlines()[0])
    assert "t_mix001_p1_0" in page["text"] and "revenue" in page["text"] and "4 rows" in page["text"].replace(",", "")
    # the parquet is the FULL data and is queryable
    cat = json.loads((out / "tables_docling" / "catalog.json").read_text())
    df = duckdb.sql(f"SELECT sum(revenue) s FROM read_parquet('{out/'tables_docling'/cat[0]['parquet']}')").df()
    assert abs(df["s"][0] - 750.75) < 1e-6


def test_parse_data_file_xlsx_multisheet(tmp_path):
    import pandas as pd

    from marag.ingest.formats import parse_data_file

    xlsx = tmp_path / "book.xlsx"
    with pd.ExcelWriter(xlsx) as w:
        pd.DataFrame({"a": [1, 2]}).to_excel(w, sheet_name="first", index=False)
        pd.DataFrame({"b": ["x", "y", "z"]}).to_excel(w, sheet_name="second", index=False)
    out = tmp_path / "out"
    meta = parse_data_file(xlsx, out, _cfg(), "mix002")
    assert meta["n_tables"] == 2 and meta["n_pages"] == 2
    pages = [json.loads(l) for l in (out / "pages.jsonl").read_text().splitlines()]
    assert "t_mix002_p1_0" in pages[0]["text"] and "t_mix002_p2_1" in pages[1]["text"]


def test_multi_index_toolbox(tmp_path):
    """Two sources in one ToolBox: doc resolution + cross-source sql in one flat namespace."""
    import pandas as pd

    from marag.config import Config, ModelsCfg, PathsCfg
    from marag.ingest.formats import parse_data_file
    from marag.ingest.pipeline import _build_duckdb

    # absolute paths: Config.path() does root / paths.X, and Path/abs == abs
    cfg = Config(
        models=ModelsCfg(orchestrator="x", utility="x", embedder="x", reranker="x"),
        paths=PathsCfg(processed=str(tmp_path / "processed"), index=str(tmp_path / "index")),
    )

    def make(ds, doc_id, df):
        csv = tmp_path / f"{doc_id}.csv"
        df.to_csv(csv, index=False)
        out = cfg.path("processed") / ds / doc_id
        meta = parse_data_file(csv, out, cfg, doc_id)
        (out / "meta.json").write_text(json.dumps({**meta, "doc_id": doc_id, "title": doc_id}))

    make("src_a", "aaa001", pd.DataFrame({"region": ["E", "A"], "rev": [10, 20]}))
    make("src_b", "bbb001", pd.DataFrame({"item": ["x", "y", "z"], "qty": [1, 2, 3]}))
    for ds in ("src_a", "src_b"):
        _build_duckdb(ds, cfg)

    from marag.agents.tools import ToolBox

    tb = ToolBox(cfg, ["src_a", "src_b"], retriever=None)
    assert tb.multi and tb.doc2ds == {"aaa001": "src_a", "bbb001": "src_b"}
    assert tb.has_tables()
    # one query spanning views from BOTH sources — globally-unique view names in one namespace
    out = tb.sql("SELECT (SELECT sum(rev) FROM t_aaa001_p1_0) a, (SELECT sum(qty) FROM t_bbb001_p1_0) b")
    assert "30" in out and "6" in out
    # read_doc resolves each doc to its owning source
    assert "aaa001" in tb.read_doc("aaa001", 1, 1)


def test_pseudo_page_split():
    from marag.ingest.formats import PSEUDO_PAGE_CHARS

    md = "x" * (PSEUDO_PAGE_CHARS * 2 + 10)
    pages = [md[i : i + PSEUDO_PAGE_CHARS] for i in range(0, len(md), PSEUDO_PAGE_CHARS)]
    assert len(pages) == 3 and "".join(pages) == md


def test_chunk_doc(tmp_path):
    md_pages = [
        {"page": 1, "md": "# Introduction\n\nSome intro text here.\n\nMore paragraph content."},
        {"page": 2, "md": "## Results\n\n| model | score |\n|---|---|\n| a | 1 |\n\nDiscussion text."},
    ]
    with open(tmp_path / "md_pages.jsonl", "w") as f:
        for p in md_pages:
            f.write(json.dumps(p) + "\n")

    from marag.config import Config, ModelsCfg

    cfg = Config(models=ModelsCfg(orchestrator="x", utility="x", embedder="x", reranker="x"))
    chunks = chunk_doc(tmp_path, "doc1", "test_ds", "Test Doc", cfg)
    assert chunks, "no chunks produced"
    assert all(c["doc_id"] == "doc1" for c in chunks)
    pages = {c["page"] for c in chunks}
    assert 1 in pages and 2 in pages
    results_chunks = [c for c in chunks if c["section"] == "Results"]
    assert results_chunks and "| model | score |" in results_chunks[0]["raw_text"]
    assert chunks[0]["text"].startswith("Test Doc ·")  # contextual header

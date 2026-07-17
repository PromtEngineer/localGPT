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

"""Model-free tests for standalone image + audio ingestion.

Every model touchpoint (VLM caption, whisper transcript) is isolated behind the
MARAG_SKIP_MODELS guard or an injected caption_fn/transcript, so nothing here loads a
model, calls a server, or transcribes/captions for real. The queued GPU run does that.
"""

import json

from PIL import Image


def _cfg(tmp_path):
    from marag.config import Config, ModelsCfg, PathsCfg

    # absolute paths: Config.path() does root / paths.X, and Path / abs == abs
    return Config(
        models=ModelsCfg(orchestrator="x", utility="x", embedder="x", reranker="x"),
        paths=PathsCfg(
            processed=str(tmp_path / "processed"),
            index=str(tmp_path / "index"),
            raw=str(tmp_path / "raw"),
            runs=str(tmp_path / "runs"),
        ),
    )


def _make_png(path, size=(3000, 2000), color=(200, 30, 30)):
    Image.new("RGB", size, color).save(path)
    return path


# ---------- extension dispatch (case-insensitive) ----------


def test_parser_for_dispatch_and_uppercase():
    from marag.ingest.formats import parser_for

    assert parser_for(".pdf") == "pdf"
    assert parser_for(".docx") == "office"
    assert parser_for(".csv") == "data"
    # uppercase image/audio extensions must route the same as lowercase
    assert parser_for(".PNG") == "image"
    assert parser_for(".png") == "image"
    assert parser_for(".JPG") == "image"
    assert parser_for(".MP3") == "audio"
    assert parser_for(".wav") == "audio"
    assert parser_for(".xyz") is None


# ---------- image ingestion ----------


def test_parse_image_file_skip_models(tmp_path, monkeypatch):
    monkeypatch.setenv("MARAG_SKIP_MODELS", "1")  # no VLM, placeholder caption
    from marag.ingest.chunk import chunk_doc
    from marag.ingest.formats import IMAGE_MAX_EDGE, parse_image_file

    src = _make_png(tmp_path / "photo.png", size=(3000, 2000))
    out = tmp_path / "out"
    meta = parse_image_file(src, out, _cfg(tmp_path), "img001")

    # a normalized page PNG exists and the longest edge was capped
    png = out / "pages" / "p0001.png"
    assert png.exists()
    with Image.open(png) as im:
        assert max(im.size) <= IMAGE_MAX_EDGE

    # meta is shaped like the other parsers + image-specific keys
    assert meta["doc_type"] == "image"
    assert meta["n_pages"] == 1 and meta["n_tables"] == 0 and meta["visual_primary"] is True
    assert meta["source_pdf"] is None
    assert meta["source_image"] == str(src.resolve())  # absolute path

    # the caption is the page text (placeholder under skip-models)
    page = json.loads((out / "pages.jsonl").read_text().splitlines()[0])
    assert page["page"] == 1 and "caption pending" in page["text"]

    # the caption flows into exactly one chunk (this is how dense/BM25 index it)
    (out / "meta.json").write_text(json.dumps({**meta, "doc_id": "img001", "title": "photo"}))
    chunks = chunk_doc(out, "img001", "ds1", "photo", _cfg(tmp_path))
    assert len(chunks) == 1 and "caption pending" in chunks[0]["raw_text"]


def test_parse_image_file_injected_caption(tmp_path):
    # no env, no model: an injected caption_fn proves the model call is fully isolated
    from marag.ingest.formats import parse_image_file

    src = _make_png(tmp_path / "diagram.webp", size=(800, 600))
    out = tmp_path / "out"
    meta = parse_image_file(
        src, out, _cfg(tmp_path), "img002", caption_fn=lambda p, c: "a red rectangle on white"
    )
    assert meta["doc_type"] == "image"
    page = json.loads((out / "pages.jsonl").read_text().splitlines()[0])
    assert page["text"] == "a red rectangle on white"


# ---------- audio ingestion ----------


def _fake_transcript(n=30, chars=100):
    # each segment ~ (7-char [mm:ss] marker + space + `chars`) so page splitting is predictable
    return [{"start": i * 10.0, "end": i * 10.0 + 9, "text": "x" * chars} for i in range(n)]


def test_parse_audio_file_injected_transcript(tmp_path):
    from marag.ingest.formats import PSEUDO_PAGE_CHARS, parse_audio_file

    out = tmp_path / "out"
    meta = parse_audio_file(
        tmp_path / "talk.mp3", out, _cfg(tmp_path), "aud001", transcript=_fake_transcript()
    )

    # doc_type + splitting: 30 uniform segments overflow the char cap into 2 pseudo-pages
    assert meta["doc_type"] == "audio"
    assert meta["n_pages"] == 2 and meta["n_tables"] == 0 and meta["visual_primary"] is False
    assert meta["source_pdf"] is None
    assert meta["source_audio"] == str((tmp_path / "talk.mp3").resolve())

    pages = [json.loads(line) for line in (out / "pages.jsonl").read_text().splitlines()]
    assert len(pages) == 2
    for p in pages:
        assert len(p["text"]) <= PSEUDO_PAGE_CHARS  # split respects the cap
    # [mm:ss] markers preserved at segment boundaries
    assert pages[0]["text"].startswith("[00:00] ")
    assert "[04:50]" in pages[1]["text"]  # last segment start = 290s

    # audio has no visual channel: no page images are written
    assert not (out / "pages").exists()


def test_parse_audio_file_skip_models(tmp_path, monkeypatch):
    # skip-guard path returns a placeholder segment without importing faster-whisper
    monkeypatch.setenv("MARAG_SKIP_MODELS", "1")
    import sys

    from marag.ingest.formats import parse_audio_file

    meta = parse_audio_file(tmp_path / "voice.wav", tmp_path / "out", _cfg(tmp_path), "aud002")
    assert meta["doc_type"] == "audio" and meta["n_pages"] == 1
    assert "faster_whisper" not in sys.modules  # never imported under the guard
    page = json.loads((tmp_path / "out" / "pages.jsonl").read_text().splitlines()[0])
    assert "transcript pending" in page["text"]


# ---------- _build_duckdb tolerates zero-table docs ----------


def test_build_duckdb_tolerates_media_docs(tmp_path, monkeypatch):
    monkeypatch.setenv("MARAG_SKIP_MODELS", "1")
    import duckdb

    from marag.ingest.formats import parse_audio_file, parse_image_file
    from marag.ingest.pipeline import _build_duckdb

    cfg = _cfg(tmp_path)
    # one image doc + one audio doc, both with n_tables == 0 and no tables catalog
    img_out = cfg.path("processed") / "ds1" / "img001"
    imeta = parse_image_file(_make_png(tmp_path / "p.png", (400, 300)), img_out, cfg, "img001")
    (img_out / "meta.json").write_text(json.dumps({**imeta, "doc_id": "img001", "title": "p"}))

    aud_out = cfg.path("processed") / "ds1" / "aud001"
    ameta = parse_audio_file(tmp_path / "a.mp3", aud_out, cfg, "aud001", transcript=_fake_transcript(3))
    (aud_out / "meta.json").write_text(json.dumps({**ameta, "doc_id": "aud001", "title": "a"}))

    db = _build_duckdb("ds1", cfg)  # must not crash on zero-table docs
    con = duckdb.connect(str(db), read_only=True)
    assert con.execute("SELECT count(*) FROM _catalog").fetchone()[0] == 0  # no rows for media
    con.close()


# ---------- view_page region fallback (no source PDF) ----------


def test_view_page_region_fallback_crops_png(tmp_path, monkeypatch):
    monkeypatch.setenv("MARAG_SKIP_MODELS", "1")
    from marag.agents.tools import ToolBox
    from marag.ingest.formats import parse_audio_file, parse_image_file

    cfg = _cfg(tmp_path)
    img_out = cfg.path("processed") / "ds1" / "img001"
    imeta = parse_image_file(_make_png(tmp_path / "p.png", (1600, 1200)), img_out, cfg, "img001")
    (img_out / "meta.json").write_text(json.dumps({**imeta, "doc_id": "img001", "title": "p"}))

    aud_out = cfg.path("processed") / "ds1" / "aud001"
    ameta = parse_audio_file(tmp_path / "a.mp3", aud_out, cfg, "aud001", transcript=_fake_transcript(2))
    (aud_out / "meta.json").write_text(json.dumps({**ameta, "doc_id": "aud001", "title": "a"}))

    tb = ToolBox(cfg, "ds1", retriever=None)

    # image doc has no source_pdf: a region must be a PIL crop of the stored PNG (bytes, not
    # the "source PDF unavailable" string that the fitz branch would return)
    region = tb._render("img001", 1, "top-left")
    assert isinstance(region, bytes)
    with Image.open(__import__("io").BytesIO(region)) as im:
        assert im.format == "PNG"
        rw, rh = im.size
    full = tb._render("img001", 1, "full")
    assert isinstance(full, bytes)
    with Image.open(__import__("io").BytesIO(full)) as im:
        fw, fh = im.size
    assert rw < fw and rh < fh  # the crop is strictly smaller than the full page

    # audio doc has no page image at all: clear message, no model touched
    msg = tb._render("aud001", 1, "full")
    assert isinstance(msg, str) and "no page image" in msg
    vp = tb.view_page("aud001", 1, "what is said?")
    assert "no page image" in vp
    tb.close()

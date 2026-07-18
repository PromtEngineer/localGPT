"""Model-free tests for the late-interaction TEXT multivector channel: MaxSim math,
path naming, sidecar round-trip, a fake-encoder search, and hybrid channel-gating."""

import numpy as np

from marag.config import Config, ModelsCfg, PathsCfg, RetrievalCfg
from marag.index.text_multivector import (
    TextMultiVectorIndex,
    _load_sidecar,
    _maxsim,
    _write_index,
)


def _cfg(tmp_path, **kw):
    return Config(
        models=ModelsCfg(orchestrator="x", utility="x", embedder="x", reranker="x", **kw.pop("models", {})),
        paths=PathsCfg(index=str(tmp_path / "index"), processed=str(tmp_path / "processed")),
        **kw,
    )


# ---------- MaxSim scoring math ----------


def test_maxsim_known_value():
    # one query token [1,0,0,0]; doc tokens' best match is the first one -> score 1.0
    q = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    doc = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    assert _maxsim(q, doc) == 1.0


def test_maxsim_sums_over_query_tokens():
    # two query tokens, each perfectly matched by a distinct doc token -> 1 + 1 = 2
    q = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    doc = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    assert _maxsim(q, doc) == 2.0


def test_maxsim_takes_max_per_query_token_not_sum():
    # query token should count its BEST doc token once, not accumulate across doc tokens
    q = np.array([[1.0, 0.0]], dtype=np.float32)
    doc = np.array([[0.9, 0.0], [0.8, 0.0], [0.7, 0.0]], dtype=np.float32)
    assert abs(_maxsim(q, doc) - 0.9) < 1e-6


def test_maxsim_ordering():
    q = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    strong = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    weak = np.array([[0.5, 0.0, 0.0, 0.0]], dtype=np.float32)
    none = np.array([[0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    assert _maxsim(q, strong) > _maxsim(q, weak) > _maxsim(q, none)


def test_maxsim_handles_fp16_and_empty():
    q = np.array([[1.0, 0.0]], dtype=np.float16)
    doc = np.array([[1.0, 0.0]], dtype=np.float16)
    assert abs(_maxsim(q, doc) - 1.0) < 1e-3
    assert _maxsim(q, np.zeros((0, 2), dtype=np.float16)) == 0.0
    assert _maxsim(np.zeros((0, 2), dtype=np.float32), doc) == 0.0


# ---------- path naming ----------


def test_index_paths(tmp_path):
    idx = TextMultiVectorIndex(_cfg(tmp_path))
    assert idx._npz_path("acme").name == "textmv_acme.npz"
    assert idx._sidecar_path("acme").name == "textmv_acme.json"
    # both live under the configured index dir, distinct from the visual index prefix
    assert idx._npz_path("acme").parent == idx._sidecar_path("acme").parent
    assert "visual_" not in idx._npz_path("acme").name


# ---------- sidecar / storage round-trip (no model) ----------


def test_write_index_sidecar_round_trip(tmp_path):
    idx = TextMultiVectorIndex(_cfg(tmp_path))
    idx.cfg.path("index")  # ensure dir exists
    vectors = {
        "doc1_c0000": np.ones((3, 4), dtype=np.float16),
        "doc1_c0001": np.zeros((2, 4), dtype=np.float16),
    }
    meta = {
        "doc1_c0000": {"doc_id": "doc1", "page": 5},
        "doc1_c0001": {"doc_id": "doc1", "page": 7},
    }
    _write_index(idx._npz_path("ds"), idx._sidecar_path("ds"), vectors, meta)
    assert idx.exists("ds")

    loaded_vecs, loaded_meta = idx._load_index("ds")
    assert set(loaded_vecs) == set(vectors)
    assert loaded_vecs["doc1_c0000"].shape == (3, 4)
    assert loaded_meta == meta  # chunk_id -> {doc_id, page} survives the round-trip
    assert _load_sidecar(idx._sidecar_path("ds"))["doc1_c0001"]["page"] == 7


def test_load_sidecar_missing_returns_empty(tmp_path):
    assert _load_sidecar(tmp_path / "nope.json") == {}


# ---------- search over a hand-built index via a fake encoder (no model) ----------


class _FakeEncoder:
    """Returns a fixed single-token query vector so search exercises real MaxSim + ranking."""

    backend = "fake"

    def encode_query(self, query, max_tokens):
        return np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)


def test_search_ranks_and_returns_locations(tmp_path):
    idx = TextMultiVectorIndex(_cfg(tmp_path))
    idx.cfg.path("index")
    vectors = {
        "d_c0000": np.array([[1.0, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float16),  # best (1.0)
        "d_c0001": np.array([[0.5, 0, 0, 0]], dtype=np.float16),                # middle (0.5)
        "d_c0002": np.array([[0, 0, 1, 0]], dtype=np.float16),                  # worst (0.0)
    }
    meta = {
        "d_c0000": {"doc_id": "d", "page": 1},
        "d_c0001": {"doc_id": "d", "page": 2},
        "d_c0002": {"doc_id": "d", "page": 3},
    }
    _write_index(idx._npz_path("ds"), idx._sidecar_path("ds"), vectors, meta)

    idx._encoder = _FakeEncoder()  # bypass _ensure_model / real model load
    idx.model_id = "fake"
    hits = idx.search("q", "ds", k=2)
    assert [h["chunk_id"] for h in hits] == ["d_c0000", "d_c0001"]  # ranked by MaxSim
    assert hits[0] == {"chunk_id": "d_c0000", "doc_id": "d", "page": 1, "score": hits[0]["score"]}
    assert hits[0]["score"] > hits[1]["score"]


# ---------- hybrid channel-gating: OFF by default keeps existing evals untouched ----------


def _retriever(tmp_path, **retrieval_kw):
    from marag.retrieve.hybrid import Retriever

    models = {}
    if retrieval_kw.pop("with_model", False):
        models = {"text_mv_retriever": "lightonai/GTE-ModernColBERT-v1"}
    cfg = _cfg(tmp_path, models=models, retrieval=RetrievalCfg(**retrieval_kw))
    return Retriever(cfg)


def test_text_mv_gated_off_by_default(tmp_path):
    r = _retriever(tmp_path)  # default config: flag False, no model
    # even if a caller explicitly requests the channel, it stays inert until configured
    assert r._text_mv_active(("dense", "fts", "text_mv")) is False
    assert r._text_mv_active(("dense", "fts")) is False


def test_text_mv_requires_flag_and_model(tmp_path):
    # flag on but no model -> off
    assert _retriever(tmp_path, text_multivector=True)._text_mv_active(("text_mv",)) is False
    # model set but flag off -> off
    assert _retriever(tmp_path, with_model=True)._text_mv_active(("text_mv",)) is False
    # both present but channel not requested -> off (existing dense/fts evals unaffected)
    r = _retriever(tmp_path, text_multivector=True, with_model=True)
    assert r._text_mv_active(("dense", "fts")) is False
    # all three conditions met -> on
    assert r._text_mv_active(("dense", "fts", "text_mv")) is True


def test_text_mv_index_not_constructed_when_off(tmp_path, monkeypatch):
    """The off path must never even instantiate the index (so no torch/model import)."""
    import marag.index.text_multivector as tmv

    def _boom(*a, **k):
        raise AssertionError("TextMultiVectorIndex must not be built when the channel is off")

    monkeypatch.setattr(tmv, "TextMultiVectorIndex", _boom)
    r = _retriever(tmp_path)  # off
    assert r._text_mv_active(("dense", "fts", "text_mv")) is False
    assert r._text_mv is None  # lazy handle untouched

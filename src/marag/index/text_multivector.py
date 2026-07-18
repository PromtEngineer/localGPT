from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import Config
from .embedder import resolve_device

# Token-length caps bound MPS attention memory. Text late-interaction models keep one vector
# per token, so an unbounded long chunk can demand a pathological buffer (the MPS "Invalid
# buffer size" lesson that already bit the dense embedder — see embedder.py max_seq_length).
_DOC_LEN = 512
_QUERY_LEN = 64

# Loader tries these (model_id, backend) in order; first that loads wins. pylate exposes the
# ColBERT multivector head natively; the sentence-transformers path is a best-effort fallback
# that reads token embeddings directly (no guarantee the projection head is applied).
_CANDIDATES = [
    ("lightonai/GTE-ModernColBERT-v1", "pylate"),
    ("lightonai/GTE-ModernColBERT-v1", "sentence_transformers"),
]


# ---------- pure scoring / storage (unit-testable without a model) ----------


def _maxsim(q_mat: np.ndarray, doc_mat: np.ndarray) -> float:
    """Late-interaction MaxSim: for each query token take its max similarity over all doc
    tokens, then sum across query tokens. q_mat is (nq, dim), doc_mat is (nd, dim)."""
    if q_mat is None or doc_mat is None or q_mat.size == 0 or doc_mat.size == 0:
        return 0.0
    sim = q_mat.astype(np.float32) @ doc_mat.astype(np.float32).T  # (nq, nd)
    return float(sim.max(axis=1).sum())


def _write_index(
    npz_path: Path, sidecar_path: Path, vectors: dict[str, np.ndarray], meta: dict[str, dict]
) -> None:
    """Persist the per-chunk multivectors (fp16 .npz) plus the chunk_id -> {doc_id, page}
    sidecar. Factored out so build() storage can be exercised without loading a model."""
    np.savez_compressed(npz_path, **vectors)
    Path(sidecar_path).write_text(json.dumps(meta))


def _load_sidecar(sidecar_path: Path) -> dict[str, dict]:
    p = Path(sidecar_path)
    return json.loads(p.read_text()) if p.exists() else {}


def _local_snapshot(model_id: str) -> str | None:
    """Prefer the local HF cache; keeps cached models loadable when the hub is down."""
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(model_id, local_files_only=True)
    except Exception:
        return None


# ---------- encoders: one thin wrapper per backend, both emit numpy multivectors ----------


class _PylateEncoder:
    backend = "pylate"

    def __init__(self, model):
        self.model = model

    def encode_docs(self, texts: list[str], max_tokens: int) -> list[np.ndarray]:
        embs = self.model.encode(
            texts, is_query=False, convert_to_numpy=True, show_progress_bar=False
        )
        return [np.asarray(e, dtype=np.float16)[:max_tokens] for e in embs]

    def encode_query(self, query: str, max_tokens: int) -> np.ndarray:
        emb = self.model.encode(
            [query], is_query=True, convert_to_numpy=True, show_progress_bar=False
        )[0]
        return np.asarray(emb, dtype=np.float32)[:max_tokens]


class _SentenceTransformersEncoder:
    """Fallback: pull per-token embeddings from a plain SentenceTransformer. Approximate —
    it may miss a ColBERT projection head the pylate loader would apply."""

    backend = "sentence_transformers"

    def __init__(self, model):
        self.model = model

    def _mv(self, text: str, max_tokens: int) -> np.ndarray:
        toks = self.model.encode(
            [text], output_value="token_embeddings", convert_to_numpy=False,
            show_progress_bar=False,
        )[0]
        arr = toks.cpu().float().numpy() if hasattr(toks, "cpu") else np.asarray(toks, np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)  # ColBERT MaxSim assumes unit tokens
        arr = arr / np.clip(norms, 1e-9, None)
        return arr[:max_tokens]

    def encode_docs(self, texts: list[str], max_tokens: int) -> list[np.ndarray]:
        return [self._mv(t, max_tokens).astype(np.float16) for t in texts]

    def encode_query(self, query: str, max_tokens: int) -> np.ndarray:
        return self._mv(query, max_tokens).astype(np.float32)


def _load_pylate(model_id: str, device: str):
    from pylate import models  # raises ImportError if pylate is not installed

    path = _local_snapshot(model_id) or model_id
    model = models.ColBERT(
        model_name_or_path=path,
        device=device,
        document_length=_DOC_LEN,
        query_length=_QUERY_LEN,
    )
    return model_id, _PylateEncoder(model)


def _load_sentence_transformers(model_id: str, device: str):
    from sentence_transformers import SentenceTransformer

    path = _local_snapshot(model_id) or model_id
    model = SentenceTransformer(path, device=device)
    model.max_seq_length = min(getattr(model, "max_seq_length", None) or _DOC_LEN, _DOC_LEN)
    return model_id, _SentenceTransformersEncoder(model)


def _load_model(preferred: str | None, device: str):
    candidates = list(_CANDIDATES)
    if preferred and preferred not in {c[0] for c in candidates}:
        # honour an arbitrary configured model id: try it on both backends, first
        candidates = [(preferred, "pylate"), (preferred, "sentence_transformers")] + candidates
    elif preferred:
        candidates.sort(key=lambda c: c[0] != preferred)  # preferred model id first (stable)
    errors = []
    for model_id, backend in candidates:
        try:
            if backend == "pylate":
                return _load_pylate(model_id, device)
            return _load_sentence_transformers(model_id, device)
        except Exception as e:  # unavailable lib/model -> record and try the next candidate
            errors.append(f"{model_id} via {backend}: {e}")
    raise RuntimeError("no text multivector retriever loadable: " + " | ".join(errors))


class TextMultiVectorIndex:
    """Late-interaction TEXT-chunk retrieval — the text-side twin of VisualIndex.

    Each chunk's raw_text is encoded to a token multivector (fp16) and stored per dataset as
    .npz keyed by chunk id, with a chunk_id -> {doc_id, page} sidecar. Query-time MaxSim is
    brute-force numpy — exact, and fine at single-node corpus sizes. Chunk ids are the store's
    ids, so hits fuse cleanly with the dense/fts channels in one rrf_fuse call.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._model = None
        self._encoder = None
        self.model_id: str | None = None
        self._cache: dict[str, tuple[dict[str, np.ndarray], dict[str, dict]]] = {}

    def _ensure_model(self) -> None:
        if self._encoder is None:
            device = resolve_device(self.cfg.embedding.device)
            self.model_id, self._encoder = _load_model(
                self.cfg.models.text_mv_retriever, device
            )

    def _npz_path(self, dataset: str) -> Path:
        return self.cfg.path("index") / f"textmv_{dataset}.npz"

    def _sidecar_path(self, dataset: str) -> Path:
        return self.cfg.path("index") / f"textmv_{dataset}.json"

    def exists(self, dataset: str) -> bool:
        return self._npz_path(dataset).exists()

    # ---------- build ----------

    def build(
        self, dataset: str, batch_size: int = 8, max_tokens: int = _DOC_LEN,
        max_chunks_per_doc: int = 0,
    ) -> dict:
        self._ensure_model()
        processed = self.cfg.path("processed", create=False) / dataset
        vectors: dict[str, np.ndarray] = {}
        meta: dict[str, dict] = {}
        n = 0
        for doc_dir in sorted(processed.iterdir()):
            f = doc_dir / "chunks.jsonl"
            if not f.is_file():
                continue
            chunks = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
            if max_chunks_per_doc:
                chunks = chunks[:max_chunks_per_doc]
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]
                texts = [c.get("raw_text") or c.get("text") or "" for c in batch]
                embs = self._encoder.encode_docs(texts, max_tokens)
                for c, e in zip(batch, embs):
                    cid = c["id"]
                    vectors[cid] = np.ascontiguousarray(e, dtype=np.float16)
                    meta[cid] = {"doc_id": c["doc_id"], "page": int(c["page"])}
                    n += 1
        _write_index(self._npz_path(dataset), self._sidecar_path(dataset), vectors, meta)
        return {"chunks": n, "model": self.model_id, "path": str(self._npz_path(dataset))}

    # ---------- search ----------

    def _load_index(self, dataset: str) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        npz = np.load(self._npz_path(dataset))
        vectors = {k: npz[k] for k in npz.files}
        return vectors, _load_sidecar(self._sidecar_path(dataset))

    def search(self, query: str, dataset: str, k: int = 10, max_tokens: int = _QUERY_LEN) -> list[dict]:
        self._ensure_model()
        if dataset not in self._cache:
            self._cache[dataset] = self._load_index(dataset)
        vectors, meta = self._cache[dataset]

        q = self._encoder.encode_query(query, max_tokens)
        scores: list[tuple[float, str]] = [
            (_maxsim(q, doc_emb), cid) for cid, doc_emb in vectors.items()
        ]
        scores.sort(reverse=True)
        out = []
        for score, cid in scores[:k]:
            loc = meta.get(cid, {})
            out.append(
                {
                    "chunk_id": cid,
                    "doc_id": loc.get("doc_id"),
                    "page": loc.get("page"),
                    "score": score,
                }
            )
        return out

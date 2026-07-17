from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import Config
from .embedder import resolve_device

# Loader tries these (model_id, engine class names) in order; first available wins.
_CANDIDATES = [
    ("ModernVBERT/colmodernvbert", "ColModernVBert", "ColModernVBertProcessor"),
    ("vidore/colqwen2.5-v0.2", "ColQwen2_5", "ColQwen2_5_Processor"),
    ("vidore/colpali-v1.3", "ColPali", "ColPaliProcessor"),
]


def _local_snapshot(model_id: str) -> str | None:
    """Prefer the local HF cache; keeps cached models loadable when the hub is down."""
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(model_id, local_files_only=True)
    except Exception:
        return None


def _load_model(preferred: str | None, device: str):
    import colpali_engine.models as cm
    import torch

    candidates = list(_CANDIDATES)
    if preferred:
        candidates.sort(key=lambda c: c[0] != preferred)  # preferred first
    errors = []
    dtype = torch.float32 if device == "cpu" else torch.float16
    for model_id, cls_name, proc_name in candidates:
        cls = getattr(cm, cls_name, None)
        proc_cls = getattr(cm, proc_name, None)
        if cls is None or proc_cls is None:
            errors.append(f"{cls_name}: not in colpali_engine")
            continue
        try:
            path = _local_snapshot(model_id) or model_id
            model = cls.from_pretrained(path, torch_dtype=dtype).to(device).eval()
            processor = proc_cls.from_pretrained(path)
            return model_id, model, processor
        except Exception as e:
            errors.append(f"{model_id}: {e}")
    raise RuntimeError("no visual retriever loadable: " + " | ".join(errors))


class VisualIndex:
    """Late-interaction page-image retrieval.

    Page embeddings are stored per dataset as .npz (fp16 multivectors); query-time
    MaxSim is brute-force numpy — exact, and fast at single-node corpus sizes
    (hundreds to low thousands of pages). Swap for LanceDB/Qdrant multivector when
    corpora outgrow memory.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = resolve_device(cfg.embedding.device)
        self._model = None
        self._processor = None
        self.model_id: str | None = None

    def _ensure_model(self) -> None:
        if self._model is None:
            self.model_id, self._model, self._processor = _load_model(
                self.cfg.models.visual_retriever, self.device
            )

    def _npz_path(self, dataset: str) -> Path:
        return self.cfg.path("index") / f"visual_{dataset}.npz"

    # ---------- build ----------

    def build(self, dataset: str, batch_size: int = 4, max_pages_per_doc: int = 0) -> dict:
        import torch
        from PIL import Image

        self._ensure_model()
        processed = self.cfg.path("processed", create=False) / dataset
        entries: dict[str, np.ndarray] = {}
        n_pages = 0
        for doc_dir in sorted(processed.iterdir()):
            pages_dir = doc_dir / "pages"
            if not pages_dir.is_dir():
                continue
            pngs = sorted(pages_dir.glob("p*.png"))
            if max_pages_per_doc:
                pngs = pngs[:max_pages_per_doc]
            for i in range(0, len(pngs), batch_size):
                batch_files = pngs[i : i + batch_size]
                images = [Image.open(p).convert("RGB") for p in batch_files]
                inputs = self._processor.process_images(images).to(self.device)
                with torch.no_grad():
                    embs = self._model(**inputs)  # (B, n_tokens, dim)
                for f, e in zip(batch_files, embs):
                    page = int(f.stem[1:])
                    key = f"{doc_dir.name}|{page}"
                    entries[key] = e.cpu().to(torch.float16).numpy()
                    n_pages += 1
        np.savez_compressed(self._npz_path(dataset), **entries)
        return {"pages": n_pages, "model": self.model_id, "path": str(self._npz_path(dataset))}

    # ---------- search ----------

    def _load_index(self, dataset: str) -> dict[str, np.ndarray]:
        npz = np.load(self._npz_path(dataset))
        return {k: npz[k] for k in npz.files}

    def exists(self, dataset: str) -> bool:
        return self._npz_path(dataset).exists()

    def search(self, query: str, dataset: str, k: int = 10) -> list[dict]:
        import torch

        self._ensure_model()
        if not hasattr(self, "_cache"):
            self._cache: dict[str, dict[str, np.ndarray]] = {}
        if dataset not in self._cache:
            self._cache[dataset] = self._load_index(dataset)
        index = self._cache[dataset]

        inputs = self._processor.process_queries([query]).to(self.device)
        with torch.no_grad():
            q = self._model(**inputs)[0].cpu().float().numpy()  # (nq, dim)

        scores: list[tuple[float, str]] = []
        for key, doc_emb in index.items():
            sim = q @ doc_emb.astype(np.float32).T  # (nq, nd)
            scores.append((float(sim.max(axis=1).sum()), key))
        scores.sort(reverse=True)
        out = []
        for score, key in scores[:k]:
            doc_id, page = key.split("|")
            out.append({"doc_id": doc_id, "page": int(page), "score": score})
        return out

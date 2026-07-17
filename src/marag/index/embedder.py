from __future__ import annotations

import numpy as np

from ..config import Config


def resolve_device(pref: str = "auto") -> str:
    import os

    env = os.environ.get("MARAG_DEVICE")  # runtime override, e.g. to serialize MPS access
    if env:
        return env
    if pref != "auto":
        return pref
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Embedder:
    """In-process dense embedder (sentence-transformers), lazy-loaded."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.cfg.models.embedder, device=resolve_device(self.cfg.embedding.device)
            )
            # Bound attention memory: chunks are <=~800 tokens by construction; without this
            # a single pathological long-line chunk can demand a 100GB+ MPS buffer.
            self._model.max_seq_length = min(self._model.max_seq_length or 1024, 1024)
        return self._model

    def embed_docs(self, texts: list[str]) -> np.ndarray:
        texts = [t[:5000] for t in texts]  # second line of defense against runaway chunks
        return self.model.encode(
            texts,
            batch_size=self.cfg.embedding.batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 256,
        ).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        q = self.cfg.embedding.query_instruction + query
        return self.model.encode([q], normalize_embeddings=True).astype(np.float32)[0]

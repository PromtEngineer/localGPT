import os
from typing import Any, Dict, List, Protocol

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from rag_system.model_registry import get_dtype as _registry_get_dtype


# We keep the protocol to ensure a consistent interface
class EmbeddingModel(Protocol):
    def create_embeddings(self, texts: List[str]) -> np.ndarray: ...


# Global cache for models - use dict to cache by model name
_MODEL_CACHE = {}


# --- New Ollama Embedder ---
class QwenEmbedder(EmbeddingModel):
    """
    An embedding model that uses a local Hugging Face transformer model.
    Model weights are loaded on the first create_embeddings() call so that
    importing this module (and creating Agent instances) doesn't block startup.
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B"):
        self.model_name = model_name
        # Weights and device are resolved on first use.
        self._loaded = False
        self.tokenizer = None
        self.model = None
        self.device = None

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        # Auto-select the best available device: CUDA > MPS > CPU
        if torch.cuda.is_available():
            self.device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        if self.model_name not in _MODEL_CACHE:
            print(
                f"Initializing HF Embedder with model '{self.model_name}' on device '{self.device}'. (first load)"
            )
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True, padding_side="left"
            )
            model = (
                AutoModel.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    torch_dtype=_registry_get_dtype(self.model_name, self.device),
                )
                .to(self.device)
                .eval()
            )
            _MODEL_CACHE[self.model_name] = (tokenizer, model)
            print(f"QwenEmbedder weights loaded and cached for {self.model_name}.")
        else:
            print(f"Reusing cached QwenEmbedder weights for {self.model_name}.")

        self.tokenizer, self.model = _MODEL_CACHE[self.model_name]
        self._loaded = True

    # Attention cost is quadratic in sequence length and activation memory
    # scales with batch × padded length. Two guards keep both bounded:
    # - a hard cap on tokens per sequence (atomic table chunks from Docling
    #   can reach several thousand tokens; embedding their first N tokens is
    #   an acceptable trade for a bounded runtime)
    # - a token budget per forward pass, packed over length-sorted texts so
    #   one giant chunk can't pad a whole batch up to its own length
    MAX_SEQ_TOKENS = int(os.getenv("EMBED_MAX_TOKENS", "2048"))
    _DEVICE_TOKEN_BUDGET = {"cuda": 49152, "mps": 12288, "cpu": 6144}
    _MAX_BATCH_COUNT = 64

    def _forward(self, texts: List[str]) -> np.ndarray:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.MAX_SEQ_TOKENS,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden = outputs.last_hidden_state  # [B, seq, dim]
            # Pool via last valid token per sequence (recommended for Qwen3)
            seq_len = inputs["attention_mask"].sum(dim=1) - 1  # index of last token
            batch_indices = torch.arange(last_hidden.size(0), device=self.device)
            embeddings = last_hidden[batch_indices, seq_len]
        return embeddings.cpu().numpy()

    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        self._ensure_loaded()
        print(f"Generating {len(texts)} embeddings with {self.model_name} model...")
        budget = self._DEVICE_TOKEN_BUDGET.get(self.device, 6144)

        # Sort by approximate token count (≈ chars/4), pack greedily into
        # batches whose padded size (count × longest) fits the budget, then
        # restore original order.
        approx = [min(max(1, len(t) // 4), self.MAX_SEQ_TOKENS) for t in texts]
        order = sorted(range(len(texts)), key=lambda i: approx[i])

        results: List[np.ndarray | None] = [None] * len(texts)
        batch: List[int] = []
        batch_max = 0

        def _flush():
            nonlocal batch, batch_max
            if not batch:
                return
            vecs = self._forward([texts[i] for i in batch])
            for idx, vec in zip(batch, vecs, strict=False):
                results[idx] = vec
            batch, batch_max = [], 0

        for i in order:
            new_max = max(batch_max, approx[i])
            if batch and (
                (len(batch) + 1) * new_max > budget
                or len(batch) >= self._MAX_BATCH_COUNT
            ):
                _flush()
                new_max = approx[i]
            batch.append(i)
            batch_max = new_max
        _flush()

        embeddings_np = np.vstack(results)

        # Check for NaN or infinite values
        if np.isnan(embeddings_np).any():
            print(
                f"⚠️ Warning: NaN values detected in embeddings from {self.model_name}"
            )
            # Replace NaN values with zeros
            embeddings_np = np.nan_to_num(
                embeddings_np, nan=0.0, posinf=0.0, neginf=0.0
            )
            print("🔄 Replaced NaN values with zeros")

        if np.isinf(embeddings_np).any():
            print(
                f"⚠️ Warning: Infinite values detected in embeddings from {self.model_name}"
            )
            # Replace infinite values with zeros
            embeddings_np = np.nan_to_num(
                embeddings_np, nan=0.0, posinf=0.0, neginf=0.0
            )
            print("🔄 Replaced infinite values with zeros")

        return embeddings_np


class EmbeddingGenerator:
    def __init__(self, embedding_model: EmbeddingModel, batch_size: int = 50):
        self.model = embedding_model
        self.batch_size = batch_size

    def generate(self, chunks: List[Dict[str, Any]]) -> List[np.ndarray]:
        """Generate embeddings for all chunks using batch processing"""
        texts_to_embed = [chunk["text"] for chunk in chunks]
        if not texts_to_embed:
            return []

        from rag_system.utils.batch_processor import (
            BatchProcessor,
            estimate_memory_usage,
        )

        memory_mb = estimate_memory_usage(chunks)
        print(f"Estimated memory usage for {len(chunks)} chunks: {memory_mb:.1f}MB")

        batch_processor = BatchProcessor(batch_size=self.batch_size)

        def process_text_batch(text_batch):
            if not text_batch:
                return []
            batch_embeddings = self.model.create_embeddings(text_batch)
            return list(batch_embeddings)

        all_embeddings = batch_processor.process_in_batches(
            texts_to_embed, process_text_batch, "Embedding Generation"
        )

        return all_embeddings


class OllamaEmbedder(EmbeddingModel):
    """Embed via Ollama — batched /api/embed with a legacy per-text fallback.

    Running embeddings through Ollama keeps torch/transformers out of this
    process entirely: Ollama serves a quantized model and unloads it when
    idle, which is a large memory win for local indexing.
    """

    def __init__(
        self,
        model_name: str,
        host: str | None = None,
        timeout: int = 120,
        batch_size: int = 64,
    ):
        self.model_name = model_name
        self.host = (
            host or os.getenv("OLLAMA_HOST") or "http://localhost:11434"
        ).rstrip("/")
        self.timeout = timeout
        self.batch_size = batch_size
        self.keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "10m")
        # Set after the first 404 from /api/embed (Ollama < 0.1.45)
        self._use_legacy_endpoint = False

    def _embed_single(self, text: str):
        import numpy as np
        import requests

        payload = {
            "model": self.model_name,
            "prompt": text,
            "keep_alive": self.keep_alive,
        }
        r = requests.post(
            f"{self.host}/api/embeddings", json=payload, timeout=self.timeout
        )
        r.raise_for_status()
        data = r.json()
        # Ollama may return {"embedding": [...]} or {"data": [...]} depending on version
        vec = data.get("embedding") or data.get("data")
        if vec is None:
            raise ValueError("Unexpected Ollama embeddings response format")
        return np.array(vec, dtype="float32")

    def _embed_batch(self, batch: List[str]):
        import numpy as np
        import requests

        payload = {
            "model": self.model_name,
            "input": batch,
            "keep_alive": self.keep_alive,
        }
        r = requests.post(f"{self.host}/api/embed", json=payload, timeout=self.timeout)
        if r.status_code == 404:
            # Old Ollama without the batched endpoint
            self._use_legacy_endpoint = True
            return [self._embed_single(t) for t in batch]
        r.raise_for_status()
        embeddings = r.json().get("embeddings")
        if not embeddings or len(embeddings) != len(batch):
            raise ValueError("Unexpected Ollama embed response format")
        return [np.array(vec, dtype="float32") for vec in embeddings]

    def create_embeddings(self, texts: List[str]):
        import numpy as np

        vectors = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            if self._use_legacy_endpoint:
                vectors.extend(self._embed_single(t) for t in batch)
            else:
                vectors.extend(self._embed_batch(batch))
        embeddings_np = np.vstack(vectors)

        # Check for NaN or infinite values
        if np.isnan(embeddings_np).any():
            print(
                f"⚠️ Warning: NaN values detected in Ollama embeddings from {self.model_name}"
            )
            # Replace NaN values with zeros
            embeddings_np = np.nan_to_num(
                embeddings_np, nan=0.0, posinf=0.0, neginf=0.0
            )
            print("🔄 Replaced NaN values with zeros")

        if np.isinf(embeddings_np).any():
            print(
                f"⚠️ Warning: Infinite values detected in Ollama embeddings from {self.model_name}"
            )
            # Replace infinite values with zeros
            embeddings_np = np.nan_to_num(
                embeddings_np, nan=0.0, posinf=0.0, neginf=0.0
            )
            print("🔄 Replaced infinite values with zeros")

        return embeddings_np


def select_embedder(model_name: str, ollama_host: str | None = None):
    """Return appropriate EmbeddingModel implementation for the given name."""
    if "/" in model_name or model_name.startswith("http"):
        # Treat as HF model path
        return QwenEmbedder(model_name=model_name)
    # Otherwise assume it's an Ollama tag
    return OllamaEmbedder(model_name=model_name, host=ollama_host)


if __name__ == "__main__":
    print("representations.py cleaned up.")
    try:
        qwen_embedder = QwenEmbedder()
        emb_gen = EmbeddingGenerator(embedding_model=qwen_embedder)

        sample_chunks = [{"text": "Hello world"}, {"text": "This is a test"}]
        embeddings = emb_gen.generate(sample_chunks)

        print(f"\nSuccessfully generated {len(embeddings)} embeddings.")
        print(f"Shape of first embedding: {embeddings[0].shape}")

    except Exception as e:
        print(f"\nAn error occurred during the QwenEmbedder test: {e}")
        print("Please ensure you have an internet connection for model downloads.")

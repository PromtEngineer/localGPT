"""Central registry of supported embedding models with their dimensions and dtype."""

from typing import Optional

import torch

EMBEDDING_REGISTRY: dict[str, dict] = {
    "Qwen/Qwen3-Embedding-0.6B": {
        "dims": 1024,
        "dtype": "float16",
        "source": "huggingface",
    },
    "Qwen/Qwen3-Embedding-4B": {
        "dims": 2048,
        "dtype": "float16",
        "source": "huggingface",
    },
    "Qwen/Qwen3-Embedding-8B": {
        "dims": 4096,
        "dtype": "float16",
        "source": "huggingface",
    },
    "BAAI/bge-small-en-v1.5": {
        "dims": 384,
        "dtype": "float16",
        "source": "huggingface",
    },
    "BAAI/bge-base-en-v1.5": {"dims": 768, "dtype": "float16", "source": "huggingface"},
    "BAAI/bge-large-en-v1.5": {
        "dims": 1024,
        "dtype": "float16",
        "source": "huggingface",
    },
    "nomic-embed-text": {"dims": 768, "dtype": None, "source": "ollama"},
    "mxbai-embed-large": {"dims": 1024, "dtype": None, "source": "ollama"},
}

# Reverse mapping: dims → representative model label (used for inference from vector size)
DIMS_TO_LABEL: dict[int, str] = {
    384: "BAAI/bge-small-en-v1.5 (or similar)",
    512: "sentence-transformers/all-MiniLM-L6-v2 (or similar)",
    768: "BAAI/bge-base-en-v1.5 (or similar)",
    1024: "Qwen/Qwen3-Embedding-0.6B (or similar)",
    1536: "text-embedding-ada-002 (or similar)",
    2048: "Qwen/Qwen3-Embedding-4B (or similar)",
    4096: "Qwen/Qwen3-Embedding-8B (or similar)",
}


def get_dims(model_name: str) -> Optional[int]:
    """Return the embedding dimension for a registered model, or None if unknown."""
    entry = EMBEDDING_REGISTRY.get(model_name)
    return entry["dims"] if entry else None


def get_dtype(model_name: str, device: str = "cpu"):
    """Return the torch dtype for a model on the given device.

    Returns torch.float16 for GPU devices when the registry says float16,
    otherwise returns None (uses the model's default dtype).
    """
    if device == "cpu":
        return None
    entry = EMBEDDING_REGISTRY.get(model_name)
    if entry and entry.get("dtype") == "float16":
        return torch.float16
    return None


def huggingface_models() -> list[str]:
    """Return all HuggingFace model names in the registry."""
    return [
        name
        for name, meta in EMBEDDING_REGISTRY.items()
        if meta["source"] == "huggingface"
    ]

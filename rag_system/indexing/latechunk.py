from __future__ import annotations

"""Late Chunking encoder.

This helper feeds the *entire* document to the embedding model, collects
per-token hidden-states and then mean-pools those vectors inside pre-defined
chunk spans.  The end result is one vector per chunk – but each vector has
been produced with knowledge of the *whole* document, alleviating context-loss
issues of vanilla chunking.

We purposefully keep this class lightweight and free of LanceDB/Chunking
logic so it can be re-used elsewhere (e.g. notebook experiments).
"""

from typing import List, Tuple

import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np

class LateChunkEncoder:
    """Generate late-chunked embeddings given character-offset spans."""

    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B", *, max_tokens: int = 8192) -> None:
        self.model_name = model_name
        self.max_len = max_tokens
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Back-compat: allow short alias without repo namespace
        repo_id = model_name
        if "/" not in model_name and not model_name.startswith("Qwen/"):
            # map common alias to official repo
            alias_map = {
                "qwen3-embedding-0.6b": "Qwen/Qwen3-Embedding-0.6B",
            }
            repo_id = alias_map.get(model_name.lower(), model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode(self, text: str, chunk_spans: List[Tuple[int, int]]) -> List[np.ndarray]:
        """Return one vector *per* span.

        Args:
            text: Full document text.
            chunk_spans: List of (char_start, char_end) offsets for each chunk.

        Returns:
            List of numpy float32 arrays – one per chunk.
        """
        if not chunk_spans:
            return []

        # Tokenise and obtain per-token hidden states
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_len,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        offsets = inputs.pop("offset_mapping").squeeze(0).cpu().tolist()  # (seq_len, 2)

        out = self.model(**inputs)
        last_hidden = out.last_hidden_state.squeeze(0)  # (seq_len, dim)
        last_hidden = last_hidden.cpu()

        # For each chunk span, gather token indices belonging to it
        vectors: List[np.ndarray] = []
        for start_char, end_char in chunk_spans:
            token_indices = [i for i, (s, e) in enumerate(offsets) if s >= start_char and e <= end_char]
            if not token_indices:
                # Fallback: if tokenizer lost the span (e.g. due to trimming) just average CLS + SEP
                token_indices = [0]
            chunk_vec = last_hidden[token_indices].mean(dim=0).numpy().astype("float32")
            
            # Check for NaN or infinite values
            if np.isnan(chunk_vec).any() or np.isinf(chunk_vec).any():
                print(f"⚠️ Warning: Invalid values detected in late chunk embedding for span ({start_char}, {end_char})")
                # Replace invalid values with zeros
                chunk_vec = np.nan_to_num(chunk_vec, nan=0.0, posinf=0.0, neginf=0.0)
                print(f"🔄 Replaced invalid values with zeros")
            
            vectors.append(chunk_vec)
        return vectors 
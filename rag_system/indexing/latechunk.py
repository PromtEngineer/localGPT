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

    def __init__(self, model_name: str | None = None, *, max_tokens: int = 8192) -> None:
        if not model_name:
            from rag_system.main import EXTERNAL_MODELS
            model_name = EXTERNAL_MODELS["embedding_model"]
        self.model_name = model_name
        self.max_len = max_tokens
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device.type != "cpu" else None,
        )
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

        # Tokenise the whole document once WITHOUT truncation, then run the
        # model over overlapping windows of max_len tokens. A single truncated
        # pass used to leave every span past max_len without tokens; the
        # windows keep global character offsets so each span maps to real
        # token vectors no matter how long the document is.
        encoding = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        input_ids = encoding["input_ids"]
        offsets = encoding["offset_mapping"]  # global (char_start, char_end) per token
        total_tokens = len(input_ids)

        if total_tokens == 0:
            # Empty/whitespace document: nothing to pool, return zero vectors.
            hidden = int(getattr(getattr(self.model, "config", None), "hidden_size", 0) or 0)
            return [np.zeros(hidden, dtype="float32") for _ in chunk_spans]

        overlap = self.max_len // 4
        stride = max(1, self.max_len - overlap)

        # Run each window and keep the per-token vectors (first window to
        # cover a token wins; overlap tokens are recomputed but equivalent).
        token_vectors = [None] * total_tokens
        for win_start in range(0, total_tokens, stride):
            win_end = min(win_start + self.max_len, total_tokens)
            window_ids = torch.tensor([input_ids[win_start:win_end]], dtype=torch.long, device=self.device)
            attention = torch.ones_like(window_ids)
            out = self.model(input_ids=window_ids, attention_mask=attention)
            last_hidden = out.last_hidden_state.squeeze(0)  # (win_len, dim)
            last_hidden = last_hidden.float().cpu()
            for i in range(win_end - win_start):
                if token_vectors[win_start + i] is None:
                    token_vectors[win_start + i] = last_hidden[i]
            if win_end >= total_tokens:
                break

        # For each chunk span, gather token indices belonging to it
        vectors: List[np.ndarray] = []
        for start_char, end_char in chunk_spans:
            token_indices = [i for i, (s, e) in enumerate(offsets) if s >= start_char and e <= end_char]
            if not token_indices:
                # Degenerate span (e.g. whitespace the tokenizer folded into a
                # neighbouring token): pool the nearest preceding token rather
                # than the old always-token-0 fallback.
                preceding = [i for i, (s, e) in enumerate(offsets) if e <= start_char]
                token_indices = [preceding[-1]] if preceding else [0]
            chunk_vec = torch.stack([token_vectors[i] for i in token_indices]).mean(dim=0).numpy().astype("float32")
            
            # Check for NaN or infinite values
            if np.isnan(chunk_vec).any() or np.isinf(chunk_vec).any():
                print(f"⚠️ Warning: Invalid values detected in late chunk embedding for span ({start_char}, {end_char})")
                # Replace invalid values with zeros
                chunk_vec = np.nan_to_num(chunk_vec, nan=0.0, posinf=0.0, neginf=0.0)
                print(f"🔄 Replaced invalid values with zeros")
            
            vectors.append(chunk_vec)
        return vectors 
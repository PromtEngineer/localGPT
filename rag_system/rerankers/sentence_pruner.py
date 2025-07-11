from __future__ import annotations

"""Sentence-level context pruning using the Provence model (ICLR 2025).

This lightweight helper wraps the HuggingFace model hosted at
`naver/provence-reranker-debertav3-v1` and exposes a thread-safe
`prune_documents()` method that converts a list of RAG chunks into their
pruned variants.

The module fails gracefully – if the model weights cannot be downloaded
(or the `transformers` / `nltk` deps are missing) we simply return the
original documents unchanged so the upstream pipeline continues
unaffected.
"""

from threading import Lock
from typing import List, Dict, Any


class SentencePruner:
    """Lightweight singleton wrapper around the Provence model."""

    _model = None  # shared across all instances
    _init_lock: Lock = Lock()

    def __init__(self, model_name: str = "naver/provence-reranker-debertav3-v1") -> None:
        self.model_name = model_name
        self._ensure_model()

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _ensure_model(self) -> None:
        """Lazily download and load the Provence model exactly once."""
        if SentencePruner._model is not None:
            return

        with SentencePruner._init_lock:
            if SentencePruner._model is not None:
                return  # another thread beat us
            try:
                from transformers import AutoModel  # local import to keep base deps light

                print("🔧 Loading Provence sentence-pruning model …")
                SentencePruner._model = AutoModel.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                )
                print("✅ Provence model loaded successfully.")
            except Exception as e:
                # Any failure leaves the singleton as None so callers can skip pruning.
                print(f"❌ Failed to load Provence model: {e}. Context pruning will be skipped.")
                SentencePruner._model = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def prune_documents(
        self,
        question: str,
        docs: List[Dict[str, Any]],
        *,
        threshold: float = 0.1,
    ) -> List[Dict[str, Any]]:
        """Return *docs* with their `text` field pruned sentence-wise.

        If the model could not be initialised we simply echo the input.
        """
        if SentencePruner._model is None:
            return docs  # model unavailable – no-op

        # Batch texts for efficiency when >1 doc
        texts = [d.get("text", "") for d in docs]

        try:
            if len(texts) == 1:
                # returns dict
                outputs = [SentencePruner._model.process(question, texts[0], threshold=threshold)]
            else:
                # Batch call expects list[list[str]] with same outer length as questions list (1)
                batched_out = SentencePruner._model.process(question, [texts], threshold=threshold)
                # HF returns List[Dict] per question
                outputs = batched_out[0] if isinstance(batched_out, list) else batched_out
                if isinstance(outputs, dict):
                    outputs = [outputs]
                if len(outputs) != len(texts):
                    print("⚠️ Provence batch size mismatch; falling back to per-doc loop")
                    raise ValueError

            pruned: List[Dict[str, Any]] = []
            for doc, out in zip(docs, outputs):
                raw = out.get("pruned_context", doc.get("text", "")) if isinstance(out, dict) else doc.get("text", "")
                new_text = raw if isinstance(raw, str) else " ".join(raw)  # HF model may return a list of sentences
                pruned.append({**doc, "text": new_text})
        except Exception as e:
            print(f"⚠️ Provence batch pruning failed ({e}); falling back to individual calls")
            pruned = []
            for doc in docs:
                text = doc.get("text", "")
                if not text:
                    pruned.append(doc)
                    continue
                try:
                    res = SentencePruner._model.process(question, text, threshold=threshold)
                    raw = res.get("pruned_context", text) if isinstance(res, dict) else text
                    new_text = raw if isinstance(raw, str) else " ".join(raw)
                    pruned.append({**doc, "text": new_text})
                except Exception as err:
                    print(f"⚠️ Provence pruning failed for chunk {doc.get('chunk_id')}: {err}")
                    pruned.append(doc)

        return pruned 
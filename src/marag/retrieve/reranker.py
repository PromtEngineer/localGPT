from __future__ import annotations

from ..config import Config
from ..index.embedder import resolve_device

_PREFIX = (
    '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query '
    'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
_INSTRUCTION = "Given a search query, retrieve relevant passages that answer the query"


class Reranker:
    """Qwen3-Reranker cross-encoder (yes/no logit scoring), lazy-loaded, optional."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._loaded = False

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from ..index.visual import _local_snapshot

        path = _local_snapshot(self.cfg.models.reranker) or self.cfg.models.reranker
        self.device = resolve_device(self.cfg.embedding.device)
        self.tok = AutoTokenizer.from_pretrained(path, padding_side="left")
        dtype = torch.float16 if self.device != "cpu" else torch.float32
        self.model = (
            AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype)
            .to(self.device)
            .eval()
        )
        self.yes_id = self.tok.convert_tokens_to_ids("yes")
        self.no_id = self.tok.convert_tokens_to_ids("no")
        self._loaded = True

    def score(self, query: str, docs: list[str], batch_size: int = 8) -> list[float]:
        import torch

        if not self._loaded:
            self._load()
        texts = [
            _PREFIX + f"<Instruct>: {_INSTRUCTION}\n<Query>: {query}\n<Document>: {d[:6000]}" + _SUFFIX
            for d in docs
        ]
        scores: list[float] = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = self.tok(
                    texts[i : i + batch_size],
                    padding=True,
                    truncation=True,
                    max_length=2048,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(**batch).logits[:, -1, :]
                pair = torch.stack([logits[:, self.no_id], logits[:, self.yes_id]], dim=1)
                probs = torch.softmax(pair.float(), dim=1)[:, 1]
                scores.extend(probs.cpu().tolist())
        return scores

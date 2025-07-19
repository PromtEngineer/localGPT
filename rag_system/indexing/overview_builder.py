from __future__ import annotations

import os, json, logging, re
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class OverviewBuilder:
    """Generates and stores a one-paragraph overview for each document.
    The overview is derived from the first *n* chunks of the document.
    """

    DEFAULT_PROMPT = (
        "You will receive the beginning of a document. "
        "In no more than 120 tokens, describe what the document is about, "
        "state its type (e.g. invoice, slide deck, policy, research paper, receipt) "
        "and mention 3-5 important entities, numbers or dates it contains.\n\n"
        "DOCUMENT_START:\n{text}\n\nOVERVIEW:"
    )

    def __init__(self, llm_client, model: str = "qwen3:0.6b", first_n_chunks: int = 5,
                 out_path: str | None = None):
        if out_path is None:
            out_path = "index_store/overviews/overviews.jsonl"
        self.llm_client = llm_client
        self.model = model
        self.first_n = first_n_chunks
        self.out_path = out_path
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def build_and_store(self, doc_id: str, chunks: List[Dict[str, Any]]):
        if not chunks:
            return
        head_text = "\n".join(c["text"] for c in chunks[: self.first_n] if c.get("text"))
        prompt = self.DEFAULT_PROMPT.format(text=head_text[:5000])  # safety cap
        try:
            resp = self.llm_client.generate_completion(model=self.model, prompt=prompt, enable_thinking=False)
            summary_raw = resp.get("response", "")
            # Remove any lingering <think>...</think> blocks just in case
            summary = re.sub(r'<think[^>]*>.*?</think>', '', summary_raw, flags=re.IGNORECASE | re.DOTALL).strip()
        except Exception as e:
            summary = f"Failed to generate overview: {e}"
        record = {"doc_id": doc_id, "overview": summary.strip()}
        with open(self.out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(f"📄 Overview generated for {doc_id} (stored in {self.out_path})") 
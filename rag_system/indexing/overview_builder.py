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
                 out_path: str | None = None, timeout: int = 60):
        if out_path is None:
            out_path = "index_store/overviews/overviews.jsonl"
        self.llm_client = llm_client
        self.model = model
        self.first_n = first_n_chunks
        self.out_path = out_path
        self.timeout = timeout
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def _read_all(self) -> list[dict]:
        """Return all valid records from the JSONL file."""
        if not os.path.exists(self.out_path):
            return []
        records = []
        try:
            with open(self.out_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            pass
        return records

    def _doc_id_exists(self, doc_id: str) -> bool:
        return any(r.get("doc_id") == doc_id for r in self._read_all())

    def _remove_entry(self, doc_id: str) -> None:
        """Rewrite the JSONL omitting any record whose doc_id matches."""
        records = [r for r in self._read_all() if r.get("doc_id") != doc_id]
        try:
            with open(self.out_path, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        except OSError as e:
            logger.warning(f"Could not rewrite overviews file: {e}")

    def compact(self) -> int:
        """Deduplicate the JSONL in-place, keeping the last entry per doc_id.

        Returns the number of duplicate lines removed.
        """
        records = self._read_all()
        seen: dict[str, dict] = {}
        for r in records:
            doc_id = r.get("doc_id")
            if doc_id:
                seen[doc_id] = r  # last one wins
        duplicates = len(records) - len(seen)
        if duplicates > 0:
            try:
                with open(self.out_path, "w", encoding="utf-8") as f:
                    for r in seen.values():
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                logger.info(f"Compacted overviews: removed {duplicates} duplicate(s)")
            except OSError as e:
                logger.warning(f"Could not compact overviews file: {e}")
        return duplicates

    def build_and_store(self, doc_id: str, chunks: List[Dict[str, Any]],
                        force: bool = False) -> None:
        """Generate and persist an overview for ``doc_id``.

        Args:
            force: When True, replace any existing entry for this doc_id
                   (used when the source file has changed content).
                   When False, skip generation if an entry already exists.
        """
        if not chunks:
            return
        if force:
            self._remove_entry(doc_id)
        elif self._doc_id_exists(doc_id):
            logger.debug(f"Overview already exists for {doc_id}, skipping.")
            return

        head_text = "\n".join(c["text"] for c in chunks[: self.first_n] if c.get("text"))
        prompt = self.DEFAULT_PROMPT.format(text=head_text[:5000])  # safety cap
        try:
            resp = self.llm_client.generate_completion(
                model=self.model,
                prompt=prompt,
                enable_thinking=False,
                timeout=self.timeout,
            )
            summary_raw = resp.get("response", "")
            summary = re.sub(r'<think[^>]*>.*?</think>', '', summary_raw, flags=re.IGNORECASE | re.DOTALL).strip()
        except Exception as e:
            summary = f"Failed to generate overview: {e}"

        record = {"doc_id": doc_id, "overview": summary.strip()}
        with open(self.out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(f"Overview generated for {doc_id} (stored in {self.out_path})")

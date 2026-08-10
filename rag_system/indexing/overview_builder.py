from __future__ import annotations

import os, json, logging, re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embedded-overview sidecar (roadmap item 4.3)
# ---------------------------------------------------------------------------
# The overviews themselves are one JSONL line per document, appended as each
# document is chunked. The overview *prefilter* needs them as vectors, so a
# sidecar `.npz` is written next to the JSONL at the end of an index build:
#
#   index_store/overviews/<index_id>.jsonl          the overviews
#   index_store/overviews/<index_id>.vectors.npz    doc_ids + L2-normalized vectors
#
# It is a sidecar and not a LanceDB table because it is one row per *document*
# (tens, not thousands), it is rebuilt wholesale rather than queried, and a
# missing sidecar has to be a graceful no-op rather than a schema problem.
#
# Vectors are written by the DOCUMENT-side embedder (no instruction prefix),
# matching how chunks are embedded, so a query-side vector can be compared
# against them with the same asymmetry the chunk index uses.

VECTORS_SUFFIX = ".vectors.npz"


def overview_vectors_path(overview_path: str) -> str:
    """The sidecar path for an overviews JSONL path."""
    if overview_path.endswith(".jsonl"):
        return overview_path[: -len(".jsonl")] + VECTORS_SUFFIX
    return overview_path + VECTORS_SUFFIX


def read_overviews(overview_path: str) -> Dict[str, str]:
    """``{doc_id: overview}`` from an appended JSONL; the last line per doc wins."""
    out: Dict[str, str] = {}
    try:
        with open(overview_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                doc_id = record.get("doc_id")
                overview = (record.get("overview") or "").strip()
                if not doc_id or not overview:
                    continue
                if overview.startswith("Failed to generate overview"):
                    continue
                out[doc_id] = overview
    except OSError:
        return {}
    return out


def load_overview_vectors(vectors_path: str) -> Optional[Dict[str, Any]]:
    """Read a sidecar. Returns ``None`` for any missing or unreadable file."""
    if not vectors_path or not os.path.exists(vectors_path):
        return None
    try:
        import numpy as np

        with np.load(vectors_path, allow_pickle=False) as data:
            doc_ids = [str(d) for d in data["doc_ids"].tolist()]
            vectors = np.asarray(data["vectors"], dtype="float32")
            meta_raw = data["meta"].item() if "meta" in data else "{}"
    except Exception as e:  # unreadable sidecar must never break retrieval
        logger.warning("Could not read overview vectors %s: %s", vectors_path, e)
        return None
    if len(doc_ids) != len(vectors) or not doc_ids:
        return None
    try:
        meta = json.loads(meta_raw.decode("utf-8") if isinstance(meta_raw, bytes) else str(meta_raw))
    except ValueError:
        meta = {}
    return {"doc_ids": doc_ids, "vectors": vectors, "meta": meta, "path": vectors_path}


def write_overview_vectors(vectors_path: str, doc_ids: List[str], vectors,
                           embedding_model: Optional[str]) -> None:
    import numpy as np

    out_dir = os.path.dirname(vectors_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    meta = json.dumps({"embedding_model": embedding_model, "normalized": True})
    np.savez(
        vectors_path,
        doc_ids=np.array(doc_ids, dtype=object).astype("U"),
        vectors=np.asarray(vectors, dtype="float32"),
        meta=np.array(meta),
    )


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

    def __init__(self, llm_client, model: str, first_n_chunks: int = 5,
                 out_path: str | None = None):
        if out_path is None:
            out_path = "index_store/overviews/overviews.jsonl"
        self.llm_client = llm_client
        self.model = model
        self.first_n = first_n_chunks
        self.out_path = out_path
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

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

    # ------------------------------------------------------------------
    # Embedded-overview sidecar (roadmap item 4.3)
    # ------------------------------------------------------------------

    @property
    def vectors_path(self) -> str:
        return overview_vectors_path(self.out_path)

    def embed_and_store_vectors(self, embedder, embedding_model: Optional[str] = None) -> int:
        """Embed every overview in the JSONL and (re)write the ``.npz`` sidecar.

        Rebuilt wholesale rather than appended: the JSONL is append-only, so a
        re-indexed document has several lines and only the last one is current.
        Returns the number of documents embedded (0 when there is nothing to do).
        """
        overviews = read_overviews(self.out_path)
        if not overviews:
            logger.info("No usable overviews in %s; skipping the vector sidecar.", self.out_path)
            return 0

        import numpy as np

        doc_ids = sorted(overviews)
        texts = [overviews[d] for d in doc_ids]
        vectors = np.asarray(embedder.create_embeddings(texts), dtype="float32")
        if vectors.ndim != 2 or len(vectors) != len(doc_ids):
            logger.warning("Overview embedding returned an unexpected shape; sidecar not written.")
            return 0

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        vectors = vectors / norms

        model_name = embedding_model or getattr(embedder, "model_name", None)
        write_overview_vectors(self.vectors_path, doc_ids, vectors, model_name)
        logger.info("🧭 Embedded %d document overview(s) into %s", len(doc_ids), self.vectors_path)
        return len(doc_ids)

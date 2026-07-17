from __future__ import annotations

import re

import lancedb
import numpy as np
import pandas as pd

from ..config import Config

_FTS_SANITIZE = re.compile(r"[^\w\s]")


class Store:
    """LanceDB-backed chunk store: dense vectors + native FTS in one embedded engine."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.db = lancedb.connect(str(cfg.path("index") / "lance"))

    @staticmethod
    def _name(dataset: str) -> str:
        return f"chunks_{dataset}"

    def build(self, dataset: str, chunks: list[dict], vectors: np.ndarray) -> None:
        name = self._name(dataset)
        rows = []
        for c, v in zip(chunks, vectors):
            rows.append(
                {
                    "id": c["id"],
                    "doc_id": c["doc_id"],
                    "page": int(c["page"]),
                    "section": c.get("section", ""),
                    "text": c["text"],
                    "raw_text": c.get("raw_text", c["text"]),
                    "vector": v.astype(np.float32),
                }
            )
        if name in self.db.table_names():
            self.db.drop_table(name)
        tbl = self.db.create_table(name, rows)
        tbl.create_fts_index("text", use_tantivy=False, replace=True)

    def exists(self, dataset: str) -> bool:
        return self._name(dataset) in self.db.table_names()

    def count(self, dataset: str) -> int:
        return self.db.open_table(self._name(dataset)).count_rows()

    def dense(self, dataset: str, qvec: np.ndarray, k: int, doc_id: str | None = None) -> pd.DataFrame:
        q = self.db.open_table(self._name(dataset)).search(qvec, vector_column_name="vector")
        if doc_id:
            q = q.where(f"doc_id = '{doc_id}'", prefilter=True)
        return q.limit(k).to_pandas()

    def fts(self, dataset: str, query: str, k: int, doc_id: str | None = None) -> pd.DataFrame:
        clean = _FTS_SANITIZE.sub(" ", query).strip()
        if not clean:
            return pd.DataFrame()
        q = self.db.open_table(self._name(dataset)).search(clean, query_type="fts")
        if doc_id:
            q = q.where(f"doc_id = '{doc_id}'", prefilter=True)
        return q.limit(k).to_pandas()

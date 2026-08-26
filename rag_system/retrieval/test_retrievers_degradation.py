"""Fault-injection tests for hybrid-leg degradation in MultiVectorRetriever.

The two hybrid legs are redundant by design: one leg down must degrade to the
surviving leg, both legs down must fail loudly, and single-leg modes must not
silently switch legs.

Run it:

    .venv/bin/python rag_system/retrieval/test_retrievers_degradation.py
"""

import os
import shutil
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import lancedb  # noqa: E402

from rag_system.indexing.embedders import LanceDBManager  # noqa: E402
from rag_system.retrieval.retrievers import MultiVectorRetriever  # noqa: E402

DIM = 8


class WorkingEmbedder:
    model_name = "test-embedder"

    def create_embeddings(self, queries):
        return [np.ones(DIM, dtype="float32") for _ in queries]


class BrokenEmbedder:
    model_name = "test-embedder"
    calls = 0

    def create_embeddings(self, queries):
        BrokenEmbedder.calls += 1
        raise RuntimeError("simulated embedder failure")


def make_table(db_path, name, with_fts=True):
    db = lancedb.connect(db_path)
    rows = [
        {"vector": np.random.rand(DIM).astype("float32"), "text": f"alpha document about topic {i}",
         "chunk_id": f"c{i}", "document_id": "doc.txt", "chunk_index": i, "metadata": "{}"}
        for i in range(5)
    ]
    tbl = db.create_table(name, rows, mode="overwrite")
    if with_fts:
        tbl.create_fts_index("text", replace=True)
    return tbl


def main():
    tmp = tempfile.mkdtemp(prefix="degradation_test_")
    failures = []

    def check(label, cond):
        print(("PASS " if cond else "FAIL ") + label)
        if not cond:
            failures.append(label)

    try:
        make_table(tmp, "t_fts", with_fts=True)
        make_table(tmp, "t_nofts", with_fts=False)
        dbm = LanceDBManager(db_path=tmp)

        # 1. Baseline: hybrid with a working embedder returns rows.
        r = MultiVectorRetriever(dbm, WorkingEmbedder())
        docs = r.retrieve("alpha topic", "t_fts", 5, "hybrid")
        check("hybrid baseline returns rows", len(docs) > 0)

        # 2. Dense leg down -> hybrid degrades to FTS-only, still returns rows.
        r = MultiVectorRetriever(dbm, BrokenEmbedder())
        docs = r.retrieve("alpha topic", "t_fts", 5, "hybrid")
        check("hybrid degrades to FTS-only when embedder fails", len(docs) > 0)
        check("broken embedder was actually exercised", BrokenEmbedder.calls > 0)

        # 3. FTS leg down (no FTS index) -> hybrid degrades to dense-only.
        r = MultiVectorRetriever(dbm, WorkingEmbedder())
        docs = r.retrieve("alpha topic", "t_nofts", 5, "hybrid")
        check("hybrid degrades to dense-only when FTS fails", len(docs) > 0)

        # 4. vector_only with a broken embedder must NOT fall back to FTS.
        r = MultiVectorRetriever(dbm, BrokenEmbedder())
        docs = r.retrieve("alpha topic", "t_fts", 5, "vector_only")
        check("vector_only with broken embedder returns no rows (no silent FTS fallback)",
              len(docs) == 0)

        # 5. Both legs down -> loud failure (empty result from the outer handler,
        #    never a partial success).
        r = MultiVectorRetriever(dbm, BrokenEmbedder())
        docs = r.retrieve("alpha topic", "t_nofts", 5, "hybrid")
        check("both legs down yields empty result, not a crash of the caller",
              docs == [])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\nALL CHECKS PASSED" if not failures else f"\nFAILURES: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

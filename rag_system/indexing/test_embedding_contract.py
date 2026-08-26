"""Tests for the v2 embedding-contract markers (dims, revision, instruction sha).

Run it:

    .venv/bin/python rag_system/indexing/test_embedding_contract.py
"""

import io
import os
import shutil
import sys
import tempfile
from contextlib import redirect_stdout

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import lancedb  # noqa: E402
import pyarrow as pa  # noqa: E402

from rag_system.indexing.embedders import (  # noqa: E402
    EmbedderMismatchError,
    LanceDBManager,
    _write_sidecar,
    assert_dims_match,
    contract_drift_warnings,
    query_instruction_sha,
    read_table_marker,
    table_schema_metadata,
)
from rag_system.retrieval.retrievers import MultiVectorRetriever  # noqa: E402

MODEL = "microsoft/harrier-oss-v1-0.6b"
DIM = 8

failures = []


def check(label, cond):
    print(("PASS " if cond else "FAIL ") + label)
    if not cond:
        failures.append(label)


class FakeEmbedder:
    model_name = MODEL

    def __init__(self, dim=DIM):
        self.dim = dim

    def create_embeddings(self, queries):
        return [np.ones(self.dim, dtype="float32") for _ in queries]


def make_table(db_path, name, meta):
    db = lancedb.connect(db_path)
    schema = pa.schema([
        pa.field("vector", pa.list_(pa.float32(), DIM)),
        pa.field("text", pa.string()),
        pa.field("chunk_id", pa.string()),
        pa.field("document_id", pa.string()),
        pa.field("chunk_index", pa.int32()),
        pa.field("metadata", pa.string()),
    ], metadata=meta)
    tbl = db.create_table(name, schema=schema)
    tbl.add([{"vector": np.random.rand(DIM).astype("float32"), "text": "alpha text",
              "chunk_id": "c0", "document_id": "d.txt", "chunk_index": 0, "metadata": "{}"}])
    tbl.create_fts_index("text", replace=True)
    return tbl


def main():
    tmp = tempfile.mkdtemp(prefix="contract_test_")
    try:
        # 1. v2 schema-metadata round-trip.
        tbl = make_table(tmp, "t_v2", table_schema_metadata(MODEL, True, dims=DIM))
        m = read_table_marker(tbl)
        check("v2 marker records model", m["embedding_model"] == MODEL)
        check("v2 marker records dims", m["dims"] == DIM)
        check("v2 marker records marker_version 2", m["marker_version"] == 2)
        check("v2 marker records instruction sha",
              m["query_instruction_sha"] == query_instruction_sha(MODEL))
        check("v2 marker records a revision string", bool(m["model_revision"]))

        # 2. v1 tolerance: old two-field metadata still reads, defaults benign.
        v1_meta = {b"localgpt_embedding_model": MODEL.encode(),
                   b"localgpt_normalized": b"true"}
        tbl1 = make_table(tmp, "t_v1", v1_meta)
        m1 = read_table_marker(tbl1)
        check("v1 marker still reads", m1 is not None and m1["embedding_model"] == MODEL)
        check("v1 marker defaults to version 1, no dims",
              m1["marker_version"] == 1 and m1["dims"] is None)
        check("v1 marker produces no dims error",
              assert_dims_match("t_v1", m1, DIM) is None)
        check("v1 marker produces no drift warnings",
              contract_drift_warnings("t_v1", m1, MODEL) == [])

        # 3. dims mismatch raises, and is NOT degraded away in hybrid retrieval.
        try:
            assert_dims_match("t_v2", m, DIM + 1)
            check("dims mismatch raises EmbedderMismatchError", False)
        except EmbedderMismatchError:
            check("dims mismatch raises EmbedderMismatchError", True)
        dbm = LanceDBManager(db_path=tmp)
        r = MultiVectorRetriever(dbm, FakeEmbedder(dim=DIM + 4))
        try:
            docs = r.retrieve("alpha", "t_v2", 5, "hybrid")
            # outer handler converts to [] — the key point is NO silent
            # FTS-only partial success
            check("hybrid with wrong-width embedder yields no partial results",
                  docs == [])
        except EmbedderMismatchError:
            check("hybrid with wrong-width embedder yields no partial results", True)

        # 4. Instruction drift warns (marker with a bogus sha).
        m_drift = dict(m, query_instruction_sha="0" * 64)
        w = contract_drift_warnings("t_v2", m_drift, MODEL)
        check("instruction drift produces a warning", len(w) == 1 and "instruction" in w[0])

        # 5. Sidecar round-trip carries v2 fields.
        _write_sidecar(tmp, "t_side", MODEL, True, dims=DIM)
        tbl_nomarker = make_table(tmp, "t_side", None)
        ms = read_table_marker(tbl_nomarker, tmp, "t_side")
        check("sidecar v2 round-trip", ms is not None and ms["dims"] == DIM
              and ms["marker_version"] == 2)

        # 6. Matched contract on the happy path: retrieval works, no warnings.
        buf = io.StringIO()
        r2 = MultiVectorRetriever(dbm, FakeEmbedder(dim=DIM))
        with redirect_stdout(buf):
            docs = r2.retrieve("alpha", "t_v2", 5, "hybrid")
        check("matched contract retrieves rows", len(docs) > 0)
        check("matched contract emits no drift warnings", "⚠️" not in buf.getvalue())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\nALL CHECKS PASSED" if not failures else f"\nFAILURES: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

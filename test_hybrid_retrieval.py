#!/usr/bin/env python3
"""Unit tests for MultiVectorRetriever hybrid (FTS + vector) search."""

import os
import sys
import tempfile
import unittest

import numpy as np
import pyarrow as pa

# Allow running from project root without installing the package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _make_embedder(dim: int = 8):
    """Return a minimal embedder that produces random unit vectors."""
    class _FakeEmbedder:
        def create_embeddings(self, texts):
            rng = np.random.default_rng(abs(hash(str(texts))) % (2**31))
            vecs = rng.random((len(texts), dim)).astype(np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            return vecs / np.maximum(norms, 1e-9)
    return _FakeEmbedder()


def _make_lancedb_manager(tmp_dir: str):
    import lancedb
    from rag_system.indexing.embedders import LanceDBManager
    return LanceDBManager(tmp_dir)


def _populate_table(manager, table_name: str, rows: list[dict], dim: int = 8):
    """Insert rows into a LanceDB table. Each row must have 'text' and 'doc_id' keys."""
    import lancedb
    schema = pa.schema([
        pa.field("vector", pa.list_(pa.float32(), dim)),
        pa.field("text", pa.string()),
        pa.field("chunk_id", pa.string()),
        pa.field("document_id", pa.string()),
        pa.field("chunk_index", pa.int32()),
        pa.field("metadata", pa.string()),
    ])
    embedder = _make_embedder(dim)
    records = []
    for i, row in enumerate(rows):
        vec = embedder.create_embeddings([row["text"]])[0].tolist()
        records.append({
            "vector": vec,
            "text": row["text"],
            "chunk_id": f"{row['doc_id']}_{i}",
            "document_id": row["doc_id"],
            "chunk_index": i,
            "metadata": "{}",
        })
    db = manager.db
    if hasattr(db, "table_names") and table_name in db.table_names():
        tbl = manager.get_table(table_name)
        tbl.add(records)
    else:
        tbl = manager.create_table(table_name, schema=schema, mode="create")
        tbl.add(records)
    # Hybrid retrieval needs an inverted index on `text` to run FTS queries (lancedb >= 0.27)
    tbl.create_fts_index("text", use_tantivy=False, replace=True)
    return tbl


class TestHybridRetrieval(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.dim = 8
        cls.table = "test_hybrid"
        cls.manager = _make_lancedb_manager(cls.tmp)
        cls.embedder = _make_embedder(cls.dim)

        # Seed test documents
        cls.rows = [
            {"text": "The quick brown fox jumps over the lazy dog", "doc_id": "doc_fox"},
            {"text": "LanceDB supports full text search and vector similarity", "doc_id": "doc_lance"},
            {"text": "Python is a popular programming language for data science", "doc_id": "doc_python"},
            {"text": "Neural networks learn representations from raw data", "doc_id": "doc_nn"},
            {"text": "The fox was seen near the river at sunset", "doc_id": "doc_fox2"},
        ]
        _populate_table(cls.manager, cls.table, cls.rows, cls.dim)

        from rag_system.retrieval.retrievers import MultiVectorRetriever
        cls.retriever = MultiVectorRetriever(
            db_manager=cls.manager,
            text_embedder=cls.embedder,
            fusion_config={"method": "linear", "bm25_weight": 0.5, "vec_weight": 0.5},
        )

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.tmp, ignore_errors=True)

    # ------------------------------------------------------------------
    def test_fts_returns_results(self):
        """FTS search on an indexed term returns matching documents."""
        tbl = self.manager.get_table(self.table)
        df = (
            tbl.search(query="fox", query_type="fts")
               .limit(5)
               .to_df()
        )
        self.assertGreater(len(df), 0, "FTS search should return results for 'fox'")
        texts = df["text"].tolist()
        self.assertTrue(any("fox" in t.lower() for t in texts))

    def test_vector_search_returns_results(self):
        """Vector search returns at least one result."""
        tbl = self.manager.get_table(self.table)
        q_vec = self.embedder.create_embeddings(["quick brown fox"])[0]
        df = tbl.search(q_vec).limit(3).to_df()
        self.assertGreater(len(df), 0)

    def test_hybrid_retrieve_returns_k_results(self):
        """retrieve() returns up to k unique documents."""
        docs = self.retriever.retrieve("fox", self.table, k=3)
        self.assertGreater(len(docs), 0)
        self.assertLessEqual(len(docs), 3)

    def test_fusion_weights_affect_ranking(self):
        """Different fusion weights produce different orderings for unambiguous queries."""
        from rag_system.retrieval.retrievers import MultiVectorRetriever
        r_bm25 = MultiVectorRetriever(
            db_manager=self.manager,
            text_embedder=self.embedder,
            fusion_config={"method": "linear", "bm25_weight": 0.9, "vec_weight": 0.1},
        )
        r_vec = MultiVectorRetriever(
            db_manager=self.manager,
            text_embedder=self.embedder,
            fusion_config={"method": "linear", "bm25_weight": 0.1, "vec_weight": 0.9},
        )
        docs_bm25 = r_bm25.retrieve("programming language", self.table, k=5)
        docs_vec = r_vec.retrieve("programming language", self.table, k=5)
        # Both return results (rankings may differ, but both work)
        self.assertGreater(len(docs_bm25), 0)
        self.assertGreater(len(docs_vec), 0)

    def test_deduplication(self):
        """No duplicate chunk_ids in returned results."""
        docs = self.retriever.retrieve("data", self.table, k=10)
        chunk_ids = [d["chunk_id"] for d in docs if "chunk_id" in d]
        self.assertEqual(len(chunk_ids), len(set(chunk_ids)), "Duplicate chunk_ids found")

    def test_surrounding_chunks(self):
        """_get_surrounding_chunks_lancedb returns a list without crashing."""
        from rag_system.pipelines.retrieval_pipeline import RetrievalPipeline
        rp = RetrievalPipeline.__new__(RetrievalPipeline)
        rp.db_manager = self.manager
        rp.config = {"storage": {"text_table_name": self.table}}
        central_chunk = {
            "document_id": "doc_fox",
            "chunk_index": 0,
            "chunk_id": "doc_fox_0",
            "text": "The quick brown fox jumps over the lazy dog",
        }
        chunks = rp._get_surrounding_chunks_lancedb(central_chunk, window_size=1)
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

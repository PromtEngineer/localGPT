"""Regression tests for failure modes fixed after external review.

Covers:
- build jobs reaching a terminal state on unexpected errors (stuck-job fix)
- enrichApiKey never persisted with job options (+ legacy-row scrub)
- session deletion with linked indexes (FK enforcement regression)
- hybrid retrieval ranking: scored before truncation, sorted, single-leg
  modes, per-call fusion override
- index builds failing when every file fails
- "all files unchanged" builds validating the existing table

- multi-index consistency: both servers resolve the active index through
  rag_system.index_selection, pinned by a source-level tripwire
"""

import json
import os
import shutil
import sqlite3
import tempfile
import unittest

import numpy as np

from backend.database import ChatDatabase
import backend.server as server


class _RecordingThread:
    """Stands in for threading.Thread so endpoint tests don't run builds."""

    last = None

    def __init__(self, target=None, args=(), kwargs=None, daemon=None):
        _RecordingThread.last = {"target": target, "args": args, "kwargs": kwargs or {}}

    def start(self):
        pass


class JobFailureStateTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.original_db = server.db
        server.db = ChatDatabase(os.path.join(self.temp_dir, "chat_data.db"))
        self.original_run = server._run_index_build

    def tearDown(self):
        server._run_index_build = self.original_run
        server.db = self.original_db
        shutil.rmtree(self.temp_dir)

    def _make_job(self, job_id):
        idx = server.db.create_index("failure-state-test")
        server.db.create_index_job(job_id, idx, {}, [])
        return idx

    def test_unexpected_runtime_error_marks_job_failed(self):
        # The exact failure raised when the isolated build child crashes.
        idx = self._make_job("job-crash")

        def _boom(*a, **k):
            raise RuntimeError("Indexing process crashed (likely out of memory)")

        server._run_index_build = _boom
        server._run_index_build_job("job-crash")

        job = server.db.get_index_job("job-crash")
        self.assertEqual(job["status"], "failed")
        self.assertIn("crashed", job["error"])
        meta = server.db.get_index(idx)["metadata"]
        self.assertEqual(meta.get("status"), "failed")

    def test_cancellation_still_marks_job_cancelled(self):
        self._make_job("job-cancel")

        def _cancelled(*a, **k):
            raise RuntimeError("indexing_cancelled")

        server._run_index_build = _cancelled
        server._run_index_build_job("job-cancel")
        self.assertEqual(server.db.get_index_job("job-cancel")["status"], "cancelled")


class ApiKeyPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.original_db = server.db
        server.db = ChatDatabase(os.path.join(self.temp_dir, "chat_data.db"))
        self.original_thread = server.threading.Thread
        self.original_preflight = server._index_build_preflight

    def tearDown(self):
        server._index_build_preflight = self.original_preflight
        server.threading.Thread = self.original_thread
        server.db = self.original_db
        shutil.rmtree(self.temp_dir)

    def test_background_build_persists_scrubbed_options(self):
        from fastapi.testclient import TestClient

        idx = server.db.create_index("key-scrub-test")
        doc = os.path.join(self.temp_dir, "d.txt")
        open(doc, "w").write("content")
        server.db.add_document_to_index(idx, "d.txt", doc)

        server._index_build_preflight = lambda *a, **k: {"ok": True, "errors": []}
        server.threading.Thread = _RecordingThread

        client = TestClient(server.app)
        resp = client.post(
            f"/indexes/{idx}/build",
            json={"background": True, "enrichApiKey": "sk-SECRET", "chunkSize": 512},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        job_id = resp.json()["job_id"]

        # Persisted options must be scrubbed...
        stored = server.db.get_index_job(job_id)["options"]
        self.assertNotIn("enrichApiKey", stored)
        self.assertEqual(stored["chunkSize"], 512)
        # ...while the worker thread receives the key in memory.
        override = _RecordingThread.last["kwargs"]["options_override"]
        self.assertEqual(override["enrichApiKey"], "sk-SECRET")

    def test_legacy_rows_scrubbed_on_init(self):
        db_path = os.path.join(self.temp_dir, "legacy.db")
        db = ChatDatabase(db_path)
        idx = db.create_index("legacy-key-test")
        db.create_index_job("legacy-job", idx, {"chunkSize": 256, "enrichApiKey": "sk-OLD"}, [])

        ChatDatabase(db_path)  # simulates a server restart

        opts = json.loads(
            sqlite3.connect(db_path)
            .execute("SELECT options FROM index_jobs WHERE id='legacy-job'")
            .fetchone()[0]
        )
        self.assertNotIn("enrichApiKey", opts)
        self.assertEqual(opts["chunkSize"], 256)


class SessionIndexDeletionTests(unittest.TestCase):
    """FK enforcement: session_indexes has no ON DELETE CASCADE."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db = ChatDatabase(os.path.join(self.temp_dir, "chat_data.db"))

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_delete_session_with_linked_index(self):
        sid = self.db.create_session("linked", "m")
        idx = self.db.create_index("linked-idx")
        self.db.link_index_to_session(sid, idx)
        self.assertTrue(self.db.delete_session(sid))
        self.assertIsNone(self.db.get_session(sid))

    def test_cleanup_empty_sessions_with_linked_index(self):
        sid = self.db.create_session("empty-linked", "m")
        idx = self.db.create_index("cleanup-idx")
        self.db.link_index_to_session(sid, idx)
        # Must not raise IntegrityError (this crashed server startup)
        deleted = self.db.cleanup_empty_sessions()
        self.assertGreaterEqual(deleted, 1)


class _FakeEmbedder:
    dim = 8

    def create_embeddings(self, texts):
        rng = np.random.default_rng(abs(hash(str(texts))) % (2**31))
        v = rng.random((len(texts), self.dim)).astype(np.float32)
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)


class HybridRankingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import lancedb
        from rag_system.indexing.embedders import LanceDBManager
        from rag_system.retrieval.retrievers import MultiVectorRetriever

        cls.temp_dir = tempfile.mkdtemp()
        db = lancedb.connect(cls.temp_dir)
        rng = np.random.default_rng(1)
        rows = [
            {
                "chunk_id": f"c{i}",
                "text": t,
                "document_id": "doc",
                "chunk_index": i,
                "metadata": json.dumps({"original_text": t}),
                "vector": (rng.random(8) / 3).tolist(),
            }
            for i, t in enumerate(
                [
                    "the quick brown fox jumps over the lazy dog",
                    "machine learning models require training data",
                    "foxes are wild canines found worldwide",
                    "a fox and another fox met a third fox",
                    "completely unrelated text about cooking pasta",
                    "fox",
                ]
            )
        ]
        tbl = db.create_table("t", rows)
        tbl.create_fts_index("text", use_tantivy=False)
        cls.retriever = MultiVectorRetriever(LanceDBManager(cls.temp_dir), _FakeEmbedder())

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)

    def test_results_sorted_by_fused_score_and_trimmed(self):
        docs = self.retriever.retrieve("fox", table_name="t", k=4)
        scores = [d["score"] for d in docs]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertLessEqual(len(docs), 4)
        self.assertGreater(len(docs), 0)

    def test_fts_only_returns_pure_fts_results(self):
        docs = self.retriever.retrieve("fox", table_name="t", k=4, fts_only=True)
        self.assertTrue(docs)
        for d in docs:
            self.assertIsNotNone(d["bm25"])
            self.assertIsNone(d["_distance"])

    def test_vector_only_returns_pure_vector_results(self):
        docs = self.retriever.retrieve("fox", table_name="t", k=4, vector_only=True)
        self.assertTrue(docs)
        for d in docs:
            self.assertIsNone(d["bm25"])
            self.assertIsNotNone(d["_distance"])

    def test_fusion_override_changes_scores(self):
        base = self.retriever.retrieve("fox", table_name="t", k=6)
        vec = self.retriever.retrieve(
            "fox", table_name="t", k=6,
            fusion_override={"bm25_weight": 0.0, "vec_weight": 1.0},
        )
        self.assertNotEqual(
            {d["chunk_id"]: round(d["score"], 4) for d in base},
            {d["chunk_id"]: round(d["score"], 4) for d in vec},
        )


class _StubLLMClient:
    def generate_completion(self, *a, **k):
        return {"response": "stub overview"}

    def stream_completion(self, *a, **k):
        yield "stub"


def _make_pipeline(temp_dir):
    """Real IndexingPipeline against temp storage, fake embedder, stub LLM."""
    import copy
    from unittest.mock import patch

    from rag_system.main import PIPELINE_CONFIGS

    config = copy.deepcopy(PIPELINE_CONFIGS["default"])
    config["storage"]["lancedb_uri"] = os.path.join(temp_dir, "lancedb")
    config["storage"]["db_path"] = os.path.join(temp_dir, "lancedb")
    config["storage"]["text_table_name"] = "t_regress"
    config["overview_path"] = os.path.join(temp_dir, "ov.jsonl")
    # Unregistered fixture name skips dimension validation and uses the
    # tokenizer-free chunker path instead of attempting a network download.
    config["embedding_model_name"] = "fixture-hash-embedder"
    config["db_path"] = os.path.join(temp_dir, "state.sqlite3")
    config["index_store_path"] = os.path.join(temp_dir, "index_store")
    config["contextual_enricher"]["enabled"] = False
    config.setdefault("retrieval", {}).setdefault("late_chunking", {})["enabled"] = False

    import rag_system.pipelines.indexing_pipeline as ip_mod

    with patch.object(ip_mod, "select_embedder", lambda *a, **k: _FakeEmbedder()):
        pipeline = ip_mod.IndexingPipeline(
            config, _StubLLMClient(), {"host": "http://localhost:1", "generation_model": "x", "enrichment_model": "x"}
        )
    pipeline._start_persistent_worker = lambda: setattr(pipeline, "_worker", None)

    def _convert_in_process(file_path, document_id):
        result = ip_mod.convert_and_chunk_document(file_path, document_id, config)
        if result.get("error"):
            raise RuntimeError(result["error"])
        return result.get("chunks", [])

    pipeline._convert_and_chunk_file = _convert_in_process
    return pipeline, config


class IndexingFailureTests(unittest.TestCase):
    """Slow tests: exercise the real pipeline against temp storage."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_all_files_failed_raises(self):
        pipeline, _ = _make_pipeline(self.temp_dir)
        bad = os.path.join(self.temp_dir, "corrupt.pdf")
        open(bad, "wb").write(b"this is not a pdf")
        with self.assertRaises(RuntimeError) as ctx:
            pipeline.run([bad], index_id="regress-fail", force_reindex=True)
        self.assertIn("failed", str(ctx.exception).lower())

    def test_all_unchanged_validates_existing_table(self):
        pipeline, config = _make_pipeline(self.temp_dir)
        doc = os.path.join(self.temp_dir, "doc.txt")
        open(doc, "w").write("The quarterly revenue was 4.2 million dollars. " * 40)

        result = pipeline.run([doc], index_id="regress-ok", force_reindex=True)
        self.assertGreater(result.get("chunks_generated", 0), 0)

        # Second run: nothing changed → previously returned success blindly.
        result = pipeline.run([doc], index_id="regress-ok")
        self.assertEqual(result.get("files_processed"), 0)

        # Drop the table out from under it: "all unchanged" must now fail.
        import lancedb

        db = lancedb.connect(config["storage"]["lancedb_uri"])
        db.drop_table("t_regress")
        pipeline2, _ = _make_pipeline(self.temp_dir)
        with self.assertRaises(RuntimeError):
            pipeline2.run([doc], index_id="regress-ok")


class MetadataFilterTests(unittest.TestCase):
    """Typed metadata schemas: strict validation, safe SQL compilation."""

    SCHEMA = [
        {"name": "project", "type": "string"},
        {"name": "year", "type": "integer"},
        {"name": "confidential", "type": "boolean"},
    ]

    def test_schema_validation(self):
        from rag_system.metadata_filters import validate_schema
        self.assertEqual(validate_schema(self.SCHEMA), [])
        self.assertTrue(validate_schema([{"name": "vector", "type": "string"}]))  # reserved
        self.assertTrue(validate_schema([{"name": "Bad Name", "type": "string"}]))
        self.assertTrue(validate_schema([{"name": "x", "type": "datetime2"}]))
        self.assertTrue(validate_schema([{"name": "x", "type": "string", "required": "yes"}]))
        self.assertTrue(validate_schema([]))

    def test_document_metadata_strict(self):
        from rag_system.metadata_filters import FilterError, validate_document_metadata
        clean = validate_document_metadata(self.SCHEMA, {"project": "alpha", "year": "2024"})
        self.assertEqual(clean, {"project": "alpha", "year": 2024, "confidential": None})
        with self.assertRaises(FilterError):  # unknown field (typo) rejected
            validate_document_metadata(self.SCHEMA, {"projet": "alpha"})
        with self.assertRaises(FilterError):  # type mismatch
            validate_document_metadata(self.SCHEMA, {"year": "not-a-year"})

    def test_filter_compilation_and_escaping(self):
        from rag_system.metadata_filters import FilterError, compile_filters
        self.assertEqual(compile_filters(self.SCHEMA, {"project": "alpha"}), "meta_project = 'alpha'")
        self.assertEqual(
            compile_filters(self.SCHEMA, {"year": {">=": 2020, "<": 2024}}),
            "meta_year >= 2020 AND meta_year < 2024",
        )
        self.assertEqual(
            compile_filters(self.SCHEMA, {"project": ["a", "b"]}),
            "meta_project IN ('a', 'b')",
        )
        # SQL injection attempt is escaped, never executed raw
        self.assertEqual(
            compile_filters(self.SCHEMA, {"project": "x' OR 1=1 --"}),
            "meta_project = 'x'' OR 1=1 --'",
        )
        with self.assertRaises(FilterError):
            compile_filters(self.SCHEMA, {"nope": 1})        # unknown field
        with self.assertRaises(FilterError):
            compile_filters(self.SCHEMA, {"project": {">": "a"}})  # bad op for type
        with self.assertRaises(FilterError):
            compile_filters(None, {"project": "alpha"})      # no schema → loud error
        self.assertIsNone(compile_filters(self.SCHEMA, None))

    def test_flatten_columns(self):
        from rag_system.metadata_filters import flatten_columns
        cols = flatten_columns(self.SCHEMA, {"project": "alpha"})
        self.assertEqual(cols, {"meta_project": "alpha", "meta_year": None, "meta_confidential": None})
        self.assertEqual(flatten_columns(None, {"x": 1}), {})

    def test_vector_index_uses_declared_types_when_first_value_is_none(self):
        import pyarrow as pa
        from rag_system.indexing.embedders import LanceDBManager, VectorIndexer

        temp_dir = tempfile.mkdtemp()
        try:
            indexer = VectorIndexer(LanceDBManager(temp_dir))
            schema = [
                {"name": "year", "type": "integer"},
                {"name": "score", "type": "float"},
                {"name": "approved", "type": "boolean"},
            ]
            first = {
                "chunk_id": "a_0",
                "text": "first",
                "metadata": {"document_id": "a", "chunk_index": 0},
                "_meta_columns": {"meta_year": None, "meta_score": None, "meta_approved": None},
            }
            second = {
                "chunk_id": "b_0",
                "text": "second",
                "metadata": {"document_id": "b", "chunk_index": 0},
                "_meta_columns": {"meta_year": 2026, "meta_score": 0.75, "meta_approved": True},
            }
            indexer.index("typed", [first], [np.ones(8, dtype=np.float32)], metadata_schema=schema)
            indexer.index("typed", [second], [np.ones(8, dtype=np.float32)], metadata_schema=schema)
            table_schema = indexer.db_manager.get_table("typed").schema
            self.assertEqual(table_schema.field("meta_year").type, pa.int64())
            self.assertEqual(table_schema.field("meta_score").type, pa.float64())
            self.assertEqual(table_schema.field("meta_approved").type, pa.bool_())
            self.assertEqual(indexer.db_manager.get_table("typed").count_rows(), 2)
        finally:
            shutil.rmtree(temp_dir)

    def test_chunk_cache_key_includes_document_identity(self):
        from rag_system.pipelines.indexing_pipeline import IndexingPipeline

        pipeline = IndexingPipeline.__new__(IndexingPipeline)
        pipeline.config = {
            "chunker_mode": "docling",
            "embedding_model_name": "fixture",
            "chunking": {"chunk_size": 512, "chunk_overlap": 64},
        }
        first = pipeline._chunk_cache_key("/tmp/first/report.txt", file_hash="same-content")
        second = pipeline._chunk_cache_key("/tmp/second/copy.txt", file_hash="same-content")
        self.assertNotEqual(first, second)


class MultiCollectionRetrievalTests(unittest.TestCase):
    """Multi-collection retrieval: per-collection search, RRF merge when no
    reranker, per-source context expansion, index-named attribution."""

    @classmethod
    def setUpClass(cls):
        import lancedb
        from rag_system.pipelines.retrieval_pipeline import RetrievalPipeline
        from rag_system.indexing.embedders import LanceDBManager
        from rag_system.retrieval.retrievers import MultiVectorRetriever

        cls.temp_dir = tempfile.mkdtemp()
        db = lancedb.connect(cls.temp_dir)
        rng = np.random.default_rng(7)

        def make_table(name, texts):
            rows = [{
                "chunk_id": f"{name}-c{i}", "text": t, "document_id": f"{name}-doc",
                "chunk_index": i, "metadata": json.dumps({"original_text": t}),
                "vector": (rng.random(8) / 3).tolist(),
            } for i, t in enumerate(texts)]
            tbl = db.create_table(name, rows)
            tbl.create_fts_index("text", use_tantivy=False)

        make_table("table_a", ["alpha mine crusher report", "alpha budget is 12 million", "alpha schedule details"])
        make_table("table_b", ["beta plant conveyor study", "beta crusher throughput data", "beta staffing plan"])

        class _StubLLM:
            def stream_completion(self, *a, **k):
                yield "stub answer"
            def generate_completion(self, *a, **k):
                return {"response": "stub"}

        config = {
            "storage": {"text_table_name": "table_a", "lancedb_uri": cls.temp_dir, "db_path": cls.temp_dir},
            "retrieval": {},
            "reranker": {"enabled": False},
            "retrieval_k": 3,
        }
        cls.pipeline = RetrievalPipeline(config, _StubLLM(), {"generation_model": "stub"})

        manager = LanceDBManager(cls.temp_dir)
        retriever = MultiVectorRetriever(manager, _FakeEmbedder())
        # Same fake retriever serves both "models"; the point under test is
        # the per-collection loop, merge, and attribution — not embedding.
        cls.pipeline._get_retriever_for_model = lambda model: retriever
        cls.pipeline._get_ai_reranker = lambda: None
        cls.pipeline._get_reranker = lambda: None

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)

    def test_multi_collection_search_merges_both_indexes(self):
        result = self.pipeline.run(
            "crusher",
            window_size_override=0,
            collections=[
                {"table_name": "table_a", "embedding_model": "m1", "index_name": "Alpha Mine"},
                {"table_name": "table_b", "embedding_model": "m2", "index_name": "Beta Plant"},
            ],
        )
        docs = result["source_documents"]
        self.assertTrue(docs)
        indexes_seen = {d.get("index_name") for d in docs}
        self.assertEqual(indexes_seen, {"Alpha Mine", "Beta Plant"})
        # Internal bookkeeping must not leak into the response
        for d in docs:
            self.assertNotIn("_source_table", d)
            self.assertNotIn("_collection_rank", d)

    def test_single_collection_path_unchanged(self):
        result = self.pipeline.run("conveyor", table_name="table_b", window_size_override=0)
        docs = result["source_documents"]
        self.assertTrue(docs)
        self.assertTrue(all(d["document_id"] == "table_b-doc" for d in docs))

    def test_collections_capped(self):
        from rag_system.pipelines.retrieval_pipeline import MAX_COLLECTIONS
        many = [{"table_name": "table_a", "embedding_model": None, "index_name": f"i{n}"} for n in range(9)]
        # Must not raise; only the first MAX_COLLECTIONS entries are searched
        result = self.pipeline.run("alpha", window_size_override=0, collections=many)
        self.assertTrue(result["source_documents"])
        self.assertLessEqual(MAX_COLLECTIONS, 5)

    def test_same_chunk_id_from_two_indexes_is_not_deduplicated(self):
        class _CollisionRetriever:
            def retrieve(self, text_query, table_name, **kwargs):
                return [{
                    "chunk_id": "report.txt_0",
                    "text": f"result from {table_name}",
                    "document_id": "report.txt",
                    "chunk_index": 0,
                    "metadata": {},
                    "score": 1.0,
                }]

        original = self.pipeline._get_retriever_for_model
        self.pipeline._get_retriever_for_model = lambda _model: _CollisionRetriever()
        try:
            result = self.pipeline.run(
                "report",
                window_size_override=0,
                collections=[
                    {"index_id": "index-a", "table_name": "table_a", "embedding_model": "m1"},
                    {"index_id": "index-b", "table_name": "table_b", "embedding_model": "m2"},
                ],
            )
        finally:
            self.pipeline._get_retriever_for_model = original
        self.assertEqual(
            {(d["index_id"], d["chunk_id"]) for d in result["source_documents"]},
            {("index-a", "report.txt_0"), ("index-b", "report.txt_0")},
        )

    def test_each_collection_uses_its_fusion_config(self):
        seen = {}

        class _RecordingRetriever:
            def retrieve(self, text_query, table_name, **kwargs):
                seen[table_name] = kwargs.get("fusion_override")
                return []

        original = self.pipeline._get_retriever_for_model
        self.pipeline._get_retriever_for_model = lambda _model: _RecordingRetriever()
        try:
            self.pipeline.run(
                "report",
                window_size_override=0,
                collections=[
                    {
                        "table_name": "table_a",
                        "embedding_model": "m1",
                        "fusion_config": {"bm25_weight": 0.8, "vec_weight": 0.2},
                    },
                    {
                        "table_name": "table_b",
                        "embedding_model": "m2",
                        "fusion_config": {"bm25_weight": 0.1, "vec_weight": 0.9},
                    },
                ],
            )
        finally:
            self.pipeline._get_retriever_for_model = original
        self.assertEqual(seen["table_a"]["bm25_weight"], 0.8)
        self.assertEqual(seen["table_b"]["bm25_weight"], 0.1)


class MultiIndexSelectionTests(unittest.TestCase):
    """Backend (table) and RAG server (embedding model/fusion) must resolve
    the same index for a session — they diverged once (last vs first)."""

    def test_active_index_is_last_linked(self):
        from rag_system.index_selection import select_active_index_id

        self.assertEqual(select_active_index_id(["a", "b", "c"]), "c")
        self.assertEqual(select_active_index_id(["only"]), "only")
        self.assertIsNone(select_active_index_id([]))
        self.assertIsNone(select_active_index_id(None))

    def test_servers_have_no_private_index_picks(self):
        # Tripwire: raw idx_ids[0]/idx_ids[-1] indexing in either server is
        # exactly how the two sides diverged. All picks must go through
        # rag_system.index_selection.select_active_index_id.
        root = os.path.dirname(os.path.abspath(__file__))
        for rel in ("backend/server.py", os.path.join("rag_system", "api_server.py")):
            src = open(os.path.join(root, rel), encoding="utf-8").read()
            self.assertNotIn("idx_ids[0]", src, f"private index pick in {rel}")
            self.assertNotIn("idx_ids[-1]", src, f"private index pick in {rel}")
            self.assertIn("select_active_index_id", src, f"{rel} no longer uses the shared helper")


if __name__ == "__main__":
    unittest.main()

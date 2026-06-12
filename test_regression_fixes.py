"""Regression tests for failure modes fixed after external review.

Covers:
- build jobs reaching a terminal state on unexpected errors (stuck-job fix)
- enrichApiKey never persisted with job options (+ legacy-row scrub)
- session deletion with linked indexes (FK enforcement regression)
- hybrid retrieval ranking: scored before truncation, sorted, single-leg
  modes, per-call fusion override
- index builds failing when every file fails
- "all files unchanged" builds validating the existing table

Not covered here (needs the live RAG server): the last-linked-index
consistency between backend table choice and api_server embedding model.
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
    # Unregistered model name: dimension validation is skipped, matching the
    # 8-dim fake embedder instead of the real model's registered 1024
    config["embedding_model_name"] = "fake-test-model"
    config["contextual_enricher"]["enabled"] = False
    config.setdefault("retrieval", {}).setdefault("late_chunking", {})["enabled"] = False

    import rag_system.pipelines.indexing_pipeline as ip_mod

    with patch.object(ip_mod, "select_embedder", lambda *a, **k: _FakeEmbedder()):
        pipeline = ip_mod.IndexingPipeline(
            config, _StubLLMClient(), {"host": "http://localhost:1", "generation_model": "x", "enrichment_model": "x"}
        )
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


if __name__ == "__main__":
    unittest.main()

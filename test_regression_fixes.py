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


class McpServerTests(unittest.TestCase):
    """MCP stdio server protocol contract — server-free (HTTP layer mocked)."""

    def setUp(self):
        from unittest.mock import patch
        import rag_system.mcp_server as mcp

        self.mcp = mcp
        self.sent = []
        self._send_patch = patch.object(mcp, "_send", self.sent.append)
        self._send_patch.start()

        def fake_http(method, url, payload=None, timeout=600):
            if url.endswith("/indexes"):
                return {"indexes": [{
                    "id": "abc12345-0000", "name": "Demo",
                    "vector_table_name": "text_pages_abc",
                    "documents": [{"filename": "a.pdf"}],
                    "metadata": {"status": "functional",
                                 "metadata_schema": [{"name": "project", "type": "string"}]},
                }]}
            if url.endswith("/chat"):
                return {"answer": "The budget is 5M.",
                        "source_documents": [{"document_id": "3f3d4465-7491-43f8-8dda-874556dd8d1a_a.pdf", "chunk_id": "c1"}]}
            return {}

        self._http_patch = patch.object(mcp, "_http_json", fake_http)
        self._http_patch.start()

    def tearDown(self):
        self._send_patch.stop()
        self._http_patch.stop()

    def _last(self):
        return self.sent[-1]

    def test_initialize_handshake(self):
        self.mcp._handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        r = self._last()["result"]
        self.assertEqual(r["protocolVersion"], "2025-03-26")
        self.assertIn("tools", r["capabilities"])

    def test_initialized_notification_is_silent(self):
        self.mcp._handle({"jsonrpc": "2.0", "method": "notifications/initialized"})
        self.assertEqual(self.sent, [])

    def test_tools_list(self):
        self.mcp._handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        names = [t["name"] for t in self._last()["result"]["tools"]]
        self.assertEqual(set(names), {"list_indexes", "ask_index"})

    def test_ask_index_returns_answer_with_sources(self):
        self.mcp._handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
            "name": "ask_index", "arguments": {"index": "abc", "question": "budget?"}}})
        r = self._last()["result"]
        self.assertFalse(r["isError"])
        text = r["content"][0]["text"]
        self.assertIn("5M", text)
        self.assertIn("a.pdf", text)        # source filename, uuid prefix stripped
        self.assertNotIn("3f3d4465", text)  # the uuid4 prefix is gone

    def test_unknown_tool_is_jsonrpc_error(self):
        self.mcp._handle({"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {
            "name": "nope", "arguments": {}}})
        self.assertEqual(self._last()["error"]["code"], -32601)

    def test_unresolvable_index_is_tool_error_not_crash(self):
        self.mcp._handle({"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {
            "name": "ask_index", "arguments": {"index": "does-not-exist", "question": "q"}}})
        r = self._last()["result"]
        self.assertTrue(r["isError"])  # surfaced as isError, never an unhandled crash


class AgenticPlannerTests(unittest.TestCase):
    """Opt-in plan-and-execute helpers: complexity gate, evidence check, retry."""

    class _StubLLM:
        def __init__(self, response="{}"):
            self.response = response
            self.calls = 0

        def generate_completion(self, *a, **k):
            self.calls += 1
            return {"response": self.response}

    def test_simple_query_skips_llm(self):
        from rag_system.agent import agentic
        llm = self._StubLLM('{"complex": true}')
        # Short, no conjunction markers → decided simple WITHOUT an LLM call
        self.assertFalse(agentic.assess_complexity(llm, "m", "what is the budget"))
        self.assertEqual(llm.calls, 0)

    def test_complex_marker_consults_llm(self):
        from rag_system.agent import agentic
        llm = self._StubLLM('{"complex": true}')
        q = "compare the budget of project A and project B and their schedules"
        self.assertTrue(agentic.assess_complexity(llm, "m", q))
        self.assertEqual(llm.calls, 1)

    def test_complexity_defaults_false_on_bad_json(self):
        from rag_system.agent import agentic
        llm = self._StubLLM("not json")
        q = "compare alpha and beta and gamma across every dimension"
        self.assertFalse(agentic.assess_complexity(llm, "m", q))

    def test_evidence_thinness(self):
        from rag_system.agent import agentic
        self.assertTrue(agentic.is_evidence_thin(None))
        self.assertTrue(agentic.is_evidence_thin({"source_documents": []}))
        self.assertTrue(agentic.is_evidence_thin(
            {"source_documents": [{"chunk_id": "c1"}], "answer": "I could not find that information in the provided documents."}
        ))
        self.assertFalse(agentic.is_evidence_thin(
            {"source_documents": [{"chunk_id": "c1"}], "answer": "The budget is 5 million."}
        ))

    def test_reformulate_returns_new_query_or_none(self):
        from rag_system.agent import agentic
        llm = self._StubLLM('{"query": "broader search terms"}')
        self.assertEqual(agentic.reformulate_task(llm, "m", "obscure phrasing"), "broader search terms")
        # Identical reformulation → None (no point retrying the same query)
        same = self._StubLLM('{"query": "same task"}')
        self.assertIsNone(agentic.reformulate_task(same, "m", "same task"))
        # Bad JSON → None
        self.assertIsNone(agentic.reformulate_task(self._StubLLM("nope"), "m", "task"))


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
        cls.pipeline._get_ai_reranker = lambda enabled_override=None: None
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
        configured_table = self.pipeline.storage_config["text_table_name"]
        result = self.pipeline.run("conveyor", table_name="table_b", window_size_override=0)
        docs = result["source_documents"]
        self.assertTrue(docs)
        self.assertTrue(all(d["document_id"] == "table_b-doc" for d in docs))
        self.assertEqual(self.pipeline.storage_config["text_table_name"], configured_table)

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


class RequestScopedConcurrencyTests(unittest.TestCase):
    def test_concurrent_synthesis_keeps_generation_models_isolated(self):
        import concurrent.futures
        import threading
        from rag_system.pipelines.retrieval_pipeline import RetrievalPipeline

        barrier = threading.Barrier(2)

        class _RecordingLLM:
            def stream_completion(self, model, prompt):
                barrier.wait(timeout=2)
                yield model

        config = {
            "storage": {"text_table_name": "default", "db_path": tempfile.mkdtemp()},
            "retrieval": {},
            "reranker": {"enabled": False},
        }
        self.addCleanup(shutil.rmtree, config["storage"]["db_path"])
        ollama_config = {"generation_model": "default-model"}
        pipeline = RetrievalPipeline(config, _RecordingLLM(), ollama_config)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(
                pipeline._synthesize_final_answer,
                "q1",
                "facts",
                generation_model="model-a",
            )
            second = executor.submit(
                pipeline._synthesize_final_answer,
                "q2",
                "facts",
                generation_model="model-b",
            )

        self.assertEqual({first.result(), second.result()}, {"model-a", "model-b"})
        self.assertEqual(ollama_config["generation_model"], "default-model")

    def test_api_chat_path_has_no_global_serialization_or_config_writes(self):
        # The in-process chat path (chat_runtime.execute_chat) replaced the
        # retired standalone 8001 api_server. It must stay lock-free and never
        # mutate shared config — request scoping is what let the global lock go.
        root = os.path.dirname(os.path.abspath(__file__))
        source = open(
            os.path.join(root, "rag_system", "chat_runtime.py"), encoding="utf-8"
        ).read()

        self.assertNotIn("_rag_agent_lock", source)
        self.assertNotIn("_apply_index_embedding_model", source)
        self.assertNotIn("ollama_config['generation_model'] =", source)
        self.assertNotIn('setdefault("provence", {})["enabled"]', source)


class MultiIndexSelectionTests(unittest.TestCase):
    """Backend (table) and RAG runtime (embedding model/fusion) must resolve
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
        for rel in ("backend/server.py", os.path.join("rag_system", "chat_runtime.py")):
            src = open(os.path.join(root, rel), encoding="utf-8").read()
            self.assertNotIn("idx_ids[0]", src, f"private index pick in {rel}")
            self.assertNotIn("idx_ids[-1]", src, f"private index pick in {rel}")
            self.assertIn("select_active_index_id", src, f"{rel} no longer uses the shared helper")


class _FakeTimingPipeline:
    """Emits the pipeline's real stage events through the callback it's given."""

    def run(self, query, *, event_callback=None, **kwargs):
        if event_callback:
            event_callback("retrieval_started", {})
            event_callback("retrieval_done", {"count": 2})
            event_callback("rerank_started", {"count": 2})
            event_callback("rerank_done", {"count": 2})
            event_callback("token", {"text": "hello"})
            event_callback("token", {"text": " there"})
        return {"answer": "hello there", "source_documents": [{"document_id": "d"}]}


class _FakeTimingAgent:
    def __init__(self):
        self.ollama_config = {"generation_model": "m"}
        self.retrieval_pipeline = _FakeTimingPipeline()

    def get_overviews_for_indexes(self, _ids):
        return []


class _FakeTimingDB:
    def list_indexes(self):
        return []

    def get_indexes_for_session(self, _sid):
        return []

    def get_index(self, _iid):
        return None


class StageTimingsTests(unittest.TestCase):
    """Opt-in per-stage timing + TTFT (LOCALGPT_TIMINGS). Off => no behavior
    change; on => timings_ms/ttft_ms ride along on the chat response."""

    def test_timings_enabled_parses_flag(self):
        from unittest.mock import patch

        from rag_system.utils.logging_utils import timings_enabled

        cases = {"1": True, "true": True, "YES": True, "on": True,
                 "0": False, "": False, "off": False}
        for value, want in cases.items():
            with patch.dict(os.environ, {"LOCALGPT_TIMINGS": value}):
                self.assertEqual(timings_enabled(), want, value)
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(timings_enabled())

    def test_stage_timings_observes_streaming_sequence(self):
        from rag_system.utils.logging_utils import StageTimings

        timer = StageTimings()
        for event in ("retrieval_started", "retrieval_done", "rerank_started",
                      "rerank_done", "prune_started", "prune_done"):
            timer.observe(event, {})
        timer.observe("token", {"text": "hi"})
        timer.observe("token", {"text": " more"})  # only the first sets TTFT

        snapshot = timer.as_dict()
        self.assertLessEqual(
            {"retrieval", "rerank", "prune", "total"}, set(snapshot["timings_ms"])
        )
        self.assertIn("ttft_ms", snapshot)
        self.assertGreaterEqual(snapshot["ttft_ms"], 0.0)

    def test_stage_timings_includes_generation(self):
        from rag_system.utils.logging_utils import StageTimings

        timer = StageTimings()
        timer.observe("generation_started", {})
        timer.observe("generation_done", {"chars": 120})
        snapshot = timer.as_dict()
        self.assertIn("generation", snapshot["timings_ms"])
        self.assertGreaterEqual(snapshot["timings_ms"]["generation"], 0.0)

    def test_stage_timings_accumulate_across_rounds(self):
        from rag_system.utils.logging_utils import StageTimings

        timer = StageTimings()
        timer.observe("retrieval_started", {})
        timer.observe("retrieval_done", {})
        after_first = timer.stages_ms["retrieval"]
        timer.observe("retrieval_started", {})
        timer.observe("retrieval_done", {})
        # Second round accumulates into the same bucket rather than overwriting.
        self.assertGreaterEqual(timer.stages_ms["retrieval"], after_first)
        self.assertEqual(list(timer.stages_ms), ["retrieval"])

    def test_stage_timings_omit_ttft_without_token(self):
        from rag_system.utils.logging_utils import StageTimings

        timer = StageTimings()
        timer.observe("retrieval_started", {})
        timer.observe("retrieval_done", {})
        snapshot = timer.as_dict()
        self.assertNotIn("ttft_ms", snapshot)
        self.assertIn("total", snapshot["timings_ms"])

    def test_execute_chat_attaches_timings_when_enabled(self):
        from unittest.mock import patch

        from rag_system import chat_runtime

        seen = []
        with patch.dict(os.environ, {"LOCALGPT_TIMINGS": "1"}):
            result = chat_runtime.execute_chat(
                _FakeTimingAgent(),
                _FakeTimingDB(),
                {"query": "q", "force_rag": True},
                event_callback=lambda event_type, _p: seen.append(event_type),
            )
        self.assertIn("timings_ms", result)
        self.assertLessEqual(
            {"retrieval", "rerank", "total"}, set(result["timings_ms"])
        )
        self.assertIn("ttft_ms", result)
        # Observer is passive: the original callback still saw every event.
        self.assertIn("retrieval_started", seen)
        self.assertIn("token", seen)

    def test_execute_chat_no_timings_when_disabled(self):
        from unittest.mock import patch

        from rag_system import chat_runtime

        with patch.dict(os.environ, {"LOCALGPT_TIMINGS": "0"}):
            result = chat_runtime.execute_chat(
                _FakeTimingAgent(),
                _FakeTimingDB(),
                {"query": "q", "force_rag": True},
                event_callback=lambda _e, _p: None,
            )
        self.assertNotIn("timings_ms", result)
        self.assertNotIn("ttft_ms", result)

    def test_execute_chat_times_stages_without_client_stream(self):
        from unittest.mock import patch

        from rag_system import chat_runtime

        # No client event_callback, but the timing sink still observes the
        # pipeline's stage events — so non-streaming /rag/chat gets the full
        # breakdown (and ttft), not just total. Behavior is unchanged because
        # generation streams internally regardless.
        with patch.dict(os.environ, {"LOCALGPT_TIMINGS": "1"}):
            result = chat_runtime.execute_chat(
                _FakeTimingAgent(),
                _FakeTimingDB(),
                {"query": "q", "force_rag": True},
                event_callback=None,
            )
        self.assertLessEqual(
            {"retrieval", "rerank", "total"}, set(result["timings_ms"])
        )
        self.assertIn("ttft_ms", result)


class _ReflectFakePipeline:
    """Scripted RetrievalPipeline: returns queued answers/sources per run(),
    records calls, and emits a stage + (suppressible) token event."""

    def __init__(self, answers, sources, synth_answer="REGENERATED"):
        self.ollama_client = self
        self._answers = list(answers)
        self._sources = list(sources)
        self._synth_answer = synth_answer
        self.run_queries = []
        self.synth_queries = []

    def run(self, query, *, event_callback=None, **_kwargs):
        self.run_queries.append(query)
        if event_callback:
            event_callback("retrieval_started", {})
            event_callback("token", {"text": "INTERMEDIATE"})  # must be suppressed
            event_callback("retrieval_done", {"count": 1})
        answer = self._answers.pop(0) if self._answers else "final"
        sources = self._sources.pop(0) if self._sources else [{"text": "t"}]
        return {"answer": answer, "source_documents": sources}

    def _synthesize_final_answer(
        self, query, facts, *, event_callback=None, generation_model=None
    ):
        self.synth_queries.append(query)
        return self._synth_answer

    def generate_completion(self, model, prompt, format=None, **kwargs):  # rewrite
        return {"response": json.dumps({"query": "REWRITTEN"})}


class _ReflectFakeVerifier:
    def __init__(self, rel, ground):
        self._rel = list(rel)
        self._ground = list(ground)

    def score_context_relevance(self, query, context, model=None):
        return self._rel.pop(0) if self._rel else 2

    def score_response_groundedness(self, query, context, answer, model=None):
        return self._ground.pop(0) if self._ground else 2


class _ReflectFakeAgent:
    def __init__(self, pipeline, verifier):
        self.ollama_config = {"generation_model": "m"}
        self.retrieval_pipeline = pipeline
        self.verifier = verifier

    def get_overviews_for_indexes(self, _ids):
        return []

    def run(self, query, **_kwargs):  # non-reflective default path
        return {"answer": "agent-path", "source_documents": []}


def _one_source():
    return [{"text": "t", "document_id": "d"}]


class ReflectionLoopTests(unittest.TestCase):
    """Two-axis (relevance + groundedness) bounded-retry self-reflection."""

    @staticmethod
    def _cfg(**override):
        base = {
            "model": "m",
            "generation_model": "m",
            "max_loops": 2,
            "relevance_threshold": 1,
            "groundedness_threshold": 1,
        }
        base.update(override)
        return base

    def test_accepts_first_answer_when_scores_pass(self):
        from rag_system.agent import reflection

        pipe = _ReflectFakePipeline(["A"], [_one_source()])
        verifier = _ReflectFakeVerifier(rel=[2], ground=[2])
        out = reflection.reflective_run(
            pipe, verifier, "q", run_kwargs={}, event_callback=None, cfg=self._cfg()
        )
        self.assertEqual(out["answer"], "A")
        self.assertEqual(out["reflection"]["rounds"], 0)
        self.assertEqual(pipe.run_queries, ["q"])  # one retrieval, original query
        self.assertEqual(pipe.synth_queries, [])

    def test_rewrites_and_reretrieves_on_low_relevance(self):
        from rag_system.agent import reflection

        pipe = _ReflectFakePipeline(["A", "B"], [_one_source(), _one_source()])
        verifier = _ReflectFakeVerifier(rel=[0, 2], ground=[2, 2])
        out = reflection.reflective_run(
            pipe, verifier, "q", run_kwargs={}, event_callback=None, cfg=self._cfg()
        )
        self.assertEqual(out["reflection"]["rounds"], 1)
        self.assertEqual(pipe.run_queries, ["q", "REWRITTEN"])  # re-retrieve broadened
        self.assertEqual(pipe.synth_queries, [])
        self.assertEqual(out["answer"], "B")

    def test_regenerates_on_low_groundedness_without_reretrieval(self):
        from rag_system.agent import reflection

        pipe = _ReflectFakePipeline(["A"], [_one_source()])
        verifier = _ReflectFakeVerifier(rel=[2, 2], ground=[0, 2])
        out = reflection.reflective_run(
            pipe, verifier, "q", run_kwargs={}, event_callback=None, cfg=self._cfg()
        )
        self.assertEqual(out["reflection"]["rounds"], 1)
        self.assertEqual(pipe.run_queries, ["q"])  # no re-retrieval
        self.assertEqual(len(pipe.synth_queries), 1)  # regenerated on same context
        self.assertEqual(out["answer"], "REGENERATED")

    def test_reports_converged_flag(self):
        from rag_system.agent import reflection

        # accepted first try -> converged
        pipe = _ReflectFakePipeline(["A"], [_one_source()])
        out = reflection.reflective_run(
            pipe,
            _ReflectFakeVerifier(rel=[2], ground=[2]),
            "q",
            run_kwargs={},
            event_callback=None,
            cfg=self._cfg(),
        )
        self.assertTrue(out["reflection"]["converged"])

        # never passes -> loop exhausts -> not converged
        pipe2 = _ReflectFakePipeline(["A", "B", "C"], [_one_source()] * 3)
        out2 = reflection.reflective_run(
            pipe2,
            _ReflectFakeVerifier(rel=[0, 0, 0], ground=[2, 2, 2]),
            "q",
            run_kwargs={},
            event_callback=None,
            cfg=self._cfg(max_loops=2),
        )
        self.assertFalse(out2["reflection"]["converged"])

    def test_respects_max_loops(self):
        from rag_system.agent import reflection

        pipe = _ReflectFakePipeline(["A", "B", "C"], [_one_source()] * 3)
        verifier = _ReflectFakeVerifier(rel=[0, 0, 0], ground=[2, 2, 2])
        out = reflection.reflective_run(
            pipe,
            verifier,
            "q",
            run_kwargs={},
            event_callback=None,
            cfg=self._cfg(max_loops=2),
        )
        self.assertEqual(out["reflection"]["rounds"], 2)
        self.assertEqual(pipe.run_queries, ["q", "REWRITTEN", "REWRITTEN"])

    def test_empty_regeneration_falls_back_to_prior_answer(self):
        # Noisy scoring can keep firing the groundedness branch, and a
        # regeneration may come back empty. The loop must never return an empty
        # answer when an earlier round produced a real one.
        from rag_system.agent import reflection

        pipe = _ReflectFakePipeline(["GOOD"], [_one_source()], synth_answer="")
        verifier = _ReflectFakeVerifier(rel=[2], ground=[0])
        out = reflection.reflective_run(
            pipe,
            verifier,
            "q",
            run_kwargs={},
            event_callback=None,
            cfg=self._cfg(max_loops=1),
        )
        self.assertEqual(out["answer"], "GOOD")  # not the empty regeneration
        self.assertEqual(out["reflection"]["rounds"], 1)

    def test_suppresses_intermediate_tokens_and_streams_only_final(self):
        from rag_system.agent import reflection

        events = []
        pipe = _ReflectFakePipeline(["FINAL"], [_one_source()])
        verifier = _ReflectFakeVerifier(rel=[2], ground=[2])
        reflection.reflective_run(
            pipe,
            verifier,
            "q",
            run_kwargs={},
            event_callback=lambda et, p: events.append((et, p)),
            cfg=self._cfg(),
        )
        types = [et for et, _ in events]
        self.assertIn("retrieval_started", types)  # stage events forwarded
        self.assertIn("retrieval_done", types)
        token_texts = [p["text"] for et, p in events if et == "token"]
        self.assertEqual(token_texts, ["FINAL"])  # intermediate token swallowed


class ReflectionScoringTests(unittest.TestCase):
    def test_scores_use_deterministic_thinking_off_decoding(self):
        # Run-to-run score drift was the noise source; scoring must call the LLM
        # greedily (temperature=0) with thinking disabled.
        from rag_system.agent.verifier import Verifier

        llm = _RewriteLLM({"response": json.dumps({"score": 2})})
        verifier = Verifier(llm, "judge-model")

        self.assertEqual(verifier.score_context_relevance("q", "ctx"), 2)
        self.assertEqual(llm.last_kwargs.get("temperature"), 0.0)
        self.assertIs(llm.last_kwargs.get("enable_thinking"), False)

        self.assertEqual(verifier.score_response_groundedness("q", "ctx", "ans"), 2)
        self.assertEqual(llm.last_kwargs.get("temperature"), 0.0)
        self.assertIs(llm.last_kwargs.get("enable_thinking"), False)

    def test_score_clamps_and_failsafe(self):
        from rag_system.agent.verifier import Verifier

        # out-of-range clamps to 0..2
        hi = Verifier(_RewriteLLM({"response": json.dumps({"score": 9})}), "m")
        self.assertEqual(hi.score_context_relevance("q", "c"), 2)
        # unparseable -> fail-safe 0
        bad = Verifier(_RewriteLLM({"response": "not json"}), "m")
        self.assertEqual(bad.score_response_groundedness("q", "c", "a"), 0)


class ReflectionConfigTests(unittest.TestCase):
    def test_defaults_and_overrides(self):
        from rag_system.agent import reflection

        default = reflection.parse_config({}, "gen-model")
        self.assertFalse(default["enabled"])
        self.assertEqual(default["max_loops"], 2)
        self.assertEqual(default["relevance_threshold"], 1)
        self.assertEqual(default["model"], "gen-model")
        self.assertEqual(default["generation_model"], "gen-model")

        custom = reflection.parse_config(
            {
                "reflect": True,
                "reflection_max_loops": 3,
                "relevance_threshold": 2,
                "groundedness_threshold": 0,
                "reflection_model": "r",
            },
            "gen",
        )
        self.assertTrue(custom["enabled"])
        self.assertEqual(custom["max_loops"], 3)
        self.assertEqual(custom["relevance_threshold"], 2)
        self.assertEqual(custom["groundedness_threshold"], 0)
        self.assertEqual(custom["model"], "r")  # judge model
        self.assertEqual(custom["generation_model"], "gen")  # answer model unchanged

    def test_bad_types_fall_back(self):
        from rag_system.agent import reflection

        # bool must not be read as int; non-int values fall back to defaults
        cfg = reflection.parse_config(
            {"reflection_max_loops": True, "relevance_threshold": "x"}, "g"
        )
        self.assertEqual(cfg["max_loops"], 2)
        self.assertEqual(cfg["relevance_threshold"], 1)

    def test_max_loops_is_clamped(self):
        from rag_system.agent import reflection

        # request-controlled value is clamped to a hard ceiling (DoS guard)
        big = reflection.parse_config(
            {"reflect": True, "reflection_max_loops": 100000}, "m"
        )
        self.assertEqual(big["max_loops"], reflection._MAX_REFLECTION_LOOPS)
        self.assertLessEqual(big["max_loops"], 5)
        # and a floor of 1
        self.assertEqual(
            reflection.parse_config({"reflection_max_loops": 0}, "m")["max_loops"], 1
        )


class ReflectionWiringTests(unittest.TestCase):
    def test_execute_chat_uses_reflection_when_requested(self):
        from rag_system import chat_runtime

        pipe = _ReflectFakePipeline(["RES"], [_one_source()])
        agent = _ReflectFakeAgent(pipe, _ReflectFakeVerifier(rel=[2], ground=[2]))
        out = chat_runtime.execute_chat(
            agent, _FakeTimingDB(), {"query": "q", "reflect": True}
        )
        self.assertIn("reflection", out)
        self.assertEqual(out["reflection"]["rounds"], 0)
        self.assertEqual(out["answer"], "RES")
        self.assertEqual(pipe.run_queries, ["q"])

    def test_execute_chat_skips_reflection_by_default(self):
        from rag_system import chat_runtime

        pipe = _ReflectFakePipeline(["RES"], [_one_source()])
        agent = _ReflectFakeAgent(pipe, _ReflectFakeVerifier(rel=[2], ground=[2]))
        out = chat_runtime.execute_chat(agent, _FakeTimingDB(), {"query": "q"})
        self.assertNotIn("reflection", out)
        self.assertEqual(out["answer"], "agent-path")


class _RewriteLLM:
    def __init__(self, response):
        self._response = response
        self.calls = 0
        self.last_kwargs = {}

    def generate_completion(self, model, prompt, format=None, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        return self._response


class _RewriteFakeDB:
    def __init__(self, messages):
        self._messages = messages

    def list_indexes(self):
        return []

    def get_indexes_for_session(self, _sid):
        return []

    def get_index(self, _iid):
        return None

    def get_messages(self, session_id, limit=100):
        return self._messages


class QueryRewriteTests(unittest.TestCase):
    """Standalone multi-turn query rewrite (opt-in, retrieval path)."""

    def test_messages_to_turns_pairs_and_caps(self):
        from rag_system.agent import query_rewrite

        msgs = [
            {"sender": "user", "content": "a"},
            {"sender": "assistant", "content": "A"},
            {"sender": "user", "content": "b"},
            {"sender": "assistant", "content": "B"},
            {"sender": "user", "content": "dangling"},  # unpaired → dropped
        ]
        self.assertEqual(
            query_rewrite.messages_to_turns(msgs, max_turns=1),
            [{"user": "b", "assistant": "B"}],
        )

    def test_standalone_query_no_history_returns_original(self):
        from rag_system.agent import query_rewrite

        llm = _RewriteLLM({"response": json.dumps({"query": "X"})})
        self.assertEqual(query_rewrite.standalone_query(llm, "m", "q", []), "q")
        self.assertEqual(llm.calls, 0)  # no LLM call without history

    def test_standalone_query_rewrites_with_history(self):
        from rag_system.agent import query_rewrite

        llm = _RewriteLLM({"response": json.dumps({"query": "standalone Q"})})
        out = query_rewrite.standalone_query(
            llm, "m", "what about it?", [{"user": "tell me about X", "assistant": "X."}]
        )
        self.assertEqual(out, "standalone Q")
        # Deterministic + thinking-off keeps the rewrite reproducible and fast.
        self.assertEqual(llm.last_kwargs.get("temperature"), 0.0)
        self.assertIs(llm.last_kwargs.get("enable_thinking"), False)

    def test_standalone_query_falls_back_on_bad_json(self):
        from rag_system.agent import query_rewrite

        llm = _RewriteLLM({"response": "not json at all"})
        out = query_rewrite.standalone_query(
            llm, "m", "q", [{"user": "a", "assistant": "A"}]
        )
        self.assertEqual(out, "q")

    def test_execute_chat_rewrites_query_for_retrieval(self):
        from rag_system import chat_runtime

        pipe = _ReflectFakePipeline(["RES"], [_one_source()])  # rewrite → "REWRITTEN"
        agent = _ReflectFakeAgent(pipe, _ReflectFakeVerifier(rel=[2], ground=[2]))
        db = _RewriteFakeDB(
            [
                {"sender": "user", "content": "about X"},
                {"sender": "assistant", "content": "X is ..."},
            ]
        )
        chat_runtime.execute_chat(
            agent,
            db,
            {
                "query": "what about it?",
                "session_id": "s",
                "force_rag": True,
                "rewrite_query": True,
            },
        )
        self.assertEqual(pipe.run_queries, ["REWRITTEN"])

    def test_execute_chat_no_rewrite_by_default(self):
        from rag_system import chat_runtime

        pipe = _ReflectFakePipeline(["RES"], [_one_source()])
        agent = _ReflectFakeAgent(pipe, _ReflectFakeVerifier(rel=[2], ground=[2]))
        db = _RewriteFakeDB(
            [
                {"sender": "user", "content": "about X"},
                {"sender": "assistant", "content": "X is ..."},
            ]
        )
        chat_runtime.execute_chat(
            agent,
            db,
            {"query": "what about it?", "session_id": "s", "force_rag": True},
        )
        self.assertEqual(pipe.run_queries, ["what about it?"])  # original, no rewrite

    def test_execute_chat_rewrite_noop_without_history(self):
        from rag_system import chat_runtime

        pipe = _ReflectFakePipeline(["RES"], [_one_source()])
        agent = _ReflectFakeAgent(pipe, _ReflectFakeVerifier(rel=[2], ground=[2]))
        db = _RewriteFakeDB([])  # no prior messages
        chat_runtime.execute_chat(
            agent,
            db,
            {
                "query": "q",
                "session_id": "s",
                "force_rag": True,
                "rewrite_query": True,
            },
        )
        self.assertEqual(pipe.run_queries, ["q"])


class RetrievalDedupTests(unittest.TestCase):
    """Within one collection, the base + late-chunk legs can return the same
    (_source_table, chunk_id); collapse those to one (best rank). The same
    chunk_id from a DIFFERENT collection is different content and is kept."""

    def test_collapses_base_and_lc_within_a_collection(self):
        from rag_system.pipelines.retrieval_pipeline import _dedup_within_collection

        docs = [
            {"_source_table": "t", "chunk_id": "d_0", "_collection_rank": 1, "text": "base"},
            {"_source_table": "t", "chunk_id": "d_1", "_collection_rank": 2, "text": "other"},
            {"_source_table": "t", "chunk_id": "d_0", "_collection_rank": 5, "text": "lc-dup"},
        ]
        out = _dedup_within_collection(docs)
        self.assertEqual([d["chunk_id"] for d in out], ["d_0", "d_1"])  # order kept
        self.assertEqual(out[0]["text"], "base")  # better-ranked copy survived

    def test_same_chunk_id_in_different_collections_is_kept(self):
        from rag_system.pipelines.retrieval_pipeline import _dedup_within_collection

        docs = [
            {"_source_table": "table_a", "chunk_id": "report.txt_0", "_collection_rank": 1},
            {"_source_table": "table_b", "chunk_id": "report.txt_0", "_collection_rank": 1},
        ]
        out = _dedup_within_collection(docs)
        self.assertEqual(len(out), 2)  # different content across indexes

    def test_keeps_best_rank_even_when_it_arrives_later(self):
        from rag_system.pipelines.retrieval_pipeline import _dedup_within_collection

        docs = [
            {"_source_table": "t", "chunk_id": "x", "_collection_rank": 9, "text": "worse"},
            {"_source_table": "t", "chunk_id": "x", "_collection_rank": 2, "text": "better"},
        ]
        out = _dedup_within_collection(docs)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["text"], "better")

    def test_docs_without_chunk_id_pass_through(self):
        from rag_system.pipelines.retrieval_pipeline import _dedup_within_collection

        docs = [{"chunk_id": None}, {"chunk_id": None}, {"chunk_id": "k", "_source_table": "t"}]
        self.assertEqual(len(_dedup_within_collection(docs)), 3)


class ChatRuntimeLimitsTests(unittest.TestCase):
    """Request-knob clamping (local DoS guard)."""

    def test_clamp_int(self):
        from rag_system.chat_runtime import _clamp_int

        self.assertEqual(_clamp_int(10_000_000, 20, 1, 500), 500)  # ceiling
        self.assertEqual(_clamp_int(-5, 1, 0, 10), 0)  # floor
        self.assertEqual(_clamp_int(50, 20, 1, 500), 50)  # in range
        self.assertEqual(_clamp_int("nope", 20, 1, 500), 20)  # non-int -> default
        self.assertEqual(_clamp_int(None, 20, 1, 500), 20)

    def test_execute_chat_rejects_overlong_query(self):
        from rag_system import chat_runtime

        with self.assertRaises(ValueError):
            chat_runtime.execute_chat(object(), object(), {"query": "x" * 20001})


class EvalSuiteTests(unittest.TestCase):
    """Pure helpers behind the extended evaluation harness (rag_eval.py).

    Categories/difficulty, failure attribution, and per-group breakdowns are
    additive: legacy golden sets (no labels) must still summarize cleanly.
    """

    def test_norm_label_coerces_to_vocabulary(self):
        from rag_eval import _norm_label, _CATEGORIES

        self.assertEqual(_norm_label(" Numeric ", _CATEGORIES, "factual"), "numeric")
        self.assertEqual(_norm_label("bogus", _CATEGORIES, "factual"), "factual")
        self.assertEqual(_norm_label(None, _CATEGORIES, "factual"), "factual")

    def test_retrieval_failure_attribution(self):
        from rag_eval import _classify_retrieval_failure

        self.assertEqual(_classify_retrieval_failure(None, False, 20), "doc_not_retrieved")
        self.assertIn("beyond_k", _classify_retrieval_failure(25, True, 20))
        self.assertEqual(_classify_retrieval_failure(2, False, 20), "answer_chunk_missed")
        self.assertIsNone(_classify_retrieval_failure(2, True, 20))  # clean

    def test_e2e_failure_reports_first_cause(self):
        from rag_eval import _classify_e2e_failure

        # retrieval miss dominates downstream judge failures
        self.assertEqual(
            _classify_e2e_failure(None, False, 20, False, False, False, False, False),
            "doc_not_retrieved",
        )
        # retrieval fine -> first failing judge axis (relevance before accuracy)
        self.assertEqual(
            _classify_e2e_failure(1, True, 20, False, True, False, True, True),
            "context_irrelevant",
        )
        self.assertEqual(
            _classify_e2e_failure(1, True, 20, True, True, True, True, False),
            "bad_citation",
        )
        # everything passes -> no failure
        self.assertIsNone(
            _classify_e2e_failure(1, True, 20, True, True, True, True, True)
        )

    def test_breakdown_groups_by_label(self):
        from rag_eval import _breakdown

        cases = [
            {"category": "numeric"}, {"category": "numeric"}, {"category": "entity"},
        ]
        ranks = [1, None, 3]
        hits = [True, False, True]
        out = _breakdown(cases, ranks, hits, "category", 20)
        self.assertEqual(out["numeric"]["n"], 2)
        self.assertAlmostEqual(out["numeric"]["recall@20"], 0.5)  # one of two found
        self.assertAlmostEqual(out["entity"]["recall@20"], 1.0)

    def test_breakdown_defaults_missing_label_to_unknown(self):
        from rag_eval import _breakdown

        out = _breakdown([{}], [1], [True], "category", 20)
        self.assertIn("unknown", out)

    def test_case_records_carry_failure_reason(self):
        from rag_eval import _case_records

        cases = [{"question": "q", "expected_doc": "d.txt", "category": "numeric"}]
        recs = _case_records(cases, [None], [False], 20)
        self.assertEqual(recs[0]["failure"], "doc_not_retrieved")
        self.assertEqual(recs[0]["category"], "numeric")
        self.assertEqual(recs[0]["difficulty"], "unknown")  # legacy default

    def test_scalar_metrics_drops_n_bools_and_nested(self):
        from rag_eval import _scalar_metrics

        out = _scalar_metrics({
            "n": 12, "mrr": 0.9, "by_category": {"numeric": {"n": 3}},
            "some_flag": True, "chunk_hit": 1.0,
        })
        self.assertEqual(out, {"mrr": 0.9, "chunk_hit": 1.0})  # n/dict/bool dropped

    def test_compare_flags_drop_beyond_tolerance(self):
        from rag_eval import _compare_to_baseline

        baseline = {"mrr": 0.90, "chunk_hit": 1.0}
        current = {"mrr": 0.80, "chunk_hit": 0.99}  # mrr -0.10, chunk -0.01
        regr = _compare_to_baseline(current, baseline, tolerance=0.02)
        self.assertEqual([r["metric"] for r in regr], ["mrr"])  # only mrr beyond tol
        self.assertAlmostEqual(regr[0]["delta"], -0.10)

    def test_compare_ignores_improvements_and_latency(self):
        from rag_eval import _compare_to_baseline

        baseline = {"mrr": 0.80, "latency_avg_s": 10.0}
        current = {"mrr": 0.95, "latency_avg_s": 99.0}  # better quality, much slower
        self.assertEqual(_compare_to_baseline(current, baseline, 0.02), [])

    def test_compare_skips_metrics_absent_from_current(self):
        from rag_eval import _compare_to_baseline

        # a baseline key the current run didn't produce must not crash or flag
        self.assertEqual(
            _compare_to_baseline({"mrr": 0.9}, {"mrr": 0.9, "helpfulness": 0.8}, 0.02), []
        )

    def test_best_per_metric_picks_winner_per_column(self):
        from rag_eval import _best_per_metric

        results = [
            ("dense=0.00", {"mrr": 0.6, "chunk_hit": 0.9}),
            ("dense=0.50", {"mrr": 0.8, "chunk_hit": 0.9}),
            ("dense=1.00", {"mrr": 0.7, "chunk_hit": 0.95}),
        ]
        best = _best_per_metric(results, ["mrr", "chunk_hit"])
        self.assertEqual(best["mrr"], "dense=0.50")
        self.assertEqual(best["chunk_hit"], "dense=1.00")

    def test_best_per_metric_ties_keep_first(self):
        from rag_eval import _best_per_metric

        results = [("a", {"mrr": 0.8}), ("b", {"mrr": 0.8})]
        self.assertEqual(_best_per_metric(results, ["mrr"])["mrr"], "a")

    def test_format_comparison_marks_winner(self):
        from rag_eval import _format_comparison

        results = [("dense=0.00", {"mrr": 0.6}), ("dense=0.50", {"mrr": 0.8})]
        table = _format_comparison(results, ["mrr"])
        self.assertIn("dense=0.00", table)
        self.assertIn("dense=0.50", table)
        self.assertIn("0.800*", table)  # winner starred
        self.assertIn("0.600 ", table)  # loser not starred


class _RecordingClient:
    """Stands in for a cloud/local enrichment client; records what it received."""

    def __init__(self, name):
        self.name = name
        self.calls = []

    def generate_completion(self, model, prompt, **kwargs):
        self.calls.append(prompt)
        return {"response": f"{self.name}:ok"}


class DataPolicyTests(unittest.TestCase):
    """Egress governance in front of optional cloud enrichment.

    Security invariants: secrets block by default, fail-closed on bad policy,
    and no matched value is ever forwarded, redacted output, or audited.
    """

    # --- detection ---

    def test_detects_common_secrets(self):
        from rag_system.utils.data_policy import scan_text

        samples = {
            "anthropic_key": "key sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAAAAAA here",
            "aws_access_key_id": "AKIAIOSFODNN7EXAMPLE",
            "github_token": "ghp_" + "a" * 36,
            "private_key_block": "-----BEGIN RSA PRIVATE KEY-----",
            "assigned_secret": 'password = "hunter2hunter2hunter2"',
        }
        for expected, text in samples.items():
            cats = {f.detector for f in scan_text(text)}
            self.assertIn(expected, cats, f"missed {expected} in {text!r}")

    def test_credit_card_requires_luhn(self):
        from rag_system.utils.data_policy import scan_text

        valid = [f.detector for f in scan_text("card 4111 1111 1111 1111")]
        invalid = [f.detector for f in scan_text("id 4111 1111 1111 1112")]
        self.assertIn("credit_card", valid)
        self.assertNotIn("credit_card", invalid)  # fails checksum -> not flagged

    def test_clean_text_has_no_findings(self):
        from rag_system.utils.data_policy import scan_text

        self.assertEqual(scan_text("The Aurora Dam has a capacity of 240 MW."), [])

    def test_findings_never_carry_the_value(self):
        from rag_system.utils.data_policy import scan_text, Finding

        f = scan_text("AKIAIOSFODNN7EXAMPLE")[0]
        self.assertIsInstance(f, Finding)
        self.assertEqual(set(vars(f)), {"detector", "category", "start", "end"})

    # --- redaction ---

    def test_redaction_masks_the_secret(self):
        from rag_system.utils.data_policy import scan_text, redact_text

        text = "before AKIAIOSFODNN7EXAMPLE after"
        out = redact_text(text, scan_text(text))
        self.assertNotIn("AKIAIOSFODNN7EXAMPLE", out)
        self.assertIn("[REDACTED:aws_access_key_id]", out)
        self.assertIn("before", out)
        self.assertIn("after", out)

    # --- policy resolution (fail-closed) ---

    def test_normalize_policy_defaults_and_fail_closed(self):
        from rag_system.utils.data_policy import normalize_policy, SECRET, PII, BLOCK, ALLOW

        self.assertEqual(normalize_policy(None), {SECRET: BLOCK, PII: ALLOW})
        self.assertEqual(normalize_policy({"pii": "redact"})[PII], "redact")
        # unknown action must not open egress
        self.assertEqual(normalize_policy({"secret": "yolo"})[SECRET], BLOCK)

    def test_evaluate_actions(self):
        from rag_system.utils.data_policy import evaluate, BLOCK, ALLOW, REDACT

        self.assertEqual(evaluate("just a normal sentence").action, ALLOW)
        self.assertEqual(evaluate("AKIAIOSFODNN7EXAMPLE").action, BLOCK)  # secret default
        self.assertEqual(evaluate("email a@b.com").action, ALLOW)  # pii default
        red = evaluate("email a@b.com", {"pii": REDACT})
        self.assertEqual(red.action, REDACT)
        self.assertNotIn("a@b.com", red.redacted_text)
        # explicit override can permit a secret (e.g. trusted internal index)
        self.assertEqual(evaluate("AKIAIOSFODNN7EXAMPLE", {"secret": "allow"}).action, ALLOW)

    # --- enforcement at the egress boundary ---

    def test_guard_blocks_to_local_fallback(self):
        from rag_system.utils.data_policy import PolicyGuardedEnricher

        cloud, local = _RecordingClient("cloud"), _RecordingClient("local")
        audited = []
        guard = PolicyGuardedEnricher(
            cloud, local_fallback=local, audit=audited.append, provider="anthropic"
        )
        out = guard.generate_completion("m", "leak AKIAIOSFODNN7EXAMPLE")
        self.assertEqual(out["response"], "local:ok")  # served locally
        self.assertEqual(cloud.calls, [])  # nothing left the machine
        self.assertEqual(local.calls[0], "leak AKIAIOSFODNN7EXAMPLE")
        self.assertEqual(audited[0]["action"], "block")
        self.assertIn("aws_access_key_id", audited[0]["findings"])
        # audit carries counts/types only, never the value
        self.assertNotIn("AKIAIOSFODNN7EXAMPLE", json.dumps(audited[0]))

    def test_guard_blocks_with_no_fallback_returns_empty(self):
        from rag_system.utils.data_policy import PolicyGuardedEnricher

        cloud = _RecordingClient("cloud")
        guard = PolicyGuardedEnricher(cloud, local_fallback=None)
        self.assertEqual(guard.generate_completion("m", "AKIAIOSFODNN7EXAMPLE"), {})
        self.assertEqual(cloud.calls, [])  # fail-closed: no egress, no result

    def test_guard_redacts_before_forwarding(self):
        from rag_system.utils.data_policy import PolicyGuardedEnricher

        cloud = _RecordingClient("cloud")
        guard = PolicyGuardedEnricher(cloud, policy={"pii": "redact"})
        guard.generate_completion("m", "contact a@b.com please")
        self.assertEqual(len(cloud.calls), 1)
        self.assertNotIn("a@b.com", cloud.calls[0])  # masked before send
        self.assertIn("[REDACTED:email]", cloud.calls[0])

    def test_guard_allows_clean_text_unchanged(self):
        from rag_system.utils.data_policy import PolicyGuardedEnricher

        cloud = _RecordingClient("cloud")
        audited = []
        guard = PolicyGuardedEnricher(cloud, audit=audited.append)
        guard.generate_completion("m", "Aurora Dam capacity is 240 MW")
        self.assertEqual(cloud.calls, ["Aurora Dam capacity is 240 MW"])
        self.assertEqual(audited, [])  # allow is not audited

    def test_safety_corpus(self):
        """Data-driven detector coverage + false-positive guards.

        The committed corpus (tests/eval_fixtures/safety_samples.jsonl) is the
        extensible safety set this branch owns: every secret must block under
        the default policy, labelled PII must be detected, and the near-miss
        rows (non-Luhn digits, git SHAs, UUIDs, versions) must NOT fire.
        """
        from rag_system.utils.data_policy import scan_text, evaluate

        path = os.path.join("tests", "eval_fixtures", "safety_samples.jsonl")
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        self.assertGreaterEqual(len(rows), 15)  # corpus didn't silently shrink
        for row in rows:
            cats = {f.category for f in scan_text(row["text"])}
            action = evaluate(row["text"]).action  # default fail-closed policy
            if row["expect_category"] is None:
                self.assertEqual(cats, set(), f"false positive on {row['name']!r}")
            else:
                self.assertIn(row["expect_category"], cats, f"missed {row['name']!r}")
            self.assertEqual(
                action, row["expect_action_default"], f"wrong action for {row['name']!r}"
            )


class _FakeReportLLM:
    """Returns a fixed planning JSON regardless of prompt."""

    def __init__(self, sections_json):
        self._json = sections_json

    def generate_completion(self, model, prompt, **kwargs):
        return {"response": self._json}


class _FakeReportPipeline:
    """Returns a canned retrieval result per section, in call order."""

    def __init__(self, ollama_client, results):
        self.ollama_client = ollama_client
        self._results = results
        self.calls = []

    def run(self, query, event_callback=None, **kwargs):
        self.calls.append(query)
        return self._results[len(self.calls) - 1]


class ReportModeTests(unittest.TestCase):
    """Local long-form report generation (rag_system.agent.report)."""

    def test_parse_sections_dedups_clamps_and_falls_back(self):
        from rag_system.agent.report import parse_sections

        out = parse_sections('{"sections": ["A", "a", "B", "C", "D"]}', max_sections=3)
        self.assertEqual(out, ["A", "B", "C"])  # case-dedup + clamp to 3
        self.assertEqual(parse_sections("not json", 4), ["Overview"])  # fallback
        self.assertEqual(parse_sections('{"sections": []}', 4), ["Overview"])

    def test_remap_citations_globalizes_and_dedups(self):
        from rag_system.agent.report import remap_citations

        s1 = {"document_id": "a.txt", "chunk_id": "a.txt_0", "_source_table": "t"}
        s2 = {"document_id": "b.txt", "chunk_id": "b.txt_1", "_source_table": "t"}
        gs, gk = [], {}
        t1, d1 = remap_citations("Alpha [1] beta [2]", [s1, s2], gs, gk)
        self.assertEqual(t1, "Alpha [1] beta [2]")
        self.assertEqual(d1, 0)
        # second section cites the SAME chunk -> reuses global index 1
        t2, d2 = remap_citations("Gamma [1]", [s1], gs, gk)
        self.assertEqual(t2, "Gamma [1]")
        self.assertEqual(len(gs), 2)  # s1, s2 only

    def test_remap_drops_out_of_range_citation(self):
        from rag_system.agent.report import remap_citations

        s1 = {"document_id": "a.txt", "chunk_id": "a.txt_0", "_source_table": "t"}
        gs, gk = [], {}
        text, dropped = remap_citations("see [5] only", [s1], gs, gk)
        self.assertEqual(dropped, 1)
        self.assertNotIn("[5]", text)
        self.assertEqual(gs, [])  # nothing valid cited

    def test_compile_report_structure(self):
        from rag_system.agent.report import compile_report

        md = compile_report("My Q", [("Sec A", "body a [1]")], [{"document_id": "a.txt"}])
        self.assertIn("# My Q", md)
        self.assertIn("## Sec A", md)
        self.assertIn("body a [1]", md)
        self.assertIn("## References", md)
        self.assertIn("1. a.txt", md)

    def test_generate_report_end_to_end(self):
        from rag_system.agent.report import generate_report

        s1 = {"document_id": "a.txt", "chunk_id": "a.txt_0", "_source_table": "t", "text": "alpha"}
        s2 = {"document_id": "b.txt", "chunk_id": "b.txt_1", "_source_table": "t", "text": "beta"}
        pipeline = _FakeReportPipeline(
            _FakeReportLLM('{"sections": ["Background", "Findings"]}'),
            results=[
                {"answer": "Alpha [1] and [2]", "source_documents": [s1, s2]},
                {"answer": "Reuse [1]", "source_documents": [s1]},  # same chunk -> [1]
            ],
        )
        events = []
        out = generate_report(
            pipeline, "model", "Explain the project",
            run_kwargs={"table_name": "t", "overrides": {}},
            event_callback=lambda t, p: events.append(t),
            max_sections=4,
        )
        self.assertEqual(len(pipeline.calls), 2)  # one retrieval per section
        self.assertEqual(out["report"]["section_count"], 2)
        self.assertEqual(len(out["source_documents"]), 2)  # deduped across sections
        self.assertIn("## Background", out["answer"])
        self.assertIn("## Findings", out["answer"])
        self.assertIn("Reuse [1]", out["answer"])  # section 2 reused global 1
        self.assertIn("report_started", events)
        self.assertIn("report_done", events)


if __name__ == "__main__":
    unittest.main()

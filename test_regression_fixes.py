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

    def test_execute_chat_total_only_without_callback(self):
        from unittest.mock import patch

        from rag_system import chat_runtime

        # No event_callback => no stage events fire (and none are forced), but
        # total latency is still recorded.
        with patch.dict(os.environ, {"LOCALGPT_TIMINGS": "1"}):
            result = chat_runtime.execute_chat(
                _FakeTimingAgent(),
                _FakeTimingDB(),
                {"query": "q", "force_rag": True},
                event_callback=None,
            )
        self.assertEqual(list(result["timings_ms"]), ["total"])
        self.assertNotIn("ttft_ms", result)


class _ReflectFakePipeline:
    """Scripted RetrievalPipeline: returns queued answers/sources per run(),
    records calls, and emits a stage + (suppressible) token event."""

    def __init__(self, answers, sources):
        self.ollama_client = self
        self._answers = list(answers)
        self._sources = list(sources)
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
        return "REGENERATED"

    def generate_completion(self, model, prompt, format=None):  # rewrite path
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


class ReflectionConfigTests(unittest.TestCase):
    def test_defaults_and_overrides(self):
        from rag_system.agent import reflection

        default = reflection.parse_config({}, "gen-model")
        self.assertFalse(default["enabled"])
        self.assertEqual(default["max_loops"], 2)
        self.assertEqual(default["relevance_threshold"], 1)
        self.assertEqual(default["model"], "gen-model")

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
        self.assertEqual(custom["model"], "r")

    def test_bad_types_fall_back(self):
        from rag_system.agent import reflection

        # bool must not be read as int; non-int values fall back to defaults
        cfg = reflection.parse_config(
            {"reflection_max_loops": True, "relevance_threshold": "x"}, "g"
        )
        self.assertEqual(cfg["max_loops"], 2)
        self.assertEqual(cfg["relevance_threshold"], 1)


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


if __name__ == "__main__":
    unittest.main()

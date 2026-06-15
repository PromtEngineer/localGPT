import asyncio
import io
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import threading
import types
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.datastructures import Headers, UploadFile

from backend.database import ChatDatabase
import backend.server as server
from rag_system.job_persistence import JobProgressTracker
from rag_system.maintenance import MaintenanceTools


def _install_indexing_pipeline_import_stubs():
    """Provide lightweight optional dependency stubs for importing IndexingPipeline."""
    modules = {}

    docling_converter = types.ModuleType("docling.document_converter")
    docling_converter.DocumentConverter = type("DoclingConverter", (), {"__init__": lambda self, *a, **k: None})
    docling_converter.PdfFormatOption = type("PdfFormatOption", (), {"__init__": lambda self, *a, **k: None})
    modules["docling.document_converter"] = docling_converter

    docling_pipeline = types.ModuleType("docling.datamodel.pipeline_options")
    docling_pipeline.PdfPipelineOptions = type("PdfPipelineOptions", (), {"__init__": lambda self: None})
    docling_pipeline.OcrMacOptions = type("OcrMacOptions", (), {"__init__": lambda self, *a, **k: None})
    modules["docling.datamodel.pipeline_options"] = docling_pipeline

    class _InputFormat:
        PDF = "PDF"
        DOCX = "DOCX"
        HTML = "HTML"
        MD = "MD"

    docling_base = types.ModuleType("docling.datamodel.base_models")
    docling_base.InputFormat = _InputFormat
    modules["docling.datamodel.base_models"] = docling_base

    for name in ("docling", "docling.datamodel"):
        modules[name] = types.ModuleType(name)

    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = type("AutoTokenizer", (), {"from_pretrained": staticmethod(lambda *a, **k: None)})
    transformers.AutoModel = type("AutoModel", (), {"from_pretrained": staticmethod(lambda *a, **k: None)})
    modules["transformers"] = transformers

    model_registry = types.ModuleType("rag_system.model_registry")
    model_registry.get_dims = lambda _model: None
    model_registry.get_dtype = lambda *_args, **_kwargs: None
    modules["rag_system.model_registry"] = model_registry

    return modules


class BackendApiContractTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "chat_data.db")
        self.original_db = server.db
        self.original_job_progress_tracker = server.job_progress_tracker
        self.original_thread_class = server.threading.Thread
        server.db = ChatDatabase(self.db_path)
        self.client = TestClient(server.app)

    def tearDown(self):
        server.db = self.original_db
        server.job_progress_tracker = self.original_job_progress_tracker
        server.threading.Thread = self.original_thread_class
        shutil.rmtree(self.temp_dir)

    def test_frontend_session_lifecycle_contract(self):
        create_response = self.client.post(
            "/sessions",
            json={"title": "Contract Test", "model": "test-model"},
        )
        self.assertEqual(create_response.status_code, 200)
        created = create_response.json()
        session_id = created["session_id"]
        self.assertEqual(created["session"]["title"], "Contract Test")

        cleanup_response = self.client.get("/sessions/cleanup")
        self.assertEqual(cleanup_response.status_code, 200)
        self.assertIn("cleanup_count", cleanup_response.json())

        create_response = self.client.post(
            "/sessions",
            json={"title": "Rename Me", "model": "test-model"},
        )
        session_id = create_response.json()["session_id"]

        rename_response = self.client.post(
            f"/sessions/{session_id}/rename",
            json={"title": "Renamed"},
        )
        self.assertEqual(rename_response.status_code, 200)
        self.assertEqual(rename_response.json()["session"]["title"], "Renamed")

        delete_response = self.client.delete(f"/sessions/{session_id}")
        self.assertEqual(delete_response.status_code, 200)
        self.assertEqual(delete_response.json()["deleted_session_id"], session_id)

        missing_response = self.client.get(f"/sessions/{session_id}")
        self.assertEqual(missing_response.status_code, 404)

    def test_persistent_index_job_contract(self):
        index_id = server.db.create_index("Job Contract")
        stored_path = os.path.join(self.temp_dir, "contract-doc.txt")
        with open(stored_path, "w", encoding="utf-8") as handle:
            handle.write("contract document")
        server.db.add_document_to_index(index_id, "contract-doc.txt", stored_path)

        job = server.db.create_index_job(
            "contract-job",
            index_id,
            {"background": True, "profile": "fast"},
            [{"filename": "contract-doc.txt", "stored_path": stored_path}],
        )
        self.assertEqual(job["status"], "queued")

        get_response = self.client.get("/index-jobs/contract-job")
        self.assertEqual(get_response.status_code, 200)
        public_job = get_response.json()
        self.assertNotIn("options", public_job)
        self.assertEqual(len(public_job["files"]), 1)
        self.assertEqual(public_job["files"][0]["status"], "pending")

        progress_response = self.client.post(
            "/index-jobs/contract-job/progress",
            json={
                "stage": "chunking",
                "progress": 35,
                "message": "Chunking contract-doc.txt",
                "file_path": stored_path,
                "file_status": "processing",
                "chunks_generated": 2,
            },
        )
        self.assertEqual(progress_response.status_code, 200)

        updated_response = self.client.get("/index-jobs/contract-job")
        updated_job = updated_response.json()
        self.assertEqual(updated_job["stage"], "chunking")
        self.assertEqual(updated_job["progress"], 35)
        self.assertEqual(updated_job["files"][0]["status"], "processing")
        self.assertEqual(updated_job["files"][0]["chunks_generated"], 2)

        completed_progress_response = self.client.post(
            "/index-jobs/contract-job/progress",
            json={
                "stage": "completed",
                "progress": 100,
                "message": "Indexing complete",
            },
        )
        self.assertEqual(completed_progress_response.status_code, 200)
        progress_job = completed_progress_response.json()
        self.assertEqual(progress_job["status"], "queued")
        self.assertEqual(progress_job["stage"], "finalizing")
        self.assertEqual(progress_job["progress"], 99)

        cancel_response = self.client.post("/index-jobs/contract-job/cancel")
        self.assertEqual(cancel_response.status_code, 200)
        cancelled = cancel_response.json()
        self.assertTrue(cancelled["cancel_requested"])
        self.assertEqual(cancelled["status"], "cancelled")
        self.assertEqual(cancelled["stage"], "cancelled")

    def test_index_build_preflight_contract(self):
        index_id = server.db.create_index("Preflight Contract")

        empty_response = self.client.post(
            f"/indexes/{index_id}/build/preflight",
            json={"checkServices": False},
        )
        self.assertEqual(empty_response.status_code, 200)
        empty_preflight = empty_response.json()
        self.assertFalse(empty_preflight["ok"])
        self.assertIn("No documents are attached", " ".join(empty_preflight["errors"]))

        stored_path = os.path.join(self.temp_dir, "preflight-doc.txt")
        with open(stored_path, "w", encoding="utf-8") as handle:
            handle.write("preflight document")
        server.db.add_document_to_index(index_id, "preflight-doc.txt", stored_path)

        with patch("backend.server.requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            ready_response = self.client.post(
                f"/indexes/{index_id}/build/preflight",
                json={"enrichModel": "gpt-oss:120b-cloud", "overviewModel": "qwen3:8b"},
            )

        self.assertEqual(ready_response.status_code, 200)
        ready_preflight = ready_response.json()
        self.assertTrue(ready_preflight["ok"])
        self.assertEqual(ready_preflight["document_count"], 1)
        self.assertGreater(ready_preflight["total_bytes"], 0)
        self.assertTrue(ready_preflight["rag_api_available"])
        self.assertTrue(any("will be replaced" in warning for warning in ready_preflight["warnings"]))

    def test_required_metadata_cannot_be_omitted_on_upload(self):
        index_id = server.db.create_index(
            "Required metadata",
            metadata={
                "status": "created",
                "metadata_schema": [
                    {"name": "project", "type": "string", "required": True},
                ],
            },
        )
        response = self.client.post(
            f"/indexes/{index_id}/upload",
            files={"files": ("report.txt", b"report contents", "text/plain")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("required metadata field missing", response.json()["detail"])
        self.assertEqual(server.db.get_index(index_id)["documents"], [])

    def test_required_metadata_must_cover_every_uploaded_file(self):
        index_id = server.db.create_index(
            "Per-file metadata",
            metadata={
                "status": "created",
                "metadata_schema": [
                    {"name": "project", "type": "string", "required": True},
                ],
            },
        )
        response = self.client.post(
            f"/indexes/{index_id}/upload",
            files=[
                ("files", ("first.txt", b"first", "text/plain")),
                ("files", ("second.txt", b"second", "text/plain")),
            ],
            data={"metadata": json.dumps({"first.txt": {"project": "Aurora"}})},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("required metadata field missing", response.json()["detail"])
        self.assertEqual(server.db.get_index(index_id)["documents"], [])

    def test_schema_cannot_change_after_upload(self):
        index_id = server.db.create_index(
            "Locked metadata",
            metadata={
                "status": "created",
                "metadata_schema": [{"name": "project", "type": "string"}],
            },
        )
        path = os.path.join(self.temp_dir, "report.txt")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("report")
        server.db.add_document_to_index(index_id, "report.txt", path, {"project": "Aurora"})

        response = self.client.put(
            f"/indexes/{index_id}/metadata-schema",
            json={"schema": [{"name": "year", "type": "integer"}]},
        )
        self.assertEqual(response.status_code, 409)
        self.assertEqual(
            server.db.get_index(index_id)["metadata"]["metadata_schema"],
            [{"name": "project", "type": "string"}],
        )

    def test_index_creation_rejects_invalid_embedded_metadata_schema(self):
        response = self.client.post(
            "/indexes",
            json={
                "name": "Invalid schema",
                "metadata": {
                    "metadata_schema": [{"name": "year", "type": "not-a-type"}],
                },
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("type must be one of", response.json()["detail"])

    def test_index_diagnostics_contract(self):
        index_id = server.db.create_index("Diagnostics Contract")
        stored_path = os.path.join(self.temp_dir, "diagnostics-doc.txt")
        with open(stored_path, "w", encoding="utf-8") as handle:
            handle.write("diagnostics document")
        server.db.add_document_to_index(index_id, "diagnostics-doc.txt", stored_path)

        response = self.client.get(f"/indexes/{index_id}/diagnostics")
        self.assertEqual(response.status_code, 200)
        diagnostics = response.json()
        self.assertEqual(diagnostics["index_id"], index_id)
        self.assertEqual(diagnostics["document_count"], 1)
        self.assertGreater(diagnostics["total_bytes"], 0)
        self.assertEqual(diagnostics["health"], "unhealthy")
        self.assertEqual(diagnostics["recommended_action"], "force_rebuild")
        self.assertTrue(diagnostics["can_repair"])
        self.assertFalse(diagnostics["vector_table"]["exists"])
        self.assertTrue(any("Vector table is missing" in error for error in diagnostics["errors"]))

        summary_response = self.client.get("/indexes/diagnostics")
        self.assertEqual(summary_response.status_code, 200)
        summaries = summary_response.json()["diagnostics"]
        summary = next(item for item in summaries if item["index_id"] == index_id)
        self.assertEqual(summary["health"], "unhealthy")
        self.assertEqual(summary["recommended_action"], "force_rebuild")
        self.assertEqual(summary["document_count"], 1)
        self.assertFalse(summary["vector_exists"])

        session_id = server.db.create_session("Guarded Link", "test-model")
        link_response = self.client.post(f"/sessions/{session_id}/indexes/{index_id}")
        self.assertEqual(link_response.status_code, 409)
        link_detail = link_response.json()["detail"]
        self.assertIn("cannot be opened safely", link_detail["message"])
        self.assertEqual(link_detail["diagnostics"]["recommended_action"], "force_rebuild")

        legacy_session_id = server.db.create_session("Legacy Guard", "test-model")
        server.db.link_index_to_session(legacy_session_id, index_id)
        chat_response = self.client.post(
            f"/sessions/{legacy_session_id}/messages",
            json={"message": "Can I trust this index?"},
        )
        self.assertEqual(chat_response.status_code, 409)
        chat_detail = chat_response.json()["detail"]
        self.assertIn("Cannot chat with unhealthy linked index", chat_detail["message"])
        self.assertEqual(chat_detail["diagnostics"][0]["recommended_action"], "force_rebuild")

    def test_delete_index_removes_owned_storage_artifacts(self):
        import lancedb

        index_id = server.db.create_index("Delete artifacts")
        index = server.db.get_index(index_id)
        table_name = index["vector_table_name"]

        upload_dir = os.path.join(self.temp_dir, "uploads")
        overview_dir = os.path.join(self.temp_dir, "overviews")
        lancedb_dir = os.path.join(self.temp_dir, "lancedb")
        os.makedirs(upload_dir)
        os.makedirs(overview_dir)

        upload_path = os.path.join(upload_dir, "owned.txt")
        with open(upload_path, "w", encoding="utf-8") as handle:
            handle.write("owned upload")
        server.db.add_document_to_index(index_id, "owned.txt", upload_path)

        conn = lancedb.connect(lancedb_dir)
        rows = [{"vector": [0.0, 1.0], "text": "owned"}]
        conn.create_table(table_name, rows)
        conn.create_table(f"{table_name}_lc", rows)

        overview_path = os.path.join(overview_dir, f"{index_id}.jsonl")
        with open(overview_path, "w", encoding="utf-8") as handle:
            handle.write('{"document_id":"owned.txt"}\n')

        with (
            patch.object(server, "_UPLOAD_DIR", upload_dir),
            patch.object(server, "_lancedb_path_candidates", return_value=[lancedb_dir]),
            patch.object(server, "_overview_path_candidates", return_value=[overview_path]),
        ):
            response = self.client.delete(f"/indexes/{index_id}")

        self.assertEqual(response.status_code, 200, response.text)
        removed = response.json()["removed"]
        self.assertEqual(len(removed["tables"]), 2)
        self.assertEqual(removed["files"], [os.path.realpath(upload_path)])
        self.assertEqual(removed["overviews"], [overview_path])
        self.assertFalse(os.path.exists(upload_path))
        self.assertFalse(os.path.exists(overview_path))
        self.assertNotIn(table_name, conn.list_tables().tables)
        self.assertNotIn(f"{table_name}_lc", conn.list_tables().tables)
        self.assertIsNone(server.db.get_index(index_id))

    def test_delete_index_preserves_external_source_file(self):
        index_id = server.db.create_index("External source")
        external_path = os.path.join(self.temp_dir, "external.txt")
        with open(external_path, "w", encoding="utf-8") as handle:
            handle.write("external source")
        server.db.add_document_to_index(index_id, "external.txt", external_path)

        upload_dir = os.path.join(self.temp_dir, "uploads")
        os.makedirs(upload_dir)
        with (
            patch.object(server, "_UPLOAD_DIR", upload_dir),
            patch.object(server, "_lancedb_path_candidates", return_value=[]),
            patch.object(server, "_overview_path_candidates", return_value=[]),
        ):
            response = self.client.delete(f"/indexes/{index_id}")

        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(os.path.exists(external_path))
        self.assertEqual(
            response.json()["removed"]["skipped_files"],
            [os.path.realpath(external_path)],
        )

    def test_delete_index_preserves_file_shared_with_another_index(self):
        """Deleting one index must not remove an owned upload that another
        index still references — mirrors the maintenance path's is_shared
        guard. Behavior-neutral under normal uploads (unique paths), but
        guards against data loss if a path is ever shared across indexes."""
        upload_dir = os.path.join(self.temp_dir, "uploads")
        os.makedirs(upload_dir)
        shared_path = os.path.join(upload_dir, "shared.txt")
        with open(shared_path, "w", encoding="utf-8") as handle:
            handle.write("shared upload")

        doomed = server.db.create_index("Doomed")
        keeper = server.db.create_index("Keeper")
        server.db.add_document_to_index(doomed, "shared.txt", shared_path)
        server.db.add_document_to_index(keeper, "shared.txt", shared_path)

        with (
            patch.object(server, "_UPLOAD_DIR", upload_dir),
            patch.object(server, "_lancedb_path_candidates", return_value=[]),
            patch.object(server, "_overview_path_candidates", return_value=[]),
        ):
            response = self.client.delete(f"/indexes/{doomed}")

        self.assertEqual(response.status_code, 200, response.text)
        removed = response.json()["removed"]
        self.assertEqual(removed["shared_files"], [os.path.realpath(shared_path)])
        self.assertEqual(removed["files"], [])
        # The file the surviving index needs is still on disk.
        self.assertTrue(os.path.exists(shared_path))
        self.assertIsNone(server.db.get_index(doomed))
        self.assertIsNotNone(server.db.get_index(keeper))

    def test_delete_index_artifact_failure_preserves_db_record(self):
        """If artifact cleanup fails, the DB record must survive so the delete
        is retryable — deletion does artifacts first (1501) then the DB row
        (1502), so a RuntimeError from _delete_index_artifacts aborts before
        any row is touched. The alternative — orphaning the row — is the bug
        this guards against."""
        index_id = server.db.create_index("Artifact failure")
        upload_dir = os.path.join(self.temp_dir, "uploads")
        os.makedirs(upload_dir)
        owned_path = os.path.join(upload_dir, "owned.txt")
        with open(owned_path, "w", encoding="utf-8") as handle:
            handle.write("owned upload")
        server.db.add_document_to_index(index_id, "owned.txt", owned_path)

        def _boom(_path):
            raise OSError("simulated disk failure")

        with (
            patch.object(server, "_UPLOAD_DIR", upload_dir),
            patch.object(server, "_lancedb_path_candidates", return_value=[]),
            patch.object(server, "_overview_path_candidates", return_value=[]),
            patch.object(server.os, "remove", side_effect=_boom),
        ):
            response = self.client.delete(f"/indexes/{index_id}")

        self.assertEqual(response.status_code, 500, response.text)
        # The record (and its document link) survived — nothing was orphaned.
        surviving = server.db.get_index(index_id)
        self.assertIsNotNone(surviving)
        self.assertEqual(
            [doc["filename"] for doc in surviving.get("documents") or []],
            ["owned.txt"],
        )
        # The file the remove() failed on is still on disk, not half-deleted.
        self.assertTrue(os.path.exists(owned_path))

    def test_delete_index_db_deletion_is_atomic(self):
        """db.delete_index commits once, after every child + parent DELETE
        (database.py:585). A failure on the final 'DELETE FROM indexes' must
        leave the already-issued child deletes uncommitted, so close() rolls
        them back — no half-deleted index left behind."""
        import backend.database as database

        index_id = server.db.create_index("Atomic delete")
        session_id = server.db.create_session("s", "m")
        server.db.add_document_to_index(index_id, "a.txt", "/tmp/a.txt")
        server.db.link_index_to_session(session_id, index_id)

        real_connect = database._connect

        class _FailOnIndexDelete:
            def __init__(self, conn):
                self._conn = conn

            def execute(self, sql, *args):
                if sql.lstrip().upper().startswith("DELETE FROM INDEXES"):
                    raise sqlite3.OperationalError("simulated mid-delete failure")
                return self._conn.execute(sql, *args)

            def commit(self):
                return self._conn.commit()

            def close(self):
                return self._conn.close()

            def __getattr__(self, name):
                return getattr(self._conn, name)

        with patch.object(
            database, "_connect", lambda path: _FailOnIndexDelete(real_connect(path))
        ):
            with self.assertRaises(sqlite3.OperationalError):
                server.db.delete_index(index_id)

        # Nothing committed: the index, its document, and its session link are
        # all still present — the child deletes rolled back with the parent.
        surviving = server.db.get_index(index_id)
        self.assertIsNotNone(surviving)
        self.assertEqual(
            [doc["filename"] for doc in surviving.get("documents") or []],
            ["a.txt"],
        )
        self.assertIn(index_id, server.db.get_indexes_for_session(session_id))

    def test_concurrent_delete_index_is_race_safe(self):
        """Two simultaneous deletes of the same index: exactly one wins
        (rowcount>0 -> True), the other sees it already gone (-> False). No
        exception, and the row is gone once. Guards the read-then-delete gap
        in delete_index under WAL + busy-timeout serialization."""
        index_id = server.db.create_index("Race delete")
        server.db.add_document_to_index(index_id, "a.txt", "/tmp/a.txt")

        barrier = threading.Barrier(2)
        results: List[object] = []
        errors: List[Exception] = []

        def worker():
            try:
                barrier.wait(timeout=10)
                results.append(server.db.delete_index(index_id))
            except Exception as exc:  # noqa: BLE001 — surface races, don't swallow
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

        self.assertEqual(errors, [], f"concurrent delete raised: {errors}")
        self.assertEqual(
            sorted(bool(r) for r in results),
            [False, True],
            f"expected exactly one winner, got {results}",
        )
        self.assertIsNone(server.db.get_index(index_id))

    def test_delete_index_twice_is_idempotent_not_found(self):
        """A second delete of an already-deleted index is a clean 404, not a
        500 — the get_index guard (1500) short-circuits before any artifact or
        DB work runs again."""
        index_id = server.db.create_index("Double delete")
        with (
            patch.object(server, "_UPLOAD_DIR", os.path.join(self.temp_dir, "u")),
            patch.object(server, "_lancedb_path_candidates", return_value=[]),
            patch.object(server, "_overview_path_candidates", return_value=[]),
        ):
            first = self.client.delete(f"/indexes/{index_id}")
            second = self.client.delete(f"/indexes/{index_id}")

        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(second.status_code, 404, second.text)

    # ---- Upload streaming: size limit + batch rollback ------------------

    @staticmethod
    def _make_upload(filename, data, size="auto", content_type="text/plain"):
        if size == "auto":
            size = len(data)
        headers = Headers({"content-type": content_type})
        return UploadFile(
            file=io.BytesIO(data), size=size, filename=filename, headers=headers
        )

    def test_upload_rejects_disallowed_extension(self):
        upload_dir = os.path.join(self.temp_dir, "up")
        bad = self._make_upload(
            "malware.exe", b"MZ\x90\x00", content_type="application/x-msdownload"
        )
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(server._save_uploads([bad], upload_dir=upload_dir))
        self.assertEqual(ctx.exception.status_code, 415)
        # Validation runs before any write — nothing landed on disk.
        self.assertEqual(
            os.listdir(upload_dir) if os.path.exists(upload_dir) else [], []
        )

    def test_upload_rejects_oversize_via_declared_size(self):
        upload_dir = os.path.join(self.temp_dir, "up")
        with patch.object(server, "_MAX_UPLOAD_BYTES", 50):
            big = self._make_upload("big.txt", b"x" * 200)  # declared size 200 > 50
            with self.assertRaises(HTTPException) as ctx:
                asyncio.run(server._save_uploads([big], upload_dir=upload_dir))
        self.assertEqual(ctx.exception.status_code, 413)
        self.assertEqual(
            os.listdir(upload_dir) if os.path.exists(upload_dir) else [], []
        )

    def test_upload_rejects_oversize_mid_stream_and_removes_partial(self):
        # size=None forces the streaming guard (not the declared-size pre-check),
        # which opens the file then aborts — the partial must be cleaned up.
        upload_dir = os.path.join(self.temp_dir, "up")
        with patch.object(server, "_MAX_UPLOAD_BYTES", 50):
            big = self._make_upload("big.txt", b"x" * 200, size=None)
            with self.assertRaises(HTTPException) as ctx:
                asyncio.run(server._save_uploads([big], upload_dir=upload_dir))
        self.assertEqual(ctx.exception.status_code, 413)
        self.assertEqual(
            os.listdir(upload_dir) if os.path.exists(upload_dir) else [], []
        )

    def test_upload_rejects_when_batch_exceeds_total_limit(self):
        # Per-file limits aren't enough — a batch of many in-limit files can
        # still blow the disk budget. The total cap trips mid-batch and the
        # whole batch rolls back.
        upload_dir = os.path.join(self.temp_dir, "up")
        with patch.object(server, "_MAX_TOTAL_UPLOAD_BYTES", 30):
            first = self._make_upload("a.txt", b"x" * 20, size=None)  # total 20 ok
            second = self._make_upload("b.txt", b"y" * 20, size=None)  # total 40 > 30
            with self.assertRaises(HTTPException) as ctx:
                asyncio.run(
                    server._save_uploads([first, second], upload_dir=upload_dir)
                )
        self.assertEqual(ctx.exception.status_code, 413)
        self.assertEqual(os.listdir(upload_dir), [])  # nothing left behind

    def test_upload_batch_rolls_back_completed_file_on_later_failure(self):
        # First file is within the limit and fully written; the second blows the
        # limit mid-stream. The already-completed first file must be rolled back
        # too, so a rejected batch leaves the directory empty.
        upload_dir = os.path.join(self.temp_dir, "up")
        with patch.object(server, "_MAX_UPLOAD_BYTES", 50):
            good = self._make_upload("ok.txt", b"y" * 10, size=None)
            bad = self._make_upload("big.txt", b"x" * 200, size=None)
            with self.assertRaises(HTTPException) as ctx:
                asyncio.run(server._save_uploads([good, bad], upload_dir=upload_dir))
        self.assertEqual(ctx.exception.status_code, 413)
        self.assertEqual(os.listdir(upload_dir), [])

    def test_upload_happy_path_streams_all_files(self):
        upload_dir = os.path.join(self.temp_dir, "up")
        f1 = self._make_upload("a.txt", b"alpha", size=None)
        f2 = self._make_upload("b.txt", b"beta", size=None)
        saved = asyncio.run(server._save_uploads([f1, f2], upload_dir=upload_dir))
        self.assertEqual([s["filename"] for s in saved], ["a.txt", "b.txt"])
        # Both on disk under uuid-prefixed names, content intact.
        names = sorted(os.listdir(upload_dir))
        self.assertEqual(len(names), 2)
        self.assertTrue(all("_" in n for n in names))
        with open(saved[0]["stored_path"], "rb") as handle:
            self.assertEqual(handle.read(), b"alpha")

    def test_reflection_defaults_endpoint(self):
        # The UI sources the reflection defaults from here instead of hard-coding
        # them; the values must match reflection.REFLECTION_DEFAULTS.
        from rag_system.agent.reflection import REFLECTION_DEFAULTS

        response = self.client.get("/rag/reflection-defaults")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), dict(REFLECTION_DEFAULTS))
        self.assertEqual(response.json()["max_loops"], 2)

    def test_index_job_timestamps_are_utc(self):
        # index_jobs/index_job_files are written by both backend.database and
        # rag_system.job_persistence (UTC). database.py must also use UTC or a
        # job's timeline mixes local + UTC stamps. created_at should be ~now-UTC.
        from backend.database import _utc_now_iso

        helper = datetime.fromisoformat(_utc_now_iso())
        utc_now = datetime.now(timezone.utc).replace(tzinfo=None)
        self.assertLess(abs((helper - utc_now).total_seconds()), 5)

        index_id = server.db.create_index("Job TZ")
        job = server.db.create_index_job("job-tz-1", index_id, {}, [])
        created = datetime.fromisoformat(job["created_at"])
        # Naive (no offset) and within seconds of UTC now — not shifted by the
        # local-vs-UTC offset the bug produced.
        self.assertIsNone(created.tzinfo)
        self.assertLess(abs((created - utc_now).total_seconds()), 30)

    def test_export_diagnostics_reads_json_body(self):
        # Options used to be query params, so a JSON body was silently ignored
        # (and logs/config exported regardless). They must now come from the body.
        class _Rec:
            def __init__(self):
                self.args = None

            def export_diagnostics_bundle(self, *args):
                self.args = args
                return {"bundle": "ok"}

        rec = _Rec()
        with patch.object(server, "maintenance_tools", rec):
            r1 = self.client.post(
                "/maintenance/export-diagnostics",
                json={"include_logs": False, "include_config": True},
            )
            self.assertEqual(r1.status_code, 200, r1.text)
            self.assertEqual(rec.args, (None, False, True))  # body respected

            rec.args = None
            r2 = self.client.post("/maintenance/export-diagnostics")  # no body
            self.assertEqual(r2.status_code, 200, r2.text)
            self.assertEqual(rec.args, (None, True, True))  # defaults

    def test_vacuum_dry_run_previews_without_running(self):
        tools = MaintenanceTools(db_path=self.db_path, project_root=self.temp_dir)
        result = tools.vacuum_database(dry_run=True)
        self.assertTrue(result["dry_run"])
        self.assertFalse(result["vacuumed"])  # never modifies on a dry run
        self.assertIn("would_vacuum", result)
        self.assertIn("fragmentation_pct", result)

        # The endpoint must pass the flag through.
        class _Rec:
            def __init__(self):
                self.dry_run = None

            def vacuum_database(self, dry_run=False):
                self.dry_run = dry_run
                return {"vacuumed": False, "dry_run": dry_run, "would_vacuum": False}

        rec = _Rec()
        with patch.object(server, "maintenance_tools", rec):
            resp = self.client.post("/maintenance/vacuum-database?dry_run=true")
            self.assertEqual(resp.status_code, 200, resp.text)
            self.assertTrue(rec.dry_run)

    def test_remove_orphan_lancedb_tables_execute_mode(self):
        import uuid as _uuid

        import lancedb

        project_root = os.path.join(self.temp_dir, "orphan-tables")
        lancedb_dir = os.path.join(project_root, "lancedb")
        os.makedirs(lancedb_dir)

        known_id = server.db.create_index("Known index")  # its table must survive
        known_table = f"text_pages_{known_id}"
        orphan_table = f"text_pages_{_uuid.uuid4()}"  # no matching index → orphan
        rows = [{"vector": [0.0, 1.0], "text": "x"}]
        conn = lancedb.connect(lancedb_dir)
        conn.create_table(known_table, rows)
        conn.create_table(orphan_table, rows)

        tools = MaintenanceTools(db_path=self.db_path, project_root=project_root)
        result = tools.remove_orphan_lancedb_tables(dry_run=False)

        self.assertIn(orphan_table, result["dropped"])
        self.assertNotIn(known_table, result["orphans"])
        remaining = conn.list_tables().tables
        self.assertNotIn(orphan_table, remaining)  # orphan gone
        self.assertIn(known_table, remaining)  # known preserved

    def test_concurrent_maintenance_sweeps_are_safe(self):
        # Concurrent maintenance sweeps share one SQLite DB; WAL + busy-timeout
        # must keep them from corrupting/erroring each other.
        project_root = os.path.join(self.temp_dir, "mt-concurrent")
        os.makedirs(os.path.join(project_root, "shared_uploads"))
        tools = MaintenanceTools(db_path=self.db_path, project_root=project_root)

        errors: List[Exception] = []
        results: List[object] = []
        barrier = threading.Barrier(4)
        sweeps = [
            lambda: tools.repair_stuck_builds(older_than_minutes=1),
            lambda: tools.remove_orphan_files(dry_run=True),
            lambda: tools.repair_stuck_builds(older_than_minutes=1),
            lambda: tools.remove_orphan_files(dry_run=True),
        ]

        def run(fn):
            try:
                barrier.wait(timeout=10)
                results.append(fn())
            except Exception as exc:  # noqa: BLE001 — surface, don't swallow
                errors.append(exc)

        threads = [threading.Thread(target=run, args=(fn,)) for fn in sweeps]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

        self.assertEqual(errors, [], f"concurrent maintenance raised: {errors}")
        self.assertEqual(len(results), 4)

    def test_maintenance_endpoints_contract(self):
        response = self.client.get("/maintenance/index-health")
        self.assertEqual(response.status_code, 200)
        health_report = response.json()
        self.assertIn("indexes", health_report)
        self.assertIn("summary", health_report)
        self.assertIsInstance(health_report["indexes"], list)

        repair_response = self.client.post("/maintenance/repair-stuck-builds?older_than_minutes=1")
        self.assertEqual(repair_response.status_code, 200)
        self.assertIn("repaired", repair_response.json())

        orphan_response = self.client.post("/maintenance/remove-orphan-files?dry_run=true")
        self.assertEqual(orphan_response.status_code, 200)
        self.assertIn("orphans_found", orphan_response.json())
        self.assertTrue(orphan_response.json()["dry_run"])

    def test_orphan_scan_preserves_absolute_and_relative_document_paths(self):
        project_root = os.path.join(self.temp_dir, "project")
        uploads_dir = os.path.join(project_root, "shared_uploads")
        os.makedirs(uploads_dir)

        absolute_file = os.path.join(uploads_dir, "absolute.txt")
        relative_file = os.path.join(uploads_dir, "relative.txt")
        orphan_file = os.path.join(uploads_dir, "orphan.txt")
        for path in (absolute_file, relative_file, orphan_file):
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(path)

        absolute_index = server.db.create_index("Absolute path")
        relative_index = server.db.create_index("Relative path")
        server.db.add_document_to_index(absolute_index, "absolute.txt", absolute_file)
        server.db.add_document_to_index(
            relative_index,
            "relative.txt",
            os.path.relpath(relative_file, project_root),
        )

        tools = MaintenanceTools(
            db_path=self.db_path,
            project_root=project_root,
        )
        report = tools.remove_orphan_files(dry_run=True)

        self.assertEqual(report["orphans_found"], 1)
        self.assertEqual(
            [entry["path"] for entry in report["orphan_files"]],
            ["shared_uploads/orphan.txt"],
        )

    def test_broken_index_cleanup_removes_owned_artifacts(self):
        project_root = os.path.join(self.temp_dir, "maintenance-project")
        uploads_dir = os.path.join(project_root, "shared_uploads")
        overview_dir = os.path.join(project_root, "index_store", "overviews")
        cache_dir = os.path.join(project_root, "index_store", "chunk_cache")
        os.makedirs(uploads_dir)
        os.makedirs(overview_dir)
        os.makedirs(cache_dir)

        index_id = server.db.create_index("Broken cleanup")
        healthy_index_id = server.db.create_index("Shared upload owner")
        owned_file = os.path.join(uploads_dir, "owned.txt")
        shared_file = os.path.join(uploads_dir, "shared.txt")
        external_file = os.path.join(self.temp_dir, "external.txt")
        for path in (owned_file, shared_file, external_file):
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("content")
        server.db.add_document_to_index(index_id, "owned.txt", owned_file)
        server.db.add_document_to_index(index_id, "shared.txt", shared_file)
        server.db.add_document_to_index(index_id, "external.txt", external_file)
        server.db.add_document_to_index(
            healthy_index_id,
            "shared.txt",
            os.path.relpath(shared_file, project_root),
        )
        server.db.update_index_metadata(index_id, {"status": "failed"})

        overview_file = os.path.join(overview_dir, f"{index_id}.jsonl")
        cache_file = os.path.join(cache_dir, f"chunks-{index_id}.json")
        for path in (overview_file, cache_file):
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("{}")

        tools = MaintenanceTools(
            db_path=self.db_path,
            project_root=project_root,
        )
        report = tools.delete_broken_indexes(dry_run=False)

        self.assertEqual(report["deleted"], 1, report)
        self.assertIsNone(server.db.get_index(index_id))
        self.assertFalse(os.path.exists(owned_file))
        self.assertTrue(os.path.exists(shared_file))
        self.assertFalse(os.path.exists(overview_file))
        self.assertFalse(os.path.exists(cache_file))
        self.assertTrue(os.path.exists(external_file))

    def test_force_rebuild_prepares_all_files_and_pauses_job(self):
        index_id = server.db.create_index("Force maintenance rebuild")
        job = server.db.create_index_job(
            "force-maintenance-job",
            index_id,
            {"background": True},
            [
                {"filename": "done.txt", "stored_path": "/tmp/done.txt"},
                {"filename": "failed.txt", "stored_path": "/tmp/failed.txt"},
            ],
        )
        job_id = job["id"]
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute(
                "UPDATE index_job_files SET status = 'done' WHERE job_id = ? AND filename = ?",
                (job_id, "done.txt"),
            )
            conn.execute(
                "UPDATE index_job_files SET status = 'failed' WHERE job_id = ? AND filename = ?",
                (job_id, "failed.txt"),
            )
            conn.commit()

        tools = MaintenanceTools(db_path=self.db_path, project_root=self.temp_dir)
        report = tools.rebuild_failed_files_only(index_id, force=True)

        self.assertEqual(report["files_prepared"], 2)
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            statuses = {
                row["status"]
                for row in conn.execute(
                    "SELECT status FROM index_job_files WHERE job_id = ?",
                    (job_id,),
                ).fetchall()
            }
            job_status = conn.execute(
                "SELECT status FROM index_jobs WHERE id = ?",
                (job_id,),
            ).fetchone()["status"]
        self.assertEqual(statuses, {"pending"})
        self.assertEqual(job_status, "paused")

    def test_health_report_marks_missing_vector_table_unhealthy(self):
        project_root = os.path.join(self.temp_dir, "health-project")
        os.makedirs(os.path.join(project_root, "lancedb"))
        index_id = server.db.create_index("Missing vector table")
        server.db.update_index_metadata(index_id, {"status": "functional"})

        tools = MaintenanceTools(db_path=self.db_path, project_root=project_root)
        report = tools.get_index_health_report(index_id)

        self.assertEqual(report["summary"]["unhealthy"], 1)
        self.assertEqual(report["indexes"][0]["health"], "unhealthy")
        self.assertFalse(report["indexes"][0]["vector_store"]["exists"])

    def test_lancedb_table_names_accepts_response_objects(self):
        class ListTablesResponse:
            tables = ["text_pages_alpha", "text_pages_beta"]

            def __iter__(self):
                yield ("tables", self.tables)
                yield ("page_token", None)

        class FakeConnection:
            def list_tables(self):
                return ListTablesResponse()

        self.assertEqual(
            MaintenanceTools._lancedb_table_names(FakeConnection()),
            ["text_pages_alpha", "text_pages_beta"],
        )

    def test_health_deep_uses_local_rag_runtime_and_ollama(self):
        class DummyResponse:
            def __init__(self, status_code):
                self.status_code = status_code

        def fake_get(url, timeout=3):
            return DummyResponse(200)

        fake_lancedb = types.ModuleType("lancedb")
        fake_lancedb.connect = lambda path: None

        with patch.dict(sys.modules, {"lancedb": fake_lancedb}), \
             patch("backend.server.requests.get", side_effect=fake_get), \
             patch("backend.server._lancedb_path_candidates", return_value=[self.temp_dir]), \
             patch("backend.server.os.path.exists", return_value=True):
            response = self.client.get("/health/deep")

        self.assertEqual(response.status_code, 200)
        status = response.json()
        self.assertEqual(status["status"], "ok")
        self.assertEqual(status["checks"]["rag_runtime"], "ready")
        self.assertEqual(status["checks"]["ollama"], "ok")
        self.assertEqual(status["checks"]["db"], "ok")
        self.assertEqual(status["checks"]["lancedb"], "ok")

    def test_fastapi_rag_chat_transport(self):
        expected = {"answer": "local answer", "source_documents": []}
        with (
            patch("backend.server._get_local_rag_agent", return_value=object()),
            patch("backend.server.execute_rag_chat", return_value=expected) as execute,
        ):
            response = self.client.post("/rag/chat", json={"query": "hello"})

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), expected)
        self.assertEqual(execute.call_args.args[2]["query"], "hello")

    def test_fastapi_rag_stream_transport(self):
        def fake_execute(agent, database, data, event_callback):
            event_callback("token", {"text": "hello"})
            return {"answer": "hello", "source_documents": []}

        with (
            patch("backend.server._get_local_rag_agent", return_value=object()),
            patch("backend.server.execute_rag_chat", side_effect=fake_execute),
        ):
            response = self.client.post("/rag/chat/stream", json={"query": "hello"})

        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn('"type": "token"', response.text)
        self.assertIn('"type": "complete"', response.text)

    def test_resume_paused_index_job(self):
        index_id = server.db.create_index("Resume API")
        stored_path = os.path.join(self.temp_dir, "resume-api-doc.txt")
        with open(stored_path, "w", encoding="utf-8") as handle:
            handle.write("resume api document")
        server.db.add_document_to_index(index_id, "resume-api-doc.txt", stored_path)

        job_id = "resume-api-job"
        server.db.create_index_job(
            job_id,
            index_id,
            {"background": True},
            [{"filename": "resume-api-doc.txt", "stored_path": stored_path}],
        )
        server.db.update_index_job(job_id, {"status": "paused", "stage": "paused", "message": "Paused for test"})
        server.job_progress_tracker = SimpleNamespace(
            mark_job_resuming=lambda job_id: {
                "job_id": job_id,
                "status": "queued",
                "message": "Resume queued",
            }
        )

        class DummyThread:
            def __init__(self, *args, **kwargs):
                pass

            def start(self):
                pass

        with patch("backend.server.threading.Thread", DummyThread):
            resume_response = self.client.post(f"/index-jobs/{job_id}/resume")

        self.assertEqual(resume_response.status_code, 200)
        self.assertEqual(resume_response.json()["status"], "queued")
        self.assertIn(job_id, server.index_jobs)
        self.assertEqual(server.index_jobs[job_id]["status"], "queued")

    def test_job_progress_tracker_mark_job_resuming_identifies_incomplete_files(self):
        index_id = server.db.create_index("Crash Resume")
        documents = []
        for filename in ("done-doc.txt", "pending-doc.txt", "failed-doc.txt"):
            stored_path = os.path.join(self.temp_dir, filename)
            with open(stored_path, "w", encoding="utf-8") as handle:
                handle.write(f"content for {filename}")
            server.db.add_document_to_index(index_id, filename, stored_path)
            documents.append({"filename": filename, "stored_path": stored_path})

        job_id = "crash-resume-job"
        server.db.create_index_job(job_id, index_id, {"background": True}, documents)
        server.db.update_index_job(job_id, {"status": "running", "stage": "embedding", "message": "Crashed mid-run"})
        server.db.update_index_job_file(job_id, filename="done-doc.txt", updates={"status": "done"})
        server.db.update_index_job_file(job_id, filename="failed-doc.txt", updates={"status": "failed", "error": "boom"})
        # pending-doc.txt is left at its initial 'pending' status, simulating work that never started

        tracker = JobProgressTracker(self.db_path)
        result = tracker.mark_job_resuming(job_id)

        self.assertEqual(result["job_id"], job_id)
        self.assertEqual(result["status"], "resuming")
        retried_filenames = {entry["filename"] for entry in result["files_to_retry"]}
        self.assertEqual(retried_filenames, {"pending-doc.txt", "failed-doc.txt"})
        self.assertEqual(result["total_files"], 2)

        refreshed_job = server.db.get_index_job(job_id)
        self.assertEqual(refreshed_job["status"], "queued")
        self.assertEqual(refreshed_job["stage"], "queued")
        self.assertEqual(refreshed_job["message"], "Queued for resume after crash")

    def test_mark_job_resuming_reports_missing_job(self):
        tracker = JobProgressTracker(self.db_path)
        result = tracker.mark_job_resuming("does-not-exist")
        self.assertEqual(result, {"error": "Job not found", "job_id": "does-not-exist"})

    def test_startup_recovery_pauses_stuck_building_index_metadata(self):
        index_id = server.db.create_index("Stuck Metadata")
        old_started_at = (datetime.now() - timedelta(minutes=30)).isoformat()
        server.db.update_index_metadata(index_id, {
            "status": "building",
            "build_job_id": "missing-runtime-job",
            "build_started_at": old_started_at,
        })

        recovered = server._recover_stale_index_builds()

        self.assertEqual(recovered, 1)
        metadata = server.db.get_index(index_id)["metadata"]
        self.assertEqual(metadata["status"], "paused")
        self.assertIn("interrupted", metadata["build_error"])

    def test_startup_recovery_syncs_metadata_for_already_paused_job(self):
        index_id = server.db.create_index("Recovered Job Metadata")
        job = server.db.create_index_job(
            "recovered-job-metadata",
            index_id,
            {"background": True},
            [{"filename": "doc.txt", "stored_path": "/tmp/doc.txt"}],
        )
        server.db.update_index_job(job["id"], {"status": "paused"})
        server.db.update_index_metadata(index_id, {
            "status": "building",
            "build_job_id": job["id"],
            "build_started_at": datetime.now().isoformat(),
        })

        recovered = server._recover_stale_index_builds()

        self.assertEqual(recovered, 1)
        metadata = server.db.get_index(index_id)["metadata"]
        self.assertEqual(metadata["status"], "paused")

    def test_stale_job_recovery_resets_interrupted_file_and_stage(self):
        tracker = JobProgressTracker(self.db_path)
        index_id = server.db.create_index("Interrupted file recovery")
        job = server.db.create_index_job(
            "interrupted-file-job",
            index_id,
            {"background": True},
            [{"filename": "doc.txt", "stored_path": "/tmp/doc.txt"}],
        )
        job_id = job["id"]
        old_time = (datetime.utcnow() - timedelta(minutes=30)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            file_id = conn.execute(
                "SELECT id FROM index_job_files WHERE job_id = ?",
                (job_id,),
            ).fetchone()[0]
            conn.execute(
                "UPDATE index_jobs SET status = 'running', updated_at = ? WHERE id = ?",
                (old_time, job_id),
            )
            conn.execute(
                "UPDATE index_job_files SET status = 'processing', stage = 'embedding' WHERE id = ?",
                (file_id,),
            )
            conn.execute(
                """
                INSERT INTO index_job_file_stages
                    (file_id, job_id, stage_name, status, started_at)
                VALUES (?, ?, 'embedding', 'in_progress', ?)
                """,
                (file_id, job_id, old_time),
            )
            conn.commit()

        result = tracker.recover_stale_jobs(older_than_minutes=5)

        self.assertEqual(result["recovered"], 1)
        with sqlite3.connect(self.db_path) as conn:
            file_status = conn.execute(
                "SELECT status, stage FROM index_job_files WHERE id = ?",
                (file_id,),
            ).fetchone()
            stage_status = conn.execute(
                """
                SELECT status FROM index_job_file_stages
                WHERE file_id = ? AND stage_name = 'embedding'
                """,
                (file_id,),
            ).fetchone()[0]
        self.assertEqual(file_status, ("pending", None))
        self.assertEqual(stage_status, "pending")

    def test_all_skipped_resume_still_validates_existing_vector_table(self):
        import_stubs = _install_indexing_pipeline_import_stubs()
        with patch.dict(sys.modules, import_stubs):
            from rag_system.pipelines.indexing_pipeline import IndexingPipeline

        index_id = server.db.create_index("Resume Validation")
        stored_path = os.path.join(self.temp_dir, "resume-doc.txt")
        with open(stored_path, "w", encoding="utf-8") as handle:
            handle.write("resume validation document")
        server.db.add_document_to_index(index_id, "resume-doc.txt", stored_path)
        job = server.db.create_index_job(
            "resume-validation-job",
            index_id,
            {"background": True},
            [{"filename": "resume-doc.txt", "stored_path": stored_path}],
        )
        file_id = job["files"][0]["id"]
        tracker = JobProgressTracker(self.db_path)
        tracker.start_stage(file_id, "resume-validation-job", "storage")
        tracker.complete_stage(file_id, "storage", output_hash="already-stored")

        pipeline = IndexingPipeline.__new__(IndexingPipeline)
        pipeline.config = {
            "db_path": self.db_path,
            "storage": {"text_table_name": "text_pages_resume_validation"},
            "retrievers": {"dense": {"enabled": True}},
        }
        pipeline.incremental_indexer = SimpleNamespace(
            get_index_stats=lambda _index_id: {
                "total_documents": 1,
                "total_chunks": 3,
                "last_indexed": "2026-05-24T00:00:00",
            },
        )
        pipeline.embedding_batch_size = 1
        pipeline.enrichment_batch_size = 1

        with patch.dict(sys.modules, {"rag_system.model_registry": import_stubs["rag_system.model_registry"]}), \
             patch.object(IndexingPipeline, "_start_persistent_worker", return_value=None), \
             patch.object(IndexingPipeline, "_stop_persistent_worker", return_value=None), \
             patch.object(IndexingPipeline, "_validate_built_index", return_value=None) as validate:
            result = pipeline.run(
                [stored_path],
                index_id=index_id,
                incremental=False,
                job_id="resume-validation-job",
            )

        validate.assert_called_once_with("text_pages_resume_validation", expected_dim=None)
        self.assertEqual(result["total_files_considered"], 1)
        self.assertEqual(result["chunks_generated"], 0)


if __name__ == "__main__":
    unittest.main()

import os
import shutil
import sys
import tempfile
import types
import unittest
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.database import ChatDatabase
import backend.server as server
from rag_system.job_persistence import JobProgressTracker


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
        server.db = ChatDatabase(self.db_path)
        self.client = TestClient(server.app)

    def tearDown(self):
        server.db = self.original_db
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

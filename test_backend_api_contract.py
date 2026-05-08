import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.database import ChatDatabase
import backend.server as server


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
                json={"enrichModel": "gpt-oss:120b-cloud", "overviewModel": "qwen3:0.6b"},
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


if __name__ == "__main__":
    unittest.main()

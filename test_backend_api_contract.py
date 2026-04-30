import os
import shutil
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()

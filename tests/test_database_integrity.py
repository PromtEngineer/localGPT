import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("LOCALGPT_DB_PATH", f"/tmp/localgpt-test-{os.getpid()}.db")

from backend.database import ChatDatabase


class DatabaseIntegrityTests(unittest.TestCase):
    def test_session_upload_rows_can_transfer_to_an_index(self):
        with tempfile.TemporaryDirectory() as directory:
            database = ChatDatabase(str(Path(directory) / "chat.db"))
            session_id = database.create_session("Upload transfer", "test-model")
            database.add_document_to_session(session_id, "/tmp/document.pdf")

            database.clear_documents_for_session(session_id)

            self.assertEqual([], database.get_documents_for_session(session_id))

    def test_deleting_session_cascades_to_all_session_owned_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "chat.db"
            database = ChatDatabase(str(database_path))
            session_id = database.create_session("Cascade test", "test-model")
            database.add_message(session_id, "hello", "user")
            database.add_document_to_session(session_id, "/tmp/document.pdf")

            self.assertTrue(database.delete_session(session_id))

            with sqlite3.connect(database_path) as connection:
                message_count = connection.execute(
                    "SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)
                ).fetchone()[0]
                document_count = connection.execute(
                    "SELECT COUNT(*) FROM session_documents WHERE session_id = ?",
                    (session_id,),
                ).fetchone()[0]

            self.assertEqual(0, message_count)
            self.assertEqual(0, document_count)


if __name__ == "__main__":
    unittest.main()

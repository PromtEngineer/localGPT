import ast
import unittest
from pathlib import Path


class ServiceBoundaryTests(unittest.TestCase):
    def test_rag_api_never_writes_chat_messages(self):
        source = Path("rag_system/api_server.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        message_writes = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_message"
        ]

        self.assertEqual([], message_writes)

    def test_index_deletion_is_delegated_to_rag_artifact_owner(self):
        backend_source = Path("backend/server.py").read_text(encoding="utf-8")
        rag_source = Path("rag_system/api_server.py").read_text(encoding="utf-8")

        self.assertIn(
            'requests.delete(f"{RAG_API_URL}/indexes/{index_id}"', backend_source
        )
        self.assertIn("def handle_delete_index_artifacts", rag_source)
        self.assertIn('f"{table_name}_lc"', rag_source)

    def test_browser_facing_chat_is_session_owned(self):
        backend_source = Path("backend/server.py").read_text(encoding="utf-8")

        self.assertNotIn("if parsed_path.path == '/chat':", backend_source)
        self.assertIn('"model": (str, "model")', backend_source)
        self.assertIn('"force_rag": (bool, "force_rag")', backend_source)


if __name__ == "__main__":
    unittest.main()

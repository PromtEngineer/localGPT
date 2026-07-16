import tempfile
import unittest
from pathlib import Path

from localgpt_runtime import (
    UploadRejected,
    inspect_upload_content,
    safe_upload_path,
    validate_index_file_paths,
)


class SafeUploadPathTests(unittest.TestCase):
    def test_rejects_parent_directory_traversal(self):
        with tempfile.TemporaryDirectory() as upload_dir:
            with self.assertRaises(UploadRejected):
                safe_upload_path(Path(upload_dir), "../../outside.txt")

    def test_indexing_rejects_server_files_outside_upload_directory(self):
        with tempfile.TemporaryDirectory() as upload_dir, tempfile.NamedTemporaryFile(
            suffix=".txt"
        ) as outside:
            with self.assertRaises(UploadRejected):
                validate_index_file_paths([outside.name], Path(upload_dir))

    def test_rejects_executable_content_disguised_as_text(self):
        with self.assertRaises(UploadRejected):
            inspect_upload_content("notes.txt", b"MZ\x00\x00binary")

    def test_rejects_invalid_pdf_signature(self):
        with self.assertRaises(UploadRejected):
            inspect_upload_content("paper.pdf", b"not a pdf")


if __name__ == "__main__":
    unittest.main()

import sqlite3
import tempfile
from pathlib import Path

import pytest

from backend.agent_runtime.data_tools import ReadOnlyDatabase, UnsafeQuery


def test_database_tool_enforces_read_only_queries() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "data.db"
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE facts (value TEXT)")
            connection.execute("INSERT INTO facts VALUES ('kept')")

        database = ReadOnlyDatabase(f"sqlite:///{path}")

        with pytest.raises(UnsafeQuery):
            database.query("DELETE FROM facts")
        assert database.query("SELECT value FROM facts")["rows"] == [["kept"]]

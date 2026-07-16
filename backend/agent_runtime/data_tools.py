from __future__ import annotations

import asyncio
import csv
import io
import json
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from backend.agent_runtime.artifacts import ArtifactStore
from backend.agent_runtime.tools import ToolContext, ToolRegistry, ToolSpec


class UnsafeQuery(ValueError):
    pass


_LEADING_COMMENTS = re.compile(r"^(?:\s|--[^\n]*(?:\n|$)|/\*.*?\*/)*", re.DOTALL)


def validate_read_only_sql(query: str) -> str:
    normalized = _LEADING_COMMENTS.sub("", query).strip()
    if not normalized:
        raise UnsafeQuery("A SQL query is required")
    statements = [part.strip() for part in normalized.split(";") if part.strip()]
    if len(statements) != 1:
        raise UnsafeQuery("Exactly one SQL statement is allowed")
    first = statements[0].split(None, 1)[0].upper()
    if first not in {"SELECT", "WITH", "EXPLAIN", "PRAGMA"}:
        raise UnsafeQuery("Only read-only SELECT, WITH, EXPLAIN, and PRAGMA queries are allowed")
    if first == "PRAGMA" and not re.match(
        r"(?is)^PRAGMA\s+(table_info|table_list|index_list|foreign_key_list)\s*\(",
        statements[0],
    ):
        raise UnsafeQuery("Only schema-inspection PRAGMAs are allowed")
    return statements[0]


@dataclass(frozen=True, slots=True)
class DatabaseConnector:
    id: str
    url: str
    description: str = ""
    max_rows: int = 1000


class ReadOnlyDatabase:
    def __init__(self, url: str, *, max_rows: int = 1000) -> None:
        self.url = url
        self.max_rows = max_rows

    def _sqlite_path(self) -> Path:
        parsed = urlsplit(self.url)
        if parsed.scheme != "sqlite":
            raise ValueError("This operation requires a sqlite connector")
        raw_path = unquote(parsed.path)
        if raw_path.startswith("//"):
            raw_path = raw_path[1:]
        return Path(raw_path).resolve()

    @staticmethod
    def _sqlite_authorizer(action: int, *_args: object) -> int:
        denied = {
            sqlite3.SQLITE_INSERT,
            sqlite3.SQLITE_UPDATE,
            sqlite3.SQLITE_DELETE,
            sqlite3.SQLITE_ALTER_TABLE,
            sqlite3.SQLITE_DROP_TABLE,
            sqlite3.SQLITE_DROP_INDEX,
            sqlite3.SQLITE_CREATE_TABLE,
            sqlite3.SQLITE_CREATE_INDEX,
            sqlite3.SQLITE_ATTACH,
            sqlite3.SQLITE_DETACH,
        }
        return sqlite3.SQLITE_DENY if action in denied else sqlite3.SQLITE_OK

    def query(self, query: str) -> dict[str, Any]:
        statement = validate_read_only_sql(query)
        if self.url.startswith("sqlite:"):
            path = self._sqlite_path()
            uri = f"file:{path}?mode=ro"
            with sqlite3.connect(uri, uri=True, timeout=10) as connection:
                connection.execute("PRAGMA query_only = ON")
                connection.set_authorizer(self._sqlite_authorizer)
                cursor = connection.execute(statement)
                columns = [item[0] for item in cursor.description or []]
                rows = cursor.fetchmany(self.max_rows + 1)
            truncated = len(rows) > self.max_rows
            rows = rows[: self.max_rows]
            return {
                "columns": columns,
                "rows": [list(row) for row in rows],
                "row_count": len(rows),
                "truncated": truncated,
            }

        try:
            from sqlalchemy import create_engine, text
        except ImportError as exc:
            raise RuntimeError(
                "Remote database connectors require the optional sqlalchemy dependency"
            ) from exc
        engine = create_engine(self.url, pool_pre_ping=True)
        try:
            with engine.connect() as connection:
                transaction = connection.begin()
                try:
                    dialect = engine.dialect.name
                    if dialect == "postgresql":
                        connection.execute(text("SET TRANSACTION READ ONLY"))
                    result = connection.execute(text(statement))
                    columns = list(result.keys())
                    rows = result.fetchmany(self.max_rows + 1)
                finally:
                    transaction.rollback()
        finally:
            engine.dispose()
        return {
            "columns": columns,
            "rows": [list(row) for row in rows[: self.max_rows]],
            "row_count": min(len(rows), self.max_rows),
            "truncated": len(rows) > self.max_rows,
        }

    def schema(self) -> dict[str, Any]:
        if self.url.startswith("sqlite:"):
            tables = self.query(
                "SELECT name, type FROM sqlite_master "
                "WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            )["rows"]
            output = []
            for name, kind in tables:
                columns = self.query(f"PRAGMA table_info({json.dumps(name)})")["rows"]
                output.append(
                    {
                        "name": name,
                        "type": kind,
                        "columns": [
                            {"name": row[1], "type": row[2], "nullable": not bool(row[3])}
                            for row in columns
                        ],
                    }
                )
            return {"objects": output}
        try:
            from sqlalchemy import create_engine, inspect
        except ImportError as exc:
            raise RuntimeError("Schema inspection requires sqlalchemy") from exc
        engine = create_engine(self.url, pool_pre_ping=True)
        try:
            inspector = inspect(engine)
            return {
                "objects": [
                    {
                        "name": name,
                        "type": "table",
                        "columns": inspector.get_columns(name),
                    }
                    for name in inspector.get_table_names()
                ]
            }
        finally:
            engine.dispose()


class DatabaseConnectorRegistry:
    def __init__(self, connectors: dict[str, DatabaseConnector] | None = None) -> None:
        self.connectors = connectors or self._from_environment()

    @staticmethod
    def _from_environment() -> dict[str, DatabaseConnector]:
        raw = os.getenv("LOCALGPT_DATABASE_CONNECTORS_JSON", "{}")
        loaded = json.loads(raw)
        return {
            connector_id: DatabaseConnector(
                id=connector_id,
                url=str(config["url"]),
                description=str(config.get("description", "")),
                max_rows=int(config.get("max_rows", 1000)),
            )
            for connector_id, config in loaded.items()
        }

    def get(self, connector_id: str) -> DatabaseConnector:
        try:
            return self.connectors[connector_id]
        except KeyError as exc:
            raise KeyError(f"Unknown database connector: {connector_id}") from exc


def analyze_csv(content: bytes, *, max_rows: int = 10_000) -> dict[str, Any]:
    text = content.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    rows = []
    for position, row in enumerate(reader):
        if position >= max_rows:
            break
        rows.append(dict(row))
    columns = reader.fieldnames or []
    summaries: dict[str, Any] = {}
    for column in columns:
        values = [row.get(column, "") for row in rows]
        numbers: list[float] = []
        for value in values:
            try:
                numbers.append(float(value))
            except (TypeError, ValueError):
                pass
        if values and len(numbers) / len(values) >= 0.8:
            summaries[column] = {
                "kind": "numeric",
                "count": len(numbers),
                "min": min(numbers) if numbers else None,
                "max": max(numbers) if numbers else None,
                "mean": sum(numbers) / len(numbers) if numbers else None,
            }
        else:
            distinct = list(dict.fromkeys(str(value) for value in values if value != ""))
            summaries[column] = {
                "kind": "text",
                "count": len(values),
                "distinct_count": len(distinct),
                "examples": distinct[:10],
            }
    return {
        "columns": columns,
        "row_count": len(rows),
        "truncated": len(rows) >= max_rows,
        "summaries": summaries,
        "sample": rows[:20],
    }


def register_data_tools(
    registry: ToolRegistry,
    *,
    connectors: DatabaseConnectorRegistry,
    artifacts: ArtifactStore,
) -> None:
    async def database_schema(
        arguments: dict[str, Any], _context: ToolContext
    ) -> dict[str, Any]:
        connector = connectors.get(arguments["connector_id"])
        return await asyncio.to_thread(
            ReadOnlyDatabase(connector.url, max_rows=connector.max_rows).schema
        )

    async def database_query(
        arguments: dict[str, Any], _context: ToolContext
    ) -> dict[str, Any]:
        connector = connectors.get(arguments["connector_id"])
        return await asyncio.to_thread(
            ReadOnlyDatabase(connector.url, max_rows=connector.max_rows).query,
            arguments["query"],
        )

    async def tabular_analysis(
        arguments: dict[str, Any], context: ToolContext
    ) -> dict[str, Any]:
        artifact = artifacts.get(arguments["artifact_id"])
        if artifact is None or artifact.session_id != context.session_id:
            raise KeyError("Artifact not found in this session")
        return await asyncio.to_thread(analyze_csv, artifacts.read_bytes(artifact.id))

    connector_schema = {
        "type": "object",
        "properties": {"connector_id": {"type": "string"}},
        "required": ["connector_id"],
        "additionalProperties": False,
    }
    registry.register(
        ToolSpec(
            name="database_schema",
            description="Inspect an allowlisted read-only database connector.",
            input_schema=connector_schema,
            handler=database_schema,
            required_permissions=frozenset({"database:read"}),
        )
    )
    registry.register(
        ToolSpec(
            name="database_query",
            description="Execute one validated read-only SQL query.",
            input_schema={
                "type": "object",
                "properties": {
                    "connector_id": {"type": "string"},
                    "query": {"type": "string", "minLength": 1},
                },
                "required": ["connector_id", "query"],
                "additionalProperties": False,
            },
            handler=database_query,
            required_permissions=frozenset({"database:read"}),
            timeout_seconds=30,
        )
    )
    registry.register(
        ToolSpec(
            name="tabular_analysis",
            description="Profile and summarize a CSV artifact from this session.",
            input_schema={
                "type": "object",
                "properties": {"artifact_id": {"type": "string"}},
                "required": ["artifact_id"],
                "additionalProperties": False,
            },
            handler=tabular_analysis,
            required_permissions=frozenset({"artifact:read"}),
        )
    )

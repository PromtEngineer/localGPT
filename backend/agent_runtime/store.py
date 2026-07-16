from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backend.agent_runtime.models import Run, RunEvent, RunStatus


def _now() -> str:
    return datetime.now(UTC).isoformat()


_TERMINAL = {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}
_ALLOWED_TRANSITIONS = {
    RunStatus.QUEUED: {RunStatus.RUNNING, RunStatus.CANCELLED, RunStatus.FAILED},
    RunStatus.RUNNING: {
        RunStatus.WAITING,
        RunStatus.COMPLETED,
        RunStatus.CANCELLED,
        RunStatus.FAILED,
    },
    RunStatus.WAITING: {
        RunStatus.RUNNING,
        RunStatus.CANCELLED,
        RunStatus.FAILED,
    },
}


class RunStore:
    """SQLite-backed run/event store with replay and cancellation semantics."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS agent_runs (
                    id TEXT PRIMARY KEY,
                    session_id TEXT,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    request TEXT NOT NULL,
                    result TEXT,
                    error TEXT,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT,
                    metadata TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_agent_runs_status
                    ON agent_runs(status, updated_at);
                CREATE INDEX IF NOT EXISTS idx_agent_runs_session
                    ON agent_runs(session_id, created_at);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_runs_idempotency
                    ON agent_runs(COALESCE(session_id, ''), kind,
                        json_extract(metadata, '$.idempotency_key'))
                    WHERE json_extract(metadata, '$.idempotency_key') IS NOT NULL;

                CREATE TABLE IF NOT EXISTS run_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    data TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES agent_runs(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_run_events_replay
                    ON run_events(run_id, id);

                CREATE TABLE IF NOT EXISTS run_checkpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES agent_runs(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_run_checkpoints_latest
                    ON run_checkpoints(run_id, id DESC);

                CREATE TABLE IF NOT EXISTS tool_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES agent_runs(id) ON DELETE CASCADE
                );
                """
            )

    def create_run(
        self,
        *,
        session_id: str | None,
        request: dict[str, Any],
        kind: str = "message",
        metadata: dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> Run:
        idempotency_key = (metadata or {}).get("idempotency_key")
        if idempotency_key:
            existing = self.find_idempotent(
                session_id=session_id,
                kind=kind,
                idempotency_key=str(idempotency_key),
            )
            if existing is not None:
                return existing
        identifier = run_id or str(uuid.uuid4())
        now = _now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO agent_runs (
                    id, session_id, kind, status, request, created_at, updated_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    identifier,
                    session_id,
                    kind,
                    RunStatus.QUEUED.value,
                    json.dumps(request),
                    now,
                    now,
                    json.dumps(metadata or {}),
                ),
            )
        run = self.get_run(identifier)
        assert run is not None
        return run

    def find_idempotent(
        self, *, session_id: str | None, kind: str, idempotency_key: str
    ) -> Run | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM agent_runs
                WHERE COALESCE(session_id, '') = COALESCE(?, '')
                  AND kind = ?
                  AND json_extract(metadata, '$.idempotency_key') = ?
                LIMIT 1
                """,
                (session_id, kind, idempotency_key),
            ).fetchone()
        return self._to_run(row) if row else None

    def get_run(self, run_id: str) -> Run | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM agent_runs WHERE id = ?", (run_id,)
            ).fetchone()
        return self._to_run(row) if row else None

    def append_event(
        self, run_id: str, event_type: str, data: dict[str, Any]
    ) -> RunEvent:
        now = _now()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO run_events (run_id, event_type, data, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (run_id, event_type, json.dumps(data), now),
            )
            event_id = int(cursor.lastrowid)
        return RunEvent(event_id, run_id, event_type, data, now)

    def transition(
        self,
        run_id: str,
        status: RunStatus,
        *,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> Run:
        current = self.get_run(run_id)
        if current is None:
            raise KeyError(f"Run not found: {run_id}")
        if current.status != status and status not in _ALLOWED_TRANSITIONS.get(
            current.status, set()
        ):
            raise ValueError(
                f"Invalid run transition: {current.status.value} -> {status.value}"
            )
        now = _now()
        completed_at = now if status in _TERMINAL else None
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE agent_runs
                SET status = ?, result = COALESCE(?, result), error = ?,
                    updated_at = ?, completed_at = ?
                WHERE id = ?
                """,
                (
                    status.value,
                    json.dumps(result) if result is not None else None,
                    error,
                    now,
                    completed_at,
                    run_id,
                ),
            )
        updated = self.get_run(run_id)
        assert updated is not None
        return updated

    def request_cancel(self, run_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE agent_runs SET cancel_requested = 1, updated_at = ?
                WHERE id = ? AND status NOT IN (?, ?, ?)
                """,
                (
                    _now(),
                    run_id,
                    RunStatus.COMPLETED.value,
                    RunStatus.FAILED.value,
                    RunStatus.CANCELLED.value,
                ),
            )
            return cursor.rowcount > 0

    def is_cancel_requested(self, run_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT cancel_requested FROM agent_runs WHERE id = ?", (run_id,)
            ).fetchone()
        return bool(row and row["cancel_requested"])

    def list_runs(
        self,
        *,
        session_id: str | None = None,
        kind: str | None = None,
        limit: int = 100,
    ) -> list[Run]:
        clauses: list[str] = []
        values: list[Any] = []
        if session_id is not None:
            clauses.append("session_id = ?")
            values.append(session_id)
        if kind is not None:
            clauses.append("kind = ?")
            values.append(kind)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        values.append(limit)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM agent_runs{where} ORDER BY created_at DESC LIMIT ?",
                values,
            ).fetchall()
        return [self._to_run(row) for row in rows]

    def delete_run(self, run_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM agent_runs WHERE id = ?", (run_id,))
            return cursor.rowcount > 0

    def reconcile_interrupted_runs(self) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id FROM agent_runs WHERE status IN (?, ?)",
                (RunStatus.RUNNING.value, RunStatus.WAITING.value),
            ).fetchall()
        recovered = [str(row["id"]) for row in rows]
        for run_id in recovered:
            self.transition(
                run_id,
                RunStatus.FAILED,
                error="Execution interrupted by service restart; retry the run.",
            )
            self.append_event(
                run_id,
                "run.failed",
                {"error": "Execution interrupted by service restart"},
            )
        return recovered

    def save_checkpoint(self, run_id: str, state: dict[str, Any]) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO run_checkpoints (run_id, state, created_at)
                VALUES (?, ?, ?)
                """,
                (run_id, json.dumps(state), _now()),
            )
            return int(cursor.lastrowid)

    def load_checkpoint(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT state FROM run_checkpoints
                WHERE run_id = ? ORDER BY id DESC LIMIT 1
                """,
                (run_id,),
            ).fetchone()
        return json.loads(row["state"]) if row else None

    def audit_tool(
        self,
        run_id: str,
        tool_name: str,
        phase: str,
        payload: dict[str, Any],
    ) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO tool_audit (run_id, tool_name, phase, payload, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, tool_name, phase, json.dumps(payload), _now()),
            )
            return int(cursor.lastrowid)

    def list_events(
        self, run_id: str, *, after_id: int = 0, limit: int = 1000
    ) -> list[RunEvent]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, run_id, event_type, data, created_at
                FROM run_events
                WHERE run_id = ? AND id > ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (run_id, after_id, limit),
            ).fetchall()
        return [
            RunEvent(
                id=int(row["id"]),
                run_id=str(row["run_id"]),
                type=str(row["event_type"]),
                data=json.loads(row["data"]),
                created_at=str(row["created_at"]),
            )
            for row in rows
        ]

    @staticmethod
    def _to_run(row: sqlite3.Row) -> Run:
        return Run(
            id=str(row["id"]),
            session_id=row["session_id"],
            kind=str(row["kind"]),
            status=RunStatus(row["status"]),
            request=json.loads(row["request"]),
            result=json.loads(row["result"]) if row["result"] else None,
            error=row["error"],
            cancel_requested=bool(row["cancel_requested"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            completed_at=row["completed_at"],
            metadata=json.loads(row["metadata"] or "{}"),
        )

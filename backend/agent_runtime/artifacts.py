from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import sqlite3
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol


class BlobStore(Protocol):
    def put_if_absent(self, key: str, content: bytes) -> None: ...
    def read(self, key: str) -> bytes: ...


class LocalBlobStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        path.relative_to(self.root)
        return path

    def put_if_absent(self, key: str, content: bytes) -> None:
        destination = self._path(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            return
        handle, temporary_name = tempfile.mkstemp(
            dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(handle, "wb") as temporary:
                temporary.write(content)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_name, destination)
        finally:
            Path(temporary_name).unlink(missing_ok=True)

    def read(self, key: str) -> bytes:
        return self._path(key).read_bytes()


class S3BlobStore:
    """Optional S3-compatible adapter; credentials are read by the AWS SDK."""

    def __init__(self, bucket: str, *, prefix: str = "", endpoint_url: str | None = None) -> None:
        try:
            import boto3
            from botocore.exceptions import ClientError
        except ImportError as exc:
            raise RuntimeError("S3 artifact storage requires the optional boto3 dependency") from exc
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.client = boto3.client("s3", endpoint_url=endpoint_url)
        self.client_error = ClientError

    def _key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def put_if_absent(self, key: str, content: bytes) -> None:
        resolved = self._key(key)
        try:
            self.client.head_object(Bucket=self.bucket, Key=resolved)
            return
        except self.client_error as exc:
            if exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode") != 404:
                raise
        self.client.put_object(Bucket=self.bucket, Key=resolved, Body=content)

    def read(self, key: str) -> bytes:
        response = self.client.get_object(Bucket=self.bucket, Key=self._key(key))
        return response["Body"].read()


@dataclass(frozen=True, slots=True)
class Artifact:
    id: str
    filename: str
    mime_type: str
    size_bytes: int
    sha256: str
    storage_key: str
    created_at: str
    session_id: str | None = None
    index_id: str | None = None
    run_id: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)


class ArtifactStore:
    """Content-addressed local artifact storage with SQLite metadata."""

    def __init__(
        self,
        db_path: str | Path,
        object_root: str | Path | None = None,
        *,
        blob_store: BlobStore | None = None,
    ) -> None:
        self.db_path = str(db_path)
        if blob_store is None and object_root is None:
            raise ValueError("object_root or blob_store is required")
        self.object_root = Path(object_root).resolve() if object_root is not None else None
        self.blob_store = blob_store or LocalBlobStore(self.object_root)  # type: ignore[arg-type]
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS artifacts (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    mime_type TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    sha256 TEXT NOT NULL,
                    storage_key TEXT NOT NULL,
                    session_id TEXT,
                    index_id TEXT,
                    run_id TEXT,
                    provenance TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_artifacts_hash ON artifacts(sha256);
                CREATE INDEX IF NOT EXISTS idx_artifacts_session
                    ON artifacts(session_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_artifacts_index
                    ON artifacts(index_id, created_at);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def put_bytes(
        self,
        content: bytes,
        *,
        filename: str,
        mime_type: str | None = None,
        session_id: str | None = None,
        index_id: str | None = None,
        run_id: str | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> Artifact:
        digest = hashlib.sha256(content).hexdigest()
        storage_key = f"sha256/{digest[:2]}/{digest}"
        self.blob_store.put_if_absent(storage_key, content)

        artifact_id = str(uuid.uuid4())
        created_at = datetime.now(UTC).isoformat()
        resolved_mime = (
            mime_type
            or mimetypes.guess_type(filename)[0]
            or "application/octet-stream"
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO artifacts (
                    id, filename, mime_type, size_bytes, sha256, storage_key,
                    session_id, index_id, run_id, provenance, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact_id,
                    Path(filename).name,
                    resolved_mime,
                    len(content),
                    digest,
                    storage_key,
                    session_id,
                    index_id,
                    run_id,
                    json.dumps(provenance or {}),
                    created_at,
                ),
            )
        artifact = self.get(artifact_id)
        assert artifact is not None
        return artifact

    def put_path(self, path: str | Path, **metadata: Any) -> Artifact:
        source = Path(path)
        return self.put_bytes(source.read_bytes(), filename=source.name, **metadata)

    def get(self, artifact_id: str) -> Artifact | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM artifacts WHERE id = ?", (artifact_id,)
            ).fetchone()
        return self._from_row(row) if row else None

    def list(
        self,
        *,
        session_id: str | None = None,
        index_id: str | None = None,
        limit: int = 100,
    ) -> list[Artifact]:
        clauses: list[str] = []
        values: list[Any] = []
        if session_id is not None:
            clauses.append("session_id = ?")
            values.append(session_id)
        if index_id is not None:
            clauses.append("index_id = ?")
            values.append(index_id)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        values.append(limit)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM artifacts{where} ORDER BY created_at DESC LIMIT ?",
                values,
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def read_bytes(self, artifact_id: str) -> bytes:
        artifact = self.get(artifact_id)
        if artifact is None:
            raise KeyError(f"Artifact not found: {artifact_id}")
        return self.blob_store.read(artifact.storage_key)

    @staticmethod
    def _from_row(row: sqlite3.Row) -> Artifact:
        return Artifact(
            id=str(row["id"]),
            filename=str(row["filename"]),
            mime_type=str(row["mime_type"]),
            size_bytes=int(row["size_bytes"]),
            sha256=str(row["sha256"]),
            storage_key=str(row["storage_key"]),
            session_id=row["session_id"],
            index_id=row["index_id"],
            run_id=row["run_id"],
            provenance=json.loads(row["provenance"] or "{}"),
            created_at=str(row["created_at"]),
        )

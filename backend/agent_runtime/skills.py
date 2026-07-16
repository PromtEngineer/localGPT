from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class SkillVersion:
    skill_id: str
    version: str
    name: str
    description: str
    content: str
    instructions: str
    allowed_tools: list[str]
    created_at: str


def parse_skill(content: str) -> tuple[dict[str, Any], str]:
    match = re.match(r"\A---\s*\n(.*?)\n---\s*\n?(.*)\Z", content, re.DOTALL)
    if not match:
        raise ValueError("Skill content must contain YAML frontmatter")
    metadata = yaml.safe_load(match.group(1)) or {}
    instructions = match.group(2).strip()
    name = metadata.get("name")
    description = metadata.get("description")
    if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{1,63}", name):
        raise ValueError("Skill name must be a lowercase identifier")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("Skill description is required")
    allowed = metadata.get("allowed_tools", [])
    if not isinstance(allowed, list) or not all(isinstance(item, str) for item in allowed):
        raise ValueError("allowed_tools must be a list of tool names")
    if not instructions:
        raise ValueError("Skill instructions are required")
    return metadata, instructions


class SkillStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS skills (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT NOT NULL,
                    current_version TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS skill_versions (
                    skill_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    content TEXT NOT NULL,
                    instructions TEXT NOT NULL,
                    allowed_tools TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(skill_id, version),
                    FOREIGN KEY(skill_id) REFERENCES skills(id) ON DELETE CASCADE
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def create(self, content: str) -> SkillVersion:
        metadata, instructions = parse_skill(content)
        skill_id = str(uuid.uuid4())
        version = self._version(content)
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO skills VALUES (?, ?, ?, ?, ?, ?)",
                (
                    skill_id,
                    metadata["name"],
                    metadata["description"],
                    version,
                    now,
                    now,
                ),
            )
            self._insert_version(
                connection, skill_id, version, content, metadata, instructions, now
            )
        result = self.get_version(skill_id, version)
        assert result is not None
        return result

    def create_version(self, skill_id: str, content: str) -> SkillVersion:
        metadata, instructions = parse_skill(content)
        version = self._version(content)
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            skill = connection.execute(
                "SELECT name FROM skills WHERE id = ?", (skill_id,)
            ).fetchone()
            if skill is None:
                raise KeyError("Skill not found")
            if skill["name"] != metadata["name"]:
                raise ValueError("A skill version cannot change the skill name")
            self._insert_version(
                connection, skill_id, version, content, metadata, instructions, now
            )
            connection.execute(
                """
                UPDATE skills SET description = ?, current_version = ?, updated_at = ?
                WHERE id = ?
                """,
                (metadata["description"], version, now, skill_id),
            )
        result = self.get_version(skill_id, version)
        assert result is not None
        return result

    @staticmethod
    def _insert_version(
        connection: sqlite3.Connection,
        skill_id: str,
        version: str,
        content: str,
        metadata: dict[str, Any],
        instructions: str,
        created_at: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO skill_versions (
                skill_id, version, name, description, content, instructions,
                allowed_tools, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                skill_id,
                version,
                metadata["name"],
                metadata["description"],
                content,
                instructions,
                json.dumps(metadata.get("allowed_tools", [])),
                created_at,
            ),
        )

    def get_version(self, skill_id: str, version: str) -> SkillVersion | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM skill_versions WHERE skill_id = ? AND version = ?
                """,
                (skill_id, version),
            ).fetchone()
        return self._from_row(row) if row else None

    def get_current(self, skill_id: str) -> SkillVersion | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT v.* FROM skills s JOIN skill_versions v
                  ON v.skill_id = s.id AND v.version = s.current_version
                WHERE s.id = ?
                """,
                (skill_id,),
            ).fetchone()
        return self._from_row(row) if row else None

    def list(self) -> list[SkillVersion]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT v.* FROM skills s JOIN skill_versions v
                  ON v.skill_id = s.id AND v.version = s.current_version
                ORDER BY s.name
                """
            ).fetchall()
        return [self._from_row(row) for row in rows]

    @staticmethod
    def _version(content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    @staticmethod
    def _from_row(row: sqlite3.Row) -> SkillVersion:
        return SkillVersion(
            skill_id=str(row["skill_id"]),
            version=str(row["version"]),
            name=str(row["name"]),
            description=str(row["description"]),
            content=str(row["content"]),
            instructions=str(row["instructions"]),
            allowed_tools=json.loads(row["allowed_tools"]),
            created_at=str(row["created_at"]),
        )

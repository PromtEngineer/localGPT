from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path


class Store:
    """Tiny SQLite store for chat sessions and their messages. Single local user."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY, title TEXT, scope TEXT, mode TEXT,
                created REAL, updated REAL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY, session_id TEXT, role TEXT,
                content TEXT, meta TEXT, created REAL
            );
            """
        )
        self.db.commit()

    def _now(self) -> float:
        return time.time()

    def create_session(self, title: str, scope: list[str], mode: str = "auto") -> dict:
        sid = uuid.uuid4().hex[:12]
        t = self._now()
        self.db.execute(
            "INSERT INTO sessions VALUES (?,?,?,?,?,?)",
            (sid, title, json.dumps(scope), mode, t, t),
        )
        self.db.commit()
        return self.get_session(sid)

    def list_sessions(self) -> list[dict]:
        rows = self.db.execute("SELECT * FROM sessions ORDER BY updated DESC").fetchall()
        out = []
        for r in rows:
            n = self.db.execute(
                "SELECT count(*) c FROM messages WHERE session_id=? AND role='user'", (r["id"],)
            ).fetchone()["c"]
            out.append({**self._session_dict(r), "turns": n})
        return out

    def _session_dict(self, r: sqlite3.Row) -> dict:
        return {
            "id": r["id"], "title": r["title"], "scope": json.loads(r["scope"]),
            "mode": r["mode"], "created": r["created"], "updated": r["updated"],
        }

    def get_session(self, sid: str) -> dict | None:
        r = self.db.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
        if not r:
            return None
        msgs = self.db.execute(
            "SELECT * FROM messages WHERE session_id=? ORDER BY created", (sid,)
        ).fetchall()
        return {
            **self._session_dict(r),
            "messages": [
                {"id": m["id"], "role": m["role"], "content": m["content"],
                 "meta": json.loads(m["meta"] or "{}"), "created": m["created"]}
                for m in msgs
            ],
        }

    def update_session(self, sid: str, *, title: str | None = None, scope: list[str] | None = None,
                       mode: str | None = None) -> None:
        cur = self.db.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
        if not cur:
            return
        self.db.execute(
            "UPDATE sessions SET title=?, scope=?, mode=?, updated=? WHERE id=?",
            (title or cur["title"], json.dumps(scope) if scope is not None else cur["scope"],
             mode or cur["mode"], self._now(), sid),
        )
        self.db.commit()

    def add_message(self, sid: str, role: str, content: str, meta: dict | None = None) -> dict:
        mid = uuid.uuid4().hex[:12]
        t = self._now()
        self.db.execute(
            "INSERT INTO messages VALUES (?,?,?,?,?,?)",
            (mid, sid, role, content, json.dumps(meta or {}), t),
        )
        self.db.execute("UPDATE sessions SET updated=? WHERE id=?", (t, sid))
        self.db.commit()
        return {"id": mid, "role": role, "content": content, "meta": meta or {}, "created": t}

    def delete_session(self, sid: str) -> None:
        self.db.execute("DELETE FROM messages WHERE session_id=?", (sid,))
        self.db.execute("DELETE FROM sessions WHERE id=?", (sid,))
        self.db.commit()

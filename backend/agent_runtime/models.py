from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class Run:
    id: str
    session_id: str | None
    kind: str
    status: RunStatus
    request: dict[str, Any]
    result: dict[str, Any] | None
    error: str | None
    cancel_requested: bool
    created_at: str
    updated_at: str
    completed_at: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RunEvent:
    id: int
    run_id: str
    type: str
    data: dict[str, Any]
    created_at: str

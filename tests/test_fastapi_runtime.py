import time

from fastapi.testclient import TestClient

from backend.api import Runtime, create_app
from backend.agent_runtime.artifacts import ArtifactStore
from backend.agent_runtime.executor import AgentExecutor
from backend.agent_runtime.providers import AssistantTurn
from backend.agent_runtime.service import RunManager
from backend.agent_runtime.skills import SkillStore
from backend.agent_runtime.store import RunStore
from backend.agent_runtime.tools import ToolRegistry, ToolSpec
from backend.database import ChatDatabase


class Provider:
    async def complete(self, **kwargs):
        return AssistantTurn(content="runtime answer", input_tokens=2, output_tokens=2)

    async def discover_models(self):
        return []


def build_runtime(tmp_path):
    chat = ChatDatabase(str(tmp_path / "chat.sqlite"))
    runs = RunStore(tmp_path / "runs.sqlite")
    artifacts = ArtifactStore(tmp_path / "artifacts.sqlite", tmp_path / "objects")
    skills = SkillStore(tmp_path / "skills.sqlite")
    tools = ToolRegistry()
    provider = Provider()
    manager = RunManager(
        store=runs,
        executor=AgentExecutor(provider=provider, tools=tools),
        tools=tools,
        database=chat,
        skills=skills,
    )
    return Runtime(chat, runs, artifacts, skills, tools, provider, manager)


def test_message_run_is_persisted_and_sse_can_replay(tmp_path):
    client = TestClient(create_app(build_runtime(tmp_path)))
    session = client.post(
        "/sessions", json={"title": "test", "model": "unit-model"}
    ).json()["session"]

    submitted = client.post(
        "/v1/runs",
        json={"session_id": session["id"], "message": "hello", "model": "unit"},
    )
    assert submitted.status_code == 202
    run_id = submitted.json()["id"]

    for _ in range(100):
        run = client.get(f"/v1/runs/{run_id}").json()
        if run["status"] == "completed":
            break
        time.sleep(0.01)
    assert run["result"]["content"] == "runtime answer"

    replay = client.get(f"/v1/runs/{run_id}/events", headers={"Last-Event-ID": "0"})
    assert replay.status_code == 200
    assert "event: run.started" in replay.text
    assert "event: run.completed" in replay.text

    detail = client.get(f"/sessions/{session['id']}").json()
    assert [message["sender"] for message in detail["messages"]] == [
        "user",
        "assistant",
    ]


def test_artifact_download_is_scoped_to_session(tmp_path):
    runtime = build_runtime(tmp_path)
    first = runtime.database.create_session("first", "unit")
    second = runtime.database.create_session("second", "unit")
    artifact = runtime.artifacts.put_bytes(
        b"private", filename="note.txt", session_id=first
    )
    client = TestClient(create_app(runtime))

    allowed = client.get(f"/v1/artifacts/{artifact.id}?session_id={first}")
    denied = client.get(f"/v1/artifacts/{artifact.id}?session_id={second}")

    assert allowed.content == b"private"
    assert denied.status_code == 404


def test_standalone_tool_cannot_self_grant_server_permission(tmp_path, monkeypatch):
    runtime = build_runtime(tmp_path)

    async def dangerous(_arguments, _context):
        return {"executed": True}

    runtime.tools.register(
        ToolSpec(
            name="dangerous",
            description="test",
            input_schema={"type": "object", "additionalProperties": False},
            handler=dangerous,
            required_permissions=frozenset({"code:execute"}),
            approval_required=True,
        )
    )
    monkeypatch.setenv("LOCALGPT_AGENT_PERMISSIONS", "knowledge:read,artifact:read")
    client = TestClient(create_app(runtime))

    response = client.post(
        "/v1/tools/dangerous/execute",
        json={"arguments": {}, "permissions": ["code:execute"], "approved": True},
    )

    assert response.status_code == 422
    assert "requires permissions" in response.json()["detail"]

import tempfile
from pathlib import Path

from backend.agent_runtime.store import RunStore
from backend.agent_runtime.models import RunStatus


def test_run_events_are_persisted_and_replayable() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        store = RunStore(Path(temp_dir) / "runtime.db")
        run = store.create_run(session_id="session-1", request={"message": "hello"})

        first = store.append_event(run.id, "run.started", {"status": "running"})
        second = store.append_event(run.id, "message.delta", {"text": "hello"})

        replay = store.list_events(run.id, after_id=first.id)

        assert [event.id for event in replay] == [second.id]
        assert replay[0].data == {"text": "hello"}


def test_interrupted_runs_are_reconciled_after_restart() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "runtime.db"
        store = RunStore(path)
        run = store.create_run(session_id="session-1", request={"message": "hello"})
        store.transition(run.id, RunStatus.RUNNING)

        restarted_store = RunStore(path)
        recovered = restarted_store.reconcile_interrupted_runs()

        assert recovered == [run.id]
        assert restarted_store.get_run(run.id).status == RunStatus.FAILED


def test_latest_checkpoint_can_resume_after_tool_results() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        store = RunStore(Path(temp_dir) / "runtime.db")
        run = store.create_run(session_id="session-1", request={"message": "hello"})

        store.save_checkpoint(run.id, {"messages": [{"role": "tool", "content": "fact"}]})

        assert store.load_checkpoint(run.id) == {
            "messages": [{"role": "tool", "content": "fact"}]
        }


def test_idempotency_key_returns_the_original_run(tmp_path) -> None:
    store = RunStore(tmp_path / "runs.sqlite")
    first = store.create_run(
        session_id="session",
        request={"message": "hello"},
        metadata={"idempotency_key": "request-123"},
    )
    second = store.create_run(
        session_id="session",
        request={"message": "hello again"},
        metadata={"idempotency_key": "request-123"},
    )

    assert second.id == first.id
    assert len(store.list_runs(session_id="session")) == 1

from pathlib import Path

from backend.agent_runtime.code_execution import (
    DockerSandboxProvider,
    safe_workspace_files,
)


def test_docker_execution_disables_network_and_drops_privileges(tmp_path: Path) -> None:
    captured: list[str] = []

    def runner(command: list[str], **_kwargs: object) -> object:
        captured.extend(command)
        return type(
            "Result",
            (),
            {"returncode": 0, "stdout": "4\n", "stderr": ""},
        )()

    provider = DockerSandboxProvider(
        workspace_root=tmp_path,
        image="python:3.12-slim",
        enabled=True,
        runner=runner,
    )

    result = provider.run_python("session-1", "print(2 + 2)")

    assert result["stdout"] == "4\n"
    assert "--network" in captured and "none" in captured
    assert ["--cap-drop", "ALL"] == captured[
        captured.index("--cap-drop") : captured.index("--cap-drop") + 2
    ]
    assert "--read-only" in captured
    assert "--ulimit" in captured


def test_workspace_artifacts_do_not_follow_symlinks(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside-secret"
    outside.write_text("secret", encoding="utf-8")
    (workspace / "safe.txt").write_text("safe", encoding="utf-8")
    (workspace / "leak.txt").symlink_to(outside)

    assert [path.name for path in safe_workspace_files(workspace)] == ["safe.txt"]

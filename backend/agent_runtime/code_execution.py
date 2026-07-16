from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import subprocess
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from backend.agent_runtime.artifacts import ArtifactStore
from backend.agent_runtime.tools import ToolContext, ToolRegistry, ToolSpec


class SandboxUnavailable(RuntimeError):
    pass


def safe_workspace_files(workspace: Path) -> list[Path]:
    files = []
    for path in workspace.iterdir():
        if path.is_symlink() or not path.is_file() or path.name.startswith("run-"):
            continue
        path.resolve().relative_to(workspace.resolve())
        files.append(path)
    return files


class DockerSandboxProvider:
    """Execute Python in a disposable, networkless Docker container.

    The provider never falls back to the host shell. If Docker is unavailable,
    code execution remains unavailable.
    """

    def __init__(
        self,
        *,
        workspace_root: str | Path,
        image: str = "python:3.12-slim",
        enabled: bool = False,
        timeout_seconds: int = 60,
        memory: str = "512m",
        cpus: str = "1.0",
        pids_limit: int = 64,
        max_output_bytes: int = 1_048_576,
        max_file_bytes: int = 64 * 1024 * 1024,
        runner: Callable[..., Any] = subprocess.run,
    ) -> None:
        allowed_images = {
            value.strip()
            for value in os.getenv(
                "LOCALGPT_SANDBOX_IMAGES", "python:3.12-slim"
            ).split(",")
            if value.strip()
        }
        if image not in allowed_images:
            raise ValueError("Sandbox image is not in LOCALGPT_SANDBOX_IMAGES")
        self.workspace_root = Path(workspace_root).resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.image = image
        self.enabled = enabled
        self.timeout_seconds = timeout_seconds
        self.memory = memory
        self.cpus = cpus
        self.pids_limit = pids_limit
        self.max_output_bytes = max_output_bytes
        self.max_file_bytes = max_file_bytes
        self.runner = runner

    def available(self) -> bool:
        if not self.enabled or shutil.which("docker") is None:
            return False
        try:
            result = self.runner(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            return result.returncode == 0
        except (OSError, subprocess.SubprocessError):
            return False

    def _workspace(self, session_id: str) -> Path:
        safe_id = hashlib.sha256(session_id.encode()).hexdigest()[:24]
        workspace = (self.workspace_root / safe_id).resolve()
        workspace.relative_to(self.workspace_root)
        workspace.mkdir(parents=True, exist_ok=True)
        workspace.chmod(0o777)
        return workspace

    def run_python(self, session_id: str, code: str) -> dict[str, Any]:
        if not self.enabled:
            raise SandboxUnavailable(
                "Code execution is disabled; set LOCALGPT_CODE_EXECUTION_ENABLED=true"
            )
        workspace = self._workspace(session_id)
        script_name = f"run-{uuid.uuid4().hex}.py"
        script = workspace / script_name
        script.write_text(code, encoding="utf-8")
        script.chmod(0o444)
        command = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            str(self.pids_limit),
            "--ulimit",
            f"fsize={self.max_file_bytes}:{self.max_file_bytes}",
            "--memory",
            self.memory,
            "--cpus",
            self.cpus,
            "--user",
            "65534:65534",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=64m",
            "--mount",
            f"type=bind,src={workspace},dst=/workspace",
            "--workdir",
            "/workspace",
            self.image,
            "python",
            "-I",
            script_name,
        ]
        try:
            result = self.runner(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                f"Sandbox execution exceeded {self.timeout_seconds}s"
            ) from exc
        finally:
            script.unlink(missing_ok=True)
        stdout = str(result.stdout or "")
        stderr = str(result.stderr or "")
        if len(stdout.encode()) > self.max_output_bytes:
            stdout = stdout.encode()[: self.max_output_bytes].decode(errors="replace")
            stdout += "\n[output truncated]"
        if len(stderr.encode()) > self.max_output_bytes:
            stderr = stderr.encode()[: self.max_output_bytes].decode(errors="replace")
            stderr += "\n[output truncated]"
        return {
            "exit_code": int(result.returncode),
            "stdout": stdout,
            "stderr": stderr,
            "success": result.returncode == 0,
        }


def register_code_execution_tool(
    registry: ToolRegistry,
    *,
    provider: DockerSandboxProvider,
    artifacts: ArtifactStore,
) -> None:
    async def execute_python(
        arguments: dict[str, Any], context: ToolContext
    ) -> dict[str, Any]:
        if not context.session_id:
            raise ValueError("Code execution requires a session")
        result = await asyncio.to_thread(
            provider.run_python, context.session_id, arguments["code"]
        )
        workspace = provider._workspace(context.session_id)
        output_artifacts = []
        for path in safe_workspace_files(workspace)[:100]:
            if path.stat().st_size > provider.max_file_bytes:
                continue
            artifact = artifacts.put_path(
                path,
                session_id=context.session_id,
                run_id=context.run_id,
                provenance={"source": "code_execution", "tool": "execute_python"},
            )
            output_artifacts.append(artifact.id)
        result["output_artifacts"] = output_artifacts
        return result

    registry.register(
        ToolSpec(
            name="execute_python",
            description=(
                "Run Python in a disposable network-disabled container and return "
                "stdout, stderr, and generated artifacts."
            ),
            input_schema={
                "type": "object",
                "properties": {"code": {"type": "string", "minLength": 1}},
                "required": ["code"],
                "additionalProperties": False,
            },
            handler=execute_python,
            required_permissions=frozenset({"code:execute"}),
            approval_required=True,
            side_effects=True,
            timeout_seconds=provider.timeout_seconds + 5,
        )
    )

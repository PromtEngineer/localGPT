from __future__ import annotations

import asyncio
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import requests

from backend.agent_runtime.executor import AgentExecutor, AgentRequest
from backend.agent_runtime.models import Run, RunStatus
from backend.agent_runtime.observability import correlation_id, redact, span
from backend.agent_runtime.skills import SkillStore
from backend.agent_runtime.store import RunStore
from backend.agent_runtime.tools import ToolContext, ToolRegistry
from backend.database import ChatDatabase


class RunManager:
    """Schedules durable agent and indexing runs on bounded local workers."""

    def __init__(
        self,
        *,
        store: RunStore,
        executor: AgentExecutor,
        tools: ToolRegistry,
        database: ChatDatabase,
        skills: SkillStore,
        rag_api_url: str | None = None,
        max_workers: int = 2,
    ) -> None:
        self.store = store
        self.executor = executor
        self.tools = tools
        self.database = database
        self.skills = skills
        self.rag_api_url = (rag_api_url or os.getenv("RAG_API_URL", "http://127.0.0.1:8001")).rstrip("/")
        self.pool = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="localgpt-run"
        )
        self._futures: dict[str, Future[Any]] = {}
        self.store.reconcile_interrupted_runs()
        for queued in self.store.list_runs(limit=10_000):
            if queued.status != RunStatus.QUEUED:
                continue
            worker = self._execute_index if queued.kind == "index" else self._execute_message
            self._futures[queued.id] = self.pool.submit(worker, queued.id)

    def submit_message(self, request: dict[str, Any]) -> Run:
        session_id = request.get("session_id")
        idempotency_key = request.get("idempotency_key")
        if idempotency_key:
            existing = self.store.find_idempotent(
                session_id=session_id,
                kind="message",
                idempotency_key=str(idempotency_key),
            )
            if existing is not None:
                return existing
        run = self.store.create_run(
            session_id=session_id,
            request=request,
            kind="message",
            metadata={"idempotency_key": request.get("idempotency_key")},
        )
        self._futures[run.id] = self.pool.submit(self._execute_message, run.id)
        return run

    def retry(self, run_id: str) -> Run:
        original = self.store.get_run(run_id)
        if original is None:
            raise KeyError("Run not found")
        request = dict(original.request)
        request["retry_of"] = run_id
        checkpoint = self.store.load_checkpoint(run_id)
        if checkpoint:
            request["resume_checkpoint"] = checkpoint
        if original.kind == "index":
            return self.submit_index(request)
        request["persist_user_message"] = False
        return self.submit_message(request)

    def submit_index(self, request: dict[str, Any]) -> Run:
        run = self.store.create_run(
            session_id=request.get("session_id"),
            request=request,
            kind="index",
            metadata={"index_id": request.get("index_id")},
        )
        self._futures[run.id] = self.pool.submit(self._execute_index, run.id)
        return run

    def cancel(self, run_id: str) -> bool:
        requested = self.store.request_cancel(run_id)
        future = self._futures.get(run_id)
        if future is not None and future.cancel():
            run = self.store.get_run(run_id)
            if run is not None and run.status == RunStatus.QUEUED:
                self.store.transition(run_id, RunStatus.CANCELLED)
                self.store.append_event(run_id, "run.cancelled", {"status": "cancelled"})
            self._futures.pop(run_id, None)
        return requested

    def wait(self, run_id: str, timeout: float = 300) -> Run:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            run = self.store.get_run(run_id)
            if run is None:
                raise KeyError("Run not found")
            if run.status in {
                RunStatus.COMPLETED,
                RunStatus.FAILED,
                RunStatus.CANCELLED,
            }:
                return run
            time.sleep(0.05)
        raise TimeoutError(f"Run {run_id} did not finish within {timeout}s")

    def _execute_message(self, run_id: str) -> None:
        token = correlation_id.set(run_id)
        try:
            asyncio.run(self._execute_message_async(run_id))
        finally:
            correlation_id.reset(token)
            self._futures.pop(run_id, None)

    def _execute_index(self, run_id: str) -> None:
        token = correlation_id.set(run_id)
        try:
            run = self.store.get_run(run_id)
            if run is None:
                return
            if self.store.is_cancel_requested(run_id):
                self.store.transition(run_id, RunStatus.CANCELLED)
                self.store.append_event(run_id, "run.cancelled", {"status": "cancelled"})
                return
            self.store.transition(run_id, RunStatus.RUNNING)
            self.store.append_event(run_id, "run.started", {"status": "running", "kind": "index"})
            request = run.request
            index_id = str(request["index_id"])
            self.database.update_index_metadata(index_id, {"status": "building", "run_id": run_id})
            self.store.save_checkpoint(run_id, {"stage": "submitted", "index_id": index_id})
            headers = {"Content-Type": "application/json"}
            api_token = os.getenv("LOCALGPT_API_TOKEN")
            if api_token:
                headers["Authorization"] = f"Bearer {api_token}"
            response = requests.post(
                f"{self.rag_api_url}/index",
                headers=headers,
                json=request["payload"],
                timeout=float(request.get("timeout_seconds", 3600)),
            )
            response.raise_for_status()
            result = response.json()
            if self.store.is_cancel_requested(run_id):
                self.store.transition(run_id, RunStatus.CANCELLED)
                self.store.append_event(run_id, "run.cancelled", {"status": "cancelled"})
                self.database.update_index_metadata(index_id, {"status": "cancelled"})
                return
            self.store.save_checkpoint(run_id, {"stage": "indexed", "result": result})
            self.database.update_index_metadata(
                index_id, {"status": "ready", "build_result": result, "run_id": run_id}
            )
            self.store.transition(run_id, RunStatus.COMPLETED, result=result)
            self.store.append_event(run_id, "index.completed", result)
            self.store.append_event(run_id, "run.completed", {"status": "completed"})
        except Exception as exc:
            safe_error = str(redact(str(exc)))
            current = self.store.get_run(run_id)
            if current and current.status not in {
                RunStatus.COMPLETED,
                RunStatus.FAILED,
                RunStatus.CANCELLED,
            }:
                self.store.transition(run_id, RunStatus.FAILED, error=safe_error)
                self.store.append_event(
                    run_id, "run.failed", {"status": "failed", "error": safe_error}
                )
                index_id = current.request.get("index_id")
                if index_id:
                    self.database.update_index_metadata(
                        str(index_id), {"status": "failed", "error": safe_error}
                    )
        finally:
            correlation_id.reset(token)
            self._futures.pop(run_id, None)

    async def _execute_message_async(self, run_id: str) -> None:
        run = self.store.get_run(run_id)
        if run is None:
            return
        self.store.transition(run_id, RunStatus.RUNNING)
        self.store.append_event(run_id, "run.started", {"status": "running"})
        request = run.request
        session_id = run.session_id

        def emit(event_type: str, payload: dict[str, Any]) -> None:
            safe_payload = redact(payload)
            self.store.append_event(run_id, event_type, safe_payload)
            if event_type.startswith("tool."):
                self.store.audit_tool(
                    run_id,
                    str(payload.get("tool", "unknown")),
                    event_type.removeprefix("tool."),
                    safe_payload,
                )

        try:
            if self.store.is_cancel_requested(run_id):
                raise asyncio.CancelledError
            messages = [dict(message) for message in request.get("messages", [])]
            checkpoint = request.get("resume_checkpoint")
            if isinstance(checkpoint, dict) and isinstance(checkpoint.get("messages"), list):
                messages = [dict(message) for message in checkpoint["messages"]]
            if session_id and not messages:
                messages = self.database.get_conversation_history(session_id)
                message = str(request.get("message") or "")
                if message:
                    messages.append({"role": "user", "content": message})
            if not messages:
                raise ValueError("At least one message is required")

            skill_tools: set[str] | None = None
            system_instructions: list[str] = []
            for skill_id in request.get("skill_ids", []):
                skill = self.skills.get_current(skill_id)
                if skill is None:
                    raise ValueError(f"Skill not found: {skill_id}")
                system_instructions.append(
                    f"Skill {skill.name}:\n{skill.instructions}"
                )
                allowed = set(skill.allowed_tools)
                skill_tools = allowed if skill_tools is None else skill_tools & allowed
            if system_instructions:
                messages.insert(
                    0,
                    {"role": "system", "content": "\n\n".join(system_instructions)},
                )

            raw_allowed_tools = request.get("allowed_tools")
            requested_tools = (
                set(raw_allowed_tools) if raw_allowed_tools is not None else None
            )
            if requested_tools is not None and skill_tools is not None:
                allowed_tools = requested_tools & skill_tools
            elif requested_tools is not None:
                allowed_tools = requested_tools
            else:
                allowed_tools = skill_tools
            if allowed_tools is None:
                allowed_tools = {
                    "search_knowledge",
                    "list_artifacts",
                    "read_artifact",
                }

            server_permissions = {
                item.strip()
                for item in os.getenv(
                    "LOCALGPT_AGENT_PERMISSIONS",
                    "knowledge:read,artifact:read",
                ).split(",")
                if item.strip()
            }
            raw_permissions = request.get("permissions")
            requested_permissions = (
                set(raw_permissions)
                if raw_permissions is not None
                else set(server_permissions)
            )
            context = ToolContext(
                run_id=run_id,
                session_id=session_id,
                permissions=server_permissions & requested_permissions,
                approved_tools=set(request.get("approved_tools") or []),
                emit=emit,
                metadata={
                    "cancel_requested": lambda: self.store.is_cancel_requested(run_id),
                    "checkpoint": lambda state: self.store.save_checkpoint(run_id, state),
                },
            )

            if request.get("force_rag"):
                if "search_knowledge" not in allowed_tools:
                    raise PermissionError(
                        "force_rag requires search_knowledge in allowed_tools"
                    )
                user_content = next(
                    (
                        str(message.get("content", ""))
                        for message in reversed(messages)
                        if message.get("role") == "user"
                    ),
                    "",
                )
                with span("agent.retrieval", run_id=run_id):
                    result = await self.tools.execute(
                        "search_knowledge",
                        {
                            "query": user_content,
                            "top_k": int(request.get("retrieval_k", 8)),
                            "search_type": request.get("search_type", "hybrid"),
                            **{
                                key: request[key]
                                for key in (
                                    "query_decompose",
                                    "compose_sub_answers",
                                    "ai_rerank",
                                    "context_expand",
                                    "verify",
                                    "context_window_size",
                                    "reranker_top_k",
                                    "dense_weight",
                                    "provence_prune",
                                    "provence_threshold",
                                )
                                if key in request
                            },
                        },
                        context,
                    )
                final = {
                    "content": result.get("answer", ""),
                    "citations": result.get("citations", []),
                    "tool_calls_executed": 1,
                    "usage": {},
                }
            else:
                with span("agent.execute", run_id=run_id):
                    agent_result = await self.executor.execute(
                        AgentRequest(
                            model=str(request.get("model") or "qwen3:8b"),
                            messages=messages,
                            allowed_tools=allowed_tools,
                            temperature=float(request.get("temperature", 0.2)),
                            max_tokens=request.get("max_tokens"),
                            max_iterations=min(int(request.get("max_iterations", 8)), 20),
                            max_tool_calls=min(int(request.get("max_tool_calls", 12)), 50),
                            max_elapsed_seconds=min(
                                float(request.get("max_elapsed_seconds", 300)), 900
                            ),
                            max_total_tokens=min(
                                int(request.get("max_total_tokens", 100_000)), 1_000_000
                            ),
                        ),
                        context,
                    )
                final = {
                    "content": agent_result.content,
                    "citations": [],
                    "tool_calls_executed": agent_result.tool_calls_executed,
                    "usage": {
                        "input_tokens": agent_result.input_tokens,
                        "output_tokens": agent_result.output_tokens,
                    },
                }

            if self.store.is_cancel_requested(run_id):
                raise asyncio.CancelledError

            if session_id:
                user_content = next(
                    (
                        str(message.get("content", ""))
                        for message in reversed(messages)
                        if message.get("role") == "user"
                    ),
                    "",
                )
                assistant_metadata = {"citations": final["citations"], "run_id": run_id}
                if request.get("persist_user_message", True):
                    user_id, assistant_id = self.database.add_exchange(
                        session_id,
                        user_content,
                        final["content"],
                        assistant_metadata,
                    )
                    final["user_message_id"] = user_id
                else:
                    assistant_id = self.database.add_message(
                        session_id,
                        final["content"],
                        "assistant",
                        assistant_metadata,
                    )
                final["assistant_message_id"] = assistant_id
            self.store.transition(run_id, RunStatus.COMPLETED, result=final)
            emit("message.completed", final)
            emit("run.completed", {"status": "completed"})
        except asyncio.CancelledError:
            self.store.transition(run_id, RunStatus.CANCELLED)
            emit("run.cancelled", {"status": "cancelled"})
        except Exception as exc:
            safe_error = str(redact(str(exc)))
            self.store.transition(run_id, RunStatus.FAILED, error=safe_error)
            emit("run.failed", {"status": "failed", "error": safe_error})

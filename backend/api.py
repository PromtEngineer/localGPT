from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, AsyncIterator

import requests
from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from backend.agent_runtime.artifacts import ArtifactStore, S3BlobStore
from backend.agent_runtime.builtin_tools import RagRetrievalClient, register_core_tools
from backend.agent_runtime.code_execution import (
    DockerSandboxProvider,
    register_code_execution_tool,
)
from backend.agent_runtime.data_tools import DatabaseConnectorRegistry, register_data_tools
from backend.agent_runtime.executor import AgentExecutor
from backend.agent_runtime.mcp_tools import MCPConnectorRegistry, register_mcp_tools
from backend.agent_runtime.models import RunStatus
from backend.agent_runtime.observability import JsonLogFormatter, correlation_id
from backend.agent_runtime.providers import ChatProvider, configured_provider
from backend.agent_runtime.service import RunManager
from backend.agent_runtime.skills import SkillStore
from backend.agent_runtime.store import RunStore
from backend.agent_runtime.tools import ToolContext, ToolRegistry
from backend.database import ChatDatabase, generate_session_title
from localgpt_runtime import (
    UploadRejected,
    env_path,
    normalize_index_options,
    inspect_upload_content,
    request_is_authorized,
    safe_upload_path,
)


logger = logging.getLogger("localgpt.api")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonLogFormatter())
    logger.addHandler(handler)
logger.setLevel(os.getenv("LOCALGPT_LOG_LEVEL", "INFO"))


class SessionCreate(BaseModel):
    title: str = Field(default="New Chat", max_length=200)
    model: str = Field(default="qwen3:8b", min_length=1, max_length=200)


class SessionRename(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class IndexCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=2000)
    options: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MessageRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str | None = None
    message: str | None = Field(default=None, max_length=1_000_000)
    messages: list[dict[str, Any]] = Field(default_factory=list, max_length=500)
    model: str | None = None
    temperature: float = Field(default=0.2, ge=0, le=2)
    max_tokens: int | None = Field(default=None, ge=1, le=1_000_000)
    max_iterations: int = Field(default=8, ge=1, le=20)
    max_tool_calls: int = Field(default=12, ge=0, le=50)
    max_elapsed_seconds: float = Field(default=300, ge=1, le=900)
    max_total_tokens: int = Field(default=100_000, ge=1, le=1_000_000)
    allowed_tools: list[str] | None = None
    approved_tools: list[str] = Field(default_factory=list)
    permissions: list[str] | None = None
    skill_ids: list[str] = Field(default_factory=list)
    force_rag: bool = False
    retrieval_k: int = Field(default=8, ge=1, le=30)
    search_type: str = Field(default="hybrid", pattern="^(hybrid|dense|lexical)$")
    query_decompose: bool | None = None
    compose_sub_answers: bool | None = None
    ai_rerank: bool | None = None
    context_expand: bool | None = None
    verify: bool | None = None
    context_window_size: int | None = Field(default=None, ge=0, le=20)
    reranker_top_k: int | None = Field(default=None, ge=1, le=100)
    dense_weight: float | None = Field(default=None, ge=0, le=1)
    provence_prune: bool | None = None
    provence_threshold: float | None = Field(default=None, ge=0, le=1)
    idempotency_key: str | None = Field(default=None, max_length=200)


class LegacyMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    message: str = Field(
        min_length=1,
        max_length=1_000_000,
        validation_alias=AliasChoices("message", "query"),
    )
    model: str | None = None
    force_rag: bool = False
    retrieval_k: int = Field(default=8, ge=1, le=30)
    search_type: str = Field(default="hybrid", pattern="^(hybrid|dense|lexical)$")
    query_decompose: bool | None = None
    compose_sub_answers: bool | None = None
    ai_rerank: bool | None = None
    context_expand: bool | None = None
    verify: bool | None = None
    context_window_size: int | None = Field(default=None, ge=0, le=20)
    reranker_top_k: int | None = Field(default=None, ge=1, le=100)
    dense_weight: float | None = Field(default=None, ge=0, le=1)
    provence_prune: bool | None = None
    provence_threshold: float | None = Field(default=None, ge=0, le=1)


class ToolExecution(BaseModel):
    arguments: dict[str, Any] = Field(default_factory=dict)
    session_id: str | None = None
    permissions: list[str] = Field(default_factory=list)
    approved: bool = False


class SkillContent(BaseModel):
    content: str = Field(min_length=1, max_length=1_000_000)


class IndexJobRequest(BaseModel):
    index_id: str
    timeout_seconds: float = Field(default=3600, ge=30, le=86400)


class MessageValidation(BaseModel):
    model: str
    messages: list[dict[str, Any]] = Field(min_length=1, max_length=500)
    allowed_tools: list[str] | None = None


@dataclass(slots=True)
class Runtime:
    database: ChatDatabase
    runs: RunStore
    artifacts: ArtifactStore
    skills: SkillStore
    tools: ToolRegistry
    provider: ChatProvider
    manager: RunManager
    upload_dir: Path | None = None
    rag_api_url: str = "http://127.0.0.1:8001"


def build_runtime() -> Runtime:
    project_root = Path(__file__).resolve().parents[1]
    state_root = env_path("LOCALGPT_STATE_DIR", project_root / "localgpt_state")
    state_root.mkdir(parents=True, exist_ok=True)
    upload_dir = env_path("LOCALGPT_UPLOAD_DIR", project_root / "shared_uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    rag_api_url = os.getenv("RAG_API_URL", "http://127.0.0.1:8001").rstrip("/")

    database = ChatDatabase(os.getenv("LOCALGPT_DB_PATH", str(state_root / "chat.sqlite")))
    runs = RunStore(state_root / "runs.sqlite")
    artifact_backend = os.getenv("LOCALGPT_ARTIFACT_BACKEND", "local").lower()
    if artifact_backend == "s3":
        blob_store = S3BlobStore(
            os.environ["LOCALGPT_S3_BUCKET"],
            prefix=os.getenv("LOCALGPT_S3_PREFIX", "localgpt"),
            endpoint_url=os.getenv("LOCALGPT_S3_ENDPOINT_URL"),
        )
        artifacts = ArtifactStore(state_root / "artifacts.sqlite", blob_store=blob_store)
    else:
        artifacts = ArtifactStore(state_root / "artifacts.sqlite", state_root / "objects")
    skills = SkillStore(state_root / "skills.sqlite")
    tools = ToolRegistry()
    register_core_tools(
        tools,
        artifacts=artifacts,
        retrieval=RagRetrievalClient(rag_api_url, database=database),
    )
    register_data_tools(
        tools,
        artifacts=artifacts,
        connectors=DatabaseConnectorRegistry(),
    )
    sandbox = DockerSandboxProvider(
        workspace_root=state_root / "sandboxes",
        image=os.getenv("LOCALGPT_SANDBOX_IMAGE", "python:3.12-slim"),
        enabled=os.getenv("LOCALGPT_CODE_EXECUTION_ENABLED", "false").lower()
        in {"1", "true", "yes"},
    )
    register_code_execution_tool(tools, provider=sandbox, artifacts=artifacts)
    provider = configured_provider()
    manager = RunManager(
        store=runs,
        executor=AgentExecutor(provider=provider, tools=tools),
        tools=tools,
        database=database,
        skills=skills,
        rag_api_url=rag_api_url,
    )
    return Runtime(
        database,
        runs,
        artifacts,
        skills,
        tools,
        provider,
        manager,
        upload_dir,
        rag_api_url,
    )


def _run_json(run: Any) -> dict[str, Any]:
    return asdict(run)


def _artifact_json(artifact: Any) -> dict[str, Any]:
    return asdict(artifact)


def _rag_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    token = os.getenv("LOCALGPT_API_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _server_tool_permissions() -> set[str]:
    return {
        item.strip()
        for item in os.getenv(
            "LOCALGPT_AGENT_PERMISSIONS", "knowledge:read,artifact:read"
        ).split(",")
        if item.strip()
    }


def create_app(runtime: Runtime | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        current = application.state.runtime
        if current is None:
            current = build_runtime()
            application.state.runtime = current
        connectors = MCPConnectorRegistry()
        if connectors.connectors:
            try:
                registered = await register_mcp_tools(current.tools, connectors)
                logger.info("mcp.registered", extra={"fields": {"tools": registered}})
            except Exception as exc:
                logger.error("mcp.registration_failed", extra={"fields": {"error": str(exc)}})
        yield

    app = FastAPI(
        title="LocalGPT API",
        version="2.0.0",
        description="Durable local document, retrieval, agent, tool, and artifact API.",
        lifespan=lifespan,
    )
    app.state.runtime = runtime
    origins = [
        item.strip()
        for item in os.getenv(
            "LOCALGPT_ALLOWED_ORIGINS",
            "http://localhost:3000,http://127.0.0.1:3000",
        ).split(",")
        if item.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "Last-Event-ID", "X-Request-ID"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        token = correlation_id.set(request_id)
        started = time.monotonic()
        try:
            response = await call_next(request)
        finally:
            elapsed = round((time.monotonic() - started) * 1000, 2)
            logger.info(
                "request.completed",
                extra={"fields": {"path": request.url.path, "method": request.method, "latency_ms": elapsed}},
            )
            correlation_id.reset(token)
        response.headers["X-Request-ID"] = request_id
        return response

    def get_runtime() -> Runtime:
        current = app.state.runtime
        if current is None:
            current = build_runtime()
            app.state.runtime = current
        return current

    def authorize(authorization: str | None = Header(default=None)) -> None:
        if not request_is_authorized(authorization):
            raise HTTPException(status_code=401, detail="Unauthorized")

    protected = [Depends(authorize)]

    @app.get("/health", dependencies=protected)
    async def health(current: Runtime = Depends(get_runtime)) -> dict[str, Any]:
        models = []
        ollama_running = False
        try:
            models = [asdict(item) for item in await current.provider.discover_models()]  # type: ignore[attr-defined]
            ollama_running = True
        except Exception:
            pass
        return {
            "status": "ok",
            "ollama_running": ollama_running,
            "available_models": [item["id"] for item in models],
            "database_stats": current.database.get_stats(),
            "capabilities": {
                "durable_runs": True,
                "event_replay": True,
                "artifacts": True,
                "tools": current.tools.names(),
            },
        }

    @app.get("/v1/models", dependencies=protected)
    async def models(current: Runtime = Depends(get_runtime)) -> dict[str, Any]:
        try:
            discovered = await current.provider.discover_models()  # type: ignore[attr-defined]
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Model discovery failed: {exc}") from exc
        return {"models": [asdict(item) for item in discovered]}

    @app.get("/models", dependencies=protected)
    async def legacy_models(current: Runtime = Depends(get_runtime)) -> dict[str, Any]:
        try:
            discovered = await current.provider.discover_models()  # type: ignore[attr-defined]
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Model discovery failed: {exc}") from exc
        return {
            "generation_models": [item.id for item in discovered if item.generation],
            "embedding_models": [item.id for item in discovered if item.embedding],
        }

    @app.post("/v1/embeddings", dependencies=protected)
    async def embeddings(payload: dict[str, Any], current: Runtime = Depends(get_runtime)) -> dict[str, Any]:
        model = str(payload.get("model") or os.getenv("LOCALGPT_EMBEDDING_MODEL", "nomic-embed-text"))
        raw_input = payload.get("input")
        inputs = [raw_input] if isinstance(raw_input, str) else raw_input
        if not isinstance(inputs, list) or not inputs or not all(isinstance(item, str) for item in inputs):
            raise HTTPException(status_code=422, detail="input must be a string or non-empty string array")
        try:
            vectors = await current.provider.embed(model=model, inputs=inputs)  # type: ignore[attr-defined]
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Embedding provider failed: {exc}") from exc
        return {
            "object": "list",
            "model": model,
            "data": [
                {"object": "embedding", "index": index, "embedding": vector}
                for index, vector in enumerate(vectors)
            ],
        }

    @app.post("/v1/messages/count_tokens", dependencies=protected)
    def count_tokens(payload: MessageValidation) -> dict[str, Any]:
        # Providers do not expose a common tokenizer contract. This stable estimate
        # is explicitly labeled so callers can budget conservatively.
        characters = sum(len(str(item.get("content", ""))) for item in payload.messages)
        return {
            "model": payload.model,
            "input_tokens": max(1, (characters + 3) // 4),
            "estimated": True,
        }

    @app.post("/v1/messages/validate", dependencies=protected)
    async def validate_message(payload: MessageValidation, current: Runtime = Depends(get_runtime)) -> dict[str, Any]:
        unknown_tools = sorted(set(payload.allowed_tools or []) - set(current.tools.names()))
        try:
            available = {item.id for item in await current.provider.discover_models()}  # type: ignore[attr-defined]
        except Exception:
            available = set()
        errors = []
        if unknown_tools:
            errors.append({"field": "allowed_tools", "message": f"Unknown tools: {', '.join(unknown_tools)}"})
        if available and payload.model not in available:
            errors.append({"field": "model", "message": "Model is not currently available"})
        return {"valid": not errors, "errors": errors}

    @app.get("/sessions", dependencies=protected)
    def list_sessions(current: Runtime = Depends(get_runtime)) -> dict[str, Any]:
        sessions = current.database.get_sessions()
        return {"sessions": sessions, "total": len(sessions)}

    @app.post("/sessions", status_code=201, dependencies=protected)
    def create_session(payload: SessionCreate, current: Runtime = Depends(get_runtime)) -> dict[str, Any]:
        session_id = current.database.create_session(payload.title, payload.model)
        return {"session": current.database.get_session(session_id), "session_id": session_id}

    @app.get("/sessions/cleanup", dependencies=protected)
    def cleanup_sessions(current: Runtime = Depends(get_runtime)) -> dict[str, Any]:
        count = current.database.cleanup_empty_sessions()
        return {"message": f"Cleaned up {count} empty sessions", "cleanup_count": count}

    @app.get("/sessions/{session_id}", dependencies=protected)
    def get_session(session_id: str, current: Runtime = Depends(get_runtime)) -> dict[str, Any]:
        session = current.database.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"session": session, "messages": current.database.get_messages(session_id)}

    @app.post("/sessions/{session_id}/rename", dependencies=protected)
    def rename_session(session_id: str, payload: SessionRename, current: Runtime = Depends(get_runtime)) -> dict[str, Any]:
        if current.database.get_session(session_id) is None:
            raise HTTPException(status_code=404, detail="Session not found")
        current.database.update_session_title(session_id, payload.title)
        return {"session": current.database.get_session(session_id)}

    @app.delete("/sessions/{session_id}", dependencies=protected)
    def delete_session(session_id: str, current: Runtime = Depends(get_runtime)) -> dict[str, Any]:
        if not current.database.delete_session(session_id):
            raise HTTPException(status_code=404, detail="Session not found")
        return {
            "message": "Session deleted",
            "deleted": True,
            "session_id": session_id,
            "deleted_session_id": session_id,
        }

    @app.post("/v1/runs", status_code=202, dependencies=protected)
    def submit_run(payload: MessageRunRequest, current: Runtime = Depends(get_runtime)) -> dict[str, Any]:
        if payload.session_id and current.database.get_session(payload.session_id) is None:
            raise HTTPException(status_code=404, detail="Session not found")
        if not payload.message and not payload.messages:
            raise HTTPException(status_code=422, detail="message or messages is required")
        request_data = payload.model_dump(exclude_none=True)
        run = current.manager.submit_message(request_data)
        return _run_json(run)

    @app.get("/v1/runs", dependencies=protected)
    def list_runs(
        session_id: str | None = None,
        kind: str | None = None,
        limit: int = Query(default=100, ge=1, le=1000),
        current: Runtime = Depends(get_runtime),
    ) -> dict[str, Any]:
        rows = current.runs.list_runs(session_id=session_id, kind=kind, limit=limit)
        return {"runs": [_run_json(row) for row in rows]}

    @app.get("/v1/runs/{run_id}", dependencies=protected)
    def get_run(run_id: str, current: Runtime = Depends(get_runtime)) -> dict[str, Any]:
        run = current.runs.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return _run_json(run)

    @app.post("/v1/runs/{run_id}/cancel", status_code=202, dependencies=protected)
    def cancel_run(run_id: str, current: Runtime = Depends(get_runtime)) -> dict[str, Any]:
        if current.runs.get_run(run_id) is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return {"cancel_requested": current.manager.cancel(run_id)}

    @app.post("/v1/runs/{run_id}/retry", status_code=202, dependencies=protected)
    def retry_run(run_id: str, current: Runtime = Depends(get_runtime)) -> dict[str, Any]:
        try:
            return _run_json(current.manager.retry(run_id))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Run not found") from exc

    @app.delete("/v1/runs/{run_id}", dependencies=protected)
    def delete_run(run_id: str, current: Runtime = Depends(get_runtime)) -> dict[str, Any]:
        run = current.runs.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        if run.status not in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}:
            raise HTTPException(status_code=409, detail="Active runs must be cancelled before deletion")
        current.runs.delete_run(run_id)
        return {"deleted": True, "run_id": run_id}

    @app.get("/v1/runs/{run_id}/events", dependencies=protected)
    async def run_events(
        run_id: str,
        last_event_id: int = Header(default=0, alias="Last-Event-ID"),
        current: Runtime = Depends(get_runtime),
    ) -> StreamingResponse:
        if current.runs.get_run(run_id) is None:
            raise HTTPException(status_code=404, detail="Run not found")

        async def stream() -> AsyncIterator[str]:
            cursor = last_event_id
            idle = 0
            while True:
                events = current.runs.list_events(run_id, after_id=cursor)
                for event in events:
                    cursor = event.id
                    idle = 0
                    yield f"id: {event.id}\nevent: {event.type}\ndata: {json.dumps(event.data)}\n\n"
                run = current.runs.get_run(run_id)
                if run is None or (
                    run.status in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}
                    and not events
                ):
                    return
                idle += 1
                if idle % 150 == 0:
                    yield ": keep-alive\n\n"
                await asyncio.sleep(0.1)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/sessions/{session_id}/messages", dependencies=protected)
    def legacy_message(session_id: str, payload: LegacyMessage, current: Runtime = Depends(get_runtime)) -> dict[str, Any]:
        session = current.database.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        if session["message_count"] == 0:
            current.database.update_session_title(session_id, generate_session_title(payload.message))
        request_data = payload.model_dump(exclude_none=True)
        request_data.update({"session_id": session_id, "model": payload.model or session["model_used"]})
        run = current.manager.submit_message(request_data)
        completed = current.manager.wait(run.id)
        if completed.status != RunStatus.COMPLETED:
            raise HTTPException(status_code=502, detail=completed.error or "Run failed")
        result = completed.result or {}
        return {
            "response": result.get("content", ""),
            "session": current.database.get_session(session_id),
            "source_documents": result.get("citations", []),
            "used_rag": payload.force_rag,
            "route": "rag" if payload.force_rag else "agent",
            "user_message_id": result.get("user_message_id"),
            "ai_message_id": result.get("assistant_message_id"),
            "run_id": run.id,
        }

    @app.post("/sessions/{session_id}/messages/stream", dependencies=protected)
    async def legacy_message_stream(session_id: str, payload: LegacyMessage, current: Runtime = Depends(get_runtime)) -> StreamingResponse:
        if current.database.get_session(session_id) is None:
            raise HTTPException(status_code=404, detail="Session not found")
        request_data = payload.model_dump(exclude_none=True)
        request_data["session_id"] = session_id
        run = current.manager.submit_message(request_data)

        async def stream() -> AsyncIterator[str]:
            cursor = 0
            while True:
                events = current.runs.list_events(run.id, after_id=cursor)
                for event in events:
                    cursor = event.id
                    event_type = "complete" if event.type == "run.completed" else event.type
                    yield f"data: {json.dumps({'type': event_type, 'data': event.data, 'id': event.id})}\n\n"
                current_run = current.runs.get_run(run.id)
                if current_run and current_run.status in {
                    RunStatus.COMPLETED,
                    RunStatus.FAILED,
                    RunStatus.CANCELLED,
                } and not events:
                    return
                await asyncio.sleep(0.1)

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.get("/indexes", dependencies=protected)
    def list_indexes(current: Runtime = Depends(get_runtime)) -> dict[str, Any]:
        rows = current.database.list_indexes()
        return {"indexes": rows, "total": len(rows)}

    @app.post("/indexes", status_code=201, dependencies=protected)
    def create_index(payload: IndexCreate, current: Runtime = Depends(get_runtime)) -> dict[str, Any]:
        try:
            options = normalize_index_options({**payload.metadata, **payload.options})
            index_id = current.database.create_index(payload.name, payload.description, options)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"index": current.database.get_index(index_id), "index_id": index_id}

    @app.get("/indexes/{index_id}", dependencies=protected)
    def get_index(index_id: str, current: Runtime = Depends(get_runtime)) -> dict[str, Any]:
        index = current.database.get_index(index_id)
        if index is None:
            raise HTTPException(status_code=404, detail="Index not found")
        return {"index": index}

    async def save_upload(
        upload: UploadFile,
        *,
        current: Runtime,
        session_id: str | None = None,
        index_id: str | None = None,
    ) -> tuple[Path, Any]:
        upload_dir = current.upload_dir or Path("shared_uploads").resolve()
        submitted = upload.filename or ""
        try:
            safe_upload_path(upload_dir, submitted)
            destination = safe_upload_path(upload_dir, f"{uuid.uuid4().hex}_{submitted}")
        except UploadRejected as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        limit = int(os.getenv("LOCALGPT_MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))
        content = await upload.read(limit + 1)
        if len(content) > limit:
            raise HTTPException(status_code=413, detail=f"Upload exceeds {limit} bytes")
        try:
            inspect_upload_content(submitted, content)
        except UploadRejected as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            with destination.open("xb") as output:
                output.write(content)
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        artifact = current.artifacts.put_bytes(
            content,
            filename=submitted,
            mime_type=upload.content_type or mimetypes.guess_type(submitted)[0],
            session_id=session_id,
            index_id=index_id,
            provenance={"source": "upload", "stored_path": str(destination)},
        )
        return destination, artifact

    @app.post("/sessions/{session_id}/upload", status_code=201, dependencies=protected)
    async def upload_to_session(
        session_id: str,
        files: list[UploadFile] = File(...),
        current: Runtime = Depends(get_runtime),
    ) -> dict[str, Any]:
        if current.database.get_session(session_id) is None:
            raise HTTPException(status_code=404, detail="Session not found")
        uploaded = []
        for file in files:
            path, artifact = await save_upload(file, current=current, session_id=session_id)
            document_id = current.database.add_document_to_session(session_id, str(path))
            uploaded.append({"filename": file.filename, "stored_path": str(path), "document_id": document_id, "artifact_id": artifact.id})
        return {"message": f"Uploaded {len(uploaded)} files", "uploaded_files": uploaded}

    @app.post("/indexes/{index_id}/upload", status_code=201, dependencies=protected)
    async def upload_to_index(
        index_id: str,
        files: list[UploadFile] = File(...),
        current: Runtime = Depends(get_runtime),
    ) -> dict[str, Any]:
        if current.database.get_index(index_id) is None:
            raise HTTPException(status_code=404, detail="Index not found")
        uploaded = []
        for file in files:
            path, artifact = await save_upload(file, current=current, index_id=index_id)
            current.database.add_document_to_index(index_id, file.filename or path.name, str(path))
            uploaded.append({"filename": file.filename or path.name, "stored_path": str(path), "artifact_id": artifact.id})
        return {"message": f"Uploaded {len(uploaded)} files", "uploaded_files": uploaded, "index": current.database.get_index(index_id)}

    @app.get("/sessions/{session_id}/documents", dependencies=protected)
    def session_documents(session_id: str, current: Runtime = Depends(get_runtime)) -> dict[str, Any]:
        session = current.database.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        docs = current.database.get_documents_for_session(session_id)
        return {"session": session, "files": [Path(item).name.split("_", 1)[-1] for item in docs], "file_count": len(docs)}

    @app.post("/indexes/{index_id}/build", dependencies=protected)
    async def build_index(index_id: str, options: dict[str, Any] | None = None, current: Runtime = Depends(get_runtime)) -> dict[str, Any]:
        if options:
            current.database.update_index_metadata(index_id, normalize_index_options(options))
        run = _submit_index_job(index_id, 3600, current)
        completed = await asyncio.to_thread(current.manager.wait, run.id, 3660)
        if completed.status != RunStatus.COMPLETED:
            raise HTTPException(status_code=502, detail=completed.error or "Index build failed")
        result = completed.result or {}
        return {**result, "message": result.get("message", "Index build complete"), "index": current.database.get_index(index_id), "run_id": run.id}

    def _submit_index_job(index_id: str, timeout_seconds: float, current: Runtime):
        index = current.database.get_index(index_id)
        if index is None:
            raise HTTPException(status_code=404, detail="Index not found")
        paths = [item["stored_path"] for item in index["documents"]]
        if not paths:
            raise HTTPException(status_code=422, detail="Index has no documents")
        options = normalize_index_options(index.get("metadata") or {})
        payload = {
            **options,
            "file_paths": paths,
            "table_name": index["vector_table_name"],
            "index_id": index_id,
        }
        return current.manager.submit_index(
            {"index_id": index_id, "payload": payload, "timeout_seconds": timeout_seconds}
        )

    @app.post("/v1/index-jobs", status_code=202, dependencies=protected)
    def submit_index_job(payload: IndexJobRequest, current: Runtime = Depends(get_runtime)) -> dict[str, Any]:
        return _run_json(_submit_index_job(payload.index_id, payload.timeout_seconds, current))

    @app.post("/sessions/{session_id}/index", dependencies=protected)
    async def index_session_documents(session_id: str, current: Runtime = Depends(get_runtime)) -> dict[str, Any]:
        session = current.database.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        paths = current.database.get_documents_for_session(session_id)
        if not paths:
            raise HTTPException(status_code=422, detail="Session has no uploaded documents")
        name = f"{session['title']} {uuid.uuid4().hex[:8]}"
        index_id = current.database.create_index(name, "Created from session uploads", normalize_index_options({}))
        for path in paths:
            current.database.add_document_to_index(index_id, Path(path).name.split("_", 1)[-1], path)
        run = _submit_index_job(index_id, 3600, current)
        completed = await asyncio.to_thread(current.manager.wait, run.id, 3660)
        if completed.status != RunStatus.COMPLETED:
            raise HTTPException(status_code=502, detail=completed.error or "Index build failed")
        current.database.link_index_to_session(session_id, index_id)
        current.database.clear_documents_for_session(session_id)
        return {
            "message": "Documents indexed and linked",
            "index_id": index_id,
            "index": current.database.get_index(index_id),
            "build": completed.result or {},
            "run_id": run.id,
        }

    @app.post("/sessions/{session_id}/indexes/{index_id}", dependencies=protected)
    def link_index(session_id: str, index_id: str, current: Runtime = Depends(get_runtime)) -> dict[str, Any]:
        if current.database.get_session(session_id) is None:
            raise HTTPException(status_code=404, detail="Session not found")
        if current.database.get_index(index_id) is None:
            raise HTTPException(status_code=404, detail="Index not found")
        current.database.link_index_to_session(session_id, index_id)
        return {"message": "Index linked", "linked": True, "session_id": session_id, "index_id": index_id}

    @app.get("/sessions/{session_id}/indexes", dependencies=protected)
    def session_indexes(session_id: str, current: Runtime = Depends(get_runtime)) -> dict[str, Any]:
        if current.database.get_session(session_id) is None:
            raise HTTPException(status_code=404, detail="Session not found")
        ids = current.database.get_indexes_for_session(session_id)
        return {"indexes": [current.database.get_index(item) for item in ids], "total": len(ids)}

    @app.delete("/indexes/{index_id}", dependencies=protected)
    async def delete_index(index_id: str, current: Runtime = Depends(get_runtime)) -> dict[str, Any]:
        if current.database.get_index(index_id) is None:
            raise HTTPException(status_code=404, detail="Index not found")

        def delete_artifacts() -> None:
            response = requests.delete(
                f"{current.rag_api_url.rstrip('/')}/indexes/{index_id}",
                headers=_rag_headers(),
                timeout=60,
            )
            response.raise_for_status()

        try:
            await asyncio.to_thread(delete_artifacts)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"RAG artifact deletion failed: {exc}") from exc
        current.database.delete_index(index_id)
        return {"message": "Index deleted", "deleted": True, "index_id": index_id}

    @app.get("/v1/artifacts", dependencies=protected)
    def list_artifacts(
        session_id: str | None = None,
        index_id: str | None = None,
        current: Runtime = Depends(get_runtime),
    ) -> dict[str, Any]:
        if session_id is None and index_id is None:
            raise HTTPException(status_code=422, detail="session_id or index_id is required")
        return {"artifacts": [_artifact_json(item) for item in current.artifacts.list(session_id=session_id, index_id=index_id)]}

    @app.get("/v1/artifacts/{artifact_id}", dependencies=protected)
    def download_artifact(
        artifact_id: str,
        session_id: str | None = None,
        index_id: str | None = None,
        current: Runtime = Depends(get_runtime),
    ) -> Response:
        artifact = current.artifacts.get(artifact_id)
        if artifact is None or not (
            (session_id is not None and artifact.session_id == session_id)
            or (index_id is not None and artifact.index_id == index_id)
        ):
            raise HTTPException(status_code=404, detail="Artifact not found")
        return Response(
            current.artifacts.read_bytes(artifact_id),
            media_type=artifact.mime_type,
            headers={"Content-Disposition": f'attachment; filename="{artifact.filename}"'},
        )

    @app.get("/v1/tools", dependencies=protected)
    def list_tools(current: Runtime = Depends(get_runtime)) -> dict[str, Any]:
        return {"tools": current.tools.descriptors()}

    @app.post("/v1/tools/{tool_name}/execute", dependencies=protected)
    async def execute_tool(tool_name: str, payload: ToolExecution, current: Runtime = Depends(get_runtime)) -> dict[str, Any]:
        context = ToolContext(
            run_id=f"standalone-{uuid.uuid4()}",
            session_id=payload.session_id,
            permissions=_server_tool_permissions() & set(payload.permissions),
            approved_tools={tool_name} if payload.approved else set(),
        )
        try:
            result = await current.tools.execute(tool_name, payload.arguments, context)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"tool": tool_name, "result": result}

    @app.get("/v1/skills", dependencies=protected)
    def list_skills(current: Runtime = Depends(get_runtime)) -> dict[str, Any]:
        return {"skills": [asdict(item) for item in current.skills.list()]}

    @app.post("/v1/skills", status_code=201, dependencies=protected)
    def create_skill(payload: SkillContent, current: Runtime = Depends(get_runtime)) -> dict[str, Any]:
        try:
            return {"skill": asdict(current.skills.create(payload.content))}
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/skills/{skill_id}/versions", status_code=201, dependencies=protected)
    def create_skill_version(skill_id: str, payload: SkillContent, current: Runtime = Depends(get_runtime)) -> dict[str, Any]:
        try:
            return {"skill": asdict(current.skills.create_version(skill_id, payload.content))}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Skill not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/v1/connectors", dependencies=protected)
    def connectors() -> dict[str, Any]:
        databases = DatabaseConnectorRegistry().connectors
        mcp = MCPConnectorRegistry().connectors
        return {
            "database": [
                {"id": item.id, "description": item.description, "max_rows": item.max_rows}
                for item in databases.values()
            ],
            "mcp": [
                {
                    "id": item.id,
                    "url": item.url,
                    "allowed_tools": sorted(item.allowed_tools) if item.allowed_tools else None,
                    "approval_required": item.approval_required,
                }
                for item in mcp.values()
            ],
        }

    @app.exception_handler(UploadRejected)
    async def upload_error(_request: Request, exc: UploadRejected) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    return app


app = create_app()

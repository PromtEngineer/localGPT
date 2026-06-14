import asyncio
import json
import os
import re
import threading
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, cast

import requests
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

load_dotenv()  # .env  — main config
load_dotenv(".env.keys", override=False)  # .env.keys — API keys (never committed)

# Add parent directory to path so we can import rag_system modules
import sys

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
for path in (BACKEND_DIR, PROJECT_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

# Import RAG system modules for complete metadata
try:
    from rag_system.main import EXTERNAL_MODELS, OLLAMA_CONFIG, PIPELINE_CONFIGS

    RAG_SYSTEM_AVAILABLE = True
    print("✅ RAG system modules accessible from backend")
except Exception as e:
    EXTERNAL_MODELS = {}
    OLLAMA_CONFIG = {}
    PIPELINE_CONFIGS = {}
    RAG_SYSTEM_AVAILABLE = False
    print(f"⚠️ RAG system modules not available: {e}")

# Enable WAL journal mode before ANY connection is opened.
# isolation_level=None (autocommit) is required because PRAGMA journal_mode
# is silently ignored when issued inside an open transaction.
import sqlite3 as _sqlite3

_DB_PATH = os.path.join(BACKEND_DIR, "chat_data.db")
try:
    _wal_conn = _sqlite3.connect(_DB_PATH, timeout=30, isolation_level=None)
    _wal_mode = _wal_conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
    _wal_conn.close()
    print(f"✅ SQLite journal mode: {_wal_mode}")
except Exception as _wal_err:
    print(f"⚠️ Could not enable WAL mode: {_wal_err}")

import simple_pdf_processor as pdf_module
from database import db, generate_session_title
from ollama_client import OllamaClient, OllamaError
from pydantic import ValidationError
from validators import (
    IndexBuildRequest,
    MessageRequest,
    RenameSessionRequest,
    SessionRequest,
    validate_file_upload,
)

from rag_system.chat_runtime import execute_chat as execute_rag_chat
from rag_system.factory import get_agent as create_rag_agent
from rag_system.index_selection import select_active_index_id
from rag_system.indexing_runtime import (
    build_config as build_indexing_config,
)
from rag_system.indexing_runtime import (
    execute_index_build,
)
from rag_system.metadata_filters import (
    FilterError,
    validate_document_metadata,
    validate_schema,
)

# Import maintenance tools
try:
    from rag_system.maintenance import MaintenanceTools

    MAINTENANCE_TOOLS_AVAILABLE = True
except ImportError:
    MAINTENANCE_TOOLS_AVAILABLE = False

# Import job persistence
try:
    from rag_system.job_persistence import JobProgressTracker

    JOB_PERSISTENCE_AVAILABLE = True
except ImportError:
    JOB_PERSISTENCE_AVAILABLE = False

# Initialize FastAPI app
app = FastAPI(title="LocalGPT Backend", version="1.0.0")


def _cors_origins_from_env() -> list[str]:
    origins = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    )
    parsed = [origin.strip() for origin in origins.split(",") if origin.strip()]
    return parsed or ["http://localhost:3000"]


_cors_origins = _cors_origins_from_env()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials="*" not in _cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import metrics singleton
try:
    from metrics import metrics as _metrics_imported

    _metrics: Any = _metrics_imported
except ImportError:
    _metrics = None


@app.middleware("http")
async def _restrict_maintenance(request: Request, call_next):
    if request.url.path.startswith("/maintenance"):
        # "testclient" is the synthetic peer address Starlette's TestClient
        # uses for in-process requests — those never cross the network.
        host = request.client.host if request.client else ""
        if host not in ("127.0.0.1", "::1", "testclient"):
            from fastapi.responses import JSONResponse

            return JSONResponse(
                status_code=403,
                content={
                    "detail": "Maintenance endpoints are only accessible from localhost"
                },
            )
    return await call_next(request)


@app.middleware("http")
async def _record_metrics(request: Request, call_next):
    if _metrics is None:
        return await call_next(request)
    _metrics.inc_active()
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    finally:
        latency_ms = (time.perf_counter() - t0) * 1000
        _metrics.dec_active()
        # Record the route template (/sessions/{session_id}/messages), not the
        # concrete URL — per-UUID paths grow the metrics dict without bound.
        route = request.scope.get("route")
        _metrics.record_request(
            getattr(route, "path", None) or request.url.path, latency_ms
        )
    return response


# Global variables
ollama_client = OllamaClient()
pdf_processor = None
index_jobs_lock = threading.Lock()
index_jobs: Dict[str, Dict[str, Any]] = {}
_rag_agent = None
_rag_agent_init_lock = threading.Lock()
STALE_BUILD_AFTER = timedelta(minutes=10)
BACKEND_BASE_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

# Upload safety limits
_MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500 MB per file
_ALLOWED_UPLOAD_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".md",
    ".rst",
    ".tex",
    ".docx",
    ".doc",
    ".pptx",
    ".ppt",
    ".xlsx",
    ".xls",
    ".html",
    ".htm",
    ".csv",
    ".json",
    ".xml",
    ".yaml",
    ".yml",
}
_UPLOAD_CHUNK_BYTES = 1024 * 1024  # stream uploads to disk 1 MB at a time
# Anchor to the repo root: a CWD-relative path creates a second uploads
# directory when the server is launched from backend/ instead of the root
_UPLOAD_DIR = os.path.join(PROJECT_ROOT, "shared_uploads")


def _get_local_rag_agent():
    """Lazily initialize the shared agent used by FastAPI chat endpoints."""
    global _rag_agent
    if _rag_agent is None:
        with _rag_agent_init_lock:
            if _rag_agent is None:
                _rag_agent = create_rag_agent(os.getenv("RAG_CONFIG_MODE", "default"))
    return _rag_agent


def _validation_error_detail(e: ValidationError) -> str:
    first = e.errors()[0]
    loc = ".".join(str(p) for p in first.get("loc", ()))
    msg = first.get("msg", "invalid value")
    return f"{loc}: {msg}" if loc else msg


async def _save_uploads(
    files: List[UploadFile], upload_dir: str = _UPLOAD_DIR
) -> List[Dict[str, str]]:
    """Validate every file, then stream them all to disk.

    Validation happens for the whole batch before anything is written, and a
    failure mid-write removes everything written so far — so a rejected batch
    never leaves partial files behind for the caller to register.
    Returns [{"filename", "stored_path"}]; DB registration is the caller's job.
    """
    os.makedirs(upload_dir, exist_ok=True)

    candidates = [f for f in files if f.filename]
    if not candidates:
        raise HTTPException(status_code=400, detail="No files were uploaded")

    # Pass 1: validate every file before writing anything
    for file in candidates:
        ext = os.path.splitext(cast(str, file.filename))[1].lower()
        if ext not in _ALLOWED_UPLOAD_EXTENSIONS:
            raise HTTPException(
                status_code=415,
                detail=f"'{file.filename}': unsupported file type '{ext}'",
            )
        result = validate_file_upload(file, max_size_bytes=_MAX_UPLOAD_BYTES)
        if (
            not result.valid
            and "not allowed" in (result.error or "")
            and "MIME" in (result.error or "")
        ):
            raise HTTPException(
                status_code=415, detail=f"'{file.filename}': {result.error}"
            )
        if file.size and file.size > _MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"'{file.filename}' exceeds the 500 MB upload limit",
            )

    # Pass 2: stream to disk in chunks (never the whole file in memory)
    saved: List[Dict[str, str]] = []
    file_path = None
    try:
        for file in candidates:
            unique_filename = (
                f"{uuid.uuid4()}_{os.path.basename(cast(str, file.filename))}"
            )
            file_path = os.path.join(upload_dir, unique_filename)
            written = 0
            with open(file_path, "wb") as out:
                while chunk := await file.read(_UPLOAD_CHUNK_BYTES):
                    written += len(chunk)
                    if written > _MAX_UPLOAD_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail=f"'{file.filename}' exceeds the 500 MB upload limit",
                        )
                    out.write(chunk)
            saved.append(
                {
                    "filename": cast(str, file.filename),
                    "stored_path": os.path.abspath(file_path),
                }
            )
            file_path = None
    except Exception:
        # Roll back: remove completed files and the partially written one
        if file_path:
            saved.append({"filename": "", "stored_path": file_path})
        for item in saved:
            try:
                if os.path.exists(item["stored_path"]):
                    os.remove(item["stored_path"])
            except OSError:
                pass
        raise

    return saved


# Initialize maintenance tools
maintenance_tools = None
if MAINTENANCE_TOOLS_AVAILABLE:
    try:
        maintenance_tools = MaintenanceTools(
            db_path=_DB_PATH,
            project_root=PROJECT_ROOT,
            lancedb_path="lancedb",
            uploads_path="shared_uploads",
            index_store_path="index_store",
        )
    except Exception as e:
        print(f"⚠️ Failed to initialize maintenance tools: {e}")
        maintenance_tools = None

# Initialize job persistence tracker
job_progress_tracker = None
if JOB_PERSISTENCE_AVAILABLE:
    try:
        job_progress_tracker = JobProgressTracker(db_path=_DB_PATH)
        print("✅ Job persistence tracker initialized")
    except Exception as e:
        print(f"⚠️ Failed to initialize job persistence: {e}")
        job_progress_tracker = None


def _format_bytes(size: int) -> str:
    value = float(max(size, 0))
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} GB"


def _large_indexing_model(model: str | None) -> bool:
    if not model:
        return False
    lowered = model.lower()
    return any(
        token in lowered for token in ("gpt-oss", "120b", "70b", "large", "cloud")
    )


def _index_build_preflight(
    index_id: str, data: Dict[str, Any] | None = None, *, check_services: bool = True
) -> Dict[str, Any]:
    data = data or {}
    index = db.get_index(index_id)
    if not index:
        raise HTTPException(status_code=404, detail="Index not found")

    errors: List[str] = []
    warnings: List[str] = []
    missing_files: List[Dict[str, str]] = []
    unreadable_files: List[Dict[str, str]] = []
    document_count = 0
    total_bytes = 0

    documents = index.get("documents", [])
    if not documents:
        errors.append("No documents are attached to this index.")

    for doc in documents:
        document_count += 1
        path = str(doc.get("stored_path") or "")
        filename = str(doc.get("filename") or os.path.basename(path) or "unknown")
        if not path or not os.path.exists(path):
            missing_files.append({"filename": filename, "stored_path": path})
            continue
        if not os.path.isfile(path) or not os.access(path, os.R_OK):
            unreadable_files.append({"filename": filename, "stored_path": path})
            continue
        size = os.path.getsize(path)
        total_bytes += size
        if size == 0:
            warnings.append(f"{filename} is empty and may be skipped.")

    if missing_files:
        sample = ", ".join(item["filename"] for item in missing_files[:5])
        errors.append(
            f"{len(missing_files)} uploaded file(s) are missing from disk: {sample}"
        )
    if unreadable_files:
        sample = ", ".join(item["filename"] for item in unreadable_files[:5])
        errors.append(
            f"{len(unreadable_files)} uploaded file(s) are not readable: {sample}"
        )

    if document_count > 100:
        warnings.append(
            f"This build has {document_count} files. Prefer Fast mode or smaller batches for best stability."
        )
    if total_bytes > 500 * 1024 * 1024:
        warnings.append(
            f"This build is {_format_bytes(total_bytes)}. Large builds can take a long time on local hardware."
        )
    if bool(data.get("forceReindex")):
        warnings.append(
            "Force reindex will rebuild all files, including unchanged documents."
        )
    if bool(data.get("enableEnrich", False)) and document_count > 50:
        warnings.append(
            "Context enrichment on large file sets can be slow. Fast mode is safer for the first pass."
        )

    for key, label in (("enrichModel", "enrichment"), ("overviewModel", "overview")):
        model = data.get(key)
        if _large_indexing_model(model):
            warnings.append(
                f"The {label} model '{model}' will be replaced with qwen3:8b for indexing safety."
            )

    rag_api_available = None
    if check_services:
        rag_api_available = RAG_SYSTEM_AVAILABLE
        if not RAG_SYSTEM_AVAILABLE:
            errors.append(
                "RAG indexing dependencies are not available in the FastAPI process. "
                "Start LocalGPT with the project virtual environment."
            )

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "document_count": document_count,
        "total_bytes": total_bytes,
        "total_size": _format_bytes(total_bytes),
        "missing_files": missing_files,
        "unreadable_files": unreadable_files,
        "rag_api_available": rag_api_available,
    }


def _lancedb_path_candidates() -> List[str]:
    candidates = [
        os.environ.get("LANCEDB_PATH"),
        os.path.join(PROJECT_ROOT, "lancedb"),
        os.path.join(PROJECT_ROOT, "rag_system", "index_store", "lancedb"),
    ]
    result: List[str] = []
    for path in candidates:
        if path and path not in result:
            result.append(path)
    return result


def _inspect_vector_table(table_name: str | None) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "expected_table": table_name,
        "exists": False,
        "path": None,
        "row_count": None,
        "latechunk_exists": False,
        "error": None,
    }
    if not table_name:
        result["error"] = "Index does not have a vector table name."
        return result

    try:
        import lancedb
    except Exception as e:
        result["error"] = f"LanceDB is not importable: {e}"
        return result

    table_names: List[str] = []
    for db_path in _lancedb_path_candidates():
        if not os.path.exists(db_path):
            continue
        try:
            conn = lancedb.connect(db_path)  # type: ignore[attr-defined]
            # list_tables() returns a ListTablesResponse that iterates as
            # (key, value) pairs: {"tables": [...], "page_token": ...}
            # table_names() is deprecated but returns a plain list.
            if hasattr(conn, "list_tables"):
                raw = conn.list_tables()
                names_map = dict(raw)  # {"tables": [...], "page_token": ...}
                names = names_map.get("tables", [])
            elif hasattr(conn, "table_names"):
                names = list(conn.table_names(limit=10_000))
            else:
                names = []
            table_names = list(names)
            if table_name not in table_names:
                continue
            table = conn.open_table(table_name)
            row_count = None
            if hasattr(table, "count_rows"):
                row_count = int(table.count_rows())
            result.update(
                {
                    "exists": True,
                    "path": db_path,
                    "row_count": row_count,
                    "latechunk_exists": f"{table_name}_lc" in table_names,
                    "error": None,
                }
            )
            return result
        except Exception as e:
            result["error"] = f"Could not inspect LanceDB at {db_path}: {e}"

    if not result["error"]:
        searched = ", ".join(_lancedb_path_candidates())
        result["error"] = (
            f"Vector table was not found in searched LanceDB paths: {searched}"
        )
    return result


def _overview_diagnostics(index_id: str) -> Dict[str, Any]:
    paths = _overview_path_candidates(index_id)
    for path in paths:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    line_count = sum(1 for line in handle if line.strip())
            except Exception:
                line_count = None
            return {"exists": True, "path": path, "line_count": line_count}
    return {"exists": False, "path": None, "line_count": 0}


def _overview_path_candidates(index_id: str) -> List[str]:
    return [
        os.path.join(PROJECT_ROOT, "index_store", "overviews", f"{index_id}.jsonl"),
        os.path.join(
            PROJECT_ROOT, "rag_system", "index_store", "overviews", f"{index_id}.jsonl"
        ),
    ]


def _lancedb_table_names(conn) -> List[str]:
    if hasattr(conn, "list_tables"):
        return list(dict(conn.list_tables()).get("tables", []))
    if hasattr(conn, "table_names"):
        return list(conn.table_names(limit=10_000))
    return []


def _delete_index_artifacts(index: Dict[str, Any]) -> Dict[str, List[str]]:
    """Delete storage artifacts while the index record still describes them."""
    removed: Dict[str, List[str]] = {
        "tables": [],
        "files": [],
        "overviews": [],
        "skipped_files": [],
        "shared_files": [],
    }
    failures: List[str] = []
    table_name = index.get("vector_table_name")

    if table_name:
        try:
            import lancedb
        except Exception as e:
            failures.append(f"LanceDB is not importable: {e}")
        else:
            for db_path in _lancedb_path_candidates():
                if not os.path.exists(db_path):
                    continue
                try:
                    conn = lancedb.connect(db_path)  # type: ignore[attr-defined]
                    names = _lancedb_table_names(conn)
                    for candidate in (table_name, f"{table_name}_lc"):
                        if candidate in names:
                            conn.drop_table(candidate)
                            removed["tables"].append(f"{db_path}:{candidate}")
                except Exception as e:
                    failures.append(f"Could not clean LanceDB at {db_path}: {e}")

    upload_root = os.path.realpath(_UPLOAD_DIR)
    # A file still referenced by another index must survive this delete. The
    # maintenance path (maintenance._delete_index_completely) already guards
    # this; mirror it here so removing one index never deletes an upload that
    # another index depends on. Behavior-neutral today (each upload gets a
    # unique uuid path), but a safety net for any future clone/import feature.
    current_id = index.get("id")
    shared_real_paths: set = set()
    try:
        for other in db.list_indexes():
            if other.get("id") == current_id:
                continue
            for other_doc in other.get("documents") or []:
                other_path = other_doc.get("stored_path")
                if other_path:
                    shared_real_paths.add(os.path.realpath(other_path))
    except Exception:
        shared_real_paths = set()

    for document in index.get("documents") or []:
        stored_path = document.get("stored_path")
        if not stored_path:
            continue
        real_path = os.path.realpath(stored_path)
        try:
            if os.path.commonpath([upload_root, real_path]) != upload_root:
                removed["skipped_files"].append(real_path)
                continue
        except ValueError:
            removed["skipped_files"].append(real_path)
            continue
        if real_path in shared_real_paths:
            # Owned and under uploads, but another index references it → keep.
            removed["shared_files"].append(real_path)
            continue
        try:
            if os.path.exists(real_path):
                os.remove(real_path)
                removed["files"].append(real_path)
        except OSError as e:
            failures.append(f"Could not remove upload {stored_path}: {e}")

    for overview_path in _overview_path_candidates(index["id"]):
        try:
            if os.path.exists(overview_path):
                os.remove(overview_path)
                removed["overviews"].append(overview_path)
        except OSError as e:
            failures.append(f"Could not remove overview {overview_path}: {e}")

    if failures:
        raise RuntimeError("; ".join(failures))
    return removed


def _clear_index_build_artifacts(index_id: str, table_name: str | None) -> None:
    """Remove generated vector/overview artifacts before a force rebuild."""
    if table_name:
        import lancedb

        for db_path in _lancedb_path_candidates():
            if not os.path.exists(db_path):
                continue
            conn = lancedb.connect(db_path)  # type: ignore[attr-defined]
            names = _lancedb_table_names(conn)
            for candidate in (table_name, f"{table_name}_lc"):
                if candidate in names:
                    conn.drop_table(candidate)

    for overview_path in _overview_path_candidates(index_id):
        if os.path.exists(overview_path):
            os.remove(overview_path)


def _index_diagnostics(index_id: str) -> Dict[str, Any]:
    index = db.get_index(index_id)
    if not index:
        raise HTTPException(status_code=404, detail="Index not found")

    errors: List[str] = []
    warnings: List[str] = []
    recommendations: List[str] = []
    metadata = index.get("metadata") or {}

    preflight = _index_build_preflight(index_id, {}, check_services=False)
    errors.extend(preflight["errors"])
    warnings.extend(preflight["warnings"])

    vector_table = _inspect_vector_table(index.get("vector_table_name"))
    if not vector_table["exists"]:
        errors.append(
            "Vector table is missing. Rebuild this index before trusting retrieval."
        )
    elif vector_table["row_count"] == 0:
        errors.append("Vector table exists but has no rows. Force rebuild this index.")

    overview = _overview_diagnostics(index_id)
    if metadata.get("enable_enrich", True) and not overview["exists"]:
        warnings.append(
            "Document overview file is missing. Routing quality may be weaker until the index is rebuilt."
        )

    latest_job = db.get_latest_index_job(
        index_id, include_options=False, include_files=True
    )
    file_status_counts: Dict[str, int] = {}
    if latest_job and latest_job.get("files"):
        for item in latest_job["files"]:
            status = str(item.get("status") or "unknown")
            file_status_counts[status] = file_status_counts.get(status, 0) + 1
        failed_count = file_status_counts.get("failed", 0)
        pending_count = file_status_counts.get("pending", 0)
        if failed_count:
            errors.append(f"{failed_count} file(s) failed in the latest build job.")
        if pending_count and latest_job.get("status") in {
            "completed",
            "failed",
            "cancelled",
        }:
            warnings.append(
                f"{pending_count} file(s) were left pending in the latest build job."
            )

    metadata_status = metadata.get("status")
    if metadata_status in {"failed", "incomplete", "empty"}:
        errors.append(f"Index metadata status is '{metadata_status}'.")
    elif metadata_status in {"building", "cancelled"}:
        warnings.append(f"Index metadata status is '{metadata_status}'.")

    source_blockers = bool(
        preflight["missing_files"]
        or preflight["unreadable_files"]
        or preflight["document_count"] == 0
    )
    if source_blockers:
        recommended_action = "fix_sources"
    elif errors:
        recommended_action = "force_rebuild"
    elif warnings:
        recommended_action = "rebuild"
    else:
        recommended_action = "none"

    if source_blockers:
        recommendations.append(
            "Re-upload missing or unreadable files before rebuilding."
        )
    elif errors:
        recommendations.append(
            "Run Force rebuild after confirming the source files still exist."
        )
    elif warnings:
        recommendations.append("A normal rebuild is recommended when convenient.")
    else:
        recommendations.append("Index diagnostics look healthy.")

    health = "unhealthy" if errors else "warning" if warnings else "healthy"
    return {
        "index_id": index_id,
        "name": index.get("name"),
        "health": health,
        "ok": health == "healthy",
        "errors": errors,
        "warnings": warnings,
        "recommendations": recommendations,
        "recommended_action": recommended_action,
        "can_repair": recommended_action in {"force_rebuild", "rebuild"},
        "document_count": preflight["document_count"],
        "total_bytes": preflight["total_bytes"],
        "total_size": preflight["total_size"],
        "missing_files": preflight["missing_files"],
        "unreadable_files": preflight["unreadable_files"],
        "metadata_status": metadata_status,
        "vector_table": vector_table,
        "overview": overview,
        "latest_job": latest_job,
        "file_status_counts": file_status_counts,
    }


def _index_diagnostics_summary(index_id: str) -> Dict[str, Any]:
    diagnostics = _index_diagnostics(index_id)
    vector_table = diagnostics.get("vector_table") or {}
    return {
        "index_id": index_id,
        "name": diagnostics.get("name"),
        "health": diagnostics.get("health"),
        "ok": diagnostics.get("ok"),
        "recommended_action": diagnostics.get("recommended_action"),
        "can_repair": diagnostics.get("can_repair"),
        "error_count": len(diagnostics.get("errors") or []),
        "warning_count": len(diagnostics.get("warnings") or []),
        "document_count": diagnostics.get("document_count"),
        "total_size": diagnostics.get("total_size"),
        "vector_exists": bool(vector_table.get("exists")),
        "vector_rows": vector_table.get("row_count"),
        "metadata_status": diagnostics.get("metadata_status"),
    }


def _raise_for_unhealthy_session_indexes(session_id: str) -> List[str]:
    idx_ids = db.get_indexes_for_session(session_id)
    unhealthy: List[Dict[str, Any]] = []
    for idx_id in idx_ids:
        diagnostics = _index_diagnostics(idx_id)
        if diagnostics.get("health") == "unhealthy":
            unhealthy.append(diagnostics)
    if unhealthy:
        names = ", ".join(
            str(item.get("name") or item.get("index_id")) for item in unhealthy
        )
        raise HTTPException(
            status_code=409,
            detail={
                "message": f"Cannot chat with unhealthy linked index(es): {names}. Run diagnostics and repair before chatting.",
                "diagnostics": unhealthy,
            },
        )
    return idx_ids


def _update_index_job(job_id: str, **updates):
    db.update_index_job(job_id, updates)
    with index_jobs_lock:
        job = index_jobs.get(job_id)
        if not job:
            return
        job.update(updates)
        job["updated_at"] = datetime.now().isoformat()


def _get_index_job(job_id: str) -> Optional[Dict[str, Any]]:
    with index_jobs_lock:
        job = index_jobs.get(job_id)
        if job:
            return dict(job)
    return db.get_index_job(job_id, include_options=True, include_files=True)


def _public_index_job(job_id: str) -> Optional[Dict[str, Any]]:
    return db.get_index_job(job_id, include_options=False, include_files=True)


def _recover_stale_index_builds() -> int:
    """Mark orphaned in-progress builds as failed after backend/RAG restarts."""
    recovered = 0
    now = datetime.now()
    for idx in db.list_indexes():
        meta = idx.get("metadata") or {}
        if meta.get("status") != "building":
            continue

        job_id = meta.get("build_job_id")
        with index_jobs_lock:
            if job_id and str(job_id) in index_jobs:
                continue
        persisted_job = db.get_index_job(str(job_id)) if job_id else None
        job_was_recovered = bool(
            persisted_job and persisted_job.get("status") == "paused"
        )

        started_raw = meta.get("build_started_at")
        try:
            started_at = (
                datetime.fromisoformat(str(started_raw)) if started_raw else None
            )
        except ValueError:
            started_at = None

        if (
            not job_was_recovered
            and started_at
            and now - started_at < STALE_BUILD_AFTER
        ):
            continue

        db.update_index_metadata(
            idx["id"],
            {
                "status": "paused",
                "build_paused_at": now.isoformat(),
                "build_error": (
                    "Previous build was interrupted or the backend restarted before the "
                    "background job could finish. Resume the build to continue."
                ),
            },
        )
        recovered += 1
    for job in db.list_unfinished_index_jobs():
        with index_jobs_lock:
            if job["id"] in index_jobs:
                continue
        started_raw = job.get("updated_at") or job.get("created_at")
        try:
            started_at = (
                datetime.fromisoformat(str(started_raw)) if started_raw else None
            )
        except ValueError:
            started_at = None
        if started_at and now - started_at < STALE_BUILD_AFTER:
            continue
        db.update_index_job(
            job["id"],
            {
                "status": "paused",
                "stage": "paused",
                "message": "Build interrupted by backend restart; resume to continue.",
                "error": "Previous build was interrupted or the backend restarted before the background job could finish.",
            },
        )
    return recovered


@app.post("/index-jobs/{job_id}/progress")
async def update_index_job_progress(job_id: str, request: Request):
    job = _get_index_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Index job not found")
    data = await request.json()
    updates = {
        key: data[key] for key in ("stage", "progress", "message") if key in data
    }
    if data.get("stage") == "completed":
        # The RAG API emits a final progress callback before _run_index_build()
        # finishes updating index metadata and storing the job result. Keep the
        # job non-terminal here; the background runner is the source of truth for
        # completion after all post-build bookkeeping is done.
        updates["stage"] = "finalizing"
        p = updates.get("progress")
        updates["progress"] = min(int(p if p is not None else 100), 99)
    elif data.get("stage") == "cancelled":
        updates["status"] = "cancelled"
        updates["finished_at"] = datetime.now().isoformat()
    _update_index_job(job_id, **updates)
    file_status = data.get("file_status")
    file_path = data.get("file_path")
    filename = data.get("filename") or data.get("document_id")
    if file_status and (file_path or filename):
        file_updates = {
            "status": file_status,
            "stage": data.get("stage"),
            "error": data.get("file_error") or data.get("error"),
        }
        if data.get("chunks_generated") is not None:
            file_updates["chunks_generated"] = int(data.get("chunks_generated") or 0)
        db.update_index_job_file(
            job_id, stored_path=file_path, filename=filename, updates=file_updates
        )
    return _public_index_job(job_id)


# Routes


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "rag_system_available": RAG_SYSTEM_AVAILABLE,
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "virtual_env": os.environ.get("VIRTUAL_ENV"),
        "ollama_running": await asyncio.to_thread(ollama_client.is_ollama_running),
        "available_models": await asyncio.to_thread(ollama_client.list_models),
        "database_stats": db.get_stats(),
    }


@app.get("/metrics", response_class=JSONResponse)
async def get_metrics(format: str = "json"):
    """Prometheus-compatible metrics endpoint.

    Use ?format=prometheus for text exposition format or default JSON.
    """
    if _metrics is None:
        raise HTTPException(status_code=503, detail="Metrics not available")
    if format == "prometheus":
        from fastapi.responses import PlainTextResponse

        return PlainTextResponse(
            _metrics.prometheus_text(), media_type="text/plain; version=0.0.4"
        )
    return _metrics.snapshot()


@app.get("/health/deep")
async def health_deep():
    """Deep health probe: checks DB, LanceDB, local RAG runtime, and Ollama."""
    checks: Dict[str, Any] = {}
    overall = "ok"

    # 1. SQLite
    try:
        import sqlite3 as _sqlite3

        _conn = _sqlite3.connect(db.db_path, timeout=3)
        _conn.execute("SELECT 1").fetchone()
        _conn.close()
        checks["db"] = "ok"
    except Exception as e:
        checks["db"] = f"error: {e}"
        overall = "degraded"

    # 2. LanceDB
    try:
        import lancedb as _lancedb

        for candidate in _lancedb_path_candidates():
            if os.path.exists(candidate):
                _lancedb.connect(candidate)  # type: ignore[attr-defined]
                checks["lancedb"] = "ok"
                break
        else:
            checks["lancedb"] = "not_found"
            overall = "degraded"
    except Exception as e:
        checks["lancedb"] = f"error: {e}"
        overall = "degraded"

    # 3. In-process RAG runtime
    try:
        checks["rag_runtime"] = "ready" if RAG_SYSTEM_AVAILABLE else "unavailable"
        if not RAG_SYSTEM_AVAILABLE:
            overall = "degraded"
    except Exception as e:
        checks["rag_runtime"] = f"error: {e}"
        overall = "degraded"

    # 4. Ollama
    try:
        ollama_base = getattr(ollama_client, "host", None) or getattr(
            ollama_client, "base_url", "http://localhost:11434"
        )
        resp = requests.get(f"{ollama_base}/api/tags", timeout=3)
        checks["ollama"] = (
            "ok" if resp.status_code == 200 else f"http_{resp.status_code}"
        )
        if resp.status_code != 200:
            overall = "degraded"
    except Exception as e:
        checks["ollama"] = f"error: {e}"
        overall = "degraded"

    return {"status": overall, "checks": checks}


@app.get("/sessions")
async def get_sessions():
    """Get all chat sessions"""
    try:
        sessions = db.get_sessions()
        return {"sessions": sessions, "total": len(sessions)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get sessions: {str(e)}")


@app.get("/sessions/cleanup")
async def cleanup_sessions():
    """Clean up empty sessions"""
    try:
        cleanup_count = db.cleanup_empty_sessions()
        return {
            "message": f"Cleaned up {cleanup_count} empty sessions",
            "cleanup_count": cleanup_count,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to cleanup sessions: {str(e)}"
        )


@app.post("/sessions")
async def create_session(request: Request):
    """Create a new chat session"""
    try:
        data = await request.json()
        try:
            req = SessionRequest(
                title=data.get("title") or "New Chat",
                model=data.get("model") or "llama3.2:latest",
            )
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=_validation_error_detail(e))

        session_id = db.create_session(req.title, req.model)
        session = db.get_session(session_id)

        return {"session": session, "session_id": session_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to create session: {str(e)}"
        )


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get a specific session with its messages"""
    try:
        session = db.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        messages = db.get_messages(session_id)

        return {"session": session, "messages": messages}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get session: {str(e)}")


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its messages"""
    try:
        deleted = db.delete_session(session_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Session not found")
        return {
            "message": "Session deleted successfully",
            "deleted_session_id": session_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}/documents")
async def get_session_documents(session_id: str):
    """Return documents and basic info for a session."""
    try:
        session = db.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        docs = db.get_documents_for_session(session_id)

        # Extract original filenames from stored paths
        filenames = [
            (
                os.path.basename(p).split("_", 1)[-1]
                if "_" in os.path.basename(p)
                else os.path.basename(p)
            )
            for p in docs
        ]

        return {"session": session, "files": filenames, "file_count": len(docs)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get documents: {str(e)}"
        )


@app.get("/sessions/{session_id}/indexes")
async def get_session_indexes(session_id: str):
    """Get indexes linked to a session"""
    try:
        idx_ids = db.get_indexes_for_session(session_id)
        indexes = []
        for idx_id in idx_ids:
            idx = db.get_index(idx_id)
            if idx:
                # Try to populate metadata for older indexes that have empty metadata
                if not idx.get("metadata") or len(idx["metadata"]) == 0:
                    print(f"🔍 Attempting to infer metadata for index {idx_id[:8]}...")
                    inferred_metadata = db.inspect_and_populate_index_metadata(idx_id)
                    if inferred_metadata:
                        # Refresh the index data with the new metadata
                        idx = db.get_index(idx_id)
                indexes.append(idx)
        return {"indexes": indexes, "total": len(indexes)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sessions/{session_id}/indexes/{index_id}")
async def link_index_to_session(session_id: str, index_id: str):
    """Link an index to a session"""
    try:
        diagnostics = _index_diagnostics(index_id)
        if diagnostics.get("health") == "unhealthy":
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "Index is unhealthy and cannot be opened safely. Run diagnostics and repair it first.",
                    "diagnostics": diagnostics,
                },
            )
        db.link_index_to_session(session_id, index_id)
        return {"message": "Index linked to session"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sessions/{session_id}/messages")
async def session_chat(session_id: str, request: Request):
    """
    Handle chat within a specific session.
    Intelligently routes between direct LLM (fast) and RAG pipeline (document-aware).
    """
    try:
        session = db.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        data = await request.json()
        try:
            message = MessageRequest(message=data.get("message", "")).message
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=_validation_error_detail(e))

        idx_ids = _raise_for_unhealthy_session_indexes(session_id)

        if session["message_count"] == 0:
            title = generate_session_title(message)
            db.update_session_title(session_id, title)

        # 🎯 SMART ROUTING: Decide between direct LLM vs RAG
        # Get overviews for routing decision
        aggregated = []
        if idx_ids:
            for idx_id in idx_ids:
                # Per-index overview paths
                overview_paths = [
                    f"../index_store/overviews/{idx_id}.jsonl",
                    f"index_store/overviews/{idx_id}.jsonl",
                    f"./index_store/overviews/{idx_id}.jsonl",
                ]
                for p in overview_paths:
                    if os.path.exists(p):
                        try:
                            with open(p, "r", encoding="utf-8") as f:
                                for line in f:
                                    if not line.strip():
                                        continue
                                    try:
                                        record = json.loads(line)
                                        overview = record.get("overview", "").strip()
                                        if overview:
                                            aggregated.append(overview)
                                    except json.JSONDecodeError:
                                        continue  # skip malformed lines
                        except Exception as e:
                            print(f"⚠️ Error reading {p}: {e}")
                            continue  # try next path variant on error
                        break  # found and read this path; stop trying variants for this index

        # 2️⃣  Fall back to legacy global file if no per-index overviews found
        if not aggregated:
            legacy_paths = [
                "../index_store/overviews/overviews.jsonl",
                "index_store/overviews/overviews.jsonl",
                "./index_store/overviews/overviews.jsonl",
            ]
            for p in legacy_paths:
                if os.path.exists(p):
                    print(f"⚠️ Falling back to legacy overviews file: {p}")
                    try:
                        with open(p, "r", encoding="utf-8") as f:
                            for line in f:
                                if not line.strip():
                                    continue
                                try:
                                    record = json.loads(line)
                                    overview = record.get("overview", "").strip()
                                    if overview:
                                        aggregated.append(overview)
                                except json.JSONDecodeError:
                                    continue
                    except Exception as e:
                        print(f"⚠️ Error reading legacy overviews file {p}: {e}")
                    break

        # Limit for performance
        if aggregated:
            print(
                f"✅ Loaded {len(aggregated)} document overviews from {len(idx_ids)} index(es)"
            )
        else:
            print(f"⚠️ No overviews found for indices {idx_ids}")
        aggregated = aggregated[:40]

        # Decide routing (force_rag / force_direct bypass heuristics entirely)
        force_rag = bool(data.get("force_rag", False))
        force_direct = bool(data.get("force_direct", False)) and not force_rag
        if force_direct:
            use_rag = False
        else:
            # _route_using_overviews makes a blocking LLM call — keep it off the event loop
            use_rag = force_rag or (
                await asyncio.to_thread(_route_using_overviews, message, aggregated)
                if aggregated
                else _simple_pattern_routing(message, idx_ids)
            )

        if use_rag:
            response_text, source_docs = await _handle_rag_query(
                session_id, message, data, idx_ids
            )
        else:
            response_text, source_docs = await _handle_direct_llm_query(
                session_id, message, session, requested_model=data.get("model")
            )

        # Store both turns only after a successful response — prevents orphaned user messages
        # if the RAG/LLM call raises an HTTPException
        db.add_message(session_id, message, "user")
        assistant_message_id = db.add_message(
            session_id, response_text, "assistant", metadata={"sources": source_docs}
        )

        return {
            "response": response_text,
            "sources": source_docs,
            "session_id": session_id,
            "message_id": assistant_message_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@app.post("/sessions/{session_id}/upload")
async def upload_files(session_id: str, files: List[UploadFile] = File(...)):
    """Handle file uploads and associate with the session."""
    uploaded_files = await _save_uploads(files)

    # Register in the DB only after every file is safely on disk, so a
    # rejected batch never leaves half its documents attached to the session
    for item in uploaded_files:
        db.add_document_to_session(session_id, item["stored_path"])

    return {
        "message": f"Successfully uploaded {len(uploaded_files)} files.",
        "uploaded_files": uploaded_files,
    }


@app.post("/sessions/{session_id}/index")
async def index_documents(session_id: str):
    """Triggers indexing for all documents in a session."""
    print(f"🔥 Received request to index documents for session {session_id[:8]}...")
    try:
        file_paths = db.get_documents_for_session(session_id)
        if not file_paths:
            return {"message": "No documents to index for this session."}

        print(f"Found {len(file_paths)} documents to index.")
        index_ids = db.get_indexes_for_session(session_id)
        active_id = select_active_index_id(index_ids)
        active_index = db.get_index(active_id) if active_id else None
        table_name = (
            active_index.get("vector_table_name")
            if active_index
            else PIPELINE_CONFIGS["default"]["storage"]["text_table_name"]
        )
        options = {"index_id": active_id or session_id, "table_name": table_name}
        config = build_indexing_config(PIPELINE_CONFIGS["default"], options)
        result = await asyncio.to_thread(
            execute_index_build,
            config,
            OLLAMA_CONFIG,
            file_paths,
            index_id=active_id or session_id,
            force_reindex=False,
            job_id=None,
            backend_base_url=BACKEND_BASE_URL,
        )
        return {
            "message": f"Indexed {len(file_paths)} document(s).",
            "table_name": table_name,
            "indexing_result": result,
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Exception during indexing: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {str(e)}"
        )


@app.post("/sessions/{session_id}/rename")
@app.put("/sessions/{session_id}/rename")
async def rename_session(session_id: str, request: Request):
    """Rename an existing session title"""
    try:
        session = db.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        data = await request.json()
        try:
            new_title = RenameSessionRequest(title=data.get("title", "")).title
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=_validation_error_detail(e))

        db.update_session_title(session_id, new_title)
        updated_session = db.get_session(session_id)

        return {"message": "Session renamed successfully", "session": updated_session}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to rename session: {str(e)}"
        )


@app.post("/rag/chat")
async def rag_chat(request: Request):
    """Run the transport-neutral RAG pipeline through FastAPI."""
    try:
        data = await request.json()
        return await asyncio.to_thread(
            execute_rag_chat, _get_local_rag_agent(), db, data
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FilterError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG processing failed: {e}")


@app.post("/rag/chat/stream")
async def rag_chat_stream(request: Request):
    """Stream RAG pipeline events over SSE from the FastAPI process."""
    data = await request.json()
    if not isinstance(data.get("query"), str) or not data["query"].strip():
        raise HTTPException(status_code=400, detail="Query is required")

    async def event_stream():
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        finished = object()

        def emit(event_type: str, payload: Any):
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {"type": event_type, "data": payload},
            )

        async def execute():
            try:
                result = await asyncio.to_thread(
                    execute_rag_chat,
                    _get_local_rag_agent(),
                    db,
                    data,
                    emit,
                )
                emit("complete", result)
            except Exception as e:
                emit("error", {"error": str(e)})
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, finished)

        task = asyncio.create_task(execute())
        try:
            while True:
                event = await queue.get()
                if event is finished:
                    break
                yield f"data: {json.dumps(event)}\n\n"
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/chat")
async def legacy_chat(request: Request):
    """Handle legacy chat requests (without sessions)"""
    try:
        data = await request.json()
        try:
            message = MessageRequest(message=data.get("message", "")).message
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=_validation_error_detail(e))
        model = data.get("model", "llama3.2:latest")
        conversation_history = data.get("conversation_history", [])

        # Check if Ollama is running
        if not await asyncio.to_thread(ollama_client.is_ollama_running):
            raise HTTPException(
                status_code=503,
                detail="Ollama is not running. Please start Ollama first.",
            )

        # Get response from Ollama (worker thread: the call blocks up to 60s)
        response = await asyncio.to_thread(
            ollama_client.chat, message, model, conversation_history
        )

        return {
            "response": response,
            "model": model,
            "message_count": len(conversation_history) + 1,
        }

    except HTTPException:
        raise
    except OllamaError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@app.get("/models")
async def get_models():
    """Get available models from both Ollama and HuggingFace, grouped by capability"""
    try:
        generation_models = []
        embedding_models = []

        # Get Ollama models if available
        if ollama_client.is_ollama_running():
            all_ollama_models = ollama_client.list_models()

            # Very naive classification - same logic as RAG API server
            ollama_embedding_models = [
                m
                for m in all_ollama_models
                if any(k in m for k in ["embed", "bge", "embedding", "text"])
            ]
            ollama_generation_models = [
                m for m in all_ollama_models if m not in ollama_embedding_models
            ]

            generation_models.extend(ollama_generation_models)
            embedding_models.extend(ollama_embedding_models)

        # Add supported HuggingFace embedding models from registry
        try:
            from rag_system.model_registry import huggingface_models

            embedding_models.extend(huggingface_models())
        except ImportError:
            embedding_models.extend(["Qwen/Qwen3-Embedding-0.6B"])

        # Sort models for consistent ordering
        generation_models.sort()
        embedding_models.sort()

        return {
            "generation_models": generation_models,
            "embedding_models": embedding_models,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not list models: {str(e)}")


@app.get("/indexes")
async def get_indexes():
    """Get all indexes"""
    try:
        data = db.list_indexes()
        return {"indexes": data, "total": len(data)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/indexes/diagnostics")
async def get_indexes_diagnostics():
    """Get compact health diagnostics for all indexes."""
    try:
        summaries = []
        for item in db.list_indexes():
            idx_id = item.get("id") or item.get("index_id")
            if not idx_id:
                continue
            try:
                summaries.append(_index_diagnostics_summary(str(idx_id)))
            except Exception as e:
                summaries.append(
                    {
                        "index_id": idx_id,
                        "name": item.get("name"),
                        "health": "unhealthy",
                        "ok": False,
                        "recommended_action": "force_rebuild",
                        "can_repair": bool(item.get("documents")),
                        "error_count": 1,
                        "warning_count": 0,
                        "document_count": len(item.get("documents") or []),
                        "total_size": "unknown",
                        "vector_exists": False,
                        "vector_rows": None,
                        "metadata_status": (item.get("metadata") or {}).get("status"),
                        "error": str(e),
                    }
                )
        return {"diagnostics": summaries, "total": len(summaries)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/indexes")
async def create_index(request: Request):
    """Create a new index"""
    try:
        data = await request.json()
        name = data.get("name")
        description = data.get("description")
        metadata = data.get("metadata", {})

        if not name:
            raise HTTPException(status_code=400, detail="Name required")
        if not isinstance(metadata, dict):
            raise HTTPException(status_code=400, detail="metadata must be an object")
        if metadata.get("metadata_schema") is not None:
            errors = validate_schema(metadata["metadata_schema"])
            if errors:
                raise HTTPException(status_code=400, detail="; ".join(errors))

        # Add complete metadata from RAG system configuration if available
        if RAG_SYSTEM_AVAILABLE and PIPELINE_CONFIGS.get("default"):
            default_config = PIPELINE_CONFIGS["default"]
            complete_metadata = {
                "status": "created",
                "metadata_source": "rag_system_config",
                "created_at": json.loads(json.dumps(datetime.now().isoformat())),
                "chunk_size": 512,  # From default config
                "chunk_overlap": 64,  # From default config
                "retrieval_mode": "hybrid",  # From default config
                "window_size": 5,  # From default config
                "embedding_model": default_config.get(
                    "embedding_model_name",
                    EXTERNAL_MODELS.get("embedding_model", "Qwen/Qwen3-Embedding-0.6B"),
                ),
                "enrich_model": OLLAMA_CONFIG.get("enrichment_model", "qwen3:8b"),
                "overview_model": OLLAMA_CONFIG.get("enrichment_model", "qwen3:8b"),
                "enable_enrich": False,  # From default config
                "latechunk": True,  # From default config
                "docling_chunk": True,  # From default config
                "note": "Default configuration from RAG system",
            }
            # Merge with any provided metadata
            complete_metadata.update(metadata)
            metadata = complete_metadata

        idx_id = db.create_index(name, description, metadata)
        return {"index_id": idx_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/indexes/{index_id}")
async def get_index(index_id: str):
    """Get a specific index"""
    try:
        data = db.get_index(index_id)
        if not data:
            raise HTTPException(status_code=404, detail="Index not found")
        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/indexes/{index_id}/fusion-weights")
async def update_fusion_weights(index_id: str, request: Request):
    """Store per-index hybrid fusion weights in the index metadata.

    Body: {"bm25_weight": 0.4, "vec_weight": 0.6}
    """
    try:
        body = await request.json()
        bm25_weight = float(body.get("bm25_weight", 0.5))
        vec_weight = float(body.get("vec_weight", 0.5))
        if abs(bm25_weight + vec_weight - 1.0) > 0.01:
            raise HTTPException(
                status_code=400, detail="bm25_weight + vec_weight must sum to 1.0"
            )
        index = db.get_index(index_id)
        if not index:
            raise HTTPException(status_code=404, detail="Index not found")
        meta = index.get("metadata") or {}
        meta["fusion_config"] = {
            "method": "linear",
            "bm25_weight": bm25_weight,
            "vec_weight": vec_weight,
        }
        db.update_index_metadata(index_id, meta)
        return {"index_id": index_id, "fusion_config": meta["fusion_config"]}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/indexes/{index_id}/diagnostics")
async def get_index_diagnostics(index_id: str):
    """Validate source files, vector artifacts, and latest build state for an index."""
    try:
        return _index_diagnostics(index_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/indexes/{index_id}")
async def delete_index(index_id: str):
    """Remove an index, its documents, links, and the underlying LanceDB table."""
    try:
        index = db.get_index(index_id)
        if not index:
            raise HTTPException(status_code=404, detail="Index not found")
        removed = _delete_index_artifacts(index)
        deleted = db.delete_index(index_id)
        if not deleted:
            raise HTTPException(
                status_code=409,
                detail="Index artifacts were removed, but the database record could not be deleted",
            )
        return {
            "message": "Index deleted successfully",
            "index_id": index_id,
            "removed": removed,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/indexes/{index_id}/upload")
async def index_file_upload(
    index_id: str,
    files: List[UploadFile] = File(...),
    metadata: Optional[str] = Form(None),
):
    """Upload files to an index.

    `metadata` (optional, JSON): either one object applied to every file in
    this upload, or {filename: object}. Validated strictly against the
    index's metadata schema — unknown fields or type mismatches are a 400.
    """
    index = db.get_index(index_id)
    if not index:
        raise HTTPException(status_code=404, detail="Index not found")
    schema = (index.get("metadata") or {}).get("metadata_schema")

    filenames = {f.filename for f in files if f.filename}
    meta_map: Dict[str, Any] = {}
    if metadata:
        try:
            parsed = json.loads(metadata)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="metadata must be valid JSON")
        if not isinstance(parsed, dict):
            raise HTTPException(
                status_code=400, detail="metadata must be a JSON object"
            )
        if not schema:
            raise HTTPException(
                status_code=400,
                detail="This index has no metadata schema — set one via PUT /indexes/{id}/metadata-schema first",
            )
        per_file = (
            bool(parsed)
            and all(isinstance(v, dict) for v in parsed.values())
            and set(parsed) <= filenames
        )
        try:
            if per_file:
                meta_map = {
                    fn: validate_document_metadata(schema, parsed.get(fn, {}))
                    for fn in filenames
                }
            else:
                shared = validate_document_metadata(schema, parsed)
                meta_map = dict.fromkeys(filenames, shared)
        except FilterError as e:
            raise HTTPException(status_code=400, detail=str(e))
    elif schema:
        try:
            empty = validate_document_metadata(schema, {})
            meta_map = dict.fromkeys(filenames, empty)
        except FilterError as e:
            raise HTTPException(status_code=400, detail=str(e))

    uploaded_files = await _save_uploads(files)

    # Register in the DB only after every file is safely on disk
    for item in uploaded_files:
        db.add_document_to_index(
            index_id,
            item["filename"],
            item["stored_path"],
            custom_metadata=meta_map.get(item["filename"]),
        )

    return {
        "message": f"Uploaded {len(uploaded_files)} files",
        "uploaded_files": uploaded_files,
    }


@app.put("/indexes/{index_id}/metadata-schema")
async def set_index_metadata_schema(index_id: str, request: Request):
    """Define the typed metadata schema for an index (before its first build)."""
    index = db.get_index(index_id)
    if not index:
        raise HTTPException(status_code=404, detail="Index not found")
    data = await request.json()
    schema = data.get("schema")
    errors = validate_schema(schema)
    if errors:
        raise HTTPException(status_code=400, detail="; ".join(errors))
    current_schema = (index.get("metadata") or {}).get("metadata_schema")
    if current_schema != schema:
        status = (index.get("metadata") or {}).get("status", "created")
        if index.get("documents") or status != "created":
            raise HTTPException(
                status_code=409,
                detail="Metadata schema cannot be changed after documents are uploaded or a build has started. Create a new index or remove its documents first.",
            )
    db.update_index_metadata(index_id, {"metadata_schema": schema})
    return {"message": "Metadata schema saved", "schema": schema}


@app.post("/indexes/{index_id}/build/preflight")
async def index_build_preflight(index_id: str, request: Request):
    try:
        body = await request.body()
        data = json.loads(body.decode("utf-8")) if body else {}
        check_services = bool(data.pop("checkServices", True))
        return _index_build_preflight(index_id, data, check_services=check_services)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _run_index_build(
    index_id: str, data: Dict[str, Any], job_id: str | None = None
) -> Dict[str, Any]:
    data = dict(
        data
    )  # shallow copy — prevents pop() from mutating the caller's job['options'] dict
    index = db.get_index(index_id)
    if not index:
        raise HTTPException(status_code=404, detail="Index not found")

    file_paths = [d["stored_path"] for d in index.get("documents", [])]
    if not file_paths:
        raise HTTPException(status_code=400, detail="No documents to index")

    preflight = _index_build_preflight(index_id, data, check_services=True)
    if not preflight["ok"]:
        detail = "; ".join(preflight["errors"])
        raise HTTPException(
            status_code=503 if preflight.get("rag_api_available") is False else 400,
            detail=detail,
        )

    latechunk = bool(data.get("latechunk", False))
    docling_chunk = bool(data.get("doclingChunk", False))
    chunk_size = int(data.get("chunkSize", 512))
    chunk_overlap = int(data.get("chunkOverlap", 64))
    retrieval_mode = str(data.get("retrievalMode", "hybrid"))
    window_size = int(data.get("windowSize", 2))
    enable_enrich = bool(data.get("enableEnrich", False))
    embedding_model = data.get("embeddingModel")
    enrich_model = data.get("enrichModel")
    enrich_provider = data.get("enrichProvider", "ollama")
    enrich_api_key = data.pop(
        "enrichApiKey", None
    )  # extracted and removed so it is never written to the DB
    batch_size_embed = int(data.get("batchSizeEmbed", 50))
    batch_size_enrich = int(data.get("batchSizeEnrich", 25))
    overview_model = data.get("overviewModel")
    force_reindex = bool(data.get("forceReindex", False))
    indexing_model_warnings = []

    # Guard only applies to local Ollama models; cloud providers manage their own quotas
    if enrich_provider == "ollama" and _large_indexing_model(enrich_model):
        indexing_model_warnings.append(
            f"Replaced enrichment model '{enrich_model}' with qwen3:8b for indexing safety."
        )
        enrich_model = "qwen3:8b"
    if _large_indexing_model(overview_model):
        indexing_model_warnings.append(
            f"Replaced overview model '{overview_model}' with qwen3:8b for indexing safety."
        )
        overview_model = "qwen3:8b"

    window_size = max(0, min(window_size, 2))
    batch_size_enrich = max(1, min(batch_size_enrich, 8))

    # Use the index's dedicated LanceDB table so retrieval matches.
    table_name = index.get("vector_table_name")
    _index_meta = index.get("metadata") or {}
    payload = {
        "index_id": index_id,
        "file_paths": file_paths,
        "session_id": index_id,  # reuse index_id for progress tracking
        "table_name": table_name,
        "metadata_schema": _index_meta.get("metadata_schema"),
        "file_metadata": {
            d["stored_path"]: d.get("custom_metadata")
            for d in index.get("documents", [])
            if d.get("custom_metadata")
        },
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "retrieval_mode": retrieval_mode,
        "window_size": window_size,
        "enable_enrich": enable_enrich,
        "batch_size_embed": batch_size_embed,
        "batch_size_enrich": batch_size_enrich,
        "force_reindex": force_reindex,
    }
    if latechunk:
        payload["enable_latechunk"] = True
    if docling_chunk:
        payload["enable_docling_chunk"] = True
    if embedding_model:
        payload["embedding_model"] = embedding_model
    if enrich_model:
        payload["enrich_model"] = enrich_model
    if enrich_provider and enrich_provider != "ollama":
        payload["enrich_provider"] = enrich_provider
        if enrich_api_key:
            payload["enrich_api_key"] = enrich_api_key
    if overview_model:
        payload["overview_model_name"] = overview_model
    if job_id:
        payload["job_id"] = job_id
        payload["backend_base_url"] = BACKEND_BASE_URL

    meta_updates = {
        "status": "building",
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "retrieval_mode": retrieval_mode,
        "window_size": window_size,
        "enable_enrich": enable_enrich,
        "latechunk": latechunk,
        "docling_chunk": docling_chunk,
        "force_reindex": force_reindex,
        "batch_size_embed": batch_size_embed,
        "batch_size_enrich": batch_size_enrich,
        "build_started_at": datetime.now().isoformat(),
    }
    if job_id:
        meta_updates["build_job_id"] = job_id
    if embedding_model:
        meta_updates["embedding_model"] = embedding_model
    if enrich_model:
        meta_updates["enrich_model"] = enrich_model
    if overview_model:
        meta_updates["overview_model"] = overview_model
    if indexing_model_warnings:
        meta_updates["indexing_model_warnings"] = indexing_model_warnings
    db.update_index_metadata(index_id, meta_updates)

    if job_id:
        _update_index_job(
            job_id,
            stage="indexing",
            progress=20,
            message="RAG pipeline is indexing documents",
        )

    try:
        if force_reindex:
            _clear_index_build_artifacts(index_id, table_name)
        config = build_indexing_config(PIPELINE_CONFIGS["default"], payload)
        indexing_result = execute_index_build(
            config,
            OLLAMA_CONFIG,
            file_paths,
            index_id=index_id,
            force_reindex=force_reindex,
            job_id=job_id,
            backend_base_url=BACKEND_BASE_URL,
        )
        final_updates = {
            **meta_updates,
            "status": "functional",
            "rebuilt_at": datetime.now().isoformat(),
        }
        db.update_index_metadata(index_id, final_updates)
        return {
            "message": f"Indexing process for {len(file_paths)} file(s) completed successfully.",
            "table_name": table_name,
            "indexing_result": indexing_result,
            "indexing_model_warnings": indexing_model_warnings,
            **final_updates,
        }
    except RuntimeError as e:
        if str(e) == "indexing_cancelled":
            db.update_index_metadata(
                index_id,
                {
                    "status": "cancelled",
                    "build_cancelled_at": datetime.now().isoformat(),
                },
            )
            raise
        db.update_index_metadata(
            index_id,
            {
                "status": "failed",
                "build_failed_at": datetime.now().isoformat(),
                "build_error": str(e),
            },
        )
        raise
    except Exception as e:
        db.update_index_metadata(
            index_id,
            {
                "status": "failed",
                "build_failed_at": datetime.now().isoformat(),
                "build_error": str(e),
            },
        )
        raise HTTPException(status_code=500, detail=f"RAG indexing failed: {e}") from e


def _run_index_build_job(job_id: str, options_override: Dict[str, Any] | None = None):
    """Run a queued build job.

    options_override carries secrets (the cloud enrichment API key) that are
    deliberately NOT persisted with the job — the DB copy is scrubbed. Resumed
    jobs read from the DB and therefore run without the key.
    """
    job = _get_index_job(job_id)
    if not job:
        return
    if options_override is not None:
        job["options"] = options_override
    if job.get("cancel_requested"):
        _update_index_job(
            job_id,
            status="cancelled",
            stage="cancelled",
            progress=100,
            message="Build cancelled before it started",
        )
        db.update_index_metadata(
            job["index_id"],
            {"status": "cancelled", "build_cancelled_at": datetime.now().isoformat()},
        )
        return

    _update_index_job(
        job_id,
        status="running",
        stage="validating",
        progress=5,
        message="Preparing index build",
    )
    try:
        result = _run_index_build(job["index_id"], job["options"], job_id=job_id)
        latest = _get_index_job(job_id) or {}
        status = "cancelled" if latest.get("cancel_requested") else "completed"
        message = (
            "Build completed after cancellation request"
            if status == "cancelled"
            else "Build completed"
        )
        if status == "cancelled":
            db.update_index_metadata(
                job["index_id"],
                {
                    "status": "cancelled",
                    "build_cancelled_at": datetime.now().isoformat(),
                },
            )
        _update_index_job(
            job_id,
            status=status,
            stage=status,
            progress=100,
            message=message,
            result=result,
            finished_at=datetime.now().isoformat(),
        )
    except Exception as e:
        if isinstance(e, RuntimeError) and str(e) == "indexing_cancelled":
            db.update_index_metadata(
                job["index_id"],
                {
                    "status": "cancelled",
                    "build_cancelled_at": datetime.now().isoformat(),
                },
            )
            _update_index_job(
                job_id,
                status="cancelled",
                stage="cancelled",
                progress=100,
                message="Indexing cancelled",
                finished_at=datetime.now().isoformat(),
            )
            return
        # Everything else marks the job failed. Re-raising here would vanish
        # into the daemon thread and leave the job "running" forever — e.g.
        # the RuntimeError raised when the build child process crashes.
        db.update_index_metadata(
            job["index_id"],
            {
                "status": "failed",
                "build_failed_at": datetime.now().isoformat(),
                "build_error": str(e),
            },
        )
        _update_index_job(
            job_id,
            status="failed",
            stage="failed",
            progress=100,
            message=str(e),
            error=str(e),
            finished_at=datetime.now().isoformat(),
        )


@app.post("/indexes/{index_id}/build")
async def build_index(index_id: str, request: Request):
    """Build an index from uploaded documents"""
    try:
        body = await request.body()
        data = json.loads(body.decode("utf-8")) if body else {}

        # Validate option types/ranges up front: a bad chunkSize should be a
        # 400, not an unhandled ValueError deep inside the build
        try:
            IndexBuildRequest(
                **{k: v for k, v in data.items() if k in IndexBuildRequest.model_fields}
            )
        except ValidationError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid build options — {_validation_error_detail(e)}",
            )

        # Reject concurrent builds: two builds of the same index write to the
        # same LanceDB table and clobber each other's metadata updates
        now_dt = datetime.now()
        for job in db.list_unfinished_index_jobs():
            if job.get("index_id") != index_id:
                continue
            updated = job.get("updated_at") or job.get("created_at")
            try:
                is_stale = (
                    bool(updated)
                    and (now_dt - datetime.fromisoformat(cast(str, updated)))
                    > STALE_BUILD_AFTER
                )
            except ValueError:
                is_stale = False
            if not is_stale:
                raise HTTPException(
                    status_code=409,
                    detail=f"A build for this index is already in progress (job {job['id']}). Cancel it or wait for it to finish.",
                )

        if bool(data.get("background", False)):
            index = db.get_index(index_id)
            if not index:
                raise HTTPException(status_code=404, detail="Index not found")
            preflight = _index_build_preflight(index_id, data, check_services=True)
            if not preflight["ok"]:
                detail = "; ".join(preflight["errors"])
                raise HTTPException(
                    status_code=(
                        503 if preflight.get("rag_api_available") is False else 400
                    ),
                    detail=detail,
                )
            job_id = str(uuid.uuid4())
            now = datetime.now().isoformat()
            # The cloud-enrichment API key is needed at runtime but must never
            # be persisted: the worker thread gets the full options in memory,
            # while the DB and the job map only ever see a scrubbed copy.
            runtime_options = dict(data)
            persisted_options = {k: v for k, v in data.items() if k != "enrichApiKey"}
            db_job = db.create_index_job(
                job_id, index_id, persisted_options, index.get("documents", [])
            )
            with index_jobs_lock:
                index_jobs[job_id] = {
                    "id": job_id,
                    "index_id": index_id,
                    "status": "queued",
                    "stage": "queued",
                    "progress": 0,
                    "message": "Build queued",
                    "cancel_requested": False,
                    "options": persisted_options,
                    "created_at": now,
                    "updated_at": now,
                    "files": db_job.get("files", []) if db_job else [],
                }
            db.update_index_metadata(
                index_id,
                {
                    "status": "building",
                    "build_job_id": job_id,
                    "build_started_at": now,
                },
            )
            thread = threading.Thread(
                target=_run_index_build_job,
                args=(job_id,),
                kwargs={"options_override": runtime_options},
                daemon=True,
            )
            thread.start()
            return {
                "message": "Index build started",
                "job_id": job_id,
                "status": "queued",
            }

        # Foreground builds block for the whole build; run in a worker thread
        # so the event loop keeps serving other requests meanwhile.
        return await asyncio.to_thread(_run_index_build, index_id, data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/index-jobs/{job_id}")
async def get_index_job(job_id: str):
    job = _public_index_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Index job not found")
    return job


@app.get("/index-jobs/{job_id}/stream")
async def stream_index_job(job_id: str):
    """SSE stream for live indexing progress. Emits events until the job finishes."""
    import asyncio as _asyncio

    async def _event_gen():
        terminal_statuses = {"completed", "failed", "cancelled"}
        last_progress = -1
        max_ticks = 7200  # 1-hour hard cap at 0.5 s/tick
        ticks = 0
        while ticks < max_ticks:
            job = _public_index_job(job_id)
            if job is None:
                yield f"data: {json.dumps({'type': 'error', 'data': {'message': 'Job not found'}})}\n\n"
                break
            progress = job.get("progress", 0)
            status = job.get("status", "unknown")
            if progress != last_progress or status in terminal_statuses:
                last_progress = progress
                payload = {
                    "type": "progress",
                    "data": {
                        "id": job.get("id", job_id),
                        "index_id": job.get("index_id", ""),
                        "status": status,
                        "stage": job.get("stage", ""),
                        "progress": progress,
                        "message": job.get("message", ""),
                        "files": job.get("files", []),
                    },
                }
                yield f"data: {json.dumps(payload)}\n\n"
            if status in terminal_statuses:
                break
            ticks += 1
            await _asyncio.sleep(0.5)

    return StreamingResponse(
        _event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/index-jobs/{job_id}/cancel")
async def cancel_index_job(job_id: str):
    job = _get_index_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Index job not found")
    if job["status"] in ("completed", "failed", "cancelled"):
        public_job = _public_index_job(job_id) or job
        return {**public_job, "message": "Job is already finished"}
    _update_index_job(job_id, cancel_requested=True, message="Cancellation requested")
    if job["status"] == "queued":
        _update_index_job(
            job_id,
            status="cancelled",
            stage="cancelled",
            progress=100,
            finished_at=datetime.now().isoformat(),
        )
        db.update_index_metadata(
            job["index_id"],
            {"status": "cancelled", "build_cancelled_at": datetime.now().isoformat()},
        )
    return _public_index_job(job_id)


# Helper functions (moved from ChatHandler)


def _route_using_overviews(query: str, overviews: List[str]) -> bool:
    """
    🎯 Use document overviews and LLM to make intelligent routing decisions.

    Returns True if RAG should be used, False for direct LLM.
    """
    if not overviews:
        return False

    # Format overviews for the routing prompt
    overviews_block = "\n".join(f"[{i+1}] {ov}" for i, ov in enumerate(overviews))

    router_prompt = f"""You are an AI router deciding whether a user question should be answered via:
• "USE_RAG" – search the user's private documents (described below)
• "DIRECT_LLM" – reply from general knowledge (greetings, public facts, unrelated topics)

CRITICAL PRINCIPLE: When documents exist in the KB, strongly prefer USE_RAG unless the query is purely conversational or completely unrelated to any possible document content.

RULES:
1. If ANY overview clearly relates to the question (entities, numbers, addresses, dates, amounts, companies, technical terms) → USE_RAG
2. For document operations (summarize, analyze, explain, extract, find) → USE_RAG
3. For greetings only ("Hi", "Hello", "Thanks") → DIRECT_LLM
4. For pure math/world knowledge clearly unrelated to documents → DIRECT_LLM
5. When in doubt → USE_RAG

DOCUMENT OVERVIEWS:
{overviews_block}

DECISION EXAMPLES:
• "What invoice amounts are mentioned?" → USE_RAG (document-specific)
• "Who is PromptX AI LLC?" → USE_RAG (entity in documents)
• "What is the DeepSeek model?" → USE_RAG (mentioned in documents)
• "Summarize the research paper" → USE_RAG (document operation)
• "What is 2+2?" → DIRECT_LLM (pure math)
• "Hi there" → DIRECT_LLM (greeting only)

USER QUERY: "{query}"

Respond with exactly one word: USE_RAG or DIRECT_LLM"""

    try:
        # Use Ollama to make the routing decision
        response = ollama_client.chat(
            message=router_prompt,
            model="qwen3:8b",  # Quality local model for routing
            enable_thinking=False,  # Fast routing
        )

        # The response is directly the text, not a dict
        decision = response.strip().upper()

        # Parse decision
        if "USE_RAG" in decision:
            print(f"🎯 Overview-based routing: USE_RAG for query: '{query[:50]}...'")
            return True
        elif "DIRECT_LLM" in decision:
            print(f"⚡ Overview-based routing: DIRECT_LLM for query: '{query[:50]}...'")
            return False
        else:
            print(f"⚠️ Unclear routing decision '{decision}', defaulting to RAG")
            return True  # Default to RAG when uncertain

    except Exception as e:
        print(f"❌ LLM routing failed: {e}, falling back to pattern matching")
        return _simple_pattern_routing(query, [])


def _simple_pattern_routing(message: str, idx_ids: List[str]) -> bool:
    """
    📝 FALLBACK: Simple pattern-based routing (original logic).
    """
    message_lower = message.lower()

    # Always use Direct LLM for greetings and casual conversation
    greeting_patterns = [
        "hello",
        "hi",
        "hey",
        "greetings",
        "good morning",
        "good afternoon",
        "good evening",
        "how are you",
        "how do you do",
        "pleasure to meet",
        "thanks",
        "thank you",
        "bye",
        "goodbye",
        "see you",
        "talk to you later",
        "test",
        "testing",
        "check",
        "ping",
        "just saying",
        "nevermind",
        "ok",
        "okay",
        "alright",
        "got it",
        "understood",
        "i see",
    ]

    # Check for greeting patterns
    for pattern in greeting_patterns:
        if pattern in message_lower:
            return False  # Use Direct LLM for greetings

    # Keywords that strongly suggest document-related queries
    rag_indicators = [
        "document",
        "doc",
        "file",
        "pdf",
        "text",
        "content",
        "page",
        "according to",
        "based on",
        "mentioned",
        "states",
        "says",
        "what does",
        "summarize",
        "summary",
        "analyze",
        "analysis",
        "quote",
        "citation",
        "reference",
        "source",
        "evidence",
        "explain from",
        "extract",
        "find in",
        "search for",
    ]

    # Check for strong RAG indicators
    for indicator in rag_indicators:
        if indicator in message_lower:
            return True

    # Question words + substantial length might benefit from RAG
    question_words = ["what", "how", "when", "where", "why", "who", "which"]
    starts_with_question = any(
        message_lower.startswith(word) for word in question_words
    )

    if starts_with_question and len(message) > 40:
        return True

    # Very short messages - use direct LLM
    if len(message.strip()) < 20:
        return False

    # Default to Direct LLM unless there's clear indication of document query
    return False


async def _handle_direct_llm_query(
    session_id: str, message: str, session: dict, requested_model: str | None = None
):
    """
    Handle query using direct Ollama client with thinking disabled for speed.

    Returns:
        tuple: (response_text, empty_source_docs)
    """
    try:
        # Get conversation history for context
        conversation_history = db.get_conversation_history(session_id)

        # The model selected in the UI for this message wins; the session's
        # stored model is the fallback. RAG answers already work this way —
        # direct answers ignoring the dropdown was an inconsistency.
        model = requested_model or session.get("model") or "qwen3:8b"

        # Direct Ollama call with thinking disabled for speed.
        # Runs in a worker thread so the blocking HTTP call (up to 60s)
        # doesn't freeze the event loop.
        response_text = await asyncio.to_thread(
            ollama_client.chat,
            message=message,
            model=model,
            conversation_history=conversation_history,
            enable_thinking=False,  # ⚡ DISABLE THINKING FOR SPEED
        )

        return response_text, []  # No source docs for direct LLM

    except Exception as e:
        print(f"❌ Direct LLM error: {e}")
        raise HTTPException(status_code=503, detail=f"Error processing query: {str(e)}")


async def _handle_rag_query(
    session_id: str, message: str, data: dict, idx_ids: List[str]
):
    """
    Handle query using the in-process transport-neutral RAG runtime.

    Returns:
        tuple[str, List[dict]]: (response_text, source_documents)
    """
    try:
        payload = dict(data)
        payload.update(
            {
                "query": message,
                "session_id": session_id,
            }
        )
        rag_data = await asyncio.to_thread(
            execute_rag_chat,
            _get_local_rag_agent(),
            db,
            payload,
        )
        response_text = rag_data.get("answer", "No answer found.")
        source_docs = rag_data.get("source_documents", [])
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ RAG processing error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error processing RAG query: {str(e)}"
        )

    # Strip any <think>/<thinking> tags that might slip through
    response_text = re.sub(
        r"<(think|thinking)>.*?</\1>",
        "",
        response_text,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()

    return response_text, source_docs


def main():
    """Main function to initialize and start the server"""
    # The embedded in-process RAG agent (and the /rag/chat routes) resolve
    # storage paths — lancedb, index_store, overviews — relative to the CWD,
    # exactly as the standalone RAG API does from the repo root. The backend
    # is commonly launched from backend/, which would point those at
    # backend/lancedb (empty) → "could not find an answer". Anchor the CWD to
    # the repo root here, at startup ONLY — never at import, which would
    # corrupt the CWD of test processes that import this module.
    os.chdir(PROJECT_ROOT)

    PORT = 8000  # 🆕 Define port
    try:
        # Initialize the database
        print("✅ Database initialized successfully")

        # Initialize the PDF processor
        try:
            pdf_module.initialize_simple_pdf_processor()
            print("📄 Initializing simple PDF processing...")
            if pdf_module.simple_pdf_processor:
                print("✅ Simple PDF processor initialized")
            else:
                print("⚠️ PDF processing could not be initialized.")
        except Exception as e:
            print(f"❌ Error initializing PDF processor: {e}")
            print(
                "⚠️ PDF processing disabled - server will run without RAG functionality"
            )

        # Set a global reference to the initialized processor if needed elsewhere
        global pdf_processor
        pdf_processor = pdf_module.simple_pdf_processor
        if pdf_processor:
            print("✅ Global PDF processor initialized")
        else:
            print(
                "⚠️ PDF processing disabled - server will run without RAG functionality"
            )

        # Cleanup empty sessions on startup
        print("🧹 Cleaning up empty sessions...")
        cleanup_count = db.cleanup_empty_sessions()
        if cleanup_count > 0:
            print(f"✨ Cleaned up {cleanup_count} empty sessions")
        else:
            print("✨ No empty sessions to clean up")

        # Start the server with uvicorn
        print(f"🚀 Starting localGPT backend server on port {PORT}")
        print(f"📍 Chat endpoint: http://localhost:{PORT}/chat")
        print(f"🔍 Health check: http://localhost:{PORT}/health")

        # Test Ollama connection
        if ollama_client.is_ollama_running():
            models = ollama_client.list_models()
            print(f"✅ Ollama is running with {len(models)} models")
            print(
                f"📋 Available models: {', '.join(models[:3])}{'...' if len(models) > 3 else ''}"
            )
        else:
            print("⚠️  Ollama is not running. Please start Ollama:")
            print("   Install: https://ollama.ai")
            print("   Run: ollama serve")

        print(f"\n🌐 Frontend should connect to: http://localhost:{PORT}")
        print("💬 Ready to chat!\n")

        # Loopback by default; set BIND_HOST=0.0.0.0 (e.g. in Docker) to
        # expose the API beyond this machine deliberately.
        uvicorn.run(app, host=os.getenv("BIND_HOST", "127.0.0.1"), port=PORT)

    except KeyboardInterrupt:
        print("\n🛑 Server stopped")


# ============================================================================
# MAINTENANCE ENDPOINTS
# ============================================================================


@app.post("/maintenance/repair-stuck-builds")
async def repair_stuck_builds(older_than_minutes: int = 30):
    """Repair stuck/stale build jobs"""
    if not maintenance_tools:
        raise HTTPException(status_code=503, detail="Maintenance tools not available")

    try:
        result = maintenance_tools.repair_stuck_builds(older_than_minutes)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error repairing stuck builds: {str(e)}"
        )


@app.post("/maintenance/remove-orphan-files")
async def remove_orphan_files(dry_run: bool = True):
    """Find and remove uploaded files not associated with any index"""
    if not maintenance_tools:
        raise HTTPException(status_code=503, detail="Maintenance tools not available")

    try:
        result = maintenance_tools.remove_orphan_files(dry_run)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error removing orphan files: {str(e)}"
        )


@app.post("/maintenance/delete-broken-indexes")
async def delete_broken_indexes(dry_run: bool = True, health_status: str = "unhealthy"):
    """Find and delete broken/unhealthy indexes"""
    if not maintenance_tools:
        raise HTTPException(status_code=503, detail="Maintenance tools not available")

    try:
        result = maintenance_tools.delete_broken_indexes(dry_run, health_status)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error deleting broken indexes: {str(e)}"
        )


@app.get("/maintenance/failed-files/{index_id}")
async def get_failed_files(index_id: str):
    """Get list of files that failed in the latest build job"""
    if not maintenance_tools:
        raise HTTPException(status_code=503, detail="Maintenance tools not available")

    try:
        result = maintenance_tools.get_failed_files_for_index(index_id)
        if result.get("error"):
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting failed files: {str(e)}"
        )


@app.post("/maintenance/rebuild-failed-files/{index_id}")
async def rebuild_failed_files(index_id: str, force: bool = False):
    """Mark failed files to be rebuilt on next indexing job"""
    if not maintenance_tools:
        raise HTTPException(status_code=503, detail="Maintenance tools not available")

    try:
        result = maintenance_tools.rebuild_failed_files_only(index_id, force)
        if result.get("error"):
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error rebuilding failed files: {str(e)}"
        )


@app.get("/maintenance/index-health")
async def get_index_health(index_id: Optional[str] = None):
    """Get detailed health report for one or all indexes"""
    if not maintenance_tools:
        raise HTTPException(status_code=503, detail="Maintenance tools not available")

    try:
        result = maintenance_tools.get_index_health_report(index_id)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting index health: {str(e)}"
        )


@app.post("/maintenance/export-diagnostics")
async def export_diagnostics(
    output_path: Optional[str] = None,
    include_logs: bool = True,
    include_config: bool = True,
):
    """Export complete diagnostics bundle (logs, configs, state)"""
    if not maintenance_tools:
        raise HTTPException(status_code=503, detail="Maintenance tools not available")

    try:
        result = maintenance_tools.export_diagnostics_bundle(
            output_path, include_logs, include_config
        )
        if result.get("errors"):
            return {"warning": "Bundle created with errors", **result}
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error exporting diagnostics: {str(e)}"
        )


@app.post("/maintenance/vacuum-database")
async def vacuum_database():
    """Run SQLite VACUUM to reclaim fragmented pages."""
    if not maintenance_tools:
        raise HTTPException(status_code=503, detail="Maintenance tools not available")
    try:
        result = maintenance_tools.vacuum_database()
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vacuum failed: {str(e)}")


@app.post("/maintenance/remove-orphan-tables")
async def remove_orphan_tables(dry_run: bool = True):
    """Drop LanceDB tables with no matching index record (dry_run=true by default)."""
    if not maintenance_tools:
        raise HTTPException(status_code=503, detail="Maintenance tools not available")
    try:
        result = maintenance_tools.remove_orphan_lancedb_tables(dry_run=dry_run)
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Orphan sweep failed: {str(e)}")


# ============================================================================
# JOB PERSISTENCE & RESUMABLE INDEXING ENDPOINTS
# ============================================================================


@app.post("/index-jobs/{job_id}/resume")
async def resume_index_job(job_id: str):
    """Resume an indexing job that was paused or crashed"""
    if not job_progress_tracker:
        raise HTTPException(status_code=503, detail="Job persistence not available")

    try:
        job = _get_index_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Index job not found")
        if job.get("status") in {"queued", "running"}:
            return {
                "job_id": job_id,
                "status": job.get("status"),
                "message": "Job is already queued or running",
            }

        result = job_progress_tracker.mark_job_resuming(job_id)
        if result.get("error"):
            raise HTTPException(status_code=404, detail=result["error"])
        with index_jobs_lock:
            index_jobs[job_id] = {
                **job,
                "status": "queued",
                "stage": "queued",
                "progress": max(int(job.get("progress") or 0), 0),
                "message": "Resume queued",
                "cancel_requested": False,
            }
        thread = threading.Thread(
            target=_run_index_build_job, args=(job_id,), daemon=True
        )
        thread.start()
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resuming job: {str(e)}")


@app.get("/index-jobs/{job_id}/timeline")
async def get_job_timeline(job_id: str):
    """Get complete timeline of events for a job"""
    if not job_progress_tracker:
        raise HTTPException(status_code=503, detail="Job persistence not available")

    try:
        result = job_progress_tracker.get_job_timeline(job_id)
        if result.get("error"):
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting timeline: {str(e)}")


@app.get("/index-jobs/{job_id}/file-status")
async def get_job_file_status(job_id: str):
    """Get detailed per-file status for a job"""
    if not job_progress_tracker:
        raise HTTPException(status_code=503, detail="Job persistence not available")

    try:
        result = job_progress_tracker.get_job_timeline(job_id)
        if result.get("error"):
            raise HTTPException(status_code=404, detail=result["error"])

        # Return file status with per-stage breakdown
        return {
            "job_id": job_id,
            "files": result.get("files", []),
            "summary": {
                "total_files": result.get("total_files", 0),
                "completed_files": result.get("completed_files", 0),
                "failed_files": result.get("failed_files", 0),
                "pending_files": result.get("pending_files", 0),
            },
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting file status: {str(e)}"
        )


@app.get("/index-jobs/{job_id}/statistics")
async def get_job_statistics(job_id: str):
    """Get performance statistics for a job"""
    if not job_progress_tracker:
        raise HTTPException(status_code=503, detail="Job persistence not available")

    try:
        result = job_progress_tracker.get_job_statistics(job_id)
        if result.get("error"):
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting statistics: {str(e)}"
        )


@app.get("/index-jobs/{job_id}/audit-trail")
async def get_job_audit_trail(job_id: str):
    """Get complete audit trail (all stage events) for a job"""
    if not job_progress_tracker:
        raise HTTPException(status_code=503, detail="Job persistence not available")

    try:
        result = job_progress_tracker.export_audit_trail(job_id)
        return {"job_id": job_id, "events": result, "total_events": len(result)}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting audit trail: {str(e)}"
        )


@app.post("/index-jobs/recover-stale")
async def recover_stale_jobs(older_than_minutes: int = 5):
    """Recover jobs that crashed (auto-run on backend startup)"""
    if not job_progress_tracker:
        raise HTTPException(status_code=503, detail="Job persistence not available")

    try:
        result = job_progress_tracker.recover_stale_jobs(older_than_minutes)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error recovering jobs: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """Auto-recover stale jobs and index metadata on backend startup"""
    if job_progress_tracker:
        try:
            result = job_progress_tracker.recover_stale_jobs(older_than_minutes=5)
            if result.get("recovered", 0) > 0:
                print(f"✅ Auto-recovered {result['recovered']} stale indexing job(s)")
        except Exception as e:
            print(f"⚠️ Error during stale job recovery: {e}")
    try:
        recovered = _recover_stale_index_builds()
        if recovered > 0:
            print(f"✅ Reset {recovered} index(es) stuck in 'building' state")
    except Exception as e:
        print(f"⚠️ Error during stale index build recovery: {e}")


if __name__ == "__main__":
    main()

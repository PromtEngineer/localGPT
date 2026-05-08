import json
import os
import uuid
from datetime import datetime, timedelta
import re
import threading
import time
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import requests

# Add parent directory to path so we can import rag_system modules
import sys
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
for path in (BACKEND_DIR, PROJECT_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

# Import RAG system modules for complete metadata
try:
    from rag_system.main import PIPELINE_CONFIGS
    RAG_SYSTEM_AVAILABLE = True
    print("✅ RAG system modules accessible from backend")
except ImportError as e:
    PIPELINE_CONFIGS = {}
    RAG_SYSTEM_AVAILABLE = False
    print(f"⚠️ RAG system modules not available: {e}")

from ollama_client import OllamaClient
from database import db, generate_session_title
import simple_pdf_processor as pdf_module
from simple_pdf_processor import initialize_simple_pdf_processor

# Initialize FastAPI app
app = FastAPI(title="LocalGPT Backend", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
ollama_client = OllamaClient()
pdf_processor = None
index_jobs_lock = threading.Lock()
index_jobs: Dict[str, Dict[str, Any]] = {}
STALE_BUILD_AFTER = timedelta(minutes=10)
RAG_API_BASE_URL = "http://localhost:8001"


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
    return any(token in lowered for token in ("gpt-oss", "120b", "70b", "large", "cloud"))


def _index_build_preflight(index_id: str, data: Dict[str, Any] | None = None, *, check_services: bool = True) -> Dict[str, Any]:
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

    if not RAG_SYSTEM_AVAILABLE:
        errors.append(
            "RAG indexing dependencies are not available in this backend Python process. "
            "Restart with './start-localgpt' or activate .venv before starting the backend."
        )

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
        errors.append(f"{len(missing_files)} uploaded file(s) are missing from disk: {sample}")
    if unreadable_files:
        sample = ", ".join(item["filename"] for item in unreadable_files[:5])
        errors.append(f"{len(unreadable_files)} uploaded file(s) are not readable: {sample}")

    if document_count > 100:
        warnings.append(f"This build has {document_count} files. Prefer Fast mode or smaller batches for best stability.")
    if total_bytes > 500 * 1024 * 1024:
        warnings.append(f"This build is {_format_bytes(total_bytes)}. Large builds can take a long time on local hardware.")
    if bool(data.get("forceReindex")):
        warnings.append("Force reindex will rebuild all files, including unchanged documents.")
    if bool(data.get("enableEnrich", True)) and document_count > 50:
        warnings.append("Context enrichment on large file sets can be slow. Fast mode is safer for the first pass.")

    for key, label in (("enrichModel", "enrichment"), ("overviewModel", "overview")):
        model = data.get(key)
        if _large_indexing_model(model):
            warnings.append(f"The {label} model '{model}' will be replaced with qwen3:0.6b for indexing safety.")

    rag_api_available = None
    if check_services:
        try:
            response = requests.get(f"{RAG_API_BASE_URL}/models", timeout=3)
            rag_api_available = response.status_code == 200
            if not rag_api_available:
                errors.append(f"RAG API responded with HTTP {response.status_code} at {RAG_API_BASE_URL}.")
        except requests.exceptions.RequestException as e:
            rag_api_available = False
            errors.append(f"RAG API is not reachable at {RAG_API_BASE_URL}: {e}")

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
            conn = lancedb.connect(db_path)
            if hasattr(conn, "list_tables"):
                raw_names = conn.list_tables()
                names = [item.name if hasattr(item, "name") else str(item) for item in raw_names]
            else:
                names = conn.table_names() if hasattr(conn, "table_names") else []
            table_names = list(names)
            if table_name not in table_names:
                continue
            table = conn.open_table(table_name)
            row_count = None
            if hasattr(table, "count_rows"):
                row_count = int(table.count_rows())
            result.update({
                "exists": True,
                "path": db_path,
                "row_count": row_count,
                "latechunk_exists": f"{table_name}_lc" in table_names,
                "error": None,
            })
            return result
        except Exception as e:
            result["error"] = f"Could not inspect LanceDB at {db_path}: {e}"

    if not result["error"]:
        searched = ", ".join(_lancedb_path_candidates())
        result["error"] = f"Vector table was not found in searched LanceDB paths: {searched}"
    return result


def _overview_diagnostics(index_id: str) -> Dict[str, Any]:
    paths = [
        os.path.join(PROJECT_ROOT, "index_store", "overviews", f"{index_id}.jsonl"),
        os.path.join(PROJECT_ROOT, "rag_system", "index_store", "overviews", f"{index_id}.jsonl"),
    ]
    for path in paths:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    line_count = sum(1 for line in handle if line.strip())
            except Exception:
                line_count = None
            return {"exists": True, "path": path, "line_count": line_count}
    return {"exists": False, "path": None, "line_count": 0}


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
        errors.append("Vector table is missing. Rebuild this index before trusting retrieval.")
    elif vector_table["row_count"] == 0:
        errors.append("Vector table exists but has no rows. Force rebuild this index.")

    overview = _overview_diagnostics(index_id)
    if metadata.get("enable_enrich", True) and not overview["exists"]:
        warnings.append("Document overview file is missing. Routing quality may be weaker until the index is rebuilt.")

    latest_job = db.get_latest_index_job(index_id, include_options=False, include_files=True)
    file_status_counts: Dict[str, int] = {}
    if latest_job and latest_job.get("files"):
        for item in latest_job["files"]:
            status = str(item.get("status") or "unknown")
            file_status_counts[status] = file_status_counts.get(status, 0) + 1
        failed_count = file_status_counts.get("failed", 0)
        pending_count = file_status_counts.get("pending", 0)
        if failed_count:
            errors.append(f"{failed_count} file(s) failed in the latest build job.")
        if pending_count and latest_job.get("status") in {"completed", "failed", "cancelled"}:
            warnings.append(f"{pending_count} file(s) were left pending in the latest build job.")

    metadata_status = metadata.get("status")
    if metadata_status in {"failed", "incomplete", "empty"}:
        errors.append(f"Index metadata status is '{metadata_status}'.")
    elif metadata_status in {"building", "cancelled"}:
        warnings.append(f"Index metadata status is '{metadata_status}'.")

    source_blockers = bool(preflight["missing_files"] or preflight["unreadable_files"] or preflight["document_count"] == 0)
    if source_blockers:
        recommended_action = "fix_sources"
    elif errors:
        recommended_action = "force_rebuild"
    elif warnings:
        recommended_action = "rebuild"
    else:
        recommended_action = "none"

    if source_blockers:
        recommendations.append("Re-upload missing or unreadable files before rebuilding.")
    elif errors:
        recommendations.append("Run Force rebuild after confirming the source files still exist.")
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

        started_raw = meta.get("build_started_at")
        try:
            started_at = datetime.fromisoformat(str(started_raw)) if started_raw else None
        except ValueError:
            started_at = None

        if started_at and now - started_at < STALE_BUILD_AFTER:
            continue

        db.update_index_metadata(idx["id"], {
            "status": "failed",
            "build_failed_at": now.isoformat(),
            "build_error": (
                "Previous build was interrupted or the backend restarted before the "
                "background job could finish. Start a rebuild to continue."
            ),
        })
        recovered += 1
    for job in db.list_unfinished_index_jobs():
        with index_jobs_lock:
            if job["id"] in index_jobs:
                continue
        started_raw = job.get("updated_at") or job.get("created_at")
        try:
            started_at = datetime.fromisoformat(str(started_raw)) if started_raw else None
        except ValueError:
            started_at = None
        if started_at and now - started_at < STALE_BUILD_AFTER:
            continue
        db.update_index_job(job["id"], {
            "status": "failed",
            "stage": "failed",
            "progress": 100,
            "message": "Build interrupted by backend restart",
            "error": "Previous build was interrupted or the backend restarted before the background job could finish.",
            "finished_at": now.isoformat(),
        })
    return recovered


@app.post("/index-jobs/{job_id}/progress")
async def update_index_job_progress(job_id: str, request: Request):
    job = _get_index_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Index job not found")
    data = await request.json()
    updates = {
        key: data[key]
        for key in ("stage", "progress", "message")
        if key in data
    }
    if data.get("stage") == "completed":
        updates["status"] = "completed"
        updates["finished_at"] = datetime.now().isoformat()
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
        db.update_index_job_file(job_id, stored_path=file_path, filename=filename, updates=file_updates)
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
        "ollama_running": ollama_client.is_ollama_running(),
        "available_models": ollama_client.list_models(),
        "database_stats": db.get_stats()
    }

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
        return {"message": f"Cleaned up {cleanup_count} empty sessions", "cleanup_count": cleanup_count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cleanup sessions: {str(e)}")

@app.post("/sessions")
async def create_session(request: Request):
    """Create a new chat session"""
    try:
        data = await request.json()
        title = data.get('title', 'New Chat')
        model = data.get('model', 'llama3.2:latest')

        session_id = db.create_session(title, model)
        session = db.get_session(session_id)

        return {"session": session, "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")

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
        filenames = [os.path.basename(p).split('_', 1)[-1] if '_' in os.path.basename(p) else os.path.basename(p) for p in docs]

        return {"session": session, "files": filenames, "file_count": len(docs)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get documents: {str(e)}")

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
                if not idx.get('metadata') or len(idx['metadata']) == 0:
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
        message = data.get('message', '')

        if not message:
            raise HTTPException(status_code=400, detail="Message is required")

        if session['message_count'] == 0:
            title = generate_session_title(message)
            db.update_session_title(session_id, title)

        # Add user message to database first
        user_message_id = db.add_message(session_id, message, "user")

        # 🎯 SMART ROUTING: Decide between direct LLM vs RAG
        idx_ids = db.get_indexes_for_session(session_id)

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
                            break  # Stop after the first existing path for this idx
                    break  # Stop after the first existing path for this idx
                break  # Stop after the first existing path for this idx

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
            print(f"✅ Loaded {len(aggregated)} document overviews from {len(idx_ids)} index(es)")
        else:
            print(f"⚠️ No overviews found for indices {idx_ids}")
        aggregated = aggregated[:40]

        # Decide routing
        use_rag = _route_using_overviews(message, aggregated) if aggregated else _simple_pattern_routing(message, idx_ids)

        if use_rag:
            response_text, source_docs = await _handle_rag_query(session_id, message, data, idx_ids)
        else:
            response_text, source_docs = await _handle_direct_llm_query(session_id, message, session)

        # Add assistant message to database
        assistant_message_id = db.add_message(session_id, response_text, "assistant", metadata={"sources": source_docs})

        return {
            "response": response_text,
            "sources": source_docs,
            "session_id": session_id,
            "message_id": assistant_message_id
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.post("/sessions/{session_id}/upload")
async def upload_files(session_id: str, files: List[UploadFile] = File(...)):
    """Handle file uploads and associate with the session."""
    uploaded_files = []
    upload_dir = "shared_uploads"
    os.makedirs(upload_dir, exist_ok=True)

    for file in files:
        if file.filename:
            # Create a unique filename to avoid overwrites
            unique_filename = f"{uuid.uuid4()}_{file.filename}"
            file_path = os.path.join(upload_dir, unique_filename)

            with open(file_path, 'wb') as f:
                content = await file.read()
                f.write(content)

            # Store the absolute path for the indexing service
            absolute_file_path = os.path.abspath(file_path)
            db.add_document_to_session(session_id, absolute_file_path)
            uploaded_files.append({"filename": file.filename, "stored_path": absolute_file_path})

    if not uploaded_files:
        raise HTTPException(status_code=400, detail="No files were uploaded")

    return {
        "message": f"Successfully uploaded {len(uploaded_files)} files.",
        "uploaded_files": uploaded_files
    }

@app.post("/sessions/{session_id}/index")
async def index_documents(session_id: str):
    """Triggers indexing for all documents in a session."""
    print(f"🔥 Received request to index documents for session {session_id[:8]}...")
    try:
        file_paths = db.get_documents_for_session(session_id)
        if not file_paths:
            return {"message": "No documents to index for this session."}

        print(f"Found {len(file_paths)} documents to index. Sending to RAG API...")

        rag_api_url = "http://localhost:8001/index"
        rag_response = requests.post(rag_api_url, json={"file_paths": file_paths, "session_id": session_id})

        if rag_response.status_code == 200:
            print("✅ RAG API successfully indexed documents.")
            # Merge key config values into index metadata
            idx_meta = {
                "session_linked": True,
                "retrieval_mode": "hybrid",
            }
            try:
                db.update_index_metadata(session_id, idx_meta)  # session_id used as index_id in text table naming
            except Exception as e:
                print(f"⚠️ Failed to update index metadata for session index: {e}")
            return rag_response.json()
        else:
            error_info = rag_response.text
            print(f"❌ RAG API indexing failed ({rag_response.status_code}): {error_info}")
            raise HTTPException(status_code=500, detail=f"Indexing failed: {error_info}")

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Exception during indexing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.post("/sessions/{session_id}/rename")
@app.put("/sessions/{session_id}/rename")
async def rename_session(session_id: str, request: Request):
    """Rename an existing session title"""
    try:
        session = db.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        data = await request.json()
        new_title: str = data.get('title', '').strip()

        if not new_title:
            raise HTTPException(status_code=400, detail="Title cannot be empty")

        db.update_session_title(session_id, new_title)
        updated_session = db.get_session(session_id)

        return {
            "message": "Session renamed successfully",
            "session": updated_session
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to rename session: {str(e)}")

@app.post("/chat")
async def legacy_chat(request: Request):
    """Handle legacy chat requests (without sessions)"""
    try:
        data = await request.json()
        message = data.get('message', '')
        model = data.get('model', 'llama3.2:latest')
        conversation_history = data.get('conversation_history', [])

        if not message:
            raise HTTPException(status_code=400, detail="Message is required")

        # Check if Ollama is running
        if not ollama_client.is_ollama_running():
            raise HTTPException(status_code=503, detail="Ollama is not running. Please start Ollama first.")

        # Get response from Ollama
        response = ollama_client.chat(message, model, conversation_history)

        return {
            "response": response,
            "model": model,
            "message_count": len(conversation_history) + 1
        }

    except HTTPException:
        raise
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
            ollama_embedding_models = [m for m in all_ollama_models if any(k in m for k in ['embed','bge','embedding','text'])]
            ollama_generation_models = [m for m in all_ollama_models if m not in ollama_embedding_models]

            generation_models.extend(ollama_generation_models)
            embedding_models.extend(ollama_embedding_models)

        # Add supported HuggingFace embedding models
        huggingface_embedding_models = [
            "Qwen/Qwen3-Embedding-0.6B",
            "Qwen/Qwen3-Embedding-4B",
            "Qwen/Qwen3-Embedding-8B"
        ]
        embedding_models.extend(huggingface_embedding_models)

        # Sort models for consistent ordering
        generation_models.sort()
        embedding_models.sort()

        return {
            "generation_models": generation_models,
            "embedding_models": embedding_models
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not list models: {str(e)}")

@app.get("/indexes")
async def get_indexes():
    """Get all indexes"""
    try:
        _recover_stale_index_builds()
        data = db.list_indexes()
        return {"indexes": data, "total": len(data)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/indexes/diagnostics")
async def get_indexes_diagnostics():
    """Get compact health diagnostics for all indexes."""
    try:
        _recover_stale_index_builds()
        summaries = []
        for item in db.list_indexes():
            idx_id = item.get("id") or item.get("index_id")
            if not idx_id:
                continue
            try:
                summaries.append(_index_diagnostics_summary(str(idx_id)))
            except Exception as e:
                summaries.append({
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
                })
        return {"diagnostics": summaries, "total": len(summaries)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/indexes")
async def create_index(request: Request):
    """Create a new index"""
    try:
        data = await request.json()
        name = data.get('name')
        description = data.get('description')
        metadata = data.get('metadata', {})

        if not name:
            raise HTTPException(status_code=400, detail="Name required")

        # Add complete metadata from RAG system configuration if available
        if RAG_SYSTEM_AVAILABLE and PIPELINE_CONFIGS.get('default'):
            default_config = PIPELINE_CONFIGS['default']
            complete_metadata = {
                'status': 'created',
                'metadata_source': 'rag_system_config',
                'created_at': json.loads(json.dumps(datetime.now().isoformat())),
                'chunk_size': 512,  # From default config
                'chunk_overlap': 64,  # From default config
                'retrieval_mode': 'hybrid',  # From default config
                'window_size': 5,  # From default config
                'embedding_model': 'Qwen/Qwen3-Embedding-0.6B',  # From default config
                'enrich_model': 'qwen3:0.6b',  # From default config
                'overview_model': 'qwen3:0.6b',  # From default config
                'enable_enrich': True,  # From default config
                'latechunk': True,  # From default config
                'docling_chunk': True,  # From default config
                'note': 'Default configuration from RAG system'
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
        deleted = db.delete_index(index_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Index not found")
        return {"message": "Index deleted successfully", "index_id": index_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/indexes/{index_id}/upload")
async def index_file_upload(index_id: str, files: List[UploadFile] = File(...)):
    """Upload files to an index"""
    uploaded_files = []
    upload_dir = 'shared_uploads'
    os.makedirs(upload_dir, exist_ok=True)

    for f in files:
        if f.filename:
            unique = f"{uuid.uuid4()}_{f.filename}"
            path = os.path.join(upload_dir, unique)
            with open(path, 'wb') as out:
                content = await f.read()
                out.write(content)
            db.add_document_to_index(index_id, f.filename, os.path.abspath(path))
            uploaded_files.append({'filename': f.filename, 'stored_path': os.path.abspath(path)})

    if not uploaded_files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    return {"message": f"Uploaded {len(uploaded_files)} files", "uploaded_files": uploaded_files}


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


def _run_index_build(index_id: str, data: Dict[str, Any], job_id: str | None = None) -> Dict[str, Any]:
    if not RAG_SYSTEM_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail=(
                "RAG indexing dependencies are not available in this backend Python process. "
                "Stop the backend, run 'source .venv/bin/activate' from the project root, "
                "then restart with 'python backend/server.py'."
            ),
        )

    index = db.get_index(index_id)
    if not index:
        raise HTTPException(status_code=404, detail="Index not found")

    file_paths = [d['stored_path'] for d in index.get('documents', [])]
    if not file_paths:
        raise HTTPException(status_code=400, detail="No documents to index")

    preflight = _index_build_preflight(index_id, data, check_services=True)
    if not preflight["ok"]:
        detail = "; ".join(preflight["errors"])
        raise HTTPException(status_code=503 if preflight.get("rag_api_available") is False else 400, detail=detail)

    latechunk = bool(data.get('latechunk', False))
    docling_chunk = bool(data.get('doclingChunk', False))
    chunk_size = int(data.get('chunkSize', 512))
    chunk_overlap = int(data.get('chunkOverlap', 64))
    retrieval_mode = str(data.get('retrievalMode', 'hybrid'))
    window_size = int(data.get('windowSize', 2))
    enable_enrich = bool(data.get('enableEnrich', True))
    embedding_model = data.get('embeddingModel')
    enrich_model = data.get('enrichModel')
    batch_size_embed = int(data.get('batchSizeEmbed', 50))
    batch_size_enrich = int(data.get('batchSizeEnrich', 25))
    overview_model = data.get('overviewModel')
    force_reindex = bool(data.get('forceReindex', False))
    indexing_model_warnings = []

    if _large_indexing_model(enrich_model):
        indexing_model_warnings.append(
            f"Replaced enrichment model '{enrich_model}' with qwen3:0.6b for indexing safety."
        )
        enrich_model = "qwen3:0.6b"
    if _large_indexing_model(overview_model):
        indexing_model_warnings.append(
            f"Replaced overview model '{overview_model}' with qwen3:0.6b for indexing safety."
        )
        overview_model = "qwen3:0.6b"

    window_size = max(0, min(window_size, 2))
    batch_size_enrich = max(1, min(batch_size_enrich, 8))

    # Set per-index overview file path
    overview_path = f"index_store/overviews/{index_id}.jsonl"

    # Delegate to advanced RAG API same as session indexing
    rag_api_url = f"{RAG_API_BASE_URL}/index"
    # Use the index's dedicated LanceDB table so retrieval matches
    table_name = index.get("vector_table_name")
    payload = {
        "file_paths": file_paths,
        "session_id": index_id,  # reuse index_id for progress tracking
        "table_name": table_name,
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
    if overview_model:
        payload["overview_model_name"] = overview_model
    if job_id:
        payload["job_id"] = job_id
        payload["backend_base_url"] = "http://localhost:8000"

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
        _update_index_job(job_id, stage="indexing", progress=20, message="RAG pipeline is indexing documents")

    try:
        rag_resp = requests.post(rag_api_url, json=payload)
    except requests.exceptions.ConnectionError as e:
        detail = (
            "Could not reach the RAG indexing server at http://localhost:8001. "
            "Start it in Terminal 2 with 'source .venv/bin/activate' and "
            "'python -m rag_system.api_server'."
        )
        db.update_index_metadata(index_id, {
            "status": "failed",
            "build_failed_at": datetime.now().isoformat(),
            "build_error": detail,
        })
        raise HTTPException(status_code=503, detail=detail) from e
    except requests.exceptions.RequestException as e:
        detail = f"RAG indexing request failed before completion: {e}"
        db.update_index_metadata(index_id, {
            "status": "failed",
            "build_failed_at": datetime.now().isoformat(),
            "build_error": detail,
        })
        raise HTTPException(status_code=503, detail=detail) from e
    if rag_resp.status_code == 200:
        final_updates = {
            **meta_updates,
            "status": "functional",
            "rebuilt_at": datetime.now().isoformat(),
        }
        db.update_index_metadata(index_id, final_updates)

        response_data = rag_resp.json()
        response_data.update(final_updates)
        return response_data

    if rag_resp.status_code == 499:
        db.update_index_metadata(index_id, {
            "status": "cancelled",
            "build_cancelled_at": datetime.now().isoformat(),
        })
        raise RuntimeError("indexing_cancelled")

    # Gracefully handle scenario where table already exists (idempotent build)
    try:
        err_json = rag_resp.json()
    except Exception:
        err_json = {}
    err_text = err_json.get('error') if isinstance(err_json, dict) else rag_resp.text
    if err_text and 'already exists' in err_text:
        db.update_index_metadata(index_id, {"status": "functional", "rebuilt_at": datetime.now().isoformat()})
        return {
            "message": "Index already built – skipping rebuild.",
            "note": err_text
        }

    db.update_index_metadata(index_id, {
        "status": "failed",
        "build_failed_at": datetime.now().isoformat(),
        "build_error": rag_resp.text,
    })
    raise HTTPException(status_code=500, detail=f"RAG indexing failed: {rag_resp.text}")


def _run_index_build_job(job_id: str):
    job = _get_index_job(job_id)
    if not job:
        return
    if job.get("cancel_requested"):
        _update_index_job(job_id, status="cancelled", stage="cancelled", progress=100, message="Build cancelled before it started")
        db.update_index_metadata(job["index_id"], {"status": "cancelled", "build_cancelled_at": datetime.now().isoformat()})
        return

    _update_index_job(job_id, status="running", stage="validating", progress=5, message="Preparing index build")
    try:
        result = _run_index_build(job["index_id"], job["options"], job_id=job_id)
        latest = _get_index_job(job_id) or {}
        status = "cancelled" if latest.get("cancel_requested") else "completed"
        message = "Build completed after cancellation request" if status == "cancelled" else "Build completed"
        if status == "cancelled":
            db.update_index_metadata(job["index_id"], {"status": "cancelled", "build_cancelled_at": datetime.now().isoformat()})
        _update_index_job(job_id, status=status, stage=status, progress=100, message=message, result=result, finished_at=datetime.now().isoformat())
    except RuntimeError as e:
        if str(e) == "indexing_cancelled":
            db.update_index_metadata(job["index_id"], {"status": "cancelled", "build_cancelled_at": datetime.now().isoformat()})
            _update_index_job(job_id, status="cancelled", stage="cancelled", progress=100, message="Indexing cancelled", finished_at=datetime.now().isoformat())
            return
        raise
    except Exception as e:
        db.update_index_metadata(job["index_id"], {
            "status": "failed",
            "build_failed_at": datetime.now().isoformat(),
            "build_error": str(e),
        })
        _update_index_job(job_id, status="failed", stage="failed", progress=100, message=str(e), error=str(e), finished_at=datetime.now().isoformat())


@app.post("/indexes/{index_id}/build")
async def build_index(index_id: str, request: Request):
    """Build an index from uploaded documents"""
    try:
        _recover_stale_index_builds()
        body = await request.body()
        data = json.loads(body.decode("utf-8")) if body else {}
        if bool(data.get("background", False)):
            index = db.get_index(index_id)
            if not index:
                raise HTTPException(status_code=404, detail="Index not found")
            preflight = _index_build_preflight(index_id, data, check_services=True)
            if not preflight["ok"]:
                detail = "; ".join(preflight["errors"])
                raise HTTPException(status_code=503 if preflight.get("rag_api_available") is False else 400, detail=detail)
            job_id = str(uuid.uuid4())
            now = datetime.now().isoformat()
            db_job = db.create_index_job(job_id, index_id, data, index.get("documents", []))
            with index_jobs_lock:
                index_jobs[job_id] = {
                    "id": job_id,
                    "index_id": index_id,
                    "status": "queued",
                    "stage": "queued",
                    "progress": 0,
                    "message": "Build queued",
                    "cancel_requested": False,
                    "options": data,
                    "created_at": now,
                    "updated_at": now,
                    "files": db_job.get("files", []) if db_job else [],
                }
            db.update_index_metadata(index_id, {
                "status": "building",
                "build_job_id": job_id,
                "build_started_at": now,
            })
            thread = threading.Thread(target=_run_index_build_job, args=(job_id,), daemon=True)
            thread.start()
            return {"message": "Index build started", "job_id": job_id, "status": "queued"}

        return _run_index_build(index_id, data)
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
        _update_index_job(job_id, status="cancelled", stage="cancelled", progress=100, finished_at=datetime.now().isoformat())
        db.update_index_metadata(job["index_id"], {"status": "cancelled", "build_cancelled_at": datetime.now().isoformat()})
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
            model="qwen3:0.6b",  # Fast model for routing
            enable_thinking=False  # Fast routing
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
        'hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening',
        'how are you', 'how do you do', 'pleasure to meet',
        'thanks', 'thank you', 'bye', 'goodbye', 'see you', 'talk to you later',
        'test', 'testing', 'check', 'ping', 'just saying', 'nevermind',
        'ok', 'okay', 'alright', 'got it', 'understood', 'i see'
    ]

    # Check for greeting patterns
    for pattern in greeting_patterns:
        if pattern in message_lower:
            return False  # Use Direct LLM for greetings

    # Keywords that strongly suggest document-related queries
    rag_indicators = [
        'document', 'doc', 'file', 'pdf', 'text', 'content', 'page',
        'according to', 'based on', 'mentioned', 'states', 'says',
        'what does', 'summarize', 'summary', 'analyze', 'analysis',
        'quote', 'citation', 'reference', 'source', 'evidence',
        'explain from', 'extract', 'find in', 'search for'
    ]

    # Check for strong RAG indicators
    for indicator in rag_indicators:
        if indicator in message_lower:
            return True

    # Question words + substantial length might benefit from RAG
    question_words = ['what', 'how', 'when', 'where', 'why', 'who', 'which']
    starts_with_question = any(message_lower.startswith(word) for word in question_words)

    if starts_with_question and len(message) > 40:
        return True

    # Very short messages - use direct LLM
    if len(message.strip()) < 20:
        return False

    # Default to Direct LLM unless there's clear indication of document query
    return False

async def _handle_direct_llm_query(session_id: str, message: str, session: dict):
    """
    Handle query using direct Ollama client with thinking disabled for speed.

    Returns:
        tuple: (response_text, empty_source_docs)
    """
    try:
        # Get conversation history for context
        conversation_history = db.get_conversation_history(session_id)

        # Use the session's model or default
        model = session.get('model', 'qwen3:8b')  # Default to fast model

        # Direct Ollama call with thinking disabled for speed
        response_text = ollama_client.chat(
            message=message,
            model=model,
            conversation_history=conversation_history,
            enable_thinking=False  # ⚡ DISABLE THINKING FOR SPEED
        )

        return response_text, []  # No source docs for direct LLM

    except Exception as e:
        print(f"❌ Direct LLM error: {e}")
        return f"Error processing query: {str(e)}", []

async def _handle_rag_query(session_id: str, message: str, data: dict, idx_ids: List[str]):
    """
    Handle query using the full RAG pipeline (delegates to the advanced RAG API running on port 8001).

    Returns:
        tuple[str, List[dict]]: (response_text, source_documents)
    """
    # Defaults
    response_text = ""
    source_docs: List[dict] = []

    # Build payload for RAG API
    rag_api_url = "http://localhost:8001/chat"
    table_name = f"text_pages_{idx_ids[-1]}" if idx_ids else None
    payload: Dict[str, Any] = {
        "query": message,
        "session_id": session_id,
    }
    if table_name:
        payload["table_name"] = table_name

    # Copy optional parameters from the incoming request
    optional_params: Dict[str, tuple[type, str]] = {
        "compose_sub_answers": (bool, "compose_sub_answers"),
        "query_decompose": (bool, "query_decompose"),
        "ai_rerank": (bool, "ai_rerank"),
        "context_expand": (bool, "context_expand"),
        "verify": (bool, "verify"),
        "retrieval_k": (int, "retrieval_k"),
        "context_window_size": (int, "context_window_size"),
        "reranker_top_k": (int, "reranker_top_k"),
        "search_type": (str, "search_type"),
        "dense_weight": (float, "dense_weight"),
        "provence_prune": (bool, "provence_prune"),
        "provence_threshold": (float, "provence_threshold"),
    }
    for key, (caster, payload_key) in optional_params.items():
        val = data.get(key)
        if val is not None:
            try:
                payload[payload_key] = caster(val)  # type: ignore[arg-type]
            except Exception:
                payload[payload_key] = val

    try:
        rag_response = requests.post(rag_api_url, json=payload)
        if rag_response.status_code == 200:
            rag_data = rag_response.json()
            response_text = rag_data.get("answer", "No answer found.")
            source_docs = rag_data.get("source_documents", [])
        else:
            response_text = f"Error from RAG API ({rag_response.status_code}): {rag_response.text}"
            print(f"❌ RAG API error: {response_text}")
    except requests.exceptions.ConnectionError:
        response_text = "Could not connect to the RAG API server. Please ensure it is running."
        print("❌ Connection to RAG API failed (port 8001).")
    except Exception as e:
        response_text = f"Error processing RAG query: {str(e)}"
        print(f"❌ RAG processing error: {e}")

    # Strip any <think>/<thinking> tags that might slip through
    response_text = re.sub(r'<(think|thinking)>.*?</\\1>', '', response_text, flags=re.DOTALL | re.IGNORECASE).strip()

    return response_text, source_docs

def main():
    """Main function to initialize and start the server"""
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
            print("⚠️ PDF processing disabled - server will run without RAG functionality")

        # Set a global reference to the initialized processor if needed elsewhere
        global pdf_processor
        pdf_processor = pdf_module.simple_pdf_processor
        if pdf_processor:
            print("✅ Global PDF processor initialized")
        else:
            print("⚠️ PDF processing disabled - server will run without RAG functionality")

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
            print(f"📋 Available models: {', '.join(models[:3])}{'...' if len(models) > 3 else ''}")
        else:
            print("⚠️  Ollama is not running. Please start Ollama:")
            print("   Install: https://ollama.ai")
            print("   Run: ollama serve")

        print(f"\n🌐 Frontend should connect to: http://localhost:{PORT}")
        print("💬 Ready to chat!\n")

        uvicorn.run(app, host="0.0.0.0", port=PORT)

    except KeyboardInterrupt:
        print("\n🛑 Server stopped")

if __name__ == "__main__":
    main()

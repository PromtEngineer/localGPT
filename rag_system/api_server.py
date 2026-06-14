import json
import os
import queue
import requests
import sys
import threading
import logging

# Add backend directory to path for database imports
backend_dir = os.path.join(os.path.dirname(__file__), '..', 'backend')
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

from backend.database import ChatDatabase, generate_session_title
from rag_system.index_selection import select_active_index_id
from rag_system.main import get_agent
from rag_system.factory import get_indexing_pipeline

# Initialize database connection once at module level
# Use auto-detection for environment-appropriate path
db = ChatDatabase()

# Get the desired agent mode from environment variables, defaulting to 'default'
# This allows us to easily switch between 'default', 'fast', 'react', etc.
AGENT_MODE = os.getenv("RAG_CONFIG_MODE", "default")
print("🧠 Initializing RAG Agent... (This may take a moment)")
RAG_AGENT = get_agent(AGENT_MODE)
INDEXING_PIPELINE = get_indexing_pipeline(AGENT_MODE)

if RAG_AGENT is None:
    raise RuntimeError("RAG Agent could not be initialized")
print("✅ RAG Agent initialized successfully.")

# Add helper near top after db & agent init
# -------------- Helper ----------------

def _get_table_name_for_session(session_id):
    """Get the correct vector table name for a session by looking up its linked indexes."""
    logger = logging.getLogger(__name__)
    
    if not session_id:
        logger.info("❌ No session_id provided")
        return None
    
    try:
        # Get indexes linked to this session
        idx_ids = db.get_indexes_for_session(session_id)
        logger.info(f"🔍 Session {session_id[:8]}... has {len(idx_ids)} indexes: {idx_ids}")
        
        if not idx_ids:
            logger.warning(f"⚠️ No indexes found for session {session_id}")
            # Use the default table name from config instead of session-specific name
            from rag_system.main import PIPELINE_CONFIGS
            default_table = PIPELINE_CONFIGS["default"]["storage"]["text_table_name"]
            logger.info(f"📊 Using default table '{default_table}' for session {session_id[:8]}...")
            return default_table
        
        # Use the ACTIVE index's vector table — collection metadata carries
        # the matching embedding model and fusion settings into retrieval.
        idx = db.get_index(select_active_index_id(idx_ids))
        if idx and idx.get('vector_table_name'):
            table_name = idx['vector_table_name']
            logger.info(f"📊 Using table '{table_name}' for session {session_id[:8]}...")
            print(f"📊 RAG API: Using table '{table_name}' for session {session_id[:8]}...")
            return table_name
        else:
            logger.warning(f"⚠️ Index found but no vector table name for session {session_id}")
            # Use the default table name from config instead of session-specific name
            from rag_system.main import PIPELINE_CONFIGS
            default_table = PIPELINE_CONFIGS["default"]["storage"]["text_table_name"]
            logger.info(f"📊 Using default table '{default_table}' for session {session_id[:8]}...")
            return default_table
            
    except Exception as e:
        logger.error(f"❌ Error getting table name for session {session_id}: {e}")
        # Use the default table name from config instead of session-specific name
        from rag_system.main import PIPELINE_CONFIGS
        default_table = PIPELINE_CONFIGS["default"]["storage"]["text_table_name"]
        logger.info(f"📊 Using default table '{default_table}' for session {session_id[:8]}...")
        return default_table

def _get_collections_for_session(session_id):
    """All linked indexes as retrieval collections (most recent last, cap 5).

    Each entry carries the table plus the embedding model that table was
    built with, so the pipeline can embed the query per collection.
    """
    if not session_id:
        return None
    try:
        idx_ids = db.get_indexes_for_session(session_id)
    except Exception as e:
        logging.getLogger(__name__).warning("collections_lookup_failed session=%s error=%s", session_id, e)
        return None
    collections = []
    for iid in idx_ids[-5:]:
        idx = db.get_index(iid)
        if idx and idx.get("vector_table_name"):
            meta = idx.get("metadata") or {}
            collections.append({
                "index_id": iid,
                "table_name": idx["vector_table_name"],
                "embedding_model": meta.get("embedding_model"),
                "index_name": idx.get("name"),
                "metadata_schema": meta.get("metadata_schema"),
                "fusion_config": meta.get("fusion_config"),
            })
    return collections or None


def _collection_for_table(table_name):
    """Single-collection entry (with metadata schema) for an explicit table."""
    try:
        for idx in db.list_indexes():
            if idx.get("vector_table_name") == table_name:
                meta = idx.get("metadata") or {}
                return [{
                    "index_id": idx.get("id"),
                    "table_name": table_name,
                    "embedding_model": meta.get("embedding_model"),
                    "index_name": idx.get("name"),
                    "metadata_schema": meta.get("metadata_schema"),
                    "fusion_config": meta.get("fusion_config"),
                }]
    except Exception as e:
        logging.getLogger(__name__).warning("table_collection_lookup_failed table=%s error=%s", table_name, e)
    return None


def _force_rag_overrides(retrieval_k, reranker_top_k, search_type, dense_weight,
                         ai_rerank, provence_prune, provence_threshold) -> dict:
    """Per-request retrieval overrides for the force_rag path (None = use config)."""
    return {
        "retrieval_k": retrieval_k,
        "reranker_top_k": reranker_top_k,
        "search_type": search_type,
        "dense_weight": dense_weight,
        "ai_rerank": ai_rerank,
        "provence_enabled": provence_prune,
        "provence_threshold": provence_threshold,
    }


def _cors_allowed_origins() -> list[str]:
    origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
    return [o.strip() for o in origins.split(",") if o.strip()]


def _execute_indexing_job(config, file_paths, *, index_id, force_reindex, job_id, backend_base_url):
    """Run an index build in a spawned child process.

    Isolation buys two things: the build's peak memory (torch, Docling,
    embeddings) is returned to the OS when the child exits, and an OOM or
    crash kills the build instead of the chat server. Set
    RAG_INDEX_IN_PROCESS=1 to run in-process (debugging/tests).
    """
    from rag_system.indexing_worker import run_indexing_job

    if os.getenv("RAG_INDEX_IN_PROCESS") == "1":
        return run_indexing_job(
            config, INDEXING_PIPELINE.ollama_config, file_paths,
            index_id, force_reindex, job_id, backend_base_url,
        )

    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool

    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as executor:
        future = executor.submit(
            run_indexing_job, config, INDEXING_PIPELINE.ollama_config,
            file_paths, index_id, force_reindex, job_id, backend_base_url,
        )
        try:
            return future.result()
        except BrokenProcessPool as e:
            raise RuntimeError(
                "Indexing process crashed (likely out of memory). Try a smaller "
                "embedding batch size, disable enrichment, or index fewer files at once."
            ) from e


# ===========================================================================
# FastAPI transport (replaces the hand-rolled threading http.server).
# Routes are thin; all retrieval/index logic lives in the module functions
# below and the request-scoped Agent — no shared mutable state, so requests
# run concurrently in the threadpool with no lock.
# ===========================================================================
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

app = FastAPI(title="LocalGPT RAG API")

_cors_origins = _cors_allowed_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials="*" not in _cors_origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def _parse_chat_request(data: dict):
    """Parse/normalize a chat body. Returns a params dict, or None if the
    query is missing (caller returns 400)."""
    query = data.get("query")
    if not query:
        return None

    requested_model = data.get("model")
    session_id = data.get("session_id")
    table_name = data.get("table_name")
    collections = None
    if not table_name and session_id:
        # No explicit table: search ALL of the session's linked indexes.
        # An explicit table_name (eval harness, API callers) pins a single
        # collection, preserving the old contract.
        collections = _get_collections_for_session(session_id)
        table_name = _get_table_name_for_session(session_id)
    elif table_name:
        # Explicit tables still need their embedding/fusion/schema metadata.
        collections = _collection_for_table(table_name)

    return {
        "query": query,
        "model": requested_model if isinstance(requested_model, str) and requested_model else None,
        "session_id": session_id,
        "table_name": table_name,
        "collections": collections,
        "compose_flag": data.get("compose_sub_answers"),
        "decomp_flag": data.get("query_decompose"),
        "ai_rerank_flag": data.get("ai_rerank"),
        "ctx_expand_flag": data.get("context_expand"),
        "verify_flag": data.get("verify"),
        "retrieval_k": data.get("retrieval_k", 20),
        "context_window_size": data.get("context_window_size", 1),
        "reranker_top_k": data.get("reranker_top_k", 10),
        "search_type": data.get("search_type", "hybrid"),
        # No default: a value here overrides the per-index fusion config
        # stored with the index, so only explicit caller overrides count
        "dense_weight": data.get("dense_weight"),
        "force_rag": bool(data.get("force_rag", False)),
        "filters": data.get("filters") if isinstance(data.get("filters"), dict) else None,
        "agentic": data.get("agentic") if isinstance(data.get("agentic"), bool) else None,
        "provence_prune": data.get("provence_prune"),
        "provence_threshold": data.get("provence_threshold"),
    }


def _run_chat(params, event_callback=None):
    """Execute a chat request (force_rag bypass or full agent) and return the
    result dict. event_callback (SSE emit) is threaded through when streaming."""
    query = params["query"]
    session_id = params["session_id"]
    table_name = params["table_name"]
    force_rag = params["force_rag"]

    generation_model = params["model"] or RAG_AGENT.ollama_config["generation_model"]
    document_overviews = []
    if session_id:
        idx_ids = db.get_indexes_for_session(session_id)
        document_overviews = RAG_AGENT.get_overviews_for_indexes(idx_ids)

    print(f"🔧 Force RAG flag: {force_rag}")
    if force_rag:
        overrides = _force_rag_overrides(
            params["retrieval_k"], params["reranker_top_k"], params["search_type"],
            params["dense_weight"], params["ai_rerank_flag"],
            params["provence_prune"], params["provence_threshold"],
        )
        overrides.update({"generation_model": generation_model, "latechunk_enabled": True})
        return RAG_AGENT.retrieval_pipeline.run(
            query,
            table_name=table_name,
            window_size_override=params["context_window_size"],
            collections=params.get("collections"),
            filters=params.get("filters"),
            overrides=overrides,
            event_callback=event_callback,
        )
    return RAG_AGENT.run(
        query,
        table_name=table_name,
        collections=params.get("collections"),
        filters=params.get("filters"),
        session_id=session_id,
        compose_sub_answers=params["compose_flag"],
        query_decompose=params["decomp_flag"],
        ai_rerank=params["ai_rerank_flag"],
        context_expand=params["ctx_expand_flag"],
        verify=params["verify_flag"],
        retrieval_k=params["retrieval_k"],
        context_window_size=params["context_window_size"],
        reranker_top_k=params["reranker_top_k"],
        search_type=params["search_type"],
        dense_weight=params["dense_weight"],
        agentic=params.get("agentic"),
        generation_model=generation_model,
        document_overviews=document_overviews,
        provence_prune=params["provence_prune"],
        provence_threshold=params["provence_threshold"],
        latechunk_enabled=True,
        event_callback=event_callback,
    )


def _stream_chat(params):
    """Bridge the agent's push-based event_callback into a pull-based SSE
    generator: the agent runs in a worker thread, pushing onto a queue."""
    q: queue.Queue = queue.Queue()
    sentinel = object()

    def emit(event_type, payload):
        q.put("data: " + json.dumps({"type": event_type, "data": payload}) + "\n\n")

    def worker():
        try:
            final_result = _run_chat(params, event_callback=emit)
            emit("complete", final_result)
        except Exception as e:  # surface as an SSE error event, then close
            print(f"❌ Stream error: {e}")
            emit("error", {"error": str(e)})
        finally:
            q.put(sentinel)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        item = q.get()
        if item is sentinel:
            break
        yield item


@app.post("/chat")
async def chat(request: Request):
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    params = _parse_chat_request(data)
    if params is None:
        return JSONResponse({"error": "Query is required"}, status_code=400)
    try:
        result = await run_in_threadpool(_run_chat, params)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": f"Server error: {str(e)}"}, status_code=500)


@app.post("/chat/stream")
async def chat_stream(request: Request):
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    params = _parse_chat_request(data)
    if params is None:
        return JSONResponse({"error": "Query is required"}, status_code=400)
    return StreamingResponse(
        _stream_chat(params),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


def _clear_index_artifacts(pipeline, table_name, index_id):
    """Remove old vector/overview artifacts before a force rebuild."""
    if table_name and hasattr(pipeline, "lancedb_manager"):
        _db = pipeline.lancedb_manager.db
        table_names = _db.table_names(limit=10_000) if hasattr(_db, "table_names") else []
        for candidate in (table_name, f"{table_name}_lc"):
            if candidate in table_names:
                try:
                    _db.drop_table(candidate)
                    print(f"🚮 Dropped existing LanceDB table '{candidate}' for force rebuild")
                except Exception as e:
                    print(f"⚠️ Could not drop LanceDB table '{candidate}': {e}")
    if index_id:
        overview_path = f"index_store/overviews/{index_id}.jsonl"
        try:
            if os.path.exists(overview_path):
                os.remove(overview_path)
                print(f"🚮 Removed overview file '{overview_path}' for force rebuild")
        except Exception as e:
            print(f"⚠️ Could not remove overview file '{overview_path}': {e}")


def _run_index(data: dict) -> dict:
    """Build/rebuild an index. Raises ValueError (->400) for bad input and
    RuntimeError('indexing_cancelled') (->499); other errors ->500."""
    import copy

    file_paths = data.get("file_paths")
    session_id = data.get("session_id")
    enable_latechunk = bool(data.get("enable_latechunk", False))
    enable_docling_chunk = bool(data.get("enable_docling_chunk", False))
    chunk_size = int(data.get("chunk_size", 512))
    chunk_overlap = int(data.get("chunk_overlap", 64))
    retrieval_mode = data.get("retrieval_mode", "hybrid")
    window_size = int(data.get("window_size", 2))
    # Default OFF: enrichment is one LLM call per chunk — opt-in only
    enable_enrich = bool(data.get("enable_enrich", False))
    embedding_model = data.get("embedding_model") or data.get("embeddingModel")
    enrich_model = data.get("enrich_model") or data.get("enrichModel")
    overview_model = data.get("overviewModel") or data.get("overview_model_name")
    enrich_provider = data.get("enrich_provider", "ollama")
    enrich_api_key = data.get("enrich_api_key")  # never logged or stored
    batch_size_embed = int(data.get("batch_size_embed", 50))
    batch_size_enrich = int(data.get("batch_size_enrich", 25))
    force_reindex = bool(data.get("force_reindex", False))
    job_id = data.get("job_id")
    backend_base_url = data.get("backend_base_url", "http://localhost:8000")
    indexing_model_warnings = []

    def is_large_indexing_model(model):
        if not model:
            return False
        lowered = str(model).lower()
        return any(token in lowered for token in ("gpt-oss", "120b", "70b", "large", "cloud"))

    if enrich_provider == "ollama" and is_large_indexing_model(enrich_model):
        indexing_model_warnings.append(
            f"Replaced enrichment model '{enrich_model}' with qwen3:8b for indexing safety."
        )
        enrich_model = "qwen3:8b"
    if is_large_indexing_model(overview_model):
        indexing_model_warnings.append(
            f"Replaced overview model '{overview_model}' with qwen3:8b for indexing safety."
        )
        overview_model = "qwen3:8b"

    window_size = max(0, min(window_size, 2))
    batch_size_enrich = max(1, min(batch_size_enrich, 8))

    if not file_paths or not isinstance(file_paths, list):
        raise ValueError("A 'file_paths' list is required.")

    table_name = data.get("table_name")
    if not table_name and session_id:
        table_name = _get_table_name_for_session(session_id)

    config_override = copy.deepcopy(INDEXING_PIPELINE.config)
    if table_name:
        config_override["storage"]["text_table_name"] = table_name
        config_override.setdefault("retrievers", {}).setdefault("dense", {})["lancedb_table_name"] = table_name

    config_override.setdefault("retrievers", {}).setdefault("latechunk", {})["enabled"] = enable_latechunk
    if enable_docling_chunk:
        config_override["chunker_mode"] = "docling"
    config_override.setdefault("contextual_enricher", {})
    config_override["contextual_enricher"]["enabled"] = enable_enrich
    config_override["contextual_enricher"]["window_size"] = window_size
    config_override.setdefault("indexing", {})
    config_override["indexing"]["embedding_batch_size"] = batch_size_embed
    config_override["indexing"]["enrichment_batch_size"] = batch_size_enrich
    config_override["indexing"].setdefault("conversion_timeout_seconds", int(os.getenv("CONVERSION_TIMEOUT_SECONDS", "900")))
    config_override["indexing"].setdefault("overview_timeout_seconds", 45)
    config_override["indexing"].setdefault("enrichment_timeout_seconds", 60)
    if data.get("metadata_schema"):
        config_override["metadata_schema"] = data["metadata_schema"]
    if data.get("file_metadata"):
        config_override["file_metadata"] = data["file_metadata"]
    config_override.setdefault("chunking", {})
    config_override["chunking"]["chunk_size"] = chunk_size
    config_override["chunking"]["chunk_overlap"] = chunk_overlap
    if embedding_model:
        config_override["embedding_model_name"] = embedding_model
    if enrich_model:
        config_override["enrich_model"] = enrich_model
    if enrich_provider and enrich_provider != "ollama":
        config_override["enrich_provider"] = enrich_provider
        if enrich_api_key:
            config_override["enrich_api_key"] = enrich_api_key
    if overview_model:
        config_override["overview_model_name"] = overview_model
    if session_id:
        config_override["overview_path"] = f"index_store/overviews/{session_id}.jsonl"

    print(f"🔧 INDEXING CONFIG: Contextual Enrichment: {enable_enrich}, Window Size: {window_size}")
    print(f"🔧 CHUNKING CONFIG: Size: {chunk_size}, Overlap: {chunk_overlap}")
    print(f"🔧 MODEL CONFIG: Embedding: {embedding_model or 'default'}, Enrichment: {enrich_model or 'default'}")

    if force_reindex:
        _clear_index_artifacts(INDEXING_PIPELINE, table_name, session_id)
    indexing_result = _execute_indexing_job(
        config_override,
        file_paths,
        index_id=session_id or table_name or "default",
        force_reindex=force_reindex,
        job_id=job_id,
        backend_base_url=backend_base_url,
    )

    if embedding_model:
        try:
            db.update_index_metadata(session_id, {"embedding_model": embedding_model})
        except Exception as e:
            print(f"⚠️ Could not update embedding_model metadata: {e}")

    return {
        "message": f"Indexing process for {len(file_paths)} file(s) completed successfully.",
        "table_name": table_name or "default_text_table",
        "latechunk": enable_latechunk,
        "docling_chunk": enable_docling_chunk,
        "indexing_config": {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "retrieval_mode": retrieval_mode,
            "window_size": window_size,
            "enable_enrich": enable_enrich,
            "embedding_model": embedding_model,
            "enrich_model": enrich_model,
            "batch_size_embed": batch_size_embed,
            "batch_size_enrich": batch_size_enrich,
            "force_reindex": force_reindex,
        },
        "indexing_result": indexing_result,
        "indexing_model_warnings": indexing_model_warnings,
    }


@app.post("/index")
async def index(request: Request):
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    try:
        result = await run_in_threadpool(_run_index, data)
        return JSONResponse(result)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except RuntimeError as e:
        if str(e) == "indexing_cancelled":
            return JSONResponse({"error": "Indexing cancelled"}, status_code=499)
        return JSONResponse({"error": f"Failed to start indexing: {str(e)}"}, status_code=500)
    except Exception as e:
        return JSONResponse({"error": f"Failed to start indexing: {str(e)}"}, status_code=500)


@app.get("/health")
def health():
    """Lightweight health probe that avoids loading large ML models."""
    checks: dict = {}
    overall = "ok"
    checks["agent"] = "ok" if RAG_AGENT is not None else "error"
    if RAG_AGENT is None:
        overall = "degraded"
    try:
        import lancedb as _lancedb
        lancedb_uri = os.getenv("LANCEDB_URI", "./lancedb")
        if os.path.exists(lancedb_uri):
            conn = _lancedb.connect(lancedb_uri)
            checks["lancedb"] = f"ok ({len(conn.table_names(limit=10_000))} tables)"
        else:
            checks["lancedb"] = "not_initialized"
    except Exception as e:
        checks["lancedb"] = f"error: {e}"
        overall = "degraded"
    try:
        if RAG_AGENT is not None:
            embedder = getattr(getattr(RAG_AGENT, "retrieval_pipeline", None), "text_embedder", None)
            checks["embedder"] = "loaded" if embedder is not None else "not_loaded"
        else:
            checks["embedder"] = "agent_unavailable"
    except Exception as e:
        checks["embedder"] = f"error: {e}"
        overall = "degraded"
    return {"status": overall, "checks": checks}


@app.get("/models")
def models():
    """Locally installed Ollama models + supported HuggingFace models."""
    generation_models, embedding_models = [], []
    try:
        resp = requests.get(f"{RAG_AGENT.ollama_config['host']}/api/tags", timeout=5)
        resp.raise_for_status()
        all_ollama = [m.get("name") for m in resp.json().get("models", [])]
        emb = [m for m in all_ollama if any(k in m for k in ["embed", "bge", "embedding", "text"])]
        generation_models.extend([m for m in all_ollama if m not in emb])
        embedding_models.extend(emb)
    except Exception as e:
        print(f"⚠️ Could not get Ollama models: {e}")
    try:
        from rag_system.model_registry import huggingface_models
        embedding_models.extend(huggingface_models())
    except ImportError:
        embedding_models.extend(["Qwen/Qwen3-Embedding-0.6B"])
    generation_models.sort()
    embedding_models.sort()
    return {"generation_models": generation_models, "embedding_models": embedding_models}


def start_server(port=8001):
    """Starts the FastAPI RAG API via uvicorn."""
    import uvicorn

    # Loopback by default: this API is unauthenticated, so exposing it on all
    # interfaces would let anyone on the LAN read indexed documents.
    # Set BIND_HOST=0.0.0.0 (e.g. in Docker) to expose it deliberately.
    bind_host = os.getenv("BIND_HOST", "127.0.0.1")
    print(f"🚀 Starting Advanced RAG API server (FastAPI) on {bind_host}:{port}")
    print(f"💬 Chat endpoint: http://localhost:{port}/chat")
    print(f"✨ Indexing endpoint: http://localhost:{port}/index")
    uvicorn.run(app, host=bind_host, port=port, log_level="info")


if __name__ == "__main__":
    # To run this server: python -m rag_system.api_server
    start_server()

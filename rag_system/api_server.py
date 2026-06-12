import json
import http.server
import socketserver
import threading
from urllib.parse import urlparse, parse_qs
import os
import requests
import sys
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
_rag_agent_lock = threading.Lock()  # guards per-request mutations of RAG_AGENT shared state

if RAG_AGENT is None:
    raise RuntimeError("RAG Agent could not be initialized")
print("✅ RAG Agent initialized successfully.")

# Add helper near top after db & agent init
# -------------- Helper ----------------

def _apply_index_embedding_model(idx_ids):
    """Ensure retrieval pipeline uses the embedding model + fusion weights of
    the ACTIVE index — the same one whose vector table the backend queries
    (see rag_system.index_selection). Using a different index here embeds
    queries with the wrong model for the table being searched."""
    logger = logging.getLogger(__name__)
    logger.debug("apply_index_embedding_model idx_ids=%s", idx_ids)

    active_idx = select_active_index_id(idx_ids)
    if not active_idx:
        logger.warning("apply_index_embedding_model called without index IDs")
        return
    try:
        idx = db.get_index(active_idx)
        logger.debug(
            "apply_index_embedding_model index_id=%s metadata=%s",
            idx.get("id"),
            idx.get("metadata", {}),
        )
        meta = idx.get("metadata") or {}
        model = meta.get("embedding_model")
        logger.debug("apply_index_embedding_model metadata_embedding_model=%s", model)
        rp = RAG_AGENT.retrieval_pipeline
        if model:
            current_model = rp.config.get("embedding_model_name")
            rp.update_embedding_model(model)
            logger.debug(
                "apply_index_embedding_model updated_embedding_model previous=%s current=%s",
                current_model,
                model,
            )
        else:
            logger.warning("apply_index_embedding_model no embedding model in index metadata")
        # Apply per-index fusion weights if stored
        fusion_config = meta.get("fusion_config")
        if fusion_config and hasattr(rp, "retriever") and hasattr(rp.retriever, "fusion_config"):
            rp.retriever.fusion_config = fusion_config
            logger.debug("apply_index_embedding_model applied_fusion_config=%s", fusion_config)
    except Exception as e:
        logger.warning("apply_index_embedding_model failed: %s", e)

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
        
        # Use the ACTIVE index's vector table — must stay consistent with
        # the backend's choice and _apply_index_embedding_model
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


class AdvancedRagApiHandler(http.server.BaseHTTPRequestHandler):
    def _cors_origin(self) -> str | None:
        """Echo the request Origin only if it is in the configured allowlist.

        A wildcard here would let any website the user visits call this
        unauthenticated API from their browser.
        """
        allowed = _cors_allowed_origins()
        origin = self.headers.get('Origin')
        if origin and (origin in allowed or '*' in allowed):
            return origin
        return None

    def _send_cors_headers(self):
        origin = self._cors_origin()
        if origin:
            self.send_header('Access-Control-Allow-Origin', origin)
            self.send_header('Vary', 'Origin')

    def do_OPTIONS(self):
        """Handle CORS preflight requests for frontend integration."""
        self.send_response(200)
        self._send_cors_headers()
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        """Handle POST requests for chat and indexing."""
        parsed_path = urlparse(self.path)

        if parsed_path.path == '/chat':
            self.handle_chat()
        elif parsed_path.path == '/chat/stream':
            self.handle_chat_stream()
        elif parsed_path.path == '/index':
            self.handle_index()
        else:
            self.send_json_response({"error": "Not Found"}, status_code=404)

    def do_GET(self):
        parsed_path = urlparse(self.path)

        if parsed_path.path == '/models':
            self.handle_models()
        elif parsed_path.path == '/health':
            self.handle_health()
        else:
            self.send_json_response({"error": "Not Found"}, status_code=404)

    def handle_health(self):
        """Lightweight health probe that avoids loading large ML models."""
        checks: dict = {}
        overall = "ok"

        # Agent
        checks["agent"] = "ok" if RAG_AGENT is not None else "error"
        if RAG_AGENT is None:
            overall = "degraded"

        # LanceDB
        try:
            import lancedb as _lancedb
            lancedb_uri = os.getenv("LANCEDB_URI", "./lancedb")
            if os.path.exists(lancedb_uri):
                conn = _lancedb.connect(lancedb_uri)
                table_names = conn.table_names()
                checks["lancedb"] = f"ok ({len(table_names)} tables)"
            else:
                checks["lancedb"] = "not_initialized"
        except Exception as e:
            checks["lancedb"] = f"error: {e}"
            overall = "degraded"

        # Embedder readiness without forcing model initialization.
        try:
            if RAG_AGENT is not None:
                retrieval_pipeline = getattr(RAG_AGENT, "retrieval_pipeline", None)
                embedder = getattr(retrieval_pipeline, "text_embedder", None)
                checks["embedder"] = "loaded" if embedder is not None else "not_loaded"
            else:
                checks["embedder"] = "agent_unavailable"
        except Exception as e:
            checks["embedder"] = f"error: {e}"
            overall = "degraded"

        self.send_json_response({"status": overall, "checks": checks})

    def _parse_chat_request(self):
        """Parse and validate a chat POST body. Returns a params dict, or None if a response was already sent."""
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode('utf-8'))

        requested_model = data.get('model')

        query = data.get('query')
        if not query:
            self.send_json_response({"error": "Query is required"}, status_code=400)
            return None

        session_id = data.get('session_id')
        table_name = data.get('table_name')
        if not table_name and session_id:
            table_name = _get_table_name_for_session(session_id)

        return {
            "query": query,
            "model": requested_model if isinstance(requested_model, str) and requested_model else None,
            "session_id": session_id,
            "table_name": table_name,
            "compose_flag": data.get('compose_sub_answers'),
            "decomp_flag": data.get('query_decompose'),
            "ai_rerank_flag": data.get('ai_rerank'),
            "ctx_expand_flag": data.get('context_expand'),
            "verify_flag": data.get('verify'),
            "retrieval_k": data.get('retrieval_k', 20),
            "context_window_size": data.get('context_window_size', 1),
            "reranker_top_k": data.get('reranker_top_k', 10),
            "search_type": data.get('search_type', 'hybrid'),
            # No default: a value here overrides the per-index fusion config
            # stored with the index, so only explicit caller overrides count
            "dense_weight": data.get('dense_weight'),
            "force_rag": bool(data.get('force_rag', False)),
            "provence_prune": data.get('provence_prune'),
            "provence_threshold": data.get('provence_threshold'),
        }

    def handle_chat(self):
        """Handles a chat query by calling the agentic RAG pipeline."""
        try:
            params = self._parse_chat_request()
            if params is None:
                return

            query = params["query"]
            session_id = params["session_id"]
            table_name = params["table_name"]
            compose_flag = params["compose_flag"]
            decomp_flag = params["decomp_flag"]
            ai_rerank_flag = params["ai_rerank_flag"]
            ctx_expand_flag = params["ctx_expand_flag"]
            verify_flag = params["verify_flag"]
            retrieval_k = params["retrieval_k"]
            context_window_size = params["context_window_size"]
            reranker_top_k = params["reranker_top_k"]
            search_type = params["search_type"]
            dense_weight = params["dense_weight"]
            force_rag = params["force_rag"]
            provence_prune = params["provence_prune"]
            provence_threshold = params["provence_threshold"]

            # Decide execution path
            print(f"🔧 Force RAG flag: {force_rag}")
            # The shared agent/pipeline are mutated per request (generation
            # model, embedding model, fusion config, active table), so chat
            # execution must be serialized across handler threads — otherwise
            # concurrent requests leak settings into each other.
            with _rag_agent_lock:
                if params["model"]:
                    RAG_AGENT.ollama_config['generation_model'] = params["model"]
                if force_rag:
                    # --- Apply runtime overrides manually because we skip Agent.run()
                    rp_cfg = RAG_AGENT.retrieval_pipeline.config
                    if retrieval_k is not None:
                        rp_cfg["retrieval_k"] = retrieval_k
                    if reranker_top_k is not None:
                        rp_cfg.setdefault("reranker", {})["top_k"] = reranker_top_k
                    if search_type is not None:
                        rp_cfg.setdefault("retrieval", {})["search_type"] = search_type
                    # Set-or-clear: a stale weight from a previous request would
                    # override the index's stored fusion config
                    if dense_weight is not None:
                        rp_cfg.setdefault("retrieval", {}).setdefault("dense", {})["weight"] = dense_weight
                    else:
                        rp_cfg.setdefault("retrieval", {}).setdefault("dense", {}).pop("weight", None)
                    if ai_rerank_flag is not None:
                        # Let force_rag callers (e.g. the eval harness) toggle
                        # the AI reranker, like the agent path already can
                        rp_cfg.setdefault("reranker", {})["enabled"] = bool(ai_rerank_flag)

                    # Provence overrides
                    if provence_prune is not None:
                        rp_cfg.setdefault("provence", {})["enabled"] = bool(provence_prune)
                    if provence_threshold is not None:
                        rp_cfg.setdefault("provence", {})["threshold"] = float(provence_threshold)

                    # 🔄 Apply embedding model for this session (same as in agent path)
                    if session_id:
                        idx_ids = db.get_indexes_for_session(session_id)
                        _apply_index_embedding_model(idx_ids)

                    # Directly invoke retrieval pipeline to bypass triage
                    result = RAG_AGENT.retrieval_pipeline.run(
                        query,
                        table_name=table_name,
                        window_size_override=context_window_size,
                    )
                else:
                    # Use full agent with smart routing
                    # Apply Provence overrides even in agent path
                    rp_cfg = RAG_AGENT.retrieval_pipeline.config
                    if provence_prune is not None:
                        rp_cfg.setdefault("provence", {})["enabled"] = bool(provence_prune)
                    if provence_threshold is not None:
                        rp_cfg.setdefault("provence", {})["threshold"] = float(provence_threshold)

                    # 🔄 Refresh document overviews for this session
                    if session_id:
                        idx_ids = db.get_indexes_for_session(session_id)
                        _apply_index_embedding_model(idx_ids)
                        RAG_AGENT.load_overviews_for_indexes(idx_ids)

                    # 🔧 Set index-specific overview path
                    if session_id:
                        rp_cfg["overview_path"] = f"index_store/overviews/{session_id}.jsonl"

                    # 🔧 Configure late chunking
                    rp_cfg.setdefault("retrievers", {}).setdefault("latechunk", {})["enabled"] = True

                    result = RAG_AGENT.run(
                        query,
                        table_name=table_name,
                        session_id=session_id,
                        compose_sub_answers=compose_flag,
                        query_decompose=decomp_flag,
                        ai_rerank=ai_rerank_flag,
                        context_expand=ctx_expand_flag,
                        verify=verify_flag,
                        retrieval_k=retrieval_k,
                        context_window_size=context_window_size,
                        reranker_top_k=reranker_top_k,
                        search_type=search_type,
                        dense_weight=dense_weight,
                    )
            
            # The result is a dict, so we need to dump it to a JSON string
            self.send_json_response(result)

        except json.JSONDecodeError:
            self.send_json_response({"error": "Invalid JSON"}, status_code=400)
        except Exception as e:
            self.send_json_response({"error": f"Server error: {str(e)}"}, status_code=500)

    def handle_chat_stream(self):
        """Stream internal phases and final answer using SSE (text/event-stream)."""
        try:
            params = self._parse_chat_request()
            if params is None:
                return

            query = params["query"]
            session_id = params["session_id"]
            table_name = params["table_name"]
            compose_flag = params["compose_flag"]
            decomp_flag = params["decomp_flag"]
            ai_rerank_flag = params["ai_rerank_flag"]
            ctx_expand_flag = params["ctx_expand_flag"]
            verify_flag = params["verify_flag"]
            retrieval_k = params["retrieval_k"]
            context_window_size = params["context_window_size"]
            reranker_top_k = params["reranker_top_k"]
            search_type = params["search_type"]
            dense_weight = params["dense_weight"]
            force_rag = params["force_rag"]
            provence_prune = params["provence_prune"]
            provence_threshold = params["provence_threshold"]

            # Prepare response headers for SSE
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            # Keep connection alive for SSE; no manual chunked encoding (Python http.server
            # does not add chunk sizes automatically, so declaring it breaks clients).
            self.send_header('Connection', 'keep-alive')
            self._send_cors_headers()
            self.end_headers()

            def emit(event_type: str, payload):
                """Send a single SSE event."""
                try:
                    data_str = json.dumps({"type": event_type, "data": payload})
                    self.wfile.write(f"data: {data_str}\n\n".encode('utf-8'))
                    self.wfile.flush()
                except BrokenPipeError:
                    # Client disconnected
                    raise

            # Run the agent synchronously, emitting checkpoints.
            # Serialized on _rag_agent_lock: the shared agent/pipeline are
            # mutated per request, so concurrent chats would corrupt each
            # other's settings (embedding model, active table, weights).
            try:
                with _rag_agent_lock:
                    if params["model"]:
                        RAG_AGENT.ollama_config['generation_model'] = params["model"]
                    if force_rag:
                        # Apply overrides same as above since we bypass Agent.run
                        rp_cfg = RAG_AGENT.retrieval_pipeline.config
                        if retrieval_k is not None:
                            rp_cfg["retrieval_k"] = retrieval_k
                        if reranker_top_k is not None:
                            rp_cfg.setdefault("reranker", {})["top_k"] = reranker_top_k
                        if search_type is not None:
                            rp_cfg.setdefault("retrieval", {})["search_type"] = search_type
                        if dense_weight is not None:
                            rp_cfg.setdefault("retrieval", {}).setdefault("dense", {})["weight"] = dense_weight
                        else:
                            rp_cfg.setdefault("retrieval", {}).setdefault("dense", {}).pop("weight", None)
                        if ai_rerank_flag is not None:
                            rp_cfg.setdefault("reranker", {})["enabled"] = bool(ai_rerank_flag)

                        # Provence overrides
                        if provence_prune is not None:
                            rp_cfg.setdefault("provence", {})["enabled"] = bool(provence_prune)
                        if provence_threshold is not None:
                            rp_cfg.setdefault("provence", {})["threshold"] = float(provence_threshold)

                        # 🔄 Apply embedding model for this session (same as in agent path)
                        if session_id:
                            idx_ids = db.get_indexes_for_session(session_id)
                            _apply_index_embedding_model(idx_ids)

                        # 🔧 Set index-specific overview path so each index writes separate file
                        if session_id:
                            rp_cfg["overview_path"] = f"index_store/overviews/{session_id}.jsonl"

                        # 🔧 Configure late chunking
                        rp_cfg.setdefault("retrievers", {}).setdefault("latechunk", {})["enabled"] = True

                        # Straight retrieval pipeline with streaming events
                        final_result = RAG_AGENT.retrieval_pipeline.run(
                            query,
                            table_name=table_name,
                            window_size_override=context_window_size,
                            event_callback=emit,
                        )
                    else:
                        # Provence overrides
                        rp_cfg = RAG_AGENT.retrieval_pipeline.config
                        if provence_prune is not None:
                            rp_cfg.setdefault("provence", {})["enabled"] = bool(provence_prune)
                        if provence_threshold is not None:
                            rp_cfg.setdefault("provence", {})["threshold"] = float(provence_threshold)

                        # 🔄 Refresh overviews for this session
                        if session_id:
                            idx_ids = db.get_indexes_for_session(session_id)
                            _apply_index_embedding_model(idx_ids)
                            RAG_AGENT.load_overviews_for_indexes(idx_ids)

                        # 🔧 Set index-specific overview path
                        if session_id:
                            rp_cfg["overview_path"] = f"index_store/overviews/{session_id}.jsonl"

                        # 🔧 Configure late chunking
                        rp_cfg.setdefault("retrievers", {}).setdefault("latechunk", {})["enabled"] = True

                        final_result = RAG_AGENT.run(
                            query,
                            table_name=table_name,
                            session_id=session_id,
                            compose_sub_answers=compose_flag,
                            query_decompose=decomp_flag,
                            ai_rerank=ai_rerank_flag,
                            context_expand=ctx_expand_flag,
                            verify=verify_flag,
                            # ✨ NEW RETRIEVAL PARAMETERS
                            retrieval_k=retrieval_k,
                            context_window_size=context_window_size,
                            reranker_top_k=reranker_top_k,
                            search_type=search_type,
                            dense_weight=dense_weight,
                            event_callback=emit,
                        )

                # Ensure the final answer is sent (in case callback missed it)
                emit("complete", final_result)
            except BrokenPipeError:
                print("🔌 Client disconnected from SSE stream.")
            except Exception as e:
                # Send error event then close
                error_payload = {"error": str(e)}
                try:
                    emit("error", error_payload)
                finally:
                    print(f"❌ Stream error: {e}")

        except json.JSONDecodeError:
            self.send_json_response({"error": "Invalid JSON"}, status_code=400)
        except Exception as e:
            self.send_json_response({"error": f"Server error: {str(e)}"}, status_code=500)

    def handle_index(self):
        """Triggers the document indexing pipeline for specific files."""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            file_paths = data.get('file_paths')
            session_id = data.get('session_id')
            compose_flag = data.get('compose_sub_answers')
            decomp_flag = data.get('query_decompose')
            ai_rerank_flag = data.get('ai_rerank')
            ctx_expand_flag = data.get('context_expand')
            enable_latechunk = bool(data.get("enable_latechunk", False))
            enable_docling_chunk = bool(data.get("enable_docling_chunk", False))
            
            # 🆕 NEW CONFIGURATION OPTIONS:
            chunk_size = int(data.get("chunk_size", 512))
            chunk_overlap = int(data.get("chunk_overlap", 64))
            retrieval_mode = data.get("retrieval_mode", "hybrid")
            window_size = int(data.get("window_size", 2))
            # Default OFF: enrichment is one LLM call per chunk — opt-in only
            enable_enrich = bool(data.get("enable_enrich", False))
            embedding_model = data.get('embedding_model') or data.get('embeddingModel')
            enrich_model = data.get('enrich_model') or data.get('enrichModel')
            overview_model = data.get('overviewModel') or data.get('overview_model_name')
            enrich_provider = data.get('enrich_provider', 'ollama')
            enrich_api_key = data.get('enrich_api_key')  # never logged or stored
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

            # Guard only applies to Ollama local models; cloud providers manage their own limits
            if enrich_provider == 'ollama' and is_large_indexing_model(enrich_model):
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
                self.send_json_response({
                    "error": "A 'file_paths' list is required."
                }, status_code=400)
                return
            indexing_result = None
            # Progress reporting and cancellation polling live in
            # rag_system.indexing_worker — the build runs in a child process.

            # Allow explicit table_name override
            table_name = data.get('table_name')
            if not table_name and session_id:
                table_name = _get_table_name_for_session(session_id)

            # The INDEXING_PIPELINE is already initialized. We just need to use it.
            # If a session-specific table is needed, we can override the config for this run.
            if table_name:
                import copy
                config_override = copy.deepcopy(INDEXING_PIPELINE.config)
                config_override["storage"]["text_table_name"] = table_name
                config_override.setdefault("retrievers", {}).setdefault("dense", {})["lancedb_table_name"] = table_name
                
                # 🔧 Configure late chunking
                if enable_latechunk:
                    config_override["retrievers"].setdefault("latechunk", {})["enabled"] = True
                else:
                    # ensure disabled if not requested
                    config_override["retrievers"].setdefault("latechunk", {})["enabled"] = False
                
                # 🔧 Configure docling chunking
                if enable_docling_chunk:
                    config_override["chunker_mode"] = "docling"
                
                # 🔧 Configure contextual enrichment (THIS WAS MISSING!)
                config_override.setdefault("contextual_enricher", {})
                config_override["contextual_enricher"]["enabled"] = enable_enrich
                config_override["contextual_enricher"]["window_size"] = window_size
                
                # 🔧 Configure indexing batch sizes
                config_override.setdefault("indexing", {})
                config_override["indexing"]["embedding_batch_size"] = batch_size_embed
                config_override["indexing"]["enrichment_batch_size"] = batch_size_enrich
                # 900s: a 500-page PDF legitimately needs >180s for layout analysis;
                # a timeout also kills the conversion worker, so the next file pays
                # a full Docling reload on top of the failure
                config_override["indexing"].setdefault("conversion_timeout_seconds", int(os.getenv("CONVERSION_TIMEOUT_SECONDS", "900")))
                config_override["indexing"].setdefault("overview_timeout_seconds", 45)
                config_override["indexing"].setdefault("enrichment_timeout_seconds", 60)
                
                # 🔧 Configure chunking parameters
                config_override.setdefault("chunking", {})
                config_override["chunking"]["chunk_size"] = chunk_size
                config_override["chunking"]["chunk_overlap"] = chunk_overlap
                
                # 🔧 Configure embedding model if specified
                if embedding_model:
                    config_override["embedding_model_name"] = embedding_model
                
                # 🔧 Configure enrichment model and provider if specified
                if enrich_model:
                    config_override["enrich_model"] = enrich_model
                if enrich_provider and enrich_provider != 'ollama':
                    config_override["enrich_provider"] = enrich_provider
                    if enrich_api_key:
                        config_override["enrich_api_key"] = enrich_api_key
                
                # 🔧 Overview model (can differ from enrichment)
                if overview_model:
                    config_override["overview_model_name"] = overview_model
                
                print(f"🔧 INDEXING CONFIG: Contextual Enrichment: {enable_enrich}, Window Size: {window_size}")
                print(f"🔧 CHUNKING CONFIG: Size: {chunk_size}, Overlap: {chunk_overlap}")
                print(f"🔧 MODEL CONFIG: Embedding: {embedding_model or 'default'}, Enrichment: {enrich_model or 'default'}")
                
                # 🔧 Set index-specific overview path so each index writes separate file
                if session_id:
                    config_override["overview_path"] = f"index_store/overviews/{session_id}.jsonl"

                config_override.setdefault("retrievers", {}).setdefault("latechunk", {})["enabled"] = enable_latechunk

                if force_reindex:
                    self._clear_index_artifacts(INDEXING_PIPELINE, table_name, session_id)
                # Run the build in an isolated child process (memory/crash isolation)
                indexing_result = _execute_indexing_job(
                    config_override,
                    file_paths,
                    index_id=session_id or table_name or "default",
                    force_reindex=force_reindex,
                    job_id=job_id,
                    backend_base_url=backend_base_url,
                )
            else:
                # Use the default pipeline with overrides
                import copy
                config_override = copy.deepcopy(INDEXING_PIPELINE.config)
                
                # 🔧 Configure late chunking
                if enable_latechunk:
                    config_override.setdefault("retrievers", {}).setdefault("latechunk", {})["enabled"] = True
                
                # 🔧 Configure docling chunking
                if enable_docling_chunk:
                    config_override["chunker_mode"] = "docling"
                
                # 🔧 Configure contextual enrichment (THIS WAS MISSING!)
                config_override.setdefault("contextual_enricher", {})
                config_override["contextual_enricher"]["enabled"] = enable_enrich
                config_override["contextual_enricher"]["window_size"] = window_size
                
                # 🔧 Configure indexing batch sizes
                config_override.setdefault("indexing", {})
                config_override["indexing"]["embedding_batch_size"] = batch_size_embed
                config_override["indexing"]["enrichment_batch_size"] = batch_size_enrich
                # 900s: a 500-page PDF legitimately needs >180s for layout analysis;
                # a timeout also kills the conversion worker, so the next file pays
                # a full Docling reload on top of the failure
                config_override["indexing"].setdefault("conversion_timeout_seconds", int(os.getenv("CONVERSION_TIMEOUT_SECONDS", "900")))
                config_override["indexing"].setdefault("overview_timeout_seconds", 45)
                config_override["indexing"].setdefault("enrichment_timeout_seconds", 60)
                
                # 🔧 Configure chunking parameters
                config_override.setdefault("chunking", {})
                config_override["chunking"]["chunk_size"] = chunk_size
                config_override["chunking"]["chunk_overlap"] = chunk_overlap
                
                # 🔧 Configure embedding model if specified
                if embedding_model:
                    config_override["embedding_model_name"] = embedding_model
                
                # 🔧 Configure enrichment model and provider if specified
                if enrich_model:
                    config_override["enrich_model"] = enrich_model
                if enrich_provider and enrich_provider != 'ollama':
                    config_override["enrich_provider"] = enrich_provider
                    if enrich_api_key:
                        config_override["enrich_api_key"] = enrich_api_key
                
                # 🔧 Overview model (can differ from enrichment)
                if overview_model:
                    config_override["overview_model_name"] = overview_model
                
                print(f"🔧 INDEXING CONFIG: Contextual Enrichment: {enable_enrich}, Window Size: {window_size}")
                print(f"🔧 CHUNKING CONFIG: Size: {chunk_size}, Overlap: {chunk_overlap}")
                print(f"🔧 MODEL CONFIG: Embedding: {embedding_model or 'default'}, Enrichment: {enrich_model or 'default'}")
                
                # 🔧 Set index-specific overview path so each index writes separate file
                if session_id:
                    config_override["overview_path"] = f"index_store/overviews/{session_id}.jsonl"

                config_override.setdefault("retrievers", {}).setdefault("latechunk", {})["enabled"] = enable_latechunk

                if force_reindex:
                    self._clear_index_artifacts(INDEXING_PIPELINE, table_name, session_id)
                # Run the build in an isolated child process (memory/crash isolation)
                indexing_result = _execute_indexing_job(
                    config_override,
                    file_paths,
                    index_id=session_id or table_name or "default",
                    force_reindex=force_reindex,
                    job_id=job_id,
                    backend_base_url=backend_base_url,
                )

            self.send_json_response({
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
            })

            if embedding_model:
                try:
                    db.update_index_metadata(session_id, {"embedding_model": embedding_model})
                except Exception as e:
                    print(f"⚠️ Could not update embedding_model metadata: {e}")

        except json.JSONDecodeError:
            self.send_json_response({"error": "Invalid JSON"}, status_code=400)
        except RuntimeError as e:
            if str(e) == "indexing_cancelled":
                self.send_json_response({"error": "Indexing cancelled"}, status_code=499)
            else:
                self.send_json_response({"error": f"Failed to start indexing: {str(e)}"}, status_code=500)
        except Exception as e:
            self.send_json_response({"error": f"Failed to start indexing: {str(e)}"}, status_code=500)

    def _clear_index_artifacts(self, pipeline, table_name: str | None, index_id: str | None):
        """Remove old vector/overview artifacts before a force rebuild."""
        if table_name and hasattr(pipeline, "lancedb_manager"):
            db = pipeline.lancedb_manager.db
            table_names = db.table_names() if hasattr(db, "table_names") else []
            for candidate in (table_name, f"{table_name}_lc"):
                if candidate in table_names:
                    try:
                        db.drop_table(candidate)
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

    def handle_models(self):
        """Return a list of locally installed Ollama models and supported HuggingFace models, grouped by capability."""
        try:
            generation_models = []
            embedding_models = []
            
            # Get Ollama models if available
            try:
                resp = requests.get(f"{RAG_AGENT.ollama_config['host']}/api/tags", timeout=5)
                resp.raise_for_status()
                data = resp.json()

                all_ollama_models = [m.get('name') for m in data.get('models', [])]

                # Very naive classification
                ollama_embedding_models = [m for m in all_ollama_models if any(k in m for k in ['embed','bge','embedding','text'])]
                ollama_generation_models = [m for m in all_ollama_models if m not in ollama_embedding_models]
                
                generation_models.extend(ollama_generation_models)
                embedding_models.extend(ollama_embedding_models)
            except Exception as e:
                print(f"⚠️ Could not get Ollama models: {e}")
            
            # Add supported HuggingFace embedding models from registry
            try:
                from rag_system.model_registry import huggingface_models
                embedding_models.extend(huggingface_models())
            except ImportError:
                embedding_models.extend(["Qwen/Qwen3-Embedding-0.6B"])
            
            # Sort models for consistent ordering
            generation_models.sort()
            embedding_models.sort()

            self.send_json_response({
                "generation_models": generation_models,
                "embedding_models": embedding_models
            })
        except Exception as e:
            self.send_json_response({"error": f"Could not list models: {e}"}, status_code=500)

    def send_json_response(self, data, status_code=200):
        """Utility to send a JSON response with CORS headers."""
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self._send_cors_headers()
        self.end_headers()
        response = json.dumps(data, indent=2)
        self.wfile.write(response.encode('utf-8'))

def start_server(port=8001):
    """Starts the API server."""
    # Use a reusable TCP server to avoid "address in use" errors on restart
    class ReusableTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
        allow_reuse_address = True
        daemon_threads = True

    # Loopback by default: this API is unauthenticated, so exposing it on all
    # interfaces would let anyone on the LAN read indexed documents.
    # Set BIND_HOST=0.0.0.0 (e.g. in Docker) to expose it deliberately.
    bind_host = os.getenv("BIND_HOST", "127.0.0.1")
    with ReusableTCPServer((bind_host, port), AdvancedRagApiHandler) as httpd:
        print(f"🚀 Starting Advanced RAG API server on {bind_host}:{port}")
        print(f"💬 Chat endpoint: http://localhost:{port}/chat")
        print(f"✨ Indexing endpoint: http://localhost:{port}/index")
        httpd.serve_forever()

if __name__ == "__main__":
    # To run this server: python -m rag_system.api_server
    start_server() 

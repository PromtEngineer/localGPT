import json
import http.server
import socketserver
from urllib.parse import urlparse
import os
import requests
import logging
from threading import RLock

from backend.database import ChatDatabase
from localgpt_runtime import (
    UploadRejected,
    cors_origin,
    env_path,
    normalize_index_options,
    request_is_authorized,
    validate_index_file_paths,
)
from rag_system.main import get_agent
from rag_system.factory import get_indexing_pipeline

# Initialize database connection once at module level
# Use auto-detection for environment-appropriate path
db = ChatDatabase()

# Get the desired agent mode from environment variables, defaulting to 'default'
# This allows us to easily switch between 'default', 'fast', 'react', etc.
AGENT_MODE = os.getenv("RAG_CONFIG_MODE", "default")
RAG_AGENT = get_agent(AGENT_MODE)
INDEXING_PIPELINE = get_indexing_pipeline(AGENT_MODE)
RAG_AGENT_LOCK = RLock()
INDEXING_LOCK = RLock()

# --- Global Singleton for the RAG Agent ---
# The agent is initialized once when the server starts.
# This avoids reloading all the models on every request.
print("🧠 Initializing RAG Agent with MAXIMUM ACCURACY... (This may take a moment)")
if RAG_AGENT is None:
    print("❌ Critical error: RAG Agent could not be initialized. Exiting.")
    exit(1)
print("✅ RAG Agent initialized successfully with MAXIMUM ACCURACY.")
# ---

# Add helper near top after db & agent init
# -------------- Helper ----------------

def _apply_index_embedding_model(idx_ids):
    """Ensure retrieval pipeline uses the embedding model stored with the first index."""
    if not idx_ids:
        return
    try:
        idx = db.get_index(idx_ids[0])
        if not idx:
            return
        model = (idx.get("metadata") or {}).get("embedding_model")
        if model:
            rp = RAG_AGENT.retrieval_pipeline
            rp.update_embedding_model(model)
    except Exception as e:
        logging.getLogger(__name__).warning(
            "Could not apply index embedding model: %s", e
        )

def _get_table_names_for_session(session_id):
    """Get every vector table linked to a session."""
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
            return []

        table_names = []
        for index_id in idx_ids:
            index = db.get_index(index_id)
            if index and index.get("vector_table_name"):
                table_names.append(index["vector_table_name"])
        if table_names:
            logger.info("Using %s linked vector tables for session %s", len(table_names), session_id[:8])
            return table_names
        else:
            logger.warning(f"⚠️ Index found but no vector table name for session {session_id}")
            return []

    except Exception as e:
        logger.error(f"❌ Error getting table name for session {session_id}: {e}")
        return []


def _restore_conversation_history(session_id, messages):
    """Restore persisted user/assistant turns into the agent's history cache."""
    if not session_id or not isinstance(messages, list):
        return
    turns = []
    pending_query = None
    for message in messages:
        role = message.get("role") or message.get("sender")
        if role == "user":
            pending_query = str(message.get("content", ""))
        elif role == "assistant" and pending_query is not None:
            turns.append(
                {"query": pending_query, "answer": str(message.get("content", ""))}
            )
            pending_query = None
    RAG_AGENT.chat_histories[session_id] = turns

class AdvancedRagApiHandler(http.server.BaseHTTPRequestHandler):
    def _authorized(self) -> bool:
        if request_is_authorized(self.headers.get("Authorization")):
            return True
        self.send_json_response({"error": "Unauthorized"}, status_code=401)
        return False

    def _send_cors(self) -> None:
        origin = cors_origin(self.headers.get("Origin"))
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")

    def do_OPTIONS(self):
        """Handle CORS preflight requests for frontend integration."""
        origin = cors_origin(self.headers.get("Origin"))
        if not origin:
            self.send_response(403)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', origin)
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, DELETE, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.end_headers()

    def do_POST(self):
        """Handle POST requests for chat and indexing."""
        if not self._authorized():
            return
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
        if not self._authorized():
            return
        parsed_path = urlparse(self.path)

        if parsed_path.path == '/health':
            self.send_json_response({"status": "ok"})
        elif parsed_path.path == '/models':
            self.handle_models()
        else:
            self.send_json_response({"error": "Not Found"}, status_code=404)

    def do_DELETE(self):
        if not self._authorized():
            return
        parsed_path = urlparse(self.path)
        if parsed_path.path.startswith("/indexes/") and parsed_path.path.count("/") == 2:
            self.handle_delete_index_artifacts(parsed_path.path.rsplit("/", 1)[-1])
        else:
            self.send_json_response({"error": "Not Found"}, status_code=404)

    def handle_delete_index_artifacts(self, index_id: str):
        """Delete RAG-owned tables and overview data for one database index."""
        index = db.get_index(index_id)
        if not index:
            self.send_json_response({"error": "Index not found"}, status_code=404)
            return
        table_name = index.get("vector_table_name")
        if not table_name:
            self.send_json_response({"error": "Index has no vector table"}, status_code=409)
            return

        try:
            import lancedb

            vector_db = lancedb.connect(
                str(env_path("LANCEDB_PATH", os.path.join(os.getcwd(), "lancedb")))
            )
            existing = set(vector_db.table_names())
            dropped = []
            for candidate in (table_name, f"{table_name}_lc"):
                if candidate in existing:
                    vector_db.drop_table(candidate)
                    dropped.append(candidate)

            overview_dir = env_path(
                "LOCALGPT_OVERVIEW_DIR",
                os.path.join(os.getcwd(), "index_store", "overviews"),
            )
            overview_path = overview_dir / f"{index_id}.jsonl"
            overview_path.unlink(missing_ok=True)
            RAG_AGENT.doc_overviews = []
            RAG_AGENT._current_overview_session = None
            self.send_json_response(
                {
                    "index_id": index_id,
                    "dropped_tables": dropped,
                    "overview_deleted": not overview_path.exists(),
                }
            )
        except Exception as exc:
            self.send_json_response({"error": str(exc)}, status_code=500)

    def handle_chat(self):
        with RAG_AGENT_LOCK:
            return self._handle_chat_locked()

    def _handle_chat_locked(self):
        """Handles a chat query by calling the agentic RAG pipeline."""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))

            query = data.get('query')
            session_id = data.get('session_id')
            compose_flag = data.get('compose_sub_answers')
            decomp_flag = data.get('query_decompose')
            ai_rerank_flag = data.get('ai_rerank')
            ctx_expand_flag = data.get('context_expand')
            verify_flag = data.get('verify')

            # ✨ NEW RETRIEVAL PARAMETERS
            retrieval_k = data.get('retrieval_k', 20)
            context_window_size = data.get('context_window_size', 1)
            reranker_top_k = data.get('reranker_top_k', 10)
            search_type = data.get('search_type', 'hybrid')
            dense_weight = data.get('dense_weight', 0.7)

            # 🚩 NEW: Force RAG override from frontend
            force_rag = bool(data.get('force_rag', False))

            # 🌿 Provence sentence pruning
            provence_prune = data.get('provence_prune')
            provence_threshold = data.get('provence_threshold')

            # User-selected generation model
            requested_model = data.get('model')
            if isinstance(requested_model,str) and requested_model:
                RAG_AGENT.ollama_config['generation_model']=requested_model

            if not query:
                self.send_json_response({"error": "Query is required"}, status_code=400)
                return
            _restore_conversation_history(session_id, data.get("conversation_history"))

            # Allow explicit table_name override
            table_name = (
                _get_table_names_for_session(session_id)
                if session_id
                else data.get('table_names') or data.get('table_name')
            )

            # Decide execution path
            print(f"🔧 Force RAG flag: {force_rag}")
            if force_rag:
                # --- Apply runtime overrides manually because we skip Agent.run()
                rp_cfg = RAG_AGENT.retrieval_pipeline.config
                if retrieval_k is not None:
                    rp_cfg["retrieval_k"] = retrieval_k
                if reranker_top_k is not None:
                    rp_cfg.setdefault("reranker", {})["top_k"] = reranker_top_k
                if search_type is not None:
                    rp_cfg.setdefault("retrieval", {})["search_type"] = search_type
                if dense_weight is not None:
                    rp_cfg.setdefault("retrieval", {}).setdefault("dense", {})["weight"] = dense_weight

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
        with RAG_AGENT_LOCK:
            return self._handle_chat_stream_locked()

    def _handle_chat_stream_locked(self):
        """Stream internal phases and final answer using SSE (text/event-stream)."""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))

            query = data.get('query')
            session_id = data.get('session_id')
            compose_flag = data.get('compose_sub_answers')
            decomp_flag = data.get('query_decompose')
            ai_rerank_flag = data.get('ai_rerank')
            ctx_expand_flag = data.get('context_expand')
            verify_flag = data.get('verify')

            # ✨ NEW RETRIEVAL PARAMETERS
            retrieval_k = data.get('retrieval_k', 20)
            context_window_size = data.get('context_window_size', 1)
            reranker_top_k = data.get('reranker_top_k', 10)
            search_type = data.get('search_type', 'hybrid')
            dense_weight = data.get('dense_weight', 0.7)

            # 🚩 NEW: Force RAG override from frontend
            force_rag = bool(data.get('force_rag', False))

            # 🌿 Provence sentence pruning
            provence_prune = data.get('provence_prune')
            provence_threshold = data.get('provence_threshold')

            # User-selected generation model
            requested_model = data.get('model')
            if isinstance(requested_model,str) and requested_model:
                RAG_AGENT.ollama_config['generation_model']=requested_model

            if not query:
                self.send_json_response({"error": "Query is required"}, status_code=400)
                return
            _restore_conversation_history(session_id, data.get("conversation_history"))

            # Allow explicit table_name override
            table_name = (
                _get_table_names_for_session(session_id)
                if session_id
                else data.get('table_names') or data.get('table_name')
            )

            # Prepare response headers for SSE
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            # Keep connection alive for SSE; no manual chunked encoding (Python http.server
            # does not add chunk sizes automatically, so declaring it breaks clients).
            self.send_header('Connection', 'keep-alive')
            self._send_cors()
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

            # Run the agent synchronously, emitting checkpoints
            try:
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
        with INDEXING_LOCK:
            return self._handle_index_locked()

    def _handle_index_locked(self):
        """Build or rebuild one validated index from canonical API options."""
        index_id = None
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(content_length).decode("utf-8"))
            options = normalize_index_options(data)
            raw_paths = options.get("file_paths")
            if not isinstance(raw_paths, list) or not raw_paths:
                raise ValueError("A non-empty 'file_paths' list is required")

            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            upload_dir = env_path(
                "LOCALGPT_UPLOAD_DIR", os.path.join(project_root, "shared_uploads")
            )
            file_paths = validate_index_file_paths(raw_paths, upload_dir)

            import copy

            config = copy.deepcopy(INDEXING_PIPELINE.config)
            table_name = options.get("table_name") or config["storage"].get(
                "text_table_name", "default_text_table"
            )
            config["storage"]["text_table_name"] = table_name
            retrieval = config.setdefault("retrieval", {})
            retrieval["search_type"] = options["retrieval_mode"]
            # LanceDB stores text and vectors in one table; embeddings are still
            # built for lexical-only indexes so the retrieval mode can later be
            # changed without rebuilding the schema.
            retrieval.setdefault("dense", {})["enabled"] = True

            retrievers = config.setdefault("retrievers", {})
            retrievers.setdefault("dense", {})["lancedb_table_name"] = table_name
            latechunk = retrievers.setdefault("latechunk", {})
            latechunk["enabled"] = bool(options["enable_latechunk"])
            latechunk["lancedb_table_name"] = f"{table_name}_lc"

            config["chunker_mode"] = (
                "docling" if options["enable_docling_chunk"] else "legacy"
            )
            config["chunking"] = {
                "chunk_size": options["chunk_size"],
                "chunk_overlap": options["chunk_overlap"],
            }
            config.setdefault("contextual_enricher", {}).update(
                {
                    "enabled": bool(options["enable_enrich"]),
                    "window_size": int(options["window_size"]),
                }
            )
            config.setdefault("indexing", {}).update(
                {
                    "embedding_batch_size": int(options["batch_size_embed"]),
                    "enrichment_batch_size": int(options["batch_size_enrich"]),
                }
            )
            if options.get("embedding_model"):
                config["embedding_model_name"] = options["embedding_model"]
            if options.get("enrich_model"):
                config["enrich_model"] = options["enrich_model"]
            if options.get("overview_model"):
                config["overview_model_name"] = options["overview_model"]

            index_id = options.get("session_id")
            overview_dir = env_path(
                "LOCALGPT_OVERVIEW_DIR", os.path.join(project_root, "index_store", "overviews")
            )
            config["overview_path"] = str(
                overview_dir / f"{index_id or table_name}.jsonl"
            )

            if index_id and db.get_index(index_id):
                db.update_index_metadata(index_id, {"status": "building"})

            pipeline = INDEXING_PIPELINE.__class__(
                config, INDEXING_PIPELINE.llm_client, INDEXING_PIPELINE.ollama_config
            )
            stats = pipeline.run(file_paths)
            metadata = {
                "status": "complete",
                "error": None,
                "chunk_size": options["chunk_size"],
                "chunk_overlap": options["chunk_overlap"],
                "retrieval_mode": options["retrieval_mode"],
                "window_size": options["window_size"],
                "enable_enrich": bool(options["enable_enrich"]),
                "embedding_model": config.get("embedding_model_name"),
                "enrich_model": config.get("enrich_model"),
                "overview_model": config.get("overview_model_name"),
                "latechunk": bool(options["enable_latechunk"]),
                "docling_chunk": bool(options["enable_docling_chunk"]),
                **(stats or {}),
            }
            if index_id and db.get_index(index_id):
                db.update_index_metadata(index_id, metadata)
            RAG_AGENT._query_cache.clear()
            response_status = 422 if metadata.get("status") == "empty" else 200
            self.send_json_response(
                {"message": "Index build completed", **metadata},
                status_code=response_status,
            )
        except (UploadRejected, ValueError, TypeError, json.JSONDecodeError) as exc:
            if index_id and db.get_index(index_id):
                db.update_index_metadata(index_id, {"status": "failed", "error": str(exc)})
            self.send_json_response({"error": str(exc)}, status_code=400)
        except Exception as exc:
            if index_id and db.get_index(index_id):
                db.update_index_metadata(index_id, {"status": "failed", "error": str(exc)})
            self.send_json_response({"error": f"Index build failed: {exc}"}, status_code=500)


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
        self._send_cors()
        self.end_headers()
        response = json.dumps(data, indent=2)
        self.wfile.write(response.encode('utf-8'))

def start_server(port=8001):
    """Starts the API server."""
    # Use a reusable TCP server to avoid "address in use" errors on restart
    class ReusableTCPServer(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    host = os.environ.get("LOCALGPT_RAG_HOST", "127.0.0.1")
    with ReusableTCPServer((host, port), AdvancedRagApiHandler) as httpd:
        print(f"🚀 Starting Advanced RAG API server on {host}:{port}")
        httpd.serve_forever()

if __name__ == "__main__":
    # To run this server: python -m rag_system.api_server
    start_server()

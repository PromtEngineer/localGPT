import copy
import json
import http.server
import socketserver
from contextlib import contextmanager
from urllib.parse import urlparse
import os
import re
import requests
import sys
import logging

# Add backend directory to path for database imports
backend_dir = os.path.join(os.path.dirname(__file__), '..', 'backend')
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

from backend.database import ChatDatabase
from rag_system.factory import get_agent, get_indexing_pipeline
from rag_system.main import LLM_BACKEND, PIPELINE_CONFIGS, WATSONX_CONFIG

logger = logging.getLogger(__name__)

# The RAG API reads/writes index metadata only. Chat message rows are owned
# exclusively by backend/server.py.
db = ChatDatabase()

# Get the desired agent mode from environment variables, defaulting to 'default'
AGENT_MODE = os.getenv("RAG_CONFIG_MODE", "default")

# --- Global Singletons ---
# The agent and indexing pipeline are initialized once when the server starts so
# that models are not reloaded on every request.
print("🧠 Initializing RAG Agent... (This may take a moment)")
RAG_AGENT = get_agent(AGENT_MODE)
INDEXING_PIPELINE = get_indexing_pipeline(AGENT_MODE)
print("✅ RAG Agent initialized successfully.")

DEFAULT_TEXT_TABLE = PIPELINE_CONFIGS.get(AGENT_MODE, PIPELINE_CONFIGS["default"])["storage"]["text_table_name"]
SUPPORTED_RETRIEVAL_MODES = ("hybrid", "vector_only", "fts_only")
OLLAMA_TIMEOUT_SECONDS = 5

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

# Wire aliases that are not a plain camelCase/snake_case pair.
_KEY_ALIASES = {
    "overview_model": "overview_model_name",
    "latechunk": "enable_latechunk",
    "docling_chunk": "enable_docling_chunk",
    "decompose": "query_decompose",
}


def normalize_request_keys(data):
    """Accept camelCase and snake_case spellings of every option.

    The frontend historically sent camelCase while the backend gateway sends
    snake_case. Both land in the same canonical snake_case key here, once, at
    parse time. Explicit snake_case values always win over a camelCase twin.
    """
    if not isinstance(data, dict):
        return data

    normalized = {}
    camel_derived = {}
    for key, value in data.items():
        if not isinstance(key, str):
            normalized[key] = value
            continue
        canonical = _CAMEL_BOUNDARY.sub("_", key).lower()
        canonical = _KEY_ALIASES.get(canonical, canonical)
        if canonical == key:
            normalized[key] = value
        else:
            camel_derived[canonical] = value

    for key, value in camel_derived.items():
        normalized.setdefault(key, value)
    return normalized


def _read_json_body(handler):
    """Read and normalize a JSON request body. Returns an empty dict on empty body."""
    content_length = int(handler.headers.get('Content-Length') or 0)
    if content_length <= 0:
        return {}
    post_data = handler.rfile.read(content_length)
    return normalize_request_keys(json.loads(post_data.decode('utf-8')))


def _model_valid_for_backend(model_name: str) -> bool:
    """A watsonx deployment cannot serve an Ollama model id (and vice versa)."""
    if LLM_BACKEND.lower() == "watsonx":
        return "/" in model_name
    return "/" not in model_name


@contextmanager
def _generation_model_override(requested_model):
    """Apply a per-request generation model without permanently mutating the singleton."""
    applied = False
    previous = RAG_AGENT.ollama_config.get("generation_model")
    if isinstance(requested_model, str) and requested_model:
        if _model_valid_for_backend(requested_model):
            RAG_AGENT.ollama_config["generation_model"] = requested_model
            applied = True
        else:
            logger.warning(
                "Ignoring requested model '%s': not valid for the '%s' backend (using '%s').",
                requested_model, LLM_BACKEND, previous,
            )
    try:
        yield
    finally:
        if applied:
            RAG_AGENT.ollama_config["generation_model"] = previous


def _apply_index_embedding_model(idx_ids):
    """Ensure the retrieval pipeline uses the embedding model stored with the first index."""
    if not idx_ids:
        logger.debug("No index IDs provided; keeping the configured embedding model.")
        return
    try:
        idx = db.get_index(idx_ids[0])
        model = (idx.get("metadata") or {}).get("embedding_model")
        if not model:
            logger.debug("Index %s has no embedding_model metadata.", idx_ids[0])
            return
        rp = RAG_AGENT.retrieval_pipeline
        logger.info(
            "Applying index embedding model '%s' (was '%s').",
            model, rp.config.get("embedding_model_name"),
        )
        rp.update_embedding_model(model)
    except Exception as e:
        logger.warning("Could not apply index embedding model: %s", e)


def _get_table_name_for_session(session_id):
    """Get the correct vector table name for a session by looking up its linked indexes."""
    if not session_id:
        return None

    try:
        idx_ids = db.get_indexes_for_session(session_id)
        if not idx_ids:
            logger.info("No indexes for session %s; using default table '%s'.", session_id, DEFAULT_TEXT_TABLE)
            return DEFAULT_TEXT_TABLE

        idx = db.get_index(idx_ids[0])
        if idx and idx.get('vector_table_name'):
            table_name = idx['vector_table_name']
            logger.info("Using table '%s' for session %s.", table_name, session_id)
            return table_name

        logger.warning("Index found but no vector table name for session %s.", session_id)
        return DEFAULT_TEXT_TABLE
    except Exception as e:
        logger.error("Error getting table name for session %s: %s", session_id, e)
        return DEFAULT_TEXT_TABLE


def _resolve_retrieval_mode(data):
    """`retrieval_mode` is the wire name for the pipeline's `search_type`; both are accepted."""
    value = data.get('retrieval_mode') or data.get('search_type')
    if value is None:
        return None, None
    if value not in SUPPORTED_RETRIEVAL_MODES:
        return None, f"Unsupported retrieval mode '{value}'. Supported: {', '.join(SUPPORTED_RETRIEVAL_MODES)}."
    return value, None


def _parse_chat_request(data):
    """Extract the canonical chat options from an already-normalized body."""
    retrieval_mode, error = _resolve_retrieval_mode(data)
    if error:
        return None, error

    return {
        "query": data.get('query'),
        "session_id": data.get('session_id'),
        "table_name": data.get('table_name'),
        "model": data.get('model'),
        "compose_sub_answers": data.get('compose_sub_answers'),
        "query_decompose": data.get('query_decompose'),
        "ai_rerank": data.get('ai_rerank'),
        "context_expand": data.get('context_expand'),
        "verify": data.get('verify'),
        "retrieval_k": data.get('retrieval_k'),
        "context_window_size": data.get('context_window_size'),
        "reranker_top_k": data.get('reranker_top_k'),
        "retrieval_mode": retrieval_mode,
        "force_rag": bool(data.get('force_rag', False)),
        "provence_prune": data.get('provence_prune'),
        "provence_threshold": data.get('provence_threshold'),
    }, None


def _build_index_config_override(base_config, *, table_name, options):
    """Build the per-request indexing config from the pipeline profile plus request options."""
    config_override = copy.deepcopy(base_config)
    retrieval_cfg = config_override.setdefault("retrieval", {})

    if table_name:
        config_override.setdefault("storage", {})["text_table_name"] = table_name
        retrieval_cfg.setdefault("dense", {})["lancedb_table_name"] = table_name

    retrieval_cfg.setdefault("latechunk", {})["enabled"] = options["enable_latechunk"]

    # `retrieval_mode` is the wire name for the pipeline's `search_type`; it is
    # recorded on the index config so the built index carries the mode it was
    # requested with.
    if options["retrieval_mode"] is not None:
        retrieval_cfg["search_type"] = options["retrieval_mode"]

    config_override["chunker_mode"] = "docling" if options["enable_docling_chunk"] else "legacy"

    enricher_cfg = config_override.setdefault("contextual_enricher", {})
    enricher_cfg["enabled"] = options["enable_enrich"]
    enricher_cfg["window_size"] = options["window_size"]

    indexing_cfg = config_override.setdefault("indexing", {})
    indexing_cfg["embedding_batch_size"] = options["batch_size_embed"]
    indexing_cfg["enrichment_batch_size"] = options["batch_size_enrich"]

    config_override.setdefault("chunking", {})["chunk_size"] = options["chunk_size"]

    if options["embedding_model"]:
        config_override["embedding_model_name"] = options["embedding_model"]
    if options["enrich_model"]:
        config_override["enrich_model"] = options["enrich_model"]
    if options["overview_model_name"]:
        config_override["overview_model_name"] = options["overview_model_name"]
    if options["session_id"]:
        config_override["overview_path"] = f"index_store/overviews/{options['session_id']}.jsonl"

    return config_override


class AdvancedRagApiHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        logger.info("%s - %s", self.address_string(), format % args)

    def do_OPTIONS(self):
        """Handle CORS preflight requests for frontend integration."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
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

        if parsed_path.path == '/health':
            self.send_json_response({"status": "ok"})
        elif parsed_path.path == '/models':
            self.handle_models()
        else:
            self.send_json_response({"error": "Not Found"}, status_code=404)

    def handle_chat(self):
        """Handles a chat query by calling the agentic RAG pipeline."""
        try:
            data = _read_json_body(self)
            params, error = _parse_chat_request(data)
            if error:
                self.send_json_response({"error": error}, status_code=400)
                return

            query = params["query"]
            if not query:
                self.send_json_response({"error": "Query is required"}, status_code=400)
                return

            session_id = params["session_id"]
            table_name = params["table_name"] or _get_table_name_for_session(session_id)

            with _generation_model_override(params["model"]):
                result = self._run_query(params, query, table_name, session_id, emit=None)

            self.send_json_response(result)

        except json.JSONDecodeError:
            self.send_json_response({"error": "Invalid JSON"}, status_code=400)
        except Exception as e:
            logger.exception("Chat request failed")
            self.send_json_response({"error": f"Server error: {str(e)}"}, status_code=500)

    def handle_chat_stream(self):
        """Stream internal phases and final answer using SSE (text/event-stream)."""
        try:
            data = _read_json_body(self)
            params, error = _parse_chat_request(data)
            if error:
                self.send_json_response({"error": error}, status_code=400)
                return

            query = params["query"]
            if not query:
                self.send_json_response({"error": "Query is required"}, status_code=400)
                return

            session_id = params["session_id"]
            table_name = params["table_name"] or _get_table_name_for_session(session_id)

            # Prepare response headers for SSE
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            # Keep connection alive for SSE; no manual chunked encoding (Python http.server
            # does not add chunk sizes automatically, so declaring it breaks clients).
            self.send_header('Connection', 'keep-alive')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            def emit(event_type: str, payload):
                """Send a single SSE event."""
                data_str = json.dumps({"type": event_type, "data": payload})
                self.wfile.write(f"data: {data_str}\n\n".encode('utf-8'))
                self.wfile.flush()

            try:
                with _generation_model_override(params["model"]):
                    final_result = self._run_query(params, query, table_name, session_id, emit=emit)
                emit("complete", final_result)
            except BrokenPipeError:
                logger.info("Client disconnected from SSE stream.")
            except Exception as e:
                logger.exception("Stream error")
                try:
                    emit("error", {"error": str(e)})
                except BrokenPipeError:
                    pass

        except json.JSONDecodeError:
            self.send_json_response({"error": "Invalid JSON"}, status_code=400)
        except Exception as e:
            logger.exception("Chat stream request failed")
            self.send_json_response({"error": f"Server error: {str(e)}"}, status_code=500)

    def _run_query(self, params, query, table_name, session_id, emit):
        """Shared execution path for /chat and /chat/stream.

        Everything, including force_rag, goes through Agent.run so that the
        verify / ai_rerank / decompose toggles are honored on every path.
        """
        rp_cfg = RAG_AGENT.retrieval_pipeline.config

        if session_id:
            idx_ids = db.get_indexes_for_session(session_id)
            _apply_index_embedding_model(idx_ids)
            rp_cfg["overview_path"] = f"index_store/overviews/{session_id}.jsonl"
            RAG_AGENT.load_overviews_for_indexes(idx_ids)

        if params["provence_prune"] is not None:
            rp_cfg.setdefault("provence", {})["enabled"] = bool(params["provence_prune"])
        if params["provence_threshold"] is not None:
            rp_cfg.setdefault("provence", {})["threshold"] = float(params["provence_threshold"])

        run_kwargs = {
            "table_name": table_name,
            "session_id": session_id,
            "compose_sub_answers": params["compose_sub_answers"],
            "query_decompose": params["query_decompose"],
            "ai_rerank": params["ai_rerank"],
            "context_expand": params["context_expand"],
            "verify": params["verify"],
            "retrieval_k": params["retrieval_k"],
            "context_window_size": params["context_window_size"],
            "reranker_top_k": params["reranker_top_k"],
            "retrieval_mode": params["retrieval_mode"],
            "force_rag": params["force_rag"],
        }
        if emit is not None:
            run_kwargs["event_callback"] = emit

        return RAG_AGENT.run(query, **run_kwargs)

    def handle_index(self):
        """Triggers the document indexing pipeline for specific files."""
        try:
            data = _read_json_body(self)

            file_paths = data.get('file_paths')
            if not file_paths or not isinstance(file_paths, list):
                self.send_json_response({"error": "A 'file_paths' list is required."}, status_code=400)
                return

            retrieval_mode, error = _resolve_retrieval_mode(data)
            if error:
                self.send_json_response({"error": error}, status_code=400)
                return

            session_id = data.get('session_id')
            options = {
                "session_id": session_id,
                "enable_latechunk": bool(data.get("enable_latechunk", False)),
                "enable_docling_chunk": bool(data.get("enable_docling_chunk", True)),
                "chunk_size": int(data.get("chunk_size", 512)),
                "retrieval_mode": retrieval_mode,
                "window_size": int(data.get("window_size", 2)),
                "enable_enrich": bool(data.get("enable_enrich", True)),
                "embedding_model": data.get('embedding_model'),
                "enrich_model": data.get('enrich_model'),
                "overview_model_name": data.get('overview_model_name'),
                "batch_size_embed": int(data.get("batch_size_embed", 50)),
                "batch_size_enrich": int(data.get("batch_size_enrich", 25)),
            }

            table_name = data.get('table_name') or _get_table_name_for_session(session_id)

            config_override = _build_index_config_override(
                INDEXING_PIPELINE.config, table_name=table_name, options=options
            )

            logger.info(
                "Indexing %d file(s) | table=%s | enrich=%s (window %s) | latechunk=%s | "
                "chunk_size=%s | embedding=%s | enrichment=%s",
                len(file_paths), table_name or DEFAULT_TEXT_TABLE, options["enable_enrich"],
                options["window_size"], options["enable_latechunk"], options["chunk_size"],
                options["embedding_model"] or config_override.get("embedding_model_name"),
                options["enrich_model"] or "default",
            )

            temp_pipeline = INDEXING_PIPELINE.__class__(
                config_override,
                INDEXING_PIPELINE.llm_client,
                INDEXING_PIPELINE.ollama_config,
            )
            temp_pipeline.run(file_paths)

            if options["embedding_model"] and session_id:
                try:
                    db.update_index_metadata(session_id, {"embedding_model": options["embedding_model"]})
                except Exception as e:
                    logger.warning("Could not update embedding_model metadata: %s", e)

            self.send_json_response({
                "message": f"Indexing process for {len(file_paths)} file(s) completed successfully.",
                "table_name": table_name or DEFAULT_TEXT_TABLE,
                "latechunk": options["enable_latechunk"],
                "docling_chunk": options["enable_docling_chunk"],
                "indexing_config": {
                    "chunk_size": options["chunk_size"],
                    "retrieval_mode": config_override.get("retrieval", {}).get("search_type"),
                    "window_size": options["window_size"],
                    "enable_enrich": options["enable_enrich"],
                    "embedding_model": config_override.get("embedding_model_name"),
                    "enrich_model": options["enrich_model"],
                    "overview_model_name": options["overview_model_name"],
                    "batch_size_embed": options["batch_size_embed"],
                    "batch_size_enrich": options["batch_size_enrich"],
                }
            })

        except json.JSONDecodeError:
            self.send_json_response({"error": "Invalid JSON"}, status_code=400)
        except Exception as e:
            logger.exception("Indexing request failed")
            self.send_json_response({"error": f"Failed to start indexing: {str(e)}"}, status_code=500)

    def handle_models(self):
        """Return the generation and embedding models available to the active backend."""
        try:
            generation_models = []
            embedding_models = [RAG_AGENT.retrieval_pipeline.config.get("embedding_model_name")]

            if LLM_BACKEND.lower() == "watsonx":
                generation_models.extend(
                    m for m in (WATSONX_CONFIG.get("generation_model"), WATSONX_CONFIG.get("enrichment_model")) if m
                )
            else:
                host = RAG_AGENT.ollama_config.get('host')
                try:
                    resp = requests.get(f"{host}/api/tags", timeout=OLLAMA_TIMEOUT_SECONDS)
                    resp.raise_for_status()
                    all_ollama_models = [m.get('name') for m in resp.json().get('models', []) if m.get('name')]

                    ollama_embedding_models = [
                        m for m in all_ollama_models if any(k in m for k in ['embed', 'bge', 'embedding'])
                    ]
                    generation_models.extend(m for m in all_ollama_models if m not in ollama_embedding_models)
                    embedding_models.extend(ollama_embedding_models)
                except Exception as e:
                    logger.warning("Could not list Ollama models from %s: %s", host, e)

            # HuggingFace embedding models loaded in-process (see rag_system/main.py EXTERNAL_MODELS).
            # harrier-oss-v1-0.6b is the shipped default; the Qwen3 family stays
            # selectable for multilingual / long-context corpora.
            embedding_models.extend([
                "microsoft/harrier-oss-v1-0.6b",
                "Qwen/Qwen3-Embedding-4B",
                "Qwen/Qwen3-Embedding-0.6B",
                "Qwen/Qwen3-Embedding-8B",
            ])

            self.send_json_response({
                "generation_models": sorted(set(generation_models)),
                "embedding_models": sorted({m for m in embedding_models if m}),
            })
        except Exception as e:
            self.send_json_response({"error": f"Could not list models: {e}"}, status_code=500)

    def send_json_response(self, data, status_code=200):
        """Utility to send a JSON response with CORS headers."""
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        response = json.dumps(data, indent=2)
        self.wfile.write(response.encode('utf-8'))


def start_server(port=8001):
    """Starts the API server."""
    # Use a reusable TCP server to avoid "address in use" errors on restart
    class ReusableTCPServer(socketserver.TCPServer):
        allow_reuse_address = True

    with ReusableTCPServer(("", port), AdvancedRagApiHandler) as httpd:
        print(f"🚀 Starting Advanced RAG API server on port {port}")
        print(f"🩺 Health endpoint: http://localhost:{port}/health")
        print(f"💬 Chat endpoint: http://localhost:{port}/chat")
        print(f"✨ Indexing endpoint: http://localhost:{port}/index")
        httpd.serve_forever()


if __name__ == "__main__":
    # To run this server: python -m rag_system.api_server
    start_server()

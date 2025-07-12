import json
import http.server
import socketserver
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
from rag_system.main import get_agent
from rag_system.factory import get_indexing_pipeline

# Initialize database connection once at module level
db = ChatDatabase("backend/chat_data.db")

# Get the desired agent mode from environment variables, defaulting to 'default'
# This allows us to easily switch between 'default', 'fast', 'react', etc.
AGENT_MODE = os.getenv("RAG_CONFIG_MODE", "default")
RAG_AGENT = get_agent(AGENT_MODE)
INDEXING_PIPELINE = get_indexing_pipeline(AGENT_MODE)

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
    debug_info = f"🔧 _apply_index_embedding_model called with idx_ids: {idx_ids}\n"
    
    if not idx_ids:
        debug_info += "⚠️ No index IDs provided\n"
        with open("logs/embedding_debug.log", "a") as f:
            f.write(debug_info)
        return
    try:
        idx = db.get_index(idx_ids[0])
        debug_info += f"🔧 Retrieved index: {idx.get('id')} with metadata: {idx.get('metadata', {})}\n"
        model = (idx.get("metadata") or {}).get("embedding_model")
        debug_info += f"🔧 Embedding model from metadata: {model}\n"
        if model:
            rp = RAG_AGENT.retrieval_pipeline
            current_model = rp.config.get("embedding_model_name")
            debug_info += f"🔧 Current embedding model: {current_model}\n"
            rp.update_embedding_model(model)
            debug_info += f"🔧 Updated embedding model to: {model}\n"
        else:
            debug_info += "⚠️ No embedding model found in metadata\n"
    except Exception as e:
        debug_info += f"⚠️ Could not apply index embedding model: {e}\n"
    
    # Write debug info to file
    with open("logs/embedding_debug.log", "a") as f:
        f.write(debug_info)

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
        
        # Use the first index's vector table name
        idx = db.get_index(idx_ids[0])
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

class AdvancedRagApiHandler(http.server.BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        """Handle CORS preflight requests for frontend integration."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
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
        else:
            self.send_json_response({"error": "Not Found"}, status_code=404)

    def handle_chat(self):
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

            # 🔄 UPDATE SESSION TITLE: If this is the first message in the session, update the title
            if session_id:
                try:
                    # Check if this is the first message by calling the backend server
                    backend_url = f"http://localhost:8000/sessions/{session_id}"
                    session_resp = requests.get(backend_url)
                    if session_resp.status_code == 200:
                        session_data = session_resp.json()
                        session = session_data.get('session', {})
                        # If message_count is 0, this is the first message
                        if session.get('message_count', 0) == 0:
                            # Generate a title from the first message
                            title = generate_session_title(query)
                            # Update the session title via backend API
                            # We'll need to add this endpoint to the backend, for now let's make a direct database call
                            # This is a temporary solution until we add a proper API endpoint
                            db.update_session_title(session_id, title)
                            print(f"📝 Updated session title to: {title}")
                            
                            # 💾 STORE USER MESSAGE: Add the user message to the database
                            user_message_id = db.add_message(session_id, query, "user")
                            print(f"💾 Stored user message: {user_message_id}")
                        else:
                            # Not the first message, but still store the user message
                            user_message_id = db.add_message(session_id, query, "user")
                            print(f"💾 Stored user message: {user_message_id}")
                except Exception as e:
                    print(f"⚠️ Failed to update session title or store user message: {e}")
                    # Continue with the request even if title update fails

            # Allow explicit table_name override
            table_name = data.get('table_name')
            if not table_name and session_id:
                table_name = _get_table_name_for_session(session_id)

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
            
            # 💾 STORE AI RESPONSE: Add the AI response to the database
            if session_id and result and result.get("answer"):
                try:
                    ai_message_id = db.add_message(session_id, result["answer"], "assistant")
                    print(f"💾 Stored AI response: {ai_message_id}")
                except Exception as e:
                    print(f"⚠️ Failed to store AI response: {e}")
                    # Continue even if storage fails

        except json.JSONDecodeError:
            self.send_json_response({"error": "Invalid JSON"}, status_code=400)
        except Exception as e:
            self.send_json_response({"error": f"Server error: {str(e)}"}, status_code=500)

    def handle_chat_stream(self):
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

            # 🔄 UPDATE SESSION TITLE: If this is the first message in the session, update the title
            if session_id:
                try:
                    # Check if this is the first message by calling the backend server
                    backend_url = f"http://localhost:8000/sessions/{session_id}"
                    session_resp = requests.get(backend_url)
                    if session_resp.status_code == 200:
                        session_data = session_resp.json()
                        session = session_data.get('session', {})
                        # If message_count is 0, this is the first message
                        if session.get('message_count', 0) == 0:
                            # Generate a title from the first message
                            title = generate_session_title(query)
                            # Update the session title via backend API
                            # We'll need to add this endpoint to the backend, for now let's make a direct database call
                            # This is a temporary solution until we add a proper API endpoint
                            db.update_session_title(session_id, title)
                            print(f"📝 Updated session title to: {title}")
                            
                            # 💾 STORE USER MESSAGE: Add the user message to the database
                            user_message_id = db.add_message(session_id, query, "user")
                            print(f"💾 Stored user message: {user_message_id}")
                        else:
                            # Not the first message, but still store the user message
                            user_message_id = db.add_message(session_id, query, "user")
                            print(f"💾 Stored user message: {user_message_id}")
                except Exception as e:
                    print(f"⚠️ Failed to update session title or store user message: {e}")
                    # Continue with the request even if title update fails

            # Allow explicit table_name override
            table_name = data.get('table_name')
            if not table_name and session_id:
                table_name = _get_table_name_for_session(session_id)

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
                
                # 💾 STORE AI RESPONSE: Add the AI response to the database
                if session_id and final_result and final_result.get("answer"):
                    try:
                        ai_message_id = db.add_message(session_id, final_result["answer"], "assistant")
                        print(f"💾 Stored AI response: {ai_message_id}")
                    except Exception as e:
                        print(f"⚠️ Failed to store AI response: {e}")
                        # Continue even if storage fails
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
            enable_enrich = bool(data.get("enable_enrich", True))
            embedding_model = data.get('embeddingModel')
            enrich_model = data.get('enrichModel')
            overview_model = data.get('overviewModel') or data.get('overview_model_name')
            batch_size_embed = int(data.get("batch_size_embed", 50))
            batch_size_enrich = int(data.get("batch_size_enrich", 25))
            
            if not file_paths or not isinstance(file_paths, list):
                self.send_json_response({
                    "error": "A 'file_paths' list is required."
                }, status_code=400)
                return

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
                
                # 🔧 Configure chunking parameters
                config_override.setdefault("chunking", {})
                config_override["chunking"]["chunk_size"] = chunk_size
                config_override["chunking"]["chunk_overlap"] = chunk_overlap
                
                # 🔧 Configure embedding model if specified
                if embedding_model:
                    config_override["embedding_model_name"] = embedding_model
                
                # 🔧 Configure enrichment model if specified
                if enrich_model:
                    config_override["enrich_model"] = enrich_model
                
                # 🔧 Overview model (can differ from enrichment)
                if overview_model:
                    config_override["overview_model_name"] = overview_model
                
                print(f"🔧 INDEXING CONFIG: Contextual Enrichment: {enable_enrich}, Window Size: {window_size}")
                print(f"🔧 CHUNKING CONFIG: Size: {chunk_size}, Overlap: {chunk_overlap}")
                print(f"🔧 MODEL CONFIG: Embedding: {embedding_model or 'default'}, Enrichment: {enrich_model or 'default'}")
                
                # 🔧 Set index-specific overview path so each index writes separate file
                if session_id:
                    config_override["overview_path"] = f"index_store/overviews/{session_id}.jsonl"

                # 🔧 Configure late chunking
                config_override.setdefault("retrievers", {}).setdefault("latechunk", {})["enabled"] = True

                # Create a temporary pipeline instance with the overridden config
                temp_pipeline = INDEXING_PIPELINE.__class__(
                    config_override, 
                    INDEXING_PIPELINE.llm_client, 
                    INDEXING_PIPELINE.ollama_config
                )
                temp_pipeline.run(file_paths)
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
                
                # 🔧 Configure chunking parameters
                config_override.setdefault("chunking", {})
                config_override["chunking"]["chunk_size"] = chunk_size
                config_override["chunking"]["chunk_overlap"] = chunk_overlap
                
                # 🔧 Configure embedding model if specified
                if embedding_model:
                    config_override["embedding_model_name"] = embedding_model
                
                # 🔧 Configure enrichment model if specified
                if enrich_model:
                    config_override["enrich_model"] = enrich_model
                
                # 🔧 Overview model (can differ from enrichment)
                if overview_model:
                    config_override["overview_model_name"] = overview_model
                
                print(f"🔧 INDEXING CONFIG: Contextual Enrichment: {enable_enrich}, Window Size: {window_size}")
                print(f"🔧 CHUNKING CONFIG: Size: {chunk_size}, Overlap: {chunk_overlap}")
                print(f"🔧 MODEL CONFIG: Embedding: {embedding_model or 'default'}, Enrichment: {enrich_model or 'default'}")
                
                # 🔧 Set index-specific overview path so each index writes separate file
                if session_id:
                    config_override["overview_path"] = f"index_store/overviews/{session_id}.jsonl"

                # 🔧 Configure late chunking
                config_override.setdefault("retrievers", {}).setdefault("latechunk", {})["enabled"] = True

                # Create temporary pipeline with overridden config
                temp_pipeline = INDEXING_PIPELINE.__class__(
                    config_override, 
                    INDEXING_PIPELINE.llm_client, 
                    INDEXING_PIPELINE.ollama_config
                )
                temp_pipeline.run(file_paths)

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
                    "batch_size_enrich": batch_size_enrich
                }
            })

            if embedding_model:
                try:
                    db.update_index_metadata(session_id, {"embedding_model": embedding_model})
                except Exception as e:
                    print(f"⚠️ Could not update embedding_model metadata: {e}")

        except json.JSONDecodeError:
            self.send_json_response({"error": "Invalid JSON"}, status_code=400)
        except Exception as e:
            self.send_json_response({"error": f"Failed to start indexing: {str(e)}"}, status_code=500)

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
        print(f"💬 Chat endpoint: http://localhost:{port}/chat")
        print(f"✨ Indexing endpoint: http://localhost:{port}/index")
        httpd.serve_forever()

if __name__ == "__main__":
    # To run this server: python -m rag_system.api_server
    start_server() 
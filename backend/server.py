import json
import http.server
import socketserver
import cgi
import os
import uuid
from urllib.parse import urlparse
import requests  # 🆕 Import requests for making HTTP calls
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path so we can import rag_system modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from localgpt_runtime import (
    UploadRejected,
    cors_origin,
    env_path,
    normalize_index_options,
    request_is_authorized,
    safe_upload_path,
    store_upload,
)

from backend.ollama_client import OllamaClient
from backend.database import db, generate_session_title
from typing import List, Dict, Any
import re

# 🆕 Reusable TCPServer with address reuse enabled
class ReusableTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = env_path("LOCALGPT_UPLOAD_DIR", os.path.join(PROJECT_ROOT, "shared_uploads"))
RAG_API_URL = os.environ.get("RAG_API_URL", "http://127.0.0.1:8001").rstrip("/")


def _rag_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    token = os.environ.get("LOCALGPT_API_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers

class ChatHandler(http.server.BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.ollama_client = OllamaClient()
        super().__init__(*args, **kwargs)

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
        """Handle CORS preflight requests"""
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

    def do_GET(self):
        """Handle GET requests"""
        if not self._authorized():
            return
        parsed_path = urlparse(self.path)

        if parsed_path.path == '/health':
            self.send_json_response({
                "status": "ok",
                "ollama_running": self.ollama_client.is_ollama_running(),
                "available_models": self.ollama_client.list_models(),
                "database_stats": db.get_stats()
            })
        elif parsed_path.path == '/sessions':
            self.handle_get_sessions()
        elif parsed_path.path == '/sessions/cleanup':
            self.handle_cleanup_sessions()
        elif parsed_path.path == '/models':
            self.handle_get_models()
        elif parsed_path.path == '/indexes':
            self.handle_get_indexes()
        elif parsed_path.path.startswith('/indexes/') and parsed_path.path.count('/') == 2:
            index_id = parsed_path.path.split('/')[-1]
            self.handle_get_index(index_id)
        elif parsed_path.path.startswith('/sessions/') and parsed_path.path.endswith('/documents'):
            session_id = parsed_path.path.split('/')[-2]
            self.handle_get_session_documents(session_id)
        elif parsed_path.path.startswith('/sessions/') and parsed_path.path.endswith('/indexes'):
            session_id = parsed_path.path.split('/')[-2]
            self.handle_get_session_indexes(session_id)
        elif parsed_path.path.startswith('/sessions/') and parsed_path.path.count('/') == 2:
            session_id = parsed_path.path.split('/')[-1]
            self.handle_get_session(session_id)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        """Handle POST requests"""
        if not self._authorized():
            return
        parsed_path = urlparse(self.path)

        if parsed_path.path == '/sessions':
            self.handle_create_session()
        elif parsed_path.path == '/indexes':
            self.handle_create_index()
        elif parsed_path.path.startswith('/indexes/') and parsed_path.path.endswith('/upload'):
            index_id = parsed_path.path.split('/')[-2]
            self.handle_index_file_upload(index_id)
        elif parsed_path.path.startswith('/indexes/') and parsed_path.path.endswith('/build'):
            index_id = parsed_path.path.split('/')[-2]
            self.handle_build_index(index_id)
        elif parsed_path.path.startswith('/sessions/') and '/indexes/' in parsed_path.path:
            parts = parsed_path.path.split('/')
            session_id = parts[2]
            index_id = parts[4]
            self.handle_link_index_to_session(session_id, index_id)
        elif parsed_path.path.startswith('/sessions/') and parsed_path.path.endswith('/messages/stream'):
            session_id = parsed_path.path.split('/')[-3]
            self.handle_session_chat_stream(session_id)
        elif parsed_path.path.startswith('/sessions/') and parsed_path.path.endswith('/messages'):
            session_id = parsed_path.path.split('/')[-2]
            self.handle_session_chat(session_id)
        elif parsed_path.path.startswith('/sessions/') and parsed_path.path.endswith('/upload'):
            session_id = parsed_path.path.split('/')[-2]
            self.handle_file_upload(session_id)
        elif parsed_path.path.startswith('/sessions/') and parsed_path.path.endswith('/index'):
            session_id = parsed_path.path.split('/')[-2]
            self.handle_index_documents(session_id)
        elif parsed_path.path.startswith('/sessions/') and parsed_path.path.endswith('/rename'):
            session_id = parsed_path.path.split('/')[-2]
            self.handle_rename_session(session_id)
        else:
            self.send_response(404)
            self.end_headers()

    def do_DELETE(self):
        """Handle DELETE requests"""
        if not self._authorized():
            return
        parsed_path = urlparse(self.path)

        if parsed_path.path.startswith('/sessions/') and parsed_path.path.count('/') == 2:
            session_id = parsed_path.path.split('/')[-1]
            self.handle_delete_session(session_id)
        elif parsed_path.path.startswith('/indexes/') and parsed_path.path.count('/') == 2:
            index_id = parsed_path.path.split('/')[-1]
            self.handle_delete_index(index_id)
        else:
            self.send_response(404)
            self.end_headers()

    def handle_get_sessions(self):
        """Get all chat sessions"""
        try:
            sessions = db.get_sessions()
            self.send_json_response({
                "sessions": sessions,
                "total": len(sessions)
            })
        except Exception as e:
            self.send_json_response({
                "error": f"Failed to get sessions: {str(e)}"
            }, status_code=500)

    def handle_cleanup_sessions(self):
        """Clean up empty sessions"""
        try:
            cleanup_count = db.cleanup_empty_sessions()
            self.send_json_response({
                "message": f"Cleaned up {cleanup_count} empty sessions",
                "cleanup_count": cleanup_count
            })
        except Exception as e:
            self.send_json_response({
                "error": f"Failed to cleanup sessions: {str(e)}"
            }, status_code=500)

    def handle_get_session(self, session_id: str):
        """Get a specific session with its messages"""
        try:
            session = db.get_session(session_id)
            if not session:
                self.send_json_response({
                    "error": "Session not found"
                }, status_code=404)
                return

            messages = db.get_messages(session_id)

            self.send_json_response({
                "session": session,
                "messages": messages
            })
        except Exception as e:
            self.send_json_response({
                "error": f"Failed to get session: {str(e)}"
            }, status_code=500)

    def handle_get_session_documents(self, session_id: str):
        """Return documents and basic info for a session."""
        try:
            session = db.get_session(session_id)
            if not session:
                self.send_json_response({"error": "Session not found"}, status_code=404)
                return

            docs = db.get_documents_for_session(session_id)

            # Extract original filenames from stored paths
            filenames = [os.path.basename(p).split('_', 1)[-1] if '_' in os.path.basename(p) else os.path.basename(p) for p in docs]

            self.send_json_response({
                "session": session,
                "files": filenames,
                "file_count": len(docs)
            })
        except Exception as e:
            self.send_json_response({"error": f"Failed to get documents: {str(e)}"}, status_code=500)

    def handle_create_session(self):
        """Create a new chat session"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))

            title = data.get('title', 'New Chat')
            model = data.get('model', 'llama3.2:latest')

            session_id = db.create_session(title, model)
            session = db.get_session(session_id)

            self.send_json_response({
                "session": session,
                "session_id": session_id
            }, status_code=201)

        except json.JSONDecodeError:
            self.send_json_response({
                "error": "Invalid JSON"
            }, status_code=400)
        except Exception as e:
            self.send_json_response({
                "error": f"Failed to create session: {str(e)}"
            }, status_code=500)

    def handle_session_chat(self, session_id: str):
        """
        Handle chat within a specific session.
        Intelligently routes between direct LLM (fast) and RAG pipeline (document-aware).
        """
        try:
            session = db.get_session(session_id)
            if not session:
                self.send_json_response({"error": "Session not found"}, status_code=404)
                return

            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            message = data.get('message', '')

            if not message:
                self.send_json_response({"error": "Message is required"}, status_code=400)
                return

            if session['message_count'] == 0:
                title = generate_session_title(message)
                db.update_session_title(session_id, title)

            # Add user message to database first
            user_message_id = db.add_message(session_id, message, "user")

            # The RAG service owns provider selection and routing.  Keeping that
            # decision in one place makes Ollama/WatsonX behavior consistent.
            idx_ids = db.get_indexes_for_session(session_id)
            response_text, source_docs, route = self._handle_rag_query(
                session_id, message, data, idx_ids
            )
            use_rag = route != "direct_answer"

            # Add AI response to database
            ai_message_id = db.add_message(session_id, response_text, "assistant")

            updated_session = db.get_session(session_id)

            # Send response with proper error handling
            self.send_json_response({
                "response": response_text,
                "session": updated_session,
                "source_documents": source_docs,
                "used_rag": use_rag,
                "route": route,
                "user_message_id": user_message_id,
                "ai_message_id": ai_message_id,
            })

        except BrokenPipeError:
            # Client disconnected - this is normal for long queries, just log it
            print(f"⚠️  Client disconnected during RAG processing for query: '{message[:30]}...'")
        except json.JSONDecodeError:
            self.send_json_response({
                "error": "Invalid JSON"
            }, status_code=400)
        except Exception as e:
            print(f"❌ Server error in session chat: {str(e)}")
            try:
                self.send_json_response({
                    "error": f"Server error: {str(e)}"
                }, status_code=500)
            except BrokenPipeError:
                print("⚠️  Client disconnected during error response")

    def handle_session_chat_stream(self, session_id: str):
        """Proxy RAG SSE while keeping the backend as the sole message owner."""
        session = db.get_session(session_id)
        if not session:
            self.send_json_response({"error": "Session not found"}, status_code=404)
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            message = str(data.get("query") or data.get("message") or "").strip()
            if not message:
                self.send_json_response({"error": "Message is required"}, status_code=400)
                return

            if session["message_count"] == 0:
                db.update_session_title(session_id, generate_session_title(message))
            history = db.get_conversation_history(session_id)
            user_message_id = db.add_message(session_id, message, "user")

            index_ids = db.get_indexes_for_session(session_id)
            table_names = [
                index["vector_table_name"]
                for index_id in index_ids
                if (index := db.get_index(index_id)) and index.get("vector_table_name")
            ]
            payload = dict(data)
            payload.update(
                {
                    "query": message,
                    "session_id": session_id,
                    "conversation_history": history,
                }
            )
            payload.pop("table_name", None)
            if table_names:
                payload["table_names"] = table_names
            if "search_type" not in data:
                index_modes = {
                    (index.get("metadata") or {}).get("retrieval_mode")
                    for index_id in index_ids
                    if (index := db.get_index(index_id))
                } - {None}
                if len(index_modes) == 1:
                    payload["search_type"] = next(iter(index_modes))

            upstream = requests.post(
                f"{RAG_API_URL}/chat/stream",
                json=payload,
                headers=_rag_headers(),
                stream=True,
                timeout=(10, 600),
            )
            if upstream.status_code != 200:
                self.send_json_response(
                    {"error": f"RAG service failed: {upstream.text}"},
                    status_code=502,
                )
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self._send_cors()
            self.end_headers()

            for raw_line in upstream.iter_lines(decode_unicode=True):
                if not raw_line or not raw_line.startswith("data:"):
                    continue
                serialized = raw_line.removeprefix("data:").strip()
                event = json.loads(serialized)
                if event.get("type") == "complete":
                    result = event.get("data") or {}
                    answer = result.get("answer", "")
                    assistant_message_id = db.add_message(
                        session_id,
                        answer,
                        "assistant",
                        {"source_documents": result.get("source_documents", [])},
                    )
                    result.update(
                        {
                            "user_message_id": user_message_id,
                            "assistant_message_id": assistant_message_id,
                            "session": db.get_session(session_id),
                        }
                    )
                    event["data"] = result
                self.wfile.write(
                    f"data: {json.dumps(event)}\n\n".encode("utf-8")
                )
                self.wfile.flush()
                if event.get("type") == "complete":
                    break
        except (json.JSONDecodeError, ValueError) as exc:
            self.send_json_response({"error": str(exc)}, status_code=400)
        except BrokenPipeError:
            print("Client disconnected from the chat stream")
        except Exception as exc:
            try:
                self.send_json_response({"error": str(exc)}, status_code=500)
            except BrokenPipeError:
                pass


    def _handle_rag_query(self, session_id: str, message: str, data: dict, idx_ids: List[str]):
        """
        Handle query using the full RAG pipeline (delegates to the advanced RAG API running on port 8001).

        Returns:
            tuple[str, List[dict]]: (response_text, source_documents)
        """
        # Defaults
        response_text = ""
        source_docs: List[dict] = []

        # Build payload for RAG API
        rag_api_url = f"{RAG_API_URL}/chat"
        table_names = [
            index["vector_table_name"]
            for index_id in idx_ids
            if (index := db.get_index(index_id)) and index.get("vector_table_name")
        ]
        payload: Dict[str, Any] = {
            "query": message,
            "session_id": session_id,
            "conversation_history": db.get_conversation_history(session_id)[:-1],
        }
        if table_names:
            payload["table_names"] = table_names
        if "search_type" not in data:
            index_modes = {
                (index.get("metadata") or {}).get("retrieval_mode")
                for index_id in idx_ids
                if (index := db.get_index(index_id))
            } - {None}
            if len(index_modes) == 1:
                payload["search_type"] = next(iter(index_modes))

        # Copy optional parameters from the incoming request
        optional_params: Dict[str, tuple[type, str]] = {
            "model": (str, "model"),
            "force_rag": (bool, "force_rag"),
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
            rag_response = requests.post(
                rag_api_url, json=payload, headers=_rag_headers(), timeout=300
            )
            if rag_response.status_code == 200:
                rag_data = rag_response.json()
                response_text = rag_data.get("answer", "No answer found.")
                source_docs = rag_data.get("source_documents", [])
                route = rag_data.get("route", "rag_query")
            else:
                response_text = f"Error from RAG API ({rag_response.status_code}): {rag_response.text}"
                print(f"❌ RAG API error: {response_text}")
        except requests.exceptions.ConnectionError:
            response_text = "Could not connect to the RAG API server. Please ensure it is running."
            print("❌ Connection to RAG API failed (port 8001).")
        except Exception as e:
            response_text = f"Error processing RAG query: {str(e)}"
            print(f"❌ RAG processing error: {e}")

        route = locals().get("route", "error")

        # Strip any <think>/<thinking> tags that might slip through
        response_text = re.sub(r'<(think|thinking)>.*?</\\1>', '', response_text, flags=re.DOTALL | re.IGNORECASE).strip()

        return response_text, source_docs, route

    def handle_delete_session(self, session_id: str):
        """Delete a session and its messages"""
        try:
            temporary_documents = db.get_documents_for_session(session_id)
            deleted = db.delete_session(session_id)
            if deleted:
                upload_root = Path(UPLOAD_DIR).resolve()
                for document_path in temporary_documents:
                    stored_path = Path(document_path).resolve()
                    try:
                        stored_path.relative_to(upload_root)
                    except ValueError:
                        continue
                    stored_path.unlink(missing_ok=True)
                self.send_json_response({'deleted': deleted})
            else:
                self.send_json_response({'error': 'Session not found'}, status_code=404)
        except Exception as e:
            self.send_json_response({'error': str(e)}, status_code=500)

    def handle_file_upload(self, session_id: str):
        """Handle file uploads, save them, and associate with the session."""
        if not db.get_session(session_id):
            self.send_json_response({"error": "Session not found"}, status_code=404)
            return
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={'REQUEST_METHOD': 'POST', 'CONTENT_TYPE': self.headers['Content-Type']}
        )

        uploaded_files = []
        staged_paths = []
        if 'files' in form:
            files = form['files']
            if not isinstance(files, list):
                files = [files]

            os.makedirs(UPLOAD_DIR, exist_ok=True)

            for file_item in files:
                if file_item.filename:
                    # Create a unique filename to avoid overwrites
                    unique_filename = f"{uuid.uuid4()}_{file_item.filename}"
                    try:
                        file_path = safe_upload_path(UPLOAD_DIR, unique_filename)
                        store_upload(file_item.file, file_path)
                        staged_paths.append(file_path)
                    except UploadRejected as exc:
                        for staged_path in staged_paths:
                            staged_path.unlink(missing_ok=True)
                        self.send_json_response({"error": str(exc)}, status_code=400)
                        return

                    uploaded_files.append({"filename": file_item.filename, "stored_path": str(file_path)})

        if not uploaded_files:
            self.send_json_response({"error": "No files were uploaded"}, status_code=400)
            return

        for uploaded in uploaded_files:
            db.add_document_to_session(session_id, uploaded["stored_path"])

        self.send_json_response({
            "message": f"Successfully uploaded {len(uploaded_files)} files.",
            "uploaded_files": uploaded_files
        })

    def handle_index_documents(self, session_id: str):
        """Turn temporary session uploads into a dedicated, linked index."""
        print(f"🔥 Received request to index documents for session {session_id[:8]}...")
        index_id = None
        try:
            file_paths = db.get_documents_for_session(session_id)
            if not file_paths:
                self.send_json_response({"message": "No documents to index for this session."}, status_code=200)
                return

            session = db.get_session(session_id)
            if not session:
                self.send_json_response({"error": "Session not found"}, status_code=404)
                return
            linked_models = {
                (linked.get("metadata") or {}).get("embedding_model")
                for linked_id in db.get_indexes_for_session(session_id)
                if (linked := db.get_index(linked_id))
            } - {None}
            embedding_model = next(iter(linked_models), "Qwen/Qwen3-Embedding-0.6B")
            index_id = db.create_index(
                name=f"{session['title']} documents",
                description="Created from documents attached in chat",
                metadata={
                    "status": "created",
                    "embedding_model": embedding_model,
                    "retrieval_mode": "hybrid",
                    "source": "session_upload",
                },
            )
            index = db.get_index(index_id)
            for file_path in file_paths:
                stored_name = Path(file_path).name
                original_name = stored_name.split("_", 1)[-1]
                db.add_document_to_index(index_id, original_name, file_path)

            rag_api_url = f"{RAG_API_URL}/index"
            rag_response = requests.post(
                rag_api_url,
                json={
                    "file_paths": file_paths,
                    "session_id": index_id,
                    "table_name": index["vector_table_name"],
                    "embedding_model": embedding_model,
                    "retrieval_mode": "hybrid",
                },
                headers=_rag_headers(),
                timeout=600,
            )

            if rag_response.status_code == 200:
                db.link_index_to_session(session_id, index_id)
                db.clear_documents_for_session(session_id)
                self.send_json_response(
                    {
                        "message": f"Indexed {len(file_paths)} attached document(s)",
                        "index_id": index_id,
                        "index": db.get_index(index_id),
                        "build": rag_response.json(),
                    }
                )
            else:
                error_info = rag_response.text
                cleanup = requests.delete(
                    f"{RAG_API_URL}/indexes/{index_id}",
                    headers=_rag_headers(),
                    timeout=60,
                )
                if cleanup.ok:
                    db.delete_index(index_id)
                self.send_json_response(
                    {"error": f"Indexing failed: {error_info}", "index_id": index_id},
                    status_code=502,
                )

        except Exception as e:
            print(f"❌ Exception during indexing: {str(e)}")
            if index_id and db.get_index(index_id):
                db.update_index_metadata(index_id, {"status": "failed", "error": str(e)})
            self.send_json_response({"error": f"An unexpected error occurred: {str(e)}"}, status_code=500)

    def handle_get_models(self):
        """Get available models from both Ollama and HuggingFace, grouped by capability"""
        try:
            generation_models = []
            embedding_models = []

            # Get Ollama models if available
            if self.ollama_client.is_ollama_running():
                all_ollama_models = self.ollama_client.list_models()

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

            self.send_json_response({
                "generation_models": generation_models,
                "embedding_models": embedding_models
            })
        except Exception as e:
            self.send_json_response({
                "error": f"Could not list models: {str(e)}"
            }, status_code=500)

    def handle_get_indexes(self):
        try:
            data = db.list_indexes()
            self.send_json_response({'indexes': data, 'total': len(data)})
        except Exception as e:
            self.send_json_response({'error': str(e)}, status_code=500)

    def handle_get_index(self, index_id: str):
        try:
            data = db.get_index(index_id)
            if not data:
                self.send_json_response({'error': 'Index not found'}, status_code=404)
                return
            self.send_json_response(data)
        except Exception as e:
            self.send_json_response({'error': str(e)}, status_code=500)

    def handle_create_index(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            name = data.get('name')
            description = data.get('description')
            metadata = data.get('metadata', {})

            if not name:
                self.send_json_response({'error': 'Name required'}, status_code=400)
                return

            complete_metadata = {
                'status': 'created',
                'metadata_source': 'backend_defaults',
                'created_at': datetime.now().isoformat(),
                'chunk_size': 512,
                'chunk_overlap': 64,
                'retrieval_mode': 'hybrid',
                'window_size': 2,
                'embedding_model': 'Qwen/Qwen3-Embedding-0.6B',
                'enrich_model': os.environ.get('OLLAMA_ENRICHMENT_MODEL', 'qwen3:0.6b'),
                'overview_model': os.environ.get('OLLAMA_ENRICHMENT_MODEL', 'qwen3:0.6b'),
                'enable_enrich': True,
                'latechunk': False,
                'docling_chunk': False,
            }
            complete_metadata.update(metadata)
            metadata = complete_metadata

            idx_id = db.create_index(name, description, metadata)
            self.send_json_response({'index_id': idx_id}, status_code=201)
        except Exception as e:
            self.send_json_response({'error': str(e)}, status_code=500)

    def handle_index_file_upload(self, index_id: str):
        """Reuse file upload logic but store docs under index."""
        if not db.get_index(index_id):
            self.send_json_response({'error': 'Index not found'}, status_code=404)
            return
        form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={'REQUEST_METHOD':'POST', 'CONTENT_TYPE': self.headers['Content-Type']})
        uploaded_files=[]
        staged_paths=[]
        if 'files' in form:
            files=form['files']
            if not isinstance(files, list):
                files=[files]
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            for f in files:
                if f.filename:
                    unique=f"{uuid.uuid4()}_{f.filename}"
                    try:
                        path = safe_upload_path(UPLOAD_DIR, unique)
                        store_upload(f.file, path)
                        staged_paths.append(path)
                    except UploadRejected as exc:
                        for staged_path in staged_paths:
                            staged_path.unlink(missing_ok=True)
                        self.send_json_response({'error': str(exc)}, status_code=400)
                        return
                    uploaded_files.append({'filename':f.filename,'stored_path':str(path)})
        if not uploaded_files:
            self.send_json_response({'error':'No files uploaded'}, status_code=400)
            return
        for uploaded in uploaded_files:
            db.add_document_to_index(index_id, uploaded['filename'], uploaded['stored_path'])
        self.send_json_response({'message':f"Uploaded {len(uploaded_files)} files","uploaded_files":uploaded_files})

    def handle_build_index(self, index_id: str):
        try:
            index=db.get_index(index_id)
            if not index:
                self.send_json_response({'error':'Index not found'}, status_code=404)
                return
            file_paths=[d['stored_path'] for d in index.get('documents',[])]
            if not file_paths:
                self.send_json_response({'error':'No documents to index'}, status_code=400)
                return

            raw_options = {}
            if 'Content-Length' in self.headers and int(self.headers['Content-Length']) > 0:
                length = int(self.headers['Content-Length'])
                raw_options = json.loads(self.rfile.read(length).decode('utf-8'))
            options = normalize_index_options(raw_options)
            latechunk = bool(options['enable_latechunk'])
            docling_chunk = bool(options['enable_docling_chunk'])
            chunk_size = options['chunk_size']
            chunk_overlap = options['chunk_overlap']
            retrieval_mode = options['retrieval_mode']
            window_size = int(options['window_size'])
            enable_enrich = bool(options['enable_enrich'])
            embedding_model = options.get('embedding_model')
            enrich_model = options.get('enrich_model')
            batch_size_embed = int(options['batch_size_embed'])
            batch_size_enrich = int(options['batch_size_enrich'])
            overview_model = options.get('overview_model')

            # Delegate to advanced RAG API same as session indexing
            rag_api_url = f"{RAG_API_URL}/index"
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
                "batch_size_enrich": batch_size_enrich
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

            rag_resp = requests.post(
                rag_api_url, json=payload, headers=_rag_headers(), timeout=600
            )
            if rag_resp.status_code==200:
                rag_data = rag_resp.json()
                meta_updates = {
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "retrieval_mode": retrieval_mode,
                    "window_size": window_size,
                    "enable_enrich": enable_enrich,
                    "latechunk": latechunk,
                    "docling_chunk": docling_chunk,
                }
                if embedding_model:
                    meta_updates["embedding_model"] = embedding_model
                if enrich_model:
                    meta_updates["enrich_model"] = enrich_model
                if overview_model:
                    meta_updates["overview_model"] = overview_model
                try:
                    db.update_index_metadata(index_id, meta_updates)
                except Exception as e:
                    print(f"⚠️ Failed to update index metadata: {e}")

                self.send_json_response({
                    "response": rag_data,
                    **meta_updates,
                    **rag_data,
                })
            else:
                try:
                    err_json = rag_resp.json()
                except Exception:
                    err_json = {'error': rag_resp.text}
                status = rag_resp.status_code if 400 <= rag_resp.status_code < 500 else 502
                self.send_json_response(err_json, status_code=status)
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            self.send_json_response({'error':str(e)}, status_code=400)
        except Exception as e:
            self.send_json_response({'error':str(e)}, status_code=500)

    def handle_link_index_to_session(self, session_id: str, index_id: str):
        try:
            if not db.get_session(session_id):
                self.send_json_response({'error': 'Session not found'}, status_code=404)
                return
            new_index = db.get_index(index_id)
            if not new_index:
                self.send_json_response({'error': 'Index not found'}, status_code=404)
                return
            new_model = (new_index.get('metadata') or {}).get('embedding_model')
            existing_models = {
                (existing.get('metadata') or {}).get('embedding_model')
                for existing_id in db.get_indexes_for_session(session_id)
                if (existing := db.get_index(existing_id))
            } - {None}
            if new_model and existing_models and new_model not in existing_models:
                self.send_json_response(
                    {
                        'error': 'Linked indexes must use the same embedding model',
                        'existing_models': sorted(existing_models),
                        'requested_model': new_model,
                    },
                    status_code=409,
                )
                return
            db.link_index_to_session(session_id, index_id)
            self.send_json_response({'message':'Index linked to session', 'session_id': session_id, 'index_id': index_id})
        except Exception as e:
            self.send_json_response({'error':str(e)}, status_code=500)

    def handle_get_session_indexes(self, session_id: str):
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
            self.send_json_response({'indexes': indexes, 'total': len(indexes)})
        except Exception as e:
            self.send_json_response({'error': str(e)}, status_code=500)

    def handle_delete_index(self, index_id: str):
        """Remove an index, its documents, links, and the underlying LanceDB table."""
        try:
            index = db.get_index(index_id)
            if not index:
                self.send_json_response({'error': 'Index not found'}, status_code=404)
                return
            artifact_response = requests.delete(f"{RAG_API_URL}/indexes/{index_id}", headers=_rag_headers(), timeout=60)
            if not artifact_response.ok:
                self.send_json_response(
                    {
                        'error': 'RAG service could not delete index artifacts',
                        'details': artifact_response.text,
                    },
                    status_code=502,
                )
                return
            deleted = db.delete_index(index_id)
            if deleted:
                upload_root = Path(UPLOAD_DIR).resolve()
                for document in index.get('documents', []):
                    stored_path = Path(document['stored_path']).resolve()
                    try:
                        stored_path.relative_to(upload_root)
                    except ValueError:
                        continue
                    stored_path.unlink(missing_ok=True)
                self.send_json_response({'message': 'Index deleted successfully', 'index_id': index_id})
            else:
                self.send_json_response({'error': 'Index not found'}, status_code=404)
        except Exception as e:
            self.send_json_response({'error': str(e)}, status_code=500)

    def handle_rename_session(self, session_id: str):
        """Rename an existing session title"""
        try:
            session = db.get_session(session_id)
            if not session:
                self.send_json_response({"error": "Session not found"}, status_code=404)
                return

            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self.send_json_response({"error": "Request body required"}, status_code=400)
                return

            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            new_title: str = data.get('title', '').strip()

            if not new_title:
                self.send_json_response({"error": "Title cannot be empty"}, status_code=400)
                return

            db.update_session_title(session_id, new_title)
            updated_session = db.get_session(session_id)

            self.send_json_response({
                "message": "Session renamed successfully",
                "session": updated_session
            })

        except json.JSONDecodeError:
            self.send_json_response({"error": "Invalid JSON"}, status_code=400)
        except Exception as e:
            self.send_json_response({"error": f"Failed to rename session: {str(e)}"}, status_code=500)

    def send_json_response(self, data, status_code: int = 200):
        """Send a JSON (UTF-8) response with CORS headers. Safe against client disconnects."""
        try:
            self.send_response(status_code)
            self.send_header('Content-Type', 'application/json')
            self._send_cors()
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
            self.end_headers()

            response_bytes = json.dumps(data, indent=2).encode('utf-8')
            self.wfile.write(response_bytes)
        except BrokenPipeError:
            # Client disconnected before we could finish sending
            print("⚠️  Client disconnected during response – ignoring.")
        except Exception as e:
            print(f"❌ Error sending response: {e}")

    def log_message(self, format, *args):
        """Custom log format"""
        print(f"[{self.date_time_string()}] {format % args}")

def main():
    """Main function to initialize and start the server"""
    PORT = int(os.environ.get("LOCALGPT_BACKEND_PORT", "8000"))
    host = os.environ.get("LOCALGPT_BACKEND_HOST", "127.0.0.1")
    try:
        # Initialize the database
        print("✅ Database initialized successfully")

        # Cleanup empty sessions on startup
        print("🧹 Cleaning up empty sessions...")
        cleanup_count = db.cleanup_empty_sessions()
        if cleanup_count > 0:
            print(f"✨ Cleaned up {cleanup_count} empty sessions")
        else:
            print("✨ No empty sessions to clean up")

        # Start the server
        with ReusableTCPServer((host, PORT), ChatHandler) as httpd:
            print(f"🚀 Starting localGPT backend server on {host}:{PORT}")
            print(f"📍 Session chat endpoint: http://localhost:{PORT}/sessions/<id>/messages")
            print(f"🔍 Health check: http://localhost:{PORT}/health")

            # Test Ollama connection
            client = OllamaClient()
            if client.is_ollama_running():
                models = client.list_models()
                print(f"✅ Ollama is running with {len(models)} models")
                print(f"📋 Available models: {', '.join(models[:3])}{'...' if len(models) > 3 else ''}")
            else:
                print("⚠️  Ollama is not running. Please start Ollama:")
                print("   Install: https://ollama.ai")
                print("   Run: ollama serve")

            print(f"\n🌐 Frontend should connect to: http://localhost:{PORT}")
            print("💬 Ready to chat!\n")

            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")

if __name__ == "__main__":
    main()

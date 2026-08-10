import json
import http.server
import socketserver
import email
import os
import uuid
from urllib.parse import urlparse
import requests  # 🆕 Import requests for making HTTP calls
import sys
from datetime import datetime

# Add parent directory to path so we can import rag_system modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import RAG system modules for complete metadata
try:
    from rag_system.main import PIPELINE_CONFIGS, OLLAMA_CONFIG
    RAG_SYSTEM_AVAILABLE = True
    print("✅ RAG system modules accessible from backend")
except ImportError as e:
    PIPELINE_CONFIGS = {}
    OLLAMA_CONFIG = {}
    RAG_SYSTEM_AVAILABLE = False
    print(f"⚠️ RAG system modules not available: {e}")

from ollama_client import OllamaClient
from database import db, generate_session_title
from typing import List, Dict, Any, Optional, Tuple
import re

PORT = 8000

# Base URL of the RAG API service. In Docker this is http://rag-api:8001.
RAG_API_URL = os.getenv("RAG_API_URL", "http://localhost:8001").rstrip("/")
RAG_API_TIMEOUT = float(os.getenv("RAG_API_TIMEOUT", "600"))
RAG_API_INDEX_TIMEOUT = float(os.getenv("RAG_API_INDEX_TIMEOUT", "3600"))

GENERATION_MODEL = os.getenv("GENERATION_MODEL") or OLLAMA_CONFIG.get("generation_model") or "qwen3.5:9b"
ENRICHMENT_MODEL = os.getenv("ENRICHMENT_MODEL") or OLLAMA_CONFIG.get("enrichment_model") or "qwen3.5:4b"

# Canonical snake_case option name -> (caster, accepted aliases).
# The frontend historically sent camelCase, so both spellings map to one key.
CHAT_OPTIONS: Dict[str, Tuple[Any, Tuple[str, ...]]] = {
    "model": (str, ()),
    "compose_sub_answers": (bool, ("composeSubAnswers",)),
    "query_decompose": (bool, ("queryDecompose", "decompose")),
    "ai_rerank": (bool, ("aiRerank",)),
    "context_expand": (bool, ("contextExpand",)),
    "verify": (bool, ()),
    "retrieval_k": (int, ("retrievalK",)),
    "context_window_size": (int, ("contextWindowSize",)),
    "reranker_top_k": (int, ("rerankerTopK",)),
    "retrieval_mode": (str, ("retrievalMode", "search_type", "searchType")),
    "provence_prune": (bool, ("provencePrune",)),
    "provence_threshold": (float, ("provenceThreshold",)),
    # Metadata filter object (roadmap 4.4). Validated by the RAG API, not here:
    # dict() copies a dict and raises on anything else, so a malformed value
    # still reaches the one validator and comes back as its 400.
    "filters": (dict, ()),
}

INDEX_OPTIONS: Dict[str, Tuple[Any, Tuple[str, ...]]] = {
    "chunk_size": (int, ("chunkSize",)),
    "window_size": (int, ("windowSize",)),
    "retrieval_mode": (str, ("retrievalMode",)),
    "enable_enrich": (bool, ("enableEnrich",)),
    "enable_latechunk": (bool, ("enableLatechunk", "latechunk")),
    "enable_docling_chunk": (bool, ("enableDoclingChunk", "doclingChunk")),
    "embedding_model": (str, ("embeddingModel",)),
    "enrich_model": (str, ("enrichModel",)),
    "overview_model_name": (str, ("overviewModelName", "overviewModel", "overview_model")),
    "batch_size_embed": (int, ("batchSizeEmbed",)),
    "batch_size_enrich": (int, ("batchSizeEnrich",)),
}


def normalize_options(data: dict, spec: Dict[str, Tuple[Any, Tuple[str, ...]]]) -> Dict[str, Any]:
    """Return canonical snake_case options from a body that may use either casing."""
    options: Dict[str, Any] = {}
    for canonical, (caster, aliases) in spec.items():
        for key in (canonical,) + tuple(aliases):
            if key not in data or data[key] is None:
                continue
            try:
                options[canonical] = caster(data[key])
            except (TypeError, ValueError):
                options[canonical] = data[key]
            break
    return options


# ---------------------------------------------------------------------------
# Gateway routing gate
# ---------------------------------------------------------------------------
# Retrieval-first cascade: escalate, don't pre-decide. When a session has
# documents linked, the gateway sends the message to the RAG API unless it is
# unmistakable smalltalk or a question about the assistant itself. There is no
# LLM call here — pre-retrieval LLM routing is measurably the weakest routing
# pattern available (see Documentation/research/), and the agent-side triage in
# rag_system/agent/loop.py remains the single LLM routing layer: it can still
# answer directly, so over-sending to RAG here is cheap and recoverable.
#
# Both regexes below are whole-message allowlists. Anything that is not an
# exact match falls through to RAG.

SMALLTALK_MAX_WORDS = 6

# Phrases that, on their own, carry no retrievable intent.
_SMALLTALK_CORE = (
    # greetings
    r"hi+", r"hey+", r"hello+", r"heya", r"yo", r"howdy", r"greetings",
    r"good\s+(?:morning|afternoon|evening|day)",
    r"how\s+are\s+(?:you|u|ya)(?:\s+doing)?", r"how'?s\s+it\s+going",
    r"what'?s\s+up", r"sup",
    # thanks
    r"thanks", r"thank\s+you", r"thx", r"ty", r"cheers",
    r"much\s+appreciated", r"appreciate\s+it",
    # farewells
    r"bye+", r"goodbye", r"good\s*night", r"see\s+(?:you|ya)",
    r"talk\s+to\s+you\s+later", r"take\s+care", r"later",
    # acknowledgements / fillers that end a turn
    r"ok(?:ay)?", r"kk", r"alright", r"got\s+it", r"understood", r"i\s+see",
    r"sounds\s+good", r"no\s+problem", r"np", r"never\s*mind", r"nvm",
    r"cool", r"nice", r"awesome", r"perfect", r"great",
    r"yes", r"yeah", r"yep", r"nope", r"no",
    r"sorry", r"please", r"lol", r"haha+",
)

# Words allowed *alongside* a core phrase but never sufficient on their own.
_SMALLTALK_FILLER = (
    r"there", r"again", r"all", r"everyone", r"folks", r"guys", r"team",
    r"friend", r"buddy", r"mate", r"man", r"dude", r"bot", r"assistant",
    r"a\s+lot", r"so\s+much", r"very\s+much", r"so", r"much", r"very",
    r"then", r"too", r"and", r"well", r"my", r"you",
)


def _alternation(*groups: Tuple[str, ...]) -> str:
    """Join regex phrases longest-first so the widest match is tried first."""
    phrases = [p for group in groups for p in group]
    phrases.sort(key=len, reverse=True)
    return "|".join(phrases)


# Whole message consists only of allowlisted smalltalk/filler phrases.
_SMALLTALK_RE = re.compile(
    r"^\W*(?:{alt})(?:\W+(?:{alt}))*\W*$".format(
        alt=_alternation(_SMALLTALK_CORE, _SMALLTALK_FILLER)
    ),
    re.IGNORECASE,
)

# ...and at least one of those phrases is a *core* smalltalk phrase.
_SMALLTALK_CORE_RE = re.compile(
    r"(?<!\w)(?:{alt})(?!\w)".format(alt=_alternation(_SMALLTALK_CORE)),
    re.IGNORECASE,
)

# Questions addressed to the assistant about itself. No document can answer
# these, so they never need retrieval.
_ASSISTANT_META_RE = re.compile(
    r"""^\W*(?:so|and|btw|hey|hi|hello)?\W*(?:
          who\s+(?:are|r)\s+(?:you|u)
        | what\s+(?:are|r)\s+(?:you|u)
        | what(?:'s|\s+is)\s+your\s+name
        | (?:what|which)\s+(?:llm\s+|ai\s+|language\s+)?model
            (?:\s+(?:are|r)\s+(?:you|u)|\s+do\s+you\s+(?:use|run(?:\s+on)?))
        | (?:what|which)\s+(?:llm|ai)\s+(?:are|r)\s+(?:you|u)
        | what\s+version\s+(?:are|r)\s+(?:you|u)
        | what\s+(?:are\s+you|do\s+you)\s+run(?:ning)?(?:\s+on)?
        | are\s+you\s+(?:an?\s+)?
            (?:ai|bot|human|robot|chatgpt|gpt-?\d*|claude|llm|language\s+model)
        | what\s+can\s+you\s+do
        | who\s+(?:made|built|created|trained)\s+you
        | tell\s+me\s+about\s+yourself
        | introduce\s+yourself
    )\W*$""",
    re.IGNORECASE | re.VERBOSE,
)


def is_smalltalk_or_meta(message: str) -> bool:
    """True when a message is pure smalltalk or a question about the assistant.

    Deterministic and allocation-cheap: two anchored regexes and a word count.
    Deliberately conservative — an unmatched message routes to RAG.
    """
    text = (message or "").strip()
    if not text:
        return True
    if _ASSISTANT_META_RE.match(text):
        return True
    if len(text.split()) > SMALLTALK_MAX_WORDS:
        return False
    return bool(_SMALLTALK_RE.match(text) and _SMALLTALK_CORE_RE.search(text))


def should_use_rag(message: str, idx_ids: Optional[List[str]], force_rag: bool = False) -> bool:
    """Decide whether one chat message goes to the RAG API or straight to Ollama.

    1. ``force_rag`` → RAG, unconditionally.
    2. No indexes linked to the session → direct LLM (nothing to retrieve from).
    3. Smalltalk / assistant-meta → direct LLM.
    4. Everything else → RAG.
    """
    if force_rag:
        return True
    if not idx_ids:
        return False
    return not is_smalltalk_or_meta(message)


def default_index_metadata() -> Dict[str, Any]:
    """Index metadata defaults taken from the live RAG pipeline configuration."""
    config = PIPELINE_CONFIGS.get('default', {}) if RAG_SYSTEM_AVAILABLE else {}
    retrieval = config.get('retrieval', {})
    indexing = config.get('indexing', {})
    return {
        'chunk_size': 512,
        'retrieval_mode': retrieval.get('search_type', 'hybrid'),
        'window_size': config.get('contextual_enricher', {}).get('window_size', 1),
        'embedding_model': os.getenv('EMBEDDING_MODEL') or config.get('embedding_model_name') or 'microsoft/harrier-oss-v1-0.6b',
        'enrich_model': ENRICHMENT_MODEL,
        'overview_model': ENRICHMENT_MODEL,
        'enable_enrich': config.get('contextual_enricher', {}).get('enabled', True),
        'latechunk': retrieval.get('latechunk', {}).get('enabled', False),
        'docling_chunk': True,
        'batch_size_embed': indexing.get('embedding_batch_size', 50),
        'batch_size_enrich': indexing.get('enrichment_batch_size', 25),
    }


# 🆕 Threaded TCPServer with address reuse enabled. Threading keeps a slow RAG
# query from blocking every other request, including the RAG API's callback.
class ReusableTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

class ChatHandler(http.server.BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.ollama_client = OllamaClient()
        super().__init__(*args, **kwargs)
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, DELETE, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_GET(self):
        """Handle GET requests"""
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
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/chat':
            self.handle_chat()
        elif parsed_path.path == '/sessions':
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
        elif parsed_path.path.startswith('/sessions/') and parsed_path.path.endswith('/messages/save'):
            session_id = parsed_path.path.split('/')[-3]
            self.handle_save_messages(session_id)
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
    
    def handle_chat(self):
        """Handle legacy chat requests (without sessions)"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            message = data.get('message', '')
            model = data.get('model', GENERATION_MODEL)
            conversation_history = data.get('conversation_history', [])
            
            if not message:
                self.send_json_response({
                    "error": "Message is required"
                }, status_code=400)
                return
            
            # Check if Ollama is running
            if not self.ollama_client.is_ollama_running():
                self.send_json_response({
                    "error": "Ollama is not running. Please start Ollama first."
                }, status_code=503)
                return
            
            # Get response from Ollama
            response = self.ollama_client.chat(message, model, conversation_history)
            
            self.send_json_response({
                "response": response,
                "model": model,
                "message_count": len(conversation_history) + 1
            })
            
        except json.JSONDecodeError:
            self.send_json_response({
                "error": "Invalid JSON"
            }, status_code=400)
        except Exception as e:
            self.send_json_response({
                "error": f"Server error: {str(e)}"
            }, status_code=500)
    
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
            model = data.get('model', GENERATION_MODEL)

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

            options = normalize_options(data, CHAT_OPTIONS)

            # 🎯 ROUTING: deterministic gate, no LLM call (see should_use_rag)
            idx_ids = db.get_indexes_for_session(session_id)
            force_rag = bool(data.get("force_rag", data.get("forceRag", False)))
            if force_rag:
                options["force_rag"] = True
            # An explicit metadata filter (roadmap 4.4) is a statement that this
            # is a document question — gateway twin of the agent-side rule.
            if options.get("filters"):
                force_rag = True
            use_rag = should_use_rag(message, idx_ids, force_rag=force_rag)

            if use_rag:
                # 🔍 --- Use RAG Pipeline for Document-Related Queries ---
                print(f"🔍 Using RAG pipeline for document query: '{message[:50]}...'")
                response_text, source_docs = self._handle_rag_query(session_id, message, options, idx_ids)
            else:
                # ⚡ --- Use Direct LLM for General Queries (FAST) ---
                print(f"⚡ Using direct LLM for general query: '{message[:50]}...'")
                response_text, source_docs = self._handle_direct_llm_query(
                    session_id, message, session, options.get('model')
                )

            # Add AI response to database (sources go into metadata so reloaded
            # sessions can still render attribution)
            ai_message_id = db.add_message(
                session_id, response_text, "assistant",
                metadata={'source_documents': source_docs} if source_docs else None
            )
            
            updated_session = db.get_session(session_id)
            
            # Send response with proper error handling
            self.send_json_response({
                "response": response_text,
                "session": updated_session,
                "source_documents": source_docs,
                "used_rag": use_rag
            })
            
        except BrokenPipeError:
            # Client disconnected - this is normal for long queries, just log it
            preview = message[:30] if 'message' in locals() else ''
            print(f"⚠️  Client disconnected during RAG processing for query: '{preview}...'")
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
                print(f"⚠️  Client disconnected during error response")
    
    def _handle_direct_llm_query(self, session_id: str, message: str, session: dict, model: Optional[str] = None):
        """
        Handle query using direct Ollama client with thinking disabled for speed.

        Returns:
            tuple: (response_text, empty_source_docs)
        """
        try:
            # Get conversation history for context
            conversation_history = db.get_conversation_history(session_id)

            # Per-request override wins, then the session's model, then the configured default
            model = model or session.get('model') or GENERATION_MODEL

            # Direct Ollama call with thinking disabled for speed
            response_text = self.ollama_client.chat(
                message=message,
                model=model,
                conversation_history=conversation_history,
                enable_thinking=False  # ⚡ DISABLE THINKING FOR SPEED
            )
            
            return response_text, []  # No source docs for direct LLM
            
        except Exception as e:
            print(f"❌ Direct LLM error: {e}")
            return f"Error processing query: {str(e)}", []
    
    def _handle_rag_query(self, session_id: str, message: str, options: Dict[str, Any], idx_ids: List[str]):
        """
        Handle query using the full RAG pipeline (delegates to the RAG API at RAG_API_URL).

        Returns:
            tuple[str, List[dict]]: (response_text, source_documents)
        """
        # Defaults
        response_text = ""
        source_docs: List[dict] = []

        # Build payload for RAG API
        rag_api_url = f"{RAG_API_URL}/chat"
        table_name = f"text_pages_{idx_ids[-1]}" if idx_ids else None
        payload: Dict[str, Any] = {
            "query": message,
            "session_id": session_id,
        }
        if table_name:
            payload["table_name"] = table_name

        payload.update(options)

        try:
            rag_response = requests.post(rag_api_url, json=payload, timeout=RAG_API_TIMEOUT)
            if rag_response.status_code == 200:
                rag_data = rag_response.json()
                response_text = rag_data.get("answer", "No answer found.")
                source_docs = rag_data.get("source_documents", [])
            else:
                response_text = f"Error from RAG API ({rag_response.status_code}): {rag_response.text}"
                print(f"❌ RAG API error: {response_text}")
        except requests.exceptions.Timeout:
            response_text = f"The RAG API did not respond within {RAG_API_TIMEOUT:.0f}s."
            print(f"❌ RAG API request timed out after {RAG_API_TIMEOUT:.0f}s ({rag_api_url}).")
        except requests.exceptions.ConnectionError:
            response_text = f"Could not connect to the RAG API server at {RAG_API_URL}. Please ensure it is running."
            print(f"❌ Connection to RAG API failed ({rag_api_url}).")
        except Exception as e:
            response_text = f"Error processing RAG query: {str(e)}"
            print(f"❌ RAG processing error: {e}")

        # Strip any <think>/<thinking> tags that might slip through
        response_text = re.sub(r'<(think|thinking)>.*?</\1>', '', response_text, flags=re.DOTALL | re.IGNORECASE).strip()

        return response_text, source_docs

    def handle_delete_session(self, session_id: str):
        """Delete a session and its messages"""
        try:
            deleted = db.delete_session(session_id)
            if deleted:
                self.send_json_response({'deleted': deleted})
            else:
                self.send_json_response({'error': 'Session not found'}, status_code=404)
        except Exception as e:
            self.send_json_response({'error': str(e)}, status_code=500)
    
    def parse_multipart_files(self, field_name: str = 'files') -> List[Tuple[str, bytes]]:
        """Parse a multipart/form-data body and return (filename, content) for one field.

        Uses the stdlib email parser; `cgi` was removed from Python 3.13.
        """
        content_type = self.headers.get('Content-Type', '') or ''
        if not content_type.lower().startswith('multipart/form-data'):
            return []

        length = int(self.headers.get('Content-Length', 0) or 0)
        if length <= 0:
            return []

        body = self.rfile.read(length)
        prologue = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode('utf-8')
        message = email.message_from_bytes(prologue + body)
        if not message.is_multipart():
            return []

        files: List[Tuple[str, bytes]] = []
        for part in message.walk():
            if part.is_multipart():
                continue
            filename = part.get_filename()
            if not filename:
                continue
            if field_name and part.get_param('name', header='content-disposition') != field_name:
                continue
            payload = part.get_payload(decode=True)
            if payload is None:
                continue
            files.append((os.path.basename(filename), payload))
        return files

    def handle_file_upload(self, session_id: str):
        """Handle file uploads, save them, and associate with the session."""
        uploaded_files = []
        incoming = self.parse_multipart_files('files')
        if incoming:
            upload_dir = "shared_uploads"
            os.makedirs(upload_dir, exist_ok=True)

            for filename, content in incoming:
                # Create a unique filename to avoid overwrites
                unique_filename = f"{uuid.uuid4()}_{filename}"
                file_path = os.path.join(upload_dir, unique_filename)

                with open(file_path, 'wb') as f:
                    f.write(content)

                # Store the absolute path for the indexing service
                absolute_file_path = os.path.abspath(file_path)
                db.add_document_to_session(session_id, absolute_file_path)
                uploaded_files.append({"filename": filename, "stored_path": absolute_file_path})

        if not uploaded_files:
            self.send_json_response({"error": "No files were uploaded"}, status_code=400)
            return
            
        self.send_json_response({
            "message": f"Successfully uploaded {len(uploaded_files)} files.",
            "uploaded_files": uploaded_files
        })

    def read_json_body(self) -> dict:
        """Read and decode an optional JSON request body. Returns {} when absent."""
        length = int(self.headers.get('Content-Length', 0) or 0)
        if length <= 0:
            return {}
        body = self.rfile.read(length)
        try:
            parsed = json.loads(body.decode('utf-8'))
        except (ValueError, UnicodeDecodeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def handle_index_documents(self, session_id: str):
        """Triggers indexing for all documents in a session."""
        print(f"🔥 Received request to index documents for session {session_id[:8]}...")
        try:
            options = normalize_options(self.read_json_body(), INDEX_OPTIONS)

            file_paths = db.get_documents_for_session(session_id)
            if not file_paths:
                self.send_json_response({"message": "No documents to index for this session."}, status_code=200)
                return

            print(f"Found {len(file_paths)} documents to index. Sending to RAG API...")

            rag_api_url = f"{RAG_API_URL}/index"
            payload: Dict[str, Any] = {"file_paths": file_paths, "session_id": session_id}
            payload.update(options)
            rag_response = requests.post(rag_api_url, json=payload, timeout=RAG_API_INDEX_TIMEOUT)

            if rag_response.status_code == 200:
                print("✅ RAG API successfully indexed documents.")
                # Merge key config values into index metadata
                idx_meta: Dict[str, Any] = {
                    "session_linked": True,
                    "retrieval_mode": options.get("retrieval_mode", "hybrid"),
                }
                idx_meta.update({k: v for k, v in options.items() if k != "retrieval_mode"})
                try:
                    db.update_index_metadata(session_id, idx_meta)  # session_id used as index_id in text table naming
                except Exception as e:
                    print(f"⚠️ Failed to update index metadata for session index: {e}")
                self.send_json_response(rag_response.json())
            else:
                error_info = rag_response.text
                print(f"❌ RAG API indexing failed ({rag_response.status_code}): {error_info}")
                self.send_json_response({"error": f"Indexing failed: {error_info}"}, status_code=500)

        except requests.exceptions.Timeout:
            print(f"❌ RAG API indexing timed out after {RAG_API_INDEX_TIMEOUT:.0f}s.")
            self.send_json_response({
                "error": f"Indexing did not complete within {RAG_API_INDEX_TIMEOUT:.0f}s."
            }, status_code=504)
        except requests.exceptions.ConnectionError:
            print(f"❌ Connection to RAG API failed ({RAG_API_URL}).")
            self.send_json_response({
                "error": f"Could not connect to the RAG API server at {RAG_API_URL}."
            }, status_code=502)
        except Exception as e:
            print(f"❌ Exception during indexing: {str(e)}")
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
                "microsoft/harrier-oss-v1-0.6b",  # shipped default
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
            
            # Add complete metadata from RAG system configuration if available
            if RAG_SYSTEM_AVAILABLE and PIPELINE_CONFIGS.get('default'):
                complete_metadata = {
                    'status': 'created',
                    'metadata_source': 'rag_system_config',
                    'created_at': datetime.now().isoformat(),
                    'note': 'Default configuration from RAG system',
                }
                complete_metadata.update(default_index_metadata())
                # Merge with any provided metadata
                complete_metadata.update(metadata)
                metadata = complete_metadata


            idx_id = db.create_index(name, description, metadata)
            self.send_json_response({'index_id': idx_id}, status_code=201)
        except Exception as e:
            self.send_json_response({'error': str(e)}, status_code=500)
    
    def handle_index_file_upload(self, index_id: str):
        """Reuse file upload logic but store docs under index."""
        uploaded_files=[]
        incoming = self.parse_multipart_files('files')
        if incoming:
            upload_dir='shared_uploads'
            os.makedirs(upload_dir, exist_ok=True)
            for filename, content in incoming:
                unique=f"{uuid.uuid4()}_{filename}"
                path=os.path.join(upload_dir, unique)
                with open(path,'wb') as out: out.write(content)
                db.add_document_to_index(index_id, filename, os.path.abspath(path))
                uploaded_files.append({'filename':filename,'stored_path':os.path.abspath(path)})
        if not uploaded_files:
            self.send_json_response({'error':'No files uploaded'}, status_code=400); return
        self.send_json_response({'message':f"Uploaded {len(uploaded_files)} files","uploaded_files":uploaded_files})
    
    def handle_build_index(self, index_id: str):
        try:
            # Parse request body for optional flags and configuration. Options the
            # caller omits are left out of the payload so the RAG pipeline defaults apply.
            # Read before any early return so the request body is always consumed.
            options = normalize_options(self.read_json_body(), INDEX_OPTIONS)

            index=db.get_index(index_id)
            if not index:
                self.send_json_response({'error':'Index not found'}, status_code=404); return
            file_paths=[d['stored_path'] for d in index.get('documents',[])]
            if not file_paths:
                self.send_json_response({'error':'No documents to index'}, status_code=400); return

            # Delegate to the RAG API, same as session indexing
            rag_api_url = f"{RAG_API_URL}/index"
            # Use the index's dedicated LanceDB table so retrieval matches
            table_name = index.get("vector_table_name")
            payload: Dict[str, Any] = {
                "file_paths": file_paths,
                "session_id": index_id,  # reuse index_id for progress tracking
                "table_name": table_name,
            }
            payload.update(options)

            rag_resp = requests.post(rag_api_url, json=payload, timeout=RAG_API_INDEX_TIMEOUT)
            if rag_resp.status_code==200:
                meta_updates: Dict[str, Any] = dict(options)
                meta_updates["status"] = "built"
                if "enable_latechunk" in meta_updates:
                    meta_updates["latechunk"] = meta_updates.pop("enable_latechunk")
                if "enable_docling_chunk" in meta_updates:
                    meta_updates["docling_chunk"] = meta_updates.pop("enable_docling_chunk")
                if "overview_model_name" in meta_updates:
                    meta_updates["overview_model"] = meta_updates.pop("overview_model_name")
                try:
                    db.update_index_metadata(index_id, meta_updates)
                except Exception as e:
                    print(f"⚠️ Failed to update index metadata: {e}")

                self.send_json_response({
                    "response": rag_resp.json(),
                    **meta_updates
                })
            else:
                # Gracefully handle scenario where table already exists (idempotent build)
                try:
                    err_json = rag_resp.json()
                except Exception:
                    err_json = {}
                err_text = err_json.get('error') if isinstance(err_json, dict) else rag_resp.text
                if err_text and 'already exists' in err_text:
                    # Treat as non-fatal; return message indicating index previously built
                    self.send_json_response({
                        "message": "Index already built – skipping rebuild.",
                        "note": err_text
                })
                else:
                    self.send_json_response({"error":f"RAG indexing failed: {rag_resp.text}"}, status_code=500)
        except requests.exceptions.Timeout:
            self.send_json_response({
                "error": f"Indexing did not complete within {RAG_API_INDEX_TIMEOUT:.0f}s."
            }, status_code=504)
        except requests.exceptions.ConnectionError:
            self.send_json_response({
                "error": f"Could not connect to the RAG API server at {RAG_API_URL}."
            }, status_code=502)
        except Exception as e:
            self.send_json_response({'error':str(e)}, status_code=500)

    def handle_link_index_to_session(self, session_id: str, index_id: str):
        try:
            db.link_index_to_session(session_id, index_id)
            self.send_json_response({'message':'Index linked to session'})
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
            deleted = db.delete_index(index_id)
            if deleted:
                self.send_json_response({'message': 'Index deleted successfully', 'index_id': index_id})
            else:
                self.send_json_response({'error': 'Index not found'}, status_code=404)
        except Exception as e:
            self.send_json_response({'error': str(e)}, status_code=500)

    def handle_save_messages(self, session_id: str):
        """Persist a completed streamed turn (the browser streams straight from the
        RAG API, so the gateway never sees those messages otherwise)."""
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
            user_message = (data.get('user_message') or '').strip()
            assistant_message = (data.get('assistant_message') or '').strip()
            source_documents = data.get('source_documents') or []
            steps = data.get('steps')

            if not user_message or not assistant_message:
                self.send_json_response({"error": "user_message and assistant_message are required"}, status_code=400)
                return

            if session['message_count'] == 0:
                db.update_session_title(session_id, generate_session_title(user_message))

            ai_metadata: Dict[str, Any] = {}
            if source_documents:
                ai_metadata['source_documents'] = source_documents
            if isinstance(steps, list) and steps:
                ai_metadata['steps'] = steps
            user_message_id = db.add_message(session_id, user_message, "user")
            ai_message_id = db.add_message(session_id, assistant_message, "assistant", metadata=ai_metadata or None)

            self.send_json_response({
                "session": db.get_session(session_id),
                "user_message_id": user_message_id,
                "ai_message_id": ai_message_id,
            })
        except Exception as e:
            self.send_json_response({"error": str(e)}, status_code=500)

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
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
            self.send_header('Access-Control-Allow-Credentials', 'true')
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
        with ReusableTCPServer(("", PORT), ChatHandler) as httpd:
            print(f"🚀 Starting localGPT backend server on port {PORT}")
            print(f"📍 Chat endpoint: http://localhost:{PORT}/chat")
            print(f"🔍 Health check: http://localhost:{PORT}/health")
            print(f"🧠 RAG API: {RAG_API_URL}")
            print(f"🤖 Default generation model: {GENERATION_MODEL}")

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
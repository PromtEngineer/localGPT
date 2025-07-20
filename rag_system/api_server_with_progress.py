import json
import threading
import time
from typing import Dict, List, Any
import logging
from urllib.parse import urlparse, parse_qs
import http.server
import socketserver

# Import the core logic and batch processing utilities
from rag_system.main import get_agent
from rag_system.utils.batch_processor import ProgressTracker, timer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global progress tracking storage
ACTIVE_PROGRESS_SESSIONS: Dict[str, Dict[str, Any]] = {}

# --- Global Singleton for the RAG Agent ---
print("🧠 Initializing RAG Agent... (This may take a moment)")
RAG_AGENT = get_agent()
if RAG_AGENT is None:
    print("❌ Critical error: RAG Agent could not be initialized. Exiting.")
    exit(1)
print("✅ RAG Agent initialized successfully.")

class ServerSentEventsHandler:
    """Handler for Server-Sent Events (SSE) for real-time progress updates"""
    
    active_connections: Dict[str, Any] = {}
    
    @classmethod
    def add_connection(cls, session_id: str, response_handler):
        """Add a new SSE connection"""
        cls.active_connections[session_id] = response_handler
        logger.info(f"SSE connection added for session: {session_id}")
    
    @classmethod
    def remove_connection(cls, session_id: str):
        """Remove an SSE connection"""
        if session_id in cls.active_connections:
            del cls.active_connections[session_id]
            logger.info(f"SSE connection removed for session: {session_id}")
    
    @classmethod
    def send_event(cls, session_id: str, event_type: str, data: Dict[str, Any]):
        """Send an SSE event to a specific session"""
        if session_id not in cls.active_connections:
            return
        
        try:
            handler = cls.active_connections[session_id]
            event_data = json.dumps(data)
            message = f"event: {event_type}\ndata: {event_data}\n\n"
            handler.wfile.write(message.encode('utf-8'))
            handler.wfile.flush()
        except Exception as e:
            logger.error(f"Failed to send SSE event: {e}")
            cls.remove_connection(session_id)

class RealtimeProgressTracker(ProgressTracker):
    """Enhanced ProgressTracker that sends updates via Server-Sent Events"""
    
    def __init__(self, total_items: int, operation_name: str, session_id: str):
        super().__init__(total_items, operation_name)
        self.session_id = session_id
        self.last_update = 0
        self.update_interval = 1  # Update every 1 second
        
        # Initialize session progress
        ACTIVE_PROGRESS_SESSIONS[session_id] = {
            "operation_name": operation_name,
            "total_items": total_items,
            "processed_items": 0,
            "errors_encountered": 0,
            "start_time": self.start_time,
            "status": "running",
            "current_step": "",
            "eta_seconds": 0,
            "throughput": 0,
            "progress_percentage": 0
        }
        
        # Send initial progress update
        self._send_progress_update()
    
    def update(self, items_processed: int, errors: int = 0, current_step: str = ""):
        """Update progress and send notification"""
        super().update(items_processed, errors)
        
        # Update session data
        session_data = ACTIVE_PROGRESS_SESSIONS.get(self.session_id)
        if session_data:
            session_data.update({
                "processed_items": self.processed_items,
                "errors_encountered": self.errors_encountered,
                "current_step": current_step,
                "progress_percentage": (self.processed_items / self.total_items) * 100,
            })
            
            # Calculate throughput and ETA
            elapsed = time.time() - self.start_time
            if elapsed > 0:
                session_data["throughput"] = self.processed_items / elapsed
                remaining = self.total_items - self.processed_items
                session_data["eta_seconds"] = remaining / session_data["throughput"] if session_data["throughput"] > 0 else 0
        
        # Send update if enough time has passed
        current_time = time.time()
        if current_time - self.last_update >= self.update_interval:
            self._send_progress_update()
            self.last_update = current_time
    
    def finish(self):
        """Mark progress as finished and send final update"""
        super().finish()
        
        # Update session status
        session_data = ACTIVE_PROGRESS_SESSIONS.get(self.session_id)
        if session_data:
            session_data.update({
                "status": "completed",
                "progress_percentage": 100,
                "eta_seconds": 0
            })
        
        # Send final update
        self._send_progress_update(final=True)
    
    def _send_progress_update(self, final: bool = False):
        """Send progress update via Server-Sent Events"""
        session_data = ACTIVE_PROGRESS_SESSIONS.get(self.session_id, {})
        
        event_data = {
            "session_id": self.session_id,
            "progress": session_data.copy(),
            "final": final,
            "timestamp": time.time()
        }
        
        ServerSentEventsHandler.send_event(self.session_id, "progress", event_data)

def run_indexing_with_progress(file_paths: List[str], session_id: str):
    """Enhanced indexing function with real-time progress tracking"""
    from rag_system.pipelines.indexing_pipeline import IndexingPipeline
    from rag_system.utils.ollama_client import OllamaClient
    import json
    
    try:
        # Send initial status
        ServerSentEventsHandler.send_event(session_id, "status", {
            "message": "Initializing indexing pipeline...",
            "session_id": session_id
        })
        
        # Load configuration
        config_file = "batch_indexing_config.json"
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            # Fallback to default config
            config = {
                "embedding_model_name": "Qwen/Qwen3-Embedding-0.6B",
                "indexing": {
                    "embedding_batch_size": 50,
                    "enrichment_batch_size": 10,
                    "enable_progress_tracking": True
                },
                "contextual_enricher": {"enabled": True, "window_size": 1},
                "retrievers": {
                    "dense": {"enabled": True, "lancedb_table_name": "default_text_table"},
                    "bm25": {"enabled": True, "index_name": "default_bm25_index"}
                },
                "storage": {
                    "chunk_store_path": "./index_store/chunks/chunks.pkl",
                    "lancedb_uri": "./index_store/lancedb",
                    "bm25_path": "./index_store/bm25"
                }
            }
        
        # Initialize components
        ollama_client = OllamaClient()
        ollama_config = {
            "generation_model": "llama3.2:1b",
            "embedding_model": "mxbai-embed-large"
        }
        
        # Create enhanced pipeline
        pipeline = IndexingPipeline(config, ollama_client, ollama_config)
        
        # Create progress tracker for the overall process
        total_steps = 6  # Rough estimate of pipeline steps
        step_tracker = RealtimeProgressTracker(total_steps, "Document Indexing", session_id)
        
        with timer("Complete Indexing Pipeline"):
            try:
                # Step 1: Document Processing
                step_tracker.update(1, current_step="Processing documents...")
                
                # Run the indexing pipeline
                pipeline.run(file_paths)
                
                # Update progress through the steps
                step_tracker.update(1, current_step="Chunking completed...")
                step_tracker.update(1, current_step="BM25 indexing completed...")
                step_tracker.update(1, current_step="Contextual enrichment completed...")
                step_tracker.update(1, current_step="Vector embeddings completed...")
                step_tracker.update(1, current_step="Indexing finalized...")
                
                step_tracker.finish()
                
                # Send completion notification
                ServerSentEventsHandler.send_event(session_id, "completion", {
                    "message": f"Successfully indexed {len(file_paths)} file(s)",
                    "file_count": len(file_paths),
                    "session_id": session_id
                })
                
            except Exception as e:
                # Send error notification
                ServerSentEventsHandler.send_event(session_id, "error", {
                    "message": str(e),
                    "session_id": session_id
                })
                raise
        
    except Exception as e:
        logger.error(f"Indexing failed for session {session_id}: {e}")
        ServerSentEventsHandler.send_event(session_id, "error", {
            "message": str(e),
            "session_id": session_id
        })
        raise

class EnhancedRagApiHandler(http.server.BaseHTTPRequestHandler):
    """Enhanced API handler with progress tracking support"""
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests for frontend integration."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        """Handle GET requests for progress status and SSE streams"""
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/progress':
            self.handle_progress_status()
        elif parsed_path.path == '/stream':
            self.handle_progress_stream()
        else:
            self.send_json_response({"error": "Not Found"}, status_code=404)

    def do_POST(self):
        """Handle POST requests for chat and indexing."""
        parsed_path = urlparse(self.path)

        if parsed_path.path == '/chat':
            self.handle_chat()
        elif parsed_path.path == '/index':
            self.handle_index_with_progress()
        else:
            self.send_json_response({"error": "Not Found"}, status_code=404)

    def handle_chat(self):
        """Handles a chat query by calling the agentic RAG pipeline."""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            query = data.get('query')
            if not query:
                self.send_json_response({"error": "Query is required"}, status_code=400)
                return

            # Use the single, persistent agent instance to run the query
            result = RAG_AGENT.run(query)
            
            # The result is a dict, so we need to dump it to a JSON string
            self.send_json_response(result)

        except json.JSONDecodeError:
            self.send_json_response({"error": "Invalid JSON"}, status_code=400)
        except Exception as e:
            self.send_json_response({"error": f"Server error: {str(e)}"}, status_code=500)

    def handle_index_with_progress(self):
        """Triggers the document indexing pipeline with real-time progress tracking."""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            file_paths = data.get('file_paths')
            session_id = data.get('session_id')
            
            if not file_paths or not isinstance(file_paths, list):
                self.send_json_response({
                    "error": "A 'file_paths' list is required."
                }, status_code=400)
                return
            
            if not session_id:
                self.send_json_response({
                    "error": "A 'session_id' is required for progress tracking."
                }, status_code=400)
                return

            # Start indexing in a separate thread to avoid blocking
            def run_indexing_thread():
                try:
                    run_indexing_with_progress(file_paths, session_id)
                except Exception as e:
                    logger.error(f"Indexing thread failed: {e}")

            thread = threading.Thread(target=run_indexing_thread)
            thread.daemon = True
            thread.start()

            # Return immediate response
            self.send_json_response({
                "message": f"Indexing started for {len(file_paths)} file(s)",
                "session_id": session_id,
                "status": "started",
                "progress_stream_url": f"http://localhost:8001/stream?session_id={session_id}"
            })
            
        except json.JSONDecodeError:
            self.send_json_response({"error": "Invalid JSON"}, status_code=400)
        except Exception as e:
            self.send_json_response({"error": f"Failed to start indexing: {str(e)}"}, status_code=500)

    def handle_progress_status(self):
        """Handle GET requests for current progress status"""
        parsed_url = urlparse(self.path)
        params = parse_qs(parsed_url.query)
        session_id = params.get('session_id', [None])[0]
        
        if not session_id:
            self.send_json_response({"error": "session_id is required"}, status_code=400)
            return
        
        progress_data = ACTIVE_PROGRESS_SESSIONS.get(session_id)
        if not progress_data:
            self.send_json_response({"error": "No active progress for this session"}, status_code=404)
            return
        
        self.send_json_response({
            "session_id": session_id,
            "progress": progress_data
        })

    def handle_progress_stream(self):
        """Handle Server-Sent Events stream for real-time progress"""
        parsed_url = urlparse(self.path)
        params = parse_qs(parsed_url.query)
        session_id = params.get('session_id', [None])[0]
        
        if not session_id:
            self.send_response(400)
            self.end_headers()
            return
        
        # Set up SSE headers
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'keep-alive')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        # Add this connection to the SSE handler
        ServerSentEventsHandler.add_connection(session_id, self)
        
        # Send initial connection message
        initial_message = json.dumps({
            "session_id": session_id,
            "message": "Progress stream connected",
            "timestamp": time.time()
        })
        self.wfile.write(f"event: connected\ndata: {initial_message}\n\n".encode('utf-8'))
        self.wfile.flush()
        
        # Keep connection alive
        try:
            while session_id in ServerSentEventsHandler.active_connections:
                time.sleep(1)
                # Send heartbeat
                heartbeat = json.dumps({"type": "heartbeat", "timestamp": time.time()})
                self.wfile.write(f"event: heartbeat\ndata: {heartbeat}\n\n".encode('utf-8'))
                self.wfile.flush()
        except Exception as e:
            logger.info(f"SSE connection closed for session {session_id}: {e}")
        finally:
            ServerSentEventsHandler.remove_connection(session_id)
    
    def send_json_response(self, data, status_code=200):
        """Utility to send a JSON response with CORS headers."""
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        response = json.dumps(data, indent=2)
        self.wfile.write(response.encode('utf-8'))

def start_enhanced_server(port=8000):
    """Start the enhanced API server with a reusable TCP socket."""
    
    # Use a custom TCPServer that allows address reuse
    class ReusableTCPServer(socketserver.TCPServer):
        allow_reuse_address = True

    with ReusableTCPServer(("", port), EnhancedRagApiHandler) as httpd:
        print(f"🚀 Starting Enhanced RAG API server on port {port}")
        print(f"💬 Chat endpoint: http://localhost:{port}/chat")
        print(f"✨ Indexing endpoint: http://localhost:{port}/index")
        print(f"📊 Progress endpoint: http://localhost:{port}/progress")
        print(f"🌊 Progress stream: http://localhost:{port}/stream")
        print(f"📈 Real-time progress tracking enabled via Server-Sent Events!")
        httpd.serve_forever()

if __name__ == '__main__':
    # Start the server on a dedicated thread
    server_thread = threading.Thread(target=start_enhanced_server)
    server_thread.daemon = True
    server_thread.start()
    
    print("🚀 Enhanced RAG API server with progress tracking is running.")
    print("Press Ctrl+C to stop.")
    
    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping server...") 
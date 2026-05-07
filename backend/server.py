import json
import os
import uuid
from datetime import datetime
import re
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

# Routes

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "ok",
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
        db.link_index_to_session(session_id, index_id)
        return {"message": "Index linked to session"}
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
        data = db.list_indexes()
        return {"indexes": data, "total": len(data)}
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

@app.post("/indexes/{index_id}/build")
async def build_index(index_id: str, request: Request):
    """Build an index from uploaded documents"""
    try:
        index = db.get_index(index_id)
        if not index:
            raise HTTPException(status_code=404, detail="Index not found")

        file_paths = [d['stored_path'] for d in index.get('documents', [])]
        if not file_paths:
            raise HTTPException(status_code=400, detail="No documents to index")

        # Parse request body for optional flags and configuration
        data = await request.json() if await request.body() else {}
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

        def is_large_indexing_model(model: str | None) -> bool:
            if not model:
                return False
            lowered = model.lower()
            return any(token in lowered for token in ("gpt-oss", "120b", "70b", "large", "cloud"))

        if is_large_indexing_model(enrich_model):
            indexing_model_warnings.append(
                f"Replaced enrichment model '{enrich_model}' with qwen3:0.6b for indexing safety."
            )
            enrich_model = "qwen3:0.6b"
        if is_large_indexing_model(overview_model):
            indexing_model_warnings.append(
                f"Replaced overview model '{overview_model}' with qwen3:0.6b for indexing safety."
            )
            overview_model = "qwen3:0.6b"

        window_size = max(0, min(window_size, 2))
        batch_size_enrich = max(1, min(batch_size_enrich, 8))

        # Set per-index overview file path
        overview_path = f"index_store/overviews/{index_id}.jsonl"

        # Delegate to advanced RAG API same as session indexing
        rag_api_url = "http://localhost:8001/index"
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

        rag_resp = requests.post(rag_api_url, json=payload)
        if rag_resp.status_code == 200:
            meta_updates = {
                "status": "functional",
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
                "rebuilt_at": datetime.now().isoformat(),
            }
            if embedding_model:
                meta_updates["embedding_model"] = embedding_model
            if enrich_model:
                meta_updates["enrich_model"] = enrich_model
            if overview_model:
                meta_updates["overview_model"] = overview_model
            if indexing_model_warnings:
                meta_updates["indexing_model_warnings"] = indexing_model_warnings
            try:
                db.update_index_metadata(index_id, meta_updates)
            except Exception as e:
                print(f"⚠️ Failed to update index metadata: {e}")

            response_data = rag_resp.json()
            response_data.update(meta_updates)
            return response_data
        else:
            # Gracefully handle scenario where table already exists (idempotent build)
            try:
                err_json = rag_resp.json()
            except Exception:
                err_json = {}
            err_text = err_json.get('error') if isinstance(err_json, dict) else rag_resp.text
            if err_text and 'already exists' in err_text:
                # Treat as non-fatal; return message indicating index previously built
                return {
                    "message": "Index already built – skipping rebuild.",
                    "note": err_text
                }
            else:
                raise HTTPException(status_code=500, detail=f"RAG indexing failed: {rag_resp.text}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

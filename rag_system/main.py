import os
import json
import sys
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# The sys.path manipulation has been removed to prevent import conflicts.
# This script should be run as a module from the project root, e.g.:
# python -m rag_system.main api

from rag_system.agent.loop import Agent
from rag_system.utils.ollama_client import OllamaClient
# Configuration is now defined in this file - no import needed

# Advanced RAG System Configuration
# ==================================
# This file contains the MASTER configuration for all models used in the RAG system.
# All components should reference these configurations to ensure consistency.

# ============================================================================
# 🎯 MASTER MODEL CONFIGURATION
# ============================================================================
# All model configurations are centralized here to prevent conflicts

# LLM Backend Configuration
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")

# Ollama Models Configuration (for inference via Ollama)
OLLAMA_CONFIG = {
    "host": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    "generation_model": "qwen3:8b",  # Main text generation model
    "enrichment_model": "qwen3:0.6b",  # Lightweight model for routing/enrichment
}

WATSONX_CONFIG = {
    "api_key": os.getenv("WATSONX_API_KEY", ""),
    "project_id": os.getenv("WATSONX_PROJECT_ID", ""),
    "url": os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com"),
    "generation_model": os.getenv("WATSONX_GENERATION_MODEL", "ibm/granite-13b-chat-v2"),
    "enrichment_model": os.getenv("WATSONX_ENRICHMENT_MODEL", "ibm/granite-8b-japanese"),  # Lightweight model
}

# External Model Configuration (HuggingFace models used directly)
EXTERNAL_MODELS = {
    "embedding_model": "Qwen/Qwen3-Embedding-0.6B",  # HuggingFace embedding model (1024 dims - fresh start)
    "reranker_model": "answerdotai/answerai-colbert-small-v1",  # ColBERT reranker
    "vision_model": "Qwen/Qwen-VL-Chat",  # Vision model for multimodal
    "fallback_reranker": "BAAI/bge-reranker-base",  # Backup reranker
}

# ============================================================================
# 🔧 PIPELINE CONFIGURATIONS
# ============================================================================

PIPELINE_CONFIGS = {
    "default": {
        "description": "Production-ready pipeline with hybrid search, AI reranking, and verification",
        "storage": {
            "lancedb_uri": "./lancedb",
            "text_table_name": "text_pages_v3", 
            "image_table_name": "image_pages_v3",
            "bm25_path": "./index_store/bm25",
            "graph_path": "./index_store/graph/knowledge_graph.gml"
        },
        "retrieval": {
            "retriever": "multivector",
            "search_type": "hybrid",
            "late_chunking": {
                "enabled": True,
                "table_suffix": "_lc_v3"
        },
            "dense": { 
                "enabled": True,
                "weight": 0.7
            },
            "bm25": { 
                "enabled": True,
                "index_name": "rag_bm25_index"
            },
            "graph": { 
                "enabled": False,
                "graph_path": "./index_store/graph/knowledge_graph.gml"
            }
        },
        # 🎯 EMBEDDING MODEL: Uses HuggingFace Qwen model directly
        "embedding_model_name": EXTERNAL_MODELS["embedding_model"],
        # 🎯 VISION MODEL: For multimodal capabilities  
        "vision_model_name": EXTERNAL_MODELS["vision_model"],
        # 🎯 RERANKER: AI-powered reranking with ColBERT
        "reranker": {
            "enabled": True, 
            "type": "ai",
            "strategy": "rerankers-lib",
            "model_name": EXTERNAL_MODELS["reranker_model"],
            "top_k": 10
        },
        "query_decomposition": {
            "enabled": True,
            "max_sub_queries": 3,
            "compose_from_sub_answers": True
        },
        "verification": {"enabled": True},
        "retrieval_k": 20,
        "context_window_size": 0,
        "semantic_cache_threshold": 0.98,
        "cache_scope": "global",
        # 🔧 Contextual enrichment configuration
        "contextual_enricher": {
            "enabled": True,
            "window_size": 1
        },
        # 🔧 Indexing configuration
        "indexing": {
            "embedding_batch_size": 50,
            "enrichment_batch_size": 10,
            "enable_progress_tracking": True
        }
    },
    "fast": {
        "description": "Speed-optimized pipeline with minimal overhead",
        "storage": {
            "lancedb_uri": "./lancedb",
            "text_table_name": "text_pages_v3",
            "image_table_name": "image_pages_v3", 
            "bm25_path": "./index_store/bm25"
        },
        "retrieval": {
            "retriever": "multivector",
            "search_type": "vector_only",
            "late_chunking": {"enabled": False},
            "dense": {"enabled": True}
        },
        "embedding_model_name": EXTERNAL_MODELS["embedding_model"],
        "reranker": {"enabled": False},
        "query_decomposition": {"enabled": False},
        "verification": {"enabled": False},
        "retrieval_k": 10,
        "context_window_size": 0,
        # 🔧 Contextual enrichment (disabled for speed)
        "contextual_enricher": {
            "enabled": False,
            "window_size": 1
        },
        # 🔧 Indexing configuration
        "indexing": {
            "embedding_batch_size": 100,
            "enrichment_batch_size": 50,
            "enable_progress_tracking": False
        }
    },
}

# ============================================================================
# 🏭 FACTORY FUNCTIONS
# ============================================================================

def get_agent(mode: str = "default") -> Agent:
    """
    Factory function to get an instance of the RAG agent based on the specified mode.
    
    Args:
        mode: Configuration mode ("default", "fast")
        
    Returns:
        Configured Agent instance
    """
    load_dotenv()
    
    # Initialize the appropriate LLM client based on backend configuration
    if LLM_BACKEND.lower() == "watsonx":
        from rag_system.utils.watsonx_client import WatsonXClient
        
        if not WATSONX_CONFIG["api_key"] or not WATSONX_CONFIG["project_id"]:
            raise ValueError(
                "Watson X configuration incomplete. Please set WATSONX_API_KEY and WATSONX_PROJECT_ID "
                "environment variables."
            )
        
        llm_client = WatsonXClient(
            api_key=WATSONX_CONFIG["api_key"],
            project_id=WATSONX_CONFIG["project_id"],
            url=WATSONX_CONFIG["url"]
        )
        llm_config = WATSONX_CONFIG
        print(f"🔧 Using Watson X backend with granite models")
    else:
        llm_client = OllamaClient(host=OLLAMA_CONFIG["host"])
        llm_config = OLLAMA_CONFIG
        print(f"🔧 Using Ollama backend")
    
    # Get the configuration for the specified mode
    config = PIPELINE_CONFIGS.get(mode, PIPELINE_CONFIGS['default'])
    
    agent = Agent(
        pipeline_configs=config, 
        llm_client=llm_client, 
        ollama_config=llm_config
    )
    return agent

def validate_model_config():
    """
    Validates the model configuration for consistency and availability.
    
    Raises:
        ValueError: If configuration conflicts are detected
    """
    print("🔍 Validating model configuration...")
    
    # Check for embedding model consistency
    default_embedding = PIPELINE_CONFIGS["default"]["embedding_model_name"]
    external_embedding = EXTERNAL_MODELS["embedding_model"]
    
    if default_embedding != external_embedding:
        raise ValueError(f"Embedding model mismatch: {default_embedding} != {external_embedding}")
    
    # Check reranker configuration
    default_reranker = PIPELINE_CONFIGS["default"]["reranker"]["model_name"]
    external_reranker = EXTERNAL_MODELS["reranker_model"]
    
    if default_reranker != external_reranker:
        raise ValueError(f"Reranker model mismatch: {default_reranker} != {external_reranker}")
    
    print("✅ Model configuration validation passed!")
    
    return True

# ============================================================================
# 🚀 UTILITY FUNCTIONS  
# ============================================================================

def run_indexing(docs_path: str, config_mode: str = "default"):
    """Runs the indexing pipeline for the specified documents."""
    print(f"📚 Starting indexing for documents in: {docs_path}")
    validate_model_config()
    
    # Local import to avoid circular dependencies
    from rag_system.pipelines.indexing_pipeline import IndexingPipeline
    
    # Get the appropriate indexing pipeline from the factory
    indexing_pipeline = IndexingPipeline(PIPELINE_CONFIGS[config_mode])
    
    # Find all PDF files in the directory
    pdf_files = [os.path.join(docs_path, f) for f in os.listdir(docs_path) if f.endswith(".pdf")]
    
    if not pdf_files:
        print("No PDF files found to index.")
        return

    # Process all documents through the pipeline
    indexing_pipeline.process_documents(pdf_files)
    print("✅ Indexing complete.")

def run_chat(query: str):
    """
    Runs the agentic RAG pipeline for a given query.
    Returns the result as a JSON string.
    """
    try:
        validate_model_config()
        ollama_client = OllamaClient(OLLAMA_CONFIG["host"])
    except ConnectionError as e:
        print(e)
        return json.dumps({"error": str(e)}, indent=2)
    except ValueError as e:
        print(f"Configuration Error: {e}")
        return json.dumps({"error": f"Configuration Error: {e}"}, indent=2)

    agent = Agent(PIPELINE_CONFIGS['default'], ollama_client, OLLAMA_CONFIG)
    result = agent.run(query)
    return json.dumps(result, indent=2, ensure_ascii=False)

def run_api_server():
    """Starts the advanced RAG API server."""
    from rag_system.api_server import start_server
    start_server()

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py [index|chat|api] [query]")
        return

    command = sys.argv[1]
    if command == "index":
        # Allow passing file paths from the command line
        files = sys.argv[2:] if len(sys.argv) > 2 else None
        run_indexing(files)
    elif command == "chat":
        if len(sys.argv) < 3:
            print("Usage: python main.py chat <query>")
            return
        query = " ".join(sys.argv[2:])
        # 🆕 Print the result for command-line usage
        print(run_chat(query))
    elif command == "api":
        run_api_server()
    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    # This allows running the script from the command line to index documents.
    parser = argparse.ArgumentParser(description="Main entry point for the RAG system.")
    parser.add_argument(
        '--index',
        type=str,
        help='Path to the directory containing documents to index.'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='default',
        help='The configuration profile to use (e.g., "default", "fast").'
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    if args.index:
        run_indexing(args.index, args.config)
    else:
        # This is where you might start a server or interactive session
        print("No action specified. Use --index to process documents.")
        # Example of how to get an agent instance
        # agent = get_agent(args.config)
        # print(f"Agent loaded with '{args.config}' config.")

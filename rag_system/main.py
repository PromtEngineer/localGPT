import os
import json
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# This module holds the MASTER configuration for the RAG system plus a thin CLI.
# Agent / pipeline construction lives in rag_system/factory.py.
# Run it as a module from the project root, e.g.:
# python -m rag_system.main api

# ============================================================================
# MASTER MODEL CONFIGURATION
# ============================================================================
# Every model default below can be overridden with an environment variable.

# LLM Backend Configuration ("ollama" or "watsonx")
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")

# Ollama Models Configuration (for inference via Ollama)
OLLAMA_CONFIG = {
    "host": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    "generation_model": os.getenv("GENERATION_MODEL", "qwen3.5:9b"),
    "enrichment_model": os.getenv("ENRICHMENT_MODEL", "qwen3.5:4b"),
}

WATSONX_CONFIG = {
    "api_key": os.getenv("WATSONX_API_KEY", ""),
    "project_id": os.getenv("WATSONX_PROJECT_ID", ""),
    "url": os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com"),
    "generation_model": os.getenv("WATSONX_GENERATION_MODEL", "ibm/granite-13b-chat-v2"),
    "enrichment_model": os.getenv("WATSONX_ENRICHMENT_MODEL", "ibm/granite-8b-japanese"),
}

# External Model Configuration (HuggingFace models loaded in-process)
#
# Defaults set at the Phase 1 adoption gate (2026-08-09); the measurements and
# the reasoning are in eval/DECISIONS.md.
#
#   embedding_model  microsoft/harrier-oss-v1-0.6b (MIT, 1024-dim). Measured
#                    mixed-corpus first-stage nDCG@10 0.915 vs 0.875 for
#                    Qwen/Qwen3-Embedding-4B, at ~3x lower latency and ~7x less
#                    memory. Qwen/Qwen3-Embedding-4B remains a supported option.
#   reranker_model   Only loaded when reranking is switched on — the "default"
#                    profile ships with reranker.enabled = False (see below).
#                    When a user does switch it on, they get the model that
#                    measured a win on top of this first stage.
EXTERNAL_MODELS = {
    "embedding_model": os.getenv("EMBEDDING_MODEL", "microsoft/harrier-oss-v1-0.6b"),
    "reranker_model": os.getenv("RERANKER_MODEL", "Qwen/Qwen3-Reranker-4B"),
}

# ============================================================================
# PIPELINE CONFIGURATIONS
# ============================================================================

PIPELINE_CONFIGS = {
    "default": {
        "description": "Production-ready pipeline with hybrid search, query decomposition, and verification",
        "storage": {
            "lancedb_uri": os.getenv("LANCEDB_PATH", "./lancedb"),
            # v4: vectors are L2-normalized at write and query time (cosine
            # ordering). v3 tables hold unnormalized vectors — see
            # rag_system/indexing/embedders.py table markers.
            "text_table_name": "text_pages_v4"
        },
        "retrieval": {
            "search_type": "hybrid",
            "latechunk": {
                "enabled": True
            },
            "dense": {
                "enabled": True
            },
            # Evidence-sufficiency retry (roadmap 2.1). One conditional second
            # retrieval when the first pass found weak evidence; hard cap of one
            # extra attempt. The signal is NOT the raw top cosine — that measured
            # anti-correlated with success — but the contrast between the best
            # candidate and the background of the rest; see
            # RetrievalPipeline._dense_evidence_score and eval/decisions/
            # phase2-pipeline.md for the calibration.
            "retry": {
                "enabled": True,
                "min_top_score": 0.12,
                "max_attempts": 1
            }
        },
        "embedding_model_name": EXTERNAL_MODELS["embedding_model"],
        # Reranking is OFF by default. Measured at the Phase 1 gate on the mixed
        # corpus: this first stage alone scores nDCG@10 0.915; the cheap
        # cross-encoder (bge-reranker-v2-m3) drops it to 0.892 for ~1.6s/query,
        # and Qwen3-Reranker-4B lifts it to 0.977 for ~12.7s/query. Neither is a
        # sensible always-on default, so the toggle (UI "AI reranker" /
        # reranker.enabled) picks the quality model and loads it lazily.
        "reranker": {
            "enabled": False,
            "model_type": "cross-encoder",
            "strategy": "rerankers-lib",
            "model_name": EXTERNAL_MODELS["reranker_model"],
            "top_k": 10
        },
        "query_decomposition": {
            "enabled": True,
            "compose_from_sub_answers": True
        },
        "verification": {"enabled": True},
        "retrieval_k": 20,
        "context_window_size": 0,
        "semantic_cache_threshold": 0.98,
        "cache_scope": "session",
        "contextual_enricher": {
            "enabled": True,
            "window_size": 1
        },
        "indexing": {
            "embedding_batch_size": 50,
            "enrichment_batch_size": 10,
            "enable_progress_tracking": True
        }
    },
    "fast": {
        "description": "Speed-optimized pipeline with minimal overhead",
        "storage": {
            "lancedb_uri": os.getenv("LANCEDB_PATH", "./lancedb"),
            "text_table_name": "text_pages_v4"
        },
        "retrieval": {
            "search_type": "vector_only",
            "latechunk": {"enabled": False},
            "dense": {"enabled": True},
            # Off in `fast`: the retry costs one enrichment-model round-trip plus
            # a second retrieval, which is exactly what this profile exists to avoid.
            "retry": {"enabled": False}
        },
        "embedding_model_name": EXTERNAL_MODELS["embedding_model"],
        "reranker": {"enabled": False},
        "query_decomposition": {"enabled": False},
        "verification": {"enabled": False},
        "retrieval_k": 10,
        "context_window_size": 0,
        "semantic_cache_threshold": 0.98,
        "cache_scope": "session",
        "contextual_enricher": {
            "enabled": False,
            "window_size": 1
        },
        "indexing": {
            "embedding_batch_size": 100,
            "enrichment_batch_size": 50,
            "enable_progress_tracking": False
        }
    }
}

# ============================================================================
# CLI
# ============================================================================

SUPPORTED_DOCUMENT_EXTENSIONS = (".pdf", ".docx", ".html", ".htm", ".md", ".txt")


def _collect_file_paths(path: str) -> list[str]:
    """Expand a file or directory argument into a list of indexable file paths."""
    if os.path.isfile(path):
        return [os.path.abspath(path)]

    if not os.path.isdir(path):
        raise FileNotFoundError(f"No such file or directory: {path}")

    collected = []
    for root, _dirs, files in os.walk(path):
        for name in sorted(files):
            if name.lower().endswith(SUPPORTED_DOCUMENT_EXTENSIONS):
                collected.append(os.path.join(root, name))
    return collected


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m rag_system.main",
        description="localGPT RAG system: indexing, one-shot chat, and API server."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    modes = sorted(PIPELINE_CONFIGS)

    index_parser = subparsers.add_parser("index", help="Index a document or a directory of documents.")
    index_parser.add_argument("path", help="File or directory to index.")
    index_parser.add_argument("--mode", default="default", choices=modes, help="Pipeline profile to use.")

    chat_parser = subparsers.add_parser("chat", help="Answer a single query and print the JSON result.")
    chat_parser.add_argument("query", help="The question to ask.")
    chat_parser.add_argument("--mode", default="default", choices=modes, help="Pipeline profile to use.")

    api_parser = subparsers.add_parser("api", help="Start the RAG API server.")
    api_parser.add_argument("--port", type=int, default=8001, help="Port to listen on.")

    args = parser.parse_args()

    if args.command == "index":
        from rag_system.factory import get_indexing_pipeline

        try:
            file_paths = _collect_file_paths(args.path)
        except FileNotFoundError as e:
            print(f"❌ {e}")
            return 1

        if not file_paths:
            print(f"No indexable documents found in {args.path} "
                  f"(supported: {', '.join(SUPPORTED_DOCUMENT_EXTENSIONS)}).")
            return 1

        print(f"📚 Indexing {len(file_paths)} file(s) with the '{args.mode}' profile...")
        get_indexing_pipeline(args.mode).run(file_paths)
        print("✅ Indexing complete.")
        return 0

    if args.command == "chat":
        from rag_system.factory import get_agent

        result = get_agent(args.mode).run(args.query)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    if args.command == "api":
        from rag_system.api_server import start_server

        start_server(port=args.port)
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

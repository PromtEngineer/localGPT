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
            },
            # Full-document escalation (roadmap 4.1). OFF until benchmarked.
            # When the evidence-sufficiency retry above has already run and the
            # evidence is STILL below threshold, reassemble the top-ranked
            # chunk's whole document in chunk_index order and append it to the
            # synthesis context, capped at token_budget. One document, no loop.
            # min_evidence defaults to retry.min_top_score when omitted. See
            # eval/decisions/phase4-escalation-tokens.md.
            "document_escalation": {
                "enabled": False,
                "max_documents": 1,
                "token_budget": 6000
            },
            # Cross-reference hop (roadmap 4.2). OFF: index-time extraction is
            # free and additive, but the query-time hop appends chunks the
            # retriever never scored, and it has not been benchmarked yet. See
            # eval/decisions/phase4-crossref-prefilter.md.
            "crossref_hop": {
                "enabled": False,
                "max_hops": 1,          # referenced documents expanded, no recursion
                "chunks_per_hop": 3
            },
            # Overview prefilter (roadmap 4.3). OFF until benchmarked. "boost"
            # is the safe mode — it reorders; "restrict" can hide a document.
            "overview_prefilter": {
                "enabled": False,
                "top_documents": 5,
                "mode": "boost"         # "boost" | "restrict"
            }
        },
        "embedding_model_name": EXTERNAL_MODELS["embedding_model"],
        # Reranking is ON with threshold selection (arm G). The Phase 1 "off by
        # default" call (first stage alone nDCG@10 0.915; bge drops it to 0.892;
        # Qwen3-Reranker-4B lifts to 0.977 at ~12.7s/query) predates the
        # synthesis context budget — back then rank order barely mattered
        # because front-truncation fed synthesis the tail of the list anyway.
        # Now the budget keeps exactly the top-ranked docs, so ordering AND
        # selection decide everything the model reads. min_score keeps only
        # candidates the calibrated Qwen scorer marks relevant to at least one
        # query (union across sub-queries), instead of a fixed 10.
        "reranker": {
            "enabled": True,
            "model_type": "cross-encoder",
            "strategy": "rerankers-lib",
            "model_name": EXTERNAL_MODELS["reranker_model"],
            "top_k": 10,
            # Qwen scorer only (calibrated P(relevant)): candidates below this
            # against every query are dropped, so easy questions send a small,
            # clean context instead of a fixed-size one (arm G, 2026-08-14).
            "min_score": 0.5,
            "min_keep": 3
        },
        "query_decomposition": {
            "enabled": True,
            # Arm H (2026-08-15): decomposed queries retrieve per-sub-query,
            # pool + dedupe the candidates, then ONE source-aware rerank pass
            # and ONE synthesis over the union context — replacing N rerank
            # passes, N synthesis calls and the compose step (where multi-hop
            # facts were measurably lost, arm E).
            "compose_from_sub_answers": False,
            "pooled_first_stage": True
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
            # Cross-reference extraction (roadmap 4.2, index-time half). Regex
            # only, no LLM; writes chunk metadata.crossrefs. Verified inert for
            # retrieval: text and vector columns are bit-identical with it on.
            "extract_crossrefs": True
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
            "retry": {"enabled": False},
            # Phase-4 flags (roadmap 4.1–4.3): all off in `fast` for the same
            # reason — this profile exists to avoid extra work per query.
            "document_escalation": {"enabled": False},
            "crossref_hop": {"enabled": False},
            "overview_prefilter": {"enabled": False}
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
            # Costs nothing even in `fast` (regex over text already in memory).
            "extract_crossrefs": True
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
    chat_parser.add_argument(
        "--filters", default=None,
        help='Metadata filter as JSON (roadmap 4.4), e.g. \'{"document_id": "nda.pdf"}\' '
             'or \'{"document_name": {"contains": "nda"}, "chunk_index": {"lte": 4}}\'.'
    )

    # Ephemeral "ask a folder" mode (roadmap 4.6).
    ask_parser = subparsers.add_parser(
        "ask",
        help="Index a folder into a throwaway index, answer, then delete it."
    )
    ask_parser.add_argument("path", help="Folder (or single file) to ask about.")
    ask_parser.add_argument("questions", nargs="*", help="One or more questions.")
    ask_parser.add_argument("--mode", default="fast", choices=modes,
                            help="Pipeline profile to use (default: fast).")
    ask_parser.add_argument("--interactive", action="store_true",
                            help="Keep asking follow-up questions against the same temp index.")
    ask_parser.add_argument("--agent", action="store_true",
                            help="Answer through the full agent loop (decomposition, "
                                 "verification) instead of the retrieval pipeline alone.")
    ask_parser.add_argument("--filters", default=None,
                            help="Metadata filter as JSON (see 'chat --filters').")
    ask_parser.add_argument("--keep", action="store_true",
                            help="Do not delete the temporary index (debugging).")

    api_parser = subparsers.add_parser("api", help="Start the RAG API server.")
    api_parser.add_argument("--port", type=int, default=8001, help="Port to listen on.")

    args = parser.parse_args()

    def _parse_filters(raw):
        """Parse and validate --filters. Returns (compiled_or_None, exit_code_or_None)."""
        if not raw:
            return None, None
        from rag_system.retrieval.filters import FilterError, compile_filters
        try:
            return compile_filters(json.loads(raw)), None
        except json.JSONDecodeError as e:
            print(f"❌ --filters is not valid JSON: {e}")
            return None, 2
        except FilterError as e:
            print(f"❌ Invalid --filters: {e}")
            return None, 2

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

        filters, error = _parse_filters(args.filters)
        if error:
            return error

        result = get_agent(args.mode).run(args.query, filters=filters)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    if args.command == "ask":
        from rag_system.ask_folder import ask_folder

        filters, error = _parse_filters(args.filters)
        if error:
            return error

        return ask_folder(
            args.path, args.questions, mode=args.mode,
            interactive=args.interactive, use_agent=args.agent,
            filters=filters, keep=args.keep,
        )

    if args.command == "api":
        from rag_system.api_server import start_server

        start_server(port=args.port)
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

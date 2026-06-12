from typing import List, Dict, Any, Callable, Optional
import copy
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import networkx as nx
from rag_system.ingestion.document_converter import DocumentConverter
from rag_system.ingestion.chunking import MarkdownRecursiveChunker
from rag_system.indexing.representations import EmbeddingGenerator, select_embedder
from rag_system.indexing.embedders import LanceDBManager, VectorIndexer
from rag_system.indexing.graph_extractor import GraphExtractor
from rag_system.utils.ollama_client import OllamaClient
from rag_system.indexing.contextualizer import ContextualEnricher
from rag_system.indexing.overview_builder import OverviewBuilder
from rag_system.utils.incremental_indexer import IncrementalIndexer
from rag_system.utils.logging_utils import indexing_logger, PerformanceTimer

try:
    from rag_system.job_persistence import JobProgressTracker
except Exception:  # pragma: no cover - job persistence is optional for standalone runs
    JobProgressTracker = None


def convert_and_chunk_document(file_path: str, document_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
    try:
        chunker_mode = config.get("chunker_mode", "docling")
        chunking_config = config.get("chunking", {})
        chunk_size = chunking_config.get("chunk_size", config.get("chunk_size", 1500))
        chunk_overlap = chunking_config.get("chunk_overlap", config.get("chunk_overlap", 200))

        if chunker_mode == "docling":
            converter = DocumentConverter()
            try:
                from rag_system.ingestion.docling_chunker import DoclingChunker
                chunker = DoclingChunker(
                    max_tokens=config.get("max_tokens", chunk_size),
                    overlap=config.get("overlap_sentences", 1),
                    tokenizer_model=config.get("embedding_model_name", "qwen3-embedding-0.6b"),
                )
            except Exception:
                chunker = MarkdownRecursiveChunker(
                    max_chunk_size=chunk_size,
                    min_chunk_size=min(chunk_overlap, chunk_size // 4),
                    tokenizer_model=config.get("embedding_model_name", "Qwen/Qwen3-Embedding-0.6B"),
                )
            pages_data = converter.convert_to_markdown(file_path)
        else:
            chunker = MarkdownRecursiveChunker(
                max_chunk_size=chunk_size,
                min_chunk_size=min(chunk_overlap, chunk_size // 4),
                tokenizer_model=config.get("embedding_model_name", "Qwen/Qwen3-Embedding-0.6B"),
            )
            pages_data = _convert_file_to_plain_markdown(file_path)

        file_chunks = []
        for tpl in pages_data:
            if len(tpl) == 3:
                markdown_text, metadata, doc_obj = tpl
                if hasattr(chunker, "chunk_document"):
                    chunks = chunker.chunk_document(doc_obj, document_id=document_id, metadata=metadata)
                else:
                    chunks = chunker.chunk(markdown_text, document_id, metadata)
            else:
                markdown_text, metadata = tpl
                chunks = chunker.chunk(markdown_text, document_id, metadata)
            file_chunks.extend(chunks)
        return {"chunks": file_chunks}
    except Exception as e:
        return {"error": str(e)}


def _convert_file_to_plain_markdown(file_path: str):
    """Low-risk conversion path for Fast indexing that avoids Docling/OCR."""
    file_ext = Path(file_path).suffix.lower()
    metadata = {"source": file_path}

    if file_ext in {".txt", ".md"}:
        return [(Path(file_path).read_text(encoding="utf-8", errors="replace"), metadata)]

    if file_ext == ".pdf":
        try:
            import fitz
            doc = fitz.open(file_path)
            text = "\n\n".join(page.get_text("text") for page in doc)
            doc.close()
            return [(text, metadata)] if text.strip() else []
        except Exception as e:
            return [(f"PDF text extraction failed for {file_path}: {e}", metadata)]

    converter = DocumentConverter()
    return converter.convert_to_markdown(file_path)


class IndexingPipeline:
    def __init__(self, config: Dict[str, Any], ollama_client: OllamaClient, ollama_config: Dict[str, str]):
        self.config = config
        self.llm_client = ollama_client
        self.ollama_config = ollama_config
        self.document_converter = DocumentConverter()
        # Chunker selection: docling (token-based) or legacy (character-based)
        chunker_mode = config.get("chunker_mode", "docling")
        
        # 🔧 Get chunking configuration from frontend parameters
        chunking_config = config.get("chunking", {})
        chunk_size = chunking_config.get("chunk_size", config.get("chunk_size", 1500))
        chunk_overlap = chunking_config.get("chunk_overlap", config.get("chunk_overlap", 200))
        
        indexing_logger.info(
            "chunking_config",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunker_mode=chunker_mode,
        )
        
        if chunker_mode == "docling":
            try:
                from rag_system.ingestion.docling_chunker import DoclingChunker
                self.chunker = DoclingChunker(
                    max_tokens=config.get("max_tokens", chunk_size),
                    overlap=config.get("overlap_sentences", 1),
                    tokenizer_model=config.get("embedding_model_name", "qwen3-embedding-0.6b"),
                )
                indexing_logger.info("chunker_selected", chunker="docling")
            except Exception as e:
                indexing_logger.warning("docling_chunker_fallback", error=str(e))
                self.chunker = MarkdownRecursiveChunker(
                    max_chunk_size=chunk_size,
                    min_chunk_size=min(chunk_overlap, chunk_size // 4),  # Sensible minimum
                    tokenizer_model=config.get("embedding_model_name", "Qwen/Qwen3-Embedding-0.6B")
                )
        else:
            self.chunker = MarkdownRecursiveChunker(
                max_chunk_size=chunk_size,
                min_chunk_size=min(chunk_overlap, chunk_size // 4),  # Sensible minimum
                tokenizer_model=config.get("embedding_model_name", "Qwen/Qwen3-Embedding-0.6B")
            )

        retriever_configs = self.config.get("retrievers") or self.config.get("retrieval", {})
        storage_config = self.config["storage"]
        
        # Get batch processing configuration
        indexing_config = self.config.get("indexing", {})
        self.embedding_batch_size = indexing_config.get("embedding_batch_size", 8)
        self.enrichment_batch_size = indexing_config.get("enrichment_batch_size", 10)
        self.enable_progress_tracking = indexing_config.get("enable_progress_tracking", True)
        self.conversion_timeout_seconds = indexing_config.get("conversion_timeout_seconds", 360)
        self.overview_timeout_seconds = indexing_config.get("overview_timeout_seconds", 60)
        self.enrichment_timeout_seconds = indexing_config.get("enrichment_timeout_seconds", 90)
        # Enrichment is one LLM call per chunk; past this many chunks the
        # enricher is disabled for the rest of the build (with a warning)
        # instead of grinding through thousands of generations.
        self.max_enrich_chunks = int(indexing_config.get("max_enrich_chunks", 1000))

        # Treat dense retrieval as enabled by default unless explicitly disabled
        dense_cfg = retriever_configs.setdefault("dense", {})
        dense_cfg.setdefault("enabled", True)

        if dense_cfg.get("enabled"):
            # Accept modern keys: db_path or lancedb_path; fall back to legacy lancedb_uri
            db_path = (
                storage_config.get("db_path")
                or storage_config.get("lancedb_path")
                or storage_config.get("lancedb_uri")
            )
            if not db_path:
                raise KeyError(
                    "Storage config must include 'db_path', 'lancedb_path', or 'lancedb_uri' for LanceDB."
                )
            self.lancedb_manager = LanceDBManager(db_path=db_path)
            self.vector_indexer = VectorIndexer(self.lancedb_manager)
            embedding_model = select_embedder(
                self.config.get("embedding_model_name", "BAAI/bge-small-en-v1.5"),
                self.ollama_config.get("host") if isinstance(self.ollama_config, dict) else None,
            )
            self.embedding_generator = EmbeddingGenerator(
                embedding_model=embedding_model, 
                batch_size=self.embedding_batch_size
            )

        if retriever_configs.get("graph", {}).get("enabled"):
            self.graph_extractor = GraphExtractor(
                llm_client=self.llm_client,
                llm_model=self.ollama_config["generation_model"]
            )

        if self.config.get("contextual_enricher", {}).get("enabled"):
            # 🔧 Use frontend enrich_model parameter if provided
            enrichment_model = (
                self.config.get("enrich_model") or  # Frontend parameter
                self.config.get("enrichment_model_name") or  # Alternative config key
                self.ollama_config.get("enrichment_model") or  # Default from ollama config
                self.ollama_config["generation_model"]  # Final fallback
            )
            indexing_logger.info("enrichment_model", model=enrichment_model)

            enrich_provider = self.config.get("enrich_provider", "ollama")
            if enrich_provider != "ollama":
                from rag_system.utils.cloud_clients import create_enrichment_client
                enrichment_client = create_enrichment_client(
                    provider=enrich_provider,
                    api_key=self.config.get("enrich_api_key"),
                    ollama_client=self.llm_client,
                )
                indexing_logger.info("enrichment_provider", provider=enrich_provider)
            else:
                enrichment_client = self.llm_client

            self.contextual_enricher = ContextualEnricher(
                llm_client=enrichment_client,
                llm_model=enrichment_model,
                batch_size=self.enrichment_batch_size,
                timeout=self.enrichment_timeout_seconds,
            )

        # Overview builder always enabled for triage routing
        ov_path = self.config.get("overview_path")
        self.overview_builder = OverviewBuilder(
            llm_client=self.llm_client,
            model=self.config.get("overview_model_name", self.ollama_config.get("enrichment_model", "qwen3:8b")),
            first_n_chunks=self.config.get("overview_first_n_chunks", 5),
            out_path=ov_path if ov_path else None,
            timeout=self.overview_timeout_seconds,
        )

        # Initialize incremental indexer
        db_path = self.config.get("db_path", "backend/chat_data.db")
        index_store_path = self.config.get("index_store_path", "index_store")
        self.incremental_indexer = IncrementalIndexer(db_path, index_store_path)
        self.chunk_cache_dir = Path(index_store_path) / "chunk_cache"
        self.chunk_cache_dir.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------
        # Late-Chunk encoder initialisation (optional)
        # ------------------------------------------------------------------
        self.latechunk_enabled = retriever_configs.get("latechunk", {}).get("enabled", False)
        if self.latechunk_enabled:
            try:
                from rag_system.indexing.latechunk import LateChunkEncoder
                self.latechunk_cfg = retriever_configs["latechunk"]
                self.latechunk_encoder = LateChunkEncoder(model_name=self.config.get("embedding_model_name", "qwen3-embedding-0.6b"))
            except Exception as e:
                indexing_logger.warning("latechunk_initialization_failed", error=str(e))
                self.latechunk_enabled = False

    def run(self, file_paths: List[str] | None = None, *, documents: List[str] | None = None,
            index_id: str = "default", incremental: bool = True, force_reindex: bool = False,
            progress_callback: Callable[..., None] | None = None,
            cancel_callback: Callable[[], bool] | None = None,
            job_id: str | None = None):
        """
        Processes and indexes documents based on the pipeline's configuration.
        Supports incremental indexing to avoid re-processing unchanged documents.

        Args:
            file_paths: List of file paths to process
            documents: Legacy alias for file_paths
            index_id: Unique identifier for this index
            incremental: Whether to use incremental indexing
            force_reindex: Force reindexing of all documents
        """
        # Back-compat shim ---------------------------------------------------
        if file_paths is None and documents is not None:
            file_paths = documents
        if file_paths is None:
            raise TypeError("IndexingPipeline.run() expects 'file_paths' (or alias 'documents') argument")

        indexing_logger.info(
            "indexing_started",
            index_id=index_id,
            total_files=len(file_paths),
            incremental=incremental,
            force_reindex=force_reindex,
        )

        def report(stage: str, progress: int, message: str, **extra):
            if progress_callback:
                progress_callback(stage, max(0, min(progress, 100)), message, **extra)

        def check_cancelled():
            if cancel_callback and cancel_callback():
                indexing_logger.warning("indexing_cancelled", index_id=index_id)
                report("cancelled", 100, "Indexing cancelled")
                raise RuntimeError("indexing_cancelled")

        tracker = None
        if job_id and JobProgressTracker is not None:
            try:
                tracker = JobProgressTracker(db_path=self.config.get("db_path", "backend/chat_data.db"))
            except Exception as e:
                indexing_logger.warning("job_progress_tracker_unavailable", job_id=job_id, error=str(e))

        def stage_output_hash(value: Any) -> str:
            try:
                payload = json.dumps(value, sort_keys=True, default=str)
            except Exception:
                payload = str(value)
            return hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest()

        def start_tracked_stage(file_id: Optional[int], stage: str) -> bool:
            if not tracker or not job_id or file_id is None:
                return False
            if not force_reindex and tracker.should_skip_stage(file_id, stage):
                return True
            tracker.start_stage(file_id, job_id, stage)
            return False

        def complete_tracked_stage(file_id: Optional[int], stage: str, output: Any = None) -> None:
            if tracker and file_id is not None:
                tracker.complete_stage(file_id, stage, output_hash=stage_output_hash(output) if output is not None else None)

        def fail_tracked_stage(file_id: Optional[int], stage: Optional[str], error: Exception | str) -> None:
            if tracker and file_id is not None and stage:
                try:
                    tracker.fail_stage(file_id, stage, str(error))
                except Exception as track_error:
                    indexing_logger.warning("stage_failure_tracking_failed", stage=stage, error=str(track_error))

        report("planning", 8, "Checking changed files")
        check_cancelled()

        # Determine which files need processing
        if incremental and not force_reindex:
            indexing_logger.info("incremental_indexing", enabled=True, force_reindex=False)
            files_to_index, unchanged_files = self.incremental_indexer.get_incremental_file_list(
                file_paths, index_id=index_id, force_reindex=force_reindex
            )

            if unchanged_files:
                indexing_logger.info("skipping_unchanged_files", count=len(unchanged_files))
                for unchanged_path in unchanged_files:
                    unchanged_doc = os.path.basename(unchanged_path)
                    report(
                        "planning",
                        8,
                        f"Skipped unchanged {unchanged_doc}",
                        file_path=unchanged_path,
                        filename=unchanged_doc,
                        document_id=unchanged_doc,
                        file_status="skipped",
                    )
            if not files_to_index:
                indexing_logger.info("indexing_not_required", message="All files are up-to-date", total_files=len(file_paths))
                # Still validate the existing table: "all unchanged" must not
                # report a missing/empty/mismatched index as a healthy build.
                _rc = self.config.get("retrievers") or self.config.get("retrieval", {})
                _table = self.config["storage"].get("text_table_name") or _rc.get("dense", {}).get("lancedb_table_name", "default_text_table")
                try:
                    from rag_system.model_registry import get_dims
                    _em = self.config.get("embedding_model_name")
                    self._validate_built_index(_table, expected_dim=get_dims(_em) if _em else None)
                except RuntimeError as val_err:
                    indexing_logger.error("post_build_validation_failed", error=str(val_err))
                    report("failed", 100, str(val_err))
                    raise
                return self._print_final_statistics(
                    len(file_paths),
                    0,
                    index_id=index_id,
                    incremental=True,
                    unchanged_count=len(unchanged_files),
                )
        else:
            if force_reindex:
                indexing_logger.info("indexing_mode", mode="force_reindex")
                self.incremental_indexer.reset_index(index_id)
            else:
                indexing_logger.info("indexing_mode", mode="full")
            files_to_index = file_paths
            unchanged_files = []

        indexing_logger.info("files_to_process", files_to_index=len(files_to_index))

        from rag_system.utils.batch_processor import timer, estimate_memory_usage

        retriever_configs = self.config.get("retrievers") or self.config.get("retrieval", {})
        table_name = self.config["storage"].get("text_table_name") or retriever_configs.get("dense", {}).get("lancedb_table_name", "default_text_table")
        enricher_config = self.config.get("contextual_enricher", {})
        enricher_enabled = enricher_config.get("enabled", False)
        total_chunks = 0
        processed_files = 0
        failed_files = 0
        skipped_completed_files = 0
        chunk_cache_hits = 0
        graph_chunks = []
        enriched_chunks_total = 0

        self._start_persistent_worker()
        # Always shut the conversion worker down — including on
        # cancellation — or each cancelled job leaks a Docling subprocess.
        try:
            with timer("Complete Indexing Pipeline"):
                for file_idx, file_path in enumerate(files_to_index, start=1):
                    document_id = os.path.basename(file_path)
                    file_base_progress = 10 + int(((file_idx - 1) / max(len(files_to_index), 1)) * 80)
                    file_done_progress = 10 + int((file_idx / max(len(files_to_index), 1)) * 80)
                    file_id = None
                    active_stage = None
                    try:
                        check_cancelled()
                        if tracker and job_id:
                            file_id = tracker.get_or_create_file_record(job_id, index_id, file_path, document_id)
                            if file_id is not None and not force_reindex and tracker.should_skip_stage(file_id, "storage"):
                                skipped_completed_files += 1
                                indexing_logger.info("skipping_previously_completed_file", document_id=document_id, job_id=job_id)
                                report(
                                    "indexing",
                                    file_done_progress,
                                    f"Skipped previously indexed {document_id}",
                                    file_path=file_path,
                                    filename=document_id,
                                    document_id=document_id,
                                    file_status="done",
                                )
                                continue

                        report(
                            "converting",
                            file_base_progress,
                            f"Converting {document_id} ({file_idx}/{len(files_to_index)})",
                            file_path=file_path,
                            filename=document_id,
                            document_id=document_id,
                            file_status="processing",
                        )
                        indexing_logger.debug("processing_file", document_id=document_id, file_path=file_path)

                        _file_hash = self.incremental_indexer.calculate_file_hash(file_path)
                        cache_key = self._chunk_cache_key(file_path, file_hash=_file_hash)
                        file_chunks = self._load_chunk_cache(cache_key)
                        _chunk_cache_hit = file_chunks is not None
                        if _chunk_cache_hit:
                            chunk_cache_hits += 1
                            indexing_logger.info("chunk_cache_hit", document_id=document_id, file_path=file_path)
                            if file_id is not None:
                                if not tracker.should_skip_stage(file_id, "conversion"):
                                    active_stage = "conversion"
                                    tracker.start_stage(file_id, job_id, active_stage)
                                    complete_tracked_stage(file_id, active_stage, {"file_hash": _file_hash, "cache": True})
                                if not tracker.should_skip_stage(file_id, "chunking"):
                                    active_stage = "chunking"
                                    tracker.start_stage(file_id, job_id, active_stage)
                                    complete_tracked_stage(file_id, active_stage, {"chunks": len(file_chunks), "cache": True})
                                active_stage = None
                        else:
                            check_cancelled()
                            active_stage = "conversion"
                            conversion_completed = start_tracked_stage(file_id, active_stage)
                            report(
                                "chunking",
                                file_base_progress + 3,
                                f"Chunking {document_id}",
                                file_path=file_path,
                                filename=document_id,
                                document_id=document_id,
                                file_status="processing",
                            )
                            if conversion_completed:
                                indexing_logger.info("conversion_stage_marked_complete_but_cache_missing", document_id=document_id)
                            file_chunks = self._convert_and_chunk_file(file_path, document_id)
                            complete_tracked_stage(file_id, "conversion", {"file_hash": _file_hash})
                            active_stage = "chunking"
                            if not start_tracked_stage(file_id, active_stage):
                                complete_tracked_stage(file_id, active_stage, {"chunks": len(file_chunks)})
                            self._save_chunk_cache(cache_key, file_chunks)
                            active_stage = None

                        for i, chunk in enumerate(file_chunks):
                            chunk.setdefault("text", "")
                            if 'metadata' not in chunk:
                                chunk['metadata'] = {}
                            chunk['metadata']['chunk_index'] = i

                        if not file_chunks:
                            indexing_logger.warning("file_no_chunks", document_id=document_id)
                            failed_files += 1
                            if tracker and file_id is not None:
                                tracker.mark_file_failed(file_id, "No chunks generated", error_code="chunking_empty")
                            report(
                                "indexing",
                                file_done_progress,
                                f"Skipped {document_id}: no chunks generated",
                                file_path=file_path,
                                filename=document_id,
                                document_id=document_id,
                                file_status="skipped",
                                chunks_generated=0,
                                file_error="No chunks generated",
                            )
                            continue

                        check_cancelled()
                        active_stage = "overview"
                        overview_completed = start_tracked_stage(file_id, active_stage)
                        if not overview_completed:
                            report(
                                "overview",
                                file_base_progress + 5,
                                f"Generating overview for {document_id}",
                                file_path=file_path,
                                filename=document_id,
                                document_id=document_id,
                                file_status="processing",
                            )
                            try:
                                self.overview_builder.build_and_store(
                                    document_id, file_chunks, force=not _chunk_cache_hit
                                )
                                complete_tracked_stage(file_id, active_stage, {"chunks": len(file_chunks)})
                            except Exception as e:
                                fail_tracked_stage(file_id, active_stage, e)
                                indexing_logger.warning("overview_creation_failed", document_id=document_id, error=str(e))
                        active_stage = None

                        check_cancelled()
                        if (hasattr(self, 'contextual_enricher') and enricher_enabled
                                and enriched_chunks_total + len(file_chunks) > self.max_enrich_chunks):
                            # Budget exhausted: keep indexing, stop enriching —
                            # otherwise huge corpora mean thousands of LLM calls.
                            enricher_enabled = False
                            indexing_logger.warning(
                                "enrichment_budget_reached",
                                budget=self.max_enrich_chunks,
                                enriched_so_far=enriched_chunks_total,
                            )
                            report(
                                "enriching",
                                file_base_progress + 8,
                                f"Enrichment budget reached ({self.max_enrich_chunks} chunks) — remaining files are indexed without enrichment",
                            )
                        if hasattr(self, 'contextual_enricher') and enricher_enabled:
                            active_stage = "enrichment"
                            enrichment_completed = start_tracked_stage(file_id, active_stage)
                            report(
                                "enriching",
                                file_base_progress + 8,
                                f"Enriching {document_id}",
                                file_path=file_path,
                                filename=document_id,
                                document_id=document_id,
                                file_status="processing",
                            )
                            window_size = enricher_config.get("window_size", 1)
                            _pre_enrich_chunks = file_chunks
                            try:
                                if enrichment_completed:
                                    indexing_logger.info("enrichment_stage_completed_previously_rerunning_for_chunks", document_id=document_id)
                                file_chunks = self.contextual_enricher.enrich_chunks(file_chunks, window_size=window_size)
                                if not file_chunks:
                                    indexing_logger.warning("enrichment_returned_empty", document_id=document_id, reverting=True)
                                    file_chunks = _pre_enrich_chunks
                                enriched_chunks_total += len(file_chunks)
                                complete_tracked_stage(file_id, active_stage, {"chunks": len(file_chunks)})
                            except Exception as _enrich_err:
                                fail_tracked_stage(file_id, active_stage, _enrich_err)
                                indexing_logger.error("enrichment_failed", document_id=document_id, error=str(_enrich_err), reverting=True)
                                file_chunks = _pre_enrich_chunks
                            active_stage = None
                        else:
                            indexing_logger.info(
                                "contextual_enrichment_skipped",
                                enabled=enricher_enabled,
                                has_enricher=hasattr(self, 'contextual_enricher'),
                            )

                        check_cancelled()
                        active_stage = "embedding"
                        embedding_completed = start_tracked_stage(file_id, active_stage)
                        report(
                            "embedding",
                            file_base_progress + 12,
                            f"Embedding {len(file_chunks)} chunks from {document_id}",
                            file_path=file_path,
                            filename=document_id,
                            document_id=document_id,
                            file_status="processing",
                            chunks_generated=len(file_chunks),
                        )
                        if hasattr(self, 'vector_indexer') and hasattr(self, 'embedding_generator'):
                            if embedding_completed:
                                indexing_logger.info("embedding_stage_completed_previously_rerunning_for_storage", document_id=document_id)
                            embeddings = self.embedding_generator.generate(file_chunks)
                            complete_tracked_stage(file_id, active_stage, {"chunks": len(file_chunks)})
                            check_cancelled()
                            active_stage = "storage"
                            storage_completed = start_tracked_stage(file_id, active_stage)
                            report(
                                "storing",
                                file_base_progress + 16,
                                f"Storing vectors for {document_id}",
                                file_path=file_path,
                                filename=document_id,
                                document_id=document_id,
                                file_status="processing",
                                chunks_generated=len(file_chunks),
                            )
                            if not storage_completed:
                                if incremental and not force_reindex:
                                    self._delete_existing_documents_from_table(table_name, [document_id])
                                self.vector_indexer.index(table_name, file_chunks, embeddings)

                                if self.latechunk_enabled:
                                    lc_table_name = self.latechunk_cfg.get("lancedb_table_name", f"{table_name}_lc")
                                    lc_vecs = self._generate_latechunk_vectors(document_id, file_chunks)
                                    if lc_vecs is not None and len(lc_vecs) > 0:
                                        if incremental and not force_reindex:
                                            self._delete_existing_documents_from_table(lc_table_name, [document_id])
                                        self.vector_indexer.index(lc_table_name, file_chunks, lc_vecs)
                                complete_tracked_stage(file_id, active_stage, {"chunks": len(file_chunks), "table": table_name})
                            active_stage = None

                        self.incremental_indexer.update_document_metadata(
                            file_path, index_id, len(file_chunks), "index", file_hash=_file_hash
                        )
                        if hasattr(self, 'graph_extractor'):
                            graph_chunks.extend(file_chunks)
                        total_chunks += len(file_chunks)
                        processed_files += 1
                        if tracker and file_id is not None:
                            tracker.mark_file_done(file_id, chunks_generated=len(file_chunks))
                        indexing_logger.info("file_indexed", document_id=document_id, chunk_count=len(file_chunks), memory_mb=estimate_memory_usage(file_chunks))
                        report(
                            "indexing",
                            file_done_progress,
                            f"Indexed {document_id}",
                            file_path=file_path,
                            filename=document_id,
                            document_id=document_id,
                            file_status="done",
                            chunks_generated=len(file_chunks),
                        )

                    except RuntimeError as e:
                        if str(e) == "indexing_cancelled":
                            raise
                        failed_files += 1
                        fail_tracked_stage(file_id, active_stage, e)
                        if tracker and file_id is not None:
                            tracker.mark_file_failed(file_id, str(e), error_code=f"{active_stage or 'processing'}_failed")
                        indexing_logger.error("file_processing_error", file_path=file_path, error=str(e))
                        report(
                            "indexing",
                            file_done_progress,
                            f"Skipped {document_id}: {e}",
                            file_path=file_path,
                            filename=document_id,
                            document_id=document_id,
                            file_status="failed",
                            file_error=str(e),
                        )
                        continue
                    except Exception as e:
                        failed_files += 1
                        fail_tracked_stage(file_id, active_stage, e)
                        if tracker and file_id is not None:
                            tracker.mark_file_failed(file_id, str(e), error_code=f"{active_stage or 'processing'}_failed")
                        indexing_logger.error("file_processing_error", file_path=file_path, error=str(e))
                        report(
                            "indexing",
                            file_done_progress,
                            f"Skipped {document_id}: {e}",
                            file_path=file_path,
                            filename=document_id,
                            document_id=document_id,
                            file_status="failed",
                            file_error=str(e),
                        )
                        continue

                check_cancelled()
                if processed_files == 0:
                    if skipped_completed_files > 0:
                        # All files were already complete from a prior run — validate the
                        # existing table so a corrupted index isn't silently reported healthy.
                        try:
                            from rag_system.model_registry import get_dims
                            embedding_model = self.config.get("embedding_model_name")
                            expected_dim = get_dims(embedding_model) if embedding_model else None
                            self._validate_built_index(table_name, expected_dim=expected_dim)
                        except RuntimeError as val_err:
                            indexing_logger.error("post_build_validation_failed", error=str(val_err))
                            report("failed", 100, str(val_err))
                            self._stop_persistent_worker()
                            raise
                    elif failed_files > 0:
                        # Nothing succeeded and nothing was skipped: this is a
                        # failed build, not a successful empty one.
                        msg = f"Indexing failed: all {failed_files} file(s) failed to process"
                        indexing_logger.error("all_files_failed", failed_files=failed_files)
                        report("failed", 100, msg)
                        raise RuntimeError(msg)
                    else:
                        indexing_logger.warning("no_chunks_generated")
                    self._stop_persistent_worker()
                    return self._print_final_statistics(
                        len(files_to_index) + len(unchanged_files),
                        0,
                        index_id=index_id,
                        incremental=incremental,
                        unchanged_count=len(unchanged_files),
                        chunk_cache_hits=chunk_cache_hits,
                        force_reindex=force_reindex,
                    )

                report("finalizing", 92, "Creating search indexes")
                self._ensure_fts_index(table_name)

                if hasattr(self, 'graph_extractor') and graph_chunks:
                    report("graph", 94, "Extracting knowledge graph")
                    self._extract_knowledge_graph(graph_chunks, retriever_configs)

            # Post-build validation: ensure the table exists, is non-empty, and has correct dims
            try:
                from rag_system.model_registry import get_dims
                embedding_model = self.config.get("embedding_model_name")
                expected_dim = get_dims(embedding_model) if embedding_model else None
                self._validate_built_index(table_name, expected_dim=expected_dim)
            except RuntimeError as val_err:
                indexing_logger.error("post_build_validation_failed", error=str(val_err))
                report("failed", 100, str(val_err))
                self._stop_persistent_worker()
                raise

            report("completed", 100, "Indexing complete")
            self._stop_persistent_worker()
            total_processed = len(files_to_index) + len(unchanged_files)
            indexing_logger.info("indexing_complete", total_processed=total_processed, total_chunks=total_chunks, unchanged_count=len(unchanged_files), incremental=incremental, failed_files=failed_files)
            return self._print_final_statistics(
                total_processed,
                total_chunks,
                index_id=index_id,
                incremental=incremental,
                unchanged_count=len(unchanged_files),
                chunk_cache_hits=chunk_cache_hits,
                force_reindex=force_reindex,
            )
        finally:
            self._stop_persistent_worker()

    def process_documents(self, file_paths: List[str], index_id: str = "default",
                         incremental: bool = True, force_reindex: bool = False):
        """
        Legacy method for processing documents with incremental indexing support.

        Args:
            file_paths: List of file paths to process
            index_id: Unique identifier for this index
            incremental: Whether to use incremental indexing
            force_reindex: Force reindexing of all documents
        """
        return self.run(file_paths, index_id=index_id, incremental=incremental, force_reindex=force_reindex)

    # ------------------------------------------------------------------
    # Persistent conversion worker
    # ------------------------------------------------------------------

    def _start_persistent_worker(self) -> None:
        """Start a long-lived conversion subprocess that keeps Docling models hot."""
        self._worker: Optional[subprocess.Popen] = None
        project_root = Path(__file__).resolve().parents[2]
        env = os.environ.copy()
        existing_pp = env.get("PYTHONPATH")
        env["PYTHONPATH"] = f"{project_root}{os.pathsep}{existing_pp}" if existing_pp else str(project_root)
        try:
            self._worker = subprocess.Popen(
                [sys.executable, "-m", "rag_system.tools.persistent_convert_worker"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                # The worker routes all logging/progress prints to stderr;
                # nothing drains it, so a PIPE would fill up and deadlock the
                # worker mid-conversion. Discard it instead.
                stderr=subprocess.DEVNULL,
                cwd=str(project_root),
                env=env,
                text=True,
                bufsize=1,
            )
            indexing_logger.info("persistent_worker_started", pid=self._worker.pid)
        except Exception as e:
            indexing_logger.warning("persistent_worker_start_failed", error=str(e))
            self._worker = None

    def _validate_built_index(self, table_name: str, expected_dim: Optional[int] = None) -> None:
        """Raise RuntimeError if the freshly-built vector table is empty or has a dimension mismatch."""
        try:
            import lancedb
        except Exception:
            return  # LanceDB not importable; skip validation

        storage = self.config.get("storage", {})
        lancedb_uri = (
            storage.get("db_path")
            or storage.get("lancedb_path")
            or storage.get("lancedb_uri")
            or "./lancedb"
        )
        try:
            conn = lancedb.connect(lancedb_uri)
            table = conn.open_table(table_name)
        except Exception as e:
            raise RuntimeError(f"Post-build validation: could not open table '{table_name}': {e}") from e

        row_count = table.count_rows() if hasattr(table, "count_rows") else None
        if row_count is not None and row_count == 0:
            raise RuntimeError(f"Post-build validation failed: table '{table_name}' is empty after indexing")

        if expected_dim is not None and row_count:
            try:
                sample = table.head(1).to_pydict()
                vec = sample.get("vector", [[]])[0]
                actual_dim = len(vec) if vec else None
                if actual_dim and actual_dim != expected_dim:
                    raise RuntimeError(
                        f"Post-build validation failed: dimension mismatch in '{table_name}' "
                        f"(expected {expected_dim}, got {actual_dim})"
                    )
            except RuntimeError:
                raise
            except Exception:
                pass  # Can't read sample; skip dim check

        indexing_logger.info("post_build_validation_passed", table=table_name, row_count=row_count)

    def _stop_persistent_worker(self) -> None:
        """Gracefully shut down the persistent conversion worker."""
        worker = getattr(self, "_worker", None)
        if worker is None:
            return
        try:
            worker.stdin.close()
            worker.wait(timeout=10)
        except Exception:
            worker.kill()
        self._worker = None
        indexing_logger.info("persistent_worker_stopped")

    def _read_worker_response(self, timeout_s: float) -> str:
        """Read one response line from the worker with a hard timeout."""
        result_q: queue.Queue = queue.Queue()

        def _reader() -> None:
            try:
                line = self._worker.stdout.readline()
                result_q.put(("ok", line))
            except Exception as exc:
                result_q.put(("err", exc))

        threading.Thread(target=_reader, daemon=True).start()
        try:
            kind, value = result_q.get(timeout=timeout_s)
        except queue.Empty:
            self._worker.kill()
            self._worker = None
            raise TimeoutError(f"Conversion worker timed out after {timeout_s}s")
        if kind == "err":
            raise value  # type: ignore[misc]
        return value  # type: ignore[return-value]

    def _convert_via_worker(self, file_path: str, document_id: str) -> List[Dict[str, Any]]:
        """Send one conversion request to the persistent worker; restart if needed."""
        if getattr(self, "_worker", None) is None or self._worker.poll() is not None:
            indexing_logger.warning("persistent_worker_dead_restarting")
            self._start_persistent_worker()

        request = json.dumps(
            {"file_path": file_path, "document_id": document_id, "config": self.config},
            default=str,
        )
        try:
            self._worker.stdin.write(request + "\n")
            self._worker.stdin.flush()
        except BrokenPipeError:
            self._start_persistent_worker()
            self._worker.stdin.write(request + "\n")
            self._worker.stdin.flush()

        raw = self._read_worker_response(self.conversion_timeout_seconds)
        if not raw or not raw.strip():
            raise RuntimeError(f"Worker closed stdout unexpectedly for {document_id}")
        result = json.loads(raw.strip())
        if result.get("error"):
            raise RuntimeError(result["error"])
        return result.get("chunks", [])

    # ------------------------------------------------------------------

    def _convert_and_chunk_file(self, file_path: str, document_id: str) -> List[Dict[str, Any]]:
        # Use persistent worker when available (models stay loaded between files).
        if getattr(self, "_worker", None) is not None:
            return self._convert_via_worker(file_path, document_id)

        project_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(prefix="localgpt_convert_") as tmpdir:
            input_path = Path(tmpdir) / "input.json"
            output_path = Path(tmpdir) / "output.json"
            input_path.write_text(
                json.dumps(
                    {
                        "file_path": file_path,
                        "document_id": document_id,
                        "config": self.config,
                        "output_path": str(output_path),
                    },
                    default=str,
                ),
                encoding="utf-8",
            )

            env = os.environ.copy()
            existing_pythonpath = env.get("PYTHONPATH")
            env["PYTHONPATH"] = (
                f"{project_root}{os.pathsep}{existing_pythonpath}"
                if existing_pythonpath
                else str(project_root)
            )

            try:
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "rag_system.tools.convert_chunk_worker",
                        str(input_path),
                    ],
                    cwd=str(project_root),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=self.conversion_timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired as e:
                raise TimeoutError(
                    f"Conversion timed out after {self.conversion_timeout_seconds}s for {document_id}"
                ) from e

            if completed.returncode != 0:
                stderr = (completed.stderr or "").strip()
                stdout = (completed.stdout or "").strip()
                details = stderr or stdout or f"exit code {completed.returncode}"
                raise RuntimeError(f"Conversion worker failed for {document_id}: {details}")

            if not output_path.exists():
                stderr = (completed.stderr or "").strip()
                raise RuntimeError(
                    f"Conversion worker produced no output for {document_id}"
                    + (f": {stderr}" if stderr else "")
                )

            content = output_path.read_text(encoding="utf-8").strip()
            if not content:
                raise RuntimeError(f"Conversion worker produced empty output for {document_id} (process may have been OOM-killed)")
            result = json.loads(content)
            if result.get("error"):
                raise RuntimeError(result["error"])
            return result.get("chunks", [])

    def _ensure_fts_index(self, table_name: str):
        if not hasattr(self, "lancedb_manager"):
            return
        indexing_logger.info("fts_index_check", table_name=table_name)
        try:
            tbl = self.lancedb_manager.get_table(table_name)
            existing_indices = [idx.name for idx in tbl.list_indices()]
            if not any(name in existing_indices for name in ("text_idx", "fts_text")):
                tbl.create_fts_index(
                    "text",
                    use_tantivy=False,
                    replace=False,
                )
                indexing_logger.info("fts_index_created", table_name=table_name)
            else:
                indexing_logger.info("fts_index_exists", table_name=table_name)
        except Exception as e:
            indexing_logger.error("fts_index_error", table_name=table_name, error=str(e))

    def _generate_latechunk_vectors(self, document_id: str, doc_chunks: List[Dict[str, Any]]):
        full_text_parts = []
        spans = []
        current_pos = 0
        for chunk in doc_chunks:
            chunk_text = chunk["text"]
            full_text_parts.append(chunk_text)
            start = current_pos
            end = start + len(chunk_text)
            spans.append((start, end))
            current_pos = end + 1
        full_doc = "\n".join(full_text_parts)

        try:
            lc_vecs = self.latechunk_encoder.encode(full_doc, spans)
        except Exception as e:
            indexing_logger.warning("latechunk_encode_failed", doc_id=document_id, error=str(e))
            return None

        if len(doc_chunks) == 0 or len(lc_vecs) == 0:
            return None
        if len(lc_vecs) != len(doc_chunks):
            indexing_logger.warning(
                "latechunk_vector_mismatch",
                doc_id=document_id,
                vecs=len(lc_vecs),
                chunks=len(doc_chunks),
            )
            return None
        return lc_vecs

    def _extract_knowledge_graph(self, chunks: List[Dict[str, Any]], retriever_configs: Dict[str, Any]):
        graph_path = retriever_configs.get("graph", {}).get("graph_path", "./index_store/graph/default_graph.gml")
        indexing_logger.info("knowledge_graph_extraction_start", graph_path=graph_path)
        graph_data = self.graph_extractor.extract(chunks)
        graph = nx.DiGraph()
        for entity in graph_data.get('entities', []):
            graph.add_node(entity['id'], type=entity.get('type', 'Unknown'), properties=entity.get('properties', {}))
        for rel in graph_data.get('relationships', []):
            graph.add_edge(rel['source'], rel['target'], label=rel['label'])

        os.makedirs(os.path.dirname(graph_path), exist_ok=True)
        nx.write_gml(graph, graph_path)
        indexing_logger.info("knowledge_graph_saved", graph_path=graph_path, entity_count=len(graph_data.get('entities', [])), relationship_count=len(graph_data.get('relationships', [])))
    
    def _chunk_cache_key(self, file_path: str, file_hash: str | None = None) -> str:
        """Build a stable cache key for conversion/chunking output."""
        if file_hash is None:
            file_hash = self.incremental_indexer.calculate_file_hash(file_path)
        chunking_config = self.config.get("chunking", {})
        cache_fingerprint = {
            "file_hash": file_hash,
            "chunker_mode": self.config.get("chunker_mode", "docling"),
            "chunk_size": chunking_config.get("chunk_size", self.config.get("chunk_size", 1500)),
            "chunk_overlap": chunking_config.get("chunk_overlap", self.config.get("chunk_overlap", 200)),
            "max_tokens": self.config.get("max_tokens", chunking_config.get("chunk_size", self.config.get("chunk_size", 1500))),
            "overlap_sentences": self.config.get("overlap_sentences", 1),
            "embedding_model_name": self.config.get("embedding_model_name", "Qwen/Qwen3-Embedding-0.6B"),
        }
        raw_key = json.dumps(cache_fingerprint, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw_key).hexdigest()

    def _load_chunk_cache(self, cache_key: str) -> List[Dict[str, Any]] | None:
        cache_path = self.chunk_cache_dir / f"{cache_key}.json"
        if not cache_path.exists():
            return None
        try:
            with cache_path.open("r", encoding="utf-8") as f:
                chunks = json.load(f)
            return copy.deepcopy(chunks)
        except Exception as e:
            indexing_logger.warning("chunk_cache_load_failed", cache_key=cache_key, error=str(e))
            return None

    def _save_chunk_cache(self, cache_key: str, chunks: List[Dict[str, Any]]):
        cache_path = self.chunk_cache_dir / f"{cache_key}.json"
        try:
            with cache_path.open("w", encoding="utf-8") as f:
                json.dump(chunks, f)
        except Exception as e:
            indexing_logger.warning("chunk_cache_save_failed", cache_key=cache_key, error=str(e))

    def _delete_existing_documents_from_table(self, table_name: str, document_ids):
        """Remove stale vectors for changed documents before appending fresh rows."""
        document_ids = list(document_ids)
        if not document_ids:
            return
        if not hasattr(self, "lancedb_manager"):
            return
        db = self.lancedb_manager.db
        if not hasattr(db, "table_names") or table_name not in db.table_names():
            return
        try:
            tbl = self.lancedb_manager.get_table(table_name)
            for document_id in document_ids:
                safe_doc_id = str(document_id).replace("'", "''")
                tbl.delete(f"document_id = '{safe_doc_id}'")
            indexing_logger.info("stale_vectors_removed", table_name=table_name, document_count=len(document_ids))
        except Exception as e:
            indexing_logger.warning("stale_vector_removal_failed", table_name=table_name, error=str(e))

    def _print_final_statistics(self, num_files: int, num_chunks: int, index_id: str = "default",
                                incremental: bool = False, unchanged_count: int = 0,
                                chunk_cache_hits: int = 0, force_reindex: bool = False):
        """Log final indexing statistics"""
        processed_files = num_files - unchanged_count if incremental else num_files
        stats = {
            'total_files_considered': num_files,
            'files_processed': processed_files,
            'chunks_generated': num_chunks,
            'average_chunks_per_processed_file': round(num_chunks / processed_files, 1) if processed_files > 0 else 0,
            'incremental': incremental,
            'unchanged_files': unchanged_count,
            'chunk_cache_hits': chunk_cache_hits,
            'force_reindex': force_reindex,
        }

        if hasattr(self, 'incremental_indexer'):
            index_stats = self.incremental_indexer.get_index_stats(index_id)
            stats.update({
                'total_indexed_documents': index_stats['total_documents'],
                'total_indexed_chunks': index_stats['total_chunks'],
                'last_indexed': index_stats['last_indexed'],
            })

        components = []
        if hasattr(self, 'incremental_indexer'):
            components.append('contextual_enrichment')
        if hasattr(self, 'vector_indexer'):
            components.append('vector_fts_index')
        if hasattr(self, 'graph_extractor'):
            components.append('knowledge_graph')

        stats['components'] = components
        stats['batch_sizes'] = {
            'embeddings': self.embedding_batch_size,
            'enrichment': self.enrichment_batch_size,
        }

        indexing_logger.info('final_indexing_statistics', **stats)
        return stats

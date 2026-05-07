from typing import List, Dict, Any, Callable
import copy
import hashlib
import json
import os
from pathlib import Path
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
        self.embedding_batch_size = indexing_config.get("embedding_batch_size", 50)
        self.enrichment_batch_size = indexing_config.get("enrichment_batch_size", 10)
        self.enable_progress_tracking = indexing_config.get("enable_progress_tracking", True)

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
            
            self.contextual_enricher = ContextualEnricher(
                llm_client=self.llm_client,
                llm_model=enrichment_model,
                batch_size=self.enrichment_batch_size
            )

        # Overview builder always enabled for triage routing
        ov_path = self.config.get("overview_path")
        self.overview_builder = OverviewBuilder(
            llm_client=self.llm_client,
            model=self.config.get("overview_model_name", self.ollama_config.get("enrichment_model", "qwen3:0.6b")),
            first_n_chunks=self.config.get("overview_first_n_chunks", 5),
            out_path=ov_path if ov_path else None,
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
            progress_callback: Callable[[str, int, str], None] | None = None,
            cancel_callback: Callable[[], bool] | None = None):
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

        def report(stage: str, progress: int, message: str):
            if progress_callback:
                progress_callback(stage, max(0, min(progress, 100)), message)

        def check_cancelled():
            if cancel_callback and cancel_callback():
                indexing_logger.warning("indexing_cancelled", index_id=index_id)
                report("cancelled", 100, "Indexing cancelled")
                raise RuntimeError("indexing_cancelled")

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
            if not files_to_index:
                indexing_logger.info("indexing_not_required", message="All files are up-to-date", total_files=len(file_paths))
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
        chunk_cache_hits = 0
        graph_chunks = []

        with timer("Complete Indexing Pipeline"):
            for file_idx, file_path in enumerate(files_to_index, start=1):
                document_id = os.path.basename(file_path)
                file_base_progress = 10 + int(((file_idx - 1) / max(len(files_to_index), 1)) * 80)
                file_done_progress = 10 + int((file_idx / max(len(files_to_index), 1)) * 80)
                try:
                    check_cancelled()
                    report("converting", file_base_progress, f"Converting {document_id} ({file_idx}/{len(files_to_index)})")
                    indexing_logger.debug("processing_file", document_id=document_id, file_path=file_path)

                    cache_key = self._chunk_cache_key(file_path)
                    file_chunks = self._load_chunk_cache(cache_key)
                    if file_chunks is not None:
                        chunk_cache_hits += 1
                        indexing_logger.info("chunk_cache_hit", document_id=document_id, file_path=file_path)
                    else:
                        pages_data = self.document_converter.convert_to_markdown(file_path)
                        check_cancelled()
                        report("chunking", file_base_progress + 3, f"Chunking {document_id}")
                        file_chunks = []
                        for tpl in pages_data:
                            if len(tpl) == 3:
                                markdown_text, metadata, doc_obj = tpl
                                if hasattr(self.chunker, "chunk_document"):
                                    chunks = self.chunker.chunk_document(doc_obj, document_id=document_id, metadata=metadata)
                                else:
                                    chunks = self.chunker.chunk(markdown_text, document_id, metadata)
                            else:
                                markdown_text, metadata = tpl
                                chunks = self.chunker.chunk(markdown_text, document_id, metadata)
                            file_chunks.extend(chunks)
                        self._save_chunk_cache(cache_key, file_chunks)

                    for i, chunk in enumerate(file_chunks):
                        chunk.setdefault("text", "")
                        if 'metadata' not in chunk:
                            chunk['metadata'] = {}
                        chunk['metadata']['chunk_index'] = i

                    if not file_chunks:
                        indexing_logger.warning("file_no_chunks", document_id=document_id)
                        failed_files += 1
                        continue

                    check_cancelled()
                    report("overview", file_base_progress + 5, f"Generating overview for {document_id}")
                    try:
                        self.overview_builder.build_and_store(document_id, file_chunks)
                    except Exception as e:
                        indexing_logger.warning("overview_creation_failed", document_id=document_id, error=str(e))

                    check_cancelled()
                    if hasattr(self, 'contextual_enricher') and enricher_enabled:
                        report("enriching", file_base_progress + 8, f"Enriching {document_id}")
                        window_size = enricher_config.get("window_size", 1)
                        file_chunks = self.contextual_enricher.enrich_chunks(file_chunks, window_size=window_size)
                    else:
                        indexing_logger.warning(
                            "contextual_enrichment_skipped",
                            enabled=enricher_enabled,
                            has_enricher=hasattr(self, 'contextual_enricher'),
                        )

                    check_cancelled()
                    report("embedding", file_base_progress + 12, f"Embedding {len(file_chunks)} chunks from {document_id}")
                    if hasattr(self, 'vector_indexer') and hasattr(self, 'embedding_generator'):
                        embeddings = self.embedding_generator.generate(file_chunks)
                        check_cancelled()
                        report("storing", file_base_progress + 16, f"Storing vectors for {document_id}")
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

                    self.incremental_indexer.update_document_metadata(
                        file_path, index_id, len(file_chunks), "index"
                    )
                    if hasattr(self, 'graph_extractor'):
                        graph_chunks.extend(file_chunks)
                    total_chunks += len(file_chunks)
                    processed_files += 1
                    indexing_logger.info("file_indexed", document_id=document_id, chunk_count=len(file_chunks), memory_mb=estimate_memory_usage(file_chunks))
                    report("indexing", file_done_progress, f"Indexed {document_id}")

                except RuntimeError:
                    raise
                except Exception as e:
                    failed_files += 1
                    indexing_logger.error("file_processing_error", file_path=file_path, error=str(e))
                    report("indexing", file_done_progress, f"Skipped {document_id}: {e}")
                    continue

            check_cancelled()
            if processed_files == 0:
                indexing_logger.warning("no_chunks_generated")
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

        report("completed", 100, "Indexing complete")
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
    
    def _chunk_cache_key(self, file_path: str) -> str:
        """Build a stable cache key for conversion/chunking output."""
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

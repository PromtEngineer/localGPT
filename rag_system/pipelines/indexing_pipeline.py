from typing import List, Dict, Any
import os
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
            index_id: str = "default", incremental: bool = True, force_reindex: bool = False):
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

        # Determine which files need processing
        if incremental and not force_reindex:
            indexing_logger.info("incremental_indexing", enabled=True, force_reindex=False)
            files_to_index, unchanged_files = self.incremental_indexer.get_incremental_file_list(
                file_paths, force_reindex
            )

            if unchanged_files:
                indexing_logger.info("skipping_unchanged_files", count=len(unchanged_files))
            if not files_to_index:
                indexing_logger.info("indexing_not_required", message="All files are up-to-date", total_files=len(file_paths))
                self._print_final_statistics(len(file_paths), 0, incremental=True)
                return
        else:
            if force_reindex:
                indexing_logger.info("indexing_mode", mode="force_reindex")
                self.incremental_indexer.reset_index(index_id)
            else:
                indexing_logger.info("indexing_mode", mode="full")
            files_to_index = file_paths
            unchanged_files = []

        indexing_logger.info("files_to_process", files_to_index=len(files_to_index))

        # Import progress tracking utilities
        from rag_system.utils.batch_processor import timer, ProgressTracker, estimate_memory_usage

        with timer("Complete Indexing Pipeline"):
            # Step 1: Document Processing and Chunking
            all_chunks = []
            doc_chunks_map = {}
            with timer("Document Processing & Chunking"):
                file_tracker = ProgressTracker(len(files_to_index), "Document Processing")

                for file_path in files_to_index:
                    try:
                        document_id = os.path.basename(file_path)
                        indexing_logger.debug("processing_file", document_id=document_id, file_path=file_path)

                        pages_data = self.document_converter.convert_to_markdown(file_path)
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

                        # Add a sequential chunk_index to each chunk within the document
                        for i, chunk in enumerate(file_chunks):
                            if 'metadata' not in chunk:
                                chunk['metadata'] = {}
                            chunk['metadata']['chunk_index'] = i

                        # Build and persist document overview (non-blocking errors)
                        try:
                            self.overview_builder.build_and_store(document_id, file_chunks)
                        except Exception as e:
                            indexing_logger.warning("overview_creation_failed", document_id=document_id, error=str(e))

                        # Update incremental indexer with chunk count
                        self.incremental_indexer.update_document_metadata(
                            file_path, index_id, len(file_chunks), "index"
                        )

                        all_chunks.extend(file_chunks)
                        doc_chunks_map[document_id] = file_chunks  # save for late-chunk step
                        indexing_logger.info("chunks_generated", document_id=document_id, chunk_count=len(file_chunks))
                        file_tracker.update(1)

                    except Exception as e:
                        indexing_logger.error("file_processing_error", file_path=file_path, error=str(e))
                        file_tracker.update(1, errors=1)
                        continue

                file_tracker.finish()

            if not all_chunks:
                indexing_logger.warning("no_chunks_generated")
                return

            indexing_logger.info("chunks_total", count=len(all_chunks))
            memory_mb = estimate_memory_usage(all_chunks)
            indexing_logger.info("estimated_memory_usage", memory_mb=memory_mb)

            retriever_configs = self.config.get("retrievers") or self.config.get("retrieval", {})

            # Step 3: Optional Contextual Enrichment (before indexing for consistency)
            enricher_config = self.config.get("contextual_enricher", {})
            enricher_enabled = enricher_config.get("enabled", False)
            
            indexing_logger.debug(
                "contextual_enrichment_debug",
                config_present=bool(enricher_config),
                enabled=enricher_enabled,
                has_enricher=hasattr(self, 'contextual_enricher'),
            )
            
            if hasattr(self, 'contextual_enricher') and enricher_enabled:
                with timer("Contextual Enrichment"):
                    window_size = enricher_config.get("window_size", 1)
                    indexing_logger.info(
                        "contextual_enrichment_started",
                        window_size=window_size,
                        model=self.contextual_enricher.llm_model,
                        batch_size=self.contextual_enricher.batch_size,
                        chunk_count=len(all_chunks),
                    )
                    
                    # This modifies the 'text' field in each chunk dictionary
                    all_chunks = self.contextual_enricher.enrich_chunks(all_chunks, window_size=window_size)
                    
                    indexing_logger.info("contextual_enrichment_complete", enriched_chunks=len(all_chunks), window_size=window_size)
            else:
                indexing_logger.warning(
                    "contextual_enrichment_skipped",
                    enabled=enricher_enabled,
                    has_enricher=hasattr(self, 'contextual_enricher'),
                )

            # Step 4: Create BM25 Index from enriched chunks (for consistency with vector index)
            if hasattr(self, 'vector_indexer') and hasattr(self, 'embedding_generator'):
                with timer("Vector Embedding & Indexing"):
                    table_name = self.config["storage"].get("text_table_name") or retriever_configs.get("dense", {}).get("lancedb_table_name", "default_text_table")
                    indexing_logger.info("vector_embedding_start", embedding_model=self.config.get('embedding_model_name'), vector_table=table_name)
                    
                    embeddings = self.embedding_generator.generate(all_chunks)
                    
                    indexing_logger.info("vector_indexing_start", vector_count=len(embeddings), table_name=table_name)
                    self.vector_indexer.index(table_name, all_chunks, embeddings)
                    indexing_logger.info("vector_embeddings_indexed", vector_count=len(embeddings), table_name=table_name)

                    # Create FTS index on the 'text' field after adding data
                    indexing_logger.info("fts_index_check", table_name=table_name)
                    try:
                        tbl = self.lancedb_manager.get_table(table_name)
                        # LanceDB's default index name is "text_idx" while older
                        # revisions of this pipeline used our own name "fts_text".
                        # Guard against both so we don't attempt to create a     
                        # duplicate index and trigger a LanceError.
                        existing_indices = [idx.name for idx in tbl.list_indices()]
                        if not any(name in existing_indices for name in ("text_idx", "fts_text")):
                            # Use LanceDB default index naming ("text_idx")
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

                    # ---------------------------------------------------
                    # Late-Chunk Embedding + Indexing (optional)
                    # ---------------------------------------------------
                    if self.latechunk_enabled:
                        with timer("Late-Chunk Embedding & Indexing"):
                            lc_table_name = self.latechunk_cfg.get("lancedb_table_name", f"{table_name}_lc")
                            indexing_logger.info("latechunk_embedding_start", table_name=lc_table_name)

                            total_lc_vecs = 0
                            for doc_id, doc_chunks in doc_chunks_map.items():
                                # Build full text and span list
                                full_text_parts = []
                                spans = []
                                current_pos = 0
                                for ch in doc_chunks:
                                    ch_text = ch["text"]
                                    full_text_parts.append(ch_text)
                                    start = current_pos
                                    end = start + len(ch_text)
                                    spans.append((start, end))
                                    current_pos = end + 1  # +1 for newline to join later
                                full_doc = "\n".join(full_text_parts)

                                try:
                                    lc_vecs = self.latechunk_encoder.encode(full_doc, spans)
                                except Exception as e:
                                    indexing_logger.warning("latechunk_encode_failed", doc_id=doc_id, error=str(e))
                                    continue

                                if len(doc_chunks) == 0 or len(lc_vecs) == 0:
                                    # Nothing to index for this document
                                    continue
                                if len(lc_vecs) != len(doc_chunks):
                                    indexing_logger.warning(
                                        "latechunk_vector_mismatch",
                                        doc_id=doc_id,
                                        vecs=len(lc_vecs),
                                        chunks=len(doc_chunks),
                                    )
                                    continue

                                self.vector_indexer.index(lc_table_name, doc_chunks, lc_vecs)
                                total_lc_vecs += len(lc_vecs)

                            indexing_logger.info("latechunk_vectors_indexed", total_lc_vecs=total_lc_vecs)
                
            # Step 6: Knowledge Graph Extraction (Optional)
            if hasattr(self, 'graph_extractor'):
                with timer("Knowledge Graph Extraction"):
                    graph_path = retriever_configs.get("graph", {}).get("graph_path", "./index_store/graph/default_graph.gml")
                    indexing_logger.info("knowledge_graph_extraction_start", graph_path=graph_path)
                    graph_data = self.graph_extractor.extract(all_chunks)
                    G = nx.DiGraph()
                    for entity in graph_data.get('entities', []):
                        G.add_node(entity['id'], type=entity.get('type', 'Unknown'), properties=entity.get('properties', {}))
                    for rel in graph_data.get('relationships', []):
                        G.add_edge(rel['source'], rel['target'], label=rel['label'])
                    
                    os.makedirs(os.path.dirname(graph_path), exist_ok=True)
                    nx.write_gml(G, graph_path)
                    indexing_logger.info("knowledge_graph_saved", graph_path=graph_path, entity_count=len(graph_data.get('entities', [])), relationship_count=len(graph_data.get('relationships', [])))
                    
        indexing_logger.info("indexing_complete", total_processed=len(files_to_index) + len(unchanged_files), total_chunks=len(all_chunks), unchanged_count=len(unchanged_files), incremental=incremental)
        total_processed = len(files_to_index) + len(unchanged_files)
        self._print_final_statistics(total_processed, len(all_chunks), incremental=incremental,
                                   unchanged_count=len(unchanged_files))

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
    
    def _print_final_statistics(self, num_files: int, num_chunks: int, incremental: bool = False, unchanged_count: int = 0):
        """Log final indexing statistics"""
        processed_files = num_files - unchanged_count if incremental else num_files
        stats = {
            'total_files_considered': num_files,
            'files_processed': processed_files,
            'chunks_generated': num_chunks,
            'average_chunks_per_processed_file': round(num_chunks / processed_files, 1) if processed_files > 0 else 0,
            'incremental': incremental,
            'unchanged_files': unchanged_count,
        }

        if hasattr(self, 'incremental_indexer'):
            index_stats = self.incremental_indexer.get_index_stats()
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

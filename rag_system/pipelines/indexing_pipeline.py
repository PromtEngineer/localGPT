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
        
        print(f"🔧 CHUNKING CONFIG: Size: {chunk_size}, Overlap: {chunk_overlap}, Mode: {chunker_mode}")
        
        if chunker_mode == "docling":
            try:
                from rag_system.ingestion.docling_chunker import DoclingChunker
                self.chunker = DoclingChunker(
                    max_tokens=config.get("max_tokens", chunk_size),
                    overlap=config.get("overlap_sentences", 1),
                    tokenizer_model=config.get("embedding_model_name", "qwen3-embedding-0.6b"),
                )
                print("🪄 Using DoclingChunker for high-recall sentence packing.")
            except Exception as e:
                print(f"⚠️  Failed to initialise DoclingChunker: {e}. Falling back to legacy chunker.")
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
            print(f"🔧 ENRICHMENT MODEL: Using '{enrichment_model}' for contextual enrichment")
            
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
                print(f"⚠️  Failed to initialise LateChunkEncoder: {e}. Disabling latechunk retrieval.")
                self.latechunk_enabled = False

    def run(self, file_paths: List[str] | None = None, *, documents: List[str] | None = None):
        """
        Processes and indexes documents based on the pipeline's configuration.
        Accepts legacy keyword *documents* as an alias for *file_paths* so that
        older callers (backend/index builder) keep working.
        """
        # Back-compat shim ---------------------------------------------------
        if file_paths is None and documents is not None:
            file_paths = documents
        if file_paths is None:
            raise TypeError("IndexingPipeline.run() expects 'file_paths' (or alias 'documents') argument")

        print(f"--- Starting indexing process for {len(file_paths)} files. ---")
        
        # Import progress tracking utilities
        from rag_system.utils.batch_processor import timer, ProgressTracker, estimate_memory_usage
        
        with timer("Complete Indexing Pipeline"):
            # Step 1: Document Processing and Chunking
            all_chunks = []
            doc_chunks_map = {}
            with timer("Document Processing & Chunking"):
                file_tracker = ProgressTracker(len(file_paths), "Document Processing")
                
                for file_path in file_paths:
                    try:
                        document_id = os.path.basename(file_path)
                        print(f"Processing: {document_id}")
                        
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
                            print(f"  ⚠️  Failed to create overview for {document_id}: {e}")
                        
                        all_chunks.extend(file_chunks)
                        doc_chunks_map[document_id] = file_chunks  # save for late-chunk step
                        print(f"  Generated {len(file_chunks)} chunks from {document_id}")
                        file_tracker.update(1)
                        
                    except Exception as e:
                        print(f"  ❌ Error processing {file_path}: {e}")
                        file_tracker.update(1, errors=1)
                        continue
                
                file_tracker.finish()

            if not all_chunks:
                print("No text chunks were generated. Skipping indexing.")
                return

            print(f"\n✅ Generated {len(all_chunks)} text chunks total.")
            memory_mb = estimate_memory_usage(all_chunks)
            print(f"📊 Estimated memory usage: {memory_mb:.1f}MB")

            retriever_configs = self.config.get("retrievers") or self.config.get("retrieval", {})

            # Step 3: Optional Contextual Enrichment (before indexing for consistency)
            enricher_config = self.config.get("contextual_enricher", {})
            enricher_enabled = enricher_config.get("enabled", False)
            
            print(f"\n🔍 CONTEXTUAL ENRICHMENT DEBUG:")
            print(f"   Config present: {bool(enricher_config)}")
            print(f"   Enabled: {enricher_enabled}")
            print(f"   Has enricher object: {hasattr(self, 'contextual_enricher')}")
            
            if hasattr(self, 'contextual_enricher') and enricher_enabled:
                with timer("Contextual Enrichment"):
                    window_size = enricher_config.get("window_size", 1)
                    print(f"\n🚀 CONTEXTUAL ENRICHMENT ACTIVE!")
                    print(f"   Window size: {window_size}")
                    print(f"   Model: {self.contextual_enricher.llm_model}")
                    print(f"   Batch size: {self.contextual_enricher.batch_size}")
                    print(f"   Processing {len(all_chunks)} chunks...")
                    
                    # Show before/after example
                    if all_chunks:
                        print(f"   Example BEFORE: '{all_chunks[0]['text'][:100]}...'")
                    
                    # This modifies the 'text' field in each chunk dictionary
                    all_chunks = self.contextual_enricher.enrich_chunks(all_chunks, window_size=window_size)
                    
                    if all_chunks:
                        print(f"   Example AFTER: '{all_chunks[0]['text'][:100]}...'")
                    
                    print(f"✅ Enriched {len(all_chunks)} chunks with context for indexing.")
            else:
                print(f"⚠️  CONTEXTUAL ENRICHMENT SKIPPED:")
                if not hasattr(self, 'contextual_enricher'):
                    print(f"   Reason: No enricher object (config enabled={enricher_enabled})")
                elif not enricher_enabled:
                    print(f"   Reason: Disabled in config")
                print(f"   Chunks will be indexed without contextual enrichment.")

            # Step 4: Create BM25 Index from enriched chunks (for consistency with vector index)
            if hasattr(self, 'vector_indexer') and hasattr(self, 'embedding_generator'):
                with timer("Vector Embedding & Indexing"):
                    table_name = self.config["storage"].get("text_table_name") or retriever_configs.get("dense", {}).get("lancedb_table_name", "default_text_table")
                    print(f"\n--- Generating embeddings with {self.config.get('embedding_model_name')} ---")
                    
                    embeddings = self.embedding_generator.generate(all_chunks)
                    
                    print(f"\n--- Indexing {len(embeddings)} vectors into LanceDB table: {table_name} ---")
                    self.vector_indexer.index(table_name, all_chunks, embeddings)
                    print("✅ Vector embeddings indexed successfully")

                    # Create FTS index on the 'text' field after adding data
                    print(f"\n--- Ensuring Full-Text Search (FTS) index on table '{table_name}' ---")
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
                            print("✅ FTS index created successfully (using Lance native FTS).")
                        else:
                            print("ℹ️  FTS index already exists – skipped creation.")
                    except Exception as e:
                        print(f"❌ Failed to create/verify FTS index: {e}")

                    # ---------------------------------------------------
                    # Late-Chunk Embedding + Indexing (optional)
                    # ---------------------------------------------------
                    if self.latechunk_enabled:
                        with timer("Late-Chunk Embedding & Indexing"):
                            lc_table_name = self.latechunk_cfg.get("lancedb_table_name", f"{table_name}_lc")
                            print(f"\n--- Generating late-chunk embeddings (table={lc_table_name}) ---")

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
                                    print(f"⚠️  LateChunk encode failed for {doc_id}: {e}")
                                    continue

                                if len(doc_chunks) == 0 or len(lc_vecs) == 0:
                                    # Nothing to index for this document
                                    continue
                                if len(lc_vecs) != len(doc_chunks):
                                    print(f"⚠️  Mismatch LC vecs ({len(lc_vecs)}) vs chunks ({len(doc_chunks)}) for {doc_id}. Skipping.")
                                    continue

                                self.vector_indexer.index(lc_table_name, doc_chunks, lc_vecs)
                                total_lc_vecs += len(lc_vecs)

                            print(f"✅ Late-chunk vectors indexed: {total_lc_vecs}")
                
            # Step 6: Knowledge Graph Extraction (Optional)
            if hasattr(self, 'graph_extractor'):
                with timer("Knowledge Graph Extraction"):
                    graph_path = retriever_configs.get("graph", {}).get("graph_path", "./index_store/graph/default_graph.gml")
                    print(f"\n--- Building and saving knowledge graph to: {graph_path} ---")
                    
                    graph_data = self.graph_extractor.extract(all_chunks)
                    G = nx.DiGraph()
                    for entity in graph_data['entities']:
                        G.add_node(entity['id'], type=entity.get('type', 'Unknown'), properties=entity.get('properties', {}))
                    for rel in graph_data['relationships']:
                        G.add_edge(rel['source'], rel['target'], label=rel['label'])
                    
                    os.makedirs(os.path.dirname(graph_path), exist_ok=True)
                    nx.write_gml(G, graph_path)
                    print(f"✅ Knowledge graph saved successfully.")
                    
        print("\n--- ✅ Indexing Complete ---")
        self._print_final_statistics(len(file_paths), len(all_chunks))
    
    def _print_final_statistics(self, num_files: int, num_chunks: int):
        """Print final indexing statistics"""
        print(f"\n📈 Final Statistics:")
        print(f"  Files processed: {num_files}")
        print(f"  Chunks generated: {num_chunks}")
        print(f"  Average chunks per file: {num_chunks/num_files:.1f}")
        
        # Component status
        components = []
        if hasattr(self, 'contextual_enricher'):
            components.append("✅ Contextual Enrichment")
        if hasattr(self, 'vector_indexer'):
            components.append("✅ Vector & FTS Index")
        if hasattr(self, 'graph_extractor'):
            components.append("✅ Knowledge Graph")
            
        print(f"  Components: {', '.join(components)}")
        print(f"  Batch sizes: Embeddings={self.embedding_batch_size}, Enrichment={self.enrichment_batch_size}")

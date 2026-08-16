from typing import List, Dict, Any
import os
from rag_system.ingestion.document_converter import DocumentConverter
from rag_system.ingestion.chunking import MarkdownRecursiveChunker
from rag_system.indexing.representations import EmbeddingGenerator, select_embedder
from rag_system.indexing.embedders import LanceDBManager, VectorIndexer
from rag_system.utils.ollama_client import OllamaClient
from rag_system.indexing.contextualizer import ContextualEnricher
from rag_system.indexing.crossref import annotate_chunks
from rag_system.indexing.overview_builder import OverviewBuilder


def _default_embedding_model() -> str:
    """The single source of truth for the embedding model default."""
    from rag_system.main import EXTERNAL_MODELS
    return EXTERNAL_MODELS["embedding_model"]


class IndexingPipeline:
    def __init__(self, config: Dict[str, Any], ollama_client: OllamaClient, ollama_config: Dict[str, str]):
        self.config = config
        self.llm_client = ollama_client
        self.ollama_config = ollama_config
        self.document_converter = DocumentConverter()
        # Chunker selection: docling (token-based) or legacy (character-based)
        chunker_mode = config.get("chunker_mode", "docling")

        self.embedding_model_name = config.get("embedding_model_name") or _default_embedding_model()

        # Chunk size is the token budget per chunk for both chunkers.
        chunking_config = config.get("chunking", {})
        chunk_size = chunking_config.get(
            "chunk_size", config.get("chunk_size", config.get("max_tokens", 1500))
        )

        print(f"🔧 CHUNKING CONFIG: Size: {chunk_size}, Mode: {chunker_mode}")

        if chunker_mode == "docling":
            try:
                from rag_system.ingestion.docling_chunker import DoclingChunker
                self.chunker = DoclingChunker(
                    max_tokens=chunk_size,
                    overlap=config.get("overlap_sentences", 1),
                    tokenizer_model=self.embedding_model_name,
                )
                print("🪄 Using DoclingChunker for high-recall sentence packing.")
            except Exception as e:
                print(f"⚠️  Failed to initialise DoclingChunker: {e}. Falling back to legacy chunker.")
                self.chunker = MarkdownRecursiveChunker(
                    max_chunk_size=chunk_size,
                    min_chunk_size=max(1, chunk_size // 4),
                    tokenizer_model=self.embedding_model_name,
                )
        else:
            self.chunker = MarkdownRecursiveChunker(
                max_chunk_size=chunk_size,
                min_chunk_size=max(1, chunk_size // 4),
                tokenizer_model=self.embedding_model_name,
            )

        retriever_configs = self.config.get("retrievers") or self.config.get("retrieval", {})
        storage_config = self.config["storage"]
        
        # Get batch processing configuration
        indexing_config = self.config.get("indexing", {})
        self.embedding_batch_size = indexing_config.get("embedding_batch_size", 50)
        self.enrichment_batch_size = indexing_config.get("enrichment_batch_size", 10)
        self.enable_progress_tracking = indexing_config.get("enable_progress_tracking", True)

        # Cross-reference extraction (roadmap item 4.2). On by default: it is a
        # few regexes over text already in memory, adds no LLM call and no second
        # pass, and only writes chunk metadata — an index built with it is
        # byte-identical on the `text` and `vector` columns, so nothing
        # downstream changes until the query-time hop flag is switched on.
        self.extract_crossrefs = bool(indexing_config.get("extract_crossrefs", True))

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
                self.embedding_model_name,
                self.ollama_config.get("host") if isinstance(self.ollama_config, dict) else None,
            )
            self.embedding_generator = EmbeddingGenerator(
                embedding_model=embedding_model, 
                batch_size=self.embedding_batch_size
            )

        enricher_config = self.config.get("contextual_enricher", {})
        self.enricher_enabled = bool(enricher_config.get("enabled", False))
        self.enricher_window_size = enricher_config.get("window_size", 1)
        self.contextual_enricher = None
        if self.enricher_enabled:
            enrichment_model = (
                self.config.get("enrich_model") or  # Per-request override
                self.config.get("enrichment_model_name") or  # Alternative config key
                self.ollama_config.get("enrichment_model") or  # Default from llm config
                self.ollama_config["generation_model"]  # Final fallback
            )
            print(f"🔧 ENRICHMENT MODEL: Using '{enrichment_model}' for contextual enrichment")

            self.contextual_enricher = ContextualEnricher(
                llm_client=self.llm_client,
                llm_model=enrichment_model,
                batch_size=self.enrichment_batch_size
            )

        # Document overviews feed the triage router; on by default.
        overview_config = self.config.get("overview", {})
        self.overview_builder = None
        # Embedded-overview sidecar for the query-time overview prefilter
        # (roadmap item 4.3). One embedding per *document*, so it is negligible
        # next to the per-chunk pass that already runs.
        self.embed_overviews = bool(overview_config.get("embed", True))
        if overview_config.get("enabled", True):
            self.overview_builder = OverviewBuilder(
                llm_client=self.llm_client,
                model=(
                    self.config.get("overview_model_name")
                    or overview_config.get("model")
                    or self.ollama_config.get("enrichment_model")
                    or self.ollama_config["generation_model"]
                ),
                first_n_chunks=self.config.get(
                    "overview_first_n_chunks", overview_config.get("max_chunks", 5)
                ),
                out_path=self.config.get("overview_path") or None,
            )

        # ------------------------------------------------------------------
        # Late-Chunk encoder initialisation (optional)
        # ------------------------------------------------------------------
        self.latechunk_cfg = (
            retriever_configs.get("latechunk")
            or retriever_configs.get("late_chunking")
            or {}
        )
        self.latechunk_enabled = bool(self.latechunk_cfg.get("enabled", False))
        if self.latechunk_enabled:
            try:
                from rag_system.indexing.latechunk import LateChunkEncoder
                self.latechunk_encoder = LateChunkEncoder(model_name=self.embedding_model_name)
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
                        if self.overview_builder is not None:
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
                raise RuntimeError(
                    "No text chunks were generated from the supplied documents — "
                    "conversion or chunking failed for every file. Check the server "
                    "log for per-file conversion errors; nothing was indexed."
                )

            print(f"\n✅ Generated {len(all_chunks)} text chunks total.")
            memory_mb = estimate_memory_usage(all_chunks)
            print(f"📊 Estimated memory usage: {memory_mb:.1f}MB")

            retriever_configs = self.config.get("retrievers") or self.config.get("retrieval", {})
            table_name = self._text_table_name(retriever_configs)

            # Step 1b: Cross-reference extraction (roadmap item 4.2)
            # Runs on the ORIGINAL chunk text, before contextual enrichment
            # rewrites it — an enriched chunk carries an LLM-written preamble
            # that can invent or drop a reference.
            if self.extract_crossrefs:
                with timer("Cross-reference extraction"):
                    known = self._existing_document_ids(table_name)
                    stats = annotate_chunks(doc_chunks_map, known_documents=known)
                    print(
                        f"🔗 Cross-references: {stats['refs']} reference(s) in "
                        f"{stats['chunks_with_refs']} chunk(s); {stats['resolved']} resolved "
                        f"to {stats['documents_linked']} document(s)."
                    )

            # Step 2: Optional Contextual Enrichment (before indexing for consistency)
            if self.contextual_enricher is not None:
                with timer("Contextual Enrichment"):
                    print(
                        f"\n🚀 Contextual enrichment: model={self.contextual_enricher.llm_model}, "
                        f"window={self.enricher_window_size}, batch={self.contextual_enricher.batch_size}, "
                        f"chunks={len(all_chunks)}"
                    )
                    # This modifies the 'text' field in each chunk dictionary
                    all_chunks = self.contextual_enricher.enrich_chunks(
                        all_chunks, window_size=self.enricher_window_size
                    )
                    print(f"✅ Enriched {len(all_chunks)} chunks with context for indexing.")
            else:
                print("\nℹ️  Contextual enrichment disabled; indexing chunks as-is.")

            # Step 3: Embed chunks into LanceDB and build the native FTS index
            if hasattr(self, 'vector_indexer') and hasattr(self, 'embedding_generator'):
                with timer("Vector Embedding & Indexing"):
                    print(f"\n--- Generating embeddings with {self.embedding_model_name} ---")
                    
                    embeddings = self.embedding_generator.generate(all_chunks)
                    
                    print(f"\n--- Indexing {len(embeddings)} vectors into LanceDB table: {table_name} ---")
                    self.vector_indexer.index(table_name, all_chunks, embeddings,
                                              embedding_model=self.embedding_model_name)
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
                            lc_table_name = self.latechunk_cfg.get("lancedb_table_name") or (
                                f"{table_name}{self.latechunk_cfg.get('table_suffix', '_lc')}"
                            )
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

                                self.vector_indexer.index(lc_table_name, doc_chunks, lc_vecs,
                                                          embedding_model=self.embedding_model_name)
                                total_lc_vecs += len(lc_vecs)

                            print(f"✅ Late-chunk vectors indexed: {total_lc_vecs}")

                            # The late-chunk table needs its own FTS index: the
                            # base-table index above does not cover it, and
                            # without one the hybrid retriever's FTS leg fails
                            # on this table and silently degrades to dense-only
                            # (retrievers.py logs "FTS leg failed").
                            if total_lc_vecs:
                                try:
                                    lc_tbl = self.lancedb_manager.get_table(lc_table_name)
                                    lc_indices = [idx.name for idx in lc_tbl.list_indices()]
                                    if not any(n in lc_indices for n in ("text_idx", "fts_text")):
                                        lc_tbl.create_fts_index("text", use_tantivy=False, replace=False)
                                        print(f"✅ FTS index created on late-chunk table '{lc_table_name}'.")
                                except Exception as e:
                                    print(f"❌ Failed to create/verify FTS index on '{lc_table_name}': {e}")

            # Step 4: Embedded-overview sidecar (roadmap item 4.3)
            if (self.overview_builder is not None and self.embed_overviews
                    and hasattr(self, "embedding_generator")):
                with timer("Overview Embedding"):
                    try:
                        n = self.overview_builder.embed_and_store_vectors(
                            self.embedding_generator.model,
                            embedding_model=self.embedding_model_name,
                        )
                        if n:
                            print(f"🧭 Embedded {n} document overview(s) → "
                                  f"{self.overview_builder.vectors_path}")
                    except Exception as e:
                        # A missing sidecar only disables an off-by-default
                        # query-time feature; it must never fail an index build.
                        print(f"⚠️  Failed to embed document overviews: {e}")

        print("\n--- ✅ Indexing Complete ---")
        self._print_final_statistics(len(file_paths), len(all_chunks))
    
    def _text_table_name(self, retriever_configs: Dict[str, Any]) -> str:
        return (
            self.config["storage"].get("text_table_name")
            or retriever_configs.get("dense", {}).get("lancedb_table_name", "default_text_table")
        )

    def _existing_document_ids(self, table_name: str) -> List[str]:
        """Document ids already in the target table, for cross-reference resolution.

        An incremental add should still be able to resolve "Exhibit B" to a
        document indexed last week. Best effort only: any failure here just means
        references resolve against the current batch alone.
        """
        if not table_name or not hasattr(self, "lancedb_manager"):
            return []
        try:
            db = self.lancedb_manager.db
            if hasattr(db, "table_names") and table_name not in db.table_names():
                return []
            tbl = self.lancedb_manager.get_table(table_name)
            arrow = tbl.to_lance().to_table(columns=["document_id"])
            return sorted({d for d in arrow.column("document_id").to_pylist() if d})
        except Exception as e:
            print(f"ℹ️  Cross-reference resolution limited to this batch ({e}).")
            return []

    def _print_final_statistics(self, num_files: int, num_chunks: int):
        """Print final indexing statistics"""
        print(f"\n📈 Final Statistics:")
        print(f"  Files processed: {num_files}")
        print(f"  Chunks generated: {num_chunks}")
        if num_files:
            print(f"  Average chunks per file: {num_chunks/num_files:.1f}")

        # Component status
        components = []
        if self.contextual_enricher is not None:
            components.append("✅ Contextual Enrichment")
        if hasattr(self, 'vector_indexer'):
            components.append("✅ Vector & FTS Index")
        if self.latechunk_enabled:
            components.append("✅ Late Chunking")
        if self.overview_builder is not None:
            components.append("✅ Document Overviews")

        print(f"  Components: {', '.join(components)}")
        print(f"  Batch sizes: Embeddings={self.embedding_batch_size}, Enrichment={self.enrichment_batch_size}")

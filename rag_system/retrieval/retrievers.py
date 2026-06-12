import lancedb
import pickle
import json
from typing import List, Dict, Any
import numpy as np
import networkx as nx
import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import logging
import pandas as pd
import math
import concurrent.futures
from functools import lru_cache

from rag_system.indexing.embedders import LanceDBManager
from rag_system.indexing.representations import QwenEmbedder
from rag_system.indexing.multimodal import LocalVisionModel
from rag_system.utils.logging_utils import log_retrieval_results

# BM25Retriever is no longer needed.
# class BM25Retriever: ...

try:
    from rapidfuzz import process
except ImportError:  # pragma: no cover - graph retrieval is optional.
    process = None
    logging.getLogger(__name__).warning(
        "rapidfuzz is not installed; graph retrieval is disabled. Run: pip install rapidfuzz"
    )

class GraphRetriever:
    def __init__(self, graph_path: str):
        self.graph = nx.read_gml(graph_path)

    def retrieve(self, query: str, k: int = 5, score_cutoff: int = 80) -> List[Dict[str, Any]]:
        print(f"\n--- Performing Graph Retrieval for query: '{query}' ---")
        if process is None:
            return []
        
        query_parts = query.split()
        entities = []
        for part in query_parts:
            match = process.extractOne(part, self.graph.nodes(), score_cutoff=score_cutoff)
            if match and isinstance(match[0], str):
                entities.append(match[0])
        
        retrieved_docs = []
        for entity in set(entities):
            for neighbor in self.graph.neighbors(entity):
                retrieved_docs.append({
                    'chunk_id': f"graph_{entity}_{neighbor}",
                    'text': f"Entity: {entity}, Neighbor: {neighbor}",
                    'score': 1.0,
                    'metadata': {'source': 'graph'}
                })
        
        print(f"Retrieved {len(retrieved_docs)} documents from the graph.")
        return retrieved_docs[:k]

# region === MultiVectorRetriever ===
class MultiVectorRetriever:
    """
    Performs hybrid (vector + FTS) or vector-only retrieval.
    """
    def __init__(self, db_manager: LanceDBManager, text_embedder: QwenEmbedder, vision_model: LocalVisionModel = None, *, fusion_config: Dict[str, Any] | None = None):
        self.db_manager = db_manager
        self.text_embedder = text_embedder
        self.vision_model = vision_model
        self.fusion_config = fusion_config or {"method": "linear", "bm25_weight": 0.5, "vec_weight": 0.5}

        # Lightweight in-memory LRU cache for single-query embeddings (256 entries)
        @lru_cache(maxsize=256)
        def _embed_single(q: str):
            return self.text_embedder.create_embeddings([q])[0]

        self._embed_single = _embed_single

    def retrieve(self, text_query: str, table_name: str, k: int, reranker=None,
                 vector_only: bool = False, fts_only: bool = False,
                 fusion_override: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
        """
        Performs a search on a single LanceDB table.
        If a reranker is provided, it performs a hybrid search.
        vector_only=True skips the FTS leg (e.g. late-chunk `_lc` tables have
        no FTS index); fts_only=True skips the vector leg. fusion_override
        replaces the instance fusion weights for this call only.
        """
        print(f"\n--- Performing Retrieval for query: '{text_query}' on table '{table_name}' ---")
        
        try:
            if table_name is None:
                table_name = "default_text_table"
            tbl = self.db_manager.get_table(table_name)
            
            # Create / fetch cached text embedding for the query
            text_query_embedding = self._embed_single(text_query)
            
            logger = logging.getLogger(__name__)

            # Always perform hybrid lexical + vector search
            logger.debug(
                "Running hybrid search on table '%s' (k=%s, have_reranker=%s)",
                table_name,
                k,
                bool(reranker),
            )

            if reranker:
                logger.debug("Hybrid + reranker path not yet implemented with manual fusion; proceeding without extra reranker.")

            # Manual two-leg hybrid: take half from each modality
            # (or everything from one leg for single-mode searches)
            if vector_only:
                fts_k = 0
            elif fts_only:
                fts_k = k
            else:
                fts_k = k // 2
            vec_k = k - fts_k

            # Run FTS and vector search in parallel to cut latency
            def _run_fts():
                if fts_k == 0:
                    return None
                # Very short queries often underperform → add fuzzy wildcard
                fts_query = text_query
                if len(text_query.split()) == 1:
                    fts_query = f"{text_query}* OR {text_query}~"
                return (
                     tbl.search(query=fts_query, query_type="fts")
                        .limit(fts_k)
                        .to_df()
                 )

            def _run_vec():
                if vec_k == 0:
                    return None
                search = tbl.search(text_query_embedding).limit(vec_k * 2)
                # Use approximate search (nprobes) when an IVF-PQ index exists
                try:
                    if tbl.list_indices():
                        search = search.nprobes(20)
                except Exception:
                    pass
                return search.to_df()

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                fts_future = executor.submit(_run_fts)
                vec_future = executor.submit(_run_vec)
                fts_df = fts_future.result()
                vec_df = vec_future.result()

            frames = [df for df in (fts_df, vec_df) if df is not None]
            if not frames:
                return []
            combined = pd.concat(frames)

            # Remove duplicates preserving first occurrence. Truncation to k
            # happens AFTER fused scores exist — trimming here would keep
            # candidates by concat order (FTS leg first) instead of by score.
            dedup_subset = ["_rowid"] if "_rowid" in combined.columns else (["chunk_id"] if "chunk_id" in combined.columns else None)
            if dedup_subset:
                combined = combined.drop_duplicates(subset=dedup_subset, keep="first")

            results_df = combined
            logger.debug(
                "Hybrid (fts=%s, vec=%s) → %s unique chunks",
                0 if fts_df is None else len(fts_df),
                0 if vec_df is None else len(vec_df),
                len(results_df),
            )
            
            def _row_value(row, *columns):
                # LanceDB returns FTS relevance in '_score' (older versions used
                # 'score'); after concat, rows from the other leg hold NaN there.
                for col in columns:
                    if col in row:
                        val = row.get(col)
                        if val is not None and not (isinstance(val, float) and math.isnan(val)):
                            return float(val)
                return None

            # BM25 scores are unbounded; normalize against the best hit so they
            # are comparable with the (0, 1] vector similarities before fusion.
            max_bm25 = 0.0
            for _, row in results_df.iterrows():
                bm25 = _row_value(row, '_score', 'score')
                if bm25 is not None and bm25 > max_bm25:
                    max_bm25 = bm25

            fusion_cfg = fusion_override or self.fusion_config
            w_bm25 = float(fusion_cfg.get('bm25_weight', 0.5))
            w_vec = float(fusion_cfg.get('vec_weight', 0.5))

            retrieved_docs = []
            for _, row in results_df.iterrows():
                metadata = json.loads(row.get('metadata', '{}'))
                # Add top-level fields back into metadata for consistency if they don't exist
                metadata.setdefault('document_id', row.get('document_id'))
                metadata.setdefault('chunk_index', row.get('chunk_index'))

                bm25 = _row_value(row, '_score', 'score')
                distance = _row_value(row, '_distance')
                vec_sim = 1.0 / (1.0 + distance) if distance is not None else None
                bm25_norm = (bm25 / max_bm25) if (bm25 is not None and max_bm25 > 0) else None

                if bm25_norm is not None and vec_sim is not None:
                    combined_score = w_bm25 * bm25_norm + w_vec * vec_sim
                elif bm25_norm is not None:
                    combined_score = w_bm25 * bm25_norm
                elif vec_sim is not None:
                    combined_score = w_vec * vec_sim
                else:
                    combined_score = 0.0

                retrieved_docs.append({
                    'chunk_id': row.get('chunk_id'),
                    'text': metadata.get('original_text', row.get('text')),
                    'score': combined_score,
                    'bm25': bm25,
                    '_distance': distance,
                    'document_id': row.get('document_id'),
                    'chunk_index': row.get('chunk_index'),
                    'metadata': metadata
                })

            # Rank by fused score and only now trim to k
            retrieved_docs.sort(key=lambda d: d['score'], reverse=True)
            retrieved_docs = retrieved_docs[:k]

            logger.debug("Hybrid search returned %s results", len(retrieved_docs))
            log_retrieval_results(retrieved_docs, k)
            print(f"Retrieved {len(retrieved_docs)} documents.")
            return retrieved_docs
        
        except Exception as e:
            print(f"Could not search table '{table_name}': {e}")
            return []
# endregion

if __name__ == '__main__':
    print("retrievers.py updated for LanceDB FTS Hybrid Search.")

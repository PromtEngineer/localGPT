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

from fuzzywuzzy import process

class GraphRetriever:
    def __init__(self, graph_path: str):
        self.graph = nx.read_gml(graph_path)

    def retrieve(self, query: str, k: int = 5, score_cutoff: int = 80) -> List[Dict[str, Any]]:
        print(f"\n--- Performing Graph Retrieval for query: '{query}' ---")
        
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

    def retrieve(self, text_query: str, table_name: str, k: int, reranker=None) -> List[Dict[str, Any]]:
        """
        Performs a search on a single LanceDB table.
        If a reranker is provided, it performs a hybrid search.
        Otherwise, it performs a standard vector search.
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
            fts_k = k // 2
            vec_k = k - fts_k

            # Run FTS and vector search in parallel to cut latency
            def _run_fts():
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
                return (
                    tbl.search(text_query_embedding)
                       .limit(vec_k * 2)  # fetch extra to allow for dedup
                       .to_df()
                )

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                fts_future = executor.submit(_run_fts)
                vec_future = executor.submit(_run_vec)
                fts_df = fts_future.result()
                vec_df = vec_future.result()

            if vec_df is not None:
                combined = pd.concat([fts_df, vec_df])
            else:
                combined = fts_df

            # Remove duplicates preserving first occurrence, then trim to k
            dedup_subset = ["_rowid"] if "_rowid" in combined.columns else (["chunk_id"] if "chunk_id" in combined.columns else None)
            if dedup_subset:
                combined = combined.drop_duplicates(subset=dedup_subset, keep="first")
            combined = combined.head(k)

            results_df = combined
            logger.debug(
                "Hybrid (fts=%s, vec=%s) → %s unique chunks",
                len(fts_df),
                0 if vec_df is None else len(vec_df),
                len(results_df),
            )
            
            retrieved_docs = []
            for _, row in results_df.iterrows():
                metadata = json.loads(row.get('metadata', '{}'))
                # Add top-level fields back into metadata for consistency if they don't exist
                metadata.setdefault('document_id', row.get('document_id'))
                metadata.setdefault('chunk_index', row.get('chunk_index'))
                
                # Determine score (vector distance or FTS). Replace NaN with 0.0
                raw_score = row.get('_distance') if '_distance' in row else row.get('score')
                try:
                    if raw_score is None or (isinstance(raw_score, float) and math.isnan(raw_score)):
                        raw_score = 0.0
                except Exception:
                    raw_score = 0.0

                combined_score = raw_score
                # Optional linear-weight fusion if both FTS & vector scores exist
                if '_distance' in row and 'score' in row:
                    try:
                        bm25 = row.get('score', 0.0)
                        vec_sim = 1.0 / (1.0 + row.get('_distance', 1.0))  # convert distance to similarity
                        w_bm25 = float(self.fusion_config.get('bm25_weight', 0.5))
                        w_vec = float(self.fusion_config.get('vec_weight', 0.5))
                        combined_score = w_bm25 * bm25 + w_vec * vec_sim
                    except Exception:
                        pass

                retrieved_docs.append({
                    'chunk_id': row.get('chunk_id'),
                    'text': metadata.get('original_text', row.get('text')),
                    'score': combined_score,
                    'bm25': row.get('score'),
                    '_distance': row.get('_distance'),
                    'document_id': row.get('document_id'),
                    'chunk_index': row.get('chunk_index'),
                    'metadata': metadata
                })

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

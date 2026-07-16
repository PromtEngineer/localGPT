import json
from typing import List, Dict, Any
import networkx as nx
import logging
import pandas as pd
import math
import concurrent.futures
from functools import lru_cache

from rag_system.indexing.embedders import LanceDBManager
from rag_system.indexing.representations import QwenEmbedder
from rag_system.utils.logging_utils import log_retrieval_results
from rag_system.retrieval.fusion import fuse_ranked_results

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
    def __init__(self, db_manager: LanceDBManager, text_embedder: QwenEmbedder, vision_model: Any = None, *, fusion_config: Dict[str, Any] | None = None):
        self.db_manager = db_manager
        self.text_embedder = text_embedder
        self.vision_model = vision_model
        self.fusion_config = fusion_config or {"method": "linear", "bm25_weight": 0.5, "vec_weight": 0.5}

        # Lightweight in-memory LRU cache for single-query embeddings (256 entries)
        @lru_cache(maxsize=256)
        def _embed_single(q: str):
            return self.text_embedder.create_embeddings([q])[0]

        self._embed_single = _embed_single

    def retrieve(
        self,
        text_query: str,
        table_name: str,
        k: int,
        reranker=None,
        search_type: str = "hybrid",
        dense_weight: float = 0.7,
    ) -> List[Dict[str, Any]]:
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
            
            logger = logging.getLogger(__name__)
            normalized_search_type = str(search_type or "hybrid").lower()
            aliases = {
                "dense": "vector",
                "vector_only": "vector",
                "semantic": "vector",
                "fts": "lexical",
                "bm25": "lexical",
                "bm25_only": "lexical",
            }
            normalized_search_type = aliases.get(normalized_search_type, normalized_search_type)
            if normalized_search_type not in {"hybrid", "vector", "lexical"}:
                raise ValueError(
                    "search_type must be one of hybrid, vector/dense, or lexical/fts/bm25"
                )

            def _run_fts():
                return (
                    tbl.search(query=text_query, query_type="fts")
                    .limit(max(k * 2, k))
                    .to_df()
                )

            def _run_vec():
                embedding = self._embed_single(text_query)
                return tbl.search(embedding).limit(max(k * 2, k)).to_df()

            fts_df = pd.DataFrame()
            vec_df = pd.DataFrame()
            if normalized_search_type == "hybrid":
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    fts_future = executor.submit(_run_fts)
                    vec_future = executor.submit(_run_vec)
                    try:
                        fts_df = fts_future.result()
                    except Exception as exc:
                        logger.warning("Lexical search failed; using dense results: %s", exc)
                    try:
                        vec_df = vec_future.result()
                    except Exception as exc:
                        logger.warning("Dense search failed; using lexical results: %s", exc)
            elif normalized_search_type == "lexical":
                fts_df = _run_fts()
            else:
                vec_df = _run_vec()

            def _records(frame):
                return [] if frame is None or frame.empty else frame.to_dict("records")

            lexical_rows = _records(fts_df)
            dense_rows = _records(vec_df)
            if normalized_search_type == "hybrid":
                rows = fuse_ranked_results(
                    lexical_rows,
                    dense_rows,
                    k=k,
                    dense_weight=dense_weight,
                )
            else:
                rows = (lexical_rows or dense_rows)[:k]

            retrieved_docs = []
            for row in rows:
                raw_metadata = row.get('metadata', '{}')
                try:
                    metadata = json.loads(raw_metadata) if isinstance(raw_metadata, str) else dict(raw_metadata or {})
                except (TypeError, json.JSONDecodeError):
                    metadata = {}
                # Read records produced by older versions, which serialized the
                # whole chunk into the metadata column.
                if isinstance(metadata.get("metadata"), dict):
                    nested = metadata["metadata"]
                    nested.setdefault("original_text", metadata.get("text"))
                    metadata = nested
                # Add top-level fields back into metadata for consistency if they don't exist
                metadata.setdefault('document_id', row.get('document_id'))
                metadata.setdefault('chunk_index', row.get('chunk_index'))
                
                # Determine score (vector distance or FTS). Replace NaN with 0.0
                raw_score = row.get('score')
                if raw_score is None and row.get('_distance') is not None:
                    raw_score = 1.0 / (1.0 + float(row['_distance']))
                if raw_score is None:
                    raw_score = row.get('_score', 0.0)
                try:
                    if raw_score is None or (isinstance(raw_score, float) and math.isnan(raw_score)):
                        raw_score = 0.0
                except Exception:
                    raw_score = 0.0

                retrieved_docs.append({
                    'chunk_id': row.get('chunk_id'),
                    'text': metadata.get('original_text', row.get('text')),
                    'score': raw_score,
                    'bm25': row.get('_score'),
                    '_distance': row.get('_distance'),
                    'document_id': row.get('document_id'),
                    'chunk_index': row.get('chunk_index'),
                    'metadata': metadata
                })

            logger.debug("%s search returned %s results", normalized_search_type, len(retrieved_docs))
            log_retrieval_results(retrieved_docs, k)
            print(f"Retrieved {len(retrieved_docs)} documents.")
            return retrieved_docs
        
        except Exception as e:
            print(f"Could not search table '{table_name}': {e}")
            return []
# endregion

if __name__ == '__main__':
    print("retrievers.py updated for LanceDB FTS Hybrid Search.")

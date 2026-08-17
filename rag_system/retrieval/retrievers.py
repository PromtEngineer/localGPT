import json
from typing import Any, Dict, List, Optional, Tuple
import logging
import math
import concurrent.futures
from functools import lru_cache

from rag_system.indexing.embedders import (
    EmbedderMismatchError,
    LanceDBManager,
    assert_embedder_matches,
    l2_normalize,
    legacy_table_warning,
    read_table_marker,
)
from rag_system.indexing.representations import QwenEmbedder
from rag_system.utils.logging_utils import log_retrieval_results

# Retrieval modes accepted by MultiVectorRetriever.retrieve().
RETRIEVAL_MODES = ("hybrid", "vector_only", "fts_only")

# Reciprocal-rank-fusion constant (Cormack et al. 2009); dampens the influence
# of the top ranks so a single leg cannot dominate the fused ordering.
_RRF_K = 60


def _is_nan(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value)


def _finite(value: Any) -> Optional[float]:
    """Return *value* as a float, or None when it is missing/NaN/Inf."""
    if value is None or _is_nan(value):
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(num) or math.isinf(num):
        return None
    return num


# GraphRetriever was removed on 2026-08-09 (roadmap item 2.5). GraphRAG loses on
# single-hop retrieval, its multi-hop gains are contested, and it costs 41–57x at
# indexing and up to ~377x in query tokens — see Documentation/research/
# academic-evidence-2026.md §6. It was also unreachable: no shipped profile ever
# set `graph_strategy`.


# region === MultiVectorRetriever ===
class MultiVectorRetriever:
    """
    Runs LanceDB full-text and/or vector search over a single table.
    """
    def __init__(self, db_manager: LanceDBManager, text_embedder: QwenEmbedder):
        self.db_manager = db_manager
        self.text_embedder = text_embedder

        # Lightweight in-memory LRU cache for single-query embeddings (256 entries).
        # The cache holds the raw model output; normalization is applied after the
        # lookup because it is a property of the table being searched, not of the
        # query (a legacy table must be searched with legacy vectors).
        @lru_cache(maxsize=256)
        def _embed_single(q: str):
            return self.text_embedder.create_embeddings([q])[0]

        self._embed_single = _embed_single

        # Tables already reported as unmarked, so the warning is printed once each.
        self._legacy_tables_warned: set = set()

    def _check_table_identity(self, tbl, table_name: str) -> bool:
        """Verify the table's embedder against ours; return whether to normalize.

        Raises ``EmbedderMismatchError`` when the table was written by a
        different embedding model — a same-width swap (harrier-oss-v1-0.6b vs
        Qwen3-Embedding-0.6B, both 1024-dim) is otherwise undetectable and would
        silently return nonsense.
        """
        marker = read_table_marker(
            tbl, getattr(self.db_manager, "db_path", None), table_name
        )
        if marker is None:
            if table_name not in self._legacy_tables_warned:
                self._legacy_tables_warned.add(table_name)
                print(legacy_table_warning(table_name))
            return False
        configured = getattr(self.text_embedder, "model_name", None)
        if configured:
            assert_embedder_matches(table_name, marker, configured)
        return bool(marker["normalized"])

    @staticmethod
    def _dedup_key(row: Dict[str, Any]) -> Tuple[str, Any]:
        for field in ("chunk_id", "_rowid"):
            value = row.get(field)
            if value is not None and not _is_nan(value):
                return (field, value)
        return ("text", row.get("text"))

    def retrieve(self, text_query: str, table_name: str, k: int, search_type: str = "hybrid",
                 *, where: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieves up to *k* chunks from *table_name*.

        `search_type` selects which LanceDB query legs run: "hybrid" (full-text
        + vector, fused with reciprocal rank fusion), "vector_only", or
        "fts_only". Unknown values fall back to "hybrid".

        `where` is a LanceDB SQL predicate applied as a **prefilter to every
        leg** (roadmap item 4.4). It is keyword-only and defaults to None, in
        which case not a single extra call is made and the behaviour is exactly
        what it was before filters existed. It must come from
        ``rag_system.retrieval.filters.compile_filters`` — nothing here
        validates or escapes it, and nothing here will repair it.

        Prefiltering (rather than filtering the k results afterwards) is the
        point: a post-filter would return "up to k, minus whatever the filter
        removed", so a filter for a rare document would return an empty list
        while the document sat in the table. Both legs are filtered, so a hybrid
        query cannot leak an excluded chunk in through the BM25 side.
        """
        mode = (search_type or "hybrid").lower()
        if mode not in RETRIEVAL_MODES:
            print(f"⚠️  Unknown retrieval mode '{search_type}'; falling back to 'hybrid'.")
            mode = "hybrid"

        print(f"\n--- Performing {mode} retrieval for query: '{text_query}' on table '{table_name}' ---")
        if where:
            print(f"🔎 Metadata filter (prefilter, both legs): {where}")

        try:
            if table_name is None:
                table_name = "default_text_table"
            tbl = self.db_manager.get_table(table_name)
            # Vectors are stored L2-normalized on v4+ tables, so LanceDB's default
            # L2 ordering equals the cosine ordering both model cards specify. The
            # query vector has to be normalized the same way — and must *not* be
            # for a legacy table, whose documents are unnormalized.
            normalize_query = self._check_table_identity(tbl, table_name)

            logger = logging.getLogger(__name__)
            logger.debug("Running %s search on table '%s' (k=%s)", mode, table_name, k)

            def _filtered(query_builder):
                """Apply the metadata prefilter, when there is one."""
                if where:
                    return query_builder.where(where, prefilter=True)
                return query_builder

            def _run_fts():
                # LanceDB's FTS parser reads double quotes as phrase syntax and
                # raises on queries it cannot parse — the decomposer emits
                # quoted sub-queries like `"Extra Fees & Costs" charges`, which
                # used to kill the whole retrieve() (hybrid included) and return
                # nothing. Quotes carry no ranking signal we rely on, so strip
                # them and search the plain terms.
                fts_query = text_query.replace('"', " ").strip() or text_query
                # Very short queries often underperform → add fuzzy wildcard
                if len(fts_query.split()) == 1:
                    fts_query = f"{fts_query}* OR {fts_query}~"
                return (
                    _filtered(tbl.search(query=fts_query, query_type="fts"))
                       .limit(k)
                       .to_pandas()
                )

            def _run_vec():
                vector = self._embed_single(text_query)
                if normalize_query:
                    vector = l2_normalize(vector)
                return (
                    _filtered(tbl.search(vector))
                       .limit(k)
                       .to_pandas()
                )

            fts_df = None
            vec_df = None
            if mode == "fts_only":
                fts_df = _run_fts()
            elif mode == "vector_only":
                vec_df = _run_vec()
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    fts_future = executor.submit(_run_fts)
                    vec_future = executor.submit(_run_vec)
                    # In hybrid mode a failed FTS leg degrades to dense-only
                    # instead of failing the whole retrieval — the two legs are
                    # redundant by design, and "one leg down" must not become
                    # "0 results". (fts_only mode still propagates, because
                    # there the caller asked for exactly that leg.)
                    try:
                        fts_df = fts_future.result()
                    except Exception as fts_err:
                        print(f"⚠️  FTS leg failed ({fts_err}); continuing dense-only.")
                        logger.warning("FTS leg failed on table '%s': %s", table_name, fts_err)
                        fts_df = None
                    vec_df = vec_future.result()

            def _records(df) -> List[Dict[str, Any]]:
                if df is None or len(df) == 0:
                    return []
                return df.to_dict("records")

            fts_rows = _records(fts_df)
            vec_rows = _records(vec_df)

            # Reciprocal rank fusion: each leg contributes 1/(_RRF_K + rank).
            # A single-leg run keeps its native ordering because RRF is
            # monotonically decreasing in rank.
            fused: Dict[Tuple[str, Any], Dict[str, Any]] = {}
            for rank, row in enumerate(fts_rows, start=1):
                entry = fused.setdefault(self._dedup_key(row), {"row": row, "rrf": 0.0, "bm25": None, "distance": None})
                entry["rrf"] += 1.0 / (_RRF_K + rank)
                # LanceDB FTS scores come back as `_score` (lancedb 0.36), not
                # `score` — reading the latter left bm25 permanently None.
                entry["bm25"] = _finite(row.get("_score"))
            for rank, row in enumerate(vec_rows, start=1):
                entry = fused.setdefault(self._dedup_key(row), {"row": row, "rrf": 0.0, "bm25": None, "distance": None})
                entry["rrf"] += 1.0 / (_RRF_K + rank)
                entry["distance"] = _finite(row.get("_distance"))

            ordered = sorted(fused.values(), key=lambda e: e["rrf"], reverse=True)[:k]

            logger.debug(
                "%s search (fts=%s, vec=%s) → %s unique chunks",
                mode,
                len(fts_rows),
                len(vec_rows),
                len(ordered),
            )

            retrieved_docs = []
            for entry in ordered:
                row = entry["row"]
                raw_metadata = row.get('metadata')
                if isinstance(raw_metadata, dict):
                    metadata = dict(raw_metadata)
                else:
                    try:
                        metadata = json.loads(raw_metadata or '{}')
                    except (TypeError, ValueError):
                        metadata = {}
                # Add top-level fields back into metadata for consistency if they don't exist
                metadata.setdefault('document_id', row.get('document_id'))
                metadata.setdefault('chunk_index', row.get('chunk_index'))

                # A single score field per mode, always "higher is better".
                if mode == "fts_only":
                    score = entry["bm25"] if entry["bm25"] is not None else 0.0
                elif mode == "vector_only":
                    distance = entry["distance"]
                    score = 1.0 / (1.0 + distance) if distance is not None else 0.0
                else:
                    score = entry["rrf"]

                doc = {
                    'chunk_id': row.get('chunk_id'),
                    'text': metadata.get('original_text') or row.get('text') or '',
                    'score': score,
                    'document_id': row.get('document_id'),
                    'chunk_index': row.get('chunk_index'),
                    'metadata': metadata
                }
                # Only carry the per-leg raw scores when that leg actually hit,
                # so downstream sorting never sees a None.
                if entry["bm25"] is not None:
                    doc['bm25'] = entry["bm25"]
                if entry["distance"] is not None:
                    doc['_distance'] = entry["distance"]
                retrieved_docs.append(doc)

            log_retrieval_results(retrieved_docs, k)
            print(f"Retrieved {len(retrieved_docs)} documents.")
            return retrieved_docs

        except EmbedderMismatchError:
            # Never degrade an embedder mismatch into "0 results" — the whole
            # point of the guard is that the user has to see it.
            raise
        except Exception as e:
            if where:
                # A filtered search that fails must not look like a filtered
                # search that matched nothing. The caller asked to be restricted
                # to part of the corpus; "0 results" and "the restriction did
                # not run" are different answers and only one of them is safe.
                print(f"❌ Filtered search failed on table '{table_name}' "
                      f"(where: {where}): {e}")
                raise
            print(f"Could not search table '{table_name}': {e}")
            return []
# endregion

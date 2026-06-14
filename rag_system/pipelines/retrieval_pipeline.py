import concurrent.futures
import json
import logging
import math
import time
from threading import Lock
from typing import Any, Dict, List, Optional

import numpy as np

import lancedb
from rag_system.indexing.embedders import LanceDBManager
from rag_system.indexing.representations import select_embedder
from rag_system.rerankers.reranker import QwenReranker
from rag_system.rerankers.sentence_pruner import SentencePruner
from rag_system.retrieval.retrievers import GraphRetriever, MultiVectorRetriever
from rag_system.utils.ollama_client import OllamaClient

# from rag_system.indexing.chunk_store import ChunkStore


logger = logging.getLogger(__name__)

# Cap on collections searched per query (NVIDIA blueprint uses the same
# limit) — each adds an embedding + retrieval + its share of rerank cost.
MAX_COLLECTIONS = 5

_UUID_PREFIX_RE = None


def _source_display_name(document_id) -> str:
    """Human-readable source name: stored filenames are '<uuid4>_<original>'."""
    global _UUID_PREFIX_RE
    import re

    if _UUID_PREFIX_RE is None:
        _UUID_PREFIX_RE = re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_",
            re.IGNORECASE,
        )
    return _UUID_PREFIX_RE.sub("", str(document_id or "unknown"))


# ---------------------------------------------------------------------------
# Thread-safety helpers
# ---------------------------------------------------------------------------

# 1. ColBERT (via `rerankers` lib) is not thread-safe.  We protect the actual
#    `.rank()` call with `_rerank_lock`.
_rerank_lock: Lock = Lock()

# 2. Loading a large cross-encoder or ColBERT model can easily take >1 GB of
#    RAM.  When multiple sub-queries are processed in parallel they may try to
#    instantiate the reranker simultaneously, which results in PyTorch meta
#    tensor errors.  We therefore guard the *initialisation* with its own
#    lock so only one thread carries out the heavy `from_pretrained()` call.
_ai_reranker_init_lock: Lock = Lock()

# Lock to serialise first-time Provence model load
_sentence_pruner_lock: Lock = Lock()

# Retriever construction is lazy and can be reached by simultaneous requests.
# Protect only first-time initialization; retrieval itself remains concurrent.
_retriever_init_lock: Lock = Lock()


class RetrievalPipeline:
    """
    Orchestrates the state-of-the-art multimodal RAG pipeline.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        ollama_client: OllamaClient,
        ollama_config: Dict[str, Any],
    ):
        self.config = config
        self.ollama_config = ollama_config
        self.ollama_client = ollama_client

        # Support both legacy "retrievers" key and newer "retrieval" key
        self.retriever_configs = self.config.get("retrievers") or self.config.get(
            "retrieval", {}
        )
        self.storage_config = self.config["storage"]

        # Defer initialization to just-in-time methods
        self.db_manager: Optional[LanceDBManager] = None
        self.text_embedder: Any = None
        self.dense_retriever: Optional[MultiVectorRetriever] = None
        # Use a private attribute to avoid clashing with the public property
        self._graph_retriever: Optional[GraphRetriever] = None
        # Holds a LanceDB internal reranker (type not visible to mypy).
        self.reranker: Any = None
        self.ai_reranker: Optional[QwenReranker] = None

    def _get_db_manager(self):
        if self.db_manager is None:
            # Accept either "db_path" (preferred) or legacy "lancedb_uri"
            db_path = self.storage_config.get("db_path") or self.storage_config.get(
                "lancedb_uri"
            )
            if not db_path:
                raise ValueError(
                    "Storage config must contain 'db_path' or 'lancedb_uri'."
                )
            self.db_manager = LanceDBManager(db_path=db_path)
        return self.db_manager

    def _get_text_embedder(self):
        if self.text_embedder is None:
            from rag_system.indexing.representations import select_embedder

            self.text_embedder = select_embedder(
                self.config.get("embedding_model_name", "BAAI/bge-small-en-v1.5"),
                (
                    self.ollama_config.get("host")
                    if isinstance(self.ollama_config, dict)
                    else None
                ),
            )
        return self.text_embedder

    def _get_dense_retriever(self):
        """Ensure a dense MultiVectorRetriever is always available unless explicitly disabled."""
        if self.dense_retriever is None:
            # If the config explicitly sets dense.enabled to False, respect it
            if self.retriever_configs.get("dense", {}).get("enabled", True) is False:
                return None

            with _retriever_init_lock:
                if self.dense_retriever is None:
                    try:
                        db_manager = self._get_db_manager()
                        text_embedder = self._get_text_embedder()
                        fusion_cfg = self.config.get("fusion", {})
                        self.dense_retriever = MultiVectorRetriever(
                            db_manager,
                            text_embedder,
                            vision_model=None,  # type: ignore[arg-type]
                            fusion_config=fusion_cfg,
                        )
                    except Exception as e:
                        logger.error(
                            "dense_retriever_initialization_failed error=%s", e
                        )
                        self.dense_retriever = None
        return self.dense_retriever

    def _get_graph_retriever(self):
        if self._graph_retriever is None and self.retriever_configs.get(
            "graph", {}
        ).get("enabled"):
            self._graph_retriever = GraphRetriever(
                graph_path=self.storage_config["graph_path"]
            )
        return self._graph_retriever

    def _get_retriever_for_model(self, embedding_model: Optional[str]):
        """Retriever for a specific embedding model (multi-collection support).

        Each collection may have been embedded with a different model, so the
        query must be embedded per-model. Instances are cached; the underlying
        HF weights are additionally cached process-wide by select_embedder.
        Falls back to the pipeline's default retriever when no model is given
        or it matches the current config.
        """
        current = self.config.get("embedding_model_name")
        if not embedding_model or embedding_model == current:
            return self._get_dense_retriever()
        if not hasattr(self, "_retrievers_by_model"):
            self._retrievers_by_model: Dict[str, MultiVectorRetriever] = {}
        if embedding_model not in self._retrievers_by_model:
            with _retriever_init_lock:
                if embedding_model not in self._retrievers_by_model:
                    try:
                        self._retrievers_by_model[embedding_model] = (
                            MultiVectorRetriever(
                                self._get_db_manager(),
                                select_embedder(embedding_model),
                                vision_model=None,  # type: ignore[arg-type]
                                fusion_config=self.config.get("fusion", {}),
                            )
                        )
                    except Exception as e:
                        logger.error(
                            "retriever_init_failed model=%s error=%s",
                            embedding_model,
                            e,
                        )
                        return None
        return self._retrievers_by_model[embedding_model]

    def _resolve_latechunk_cfg(self) -> Dict[str, Any]:
        """Resolve late-chunk config at query time.

        Callers (the API server) toggle config["retrievers"]["latechunk"]
        after construction, and older configs use retrieval.late_chunking —
        the binding captured in __init__ sees neither.
        """
        cfg = (self.config.get("retrievers") or {}).get("latechunk")
        if cfg is None:
            cfg = (self.config.get("retrieval") or {}).get("late_chunking") or {}
        return cfg

    @staticmethod
    def _lancedb_table_exists(retriever, table_name: str) -> bool:
        try:
            return table_name in set(retriever.db_manager.db.table_names(limit=10_000))
        except Exception:
            return False

    def _get_reranker(self):
        """Initializes the reranker for hybrid search score fusion."""
        reranker_config = self.config.get("reranker", {})
        # This is for the LanceDB internal reranker, not the AI one.
        if (
            self.reranker is None
            and reranker_config.get("type") == "linear_combination"
        ):
            rerank_weight = reranker_config.get("weight", 0.5)
            self.reranker = lancedb.rerankers.LinearCombinationReranker(  # type: ignore[attr-defined]
                weight=rerank_weight
            )
            logger.info(
                "linear_combination_reranker_initialized weight=%s", rerank_weight
            )
        return self.reranker

    def _get_ai_reranker(self, enabled_override: Optional[bool] = None):
        """Initializes a dedicated AI-based reranker.

        enabled_override lets a single request turn reranking on/off without
        mutating shared config: None → use config's reranker.enabled; False →
        skip (return None); True → build even if config defaults it off,
        falling back to sane model/strategy defaults when unset.
        """
        reranker_config = self.config.get("reranker", {})
        enabled = (
            reranker_config.get("enabled")
            if enabled_override is None
            else enabled_override
        )
        if not enabled:
            return None
        if self.ai_reranker is None:
            # Serialise first-time initialisation so only one thread attempts
            # to load the (very large) model.  Other threads will wait and use
            # the instance once ready, preventing the meta-tensor crash.
            with _ai_reranker_init_lock:
                # Another thread may have completed init while we waited
                if self.ai_reranker is None:
                    try:
                        # Defaults cover a request that enables reranking when
                        # config left it off (e.g. fast mode) — previously the
                        # caller mutated config to inject these.
                        model_name = reranker_config.get(
                            "model_name"
                        ) or self.ollama_config.get(
                            "rerank_model", "answerai-colbert-small-v1"
                        )
                        strategy = reranker_config.get("strategy") or (
                            "rerankers-lib"
                            if reranker_config.get("model_name") is None
                            else "qwen"
                        )

                        logger.info(
                            "ai_reranker_initializing strategy=%s model_name=%s",
                            strategy,
                            model_name,
                        )
                        if strategy == "rerankers-lib":
                            from rerankers import Reranker

                            self.ai_reranker = Reranker(
                                model_name, model_type="colbert"
                            )
                        else:
                            self.ai_reranker = QwenReranker(model_name=model_name)

                        logger.info(
                            "ai_reranker_initialized strategy=%s model_name=%s",
                            strategy,
                            model_name,
                        )
                    except Exception as e:
                        # Leave as None so the pipeline can proceed without reranking
                        logger.error(
                            "ai_reranker_initialization_failed model_name=%s error=%s",
                            model_name,
                            e,
                        )
        return self.ai_reranker

    def _get_sentence_pruner(self):
        if getattr(self, "_sentence_pruner", None) is None:
            with _sentence_pruner_lock:
                if getattr(self, "_sentence_pruner", None) is None:
                    self._sentence_pruner = SentencePruner()
        return self._sentence_pruner

    def _get_surrounding_chunks_lancedb(
        self, chunk: Dict[str, Any], window_size: int
    ) -> List[Dict[str, Any]]:
        """
        Retrieves a window of chunks around a central chunk using LanceDB.
        """
        db_manager = self._get_db_manager()
        if not db_manager:
            return [chunk]

        # Extract identifiers needed for the query
        document_id = chunk.get("document_id")
        chunk_index = chunk.get("chunk_index")

        # If essential identifiers are missing, return the chunk itself
        if document_id is None or chunk_index is None or chunk_index == -1:
            return [chunk]

        # Multi-collection: neighbors must come from the doc's OWN table,
        # not whatever table the pipeline config currently points at
        table_name = (
            chunk.get("_source_table") or self.config["storage"]["text_table_name"]
        )
        try:
            tbl = db_manager.get_table(table_name)
        except Exception:
            # If the table can't be opened, we can't get surrounding chunks
            return [chunk]

        # Define the window for the search
        start_index = max(0, chunk_index - window_size)
        end_index = chunk_index + window_size

        # Construct the SQL filter for an efficient metadata-based search.
        # document_id is the uploaded filename — escape quotes (O'Brien.pdf)
        # the same way _delete_existing_documents_from_table does.
        escaped_document_id = str(document_id).replace("'", "''")
        sql_filter = f"document_id = '{escaped_document_id}' AND chunk_index >= {int(start_index)} AND chunk_index <= {int(end_index)}"

        try:
            # Execute a filter-only search, which is very fast on indexed metadata
            results = tbl.search().where(sql_filter).to_list()

            # The results must be sorted by chunk_index to maintain logical order
            results.sort(key=lambda c: c["chunk_index"])

            # The 'metadata' field is a JSON string and needs to be parsed
            for res in results:
                if isinstance(res.get("metadata"), str):
                    try:
                        res["metadata"] = json.loads(res["metadata"])
                    except json.JSONDecodeError:
                        res["metadata"] = {}  # Handle corrupted metadata gracefully
            return results
        except Exception:
            # If the query fails for any reason, fall back to the single chunk
            return [chunk]

    def _synthesize_final_answer(
        self,
        query: str,
        facts: str,
        *,
        event_callback=None,
        generation_model: Optional[str] = None,
    ) -> str:
        """Uses a text LLM to synthesize a final answer from extracted facts."""
        prompt = f"""
You are an AI assistant specialised in answering questions from retrieved context.

Context you receive
• VERIFIED FACTS – text snippets retrieved from the user's documents, each labelled with its source document. Some may be irrelevant noise.
• ORIGINAL QUESTION – the user's actual query.

Instructions
1. Evaluate each snippet for relevance to the ORIGINAL QUESTION; ignore those that do not help answer it.
2. Synthesise an answer **using only information from the relevant snippets**.
3. If snippets contradict one another, mention the contradiction explicitly.
4. If the snippets do not contain the needed information, reply exactly with:
   "I could not find that information in the provided documents."
5. Provide a thorough, well-structured answer. Use paragraphs or bullet points where helpful, and include any relevant numbers/names exactly as they appear. There is **no strict sentence limit**, but aim for clarity over brevity.
6. Do **not** introduce external knowledge unless step 4 applies; in that case you may add a clearly-labelled "General knowledge" sentence after the required statement.
7. **Cite your sources inline.** Each snippet is numbered, e.g. "[Source 3: report.pdf]". After every fact you state, append the number(s) of the snippet(s) it came from in square brackets, e.g. [3] or [1][4]. Never attribute a fact to a snippet it did not come from.

Output format
Answer:
<your answer here, with [N] after each fact>

–––––  Retrieved Snippets  –––––
{facts}
––––––––––––––––––––––––––––––

ORIGINAL QUESTION: "{query}"

Reminder: every fact in your answer must end with its source number in square brackets, e.g. "The budget is 7.3M [2]." — matching the numbered [Source N: ...] labels above. Do not skip this.
"""
        # Stream the answer token-by-token so the caller can forward them as SSE
        answer_parts: list[str] = []
        for tok in self.ollama_client.stream_completion(
            model=generation_model or self.ollama_config["generation_model"],
            prompt=prompt,
        ):
            answer_parts.append(tok)
            if event_callback:
                event_callback("token", {"text": tok})

        return "".join(answer_parts)

    def run(
        self,
        query: str,
        table_name: Optional[str] = None,
        window_size_override: Optional[int] = None,
        event_callback=None,
        collections: Optional[List[Dict[str, Any]]] = None,
        filters: Optional[Dict[str, Any]] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Retrieve, rerank, and synthesize.

        collections: optional list of {"table_name", "embedding_model",
        "index_id", "index_name", "metadata_schema", "fusion_config"} dicts to
        search together (capped at
        MAX_COLLECTIONS). When omitted, the single table_name/config table is
        used. filters: optional typed metadata filters, validated against
        each collection's schema and compiled to a LanceDB where clause —
        collections whose schema can't satisfy the filter are excluded.

        overrides: per-request retrieval knobs applied locally for THIS call
        only (retrieval_k, reranker_top_k, ai_rerank, search_type,
        dense_weight, provence_enabled, provence_threshold, latechunk_enabled,
        generation_model). These used to be
        written into self.config by callers, which forced a global lock to
        stop concurrent requests clobbering each other; resolving them per
        call instead keeps the shared pipeline config immutable.
        """
        start_time = time.time()
        ov = overrides or {}

        def _ov(key, default):
            v = ov.get(key)
            return default if v is None else v

        retrieval_k = _ov("retrieval_k", self.config.get("retrieval_k", 10))

        active_table = table_name or self.storage_config.get("text_table_name")
        logger.debug(
            "--- Running Hybrid Search for query '%s' (table=%s) ---",
            query,
            active_table,
        )

        if event_callback:
            event_callback("retrieval_started", {})
        # Get the LanceDB reranker for initial score fusion
        lancedb_reranker = self._get_reranker()

        # Search mode + dense weight: per-request override wins, else config.
        retrieval_cfg = self.config.get("retrieval", {}) or {}
        search_type = str(
            _ov("search_type", retrieval_cfg.get("search_type") or "hybrid")
        ).lower()
        vector_only_search = search_type in ("vector_only", "vector")
        fts_only_search = search_type in ("bm25", "fts", "keyword")
        fusion_override = None
        dense_weight = _ov(
            "dense_weight", (retrieval_cfg.get("dense") or {}).get("weight")
        )
        if dense_weight is not None:
            try:
                w = max(0.0, min(1.0, float(dense_weight)))
                fusion_override = {"bm25_weight": 1.0 - w, "vec_weight": w}
            except (TypeError, ValueError):
                pass

        # ---------------------------------------------------------------
        # Multi-collection retrieval (NVIDIA blueprint contract): retrieve
        # retrieval_k from EACH collection with that collection's embedding
        # model, then rank globally across collections. Single-collection
        # requests are the degenerate case of the same path.
        # ---------------------------------------------------------------
        if not collections:
            collections = [
                {
                    "table_name": active_table,
                    "embedding_model": self.config.get("embedding_model_name"),
                    "index_name": None,
                }
            ]
        collections = collections[:MAX_COLLECTIONS]
        multi_collection = len(collections) > 1

        lc_cfg = self._resolve_latechunk_cfg()
        lc_enabled = bool(_ov("latechunk_enabled", lc_cfg.get("enabled")))

        from rag_system.metadata_filters import FilterError, compile_filters

        retrieved_docs = []
        filter_errors: List[str] = []
        searched = 0
        for coll in collections:
            coll_table = coll.get("table_name")
            if not coll_table:
                continue

            # Typed metadata filters: validated against this collection's
            # schema and compiled to SQL. A collection whose schema cannot
            # satisfy the filter is EXCLUDED (it can't contain matches) —
            # never searched unfiltered, and never given raw filter input.
            where = None
            if filters:
                try:
                    where = compile_filters(coll.get("metadata_schema"), filters)
                except FilterError as fe:
                    logger.warning(
                        "filters_exclude_collection table=%s error=%s", coll_table, fe
                    )
                    filter_errors.append(str(fe))
                    continue

            retriever = self._get_retriever_for_model(coll.get("embedding_model"))
            if not retriever:
                logger.warning(
                    "collection_skipped_no_compatible_embedder table=%s model=%s",
                    coll_table,
                    coll.get("embedding_model"),
                )
                continue
            collection_fusion = fusion_override or coll.get("fusion_config")
            try:
                docs = retriever.retrieve(
                    text_query=query,
                    table_name=coll_table,
                    k=retrieval_k,
                    reranker=lancedb_reranker,
                    vector_only=vector_only_search,
                    fts_only=fts_only_search,
                    fusion_override=collection_fusion,
                    where=where,
                )
                searched += 1
            except Exception as e:
                logger.warning(
                    "collection_retrieval_failed table=%s error=%s", coll_table, e
                )
                continue

            # Late-chunk leg per collection (vector_only: no FTS index there;
            # skipped for filtered queries — lc tables carry no meta columns)
            if lc_enabled and not where:
                lc_table = lc_cfg.get("lancedb_table_name") or f"{coll_table}_lc"
                if self._lancedb_table_exists(retriever, lc_table):
                    try:
                        docs.extend(
                            retriever.retrieve(
                                text_query=query,
                                table_name=lc_table,
                                k=retrieval_k,
                                vector_only=True,
                            )
                        )
                    except Exception as e:
                        logger.warning(
                            "latechunk_retrieval_failed table=%s error=%s", lc_table, e
                        )

            for rank, d in enumerate(docs, start=1):
                # Source bookkeeping: expansion must fetch neighbors from the
                # doc's own table, and citations should name the index
                d["_source_table"] = coll_table
                d["_collection_rank"] = rank
                if coll.get("index_id"):
                    d["index_id"] = coll["index_id"]
                if coll.get("index_name"):
                    d["index_name"] = coll["index_name"]
            retrieved_docs.extend(docs)

        if filters and searched == 0:
            detail = (
                "; ".join(sorted(set(filter_errors)))
                or "no collection accepts this filter"
            )
            raise FilterError(f"Metadata filter could not be applied: {detail}")

        if multi_collection:
            logger.info(
                "multi_collection_retrieval collections=%s docs=%s",
                [c.get("table_name") for c in collections],
                len(retrieved_docs),
            )

        if event_callback:
            event_callback("retrieval_done", {"count": len(retrieved_docs)})

        retrieval_time = time.time() - start_time
        logger.debug(
            "Retrieved %s chunks in %.2fs", len(retrieved_docs), retrieval_time
        )

        # -----------------------------------------------------------
        #  LATE-CHUNK MERGING (merge ±1 sub-vector into central hit)
        # -----------------------------------------------------------
        if lc_enabled and retrieved_docs:
            merged_count = 0
            for doc in retrieved_docs:
                try:
                    cid = doc.get("chunk_id")
                    meta = doc.get("metadata", {})
                    if meta.get("latechunk_merged"):
                        continue  # already processed
                    doc_id = doc.get("document_id")
                    cidx = doc.get("chunk_index")
                    if doc_id is None or cidx is None or cidx == -1:
                        continue
                    # Fetch neighbouring late-chunks inside same document (±1)
                    siblings = self._get_surrounding_chunks_lancedb(doc, window_size=1)
                    # Keep only same document_id and ordered by chunk_index
                    siblings = [s for s in siblings if s.get("document_id") == doc_id]
                    siblings.sort(key=lambda s: s.get("chunk_index", 0))
                    merged_text = " \n".join(s.get("text", "") for s in siblings)
                    if merged_text:
                        doc["text"] = merged_text
                        meta["latechunk_merged"] = True
                        merged_count += 1
                except Exception as e:
                    logger.warning(
                        "latechunk_merge_failed chunk_id=%s error=%s",
                        doc.get("chunk_id"),
                        e,
                    )
            if merged_count:
                logger.info("latechunk_merge_applied merged_count=%s", merged_count)

        # --- AI Reranking Step ---
        ai_reranker = self._get_ai_reranker(enabled_override=ov.get("ai_rerank"))
        if ai_reranker and retrieved_docs:
            if event_callback:
                event_callback("rerank_started", {"count": len(retrieved_docs)})
            logger.info("ai_reranking_started retrieved_count=%s", len(retrieved_docs))
            start_rerank_time = time.time()

            rerank_cfg = self.config.get("reranker", {})
            top_k_cfg = _ov("reranker_top_k", rerank_cfg.get("top_k"))
            top_percent = rerank_cfg.get("top_percent")  # value in range 0–1

            if top_percent is not None:
                try:
                    pct = float(top_percent)
                    assert 0 < pct <= 1
                    top_k = max(1, int(len(retrieved_docs) * pct))
                except Exception:
                    logger.warning("invalid_top_percent top_percent=%s", top_percent)
                    top_k = top_k_cfg or len(retrieved_docs)
            else:
                top_k = top_k_cfg or len(retrieved_docs)

            strategy = self.config.get("reranker", {}).get("strategy", "qwen")

            if strategy == "rerankers-lib":
                texts = [d["text"] for d in retrieved_docs]
                # ColBERT's Rust backend isn't Sync; serialise calls.
                with _rerank_lock:
                    ranked = ai_reranker.rank(query=query, docs=texts)
                # ranked is RankedResults; convert to list of (score, idx)
                try:
                    pairs = [(r.score, r.document.doc_id) for r in ranked.results]
                    if any(p[1] is None for p in pairs):
                        pairs = [(r.score, i) for i, r in enumerate(ranked.results)]
                except Exception:
                    pairs = ranked
                # Keep only top_k results if requested
                if top_k is not None and len(pairs) > top_k:
                    pairs = pairs[:top_k]
                reranked_docs = [
                    retrieved_docs[idx] | {"rerank_score": score}
                    for score, idx in pairs
                ]
            else:
                try:
                    reranked_docs = ai_reranker.rerank(
                        query, retrieved_docs, top_k=top_k
                    )
                except TypeError:
                    texts = [d["text"] for d in retrieved_docs]
                    pairs = ai_reranker.rank(query, texts, top_k=top_k)
                    reranked_docs = [
                        retrieved_docs[idx] | {"rerank_score": score}
                        for score, idx in pairs
                    ]

            rerank_time = time.time() - start_rerank_time
            logger.info(
                "ai_reranking_completed duration_s=%s result_count=%s",
                round(rerank_time, 2),
                len(reranked_docs),
            )
            if event_callback:
                event_callback("rerank_done", {"count": len(reranked_docs)})
        else:
            # No AI reranker. A single collection keeps its fused retrieval
            # order; across collections the scores are not comparable
            # (different tables/embedding models), so fall back to Reciprocal
            # Rank Fusion over each collection's own ranking.
            if multi_collection:
                reranked_docs = sorted(
                    retrieved_docs,
                    key=lambda d: 1.0 / (60 + d.get("_collection_rank", 10**6)),
                    reverse=True,
                )
            else:
                reranked_docs = retrieved_docs

        # Keep-union rescue: cross-encoder rerankers score flattened table
        # chunks poorly against natural-language questions, so the chunks
        # most likely to hold precise values get cut. Always retain EACH
        # collection's top few by retrieval rank, slotted just below the
        # reranker's picks so they outrank expansion neighbors and survive
        # the context budget. Measured on the eval set: this is where
        # answer-bearing chunks were lost between retrieval (86%) and
        # synthesis (71%).
        keep_n = int(self.config.get("reranker", {}).get("keep_top_retrieval", 3))
        if ai_reranker and keep_n > 0 and reranked_docs is not retrieved_docs:
            have = {(d.get("_source_table"), d.get("chunk_id")) for d in reranked_docs}
            rescued = [
                d
                for d in retrieved_docs
                if d.get("_collection_rank", 10**6) <= keep_n
                and (d.get("_source_table"), d.get("chunk_id")) not in have
            ]
            if rescued:
                floor = min(
                    (d.get("rerank_score", 0.0) for d in reranked_docs), default=0.0
                )
                for j, doc in enumerate(rescued, start=1):
                    doc["rerank_score"] = floor - 0.001 * j
                reranked_docs.extend(rescued)
                logger.info("rerank_keep_union_rescued count=%s", len(rescued))

        window_size = _ov(
            "context_window_size", self.config.get("context_window_size", 1)
        )
        if window_size_override is not None:
            window_size = window_size_override
        if window_size > 0 and reranked_docs:
            if event_callback:
                event_callback("context_expand_started", {"count": len(reranked_docs)})
            logger.info(
                "context_expansion_started top_docs=%s window_size=%s",
                len(reranked_docs),
                window_size,
            )
            expanded_chunks = {}
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_chunk = {
                    executor.submit(
                        self._get_surrounding_chunks_lancedb, chunk, window_size
                    ): chunk
                    for chunk in reranked_docs
                }
                for future in concurrent.futures.as_completed(future_to_chunk):
                    try:
                        seed_chunk = future_to_chunk[future]
                        surrounding_chunks = future.result()
                        for surrounding_chunk in surrounding_chunks:
                            cid = surrounding_chunk["chunk_id"]
                            identity = (seed_chunk.get("_source_table"), cid)
                            if identity not in expanded_chunks:
                                # If this is the *central* chunk we already reranked, carry over its score
                                if (
                                    cid == seed_chunk.get("chunk_id")
                                    and "rerank_score" in seed_chunk
                                ):
                                    surrounding_chunk["rerank_score"] = seed_chunk[
                                        "rerank_score"
                                    ]
                                # Neighbors inherit the seed's collection attribution
                                if seed_chunk.get("_source_table"):
                                    surrounding_chunk.setdefault(
                                        "_source_table", seed_chunk["_source_table"]
                                    )
                                if seed_chunk.get("index_name"):
                                    surrounding_chunk.setdefault(
                                        "index_name", seed_chunk["index_name"]
                                    )
                                if seed_chunk.get("index_id"):
                                    surrounding_chunk.setdefault(
                                        "index_id", seed_chunk["index_id"]
                                    )
                                expanded_chunks[identity] = surrounding_chunk
                    except Exception as e:
                        logger.error(
                            "context_expansion_chunk_failed chunk_id=%s error=%s",
                            seed_chunk.get("chunk_id"),
                            e,
                        )

            final_docs = list(expanded_chunks.values())
            # Sort by reranker score if present, otherwise by fused similarity
            # (the retriever already converts vector distances to similarities,
            # so higher `score` is better for both legs).
            if any("rerank_score" in d for d in final_docs):
                final_docs.sort(key=lambda c: c.get("rerank_score", -1), reverse=True)
            elif any(d.get("score") is not None for d in final_docs):
                final_docs.sort(key=lambda c: c.get("score") or 0, reverse=True)
            else:
                # Fallback to document order
                final_docs.sort(
                    key=lambda c: (c.get("document_id", ""), c.get("chunk_index", 0))
                )

            logger.info("context_expansion_completed unique_chunks=%s", len(final_docs))
            if event_callback:
                event_callback("context_expand_done", {"count": len(final_docs)})
        else:
            final_docs = reranked_docs

        # NOTE: there used to be a filter here dropping every doc without a
        # rerank_score. The reranker only ever returns scored docs and the
        # no-rerank path has no scores at all — so its sole effect was to
        # strip the neighbor chunks that context expansion just fetched,
        # making expansion dead code whenever reranking was on (the default).

        # ------------------------------------------------------------------
        # Sentence-level pruning (Provence)
        # ------------------------------------------------------------------
        prov_cfg = self.config.get("provence", {})
        prov_enabled = _ov("provence_enabled", prov_cfg.get("enabled"))
        if prov_enabled:
            if event_callback:
                event_callback("prune_started", {"count": len(final_docs)})
            thresh = float(_ov("provence_threshold", prov_cfg.get("threshold", 0.1)))
            logger.info("provence_pruning_enabled threshold=%s", thresh)
            pruner = self._get_sentence_pruner()
            final_docs = pruner.prune_documents(query, final_docs, threshold=thresh)
            # Remove any chunks that were fully pruned (empty text)
            final_docs = [d for d in final_docs if d.get("text", "").strip()]
            if event_callback:
                event_callback("prune_done", {"count": len(final_docs)})

        if not final_docs:
            logger.info("no_documents_for_synthesis")
        else:
            logger.info(
                "final_documents_for_synthesis document_count=%s", len(final_docs)
            )
            for i, doc in enumerate(final_docs):
                logger.debug(  # type: ignore[call-arg]
                    "final_document_summary",
                    position=i + 1,
                    chunk_id=doc.get("chunk_id"),
                    score=doc.get("score", "N/A"),
                    rerank_score=doc.get("rerank_score"),
                    text_preview=doc.get("text", "").strip()[:200],
                )

        if not final_docs:
            return {
                "answer": "I could not find an answer in the documents.",
                "source_documents": [],
            }

        # --- Sanitize docs for JSON serialization (no NaN/Inf types) ---
        def _clean_val(v):
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return None
            if isinstance(v, (np.floating,)):
                try:
                    f = float(v)
                    if math.isnan(f) or math.isinf(f):
                        return None
                    return f
                except Exception:
                    return None
            return v

        for doc in final_docs:
            # Remove heavy or internal-only fields before serialising
            # (index_name stays: the UI sources list can show it)
            doc.pop("vector", None)
            doc.pop("_distance", None)
            doc.pop("_source_table", None)
            doc.pop("_collection_rank", None)
            # Clean numeric fields
            for key in ["score", "_distance", "rerank_score"]:
                if key in doc:
                    doc[key] = _clean_val(doc[key])

        # Budget the synthesis context to fit the model's allocated window
        # (see OLLAMA_NUM_CTX): anything beyond it would be SILENTLY truncated
        # from the front of the prompt — the instructions and the top-ranked
        # snippets. final_docs is sorted best-first, so trimming keeps winners.
        from rag_system.utils.ollama_client import NUM_CTX

        max_context_chars = (
            NUM_CTX - 2500
        ) * 4  # ≈4 chars/token, reserve for instructions+answer
        kept: List[Dict[str, Any]] = []
        used = 0
        for doc in final_docs:
            used += len(doc.get("text", "")) + 60
            if used > max_context_chars and kept:
                logger.info(
                    "synthesis_context_trimmed kept=%s of=%s",
                    len(kept),
                    len(final_docs),
                )
                break
            kept.append(doc)
        final_docs = kept

        # Number each snippet and label it with its source so the model can
        # cite inline as [N] — the numbers match the order of the sources
        # list shown in the UI, and blends across similar documents become
        # visible in the answer. With multiple indexes, the index name is
        # part of the label so cross-index blends are visible too.
        def _source_label(doc):
            name = _source_display_name(doc.get("document_id"))
            if doc.get("index_name"):
                return f"{doc['index_name']} — {name}"
            return name

        context = "\n\n".join(
            f"[Source {i}: {_source_label(doc)}]\n{doc['text']}"
            for i, doc in enumerate(final_docs, start=1)
        )

        # 👀 DEBUG: Show the exact context passed to the LLM after pruning
        logger.debug(  # type: ignore[call-arg]
            "context_passed_to_llm",
            length=len(context),
            preview=(
                (context[:2000] + f"…\n[truncated] (total {len(context)} chars)")
                if len(context) > 2000
                else context
            ),
        )

        final_answer = self._synthesize_final_answer(
            query,
            context,
            event_callback=event_callback,
            generation_model=ov.get("generation_model"),
        )

        return {"answer": final_answer, "source_documents": final_docs}

    # ------------------------------------------------------------------
    # Public utility
    # ------------------------------------------------------------------
    def list_document_titles(self, max_items: int = 25) -> List[str]:
        """Return up to *max_items* distinct document titles (or IDs).

        This is used only for prompt-routing, so we favour robustness over
        perfect recall. If anything goes wrong we return an empty list so
        the caller can degrade gracefully.
        """
        try:
            tbl_name = self.storage_config.get("text_table_name")
            if not tbl_name:
                return []

            tbl = self._get_db_manager().get_table(tbl_name)

            field_name = (
                "document_title"
                if "document_title" in tbl.schema.names
                else "document_id"
            )

            # Use a cheap SQL filter to grab distinct values; fall back to a
            # simple scan if the driver lacks DISTINCT support.
            try:
                sql = f"SELECT DISTINCT {field_name} FROM tbl LIMIT {max_items}"
                rows = tbl.search().where("true").sql(sql).to_list()  # type: ignore
                titles = [r[field_name] for r in rows if r.get(field_name)]
            except Exception:
                # Fallback: scan first N rows
                rows = tbl.search().select(field_name).limit(max_items * 4).to_list()
                seen = set()
                titles = []
                for r in rows:
                    val = r.get(field_name)
                    if val and val not in seen:
                        titles.append(val)
                        seen.add(val)
                        if len(titles) >= max_items:
                            break

            # Ensure we don't exceed max_items
            return titles[:max_items]
        except Exception:
            # Any issues (missing table, bad schema, etc.) –> just return []
            return []

    # -------------------- Public helper properties --------------------
    @property
    def retriever(self):
        """Lazily exposes the main (dense) retriever so external components
        like the ReAct agent tools can call `.retrieve()` directly without
        reaching into private helpers. If the retriever has not yet been
        instantiated, it is created on first access via `_get_dense_retriever`."""
        return self._get_dense_retriever()

    def update_embedding_model(self, model_name: str):
        """Switch embedding model at runtime and clear cached objects so they re-initialize."""
        if self.config.get("embedding_model_name") == model_name:
            return  # nothing to do
        logger.info(  # type: ignore[call-arg]
            "embedding_model_switch",
            new_model=model_name,
            previous_model=self.config.get("embedding_model_name"),
        )
        self.config["embedding_model_name"] = model_name
        # Reset caches so new instances are built on demand
        self.text_embedder = None
        self.dense_retriever = None

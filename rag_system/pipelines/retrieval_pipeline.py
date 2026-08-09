from typing import List, Dict, Any, Optional
import concurrent.futures
import time
import json
import logging
import math
import os
import numpy as np
from threading import Lock

from rag_system.utils.ollama_client import OllamaClient
from rag_system.retrieval.retrievers import MultiVectorRetriever
from rag_system.indexing.representations import default_query_instruction, select_embedder
from rag_system.indexing.embedders import LanceDBManager
from rag_system.rerankers.reranker import CrossEncoderReranker, QwenRerankerScorer, is_qwen3_reranker
from rag_system.rerankers.sentence_pruner import SentencePruner

# ---------------------------------------------------------------------------
# Thread-safety helpers
# ---------------------------------------------------------------------------

# 1. The `rerankers` lib backends are not thread-safe.  We protect the actual
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

class RetrievalPipeline:
    """
    Orchestrates retrieval, reranking, context expansion, pruning and synthesis.
    """
    def __init__(self, config: Dict[str, Any], ollama_client: OllamaClient, ollama_config: Dict[str, Any]):
        self.config = config
        self.ollama_config = ollama_config
        self.ollama_client = ollama_client

        self.storage_config = self.config["storage"]

        # Defer initialization to just-in-time methods
        self.db_manager = None
        self.text_embedder = None
        self.dense_retriever = None
        self.ai_reranker = None

    def _retriever_config(self, name: str, *aliases: str) -> Dict[str, Any]:
        """Look a retriever sub-config up in "retrievers" then "retrieval".

        Resolved on every access instead of cached in ``__init__`` so runtime
        overrides written by the API land in the pipeline.
        """
        for container_key in ("retrievers", "retrieval"):
            container = self.config.get(container_key) or {}
            for key in (name, *aliases):
                if container.get(key):
                    return container[key]
        return {}

    def _retrieval_mode(self) -> str:
        """Query-time search mode: "hybrid" (default), "vector_only" or "fts_only"."""
        retrieval_cfg = self.config.get("retrieval") or {}
        return retrieval_cfg.get("search_type") or self.config.get("search_type") or "hybrid"

    def _retry_config(self) -> Dict[str, Any]:
        """The evidence-sufficiency retry block (roadmap item 2.1).

        Merged across both container spellings so a runtime override written by
        the API under ``retrievers.retry`` beats the profile's
        ``retrieval.retry``, matching how ``_latechunk_config`` behaves.
        """
        merged: Dict[str, Any] = {}
        for container_key in ("retrieval", "retrievers"):
            block = (self.config.get(container_key) or {}).get("retry")
            if isinstance(block, dict):
                merged.update(block)
        return merged

    def _latechunk_config(self) -> Dict[str, Any]:
        """Merge the late-chunk block across both container and key spellings.

        The profile declares it under ``retrieval.late_chunking`` while the API
        toggles it at runtime under ``retrievers.latechunk``; later writes win.
        """
        merged: Dict[str, Any] = {}
        for container_key in ("retrieval", "retrievers"):
            container = self.config.get(container_key) or {}
            for key in ("late_chunking", "latechunk"):
                block = container.get(key)
                if isinstance(block, dict):
                    merged.update(block)
        return merged

    @staticmethod
    def _latechunk_table_name(latechunk_cfg: Dict[str, Any], base_table: str) -> Optional[str]:
        explicit = latechunk_cfg.get("lancedb_table_name")
        if explicit:
            return explicit
        # "_lc" is the suffix IndexingPipeline writes when none is configured.
        suffix = latechunk_cfg.get("table_suffix", "_lc")
        return f"{base_table}{suffix}" if suffix and base_table else None

    def _get_db_manager(self):
        if self.db_manager is None:
            # Accept either "db_path" (preferred) or legacy "lancedb_uri"
            db_path = self.storage_config.get("db_path") or self.storage_config.get("lancedb_uri")
            if not db_path:
                raise ValueError("Storage config must contain 'db_path' or 'lancedb_uri'.")
            self.db_manager = LanceDBManager(db_path=db_path)
        return self.db_manager

    def _query_instruction(self, model_name: str) -> str:
        """The query-side instruction prefix for this pipeline's embedder.

        Resolution order, most explicit first:

        1. ``config["embedding_instruction"]`` — set it to ``""`` to switch the
           prefix off for a model whose family would otherwise get one.
        2. ``EMBEDDING_INSTRUCTION`` env var — same semantics, for A/B runs.
        3. The model family's official retrieval instruction (Qwen3-Embedding
           and harrier-oss-v1), or ``""`` for everything else.

        This is the QUERY side only. Documents are embedded by
        ``IndexingPipeline``, which calls ``select_embedder`` without an
        instruction, so an index built before this existed remains valid.
        """
        configured = self.config.get("embedding_instruction")
        if configured is not None:
            return configured
        env = os.getenv("EMBEDDING_INSTRUCTION")
        if env is not None:
            return env
        return default_query_instruction(model_name)

    def _get_text_embedder(self):
        if self.text_embedder is None:
            model_name = self.config.get("embedding_model_name")
            if not model_name:
                raise ValueError(
                    "Config must contain 'embedding_model_name'. Falling back to a hard-coded "
                    "default here would silently produce vectors whose dimensionality does not "
                    "match the index."
                )
            instruction = self._query_instruction(model_name)
            if instruction:
                print(f"🔧 Query-side embedding instruction active: '{instruction}'")
            self.text_embedder = select_embedder(
                model_name,
                self.ollama_config.get("host") if isinstance(self.ollama_config, dict) else None,
                query_instruction=instruction,
            )
        return self.text_embedder

    def _get_dense_retriever(self):
        """Ensure a MultiVectorRetriever is always available unless explicitly disabled."""
        if self.dense_retriever is None:
            # If the config explicitly sets dense.enabled to False, respect it
            if self._retriever_config("dense").get("enabled", True) is False:
                return None

            try:
                self.dense_retriever = MultiVectorRetriever(
                    self._get_db_manager(),
                    self._get_text_embedder(),
                )
            except Exception as e:
                print(f"❌ Failed to initialise dense retriever: {e}")
                self.dense_retriever = None
        return self.dense_retriever

    def _get_ai_reranker(self):
        """Initializes a dedicated AI-based reranker."""
        reranker_config = self.config.get("reranker", {})
        if self.ai_reranker is None and reranker_config.get("enabled"):
            # Serialise first-time initialisation so only one thread attempts
            # to load the (very large) model.  Other threads will wait and use
            # the instance once ready, preventing the meta-tensor crash.
            with _ai_reranker_init_lock:
                # Another thread may have completed init while we waited
                if self.ai_reranker is None:
                    model_name = reranker_config.get("model_name")
                    if not model_name:
                        print("⚠️  Reranking is enabled but 'reranker.model_name' is not configured; skipping reranking.")
                        return None
                    try:
                        strategy = reranker_config.get("strategy", "rerankers-lib")
                        model_type = reranker_config.get("model_type", "cross-encoder")

                        # Qwen3-Reranker is a causal-LM yes/no-logit scorer, not a
                        # SequenceClassification model. The rerankers lib silently
                        # loads it with a randomly-initialised score head, so route
                        # the whole family to our own scorer — either by explicit
                        # `reranker.model_type: "qwen3"` or by model name.
                        if model_type == "qwen3" or is_qwen3_reranker(model_name):
                            print(f"🔧 Initialising Qwen3 yes/no-logit reranker ({model_name})…")
                            self.ai_reranker = QwenRerankerScorer(model_name=model_name)
                        elif strategy == "rerankers-lib":
                            print(f"🔧 Initialising {model_type} reranker ({model_name}) via rerankers lib…")
                            from rerankers import Reranker
                            self.ai_reranker = Reranker(model_name, model_type=model_type)
                        else:
                            print(f"🔧 Lazily initializing local cross-encoder reranker ({model_name})…")
                            self.ai_reranker = CrossEncoderReranker(model_name=model_name)

                        print("✅ AI reranker initialized successfully.")
                    except Exception as e:
                        # Leave as None so the pipeline can proceed without reranking
                        print(f"⚠️  Could not load reranker '{model_name}' ({e}). Continuing without reranking.")
        return self.ai_reranker

    def _get_sentence_pruner(self):
        if getattr(self, "_sentence_pruner", None) is None:
            with _sentence_pruner_lock:
                if getattr(self, "_sentence_pruner", None) is None:
                    self._sentence_pruner = SentencePruner()
        return self._sentence_pruner

    # ------------------------------------------------------------------
    # Evidence-sufficiency retry (roadmap item 2.1)
    # ------------------------------------------------------------------

    # Candidates from this rank onwards are treated as "background": chunks the
    # query pulled in because they are documents in this corpus, not because
    # they answer it. Rank 6 of 20 leaves the plausible answers out of the
    # background estimate while still averaging over enough rows to be stable.
    _EVIDENCE_BACKGROUND_FROM = 5

    @classmethod
    def _dense_evidence_score(cls, docs: List[Dict[str, Any]]) -> Optional[float]:
        """A 0–1 "did we actually find something" score from the dense leg.

        The naive choice — the raw top cosine similarity — was **measured and
        rejected**: on the gold set it is *anti*-correlated with success,
        because absolute similarity mostly encodes how close the query's
        phrasing sits to the corpus's register, not whether the answer-bearing
        chunk was found. The three `mixed` first-stage misses all scored a
        *higher* top cosine than the median successful query.

        What does carry signal is **contrast**: how far the best candidate
        stands above the background of everything else the query pulled in.

            score = (cos_top − cos_background) / (1 − cos_background)

        where ``cos_background`` is the mean cosine of the candidates from rank
        ``_EVIDENCE_BACKGROUND_FROM`` down. The denominator rescales against the
        headroom that is actually reachable for this query, keeping the result
        in 0–1 and comparable across queries whose background level differs.

        Returns ``None`` when the dense leg did not run (``fts_only``) or the
        table predates cosine normalization, in which case the caller must not
        retry — the number would not mean anything. Requires L2-normalized
        vectors (v4+ tables), where LanceDB's squared-L2 ``_distance`` maps to
        cosine as ``cos = 1 − d/2``.
        """
        sims = []
        for doc in docs:
            distance = doc.get("_distance")
            if distance is None:
                continue
            try:
                sims.append(1.0 - float(distance) / 2.0)
            except (TypeError, ValueError):
                continue
        if len(sims) < 2:
            return None
        sims.sort(reverse=True)
        tail = sims[cls._EVIDENCE_BACKGROUND_FROM:] or sims[1:]
        background = sum(tail) / len(tail)
        headroom = 1.0 - background
        if headroom <= 1e-6:
            return None
        return max(0.0, min(1.0, (sims[0] - background) / headroom))

    @staticmethod
    def _rerank_evidence_score(docs: List[Dict[str, Any]]) -> Optional[float]:
        """Top reranker score, when the reranker produces a calibrated 0–1 one.

        ``QwenRerankerScorer`` returns P("yes") per candidate, which is directly
        interpretable. Other backends return arbitrary logits, so anything
        outside 0–1 is rejected rather than silently compared to a probability
        threshold.
        """
        scores = [d.get("rerank_score") for d in docs if d.get("rerank_score") is not None]
        if not scores:
            return None
        try:
            top = float(max(scores))
        except (TypeError, ValueError):
            return None
        if not (0.0 <= top <= 1.0):
            return None
        return top

    def _reformulate_query(self, query: str) -> Optional[str]:
        """One rewrite of a query whose first pass found weak evidence.

        Runs on the enrichment (utility) model, not the generation model, and
        asks for JSON so the "thinking" preamble small models emit cannot leak
        into the rewritten query.
        """
        model = (self.ollama_config.get("enrichment_model")
                 or self.ollama_config.get("generation_model"))
        if not model:
            return None
        prompt = (
            "A document search for the question below returned weak matches.\n"
            "Rewrite it once as a single self-contained search query that uses the "
            "concrete nouns, technical terms and synonyms a document would actually "
            "use, instead of the asker's phrasing. Keep every entity, number and "
            "constraint from the original. Do not answer the question.\n\n"
            f'Question: "{query}"\n\n'
            'Respond with JSON: {"query": "<rewritten query>"}'
        )
        try:
            resp = self.ollama_client.generate_completion(model=model, prompt=prompt, format="json")
            data = json.loads(resp.get("response", "{}"))
        except Exception as e:
            print(f"⚠️  Retry reformulation failed ({e}); keeping the original query.")
            return None
        rewritten = (data.get("query") or "").strip() if isinstance(data, dict) else ""
        if not rewritten or rewritten.lower() == query.strip().lower():
            return None
        return rewritten

    def _get_surrounding_chunks_lancedb(self, chunk: Dict[str, Any], window_size: int) -> List[Dict[str, Any]]:
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

        table_name = self.config["storage"]["text_table_name"]
        try:
            tbl = db_manager.get_table(table_name)
        except Exception:
            # If the table can't be opened, we can't get surrounding chunks
            return [chunk]

        # Define the window for the search
        start_index = max(0, chunk_index - window_size)
        end_index = chunk_index + window_size
        
        # Construct the SQL filter for an efficient metadata-based search
        sql_filter = f"document_id = '{document_id}' AND chunk_index >= {start_index} AND chunk_index <= {end_index}"
        
        try:
            # Execute a filter-only search, which is very fast on indexed metadata
            results = tbl.search().where(sql_filter).to_list()
            
            # The results must be sorted by chunk_index to maintain logical order
            results.sort(key=lambda c: c['chunk_index'])

            # The 'metadata' field is a JSON string and needs to be parsed
            for res in results:
                if isinstance(res.get('metadata'), str):
                    try:
                        res['metadata'] = json.loads(res['metadata'])
                    except json.JSONDecodeError:
                        res['metadata'] = {} # Handle corrupted metadata gracefully
            return results
        except Exception:
            # If the query fails for any reason, fall back to the single chunk
            return [chunk]

    def _synthesize_final_answer(self, query: str, facts: str, *, event_callback=None) -> str:
        """Uses a text LLM to synthesize a final answer from extracted facts."""
        prompt = f"""
You are an AI assistant specialised in answering questions from retrieved context.

Context you receive
• VERIFIED FACTS – text snippets retrieved from the user's documents. Some may be irrelevant noise.  
• ORIGINAL QUESTION – the user's actual query.

Instructions
1. Evaluate each snippet for relevance to the ORIGINAL QUESTION; ignore those that do not help answer it.  
2. Synthesise an answer **using only information from the relevant snippets**.  
3. If snippets contradict one another, mention the contradiction explicitly.  
4. If the snippets do not contain the needed information, reply exactly with:  
   "I could not find that information in the provided documents."  
5. Provide a thorough, well-structured answer. Use paragraphs or bullet points where helpful, and include any relevant numbers/names exactly as they appear. There is **no strict sentence limit**, but aim for clarity over brevity.  
6. Do **not** introduce external knowledge unless step 4 applies; in that case you may add a clearly-labelled "General knowledge" sentence after the required statement.

Output format
Answer:
<your answer here>

–––––  Retrieved Snippets  –––––
{facts}
––––––––––––––––––––––––––––––

ORIGINAL QUESTION: "{query}"
"""
        # Stream the answer token-by-token so the caller can forward them as SSE
        answer_parts: list[str] = []
        for tok in self.ollama_client.stream_completion(
            model=self.ollama_config["generation_model"],
            prompt=prompt,
        ):
            answer_parts.append(tok)
            if event_callback:
                event_callback("token", {"text": tok})

        return "".join(answer_parts)

    def _first_stage(self, query: str, base_table: str, retrieval_k: int, retrieval_mode: str,
                     event_callback=None) -> List[Dict[str, Any]]:
        """Hybrid/vector/FTS retrieval plus the optional late-chunk table and merge.

        Split out of ``run()`` so the evidence-sufficiency retry can call it a
        second time with a reformulated query without duplicating any of it.
        """
        start_time = time.time()
        logger = logging.getLogger(__name__)
        dense_retriever = self._get_dense_retriever()

        retrieved_docs = []
        if dense_retriever:
            retrieved_docs = dense_retriever.retrieve(
                text_query=query,
                table_name=base_table,
                k=retrieval_k,
                search_type=retrieval_mode,
            )

        # ---------------------------------------------------------------
        # Late-Chunk retrieval (optional)
        # ---------------------------------------------------------------
        latechunk_cfg = self._latechunk_config()
        if dense_retriever and latechunk_cfg.get("enabled"):
            lc_table = self._latechunk_table_name(latechunk_cfg, base_table)
            if lc_table:
                try:
                    lc_docs = dense_retriever.retrieve(
                        text_query=query,
                        table_name=lc_table,
                        k=retrieval_k,
                        search_type=retrieval_mode,
                    )
                    retrieved_docs.extend(lc_docs)
                except Exception as e:
                    print(f"⚠️  Late-chunk retrieval failed: {e}")

        if event_callback:
            event_callback("retrieval_done", {"count": len(retrieved_docs)})

        logger.debug("Retrieved %s chunks in %.2fs", len(retrieved_docs), time.time() - start_time)

        # -----------------------------------------------------------
        #  LATE-CHUNK MERGING (merge ±1 sub-vector into central hit)
        # -----------------------------------------------------------
        if latechunk_cfg.get("enabled") and retrieved_docs:
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
                    print(f"⚠️  Late-chunk merge failed for chunk {doc.get('chunk_id')}: {e}")
            if merged_count:
                print(f"🪄 Late-chunk merging applied to {merged_count} retrieved chunks.")

        return retrieved_docs

    # ------------------------------------------------------------------
    # Reranking (roadmap item 2.2: decomposition applies HERE, not first stage)
    # ------------------------------------------------------------------

    @staticmethod
    def _score_pairs(ai_reranker, strategy: str, query: str, texts: List[str]) -> Dict[int, float]:
        """Score every candidate against one query. Returns {candidate index: score}."""
        # Some rerankers-lib backends are not thread-safe; serialise calls.
        with _rerank_lock:
            if strategy == "rerankers-lib":
                ranked = ai_reranker.rank(query=query, docs=texts)
                try:
                    pairs = [(r.score, r.document.doc_id) for r in ranked.results]
                    if any(not isinstance(p[1], int) for p in pairs):
                        pairs = [(r.score, i) for i, r in enumerate(ranked.results)]
                except Exception:
                    pairs = ranked
            else:
                pairs = ai_reranker.rank(query, texts)
        return {int(idx): float(score) for score, idx in pairs}

    def _rerank_stage(self, query: str, retrieved_docs: List[Dict[str, Any]],
                      sub_queries: Optional[List[str]] = None,
                      event_callback=None) -> List[Dict[str, Any]]:
        """Reorder the first-stage candidates. No-op when reranking is off.

        When *sub_queries* is supplied (query decomposition is on **and** the
        reranker is on), each candidate is scored against every sub-query and
        the per-sub-query scores are aggregated with
        ``query_decomposition.rerank_aggregate`` (``"mean"``, the default, or
        ``"max"``). This is the whole of roadmap item 2.2: the first stage
        always ran on the full original query, because decomposing *there*
        dilutes the query semantically, while decomposition applied at reranking
        is where the 2026 evidence puts the win.
        """
        ai_reranker = self._get_ai_reranker()
        if not ai_reranker or not retrieved_docs:
            return retrieved_docs

        if event_callback:
            event_callback("rerank_started", {"count": len(retrieved_docs)})
        print(f"\n--- Reranking top {len(retrieved_docs)} docs with AI model... ---")
        start_rerank_time = time.time()

        rerank_cfg = self.config.get("reranker", {})
        top_k_cfg = rerank_cfg.get("top_k")
        top_percent = rerank_cfg.get("top_percent")  # value in range 0–1

        if top_percent is not None:
            try:
                pct = float(top_percent)
                assert 0 < pct <= 1
                top_k = max(1, int(len(retrieved_docs) * pct))
            except Exception:
                print("⚠️  Invalid top_percent value; falling back to top_k")
                top_k = top_k_cfg or len(retrieved_docs)
        else:
            top_k = top_k_cfg or len(retrieved_docs)

        strategy = rerank_cfg.get("strategy", "rerankers-lib")
        texts = [d["text"] for d in retrieved_docs]

        queries = [q for q in (sub_queries or []) if q and q.strip()] or [query]
        # "mean" is the default because it measured better than "max" on both
        # subsets of the item-2.2 A/B (eval/decisions/phase2-pipeline.md §3).
        aggregate = (self.config.get("query_decomposition", {}) or {}).get(
            "rerank_aggregate", "mean")
        if len(queries) > 1:
            print(f"🔀 Scoring candidates against {len(queries)} sub-queries "
                  f"(aggregate={aggregate}).")

        per_query = [self._score_pairs(ai_reranker, strategy, q, texts) for q in queries]

        scores: Dict[int, float] = {}
        for idx in range(len(texts)):
            values = [m[idx] for m in per_query if idx in m]
            if not values:
                continue
            scores[idx] = (sum(values) / len(values)) if aggregate == "mean" else max(values)

        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        if top_k is not None and len(ordered) > top_k:
            ordered = ordered[:top_k]
        reranked_docs = [retrieved_docs[idx] | {"rerank_score": score} for idx, score in ordered]

        rerank_time = time.time() - start_rerank_time
        print(f"✅ Reranking completed in {rerank_time:.2f}s. Refined to {len(reranked_docs)} docs.")
        if event_callback:
            event_callback("rerank_done", {"count": len(reranked_docs)})
        return reranked_docs

    # ------------------------------------------------------------------

    def retrieve_candidates(self, query: str, table_name: Optional[str] = None,
                            sub_queries: Optional[List[str]] = None,
                            event_callback=None) -> Dict[str, Any]:
        """First stage + rerank + the evidence-sufficiency retry around both.

        This is the whole candidate-selection path, factored out of ``run()`` so
        that ``eval/run_eval.py`` measures exactly what ships rather than a
        reimplementation of it.

        Returns::

            {"first_stage": [...],   # post-retry first-stage ordering
             "documents":   [...],   # after reranking, or == first_stage
             "query_used":  str,
             "retry":       {...} | None}
        """
        retrieval_k = self.config.get("retrieval_k", 10)
        retrieval_mode = self._retrieval_mode()
        base_table = table_name or self.storage_config["text_table_name"]

        if event_callback:
            event_callback("retrieval_started", {"mode": retrieval_mode})

        first_stage = self._first_stage(query, base_table, retrieval_k, retrieval_mode,
                                        event_callback)
        documents = self._rerank_stage(query, first_stage, sub_queries, event_callback)

        result = {"first_stage": first_stage, "documents": documents,
                  "query_used": query, "retry": None}

        retry_cfg = self._retry_config()
        if not retry_cfg.get("enabled") or not first_stage:
            return result

        # Prefer the reranker's calibrated probability when it produced one;
        # otherwise fall back to the dense contrast score. RRF ranks are
        # deliberately never used — they carry no absolute information.
        score = self._rerank_evidence_score(documents)
        signal = "rerank"
        threshold = retry_cfg.get("min_rerank_score", retry_cfg.get("min_top_score"))
        if score is None:
            score = self._dense_evidence_score(first_stage)
            signal = "dense_contrast"
            threshold = retry_cfg.get("min_top_score")
        if score is None or threshold is None:
            # fts_only, or a legacy unnormalized table: no meaningful signal.
            return result

        if score >= float(threshold):
            return result

        max_attempts = int(retry_cfg.get("max_attempts", 1) or 0)
        if max_attempts < 1:
            return result

        print(f"\n🔁 Evidence-sufficiency retry: {signal} score {score:.3f} "
              f"< {float(threshold):.3f} — reformulating once.")
        reformulated = self._reformulate_query(query)
        info = {"signal": signal, "threshold": float(threshold),
                "score_before": round(score, 4), "reformulated": reformulated,
                "attempted": reformulated is not None, "kept": "original",
                "score_after": None}

        if reformulated:
            retry_first = self._first_stage(reformulated, base_table, retrieval_k,
                                            retrieval_mode, event_callback)
            retry_docs = self._rerank_stage(reformulated, retry_first, sub_queries,
                                            event_callback)
            retry_score = (self._rerank_evidence_score(retry_docs) if signal == "rerank"
                           else self._dense_evidence_score(retry_first))
            info["score_after"] = None if retry_score is None else round(retry_score, 4)
            # Keep whichever attempt scored better on the same signal. A retry
            # that did not improve the evidence is discarded, not merged.
            if retry_score is not None and retry_score > score:
                result["first_stage"] = retry_first
                result["documents"] = retry_docs
                result["query_used"] = reformulated
                info["kept"] = "retry"

        result["retry"] = info
        if event_callback:
            event_callback("retrieval_retry", info)
        print(f"🔁 Retry kept the {info['kept']} result set "
              f"(score_after={info['score_after']}).")
        return result

    def run(self, query: str, table_name: str = None, window_size_override: Optional[int] = None,
            event_callback=None, sub_queries: Optional[List[str]] = None) -> Dict[str, Any]:
        base_table = table_name or self.storage_config["text_table_name"]

        logger = logging.getLogger(__name__)
        logger.debug("--- Running search for query '%s' (table=%s) ---", query, base_table)

        # If a custom table_name is provided, propagate it to storage config so helper methods use it
        if table_name:
            self.storage_config["text_table_name"] = table_name

        candidates = self.retrieve_candidates(query, base_table, sub_queries, event_callback)
        reranked_docs = candidates["documents"]

        window_size = self.config.get("context_window_size", 1)
        if window_size_override is not None:
            window_size = window_size_override
        if window_size > 0 and reranked_docs:
            if event_callback:
                event_callback("context_expand_started", {"count": len(reranked_docs)})
            print(f"\n--- Expanding context for {len(reranked_docs)} top documents (window size: {window_size})... ---")
            expanded_chunks = {}
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_chunk = {executor.submit(self._get_surrounding_chunks_lancedb, chunk, window_size): chunk for chunk in reranked_docs}
                for future in concurrent.futures.as_completed(future_to_chunk):
                    try:
                        seed_chunk = future_to_chunk[future]
                        surrounding_chunks = future.result()
                        for surrounding_chunk in surrounding_chunks:
                            cid = surrounding_chunk['chunk_id']
                            if cid not in expanded_chunks:
                                # If this is the *central* chunk we already reranked, carry over its score
                                if cid == seed_chunk.get('chunk_id') and 'rerank_score' in seed_chunk:
                                    surrounding_chunk['rerank_score'] = seed_chunk['rerank_score']
                                expanded_chunks[cid] = surrounding_chunk
                    except Exception as e:
                        print(f"Error expanding context for a chunk: {e}")

            final_docs = list(expanded_chunks.values())
            # Sort by reranker score if present, otherwise by raw score/distance
            if any('rerank_score' in d for d in final_docs):
                final_docs.sort(key=lambda c: c.get('rerank_score', -1), reverse=True)
            elif any('_distance' in d for d in final_docs):
                # For vector search smaller distance is better
                final_docs.sort(key=lambda c: c.get('_distance', 1e9))
            elif any('score' in d for d in final_docs):
                final_docs.sort(key=lambda c: c.get('score', 0), reverse=True)
            else:
                # Fallback to document order
                final_docs.sort(key=lambda c: (c.get('document_id', ''), c.get('chunk_index', 0)))

            print(f"Expanded to {len(final_docs)} unique chunks for synthesis.")
            if event_callback:
                event_callback("context_expand_done", {"count": len(final_docs)})
        else:
            final_docs = reranked_docs

        # Optionally hide non-reranked chunks: if any chunk carries a
        # `rerank_score`, we assume the caller wants to focus on those.
        if any('rerank_score' in d for d in final_docs):
            final_docs = [d for d in final_docs if 'rerank_score' in d]

        # ------------------------------------------------------------------
        # Sentence-level pruning (Provence)
        # ------------------------------------------------------------------
        prov_cfg = self.config.get("provence", {})
        if prov_cfg.get("enabled"):
            if event_callback:
                event_callback("prune_started", {"count": len(final_docs)})
            thresh = float(prov_cfg.get("threshold", 0.1))
            print(f"\n--- Provence pruning enabled (threshold={thresh}) ---")
            pruner = self._get_sentence_pruner()
            final_docs = pruner.prune_documents(query, final_docs, threshold=thresh)
            # Remove any chunks that were fully pruned (empty text)
            final_docs = [d for d in final_docs if d.get('text', '').strip()]
            if event_callback:
                event_callback("prune_done", {"count": len(final_docs)})

        print("\n--- Final Documents for Synthesis ---")
        if not final_docs:
            print("No documents to synthesize.")
        else:
            for i, doc in enumerate(final_docs):
                print(f"  [{i+1}] Chunk ID: {doc.get('chunk_id')}")
                print(f"      Score: {doc.get('score', 'N/A')}")
                if 'rerank_score' in doc:
                    print(f"      Rerank Score: {doc.get('rerank_score'):.4f}")
                print(f"      Text: \"{doc.get('text', '').strip()}\"")
        print("------------------------------------")

        if not final_docs:
            return {"answer": "I could not find an answer in the documents.", "source_documents": []}
        
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
            doc.pop("vector", None)
            doc.pop("_distance", None)
            # Clean numeric fields
            for key in ['score', '_distance', 'rerank_score']:
                if key in doc:
                    doc[key] = _clean_val(doc[key])

        context = "\n\n".join([doc['text'] for doc in final_docs])

        # 👀 DEBUG: Show the exact context passed to the LLM after pruning
        print("\n=== Context passed to LLM (post-pruning) ===")
        if len(context) > 2000:
            print(context[:2000] + "…\n[truncated] (total {} chars)".format(len(context)))
        else:
            print(context)
        print("=== End of context ===\n")

        final_answer = self._synthesize_final_answer(query, context, event_callback=event_callback)
        
        return {"answer": final_answer, "source_documents": final_docs}

    # -------------------- Public helper properties --------------------
    @property
    def retriever(self):
        """Lazily exposes the MultiVectorRetriever so external components can
        call `.retrieve()` directly without reaching into private helpers. If
        the retriever has not yet been instantiated, it is created on first
        access via `_get_dense_retriever`."""
        return self._get_dense_retriever()

    def update_embedding_model(self, model_name: str):
        """Switch embedding model at runtime and clear cached objects so they re-initialize."""
        if self.config.get("embedding_model_name") == model_name:
            return  # nothing to do
        print(f"🔧 RetrievalPipeline switching embedding model to '{model_name}' (was '{self.config.get('embedding_model_name')}')")
        self.config["embedding_model_name"] = model_name
        # Reset caches so new instances are built on demand
        self.text_embedder = None
        self.dense_retriever = None
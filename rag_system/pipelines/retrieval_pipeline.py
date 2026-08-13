from typing import List, Dict, Any, Optional
import concurrent.futures
import contextlib
import threading
import time
import json
import logging
import math
import os
import numpy as np
from threading import Lock

from rag_system.utils.ollama_client import OllamaClient
from rag_system.retrieval.filters import CompiledFilter, combine, compile_filters
from rag_system.retrieval.retrievers import MultiVectorRetriever
from rag_system.indexing.representations import default_query_instruction, select_embedder
from rag_system.indexing.embedders import (
    EmbedderMismatchError,
    LanceDBManager,
    assert_embedder_matches,
    l2_normalize,
    read_table_marker,
)
from rag_system.indexing.overview_builder import load_overview_vectors, overview_vectors_path
from rag_system.rerankers.reranker import CrossEncoderReranker, QwenRerankerScorer, is_qwen3_reranker
from rag_system.rerankers.sentence_pruner import SentencePruner

# Reciprocal-rank-fusion constant, kept identical to the retriever's so a fused
# ordering produced here is comparable with one produced there.
_RRF_K = 60

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

        # Overview prefilter (roadmap item 4.3): the sidecar is read once per
        # pipeline and its absence is reported once, not per query.
        self._overview_vectors_loaded = False
        self._overview_vectors_data = None

        # Metadata filter (roadmap item 4.4). Thread-local, because the agent
        # runs sub-queries of one user question through this same pipeline
        # object in parallel and one sub-query's filter must not leak into
        # another's search. Empty unless a caller opened a filter scope.
        self._filter_local = threading.local()

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

    # ------------------------------------------------------------------
    # Metadata filter scope (roadmap item 4.4)
    # ------------------------------------------------------------------

    def active_filter(self) -> Optional[CompiledFilter]:
        """The compiled filter in force on *this thread*, or None."""
        return getattr(self._filter_local, "compiled", None)

    @contextlib.contextmanager
    def filter_scope(self, compiled: Optional[CompiledFilter]):
        """Make *compiled* the active filter for the duration of the block.

        ``run()`` opens this scope rather than threading a ``filters`` argument
        down through every internal call, for one concrete reason:
        ``EscalatingRetrievalPipeline`` (roadmap 4.1) overrides
        ``retrieve_candidates`` with a fixed four-argument signature and calls
        ``super()`` positionally. Adding a parameter that ``run()`` had to pass
        would break that subclass. A thread-local scope is invisible to it, and
        the agent's parallel sub-query fan-out enters ``run()`` **inside** each
        worker thread, so each worker sets its own.
        """
        previous = getattr(self._filter_local, "compiled", None)
        self._filter_local.compiled = compiled
        try:
            yield compiled
        finally:
            self._filter_local.compiled = previous

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

    def _merged_block(self, key: str) -> Dict[str, Any]:
        """A config block merged across the ``retrieval`` / ``retrievers`` spellings.

        Same layering as ``_retry_config``: a runtime override written by the API
        under ``retrievers.<key>`` beats the profile's ``retrieval.<key>``.
        """
        merged: Dict[str, Any] = {}
        for container_key in ("retrieval", "retrievers"):
            block = (self.config.get(container_key) or {}).get(key)
            if isinstance(block, dict):
                merged.update(block)
        return merged

    def _crossref_hop_config(self) -> Dict[str, Any]:
        """``retrieval.crossref_hop`` (roadmap item 4.2). Default OFF."""
        return self._merged_block("crossref_hop")

    def _overview_prefilter_config(self) -> Dict[str, Any]:
        """``retrieval.overview_prefilter`` (roadmap item 4.3). Default OFF."""
        return self._merged_block("overview_prefilter")

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
        # A caller's metadata filter (item 4.4) also bounds context expansion:
        # otherwise a `chunk_index <= 0` filter would still pull chunk 1 back in
        # as a neighbour, and the guarantee "nothing that fails the filter
        # reaches synthesis" would be false.
        sql_filter = combine(sql_filter, self.active_filter())

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
        # Arm-C prompt from the synthesis-grounding A/B
        # (eval/decisions/synthesis-grounding-ab-2026-08-13.md): deletes the
        # old "General knowledge" escape hatch that produced fabricated
        # citations on unseen corpora, and forbids quotes/section numbers/
        # document names not present in the snippets.
        prompt = f"""
You are answering strictly from the retrieved snippets below.

Hard rules — these override anything you believe you know:
1. Use ONLY information stated in the snippets. Your own knowledge of the topic, however confident, must not appear in the answer.
2. If the snippets disagree with what you remember, the snippets are correct.
3. Copy every number, identifier, code and quoted phrase character-for-character from a snippet. Never write a quotation, section number or document name that does not appear in the snippets.
4. If the snippets do not contain the needed information, reply exactly:
   "I could not find that information in the provided documents."
   Do not add a general-knowledge answer after it.
5. If snippets contradict one another, state the contradiction explicitly.
6. Be thorough and well-structured, but stay within the snippets; include relevant numbers and names exactly as they appear.

Output format
Answer:
<your answer here>

–––––  Retrieved Snippets  –––––
{facts}
––––––––––––––––––––––––––––––

ORIGINAL QUESTION: "{query}"
"""
        # Stream the answer token-by-token so the caller can forward them as SSE.
        # Thinking must be OFF here: with it on, the model spends its window on
        # chain-of-thought that never enters `response` and can return an empty
        # answer (measured: prompt 9351 + thinking 7033 = window exactly, "" out).
        answer_parts: list[str] = []
        for tok in self.ollama_client.stream_completion(
            model=self.ollama_config["generation_model"],
            prompt=prompt,
            enable_thinking=False,
            options={"temperature": 0},  # greedy decode: measured fewer prior-driven drifts, zero judge splits
        ):
            answer_parts.append(tok)
            if event_callback:
                event_callback("token", {"text": tok})

        return "".join(answer_parts)

    # ------------------------------------------------------------------
    # Document-scoped search (shared by the cross-reference hop, item 4.2,
    # and the overview prefilter's "restrict" mode, item 4.3)
    # ------------------------------------------------------------------

    def _embed_query(self, query: str):
        """The query vector, reusing the retriever's LRU cache when it exists."""
        retriever = self._get_dense_retriever()
        cached = getattr(retriever, "_embed_single", None) if retriever else None
        if cached is not None:
            return cached(query)
        return self._get_text_embedder().create_embeddings([query])[0]

    def _table_normalizes(self, tbl, table_name: str) -> bool:
        """Whether *tbl* holds L2-normalized vectors; also re-checks the embedder."""
        marker = read_table_marker(tbl, getattr(self._get_db_manager(), "db_path", None),
                                   table_name)
        if marker is None:
            return False
        configured = self.config.get("embedding_model_name")
        if configured:
            assert_embedder_matches(table_name, marker, configured)
        return bool(marker["normalized"])

    @staticmethod
    def _doc_id_filter(doc_ids: List[str]) -> str:
        quoted = ", ".join("'" + str(d).replace("'", "''") + "'" for d in doc_ids)
        return f"document_id IN ({quoted})"

    @staticmethod
    def _row_to_doc(row: Dict[str, Any], score: float) -> Dict[str, Any]:
        """A LanceDB row in the shape ``MultiVectorRetriever.retrieve`` returns."""
        raw_metadata = row.get("metadata")
        if isinstance(raw_metadata, dict):
            metadata = dict(raw_metadata)
        else:
            try:
                metadata = json.loads(raw_metadata or "{}")
            except (TypeError, ValueError):
                metadata = {}
        metadata.setdefault("document_id", row.get("document_id"))
        metadata.setdefault("chunk_index", row.get("chunk_index"))
        doc = {
            "chunk_id": row.get("chunk_id"),
            "text": metadata.get("original_text") or row.get("text") or "",
            "score": score,
            "document_id": row.get("document_id"),
            "chunk_index": row.get("chunk_index"),
            "metadata": metadata,
        }
        distance = row.get("_distance")
        if distance is not None:
            try:
                doc["_distance"] = float(distance)
            except (TypeError, ValueError):
                pass
        return doc

    def _search_within_documents(self, query: str, table_name: str, doc_ids: List[str],
                                 k: int, mode: str = "vector_only") -> List[Dict[str, Any]]:
        """Retrieve up to *k* chunks, restricted to *doc_ids* with a LanceDB filter.

        ``MultiVectorRetriever.retrieve`` has no filter parameter and belongs to
        another module, so the document-scoped variant lives here. It mirrors the
        retriever exactly — same prefiltered legs, same RRF fusion at the same
        ``_RRF_K``, same output shape — so a chunk pulled by a hop is
        indistinguishable downstream from one pulled by the first stage.
        """
        if not doc_ids or k < 1:
            return []
        try:
            tbl = self._get_db_manager().get_table(table_name)
            normalize = self._table_normalizes(tbl, table_name)
        except EmbedderMismatchError:
            raise
        except Exception as e:
            print(f"⚠️  Document-scoped search: cannot open table '{table_name}': {e}")
            return []

        # A caller's metadata filter (item 4.4) narrows every internally-scoped
        # search too. The cross-reference hop must not be a way to reach a
        # document the caller filtered out.
        where = combine(self._doc_id_filter(doc_ids), self.active_filter())
        mode = (mode or "vector_only").lower()
        fts_rows: List[Dict[str, Any]] = []
        vec_rows: List[Dict[str, Any]] = []

        if mode != "fts_only":
            try:
                vector = self._embed_query(query)
                if normalize:
                    vector = l2_normalize(vector)
                vec_rows = (tbl.search(vector).where(where, prefilter=True)
                            .limit(k).to_pandas().to_dict("records"))
            except Exception as e:
                print(f"⚠️  Document-scoped vector search failed: {e}")
        if mode != "vector_only":
            try:
                fts_query = query
                if len(query.split()) == 1:
                    fts_query = f"{query}* OR {query}~"
                fts_rows = (tbl.search(query=fts_query, query_type="fts")
                            .where(where, prefilter=True)
                            .limit(k).to_pandas().to_dict("records"))
            except Exception as e:
                print(f"⚠️  Document-scoped full-text search failed: {e}")

        fused: Dict[Any, Dict[str, Any]] = {}
        for rows in (fts_rows, vec_rows):
            for rank, row in enumerate(rows, start=1):
                key = row.get("chunk_id") or row.get("text")
                entry = fused.setdefault(key, {"row": row, "rrf": 0.0})
                entry["rrf"] += 1.0 / (_RRF_K + rank)
        ordered = sorted(fused.values(), key=lambda e: e["rrf"], reverse=True)[:k]
        return [self._row_to_doc(e["row"], e["rrf"]) for e in ordered]

    # ------------------------------------------------------------------
    # Overview prefilter (roadmap item 4.3)
    # ------------------------------------------------------------------

    def _overview_vectors(self) -> Optional[Dict[str, Any]]:
        """The embedded-overview sidecar for this index, or None (logged once).

        Path resolution, most explicit first:

        1. ``retrieval.overview_prefilter.vectors_path``
        2. ``config["overview_path"]`` with its ``.jsonl`` swapped for
           ``.vectors.npz`` — this is what ``api_server`` already sets per
           session, so the HTTP path needs no extra plumbing.
        3. ``index_store/overviews/<index_id>.vectors.npz``.

        Nothing is guessed beyond that: silently prefiltering a query against
        *some other index's* overviews would be worse than not prefiltering.
        """
        if self._overview_vectors_loaded:
            return self._overview_vectors_data
        self._overview_vectors_loaded = True

        cfg = self._overview_prefilter_config()
        path = cfg.get("vectors_path")
        if not path:
            overview_path = (self.config.get("overview_path")
                             or (self.config.get("overview") or {}).get("path"))
            if overview_path:
                path = overview_vectors_path(overview_path)
        if not path:
            index_id = cfg.get("index_id") or self.config.get("index_id")
            if index_id:
                path = os.path.join("index_store", "overviews", f"{index_id}.vectors.npz")
        if not path:
            print("ℹ️  Overview prefilter is on but no overview path is configured "
                  "(set retrieval.overview_prefilter.vectors_path or overview_path); "
                  "continuing without it.")
            return None

        data = load_overview_vectors(path)
        if data is None:
            print(f"ℹ️  Overview prefilter is on but no embedded overviews were found at "
                  f"{path}; continuing without it. Re-index to build the sidecar.")
            return None

        recorded = (data.get("meta") or {}).get("embedding_model")
        configured = self.config.get("embedding_model_name")
        if recorded and configured and recorded != configured:
            print(f"ℹ️  Overview prefilter disabled: the sidecar at {path} was written by "
                  f"'{recorded}' but this pipeline uses '{configured}'.")
            return None

        print(f"🧭 Overview prefilter: {len(data['doc_ids'])} document overview(s) loaded "
              f"from {path}.")
        self._overview_vectors_data = data
        return data

    def _overview_prefilter_documents(self, query: str) -> Optional[List[str]]:
        """The top-N document ids by query-vs-overview similarity, or None."""
        cfg = self._overview_prefilter_config()
        if not cfg.get("enabled"):
            return None
        data = self._overview_vectors()
        if data is None:
            return None
        top_n = int(cfg.get("top_documents", 5) or 0)
        if top_n < 1:
            return None
        try:
            vector = l2_normalize(self._embed_query(query))
            scores = np.asarray(data["vectors"], dtype="float32") @ np.asarray(vector,
                                                                               dtype="float32")
        except Exception as e:
            print(f"⚠️  Overview prefilter scoring failed ({e}); continuing without it.")
            return None
        order = np.argsort(-scores)[:top_n]
        selected = [data["doc_ids"][int(i)] for i in order]
        print(f"🧭 Overview prefilter selected {len(selected)} document(s): "
              f"{', '.join(selected)}")
        return selected

    @staticmethod
    def _apply_overview_boost(docs: List[Dict[str, Any]],
                              prefiltered: List[str]) -> List[Dict[str, Any]]:
        """Fuse the candidate ordering with the document-overview ordering by RRF.

        A rank bonus rather than a score bonus, and RRF rather than a weighted
        sum, for the same reason the retriever fuses its two legs that way: the
        two orderings are not on a common scale and there is no validation split
        here to tune a weight against (design_rationale §4).
        """
        doc_rank = {doc_id: rank for rank, doc_id in enumerate(prefiltered)}
        scored = []
        for position, doc in enumerate(docs):
            fused = 1.0 / (_RRF_K + position + 1)
            rank = doc_rank.get(doc.get("document_id"))
            if rank is not None:
                fused += 1.0 / (_RRF_K + rank + 1)
                doc["overview_prefilter_rank"] = rank
            scored.append((fused, position, doc))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [doc for _, _, doc in scored]

    def _first_stage(self, query: str, base_table: str, retrieval_k: int, retrieval_mode: str,
                     event_callback=None) -> List[Dict[str, Any]]:
        """Hybrid/vector/FTS retrieval plus the optional late-chunk table and merge.

        Split out of ``run()`` so the evidence-sufficiency retry can call it a
        second time with a reformulated query without duplicating any of it.

        The caller's metadata filter (item 4.4) is read from the thread-local
        scope rather than passed in, so the retry re-applies it automatically
        and no existing call site changes.
        """
        start_time = time.time()
        logger = logging.getLogger(__name__)
        dense_retriever = self._get_dense_retriever()
        active_filter = self.active_filter()
        filter_where = active_filter.where if active_filter is not None else None

        # Overview prefilter (roadmap item 4.3). Computed here rather than in
        # retrieve_candidates so the evidence-sufficiency retry re-scores the
        # documents against its reformulated query too.
        prefilter_cfg = self._overview_prefilter_config()
        prefilter_mode = (prefilter_cfg.get("mode") or "boost").lower()
        prefilter_docs = self._overview_prefilter_documents(query)
        restrict_docs = prefilter_docs if (prefilter_docs and prefilter_mode == "restrict") else None

        retrieved_docs = []
        if restrict_docs:
            retrieved_docs = self._search_within_documents(
                query, base_table, restrict_docs, retrieval_k, retrieval_mode)
            if not retrieved_docs:
                print("⚠️  Overview prefilter (restrict) matched no chunks; falling back "
                      "to unrestricted retrieval for this query.")
                restrict_docs = None
        if not retrieved_docs and dense_retriever:
            retrieved_docs = dense_retriever.retrieve(
                text_query=query,
                table_name=base_table,
                k=retrieval_k,
                search_type=retrieval_mode,
                where=filter_where,
            )

        # ---------------------------------------------------------------
        # Late-Chunk retrieval (optional)
        # ---------------------------------------------------------------
        latechunk_cfg = self._latechunk_config()
        if dense_retriever and latechunk_cfg.get("enabled"):
            lc_table = self._latechunk_table_name(latechunk_cfg, base_table)
            if lc_table:
                try:
                    if restrict_docs:
                        lc_docs = self._search_within_documents(
                            query, lc_table, restrict_docs, retrieval_k, retrieval_mode)
                    else:
                        lc_docs = dense_retriever.retrieve(
                            text_query=query,
                            table_name=lc_table,
                            k=retrieval_k,
                            search_type=retrieval_mode,
                            where=filter_where,
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

        if prefilter_docs and prefilter_mode == "boost" and retrieved_docs:
            before = [d.get("chunk_id") for d in retrieved_docs[:3]]
            retrieved_docs = self._apply_overview_boost(retrieved_docs, prefilter_docs)
            if [d.get("chunk_id") for d in retrieved_docs[:3]] != before:
                print("🧭 Overview prefilter (boost) reordered the candidate list.")

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
    # Cross-reference hop (roadmap item 4.2)
    # ------------------------------------------------------------------

    # Only the head of the candidate list gets to trigger a hop. A reference
    # carried by candidate 17 is not evidence that the query is about the
    # referenced document; it is evidence that candidate 17 is noise.
    _CROSSREF_TRIGGER_DEPTH = 3

    @staticmethod
    def _chunk_crossrefs(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """The ``crossrefs`` list on a candidate, at either metadata nesting level.

        ``VectorIndexer`` stores ``json.dumps(chunk)`` in the ``metadata`` column,
        so what the retriever hands back as ``doc["metadata"]`` is the whole chunk
        dict with the real metadata nested one level down. Both shapes are
        accepted so this keeps working if that ever gets straightened out.
        """
        metadata = doc.get("metadata")
        if not isinstance(metadata, dict):
            return []
        refs = metadata.get("crossrefs")
        if refs is None:
            inner = metadata.get("metadata")
            if isinstance(inner, dict):
                refs = inner.get("crossrefs")
        if not isinstance(refs, list):
            return []
        return [r for r in refs if isinstance(r, dict)]

    def _crossref_hop(self, query: str, base_table: str, result: Dict[str, Any],
                      event_callback=None) -> Dict[str, Any]:
        """One bounded hop from a top candidate's reference to the referenced doc.

        Runs only when ``retrieval.crossref_hop.enabled`` is true. No LLM, no
        recursion: chunks pulled by a hop can never trigger another one, so the
        worst case is ``max_hops`` extra filtered searches per query.
        """
        cfg = self._crossref_hop_config()
        if not cfg.get("enabled"):
            return result

        documents = result.get("documents") or []
        max_hops = int(cfg.get("max_hops", 1) or 0)
        chunks_per_hop = int(cfg.get("chunks_per_hop", 3) or 0)
        if not documents or max_hops < 1 or chunks_per_hop < 1:
            return result

        represented = {d.get("document_id") for d in documents}
        targets: List[Dict[str, Any]] = []
        for rank, candidate in enumerate(documents[: self._CROSSREF_TRIGGER_DEPTH]):
            for ref in self._chunk_crossrefs(candidate):
                target = ref.get("target_doc")
                if not target or target in represented:
                    continue
                if any(t["target_doc"] == target for t in targets):
                    continue
                targets.append({
                    "target_doc": target,
                    "kind": ref.get("kind"),
                    "ref": ref.get("ref"),
                    "from_chunk_id": candidate.get("chunk_id"),
                    "from_document_id": candidate.get("document_id"),
                    "from_rank": rank,
                })
        targets = targets[:max_hops]
        if not targets:
            return result

        hopped: List[Dict[str, Any]] = []
        for target in targets:
            # Dense-only on purpose: the hop already knows *which* document it
            # wants, so all that is left is picking its most on-topic chunks.
            pulled = self._search_within_documents(
                query, base_table, [target["target_doc"]], chunks_per_hop,
                mode="vector_only")
            for doc in pulled:
                doc["via_crossref"] = True
                doc["crossref"] = {k: target[k] for k in
                                   ("kind", "ref", "from_chunk_id", "from_document_id")}
                if isinstance(doc.get("metadata"), dict):
                    doc["metadata"]["via_crossref"] = True
                    doc["metadata"]["crossref"] = doc["crossref"]
                hopped.append(doc)
            target["chunks_added"] = len(pulled)

        info = {"targets": targets, "chunks_added": len(hopped)}
        if hopped:
            # A new list: with reranking off, ``documents`` and ``first_stage``
            # are the same object, and the hop must not rewrite the first stage.
            result["documents"] = list(documents) + hopped
        result["crossref_hop"] = info
        print(f"🔗 Cross-reference hop: pulled {len(hopped)} chunk(s) from "
              f"{len(targets)} referenced document(s) "
              f"({', '.join(t['target_doc'] for t in targets)}).")
        if event_callback:
            event_callback("crossref_hop", info)
        return result

    def _post_candidates(self, query: str, base_table: str, result: Dict[str, Any],
                         event_callback=None) -> Dict[str, Any]:
        """Tail hook for every ``retrieve_candidates`` exit path.

        ``retrieve_candidates`` returns from five places (the retry has four
        early outs). Anything that must run on a *final* candidate set goes here
        once instead of being copy-pasted into each of them.
        """
        return self._crossref_hop(query, base_table, result, event_callback)

    def retrieve_candidates(self, query: str, table_name: Optional[str] = None,
                            sub_queries: Optional[List[str]] = None,
                            event_callback=None, *,
                            filters: Any = None) -> Dict[str, Any]:
        """First stage + rerank + the evidence-sufficiency retry around both.

        This is the whole candidate-selection path, factored out of ``run()`` so
        that ``eval/run_eval.py`` measures exactly what ships rather than a
        reimplementation of it.

        *filters* (roadmap item 4.4) is **keyword-only with a default**, so the
        four positional arguments are exactly what they were and
        ``EscalatingRetrievalPipeline``'s ``super()`` call is unaffected. It
        accepts a raw filter object or an already-compiled one and raises
        ``FilterError`` on anything invalid. When omitted, the thread-local
        scope opened by ``run()`` is used — which is how a filter reaches this
        method through the escalation subclass at all.

        Returns::

            {"first_stage": [...],   # post-retry first-stage ordering
             "documents":   [...],   # after reranking, or == first_stage
             "query_used":  str,
             "retry":       {...} | None,
             # present only when a metadata filter was applied:
             "filters":     {"spec": {...}, "where": str},
             # present only when the cross-reference hop actually hopped:
             "crossref_hop": {"targets": [...], "chunks_added": int}}
        """
        retrieval_k = self.config.get("retrieval_k", 10)
        retrieval_mode = self._retrieval_mode()
        base_table = table_name or self.storage_config["text_table_name"]

        compiled = compile_filters(filters) if filters is not None else self.active_filter()

        if event_callback:
            event_callback("retrieval_started", {"mode": retrieval_mode})

        with self.filter_scope(compiled):
            return self._retrieve_candidates_filtered(
                query, base_table, sub_queries, event_callback, compiled,
                retrieval_k, retrieval_mode)

    def _retrieve_candidates_filtered(self, query, base_table, sub_queries, event_callback,
                                      compiled, retrieval_k, retrieval_mode) -> Dict[str, Any]:
        """``retrieve_candidates``'s body, with the filter scope already open."""
        first_stage = self._first_stage(query, base_table, retrieval_k, retrieval_mode,
                                        event_callback)
        documents = self._rerank_stage(query, first_stage, sub_queries, event_callback)

        result = {"first_stage": first_stage, "documents": documents,
                  "query_used": query, "retry": None}

        if compiled is not None:
            # Only present when a filter was actually applied, so an unfiltered
            # result dict is byte-identical to what it was before item 4.4.
            result["filters"] = {"spec": compiled.spec, "where": compiled.where}
            print(f"🔎 Metadata filter applied: {compiled.where} "
                  f"→ {len(first_stage)} candidate(s).")
            if event_callback:
                event_callback("filters_applied",
                               {"spec": compiled.spec, "where": compiled.where,
                                "candidates": len(first_stage)})

        retry_cfg = self._retry_config()
        if not retry_cfg.get("enabled") or not first_stage:
            return self._post_candidates(query, base_table, result, event_callback)

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
            return self._post_candidates(query, base_table, result, event_callback)

        if score >= float(threshold):
            return self._post_candidates(query, base_table, result, event_callback)

        max_attempts = int(retry_cfg.get("max_attempts", 1) or 0)
        if max_attempts < 1:
            return self._post_candidates(query, base_table, result, event_callback)

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
        return self._post_candidates(query, base_table, result, event_callback)

    def run(self, query: str, table_name: str = None, window_size_override: Optional[int] = None,
            event_callback=None, sub_queries: Optional[List[str]] = None,
            *, filters: Any = None) -> Dict[str, Any]:
        base_table = table_name or self.storage_config["text_table_name"]

        logger = logging.getLogger(__name__)
        logger.debug("--- Running search for query '%s' (table=%s) ---", query, base_table)

        # If a custom table_name is provided, propagate it to storage config so helper methods use it
        if table_name:
            self.storage_config["text_table_name"] = table_name

        # Compiled here so an invalid filter raises before any retrieval work,
        # and opened as a thread-local scope so it reaches `retrieve_candidates`
        # without changing the positional signature the escalation subclass
        # overrides. `filters=None` skips both and leaves this path untouched.
        compiled = compile_filters(filters)
        with self.filter_scope(compiled) if compiled is not None else contextlib.nullcontext():
            candidates = self.retrieve_candidates(query, base_table, sub_queries, event_callback)
            return self._run_after_candidates(query, candidates, window_size_override,
                                              event_callback)

    def _run_after_candidates(self, query: str, candidates: Dict[str, Any],
                              window_size_override: Optional[int],
                              event_callback) -> Dict[str, Any]:
        """``run()``'s post-candidate half: expansion, pruning and synthesis."""
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
                                # Same for the cross-reference marker: expansion
                                # re-reads the row from LanceDB, which knows
                                # nothing about how the chunk got here.
                                if cid == seed_chunk.get('chunk_id') and seed_chunk.get('via_crossref'):
                                    surrounding_chunk['via_crossref'] = True
                                    if seed_chunk.get('crossref'):
                                        surrounding_chunk['crossref'] = seed_chunk['crossref']
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
        # Cross-reference hops are exempt — they are appended *after* reranking
        # by design (the reranker never saw them, and scoring them against the
        # query would defeat the point: the referenced document is exactly the
        # one whose text does not look like the query).
        if any('rerank_score' in d for d in final_docs):
            final_docs = [d for d in final_docs
                          if 'rerank_score' in d or d.get('via_crossref')]

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
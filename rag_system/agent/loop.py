from typing import Dict, Any, Optional
import contextlib
import contextvars
import copy
import json
import re
import time, asyncio, os
import numpy as np
import concurrent.futures
from cachetools import TTLCache, LRUCache
from rag_system.utils.ollama_client import (
    OllamaClient,
    TokenUsageTracker,
    token_stage,
    track_token_usage,
)
from rag_system.agent.escalation import EscalatingRetrievalPipeline
from rag_system.agent.verifier import Verifier
from rag_system.retrieval.filters import compile_filters
from rag_system.retrieval.query_transformer import QueryDecomposer

# Per-session history cap, in turns. The LRUCache on chat_histories bounds
# session COUNT; nothing else bounded turns within a session.
_MAX_HISTORY_TURNS = 40

# The verifier's answer suffixes: " [Confidence: 85%]" plus the optional
# low-confidence warning. Per-response UX only — they must not be stored in
# history, where they would leak into later prompts (and into the
# decomposer's last_assistant_answer).
_ANSWER_TAGS_RE = re.compile(
    r"\s*\[Confidence: \d+%\](\s*\[Warning: Low confidence\. Groundedness: [^\]]*\])?\s*$"
)


def _strip_answer_tags(answer: str) -> str:
    """Return *answer* without the verifier's trailing confidence/warning tags."""
    return _ANSWER_TAGS_RE.sub("", answer) if answer else answer


def _in_copied_context(fn):
    """Run *fn* in a copy of the caller's context, for ThreadPoolExecutor.

    ``submit()`` does not propagate context variables to the worker thread, so
    without this the per-query token tracker (roadmap 4.5) would silently miss
    every LLM call made by a parallel sub-query. One fresh copy per submitted
    task — a ``Context`` cannot be entered twice concurrently.
    """
    ctx = contextvars.copy_context()

    def _runner(*args, **kwargs):
        return ctx.run(fn, *args, **kwargs)

    return _runner

class Agent:
    """
    The main agent, now fully wired to use a live Ollama client.
    """
    def __init__(self, pipeline_configs: Dict[str, Dict], llm_client: OllamaClient, ollama_config: Dict[str, str]):
        self.pipeline_configs = pipeline_configs
        self.llm_client = llm_client
        self.ollama_config = ollama_config

        # Utility work (routing, triage, decomposition, verification) runs on the
        # small enrichment model; only user-facing answers use the generation model.
        utility_model = self._utility_model()

        # Initialize the single, persistent retrieval pipeline for this agent.
        # EscalatingRetrievalPipeline is a plain RetrievalPipeline with roadmap
        # 4.1's full-document escalation hooked in; the hooks are inert unless
        # `retrieval.document_escalation.enabled` is set, which it is not by
        # default. See rag_system/agent/escalation.py.
        self.retrieval_pipeline = EscalatingRetrievalPipeline(
            pipeline_configs, self.llm_client, self.ollama_config
        )

        # `verification.model` (or the VERIFIER_MODEL env var, read inside
        # Verifier) swaps the LLM-prompt verifier for a local NLI/verifier model.
        # Unset by default — the LLM prompt is what ships (roadmap 2.4).
        verification_config = self.pipeline_configs.get("verification", {}) or {}
        self.verifier = Verifier(
            llm_client,
            utility_model,
            model_name=verification_config.get("model"),
            threshold=float(verification_config.get("threshold", 0.5)),
        )
        self.query_decomposer = QueryDecomposer(llm_client, utility_model)

        # 🚀 OPTIMIZED: TTL cache now stores embeddings for semantic matching
        self._query_cache: TTLCache = TTLCache(maxsize=100, ttl=300)
        self.semantic_cache_threshold = self.pipeline_configs.get("semantic_cache_threshold", 0.98)
        # If set to "global", semantic-cache hits are reused across chat sessions.
        # The default keeps answers (and therefore document content) inside one session.
        self.cache_scope = self.pipeline_configs.get("cache_scope", "session")  # 'global' or 'session'

        # 🚀 NEW: In-memory store for conversational history per session
        self.chat_histories: LRUCache = LRUCache(maxsize=100) # Stores history for 100 recent sessions

        # ---- Load document overviews for fast routing ----
        self._global_overview_path = os.path.join("index_store", "overviews", "overviews.jsonl")
        self.doc_overviews: list[str] = []
        self._current_overview_session: str | None = None  # cache key to avoid rereading on every query
        self._load_overviews(self._global_overview_path)

    def _utility_model(self) -> str:
        """Model used for routing, triage, decomposition and verification."""
        return self.ollama_config.get("enrichment_model") or self.ollama_config["generation_model"]

    def _load_overviews(self, path: str):
        """Helper to load overviews from a .jsonl file into self.doc_overviews."""
        import json, os
        self.doc_overviews.clear()
        if not os.path.exists(path):
            return
        try:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                        if isinstance(rec, dict) and rec.get("overview"):
                            self.doc_overviews.append(rec["overview"].strip())
                    except Exception:
                        continue
            print(f"📖 Loaded {len(self.doc_overviews)} overviews from {path}")
        except Exception as e:
            print(f"⚠️  Failed to load document overviews from {path}: {e}")

    def load_overviews_for_indexes(self, idx_ids: list[str]):
        """Aggregate overviews for the given indexes or fall back to global file."""
        import os, json
        aggregated: list[str] = []
        for idx in idx_ids:
            path = os.path.join("index_store", "overviews", f"{idx}.jsonl")
            if os.path.exists(path):
                try:
                    with open(path, encoding="utf-8") as fh:
                        for line in fh:
                            if not line.strip():
                                continue
                            try:
                                rec = json.loads(line)
                                ov = rec.get("overview", "").strip()
                                if ov:
                                    aggregated.append(ov)
                            except json.JSONDecodeError:
                                continue
                except Exception as e:
                    print(f"⚠️  Error reading {path}: {e}")
        if aggregated:
            self.doc_overviews = aggregated
            self._current_overview_session = "|".join(idx_ids)  # cache composite key so no overwrite
            print(f"📖 Loaded {len(aggregated)} overviews for indexes {[i[:8] for i in idx_ids]}")
        else:
            print(f"⚠️  No per-index overviews found for {idx_ids}. Using global overview file.")
            self._load_overviews(self._global_overview_path)
            self._current_overview_session = "GLOBAL"

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Computes cosine similarity between two vectors."""
        if not isinstance(v1, np.ndarray): v1 = np.array(v1)
        if not isinstance(v2, np.ndarray): v2 = np.array(v2)
        
        if v1.shape != v2.shape:
            raise ValueError("Vectors must have the same shape for cosine similarity.")

        if np.all(v1 == 0) or np.all(v2 == 0):
            return 0.0
            
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        # Avoid division by zero
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        
        return dot_product / (norm_v1 * norm_v2)

    def _find_in_semantic_cache(self, query_embedding: np.ndarray, session_id: Optional[str] = None,
                                filter_signature: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Finds a semantically similar query in the cache."""
        if not self._query_cache or query_embedding is None:
            return None

        for key, cached_item in self._query_cache.items():
            cached_embedding = cached_item.get('embedding')
            if cached_embedding is None:
                continue

            # Respect cache scoping: if scope is session-level, skip results from other sessions
            if self.cache_scope != "global" and cached_item.get("session_id") != session_id:
                continue

            # A metadata filter (item 4.4) is part of the question, not a view
            # over the answer: "what does the NDA say" and "what do all ten
            # documents say" have near-identical embeddings and different right
            # answers. Both sides are None on the unfiltered path, so this is a
            # no-op unless someone actually filtered.
            if cached_item.get("filters") != filter_signature:
                continue

            try:
                similarity = self._cosine_similarity(query_embedding, cached_embedding)

                if similarity >= self.semantic_cache_threshold:
                    print(f"🚀 Semantic cache hit! Similarity: {similarity:.3f} with cached query '{key}'")
                    return cached_item.get('result')
            except ValueError:
                # In case of shape mismatch, just skip
                continue

        return None

    def _format_query_with_history(self, query: str, history: list) -> str:
        """Formats the user query with conversation history for context."""
        if not history:
            return query
        
        formatted_history = "\n".join([f"User: {turn['query']}\nAssistant: {turn['answer']}" for turn in history])
        
        prompt = f"""
Given the following conversation history, answer the user's latest query. The history provides context for resolving pronouns or follow-up questions.

--- Conversation History ---
{formatted_history}
---

Latest User Query: "{query}"
"""
        return prompt

    # ---------------- Asynchronous triage using Ollama ----------------
    @staticmethod
    def _normalize_triage(decision: str) -> str:
        """Collapse a triage label to one of the two categories that still exist.

        ``graph_query`` was a third outcome until the graph module was removed
        (roadmap 2.5, 2026-08-09). Small utility models still emit it from time to
        time — they have seen the label in their training data — so anything that
        is not an explicit ``direct_answer`` is answered from the documents.
        """
        return "direct_answer" if (decision or "").strip() == "direct_answer" else "rag_query"

    async def _triage_query_async(self, query: str, history: list) -> str:

        print(f"🔍 ROUTING DEBUG: Starting triage for query: '{query[:100]}...'")
        
        # 1️⃣ Fast routing using precomputed overviews (if available)
        print(f"📖 ROUTING DEBUG: Attempting overview-based routing...")
        routed = self._route_via_overviews(query)
        if routed:
            routed = self._normalize_triage(routed)
            print(f"✅ ROUTING DEBUG: Overview routing decided: '{routed}'")
            return routed
        else:
            print(f"❌ ROUTING DEBUG: Overview routing returned None, falling back to LLM triage")

        if history:
            # If there's history, the query is likely a follow-up, so we default to RAG.
            # A more advanced implementation could use an LLM to see if the new query
            # changes the topic entirely.
            print(f"📜 ROUTING DEBUG: History exists, defaulting to 'rag_query'")
            return "rag_query"

        print(f"🤖 ROUTING DEBUG: No history, using LLM fallback triage...")
        prompt = f"""
You are a query routing expert. Analyze the user's question and decide which backend should handle it.

Choose **exactly one** category:

1. "rag_query" – Questions about the user's uploaded documents or specific document content that should be searched. Examples: "What is the invoice amount?", "Summarize the research paper", "What companies are mentioned?"

2. "direct_answer" – General knowledge questions, greetings, or queries unrelated to uploaded documents. Examples: "Who are the CEOs of Tesla and Amazon?", "What is the capital of France?", "Hello", "Explain quantum physics"

IMPORTANT: For general world knowledge about well-known companies, people, or facts NOT related to uploaded documents, choose "direct_answer".

User query: "{query}"

Respond with JSON: {{"category": "<your_choice>"}}
"""
        resp = self.llm_client.generate_completion(
            model=self._utility_model(), prompt=prompt, format="json",
            options={"temperature": 0},  # deterministic routing (same pin the eval judge got in ebcc88b)
        )
        try:
            data = json.loads(resp.get("response", "{}"))
            decision = self._normalize_triage(data.get("category", "rag_query"))
            print(f"🤖 ROUTING DEBUG: LLM fallback triage decided: '{decision}'")
            return decision
        except json.JSONDecodeError:
            print(f"❌ ROUTING DEBUG: LLM fallback triage JSON parsing failed, defaulting to 'rag_query'")
            return "rag_query"

    # ---------------- Public sync API (kept for backwards compatibility) --------------
    def run(self, query: str, table_name: str = None, session_id: str = None, compose_sub_answers: Optional[bool] = None, query_decompose: Optional[bool] = None, ai_rerank: Optional[bool] = None, context_expand: Optional[bool] = None, verify: Optional[bool] = None, retrieval_k: Optional[int] = None, context_window_size: Optional[int] = None, reranker_top_k: Optional[int] = None, retrieval_mode: Optional[str] = None, force_rag: bool = False, event_callback: Optional[callable] = None, *, filters: Any = None) -> Dict[str, Any]:
        """Synchronous helper. If *event_callback* is supplied, important
        milestones will be forwarded to that callable as

            event_callback(phase:str, payload:Any)

        *filters* (roadmap item 4.4) is a metadata filter object; it is
        keyword-only and defaults to None, in which case nothing about this
        call changes.
        """
        return asyncio.run(self._run_async(query, table_name, session_id, compose_sub_answers, query_decompose, ai_rerank, context_expand, verify, retrieval_k, context_window_size, reranker_top_k, retrieval_mode, force_rag, event_callback, filters=filters))

    # ---------------- Per-query bookkeeping wrapper -----------------------------------
    @contextlib.contextmanager
    def _pipeline_config_overrides(self, *, ai_rerank: Optional[bool] = None, retrieval_k: Optional[int] = None, context_window_size: Optional[int] = None, reranker_top_k: Optional[int] = None, retrieval_mode: Optional[str] = None):
        """Apply per-request retrieval overrides, then restore the shared config.

        The retrieval pipeline is a process-wide singleton, so writing these
        straight into ``pipeline.config`` made one request's toggle sticky for
        every later request. Snapshot the affected subtrees, apply, and restore
        in ``finally`` — the same pattern as ``_generation_model_override`` in
        api_server.py. Snapshot/restore is safe because the RAG API server is
        single-threaded by design; a concurrent server would need a lock.
        """
        config = self.retrieval_pipeline.config
        missing = object()
        snapshot = {}
        for key in ("reranker", "retrieval", "retrieval_k", "context_window_size"):
            value = config.get(key, missing)
            snapshot[key] = copy.deepcopy(value) if value is not missing else value

        if ai_rerank is not None:
            rr_cfg = config.setdefault("reranker", {})
            rr_cfg["enabled"] = bool(ai_rerank)
        if retrieval_k is not None:
            config["retrieval_k"] = retrieval_k
            print(f"🔍 Retrieval K set to: {retrieval_k}")
        if context_window_size is not None:
            config["context_window_size"] = context_window_size
            print(f"🔍 Context window size set to: {context_window_size}")
        if reranker_top_k is not None:
            rr_cfg = config.setdefault("reranker", {})
            rr_cfg["top_k"] = reranker_top_k
            print(f"🔍 Reranker top K set to: {reranker_top_k}")
        if retrieval_mode is not None:
            retrieval_cfg = config.setdefault("retrieval", {})
            retrieval_cfg["search_type"] = retrieval_mode
            print(f"🔍 Retrieval mode set to: {retrieval_mode}")

        try:
            yield
        finally:
            for key, value in snapshot.items():
                if value is missing:
                    config.pop(key, None)
                else:
                    config[key] = value

    async def _run_async(self, query: str, table_name: str = None, session_id: str = None, compose_sub_answers: Optional[bool] = None, query_decompose: Optional[bool] = None, ai_rerank: Optional[bool] = None, context_expand: Optional[bool] = None, verify: Optional[bool] = None, retrieval_k: Optional[int] = None, context_window_size: Optional[int] = None, reranker_top_k: Optional[int] = None, retrieval_mode: Optional[str] = None, force_rag: bool = False, event_callback: Optional[callable] = None, *, filters: Any = None) -> Dict[str, Any]:
        """Wraps one user query in its token-usage, config-override and escalation scopes.

        All three are per-*query*, not per-retrieval, which is why they live out
        here rather than inside the pipeline: a decomposed query runs the
        pipeline several times and must still report one token total, see one
        set of runtime overrides and escalate at most one document
        (roadmap 4.1 / 4.5).
        """
        tracker = TokenUsageTracker()
        pipeline = self.retrieval_pipeline
        can_escalate = hasattr(pipeline, "begin_escalation_request")

        with track_token_usage(tracker):
            if can_escalate:
                pipeline.begin_escalation_request()
            try:
                with self._pipeline_config_overrides(
                    ai_rerank=ai_rerank,
                    retrieval_k=retrieval_k,
                    context_window_size=context_window_size,
                    reranker_top_k=reranker_top_k,
                    retrieval_mode=retrieval_mode,
                ):
                    result = await self._run_async_inner(query, table_name, session_id, compose_sub_answers, query_decompose, ai_rerank, context_expand, verify, retrieval_k, context_window_size, reranker_top_k, retrieval_mode, force_rag, event_callback, filters=filters)
            finally:
                budget = pipeline.end_escalation_request() if can_escalate else None

        if not isinstance(result, dict):
            return result

        # Shallow-copied so writing the usage of *this* run cannot mutate an
        # entry the semantic cache is holding on to.
        result = dict(result)
        result["token_usage"] = tracker.as_dict()
        if budget is not None and budget.events:
            result["document_escalation"] = budget.events
        return result

    # ---------------- Main async implementation --------------------------------------
    async def _run_async_inner(self, query: str, table_name: str = None, session_id: str = None, compose_sub_answers: Optional[bool] = None, query_decompose: Optional[bool] = None, ai_rerank: Optional[bool] = None, context_expand: Optional[bool] = None, verify: Optional[bool] = None, retrieval_k: Optional[int] = None, context_window_size: Optional[int] = None, reranker_top_k: Optional[int] = None, retrieval_mode: Optional[str] = None, force_rag: bool = False, event_callback: Optional[callable] = None, *, filters: Any = None) -> Dict[str, Any]:
        start_time = time.time()

        # Compiled once per user query (roadmap 4.4) and handed to every
        # retrieval this query performs, including each parallel sub-query.
        # Raises FilterError on an invalid filter — /chat validates first and
        # returns 400, so reaching here with a bad filter means a non-HTTP caller.
        compiled_filters = compile_filters(filters)
        filter_signature = compiled_filters.signature if compiled_filters else None

        # Emit analyze event at the start
        if event_callback:
            event_callback("analyze", {"query": query})
        
        # 🚀 NEW: Get conversation history
        history = self.chat_histories.get(session_id, []) if session_id else []

        if force_rag or compiled_filters is not None:
            # A metadata filter is an instruction to search a named part of the
            # corpus (roadmap 4.4). Letting triage answer it from the model's
            # general knowledge instead would ignore the filter completely and
            # look, from the outside, exactly like a filter that matched
            # nothing — so a filter skips triage the way force_rag does.
            query_type = "rag_query"
            reason = "force_rag" if force_rag else "metadata filter"
            print(f"🎯 ROUTING DEBUG: {reason} set – triage skipped, using 'rag_query'")
        else:
            with token_stage("triage"):
                query_type = await self._triage_query_async(query, history)
            print(f"🎯 ROUTING DEBUG: Final triage decision: '{query_type}'")
        print(f"Agent Triage Decision: '{query_type}'")

        # Create a contextual query that includes history for most operations
        contextual_query = self._format_query_with_history(query, history)
        raw_query = query.strip()

        # Runtime retrieval overrides (ai_rerank, retrieval_k, context_window_size,
        # reranker_top_k, retrieval_mode) are applied around this whole method by
        # _run_async's _pipeline_config_overrides scope and rolled back there, so
        # everything below already sees this request's config.

        query_embedding = None
        # 🚀 OPTIMIZED: Semantic Cache Check
        if query_type != "direct_answer":
            text_embedder = self.retrieval_pipeline._get_text_embedder()
            if text_embedder:
                # The embedder expects a list, so we wrap the *raw* query only.
                query_embedding_list = text_embedder.create_embeddings([raw_query])
                if isinstance(query_embedding_list, np.ndarray):
                    query_embedding = query_embedding_list[0]
                else:
                    # Some embedders return a list – convert if necessary
                    query_embedding = np.array(query_embedding_list[0])

                cached_result = self._find_in_semantic_cache(query_embedding, session_id,
                                                             filter_signature)

                if cached_result:
                    # Update history even on cache hit
                    if session_id:
                        history.append({"query": query, "answer": _strip_answer_tags(cached_result.get('answer', 'Cached answer not found.'))})
                        self.chat_histories[session_id] = history[-_MAX_HISTORY_TURNS:]
                    return cached_result

        if query_type == "direct_answer":
            print(f"✅ ROUTING DEBUG: Executing DIRECT_ANSWER path")
            if event_callback:
                event_callback("direct_answer", {})

            prompt = (
                "You are a helpful assistant. Read the conversation history below. "
                "If the answer to the user's latest question is already present in the history, quote it concisely. "
                "Otherwise answer from your general world knowledge. Provide a short, factual reply (1‒2 sentences).\n\n"
                f"Conversation + Latest Question:\n{contextual_query}\n\nAssistant:"
            )

            async def _run_stream():
                answer_parts: list[str] = []

                def _blocking_stream():
                    with token_stage("synthesis"):
                        for tok in self.llm_client.stream_completion(
                            model=self.ollama_config["generation_model"], prompt=prompt,
                            enable_thinking=False,  # thinking burns the window and can yield an empty answer
                            options={"temperature": 0},  # greedy decode, matches synthesis (arm C config)
                        ):
                            answer_parts.append(tok)
                            if event_callback:
                                event_callback("token", {"text": tok})

                # Run the blocking generator in a thread so the event loop stays responsive
                await asyncio.to_thread(_blocking_stream)
                return "".join(answer_parts)

            final_answer = await _run_stream()
            result = {"answer": final_answer, "source_documents": []}

        # --- RAG Query Processing with Optional Query Decomposition ---
        else: # Default to rag_query
            print(f"✅ ROUTING DEBUG: Executing RAG_QUERY path (query_type='{query_type}')")
            query_decomp_config = self.pipeline_configs.get("query_decomposition", {})
            decomp_enabled = query_decomp_config.get("enabled", False)
            if query_decompose is not None:
                decomp_enabled = query_decompose

            if decomp_enabled:
                print(f"\n--- Query Decomposition Enabled ---")
                # Use the raw user query (without conversation history) for decomposition to avoid leakage of prior context
                # Pass the last 5 conversation turns for context resolution within the decomposer
                recent_history = history[-5:] if history else []
                with token_stage("decomposition"):
                    sub_queries = self.query_decomposer.decompose(
                        raw_query,
                        recent_history,
                        max_sub_queries=query_decomp_config.get("max_sub_queries", 10),
                        resolve_only=bool(query_decomp_config.get("resolve_only", False)),
                    )
                if event_callback:
                    event_callback("decomposition", {"sub_queries": sub_queries})
                print(f"Original query: '{query}' (Contextual: '{contextual_query}')")
                print(f"Decomposed into {len(sub_queries)} sub-queries: {sub_queries}")

                # retrieval_started is emitted per branch below, with the count
                # that branch actually retrieves with — 1 for the single and
                # pooled paths, N for compose. One agent-level event per
                # logical retrieval, so the frontend never sees contradictory
                # counts (the pipeline emits its own retrieval_started too).
                # If decomposition produced only a single sub-query, skip the
                # parallel/composition machinery for efficiency.
                if len(sub_queries) == 1:
                    print("--- Only one sub-query after decomposition; using direct retrieval path ---")
                    if event_callback:
                        event_callback("retrieval_started", {"count": 1})
                    with token_stage("synthesis"):
                        result = self.retrieval_pipeline.run(
                            sub_queries[0],
                            table_name,
                            0 if context_expand is False else None,
                            event_callback=event_callback,
                            filters=compiled_filters,
                        )
                    if event_callback:
                        event_callback("single_query_result", result)
                    # Emit retrieval_done and rerank_done for single sub-query
                    if event_callback:
                        event_callback("retrieval_done", {"count": 1})
                        event_callback("rerank_started", {"count": 1})
                        event_callback("rerank_done", {"count": 1})
                else:
                    compose_from_sub_answers = query_decomp_config.get("compose_from_sub_answers", True)
                    if compose_sub_answers is not None:
                        compose_from_sub_answers = compose_sub_answers

                    if not compose_from_sub_answers:
                        # ---- Roadmap item 2.2: decomposition applies at RERANK ----
                        # The first stage runs ONCE, on the full original query.
                        # Fanning the *first stage* out over sub-queries dilutes
                        # it semantically (2026 MultiConIR/SSRB finding); the
                        # sub-queries earn their keep at the rerank stage, where
                        # every candidate is scored against every sub-query and
                        # the scores are aggregated with
                        # `query_decomposition.rerank_aggregate` ("max" or "mean").
                        # With reranking off there is no rerank stage, so the
                        # sub-queries go unused and this is plain single-query
                        # retrieval — which is the shipped default.
                        if query_decomp_config.get("pooled_first_stage"):
                            print("\n--- Pooled decomposition: per-sub-query retrieval, "
                                  "single rerank + synthesis ---")
                        else:
                            print("\n--- Decomposition applied at rerank; first stage uses the full query ---")
                        if event_callback:
                            event_callback("retrieval_started", {"count": 1})
                        with token_stage("synthesis"):
                            result = self.retrieval_pipeline.run(
                                contextual_query,
                                table_name,
                                0 if context_expand is False else None,
                                event_callback=event_callback,
                                sub_queries=sub_queries,
                                filters=compiled_filters,
                            )
                        if event_callback:
                            event_callback("final_answer", result)
                    else:
                        # `compose_from_sub_answers` keeps per-sub-query *retrieval*
                        # on purpose, and is the only thing that does: it needs a
                        # separate answer per sub-question to compose from, which a
                        # single shared candidate set cannot produce. One full
                        # RetrievalPipeline.run() — retrieval, rerank, synthesis —
                        # per sub-query, in parallel.
                        print(f"\n--- Processing {len(sub_queries)} sub-queries in parallel ---")
                        start_time_inner = time.time()

                        sub_answers = []
                        all_source_docs = []
                        citations_seen = set()

                        # One retrieval_started for the whole fan-out: each
                        # sub-query runs a full retrieval below.
                        if event_callback:
                            event_callback("retrieval_started", {"count": len(sub_queries)})

                        # Emit rerank_started before the parallel retrievals (each sub-query reranks).
                        if event_callback:
                            event_callback("rerank_started", {"count": len(sub_queries)})

                        # Emit token chunks as soon as we receive them. The UI
                        # keeps answers separated by `index`, so interleaving is
                        # harmless and gives continuous feedback.

                        def make_cb(idx: int):
                            def _cb(ev_type: str, payload):
                                if event_callback is None:
                                    return
                                if ev_type == "token":
                                    event_callback("sub_query_token", {"index": idx, "text": payload.get("text", ""), "question": sub_queries[idx]})
                                else:
                                    event_callback(ev_type, payload)
                            return _cb

                        with concurrent.futures.ThreadPoolExecutor(max_workers=min(3, len(sub_queries))) as executor:
                            # Each task runs in its own copy of the current
                            # context so the per-query token tracker and the
                            # "synthesis" stage label reach the worker thread
                            # (submit() does not propagate context variables).
                            with token_stage("synthesis"):
                                future_to_query = {
                                    executor.submit(
                                        _in_copied_context(self.retrieval_pipeline.run),
                                        sub_query,
                                        table_name,
                                        0 if context_expand is False else None,
                                        make_cb(i),
                                        filters=compiled_filters,
                                    ): (i, sub_query)
                                    for i, sub_query in enumerate(sub_queries)
                                }

                            for future in concurrent.futures.as_completed(future_to_query):
                                i, sub_query = future_to_query[future]
                                try:
                                    sub_result = future.result()
                                    print(f"✅ Sub-Query {i+1} completed: '{sub_query}'")

                                    if event_callback:
                                        event_callback("sub_query_result", {
                                            "index": i,
                                            "query": sub_query,
                                            "answer": sub_result.get("answer", ""),
                                            "source_documents": sub_result.get("source_documents", []),
                                        })

                                    sub_answers.append({
                                        "question": sub_query,
                                        "answer": sub_result.get("answer", "")
                                    })
                                    # Keep up to 5 citations per sub-query for traceability
                                    for doc in sub_result.get("source_documents", [])[:5]:
                                        if doc['chunk_id'] not in citations_seen:
                                            all_source_docs.append(doc)
                                            citations_seen.add(doc['chunk_id'])
                                except Exception as e:
                                    print(f"❌ Sub-Query {i+1} failed: '{sub_query}' - {e}")

                        print(f"🚀 Parallel processing completed in {time.time() - start_time_inner:.2f}s")

                        # Every sub-query raised: composing from "[]" would dress
                        # total failure up as a 200. Fail the way the single-query
                        # path does (raise -> 500 / SSE error); a partial failure
                        # stays best-effort.
                        if not sub_answers:
                            raise RuntimeError(
                                f"All {len(sub_queries)} sub-queries failed for query: '{raw_query}'"
                            )

                        if event_callback:
                            event_callback("retrieval_done", {"count": len(sub_queries)})
                            event_callback("rerank_done", {"count": len(sub_queries)})

                        print("\n--- Composing final answer from sub-answers ---")
                        compose_prompt = f"""
You are an expert answer composer for a Retrieval-Augmented Generation (RAG) system.

Context:
• The ORIGINAL QUESTION from the user is shown below.
• That question was automatically decomposed into simpler SUB-QUESTIONS.
• Each sub-question has already been answered by an earlier step and the resulting Question→Answer pairs are provided to you in JSON.

Your task:
1. Read every sub-answer carefully.
2. Write a single, final answer to the ORIGINAL QUESTION **using only the information contained in the sub-answers**. Do NOT invent facts that are not present.
3. If the original question includes a comparison (e.g., "Which, A or B, …") clearly state the outcome (e.g., "A > B"). Quote concrete numbers when available.
4. If any aspect of the original question cannot be answered with the given sub-answers, explicitly say so (e.g., "The provided context does not mention …").
5. Keep the answer concise (≤ 5 sentences) and use a factual, third-person tone.

Input
------
ORIGINAL QUESTION:
"{contextual_query}"

SUB-ANSWERS (JSON):
{json.dumps(sub_answers, indent=2)}

------
FINAL ANSWER:
"""
                        # --- Stream composition answer token-by-token ---
                        answer_parts: list[str] = []

                        def _blocking_compose_stream():
                            with token_stage("synthesis"):
                                for tok in self.llm_client.stream_completion(
                                    model=self.ollama_config["generation_model"],
                                    prompt=compose_prompt,
                                    enable_thinking=False,  # thinking burns the window and can yield an empty answer
                                    options={"temperature": 0},  # greedy decode, matches synthesis (arm C config)
                                ):
                                    answer_parts.append(tok)
                                    if event_callback:
                                        event_callback("token", {"text": tok})

                        # Run the blocking generator in a thread so the event loop stays responsive
                        await asyncio.to_thread(_blocking_compose_stream)

                        final_answer = "".join(answer_parts) or "Unable to generate an answer."

                        result = {
                            "answer": final_answer,
                            "source_documents": all_source_docs
                        }
                        if event_callback:
                            event_callback("final_answer", result)
            else:
                # Standard retrieval (single-query)
                with token_stage("synthesis"):
                    result = self.retrieval_pipeline.run(contextual_query, table_name, 0 if context_expand is False else None, event_callback=event_callback, filters=compiled_filters)


        # Verification step (simplified for now) - Skip in fast mode
        verification_enabled = self.pipeline_configs.get("verification", {}).get("enabled", True)
        if verify is not None:
            verification_enabled = verify
            
        if verification_enabled and result.get("source_documents"):
            context_str = "\n".join([doc['text'] for doc in result['source_documents']])
            with token_stage("verification"):
                verification = await self.verifier.verify_async(contextual_query, context_str, result['answer'])
            
            score = verification.confidence_score

            # Only include confidence details if we received a non-zero score (0 usually means JSON parse failure)
            if score > 0:
                result['answer'] += f" [Confidence: {score}%]"
                # Add warning only when the verifier explicitly reported low confidence / not grounded
                if (not verification.is_grounded) or score < 50:
                    result['answer'] += f" [Warning: Low confidence. Groundedness: {verification.is_grounded}]"
            else:
                # Skip appending any verifier note – 0 likely indicates a parser error
                print("⚠️  Verifier returned 0 confidence – likely JSON parse error; omitting tags.")
        else:
            print("🚀 Skipping verification for speed or lack of sources")
        
        # 🚀 NEW: Update history — store the answer WITHOUT the verifier's
        # confidence tags; they are per-response UX and would otherwise leak
        # into later prompts (and the decomposer's last_assistant_answer).
        if session_id:
            history.append({"query": query, "answer": _strip_answer_tags(result['answer'])})
            self.chat_histories[session_id] = history[-_MAX_HISTORY_TURNS:]
            
        # 🚀 OPTIMIZED: Cache the result for future queries
        if query_type != "direct_answer" and query_embedding is not None:
            cache_key = raw_query  # Key is for logging/debugging
            self._query_cache[cache_key] = {
                "embedding": query_embedding,
                "result": result,
                "session_id": session_id,
                "filters": filter_signature,
            }
        
        total_time = time.time() - start_time
        print(f"🚀 Total query processing time: {total_time:.2f}s")
        
        return result

    # ------------------------------------------------------------------
    def _route_via_overviews(self, query: str) -> str | None:
        """Use document overviews and a small model to decide routing.
        Returns 'rag_query', 'direct_answer', or None if unsure/disabled."""
        if not self.doc_overviews:
            print(f"📖 ROUTING DEBUG: No document overviews available, returning None")
            return None
        
        print(f"📖 ROUTING DEBUG: Found {len(self.doc_overviews)} document overviews, using LLM routing...")

        # Keep prompt concise: if more than 40 overviews, take first 40
        overviews_snip = self.doc_overviews[:40]
        overviews_block = "\n".join(f"[{i+1}] {ov}" for i, ov in enumerate(overviews_snip))

        router_prompt = f"""Task: Route query to correct system.

DOCUMENT OVERVIEWS:
{overviews_block}

Query: "{query}"

Is this query asking about:
A) Greetings/social: "Hi", "Hello", "Thanks", "What's up", "How are you"
B) General knowledge unrelated to the documents above: "CEO of Tesla", "capital of France", "what is 2+2"
C) Anything covered by, or plausibly contained in, the documents above

If A or B → {{"category": "direct_answer"}}
If C → {{"category": "rag_query"}}

Response:"""

        resp = self.llm_client.generate_completion(
            model=self._utility_model(), prompt=router_prompt, format="json",
            options={"temperature": 0},  # deterministic routing (same pin the eval judge got in ebcc88b)
        )
        try:
            raw_response = resp.get("response", "{}")
            print(f"📖 ROUTING DEBUG: Overview LLM raw response: '{raw_response[:200]}...'")
            data = json.loads(raw_response)
            decision = data.get("category", "rag_query")
            print(f"📖 ROUTING DEBUG: Overview routing final decision: '{decision}'")
            return decision
        except json.JSONDecodeError as e:
            print(f"❌ ROUTING DEBUG: Overview routing JSON parsing failed: {e}, defaulting to 'rag_query'")
            return "rag_query"

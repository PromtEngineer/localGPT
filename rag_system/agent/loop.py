from typing import Dict, Any, Optional
import json
import time, asyncio, os
import numpy as np
import concurrent.futures
from cachetools import TTLCache, LRUCache
from rag_system.utils.ollama_client import OllamaClient
from rag_system.pipelines.retrieval_pipeline import RetrievalPipeline
from rag_system.agent.verifier import Verifier
from rag_system.retrieval.query_transformer import QueryDecomposer, GraphQueryTranslator
from rag_system.retrieval.retrievers import GraphRetriever

class Agent:
    """
    The main agent, now fully wired to use a live Ollama client.
    """
    def __init__(self, pipeline_configs: Dict[str, Dict], llm_client: OllamaClient, ollama_config: Dict[str, str]):
        self.pipeline_configs = pipeline_configs
        self.llm_client = llm_client
        self.ollama_config = ollama_config
        
        gen_model = self.ollama_config["generation_model"]
        
        # Initialize the single, persistent retrieval pipeline for this agent
        self.retrieval_pipeline = RetrievalPipeline(pipeline_configs, self.llm_client, self.ollama_config)
        
        self.verifier = Verifier(llm_client, gen_model)
        self.query_decomposer = QueryDecomposer(llm_client, gen_model)
        
        # 🚀 OPTIMIZED: TTL cache now stores embeddings for semantic matching
        self._cache_max_size = 100  # fallback size limit for manual eviction helper
        self._query_cache: TTLCache = TTLCache(maxsize=self._cache_max_size, ttl=300)
        self.semantic_cache_threshold = self.pipeline_configs.get("semantic_cache_threshold", 0.98)
        # If set to "session", semantic-cache hits will be restricted to the same chat session.
        # Otherwise (default "global") answers can be reused across sessions.
        self.cache_scope = self.pipeline_configs.get("cache_scope", "global")  # 'global' or 'session'
        
        # 🚀 NEW: In-memory store for conversational history per session
        self.chat_histories: LRUCache = LRUCache(maxsize=100) # Stores history for 100 recent sessions

        graph_config = self.pipeline_configs.get("graph_strategy", {})
        if graph_config.get("enabled"):
            self.graph_query_translator = GraphQueryTranslator(llm_client, gen_model)
            self.graph_retriever = GraphRetriever(graph_config["graph_path"])
            print("Agent initialized with live GraphRAG capabilities.")
        else:
            print("Agent initialized (GraphRAG disabled).")

        # ---- Load document overviews for fast routing ----
        self._global_overview_path = os.path.join("index_store", "overviews", "overviews.jsonl")
        self.doc_overviews: list[str] = []
        self._current_overview_session: str | None = None  # cache key to avoid rereading on every query
        self._load_overviews(self._global_overview_path)

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

    def _find_in_semantic_cache(self, query_embedding: np.ndarray, session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Finds a semantically similar query in the cache."""
        if not self._query_cache or query_embedding is None:
            return None

        for key, cached_item in self._query_cache.items():
            cached_embedding = cached_item.get('embedding')
            if cached_embedding is None:
                continue

            # Respect cache scoping: if scope is session-level, skip results from other sessions
            if self.cache_scope == "session" and session_id is not None:
                if cached_item.get("session_id") != session_id:
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
    async def _triage_query_async(self, query: str, history: list) -> str:
        
        print(f"🔍 ROUTING DEBUG: Starting triage for query: '{query[:100]}...'")
        
        # 1️⃣ Fast routing using precomputed overviews (if available)
        print(f"📖 ROUTING DEBUG: Attempting overview-based routing...")
        routed = self._route_via_overviews(query)
        if routed:
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

3. "graph_query" – Specific factual relations for knowledge-graph lookup (currently limited use)

IMPORTANT: For general world knowledge about well-known companies, people, or facts NOT related to uploaded documents, choose "direct_answer".

User query: "{query}"

Respond with JSON: {{"category": "<your_choice>"}}
"""
        resp = self.llm_client.generate_completion(
            model=self.ollama_config["generation_model"], prompt=prompt, format="json"
        )
        try:
            data = json.loads(resp.get("response", "{}"))
            decision = data.get("category", "rag_query")
            print(f"🤖 ROUTING DEBUG: LLM fallback triage decided: '{decision}'")
            return decision
        except json.JSONDecodeError:
            print(f"❌ ROUTING DEBUG: LLM fallback triage JSON parsing failed, defaulting to 'rag_query'")
            return "rag_query"

    def _run_graph_query(self, query: str, history: list) -> Dict[str, Any]:
        contextual_query = self._format_query_with_history(query, history)
        structured_query = self.graph_query_translator.translate(contextual_query)
        if not structured_query.get("start_node"):
            return self.retrieval_pipeline.run(contextual_query, window_size_override=0)
        results = self.graph_retriever.retrieve(structured_query)
        if not results:
            return self.retrieval_pipeline.run(contextual_query, window_size_override=0)
        answer = ", ".join([res['details']['node_id'] for res in results])
        return {"answer": f"From the knowledge graph: {answer}", "source_documents": results}

    def _get_cache_key(self, query: str, query_type: str) -> str:
        """Generate a cache key for the query"""
        # Simple cache key based on query and type
        return f"{query_type}:{query.strip().lower()}"
    
    def _cache_result(self, cache_key: str, result: Dict[str, Any], session_id: Optional[str] = None):
        """Cache a result with size limit"""
        if len(self._query_cache) >= self._cache_max_size:
            # Remove oldest entry (simple FIFO eviction)
            oldest_key = next(iter(self._query_cache))
            del self._query_cache[oldest_key]
        
        self._query_cache[cache_key] = {
            'result': result,
            'timestamp': time.time(),
            'session_id': session_id
        }

    # ---------------- Public sync API (kept for backwards compatibility) --------------
    def run(self, query: str, table_name: str = None, session_id: str = None, compose_sub_answers: Optional[bool] = None, query_decompose: Optional[bool] = None, ai_rerank: Optional[bool] = None, context_expand: Optional[bool] = None, verify: Optional[bool] = None, retrieval_k: Optional[int] = None, context_window_size: Optional[int] = None, reranker_top_k: Optional[int] = None, search_type: Optional[str] = None, dense_weight: Optional[float] = None, max_retries: int = 1, event_callback: Optional[callable] = None) -> Dict[str, Any]:
        """Synchronous helper. If *event_callback* is supplied, important
        milestones will be forwarded to that callable as

            event_callback(phase:str, payload:Any)
        """
        return asyncio.run(self._run_async(query, table_name, session_id, compose_sub_answers, query_decompose, ai_rerank, context_expand, verify, retrieval_k, context_window_size, reranker_top_k, search_type, dense_weight, max_retries, event_callback))

    # ---------------- Main async implementation --------------------------------------
    async def _run_async(self, query: str, table_name: str = None, session_id: str = None, compose_sub_answers: Optional[bool] = None, query_decompose: Optional[bool] = None, ai_rerank: Optional[bool] = None, context_expand: Optional[bool] = None, verify: Optional[bool] = None, retrieval_k: Optional[int] = None, context_window_size: Optional[int] = None, reranker_top_k: Optional[int] = None, search_type: Optional[str] = None, dense_weight: Optional[float] = None, max_retries: int = 1, event_callback: Optional[callable] = None) -> Dict[str, Any]:
        start_time = time.time()
        
        # Emit analyze event at the start
        if event_callback:
            event_callback("analyze", {"query": query})
        
        # 🚀 NEW: Get conversation history
        history = self.chat_histories.get(session_id, []) if session_id else []
        
        # 🔄 Refresh overviews for this session if available
        # if session_id and session_id != getattr(self, "_current_overview_session", None):
        #     candidate_path = os.path.join("index_store", "overviews", f"{session_id}.jsonl")
        #     if os.path.exists(candidate_path):
        #         self._load_overviews(candidate_path)
        #         self._current_overview_session = session_id
        #     else:
        #         # Fall back to global overviews if per-session file not found
        #         if self._current_overview_session != "GLOBAL":
        #             self._load_overviews(self._global_overview_path)
        #             self._current_overview_session = "GLOBAL"
        
        query_type = await self._triage_query_async(query, history)
        print(f"🎯 ROUTING DEBUG: Final triage decision: '{query_type}'")
        print(f"Agent Triage Decision: '{query_type}'")
        
        # Create a contextual query that includes history for most operations
        contextual_query = self._format_query_with_history(query, history)
        raw_query = query.strip()
        
        # --- Apply runtime AI reranker override (must happen before any retrieval calls) ---
        if ai_rerank is not None:
            rr_cfg = self.retrieval_pipeline.config.setdefault("reranker", {})
            rr_cfg["enabled"] = bool(ai_rerank)
            if ai_rerank:
                # Ensure the pipeline knows to use the external ColBERT reranker
                rr_cfg.setdefault("type", "ai")
                rr_cfg.setdefault("strategy", "rerankers-lib")
                rr_cfg.setdefault(
                    "model_name",
                    # Falls back to ColBERT-small if the caller did not supply one
                    self.ollama_config.get("rerank_model", "answerai-colbert-small-v1"),
                )

        # --- Apply runtime retrieval configuration overrides ---
        if retrieval_k is not None:
            self.retrieval_pipeline.config["retrieval_k"] = retrieval_k
            print(f"🔍 Retrieval K set to: {retrieval_k}")
            
        if context_window_size is not None:
            self.retrieval_pipeline.config["context_window_size"] = context_window_size
            print(f"🔍 Context window size set to: {context_window_size}")
            
        if reranker_top_k is not None:
            rr_cfg = self.retrieval_pipeline.config.setdefault("reranker", {})
            rr_cfg["top_k"] = reranker_top_k
            print(f"🔍 Reranker top K set to: {reranker_top_k}")
            
        if search_type is not None:
            retrieval_cfg = self.retrieval_pipeline.config.setdefault("retrieval", {})
            retrieval_cfg["search_type"] = search_type
            print(f"🔍 Search type set to: {search_type}")
            
        if dense_weight is not None:
            dense_cfg = self.retrieval_pipeline.config.setdefault("retrieval", {}).setdefault("dense", {})
            dense_cfg["weight"] = dense_weight
            print(f"🔍 Dense search weight set to: {dense_weight}")

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

                cached_result = self._find_in_semantic_cache(query_embedding, session_id)

                if cached_result:
                    # Update history even on cache hit
                    if session_id:
                        history.append({"query": query, "answer": cached_result.get('answer', 'Cached answer not found.')})
                        self.chat_histories[session_id] = history
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
                    for tok in self.llm_client.stream_completion(
                        model=self.ollama_config["generation_model"], prompt=prompt
                    ):
                        answer_parts.append(tok)
                        if event_callback:
                            event_callback("token", {"text": tok})

                # Run the blocking generator in a thread so the event loop stays responsive
                await asyncio.to_thread(_blocking_stream)
                return "".join(answer_parts)

            final_answer = await _run_stream()
            result = {"answer": final_answer, "source_documents": []}
        
        elif query_type == "graph_query" and hasattr(self, 'graph_retriever'):
            print(f"✅ ROUTING DEBUG: Executing GRAPH_QUERY path")
            result = self._run_graph_query(query, history)

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
                sub_queries = self.query_decomposer.decompose(raw_query, recent_history)
                if event_callback:
                    event_callback("decomposition", {"sub_queries": sub_queries})
                print(f"Original query: '{query}' (Contextual: '{contextual_query}')")
                print(f"Decomposed into {len(sub_queries)} sub-queries: {sub_queries}")
                
                # Emit retrieval_started event before any retrievals
                if event_callback:
                    event_callback("retrieval_started", {"count": len(sub_queries)})
                
                # If decomposition produced only a single sub-query, skip the
                # parallel/composition machinery for efficiency.
                if len(sub_queries) == 1:
                    print("--- Only one sub-query after decomposition; using direct retrieval path ---")
                    result = self.retrieval_pipeline.run(
                        sub_queries[0],
                        table_name,
                        0 if context_expand is False else None,
                        event_callback=event_callback
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

                    print(f"\n--- Processing {len(sub_queries)} sub-queries in parallel ---")
                    start_time_inner = time.time()

                    # Shared containers
                    sub_answers = []  # For two-stage composition
                    all_source_docs = []  # For single-stage aggregation
                    citations_seen = set()

                    # Emit rerank_started event before parallel retrievals (since each sub-query will rerank)
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
                        future_to_query = {
                            executor.submit(
                                self.retrieval_pipeline.run,
                                sub_query,
                                table_name,
                                0 if context_expand is False else None,
                                make_cb(i),
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

                                if compose_from_sub_answers:
                                    sub_answers.append({
                                        "question": sub_query,
                                        "answer": sub_result.get("answer", "")
                                    })
                                    # Keep up to 5 citations per sub-query for traceability
                                    for doc in sub_result.get("source_documents", [])[:5]:
                                        if doc['chunk_id'] not in citations_seen:
                                            all_source_docs.append(doc)
                                            citations_seen.add(doc['chunk_id'])
                                else:
                                    # Aggregate unique docs (single-stage path)
                                    for doc in sub_result.get('source_documents', []):
                                        if doc['chunk_id'] not in citations_seen:
                                            all_source_docs.append(doc)
                                            citations_seen.add(doc['chunk_id'])
                            except Exception as e:
                                print(f"❌ Sub-Query {i+1} failed: '{sub_query}' - {e}")

                    parallel_time = time.time() - start_time_inner
                    print(f"🚀 Parallel processing completed in {parallel_time:.2f}s")

                    # Emit retrieval_done and rerank_done after all sub-queries are processed
                    if event_callback:
                        event_callback("retrieval_done", {"count": len(sub_queries)})
                        event_callback("rerank_done", {"count": len(sub_queries)})

                    if compose_from_sub_answers:
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

                        for tok in self.llm_client.stream_completion(
                            model=self.ollama_config["generation_model"],
                            prompt=compose_prompt,
                        ):
                            answer_parts.append(tok)
                            if event_callback:
                                event_callback("token", {"text": tok})

                        final_answer = "".join(answer_parts) or "Unable to generate an answer."

                        result = {
                            "answer": final_answer,
                            "source_documents": all_source_docs
                        }
                        if event_callback:
                            event_callback("final_answer", result)
                    else:
                        print(f"\n--- Aggregated {len(all_source_docs)} unique documents from all sub-queries ---")

                        if all_source_docs:
                            aggregated_context = "\n\n".join([doc['text'] for doc in all_source_docs])
                            final_answer = self.retrieval_pipeline._synthesize_final_answer(contextual_query, aggregated_context)
                            result = {
                                "answer": final_answer,
                                "source_documents": all_source_docs
                            }
                            if event_callback:
                                event_callback("final_answer", result)
                        else:
                            result = {
                                "answer": "I could not find relevant information to answer your question.",
                                "source_documents": []
                            }
                            if event_callback:
                                event_callback("final_answer", result)
            else:
                # Standard retrieval (single-query)
                retrieved_docs = (self.retrieval_pipeline.retriever.retrieve(
                    text_query=contextual_query,
                    table_name=table_name or self.retrieval_pipeline.storage_config["text_table_name"],
                    k=self.retrieval_pipeline.config.get("retrieval_k", 10),
                ) if hasattr(self.retrieval_pipeline, "retriever") and self.retrieval_pipeline.retriever else [])

                print("\n=== DEBUG: Original retrieval order ===")
                for i, d in enumerate(retrieved_docs[:10]):
                    snippet = (d.get('text','') or '')[:200].replace('\n',' ')
                    print(f"Orig[{i}] id={d.get('chunk_id')} dist={d.get('_distance','') or d.get('score','')}  {snippet}")

                result = self.retrieval_pipeline.run(contextual_query, table_name, 0 if context_expand is False else None, event_callback=event_callback)

                # After run, result['source_documents'] is reranked list
                reranked_docs = result.get('source_documents', [])
                print("\n=== DEBUG: Reranked docs order ===")
                for i, d in enumerate(reranked_docs[:10]):
                    snippet = (d.get('text','') or '')[:200].replace('\n',' ')
                    print(f"ReRank[{i}] id={d.get('chunk_id')} score={d.get('rerank_score','')} {snippet}")
        
        # Verification step (simplified for now) - Skip in fast mode
        verification_enabled = self.pipeline_configs.get("verification", {}).get("enabled", True)
        if verify is not None:
            verification_enabled = verify
            
        if verification_enabled and result.get("source_documents"):
            context_str = "\n".join([doc['text'] for doc in result['source_documents']])
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
        
        # 🚀 NEW: Update history
        if session_id:
            history.append({"query": query, "answer": result['answer']})
            self.chat_histories[session_id] = history
            
        # 🚀 OPTIMIZED: Cache the result for future queries
        if query_type != "direct_answer" and query_embedding is not None:
            cache_key = raw_query  # Key is for logging/debugging
            self._query_cache[cache_key] = {
                "embedding": query_embedding,
                "result": result,
                "session_id": session_id,
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

Documents available: Invoices, DeepSeek-V3 research papers

Query: "{query}"

Is this query asking about:
A) Greetings/social: "Hi", "Hello", "Thanks", "What's up", "How are you"
B) General knowledge: "CEO of Tesla", "capital of France", "what is 2+2"  
C) Document content: invoice amounts, DeepSeek-V3 details, companies mentioned

If A or B → {{"category": "direct_answer"}}
If C → {{"category": "rag_query"}}

Response:"""
        
        resp = self.llm_client.generate_completion(
            model=self.ollama_config["generation_model"], prompt=router_prompt, format="json"
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

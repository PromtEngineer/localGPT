"""Transport-neutral execution for LocalGPT RAG chat requests."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from rag_system.agent import reflection
from rag_system.index_selection import select_active_index_id
from rag_system.utils.logging_utils import StageTimings, timings_enabled

EventCallback = Optional[Callable[[str, Any], None]]


def _collection_from_index(index: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    table_name = index.get("vector_table_name")
    if not table_name:
        return None
    metadata = index.get("metadata") or {}
    return {
        "index_id": index.get("id"),
        "table_name": table_name,
        "embedding_model": metadata.get("embedding_model"),
        "index_name": index.get("name"),
        "metadata_schema": metadata.get("metadata_schema"),
        "fusion_config": metadata.get("fusion_config"),
    }


def _resolve_targets(db, session_id: Optional[str], table_name: Optional[str]):
    if table_name:
        for index in db.list_indexes():
            if index.get("vector_table_name") == table_name:
                collection = _collection_from_index(index)
                return table_name, [collection] if collection else None
        return table_name, None

    if not session_id:
        return None, None

    index_ids = db.get_indexes_for_session(session_id)
    collections = []
    for index_id in index_ids[-5:]:
        index = db.get_index(index_id)
        if index:
            collection = _collection_from_index(index)
            if collection:
                collections.append(collection)

    active_id = select_active_index_id(index_ids)
    active_index = db.get_index(active_id) if active_id else None
    active_table = active_index.get("vector_table_name") if active_index else None
    return active_table, collections or None


def execute_chat(agent, db, data: Dict[str, Any], event_callback: EventCallback = None):
    """Execute one request without mutating shared agent or pipeline config."""
    query = data.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query is required")

    session_id = data.get("session_id")
    table_name, collections = _resolve_targets(db, session_id, data.get("table_name"))
    generation_model = (
        data.get("model")
        if isinstance(data.get("model"), str) and data.get("model")
        else agent.ollama_config["generation_model"]
    )

    document_overviews = []
    if session_id:
        document_overviews = agent.get_overviews_for_indexes(
            db.get_indexes_for_session(session_id)
        )

    retrieval_k = data.get("retrieval_k", 20)
    context_window_size = data.get("context_window_size", 1)
    reranker_top_k = data.get("reranker_top_k", 10)
    search_type = data.get("search_type", "hybrid")
    dense_weight = data.get("dense_weight")
    provence_prune = data.get("provence_prune")
    provence_threshold = data.get("provence_threshold")
    ai_rerank = data.get("ai_rerank")

    # Opt-in per-stage timing. The observer wraps the event stream (so it only
    # adds detail on the streaming path, never flipping a non-streaming request
    # into one); total latency is captured regardless.
    timer = StageTimings() if timings_enabled() else None
    callback = event_callback
    if timer is not None and event_callback is not None:

        def _timed_callback(
            event_type: str, payload: Any, _t=timer, _orig=event_callback
        ) -> None:
            _t.observe(event_type, payload)
            _orig(event_type, payload)

        callback = _timed_callback

    # Two-axis self-reflection is opt-in per request. It runs on the retrieval
    # pipeline directly (like force_rag), so enabling it implies that path.
    reflect_cfg = reflection.parse_config(data, generation_model)
    verifier = getattr(agent, "verifier", None)
    reflect_on = reflect_cfg["enabled"] and verifier is not None

    force_rag = bool(data.get("force_rag", False))
    if force_rag or reflect_on:
        overrides = {
            "retrieval_k": retrieval_k,
            "reranker_top_k": reranker_top_k,
            "search_type": search_type,
            "dense_weight": dense_weight,
            "ai_rerank": ai_rerank,
            "provence_enabled": provence_prune,
            "provence_threshold": provence_threshold,
            "generation_model": generation_model,
            "latechunk_enabled": True,
        }
        run_kwargs = {
            "table_name": table_name,
            "window_size_override": context_window_size,
            "collections": collections,
            "filters": (
                data.get("filters") if isinstance(data.get("filters"), dict) else None
            ),
            "overrides": overrides,
        }
        if reflect_on:
            result = reflection.reflective_run(
                agent.retrieval_pipeline,
                verifier,
                query,
                run_kwargs=run_kwargs,
                event_callback=callback,
                cfg=reflect_cfg,
            )
        else:
            result = agent.retrieval_pipeline.run(
                query, event_callback=callback, **run_kwargs
            )
    else:
        result = agent.run(
            query,
            table_name=table_name,
            collections=collections,
            filters=(
                data.get("filters") if isinstance(data.get("filters"), dict) else None
            ),
            session_id=session_id,
            compose_sub_answers=data.get("compose_sub_answers"),
            query_decompose=data.get("query_decompose"),
            ai_rerank=ai_rerank,
            context_expand=data.get("context_expand"),
            verify=data.get("verify"),
            retrieval_k=retrieval_k,
            context_window_size=context_window_size,
            reranker_top_k=reranker_top_k,
            search_type=search_type,
            dense_weight=dense_weight,
            agentic=(
                data.get("agentic") if isinstance(data.get("agentic"), bool) else None
            ),
            generation_model=generation_model,
            document_overviews=document_overviews,
            provence_prune=provence_prune,
            provence_threshold=provence_threshold,
            latechunk_enabled=True,
            event_callback=callback,
        )

    if timer is not None and isinstance(result, dict):
        result.update(timer.as_dict())
        timer.log(
            session_id=session_id,
            table_name=table_name,
            force_rag=force_rag,
            source_count=len(result.get("source_documents") or []),
        )
    return result

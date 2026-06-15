"""Two-axis self-reflection loop for retrieval chat (opt-in, request-gated).

Inspired by the NVIDIA RAG blueprint's reflection feature, scaled to
LocalGPT's local-first stack. A bounded retry loop wraps the retrieval
pipeline's existing entry points:

  * context relevance — score the retrieved context; if it's too weak, rewrite
    the query and re-retrieve (a fresh ``pipeline.run``).
  * response groundedness — score the answer against its context; if it drifts,
    regenerate on the same context with stronger source-adherence
    (``pipeline._synthesize_final_answer``).

This is a pure orchestrator: it adds no new model infrastructure, reuses the
pipeline for retrieval/generation and a ``Verifier`` for the 0-2 scoring, and
is a no-op unless a request sets ``reflect``. Intermediate (possibly-rejected)
answers never stream to the client — only the final accepted answer does.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

EventCallback = Optional[Callable[[str, Any], None]]

# Single source of truth for the reflection defaults — also served to the UI
# via GET /rag/reflection-defaults so the frontend slider doesn't hard-code them.
REFLECTION_DEFAULTS: Dict[str, int] = {
    "max_loops": 2,
    "relevance_threshold": 1,
    "groundedness_threshold": 1,
}

# Hard ceiling on reflection rounds, regardless of the request value — each
# round is a full retrieval+generation (or synthesis) pass.
_MAX_REFLECTION_LOOPS = 5


def parse_config(data: Dict[str, Any], generation_model: Any) -> Dict[str, Any]:
    """Parse per-request reflection knobs from a chat request payload."""

    def _int(key: str, default: int) -> int:
        value = data.get(key)
        # bool is an int subclass — exclude it so reflect=True isn't read as 1
        if isinstance(value, bool) or not isinstance(value, int):
            return default
        return value

    judge = data.get("reflection_model")
    return {
        "enabled": bool(data.get("reflect", False)),
        # Clamp to a hard ceiling: each loop is a full retrieval+generation (or a
        # synthesis) call, so an unbounded request value is a resource-exhaustion
        # lever.
        "max_loops": max(
            1,
            min(
                _MAX_REFLECTION_LOOPS,
                _int("reflection_max_loops", REFLECTION_DEFAULTS["max_loops"]),
            ),
        ),
        "relevance_threshold": _int(
            "relevance_threshold", REFLECTION_DEFAULTS["relevance_threshold"]
        ),
        "groundedness_threshold": _int(
            "groundedness_threshold", REFLECTION_DEFAULTS["groundedness_threshold"]
        ),
        # "model" judges (scores + rewrites) — point reflection_model at a small
        # fast model to cut latency; "generation_model" always regenerates the
        # answer, so a fast judge never degrades answer quality.
        "model": judge if isinstance(judge, str) and judge else generation_model,
        "generation_model": generation_model,
    }


def _context_from_sources(sources: List[Dict[str, Any]]) -> str:
    """Rebuild a labelled context string from a result's source documents."""
    parts = []
    for i, doc in enumerate(sources, start=1):
        name = doc.get("document_id") or "source"
        if doc.get("index_name"):
            name = f"{doc['index_name']} — {name}"
        parts.append(f"[Source {i}: {name}]\n{doc.get('text', '')}")
    return "\n\n".join(parts)


def _budget_context(sources: List[Dict[str, Any]]) -> str:
    """Build a synthesis context trimmed to the model's window.

    The pipeline's run() budgets its context before generating; the
    groundedness-regeneration path must do the same or a large source set
    overflows num_ctx and the model can return an empty/garbled answer.
    """
    from rag_system.utils.ollama_client import NUM_CTX

    max_chars = (NUM_CTX - 2500) * 4  # ≈4 chars/token, reserve instructions+answer
    kept: List[Dict[str, Any]] = []
    used = 0
    for doc in sources:
        used += len(doc.get("text", "")) + 60
        if used > max_chars and kept:
            break
        kept.append(doc)
    return _context_from_sources(kept)


def _has_answer(result: Dict[str, Any]) -> bool:
    return bool((result.get("answer") or "").strip())


def _suppress_tokens(callback: EventCallback) -> EventCallback:
    """Forward stage events but swallow token events.

    During reflection rounds the pipeline still emits its retrieval/rerank/prune
    stage events (so progress UI and timing keep working) but candidate answers
    must not stream — only the final accepted answer is emitted, by the caller.
    """
    if callback is None:
        return None

    def wrapped(event_type: str, payload: Any) -> None:
        if event_type == "token":
            return
        callback(event_type, payload)

    return wrapped


def _rewrite_query(llm_client: Any, model: str, query: str, context: str) -> str:
    """Broaden the query after weak retrieval; falls back to the original."""
    prompt = (
        "The context retrieved for the QUESTION was judged not relevant enough. "
        "Rewrite the QUESTION as a single, broader search query more likely to "
        "retrieve relevant passages — keep the key entities and topic, drop rare "
        'phrasings. Reply with JSON only: {"query": "..."}.\n\n'
        f"QUESTION: {query}\n\nWEAK CONTEXT:\n{context[:2000]}"
    )
    try:
        resp = llm_client.generate_completion(
            model, prompt, format="json", temperature=0.0, enable_thinking=False
        )
        rewritten = json.loads(resp.get("response", "{}")).get("query")
        if isinstance(rewritten, str) and rewritten.strip():
            return rewritten
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass
    return query


def _adherence_wrap(query: str) -> str:
    """Strengthen source-adherence for a groundedness regeneration."""
    return (
        f"{query}\n\n[Answer using ONLY the provided sources. Do not add facts "
        "that are not present in the sources. If the sources do not contain the "
        "answer, say so explicitly.]"
    )


def _emit_answer(callback: EventCallback, answer: str) -> None:
    """Stream the final accepted answer to the client in modest chunks."""
    if not callback or not answer:
        return
    step = 280
    for start in range(0, len(answer), step):
        callback("token", {"text": answer[start : start + step]})


def reflective_run(
    pipeline: Any,
    verifier: Any,
    query: str,
    *,
    run_kwargs: Dict[str, Any],
    event_callback: EventCallback,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Run retrieval + generation under the two-axis reflection loop.

    Returns the usual ``{"answer", "source_documents"}`` dict plus a
    ``"reflection"`` block recording rounds and the last scores.
    """
    model = cfg["model"]  # judge: scoring + query rewrite
    gen_model = cfg.get("generation_model") or model  # regeneration (quality)
    rel_threshold = cfg["relevance_threshold"]
    ground_threshold = cfg["groundedness_threshold"]
    max_loops = cfg["max_loops"]

    stage_cb = _suppress_tokens(event_callback)
    current_query = query
    result = pipeline.run(current_query, event_callback=stage_cb, **run_kwargs)
    # Best non-empty answer seen so far. Reflection scoring on small local
    # models is noisy and a regeneration can come back empty (thinking models
    # occasionally emit no final text), so we never let the loop downgrade a
    # good answer to an empty one.
    best = result if _has_answer(result) else None

    rounds = 0
    relevance: Optional[int] = None
    groundedness: Optional[int] = None
    converged = False

    for _ in range(max_loops):
        sources = result.get("source_documents") or []
        if not sources:
            break
        context = _context_from_sources(sources)
        relevance = verifier.score_context_relevance(query, context, model)
        groundedness = verifier.score_response_groundedness(
            query, context, result.get("answer", ""), model
        )
        if relevance >= rel_threshold and groundedness >= ground_threshold:
            converged = True
            break
        rounds += 1
        if relevance < rel_threshold:
            # Weak context → broaden the query and retrieve again.
            current_query = _rewrite_query(
                pipeline.ollama_client, model, query, context
            )
            result = pipeline.run(current_query, event_callback=stage_cb, **run_kwargs)
        else:
            # Context is fine but the answer drifts → regenerate on the same
            # context (budgeted to the window) with stronger adherence.
            # stage_cb (not None): suppresses the regeneration's tokens but lets
            # its generation_started/done through so its latency is still timed.
            answer = pipeline._synthesize_final_answer(
                _adherence_wrap(query),
                _budget_context(sources),
                event_callback=stage_cb,
                generation_model=gen_model,
            )
            result = {"answer": answer, "source_documents": sources}
        if _has_answer(result):
            best = result

    # Prefer the loop's final result, but fall back to the best non-empty
    # answer rather than ever returning an empty one.
    final = result if _has_answer(result) else (best or result)
    _emit_answer(event_callback, final.get("answer", ""))
    # converged=False means max_loops was hit: the scores describe the
    # last-evaluated answer, which may predate a final groundedness regeneration.
    final["reflection"] = {
        "rounds": rounds,
        "relevance": relevance,
        "groundedness": groundedness,
        "converged": converged,
    }
    return final

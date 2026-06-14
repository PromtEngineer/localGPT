"""Condense a multi-turn follow-up into a standalone retrieval query.

When a session has prior turns, the latest user message is often a follow-up
("what about its limits?") that retrieves poorly verbatim. Rewriting it into a
self-contained query — resolving pronouns/references against the recent
history — improves first-pass retrieval (and, with reflection on, cuts the
rounds needed to reach a grounded answer).

Opt-in per request (``rewrite_query``). It falls back to the original query
whenever there is no history or anything goes wrong, so it can only help
retrieval, never break it.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List


def messages_to_turns(
    messages: List[Dict[str, Any]], max_turns: int = 5
) -> List[Dict[str, str]]:
    """Pair persisted user/assistant messages (chronological) into recent turns."""
    turns: List[Dict[str, str]] = []
    pending_user = None
    for msg in messages:
        sender = msg.get("sender")
        content = (msg.get("content") or "").strip()
        if sender == "user":
            pending_user = content
        elif sender == "assistant" and pending_user is not None:
            turns.append({"user": pending_user, "assistant": content})
            pending_user = None
    return turns[-max_turns:]


def _format_history(turns: List[Dict[str, str]]) -> str:
    return "\n".join(f"User: {t['user']}\nAssistant: {t['assistant']}" for t in turns)


def standalone_query(
    llm_client: Any, model: Any, query: str, turns: List[Dict[str, str]]
) -> str:
    """Rewrite the latest query into a self-contained one using recent turns.

    Returns the original query unchanged when there is no history or on any
    LLM/parse failure.
    """
    if not turns:
        return query
    prompt = (
        "Given the conversation history, rewrite the user's LATEST message into a "
        "single, self-contained search query that stands on its own without the "
        "history: resolve pronouns and references to concrete entities, stay "
        "concise, and do NOT answer it. If it is already self-contained, return it "
        'unchanged. Reply with JSON only: {"query": "..."}.\n\n'
        f"HISTORY:\n{_format_history(turns)}\n\nLATEST MESSAGE: {query}"
    )
    try:
        resp = llm_client.generate_completion(model, prompt, format="json")
        rewritten = json.loads(resp.get("response", "{}")).get("query")
        if isinstance(rewritten, str) and rewritten.strip():
            return rewritten.strip()
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass
    return query

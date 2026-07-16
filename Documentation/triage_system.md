# Triage and Routing

Routing has one owner: `rag_system/agent/loop.py`. The backend no longer runs a separate regex/LLM router; it delegates every session query to the RAG agent so Ollama and WatsonX follow the same path.

## Decisions

- `direct_answer`: greetings, conversational follow-ups, and general-knowledge questions that do not need private documents.
- `rag_query`: questions related to linked document overviews or explicit document operations such as summarize, extract, compare, or find.
- `graph_query`: only available when graph retrieval is explicitly configured; disabled by default.

## Signals and fallback

1. Persisted user/assistant history is restored into the agent for each backend request.
2. Overviews from every linked index are loaded.
3. `_route_via_overviews()` sends the actual overview text and query to the configured generation provider.
4. The broader triage prompt considers query semantics and history.
5. Router parse failures and ambiguous document-related queries default to RAG, which avoids answering a private-document question from general model knowledge.

The old hard-coded “Invoices, DeepSeek-V3 research papers” prompt, backend fast router, `PIPELINE_CONFIGS.triage.enabled`, and `TRIAGE_OVERVIEW_THRESHOLD` are not part of the active implementation.

Routing results are returned as the `route` field and drive the backend's `used_rag` response field.

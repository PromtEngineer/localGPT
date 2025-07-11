# 📜 Prompt Inventory (Ground-Truth)

_All generation / verification prompts currently hard-coded in the codebase._  
_Last updated: 2025-07-06_

> Edit process: if you change a prompt in code, please **update this file** or, once we migrate to the central registry, delete the entry here.

---

## 1. Indexing / Context Enrichment

| ID | File & Lines | Variable / Builder | Purpose |
|----|--------------|--------------------|---------|
| `overview_builder.default` | `rag_system/indexing/overview_builder.py` `12-21` | `DEFAULT_PROMPT` | Generate 1-paragraph document overview for search-time routing.
| `contextualizer.system` | `rag_system/indexing/contextualizer.py` `11` | `SYSTEM_PROMPT` | System instruction: explain summarisation role.
| `contextualizer.local_context` | same file `13-15` | `LOCAL_CONTEXT_PROMPT_TEMPLATE` | Human message – wraps neighbouring chunks.
| `contextualizer.chunk` | same file `17-19` | `CHUNK_PROMPT_TEMPLATE` | Human message – shows the target chunk.
| `graph_extractor.entities` | `rag_system/indexing/graph_extractor.py` `20-31` | `entity_prompt` | Ask LLM to list entities.
| `graph_extractor.relationships` | same file `53-64` | `relationship_prompt` | Ask LLM to list relationships.

## 2. Retrieval / Query Transformation

| ID | File & Lines | Purpose |
|----|--------------|---------|
| `query_transformer.expand` | `rag_system/retrieval/query_transformer.py` `10-26` | Produce query rewrites (keywords, boolean). |
| `hyde.hypothetical_doc` | same `115-122` | HyDE hypothetical document generator. |
| `graph_query.translate` | same `124-140` | Translate user question to JSON KG query. |

## 3. Pipeline Answer Synthesis

| ID | File & Lines | Purpose |
|----|--------------|---------|
| `retrieval_pipeline.synth_final` | `rag_system/pipelines/retrieval_pipeline.py` `217-256` | Turn verified facts into answer (with directives 1-6). |

## 4. Agent – Classical Loop

| ID | File & Lines | Purpose |
|----|--------------|---------|
| `agent.loop.initial_thought` | `rag_system/agent/loop.py` `157-180` | First LLM call to think about query. |
| `agent.loop.verify_path` | same `190-205` | Secondary thought loop. |
| `agent.loop.compose_sub` | same `506-542` | Compose answer from sub-answers. |
| `agent.loop.router` | same `648-660` | Decide which subsystem handles query. |

## 5. Verifier

| ID | File & Lines | Purpose |
|----|--------------|---------|
| `verifier.fact_check` | `rag_system/agent/verifier.py` `18-58` | Strict JSON-format grounding verifier. |

## 6. Backend Router (Fast path)

| ID | File & Lines | Purpose |
|----|--------------|---------|
| `backend.router` | `backend/server.py` `435-448` | Decide "RAG vs direct LLM" before heavy processing. |

## 7. Miscellaneous

| ID | File & Lines | Purpose |
|----|--------------|---------|
| `vision.placeholder` | `rag_system/utils/ollama_client.py` `169` | Dummy prompt for VLM colour check. |

---

### Missing / To-Do
1. Verify whether **ReActAgent.PROMPT_TEMPLATE** captures every placeholder – some earlier lines may need explicit ID when we move to central registry.
2. Search TS/JS code once the backend prompts are ported (currently none).

---

**Next step:** create `rag_system/prompts/registry.yaml` and start moving each prompt above into a key–value entry with identical IDs. Update callers gradually using the helper proposed earlier. 
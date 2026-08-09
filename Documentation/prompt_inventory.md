# 📜 Prompt Inventory (Ground-Truth)

_Every prompt hard-coded in the codebase, re-derived from the current source._

> Edit process: if you change a prompt in code, update the line range here in the same commit.

## Which model runs which prompt

There are two model roles (`rag_system/main.py` `OLLAMA_CONFIG`):

| Role | Config key | Default | Env override |
|------|-----------|---------|--------------|
| Generation — user-facing answers | `generation_model` | `qwen3.5:9b` | `GENERATION_MODEL` |
| Utility — routing, triage, decomposition, enrichment, overviews, verification | `enrichment_model` | `qwen3.5:4b` | `ENRICHMENT_MODEL` |

Inside the agent the utility model is resolved by `Agent._utility_model()` (`rag_system/agent/loop.py:56-58`), which returns `ollama_config["enrichment_model"]` and falls back to `generation_model` when that key is absent. No prompt hard-codes a model name.

---

## 1. Indexing / context enrichment

| ID | File & lines | Variable / builder | Model | Purpose |
|----|--------------|--------------------|-------|---------|
| `overview_builder.default` | `rag_system/indexing/overview_builder.py` `13-19` | `OverviewBuilder.DEFAULT_PROMPT` | overview model (resolved at `indexing_pipeline.py:128-133`; utility model by default) | One-paragraph document overview used by the triage routers. Input is the first `first_n_chunks` chunks (default 5), truncated to 5000 characters at `overview_builder.py:37`. |
| `contextualizer.system` | `rag_system/indexing/contextualizer.py` `12` | `SYSTEM_PROMPT` | enrichment model | Role instruction for the summariser. |
| `contextualizer.local_context` | same file `14-16` | `LOCAL_CONTEXT_PROMPT_TEMPLATE` | — | Wraps the neighbouring-chunk window in `<local_context>` tags. |
| `contextualizer.chunk` | same file `18-26` | `CHUNK_PROMPT_TEMPLATE` | — | Shows the target chunk and carries the actual instruction: a 2-5 sentence context summary, "Answer *only* with the succinct context and nothing else." |

The three contextualizer parts are concatenated into a single `/api/generate` completion prompt at `contextualizer.py:42-51` (no chat roles) and sent at `contextualizer.py:53` with `enable_thinking=False`.

## 2. Retrieval / query transformation

| ID | File & lines | Model | Purpose |
|----|--------------|-------|---------|
| `query_transformer.decompose` | `rag_system/retrieval/query_transformer.py` `38-84` (system) + `87-196` (few-shot examples) + `250-265` (assembly) | utility model (`loop.py:30`) | Resolve pronouns/ellipsis against the last 5 turns, then split the query into standalone sub-queries. Returns RFC-8259 JSON `{requires_decomposition, reasoning, resolved_query, sub_queries}`; sent with `format="json"` at `:268`; the list is deduplicated and capped by `max_sub_queries` at `:290` (default 10). |

| `retrieval.retry.reformulate` | `rag_system/pipelines/retrieval_pipeline.py`, `_reformulate_query` | enrichment/utility model, `format="json"` | Fires **only** when the evidence-sufficiency retry triggers (roadmap 2.1). Asks for one rewrite of a weak-evidence query into the concrete nouns and synonyms a document would use, preserving every entity and constraint. Returns `{"query": "…"}`; `format="json"` is what keeps a small model's thinking preamble out of the rewritten query. |

Two legacy example blocks still live in the file at `199-203` and `205-248`, but they are excluded from the assembled prompt — the two lines that would concatenate them are commented out at `:253-254`.

## 3. Answer synthesis

| ID | File & lines | Model | Purpose |
|----|--------------|-------|---------|
| `retrieval_pipeline.synth_final` | `rag_system/pipelines/retrieval_pipeline.py` `225-250` | generation model | Turn the retrieved snippets into the final answer (6 numbered directives; instructs the model to reply exactly "I could not find that information in the provided documents." when the snippets do not cover the question). Streamed token-by-token via `stream_completion` at `:253-256`; each token is forwarded to the `event_callback` as a `token` event. |

## 4. Agent loop (`rag_system/agent/loop.py`)

| ID | Lines | Model | Purpose |
|----|-------|-------|---------|
| `agent.loop.history_wrapper` | `163-171` | none — template only | `_format_query_with_history` builds the `contextual_query` string that is embedded into downstream prompts. It makes no LLM call. |
| `agent.loop.overview_router` | `615-630` | utility model (call at `632-634`, `format="json"`) | First routing pass. Interpolates the loaded document overviews (first 40, `:612-613`) under a `DOCUMENT OVERVIEWS:` header and returns `{"category": "direct_answer"}` or `{"category": "rag_query"}`. |
| `agent.loop.triage_fallback` | `agent/loop.py`, `_triage_query_async` | utility model, `format="json"` | Last-resort routing. Reached only when the overview router returns `None` (no overviews loaded) **and** there is no chat history. Two-way vocabulary: `rag_query` / `direct_answer`; `_normalize_triage()` maps anything else to `rag_query`. |
| `agent.loop.direct_answer` | `331-336` | generation model (streamed at `342-344`) | Answers on the `direct_answer` route from conversation history or general knowledge. Caps the reply at 1-2 sentences. |
| `agent.loop.compose_sub` | `490-515` | generation model (streamed at `519-522`) | Compose one final answer from the JSON list of sub-question/sub-answer pairs. Used when `query_decomposition.compose_from_sub_answers` is true and decomposition produced more than one sub-query. |

## 5. Verifier

| ID | File & lines | Model | Purpose |
|----|--------------|-------|---------|
| `verifier.fact_check` | `rag_system/agent/verifier.py` `25-85` | utility model (`loop.py:29`) | Grounding check with three few-shot examples and a `# TASK` block. The prompt is built in four appends: the base f-string ends at `:75`, the context is appended clamped to 4000 characters at `:76`, the answer at `:81`, and the `<OUTPUT>` tag at `:82-85`. Sent asynchronously with `format="json"`. Verdict labels: `SUPPORTED` / `NOT_SUPPORTED` / `NEEDS_CLARIFICATION`. **Skipped entirely when `VERIFIER_MODEL` / `verification.model` names a local NLI verifier** — that backend makes no LLM call at all (roadmap 2.4, `verifier.md`). |

## 6. Backend router (fast path)

| ID | File & lines | Model | Purpose |
|----|--------------|-------|---------|
| `backend.router` | `backend/server.py` `534-560` | `ENRICHMENT_MODEL` (call at `:564-568`, `enable_thinking=False`) | Decides RAG vs direct LLM inside the backend gateway before it calls the RAG API. Unlike the agent routers this one returns **plain text**, not JSON: "Respond with exactly one word: USE_RAG or DIRECT_LLM" (`:560`), substring-matched at `:574-579` with an "unclear ⇒ RAG" default at `:580-582`. |

## 7. Miscellaneous

| ID | File & lines | Purpose |
|----|--------------|---------|
| `vision.placeholder` | `rag_system/utils/ollama_client.py` `158` | `prompt="What color is this image?"` inside that module's `if __name__ == '__main__'` demo block. Not part of any pipeline. |

---

### Notes

* `rag_system/utils/watsonx_client.py:222` contains the string `prompt="What is AI?"`, but it is literal text inside a `print()` usage banner — it is never sent to a model and is therefore not inventoried.
* There is no prompt registry module; every prompt above is an inline literal at the cited location.
* There is no ReAct-style think/act/observe prompt anywhere. The agent's stages are triage → (optional) decomposition → retrieval → rerank → expand → prune → synthesis → verification.

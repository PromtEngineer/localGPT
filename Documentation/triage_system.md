# 🔀 Triage / Routing System

_Maps to `rag_system/agent/loop.Agent._should_use_rag`, `_route_using_overviews`, and the fast-path router in `backend/server.py`._

## Purpose
Determine, for every incoming query, whether it should be answered by:
1. **Direct LLM Generation** (no retrieval) — faster, cheaper.
2. **Retrieval-Augmented Generation (RAG)** — when the answer likely requires document context.

## Decision Signals
| Signal | Source | Notes |
|--------|--------|-------|
| Keyword/regex check | `backend/server.py` (fast path) | Hard-coded quick wins (`what time`, `define`, etc.). |
| Index presence | SQLite (session → indexes) | If no indexes linked, direct LLM. |
| Overview routing | `_route_using_overviews()` | Uses document overviews and enrichment model to predict relevance. |
| LLM router prompt | `agent/loop.py` lines 648-665 | Final arbitrator (Ollama call, JSON output). |

## High-level Flow
```mermaid
flowchart TD
    Q["Incoming Query"] --> S1{Session\nHas Indexes?}
    S1 -- no --> LLM["Direct LLM Generation"]
    S1 -- yes --> S2{Fast Regex\nHeuristics}
    S2 -- match--> LLM
    S2 -- no --> S3{Overview\nRelevance > τ?}
    S3 -- low --> LLM
    S3 -- high --> S4[LLM Router\n(prompt @648)]
    S4 -- "route: RAG" --> RAG["Retrieval Pipeline"]
    S4 -- "route: DIRECT" --> LLM
```

## Detailed Sequence (Code-level)
1. **backend/server.py**
   * `handle_session_chat()` builds `router_prompt` (line ~435) and makes a **first pass** decision before calling the heavy agent code.
2. **agent.loop._should_use_rag()**
   * Re-evaluates using richer features (e.g., token count, query type).
3. **Overviews Phase** (`_route_using_overviews()`)
   * Loads JSONL overviews file per index.
   * Calls enrichment model (`qwen3:0.6b`) with prompt: _"Does this overview mention … ? "_ → returns yes/no.
4. **LLM Router** (prompt lines 648-665)
   * JSON-only response `{ "route": "RAG" | "DIRECT" }`.

## Interfaces & Dependencies
| Component | Calls / Data |
|-----------|--------------|
| SQLite `chat_sessions` | Reads `indexes` column to know linked index IDs. |
| LanceDB Overviews | Reads `index_store/overviews/<idx>.jsonl`. |
| `OllamaClient` | Generates LLM router decision. |

## Config Flags
* `PIPELINE_CONFIGS.triage.enabled` – global toggle.
* Env var `TRIAGE_OVERVIEW_THRESHOLD` – min similarity score to prefer RAG (default 0.35).

## Failure / Fallback Modes
1. If overview file missing → skip to LLM router.
2. If LLM router errors → default to RAG (safer) but log warning.

---

_Keep this document updated whenever routing heuristics, thresholds, or prompt wording change._ 
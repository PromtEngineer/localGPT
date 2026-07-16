# Prompt inventory

Prompts remain co-located with their implementation; there is no central prompt registry. This inventory names the active call sites without brittle line-number claims.

| Area | File | Purpose | Active by default |
|---|---|---|---|
| Overview | `rag_system/indexing/overview_builder.py` | Per-document routing overview | yes during builds |
| Context enrichment | `rag_system/indexing/contextualizer.py` | Situate a chunk in its document-local neighbor window | when enrichment enabled |
| Query transformation | `rag_system/retrieval/query_transformer.py` | Query rewriting/decomposition and optional HyDE | decomposition path |
| Agent analysis | `rag_system/agent/loop.py` | Initial structured analysis and follow-up reasoning | yes |
| Routing | `rag_system/agent/loop.py` | Choose direct answer, document RAG, or configured graph route using scoped overviews | yes |
| Sub-answer composition | `rag_system/agent/loop.py` | Combine decomposed answers | when enabled |
| Answer synthesis | `rag_system/pipelines/retrieval_pipeline.py` | Answer from retrieved context | document RAG |
| Verification | `rag_system/agent/verifier.py` | Grounding verdict as structured JSON | when enabled |

Dormant/experimental prompts:

- `rag_system/indexing/graph_extractor.py` contains entity and relationship extraction prompts, but graph indexing/retrieval is disabled in normal configuration.
- `QueryTransformer` contains HyDE and graph-query helpers that are not the default retrieval route.
- `rag_system/utils/ollama_client.py` has a vision smoke-test helper; active indexing/retrieval does not call it.

When a prompt changes, update this inventory and add a behavioral test around its parser/fallback when the prompt requires structured output. A future prompt registry should retain provider/model, schema, version, and evaluation metadata rather than only moving strings into YAML.

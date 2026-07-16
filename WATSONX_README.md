# WatsonX provider

WatsonX is an optional generation/enrichment provider. It is not installed by the base requirements and it is not a fully local mode: prompts and retrieved document context are sent to IBM Cloud.

## Install and configure

```bash
pip install -r requirements-watsonx.txt
```

Set:

```bash
LLM_BACKEND=watsonx
WATSONX_API_KEY=...
WATSONX_PROJECT_ID=...
WATSONX_URL=https://us-south.ml.cloud.ibm.com
WATSONX_GENERATION_MODEL=ibm/granite-13b-chat-v2
WATSONX_ENRICHMENT_MODEL=ibm/granite-8b-japanese
```

Model IDs are examples and must be available to the configured WatsonX project/region. The implementation validates that API key and project ID are present when constructing the agent or indexing pipeline.

## Implemented adapter behavior

`rag_system/utils/watsonx_client.py` supplies the completion, asynchronous completion, batch, chat-compatible, and streaming methods expected by the rest of LocalGPT. Streaming behavior depends on the installed IBM SDK/model; the adapter can yield provider chunks or return a completed response through its compatibility path.

Hugging Face embedding and reranking remain local Python components. Only the LLM client switches providers. Ollama model discovery in parts of the UI may therefore not enumerate WatsonX deployment model IDs; configure the provider model variables explicitly.

Use IBM-side access controls, retention settings, and regional configuration appropriate to the sensitivity of uploaded documents. Do not describe this mode as private/offline solely because LanceDB and SQLite remain local.

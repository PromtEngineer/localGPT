# Watson X Integration with Granite Models

localGPT can run its LLM calls against IBM watsonx.ai Granite models instead of a local
Ollama server.

## Overview

`rag_system` supports two LLM backends, selected with the `LLM_BACKEND` environment
variable:

1. **Ollama** (`ollama`, the default) — models run locally.
2. **Watson X** (`watsonx`) — Granite models hosted on IBM watsonx.ai.

The switch is made in `rag_system/factory.py::_build_llm_client()`, which returns either an
`OllamaClient` or a `WatsonXClient` (`rag_system/utils/watsonx_client.py`) together with
the matching config dict (`OLLAMA_CONFIG` or `WATSONX_CONFIG` from `rag_system/main.py`).
Both dicts expose the same `generation_model` / `enrichment_model` keys, so the agent, the
retrieval pipeline and the indexing pipeline are unchanged.

### What the backend switch does and does not cover

Switched to Watson X:

- Answer generation and sub-answer composition (`generation_model`).
- Query routing, triage, query decomposition, contextual enrichment, document overviews and
  answer verification (`enrichment_model`).

**Always local, regardless of `LLM_BACKEND`:**

- **Embeddings.** `rag_system/indexing/representations.py::select_embedder()` returns a
  Hugging Face model when `EMBEDDING_MODEL` contains a `/`, and an Ollama embedder
  otherwise. There is no Watson X embedding path, and `WatsonXClient` exposes no embedding
  method. Point `EMBEDDING_MODEL` at a Hugging Face repo (the default
  `microsoft/harrier-oss-v1-0.6b`) so no Ollama server is needed for indexing.
- **Reranking** (on by default; `Qwen/Qwen3-Reranker-4B`, loaded lazily on the first
  reranked query) and **Provence
  sentence pruning** — both are local `transformers` models.
- **The backend gateway's direct-LLM path.** `backend/server.py` answers non-document
  questions through `backend/ollama_client.py`, which always
  talks to `OLLAMA_HOST`. The routing decision itself is the deterministic, no-LLM
  `should_use_rag` gate. Watson X only serves requests that reach the RAG API on port
  8001.

## Prerequisites

1. IBM Cloud account with watsonx.ai access
2. A watsonx.ai API key
3. A watsonx.ai project ID

### Getting your credentials

1. Go to [IBM Cloud](https://cloud.ibm.com/)
2. Navigate to the watsonx.ai service
3. Create or select a project
4. Get your API key from IBM Cloud IAM
5. Copy your project ID from the project settings

## Installation

The SDK is **not** installed by the root `requirements.txt` (it is listed there as a
commented-out optional extra). Install it explicitly:

```bash
pip install "ibm-watsonx-ai>=1.3.39"
```

`rag_system/requirements.txt` — the RAG-only dependency list — does pin it, so
`pip install -r rag_system/requirements.txt` also gets you the SDK.

Without the package, `WatsonXClient.__init__` raises
`ImportError: ibm-watsonx-ai package is required.` as soon as the agent is constructed.

## Configuration

Copy the example file and fill in your credentials:

```bash
cp env.example.watsonx .env
```

The variables, with the defaults from `rag_system/main.py`:

```bash
# Choose LLM backend (default: ollama)
LLM_BACKEND=watsonx

# Watson X credentials
WATSONX_API_KEY=your_api_key_here
WATSONX_PROJECT_ID=your_project_id_here
WATSONX_URL=https://us-south.ml.cloud.ibm.com

# Model configuration
WATSONX_GENERATION_MODEL=ibm/granite-13b-chat-v2
WATSONX_ENRICHMENT_MODEL=ibm/granite-8b-japanese
```

`WATSONX_API_KEY` and `WATSONX_PROJECT_ID` are mandatory: `_build_llm_client()` raises
`ValueError: Watson X configuration incomplete.` when either is empty.

Use model ids that exist in your watsonx.ai instance — the two above are only the code
defaults, and `ibm/granite-8b-japanese` in particular is unlikely to be the utility model
you want. IBM's
[supported foundation models](https://www.ibm.com/docs/en/watsonx/saas?topic=solutions-supported-foundation-models)
page lists what is currently available.

## Usage

### Running with Watson X

```bash
export LLM_BACKEND=watsonx
python -m rag_system.main api          # RAG API on port 8001
```

`python -m rag_system.api_server` is equivalent. To index or ask a one-off question from
the CLI:

```bash
python -m rag_system.main index ./shared_uploads
python -m rag_system.main chat "What is in these documents?"
```

Or programmatically:

```python
import os
os.environ['LLM_BACKEND'] = 'watsonx'

from rag_system.factory import get_agent

agent = get_agent(mode="default")
result = agent.run("What is artificial intelligence?")
print(result["answer"])
```

### Switching between backends

```bash
# Local Ollama
export LLM_BACKEND=ollama
python -m rag_system.main api

# Watson X
export LLM_BACKEND=watsonx
python -m rag_system.main api
```

### Using Watson X from the web UI

Document-grounded answers do come from Watson X, with two caveats.

**The model dropdown still lists Ollama tags.** The UI populates it from the *backend
gateway's* `GET :8000/models` (`src/lib/api.ts`), which always queries Ollama. Only the
RAG API's own `GET :8001/models` is backend-aware and returns the configured Granite ids
when `LLM_BACKEND=watsonx`.

**That mismatch is harmless.** A per-request `model` is applied only when it is valid for
the active backend — under Watson X the id must contain a `/`, so a stray Ollama tag such
as `qwen3.5:9b` is ignored with a warning and `WATSONX_GENERATION_MODEL` is used instead.
The override is also scoped to that single request and restored afterwards, so one user's
choice cannot leak into the next request.

The backend gateway also keeps using Ollama for its non-document fast path (its routing
decision is the deterministic, no-LLM `should_use_rag` gate — see above), so a fully
Ollama-free setup means talking to the RAG API on
port 8001 directly.

## API compatibility

`WatsonXClient` implements the three methods the RAG system uses from `OllamaClient`:

```python
from rag_system.utils.watsonx_client import WatsonXClient

client = WatsonXClient(
    api_key="your_api_key",
    project_id="your_project_id",
)

# Blocking completion -> {"response": str, "model": str, "done": True}
response = client.generate_completion(
    model="ibm/granite-13b-chat-v2",
    prompt="Explain quantum computing",
)
print(response['response'])

# Streaming completion -> yields text chunks
for chunk in client.stream_completion(
    model="ibm/granite-13b-chat-v2",
    prompt="Write a story about AI",
):
    print(chunk, end='', flush=True)
```

There is also `generate_completion_async()`, which runs the blocking call in an executor
(the IBM SDK has no native async API). The verifier uses it.

## Limitations

1. **No JSON mode.** `OllamaClient.generate_completion()` forwards `format="json"` to
   Ollama; `WatsonXClient.generate_completion()` accepts the argument and ignores it.
   Triage, the overview router, query decomposition and the verifier all rely on the model
   returning parseable JSON. When parsing fails the code falls back to safe defaults
   (route to RAG, do not decompose, no confidence tag), so answers still work but routing
   and verification are less reliable than on Ollama.

2. **Embeddings and rerankers are never Watson X.** See the section above — they load from
   Hugging Face (or Ollama) in-process.

3. **Streaming.** `stream_completion()` uses the SDK's `generate_text_stream()` and falls
   back to yielding the full response as a single chunk if that method is unavailable.

4. **Errors are swallowed into empty responses.** `generate_completion()` catches
   exceptions and returns `{"response": "", "error": ...}`, so an authentication or quota
   failure shows up as an empty answer rather than an HTTP error. Check the server log.

5. **Rate limits and cost.** Watson X is a metered cloud service; local Ollama is not.

## Troubleshooting

### `ImportError: ibm-watsonx-ai package is required`

Run `pip install "ibm-watsonx-ai>=1.3.39"` in the environment that runs the RAG API.

### `ValueError: Watson X configuration incomplete`

`WATSONX_API_KEY` or `WATSONX_PROJECT_ID` is empty. If you put them in `.env`, make sure the
process is started from the repository root — `load_dotenv()` resolves the file relative to
the working directory.

### Authentication errors

- Verify the API key is correct
- Check that the project ID matches an existing watsonx.ai project
- Ensure your IBM Cloud account has watsonx.ai access

### Model not found

- Verify the model id (e.g. `ibm/granite-13b-chat-v2`)
- Check that the model is available in your instance and region
- Some models require additional entitlements

### Connection errors

- Check your internet connection
- Verify `WATSONX_URL` matches your region
- Check the IBM Cloud status page

### Empty answers with no visible error

See limitation 4 — look at the RAG API's stdout for `Error generating completion: …`.

## Reverting to Ollama

```bash
unset LLM_BACKEND   # or set LLM_BACKEND=ollama
python -m rag_system.main api
```

Indexes built while Watson X was active remain valid: the embeddings were produced locally
and never touched Watson X.

## Support

For Watson X issues:
- [IBM watsonx Documentation](https://www.ibm.com/docs/en/watsonx/saas)
- [watsonx Developer Hub](https://www.ibm.com/watsonx/developer/)
- [IBM Cloud Support](https://cloud.ibm.com/docs/get-support)

For localGPT issues:
- [localGPT GitHub Issues](https://github.com/PromtEngineer/localGPT/issues)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

This integration follows the same license as localGPT (MIT License).

# LocalGPT

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=flat-square)](https://www.python.org/)
[![Next.js](https://img.shields.io/badge/next.js-15-black.svg?style=flat-square)](https://nextjs.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg?style=flat-square)](LICENSE)

LocalGPT is a local-first document chat system with persistent sessions and indexes, structured document conversion, hybrid LanceDB retrieval, optional reranking/decomposition/context expansion/pruning/verification, and Ollama generation. WatsonX is available as an explicit cloud alternative.

![Home](Documentation/images/Home.png)

## Implemented capabilities

- PDF, DOCX, PPTX, XLSX, HTML/HTM, Markdown, text, CSV, JSON, and email uploads. Textual formats use lightweight native parsers; Docling provides layout-aware PDF/Office conversion and PDF OCR fallback.
- Legacy recursive or Docling structural chunking, deterministic overlap, optional document-local contextual enrichment, and per-index overview generation.
- Dedicated LanceDB tables per index, FTS plus dense-vector retrieval, weighted reciprocal-rank hybrid fusion, optional late-chunk tables, and cross-index deduplication.
- Agent routing between direct generation and document RAG using only the overviews linked to the session.
- Optional query decomposition, ColBERT reranking, neighbor expansion, Provence sentence pruning, sub-answer composition, and answer verification.
- SQLite-backed sessions, messages, index metadata, many-index session links, and conversation restoration after restart.
- A typed FastAPI gateway with OpenAPI, provider/model discovery, generation and embedding endpoints, durable message/index runs, cancellation/retry, checkpoints, and replayable SSE events.
- A bounded provider-neutral tool agent with schema validation, permissions, approvals, time/output/tool/iteration budgets, retrieval and artifact tools, safe public-web access, read-only database tools, tabular profiling, configured MCP connectors, and Docker-isolated Python execution.
- Content-addressed artifacts with provenance, session/index ownership, local or optional S3-compatible storage, immutable versioned skills, structured redacted logs, and a grounded-answer evaluation harness.
- Non-streaming and SSE session chat through one browser-facing backend; the backend is the sole message-persistence owner. The legacy UI routes remain compatible with the durable runtime.
- Same-origin Next.js backend proxy, strict configurable CORS, loopback defaults, optional bearer authentication, safe upload paths, and remote-model-code opt-in.
- Native launcher plus Docker Compose deployment with shared persistent storage and host or containerized Ollama.

Document-RAG replies include retrieved source rows. Direct-model replies intentionally do not invent document citations.

## Current boundaries

- Ollama generation stays local. WatsonX sends prompts and retrieved context to IBM Cloud.
- Hugging Face embedding/reranking weights may download on first use. Model-repository custom Python is disabled unless `LOCALGPT_TRUST_REMOTE_CODE=true`.
- PDF OCR and table-to-Markdown preservation are active. Page-image embeddings and VLM answer synthesis are experimental scaffolding, not active multimodal RAG.
- Graph extraction/retrieval is disabled by default.
- The supplied deployment is intended for trusted single-user/single-host use. A shared bearer token is not multi-user authorization.
- Code execution is disabled by default and never falls back to the host interpreter. Enabling it requires a running Docker daemon, an allowlisted image, the `code:execute` server permission, and per-run tool approval. Containers have no network, a read-only root, dropped capabilities, resource limits, and a non-root user.
- Cancellation is cooperative. Queued work stops immediately, agent loops stop at tool/model boundaries, and index results are discarded when cancellation is observed; the internal synchronous model/index HTTP call cannot be preempted mid-request.

## Requirements

- Python 3.10+
- Node.js 18+ and npm
- Ollama for the default provider
- 8 GB RAM minimum; 16 GB or more recommended
- Docker/Compose only for container deployment

## Native quick start

```bash
git clone https://github.com/PromtEngineer/localGPT.git
cd localGPT
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
npm ci
cp .env.example .env
ollama pull qwen3:0.6b
ollama pull qwen3:8b
ollama pull nomic-embed-text
python run_system.py
```

Open `http://127.0.0.1:3000`.

The launcher also supports:

```bash
python run_system.py --health
python run_system.py --logs-only
python run_system.py --stop
python run_system.py --mode prod
python run_system.py --no-frontend
```

Manual startup from the repository root:

```bash
ollama serve
python -m rag_system.api_server
python -m backend.server
npm run dev
```

## Docker quick start

Use host Ollama:

```bash
ollama serve
./start-docker.sh local
```

Or use the Compose profile:

```bash
./start-docker.sh container
docker compose --profile with-ollama exec ollama ollama pull qwen3:0.6b
docker compose --profile with-ollama exec ollama ollama pull qwen3:8b
docker compose --profile with-ollama exec ollama ollama pull nomic-embed-text
```

Ports are published only on host loopback. Application data persists in `data/`, `lancedb/`, `index_store/`, and `shared_uploads/`.

## Configuration

Copy `.env.example` to `.env`. Primary settings:

- `OLLAMA_HOST`, `OLLAMA_GENERATION_MODEL`, `OLLAMA_ENRICHMENT_MODEL`, `OLLAMA_EMBEDDING_MODEL`
- `LOCALGPT_DB_PATH`, `LOCALGPT_UPLOAD_DIR`, `LOCALGPT_OVERVIEW_DIR`, `LANCEDB_PATH`
- `LOCALGPT_MAX_UPLOAD_BYTES`, `LOCALGPT_ALLOWED_ORIGINS`, optional `LOCALGPT_API_TOKEN`
- `LOCALGPT_BACKEND_HOST`, `LOCALGPT_BACKEND_PORT`, `LOCALGPT_RAG_HOST`, `RAG_API_URL`, `BACKEND_INTERNAL_URL`
- `RAG_CONFIG_MODE=default|fast`
- `LLM_BACKEND=watsonx` and WatsonX credentials/models for optional cloud generation
- `LOCALGPT_STATE_DIR`, `LOCALGPT_AGENT_PERMISSIONS`, optional tool/connector/artifact-backend settings shown in `.env.example`

Index-time model/chunking/enrichment/late-chunk/retrieval choices are API request fields. Retrieval-time search/fusion/decomposition/reranking/expansion/pruning/verification choices are session-message request fields.

## Basic API flow

The backend is at `http://127.0.0.1:8000`. If `LOCALGPT_API_TOKEN` is set, add `Authorization: Bearer <token>` to direct requests.

```bash
# Create an index
curl -X POST http://127.0.0.1:8000/indexes \
  -H 'Content-Type: application/json' \
  -d '{"name":"Research","description":"Research documents"}'

# Upload and build (replace INDEX_ID)
curl -X POST http://127.0.0.1:8000/indexes/INDEX_ID/upload \
  -F 'files=@paper.pdf'
curl -X POST http://127.0.0.1:8000/indexes/INDEX_ID/build \
  -H 'Content-Type: application/json' \
  -d '{"retrieval_mode":"hybrid","chunk_size":512,"chunk_overlap":64}'

# Create a session, link it, then chat (replace IDs)
curl -X POST http://127.0.0.1:8000/sessions \
  -H 'Content-Type: application/json' \
  -d '{"title":"Research chat","model":"qwen3:8b"}'
curl -X POST http://127.0.0.1:8000/sessions/SESSION_ID/indexes/INDEX_ID
curl -X POST http://127.0.0.1:8000/sessions/SESSION_ID/messages \
  -H 'Content-Type: application/json' \
  -d '{"message":"What are the main findings?","search_type":"hybrid","dense_weight":0.7}'
```

Durable clients submit `/v1/runs`, consume `/v1/runs/{id}/events` with `Last-Event-ID`, and use the cancel/retry endpoints. Interactive API documentation is available at `/docs`; the versioned schema is `Documentation/openapi.json`.

The browser uses the same-origin `/api/backend` proxy instead of calling ports 8000 or 8001 directly.

The same safe API flow is available for scripts:

```bash
python create_index_script.py --name "Research" paper.pdf notes.md
python demo_batch_indexing.py --config batch_indexing_config.json
```

## Verification

```bash
python -m pytest -q
python -m compileall -q backend rag_system run_system.py localgpt_runtime.py
ruff check backend rag_system run_system.py localgpt_runtime.py tests
npm test
npm run typecheck
npm run lint
npm run build
npm run api:types
docker compose config
```

With Ollama and both Python services running, the real model-backed workflow is:

```bash
python scripts/e2e_real_workflow.py \
  --document /absolute/path/to/knowledge.txt \
  --generation-model qwen3:0.6b \
  --embedding-model nomic-embed-text
```

It uploads a real document, requests an Ollama embedding, builds a LanceDB index, links a session, performs a grounded query, checks citation text and SSE replay, and deletes the ephemeral resources.

## Documentation

- [System overview](Documentation/system_overview.md)
- [Architecture overview](Documentation/architecture_overview.md)
- [API reference](Documentation/api_reference.md)
- [Indexing pipeline](Documentation/indexing_pipeline.md)
- [Retrieval pipeline](Documentation/retrieval_pipeline.md)
- [Triage/router](Documentation/triage_system.md)
- [Installation](Documentation/installation_guide.md)
- [Docker usage](Documentation/docker_usage.md)
- [Deployment and security](Documentation/deployment_guide.md)
- [Remaining improvement plan](Documentation/improvement_plan.md)

## License

Apache License 2.0. See [LICENSE](LICENSE).

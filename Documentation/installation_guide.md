# Installation guide

## Requirements

- Python 3.10 or newer; Docker images use Python 3.11.
- Node.js 18 or newer and npm.
- Ollama for the default local generation provider.
- 8 GB RAM minimum; 16 GB or more is recommended for the default models.
- Internet access on first install for Python/npm packages and model weights.

## Native installation

```bash
git clone https://github.com/PromtEngineer/localGPT.git
cd localGPT
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
npm ci
cp .env.example .env
```

Linux and macOS use the same base requirements. The root requirements include the platform-specific OCR dependency used on macOS. If an optional library cannot be installed on your platform, use Docker or edit only the platform marker rather than installing a second, conflicting requirements set.

Install and prepare Ollama:

```bash
ollama serve
ollama pull qwen3:0.6b
ollama pull qwen3:8b
```

Then start the system:

```bash
python run_system.py
```

For manual startup in separate terminals:

```bash
ollama serve
python -m rag_system.api_server
python -m backend.server
npm run dev
```

All Python module commands must be run from the repository root.

## Environment configuration

`.env.example` is the authoritative list. Important settings include:

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama endpoint |
| `OLLAMA_GENERATION_MODEL` | `qwen3:8b` | Answer model |
| `OLLAMA_ENRICHMENT_MODEL` | `qwen3:0.6b` | Routing/enrichment model |
| `LOCALGPT_BACKEND_HOST` | `127.0.0.1` | Backend bind address |
| `LOCALGPT_BACKEND_PORT` | `8000` | Backend port |
| `LOCALGPT_RAG_HOST` | `127.0.0.1` | Internal RAG bind address |
| `RAG_API_URL` | `http://127.0.0.1:8001` | Backend-to-RAG URL |
| `BACKEND_INTERNAL_URL` | `http://127.0.0.1:8000` | Next.js server proxy target |
| `LOCALGPT_DB_PATH` | `./data/chat_data.db` | SQLite persistence |
| `LOCALGPT_UPLOAD_DIR` | `./shared_uploads` | Validated uploads |
| `LANCEDB_PATH` | `./lancedb` | Vector/FTS tables |
| `LOCALGPT_API_TOKEN` | unset | Optional bearer token |

Do not put a secret in a `NEXT_PUBLIC_*` variable. The browser calls the same-origin Next.js proxy, which reads `LOCALGPT_API_TOKEN` only on the server.

## Docker installation

With host Ollama:

```bash
ollama serve
./start-docker.sh local
```

With Ollama in Compose:

```bash
./start-docker.sh container
docker compose --profile with-ollama exec ollama ollama pull qwen3:0.6b
docker compose --profile with-ollama exec ollama ollama pull qwen3:8b
```

Compose persists `./data`, `./lancedb`, `./index_store`, and `./shared_uploads`. Published ports are loopback-only by default.

## Optional WatsonX provider

```bash
pip install -r requirements-watsonx.txt
```

Set `LLM_BACKEND=watsonx`, `WATSONX_API_KEY`, `WATSONX_PROJECT_ID`, and the optional URL/model settings shown in `.env.example`. This mode sends prompts and retrieved context to IBM Cloud and is therefore not fully local.

## Verification

```bash
python run_system.py --health
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8001/health
npm test
python -m pytest -q
```

If `LOCALGPT_API_TOKEN` is set, add `-H "Authorization: Bearer $LOCALGPT_API_TOKEN"` to direct curl calls.

# Quick start

## Local development

Requirements: Python 3.10+, Node.js 18+, npm, Ollama, 8 GB RAM minimum, and 16 GB recommended.

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
python run_system.py
```

`run_system.py` starts or reuses Ollama, then starts the RAG API, backend, and Next.js development server. Open `http://localhost:3000`.

Useful launcher commands:

```bash
python run_system.py --health
python run_system.py --logs-only
python run_system.py --stop
python run_system.py --mode prod
```

Production mode runs the Next.js production build before starting it. `--stop` terminates only processes recorded by this launcher.

## Docker with host Ollama

Start Ollama and pull the models on the host, then run:

```bash
cp .env.example .env
ollama serve
./start-docker.sh local
docker compose ps
```

The application is at `http://localhost:3000`. The backend and internal RAG diagnostics are published on host loopback at ports 8000 and 8001.

## Docker with containerized Ollama

```bash
./start-docker.sh container
docker compose --profile with-ollama exec ollama ollama pull qwen3:0.6b
docker compose --profile with-ollama exec ollama ollama pull qwen3:8b
```

The first start is not ready for chat until the model pulls complete.

## First document chat

1. Open the web application.
2. Choose **Create index**.
3. Upload PDF, DOCX, HTML/HTM, Markdown, or text files.
4. Select chunking, embedding, enrichment, late-chunk, and retrieval options and build the index.
5. Open or create a session linked to that index and send a message.

The build endpoint reports its final status and statistics. RAG replies include the retrieved source rows. Direct-model replies intentionally do not claim document citations.

## Authenticated local use

Set the same `LOCALGPT_API_TOKEN` in the service environment (or `docker.env`). The Next.js server-side proxy adds it to backend requests, so it is not shipped in browser JavaScript. Direct command-line calls must add `Authorization: Bearer <token>`.

See the [installation guide](installation_guide.md), [Docker guide](docker_usage.md), and [API reference](api_reference.md) for more detail.

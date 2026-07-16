# LocalGPT Docker guide

The repository supports host Ollama and an optional Compose-managed Ollama service.

## Host Ollama

```bash
ollama serve
ollama pull qwen3:0.6b
ollama pull qwen3:8b
./start-docker.sh local
```

## Containerized Ollama

```bash
./start-docker.sh container
docker compose --profile with-ollama exec ollama ollama pull qwen3:0.6b
docker compose --profile with-ollama exec ollama ollama pull qwen3:8b
```

Open `http://127.0.0.1:3000`. Backend and RAG diagnostics are available only on host loopback at ports 8000 and 8001.

## Operations

```bash
./start-docker.sh status
./start-docker.sh logs
./start-docker.sh stop
docker compose config
```

Set `LOCALGPT_API_TOKEN` in `docker.env` before startup to enable shared bearer authentication. The Next.js server injects that secret into same-origin backend proxy calls; it is not included in the browser bundle.

Persistent application data lives in `data/`, `lancedb/`, `index_store/`, and `shared_uploads/`. Ollama models use the `ollama_data` named volume when the profile is enabled.

See [Docker usage](Documentation/docker_usage.md) for service, storage, configuration, and troubleshooting details, and [deployment guide](Documentation/deployment_guide.md) for security and backup boundaries.

# Docker usage

## Commands

```bash
# Host Ollama (default)
./start-docker.sh local

# Compose-managed Ollama
./start-docker.sh container

# Operations
./start-docker.sh status
./start-docker.sh logs
./start-docker.sh stop
```

Equivalent Compose commands:

```bash
docker compose --env-file docker.env up --build -d
docker compose ps
docker compose logs -f
docker compose down
```

For the optional Ollama service, add `--profile with-ollama`.

## Services

| Service | Container | Host endpoint | Role |
|---|---|---|---|
| frontend | `rag-frontend` | `127.0.0.1:3000` | Next.js UI and authenticated same-origin proxy |
| backend | `rag-backend` | `127.0.0.1:8000` | Public API, SQLite persistence, upload/index gateway |
| rag-api | `rag-api` | `127.0.0.1:8001` | Internal indexing and agent pipeline |
| ollama | `rag-ollama` | `127.0.0.1:11434` | Optional local model server |

All published ports are loopback-only. Services communicate on `rag-network`.

## Host Ollama

Compose maps `host.docker.internal` to the Docker host gateway, including on Linux. `docker.env` points `OLLAMA_HOST` there. Confirm the host service is reachable before starting:

```bash
curl http://127.0.0.1:11434/api/tags
```

## Compose Ollama

The profile starts the model server but does not automatically download models:

```bash
./start-docker.sh container
docker compose --profile with-ollama exec ollama ollama pull qwen3:0.6b
docker compose --profile with-ollama exec ollama ollama pull qwen3:8b
```

The `ollama_data` named volume persists downloaded models.

## Storage

| Host path | Container path | Consumers |
|---|---|---|
| `./data` | `/app/data` | backend and RAG |
| `./shared_uploads` | `/app/shared_uploads` | backend and RAG |
| `./lancedb` | `/app/lancedb` | RAG |
| `./index_store` | `/app/index_store` | RAG |

The shared upload mount is required because the backend records validated file paths and the RAG service opens those same paths during a build. The shared database mount keeps session/index metadata consistent.

## Configuration and auth

Compose expands `OLLAMA_HOST` and `LOCALGPT_API_TOKEN` from `docker.env` or the shell. Set one non-empty token value for the whole stack. The frontend reads it only in its server runtime and injects it into proxied backend calls.

```bash
LOCALGPT_API_TOKEN='replace-with-a-long-random-secret' docker compose up --build -d
```

Health checks automatically send the same token. For manual direct calls:

```bash
curl -H "Authorization: Bearer $LOCALGPT_API_TOKEN" http://127.0.0.1:8000/health
curl -H "Authorization: Bearer $LOCALGPT_API_TOKEN" http://127.0.0.1:8001/health
```

## Rebuild and reset

```bash
docker compose build --pull
docker compose up -d
```

`docker compose down` preserves bind-mounted application data and the named Ollama volume. To reset application data, stop the stack and deliberately remove the four host persistence directories; that action is irreversible and is not performed by the startup script.

## Troubleshooting

```bash
docker compose config
docker compose ps
docker compose logs --tail=200 rag-api
docker compose logs --tail=200 backend
docker compose logs --tail=200 frontend
docker compose exec rag-api curl -fsS http://localhost:8001/health
docker compose exec backend curl -fsS http://localhost:8000/health
```

Add the bearer header to the last two commands when authentication is enabled. See [deployment guide](deployment_guide.md) for security and backup guidance.

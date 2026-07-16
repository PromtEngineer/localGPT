# Docker troubleshooting

Start with:

```bash
docker compose config
docker compose ps
docker compose logs --tail=200 rag-api backend frontend
```

## Ollama is unreachable

For host Ollama, confirm `curl http://127.0.0.1:11434/api/tags` works on the host and `OLLAMA_HOST=http://host.docker.internal:11434` is present in `docker.env`. Compose maps that hostname to the host gateway on Linux and Docker Desktop.

For container Ollama, confirm the profile is running and the required models exist:

```bash
docker compose --profile with-ollama ps
docker compose --profile with-ollama exec ollama ollama list
```

## Service remains unhealthy

```bash
docker compose exec rag-api curl -v http://localhost:8001/health
docker compose exec backend curl -v http://localhost:8000/health
docker compose exec frontend curl -v http://localhost:3000
```

If `LOCALGPT_API_TOKEN` is non-empty, add `-H "Authorization: Bearer $LOCALGPT_API_TOKEN"` inside the RAG/backend containers. Ensure the same token was supplied to every service by inspecting `docker compose config` without sharing its output publicly.

## Index build cannot open uploads

Both backend and RAG must mount `./shared_uploads` at `/app/shared_uploads`. Both must use that same `LOCALGPT_UPLOAD_DIR`. Do not submit arbitrary host paths to the RAG API; indexing intentionally rejects files outside the shared upload root.

## Chats or indexes disappear after restart

Confirm both backend and RAG mount `./data:/app/data`, with `LOCALGPT_DB_PATH=/app/data/chat_data.db`. Confirm the RAG service also mounts `./lancedb` and `./index_store`. Avoid running the stack from a different working directory, which creates a different set of relative bind mounts.

## Frontend cannot reach backend

The container setting must be `BACKEND_INTERNAL_URL=http://backend:8000`. Browser calls should target `/api/backend`; a `NEXT_PUBLIC_API_URL` is neither needed nor recommended. Check frontend proxy errors with:

```bash
docker compose logs --tail=200 frontend backend
```

## Slow first request or build

The first run downloads Hugging Face weights and loads embedding/reranking models. Ollama models must also be pulled separately. Subsequent container rebuilds may redownload Hugging Face weights unless a cache volume is added.

## Port conflict

Ports 3000, 8000, 8001, and optional 11434 are bound to `127.0.0.1`. Stop the conflicting process or change the host side of the relevant Compose port mapping. Internal container ports and service URLs should remain unchanged.

## Clean rebuild

```bash
docker compose down
docker compose build --no-cache
docker compose up -d
```

This preserves bind-mounted application data. Do not remove `data/`, `lancedb/`, `index_store/`, `shared_uploads/`, or the Ollama volume unless intentional data loss is acceptable.

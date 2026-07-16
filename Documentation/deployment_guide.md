# Deployment guide

LocalGPT ships a single-host Compose deployment. It is not a multi-tenant or internet-hardened SaaS template.

## Recommended single-host deployment

1. Install Docker/Compose and decide whether Ollama runs on the host or in Compose.
2. Set a strong `LOCALGPT_API_TOKEN` in `docker.env` or the shell environment.
3. Keep the supplied loopback-only port mappings unless a trusted reverse proxy is in front of the application.
4. Start the stack and pull the required Ollama models.

Host Ollama:

```bash
ollama serve
ollama pull qwen3:0.6b
ollama pull qwen3:8b
./start-docker.sh local
```

Container Ollama:

```bash
./start-docker.sh container
docker compose --profile with-ollama exec ollama ollama pull qwen3:0.6b
docker compose --profile with-ollama exec ollama ollama pull qwen3:8b
```

Inspect the deployment:

```bash
docker compose ps
docker compose logs -f backend rag-api frontend
curl -H "Authorization: Bearer $LOCALGPT_API_TOKEN" http://127.0.0.1:8000/health
curl -H "Authorization: Bearer $LOCALGPT_API_TOKEN" http://127.0.0.1:8001/health
```

## Network boundary

The browser reaches Next.js at port 3000. Next.js proxies `/api/backend/*` to the backend over the Compose network and injects the optional bearer token. Backend-to-RAG calls use `http://rag-api:8001`. The RAG API should be treated as internal even though a loopback diagnostic port is published.

For LAN or internet access, place a TLS reverse proxy in front of port 3000 and leave ports 8000/8001 unexposed. Configure proxy body-size and timeout limits consistently with `LOCALGPT_MAX_UPLOAD_BYTES` and long index/model operations. Set `LOCALGPT_ALLOWED_ORIGINS` only if a separate browser origin must call the backend directly.

The bearer token is a single shared service token, not user authentication or authorization. Public/multi-user deployment additionally needs an identity provider, per-user authorization, rate limiting, request/audit logging, secret management, malware/content scanning for uploads, resource quotas, and tenant-isolated storage.

## Persistence and backup

Back up these host paths together while writes are quiesced:

- `data/` — SQLite session, message, and index metadata.
- `lancedb/` — vector and FTS tables.
- `index_store/` — overview manifests and optional artifacts.
- `shared_uploads/` — original uploaded documents.

SQLite uses WAL mode. Stop the stack before a simple filesystem copy so the SQLite database and LanceDB artifacts represent one consistent point in time.

## Updates

```bash
docker compose down
git pull --ff-only
docker compose build --pull
docker compose up -d
```

Back up persistent paths before updating. There is no formal schema migration framework yet; database initialization performs backward-compatible table/index cleanup at startup.

## Resource planning

Model weights dominate disk and memory use. Hugging Face embedding/reranker weights are downloaded into the container cache unless you add a persistent cache volume. Concurrent HTTP handling is threaded, while model/index operations are serialized inside the RAG process to prevent mutable configuration races. Scale-up is safer than horizontally replicating the RAG service against the same writable LanceDB directory.

## Health and shutdown

Compose health checks cover Ollama, RAG, backend, and frontend. Stop cleanly with:

```bash
./start-docker.sh stop
```

Do not use `docker compose down -v` unless deleting Ollama's named model volume is intended.

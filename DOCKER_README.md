# 🐳 LocalGPT Docker Deployment Guide

This guide covers running LocalGPT in Docker containers, with Ollama either on the
host (default, best performance) or as a container.

## 🚀 Quick Start

### Complete Setup
```bash
# 1. Install Ollama locally
curl -fsSL https://ollama.ai/install.sh | sh

# 2. Start Ollama server
ollama serve

# 3. Install the models (in another terminal)
ollama pull qwen3.5:9b     # answer generation
ollama pull qwen3.5:4b     # routing, triage, enrichment, verification

# 4. Clone and start LocalGPT
git clone https://github.com/PromtEngineer/localGPT.git
cd localGPT
./start-docker.sh

# 5. Access the application
open http://localhost:3000
```

The first `up --build` installs the Python and Node dependencies and then the
`rag-api` container downloads its embedding model from HuggingFace (the reranker follows lazily on the first reranked query),
so allow several minutes before the UI is reachable.

Running this from a script or CI? Add `-y` so the fallback prompt never blocks:

```bash
./start-docker.sh local -y      # or: NONINTERACTIVE=1 ./start-docker.sh
```

## 📋 Prerequisites

- **Docker Desktop** (or Docker Engine 24+ with the Compose plugin), running
- **Ollama** on the host (recommended) or the containerized fallback below
- **8GB+ RAM** (16GB recommended)
- **20GB+ free disk space** — images, plus ~10GB of HuggingFace weights inside
  `rag-api`, plus the Ollama models

## 🏗️ Architecture

### Default Setup (Local Ollama + Docker Containers)
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │────│    Backend      │────│    RAG API      │
│  (Container)    │    │  (Container)    │    │  (Container)    │
│   Port: 3000    │    │   Port: 8000    │    │   Port: 8001    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                                              │
        │  browser streams directly to :8001           │
        └──────────────────────────────────────────────┤
                                                       │ API calls
                                                       ▼
                                              ┌─────────────────┐
                                              │     Ollama      │
                                              │ (Host, default) │
                                              │   Port: 11434   │
                                              └─────────────────┘
```

**Why host Ollama by default?**
- ✅ Better performance (direct GPU access)
- ✅ Simpler setup (one less container)
- ✅ Easier model management
- ✅ Models survive `docker system prune`

### Containerized Ollama

`docker-compose.yml` also defines an `ollama` service behind the `with-ollama`
profile, for machines where you would rather not install Ollama:

```bash
./start-docker.sh container
# equivalently:
#   OLLAMA_HOST=http://ollama:11434 \
#   docker compose --env-file docker.env --profile with-ollama up --build -d

# Pull the models inside the container the first time
docker compose --profile with-ollama exec ollama ollama pull qwen3.5:9b
docker compose --profile with-ollama exec ollama ollama pull qwen3.5:4b
```

Its models persist in the named volume `ollama_data`.

### Startup Order

```
rag-api (healthy: GET /health)  →  backend (healthy: GET /health)  →  frontend
```

`backend` declares `depends_on: rag-api: condition: service_healthy`, and `rag-api`
does not answer `/health` until the agent and the embedding model are loaded (the
reranker is fetched lazily on the first query that reranks).
Seeing `backend` sit in `created` for a few minutes on a cold start is normal —
follow `docker compose logs -f rag-api`.

## 🛠️ Container Details

### Frontend Container (rag-frontend)
- **Image**: `node:20-alpine`, built by `Dockerfile.frontend`
- **Port**: 3000
- **Purpose**: Next.js web interface
- **Health Check**: busybox `wget -qO- http://localhost:3000` (the alpine image has no curl)
- **Build args**: `NEXT_PUBLIC_API_URL`, `NEXT_PUBLIC_RAG_API_URL` — inlined by `next build`

### Backend Container (rag-backend)
- **Image**: `python:3.11-slim`, built by `Dockerfile.backend`
- **Port**: 8000
- **Purpose**: sessions, indexes, uploads, chat history, gateway to the RAG API
- **Health Check**: `curl -f http://localhost:8000/health`
- **Working directory**: `/app`, started with `python backend/server.py`
- **Extras**: `sqlite3` CLI installed for database troubleshooting

### RAG API Container (rag-api)
- **Image**: `python:3.11-slim`, built by `Dockerfile.rag-api`
- **Port**: 8001
- **Purpose**: document indexing, retrieval, the agent loop
- **Health Check**: `curl -f http://localhost:8001/health`
- **Memory**: dominated by the resident embedding and reranker models
- **Concurrency**: single-threaded — one chat or indexing run at a time

## 📂 Volume Mounts & Data

### Persistent Data (bind mounts to host directories)
- `./lancedb/` → `/app/lancedb` — vectors and the native full-text index
- `./index_store/` → `/app/index_store` — document overviews
- `./shared_uploads/` → `/app/shared_uploads` — uploaded documents
- `./backend/` → `/app/backend` — `chat_data.db`

### Named volumes
- `ollama_data` — only used by the optional containerized Ollama

### Shared Between Containers
`backend` and `rag-api` mount the same four directories and both set
`DB_PATH=/app/backend/chat_data.db`, so they share one SQLite file and one vector
store. HuggingFace model weights are **not** mounted — they live in the container's
writable layer and are re-downloaded whenever `rag-api` is recreated.

## 🔧 Configuration

### Environment Variables (docker.env)
```bash
# Ollama on the host. Every service declares
# extra_hosts: ["host.docker.internal:host-gateway"], so this resolves on Linux too.
OLLAMA_HOST=http://host.docker.internal:11434
# Containerized alternative: OLLAMA_HOST=http://ollama:11434

# Service wiring
NODE_ENV=production
RAG_API_URL=http://rag-api:8001

# Browser-facing URLs, inlined into the frontend at build time
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_RAG_API_URL=http://localhost:8001

# Shared SQLite database and vector store
DB_PATH=/app/backend/chat_data.db
LANCEDB_PATH=/app/lancedb

# Models
GENERATION_MODEL=qwen3.5:9b
ENRICHMENT_MODEL=qwen3.5:4b
EMBEDDING_MODEL=microsoft/harrier-oss-v1-0.6b
RERANKER_MODEL=Qwen/Qwen3-Reranker-4B
```

Pass it explicitly (`start-docker.sh` does this for you):

```bash
docker compose --env-file docker.env up --build -d
```

Variables already exported in your shell take precedence over `--env-file`.

`NEXT_PUBLIC_*` are **build-time** for Next.js. `docker-compose.yml` forwards them
to `Dockerfile.frontend` as build args; changing them at runtime does nothing to an
already-built image, so rebuild:

```bash
NEXT_PUBLIC_API_URL=https://gpt.example.com/api \
NEXT_PUBLIC_RAG_API_URL=https://gpt.example.com/rag \
docker compose --env-file docker.env up -d --build frontend
```

### Model Configuration

| Role | Default | Documented options |
|------|---------|--------------------|
| Generation | `qwen3.5:9b` (Ollama) | `qwen3.6:27b` (high-end, ~17GB), `qwen3.5:4b` (light) |
| Enrichment / utility | `qwen3.5:4b` (Ollama) | `qwen3.5:2b` (light) |
| Embedding | `microsoft/harrier-oss-v1-0.6b` — HuggingFace, MIT, 1024 dims | `Qwen/Qwen3-Embedding-4B` (2560 dims, 32K context), `Qwen/Qwen3-Embedding-0.6B` (1024 dims) |
| Reranking (**on by default**) | `Qwen/Qwen3-Reranker-4B` — HuggingFace, loaded lazily on the first reranked query | `BAAI/bge-reranker-v2-m3`, `answerdotai/answerai-colbert-small-v1`, `Qwen/Qwen3-Reranker-0.6B` |

Only the two Ollama models need `ollama pull`. Embedding dimensions are read from
the loaded model, never hardcoded — **changing `EMBEDDING_MODEL` requires rebuilding
existing indexes**, and appending mismatched vectors to a LanceDB table raises an
explicit error. If the reranker cannot be loaded, the pipeline logs a warning and
continues without reranking.

## 🎯 Management Commands

### Start/Stop Services
```bash
# Start all services (local Ollama)
./start-docker.sh

# Start with containerized Ollama
./start-docker.sh container

# Stop all services
./start-docker.sh stop

# Restart services
./start-docker.sh stop && ./start-docker.sh

# Non-interactive (CI)
./start-docker.sh local -y
```

### Monitor Services
```bash
# Check container status
./start-docker.sh status
docker compose ps

# View live logs
./start-docker.sh logs
docker compose logs -f

# View specific service logs
docker compose logs -f rag-api
docker compose logs -f backend
docker compose logs -f frontend
```

### Manual Docker Compose
```bash
# Start manually
docker compose --env-file docker.env up --build -d

# Stop manually
docker compose down

# Rebuild a specific service (code is COPY-ed in, not mounted -
# a restart alone will not pick up source changes)
docker compose build --no-cache rag-api
docker compose up -d rag-api
```

### Health Checks
```bash
curl -f http://localhost:3000 && echo "✅ Frontend OK"
curl -f http://localhost:8000/health && echo "✅ Backend OK"
curl -f http://localhost:8001/health && echo "✅ RAG API OK"
curl -f http://localhost:11434/api/tags && echo "✅ Ollama OK"
```

`test_docker_build.sh` builds and smoke-tests each image individually against the
same endpoints.

## 🐞 Debugging

### Access Container Shells
```bash
# RAG API container (most debugging happens here)
docker compose exec rag-api bash

# Backend container
docker compose exec backend bash

# Frontend container (alpine -> sh, not bash)
docker compose exec frontend sh
```

### Common Debug Commands
```bash
# Test RAG system initialization
docker compose exec rag-api python -c "
from rag_system.factory import get_agent
agent = get_agent('default')
print('✅ RAG System OK')
"

# Test Ollama connection from the container
docker compose exec rag-api curl -s http://host.docker.internal:11434/api/tags

# Check the wiring the containers actually got
docker compose exec rag-api env | grep -E "OLLAMA|MODEL|DB_PATH|LANCEDB"
docker compose exec backend env | grep RAG_API_URL

# Verify the backend can reach the RAG API by service name
docker compose exec backend curl -s http://rag-api:8001/health

# View Python packages
docker compose exec rag-api pip list | grep -E "(torch|transformers|lancedb|docling|rerankers)"
```

### Resource Monitoring
```bash
docker stats

docker system df
du -sh lancedb shared_uploads index_store

docker stats --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"
```

## 🚨 Troubleshooting

### Common Issues

#### Container Won't Start
```bash
# Check logs for the specific error
docker compose logs [service-name]

# Rebuild from scratch
./start-docker.sh stop
docker system prune -f
./start-docker.sh

# Check for port conflicts
lsof -i :3000 -i :8000 -i :8001
```

#### `backend` never starts
It is waiting for `rag-api` to report healthy. Check:
```bash
docker compose ps
docker compose logs -f rag-api
docker compose exec rag-api curl -s http://localhost:8001/health
```

#### Can't Connect to Ollama
```bash
# Verify Ollama is running on the host
curl http://localhost:11434/api/tags

# Restart Ollama
pkill ollama
ollama serve

# Test from the container
docker compose exec rag-api curl -s http://host.docker.internal:11434/api/tags
```

#### Every chat answers "Could not connect to the RAG API server"
The backend builds its URLs from `RAG_API_URL`; inside compose that must be
`http://rag-api:8001`, since `localhost` there is the backend container itself.
```bash
docker compose exec backend env | grep RAG_API_URL
docker compose exec backend curl -s http://rag-api:8001/health
```

#### Memory Issues
```bash
docker stats --no-stream
free -h  # on the host

# Docker Desktop → Settings → Resources → Memory → 8GB+

# Lighter configuration
GENERATION_MODEL=qwen3.5:4b \
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B \
docker compose --env-file docker.env up -d rag-api backend
# (changing the embedding model requires rebuilding your indexes)
```

#### Frontend Build Errors
```bash
docker compose build --no-cache frontend
docker compose up -d frontend
docker compose logs frontend
```

#### Frontend talks to the wrong host
`NEXT_PUBLIC_API_URL` / `NEXT_PUBLIC_RAG_API_URL` are baked in at build time.
Rebuild the frontend image after changing them.

#### Database/Storage Issues
```bash
# Check file permissions
ls -la backend/chat_data.db
ls -la lancedb/

# Reset permissions
chmod 664 backend/chat_data.db
chmod -R 755 lancedb/ shared_uploads/

# Inspect the database (sqlite3 is installed in the image)
docker compose exec backend sqlite3 /app/backend/chat_data.db ".tables"
```

### Performance Notes

- The RAG API is single-threaded: a chat request queued behind an indexing run
  looks like a hang. Check `docker compose logs -f rag-api`.
- Contextual enrichment makes one LLM call per chunk; it dominates indexing time.
  Turn it off in the index build options for the fastest ingest.
- `RAG_CONFIG_MODE=fast` switches the RAG API to vector-only retrieval with no
  reranking, decomposition or verification.

### Complete Reset
```bash
# Nuclear option - resets everything, including your documents and chat history
./start-docker.sh stop
docker compose down
rm -rf lancedb/* index_store/* shared_uploads/* backend/chat_data.db
docker system prune -a --volumes
./start-docker.sh
```

`docker compose down -v` alone only drops the `ollama_data` volume — application
data is in host directories and must be deleted explicitly.

## 🏆 Success Criteria

Your Docker deployment is successful when:

- ✅ `docker compose ps` shows all services healthy
- ✅ All health checks pass (see commands above)
- ✅ You can access http://localhost:3000
- ✅ You can upload documents and create indexes
- ✅ You can chat with your documents
- ✅ No errors in container logs

### What to Expect

- **First build**: installs torch/transformers/docling and builds the Next.js
  bundle; then `rag-api` downloads the ~1.2GB embedding model before it reports
  healthy (the ~7.5GB reranker downloads lazily, on the first reranked query)
- **Restarts**: fast. **Recreating** `rag-api` re-downloads the model weights,
  because no volume is mounted for the HuggingFace cache
- **Concurrency**: one RAG request at a time

## 📚 Additional Resources

- **Detailed Troubleshooting**: See `DOCKER_TROUBLESHOOTING.md`
- **Complete Docker Guide**: See `Documentation/docker_usage.md`
- **Deployment Guide**: See `Documentation/deployment_guide.md`
- **System Architecture**: See `Documentation/architecture_overview.md`
- **Direct Development**: See the main `README.md`

---

**Happy Dockerizing! 🐳** Need help? Check the troubleshooting guide or open an issue.

# 🐳 Docker Usage Guide - LocalGPT

_Last updated: 2026-08-08_

Practical Docker commands and procedures for running LocalGPT in containers.

---

## 📋 Prerequisites

### Required Setup
- Docker Desktop (or Docker Engine 24+ with the Compose plugin) running
- Ollama, either on the host (default) or as a container
- 8GB+ RAM available

### Architecture Overview
```
┌─────────────────────────────────────────────────┐
│                Docker Containers                │
├─────────────────────────────────────────────────┤
│ frontend (3000)  →  backend (8000)  →  rag-api  │
│                                         (8001)  │
└─────────────────────────────────────────────────┘
            │                               │
            │  browser streams              │
            │  directly to :8001            ▼
            └──────────────────►  ┌────────────────────┐
                                  │ Ollama (11434)     │
                                  │ host.docker.internal│
                                  └────────────────────┘
```

`backend` and `rag-api` share `./backend` (SQLite), `./lancedb`, `./index_store`
and `./shared_uploads` as bind mounts.

---

## 1. Quick Start Commands

### Step 1: Clone

```bash
git clone https://github.com/PromtEngineer/localGPT.git
cd localGPT

# Verify Docker is running
docker version
```

### Step 2: Ollama

The compose files point the containers at `host.docker.internal:11434` by default,
and declare `extra_hosts: ["host.docker.internal:host-gateway"]` so that also
resolves on Linux.

```bash
# Install Ollama on the host
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama (in one terminal)
ollama serve

# Install the models (in another terminal)
ollama pull qwen3.5:9b      # answer generation
ollama pull qwen3.5:4b      # routing, triage, enrichment, verification

ollama list
curl http://localhost:11434/api/tags
```

**Or run Ollama in a container.** `docker-compose.yml` defines an `ollama` service
behind the `with-ollama` profile:

```bash
./start-docker.sh container
# equivalently:
#   OLLAMA_HOST=http://ollama:11434 \
#   docker compose --env-file docker.env --profile with-ollama up --build -d

# Models must be pulled inside the container the first time
docker compose --profile with-ollama exec ollama ollama pull qwen3.5:9b
docker compose --profile with-ollama exec ollama ollama pull qwen3.5:4b
```

The embedding model (`microsoft/harrier-oss-v1-0.6b`) is a HuggingFace download
inside `rag-api`, not an Ollama model — nothing to pull for it. The reranker
(`Qwen/Qwen3-Reranker-4B`) is only downloaded if you switch reranking on, which
is off by default.

### Step 3: Start Containers

```bash
./start-docker.sh              # local Ollama
./start-docker.sh container    # containerized Ollama
./start-docker.sh stop
./start-docker.sh logs
./start-docker.sh status
./start-docker.sh help
```

`./start-docker.sh` with no argument probes port 11434. If nothing is listening it
offers the containerized fallback; pass `-y`/`--yes` (or set `NONINTERACTIVE=1`) to
take it without a prompt, which is what CI should do. With no TTY and no `-y` it
exits 1 with instructions instead of hanging.

### 1.2 Service Access

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **RAG API**: http://localhost:8001
- **Ollama**: http://localhost:11434

### 1.3 Startup Order

```
rag-api (healthy)  →  backend (healthy)  →  frontend
```

`rag-api` loads the embedding and reranker models before `/health` answers, so its
check has a 60s start period and `backend` intentionally waits in `created` until
then. `docker compose logs -f rag-api` shows the progress.

---

## 2. Container Management

### 2.1 Using the Convenience Script

```bash
./start-docker.sh
./start-docker.sh stop
./start-docker.sh logs
./start-docker.sh status

# Restart
./start-docker.sh stop && ./start-docker.sh
```

### 2.2 Manual Docker Compose Commands

```bash
# Start all services
docker compose --env-file docker.env up --build -d

# Check status
docker compose ps

# View logs
docker compose logs -f

# Stop all services
docker compose down

# Force rebuild
docker compose build --no-cache
docker compose --env-file docker.env up --build -d
```

Always pass `--env-file docker.env` (as `start-docker.sh` does). Without it the
compose defaults still apply — they mirror `docker.env` — but any value you edit in
`docker.env` is ignored.

### 2.3 Individual Service Management

```bash
docker compose up -d frontend
docker compose up -d backend
docker compose up -d rag-api

docker compose restart rag-api
docker compose stop backend
docker compose logs -f rag-api
```

---

## 3. Development Workflow

### 3.1 Code Changes

No source is bind-mounted — code is `COPY`-ed into the images at build time, so a
restart alone will not pick up an edit. Rebuild the affected service:

```bash
docker compose up -d --build frontend
docker compose up -d --build backend
docker compose up -d --build rag-api

# After a dependency change
docker compose build --no-cache rag-api
docker compose up -d rag-api
```

Editing `NEXT_PUBLIC_API_URL` or `NEXT_PUBLIC_RAG_API_URL` also needs a frontend
**rebuild** — Next.js inlines those at `next build` time and they are passed to
`Dockerfile.frontend` as build args.

### 3.2 Debugging Containers

```bash
# Shells
docker compose exec frontend sh      # node:20-alpine
docker compose exec backend bash     # python:3.11-slim
docker compose exec rag-api bash     # python:3.11-slim

# Run commands in a container
docker compose exec rag-api python -c "from rag_system.factory import get_agent; get_agent('default'); print('✅ RAG System OK')"
docker compose exec backend curl -s http://localhost:8000/health

# Environment
docker compose exec rag-api env | grep -E "OLLAMA|MODEL|DB_PATH|LANCEDB"
```

### 3.3 Compose File Variants

| File | Contents |
|------|----------|
| `docker-compose.yml` | `rag-api`, `backend`, `frontend`, plus an optional `ollama` service behind the `with-ollama` profile |
| `docker-compose.local-ollama.yml` | The same three application services, no optional `ollama` service |

There is no `docker-compose.dev.yml`.

---

## 4. Logging & Monitoring

### 4.1 Log Management

```bash
docker compose logs
docker compose logs frontend
docker compose logs backend
docker compose logs rag-api

docker compose logs -f
docker compose logs --tail=100
docker compose logs -t
docker compose logs > system.log 2>&1
docker compose logs --since=2h
```

### 4.2 System Monitoring

```bash
docker stats
docker stats rag-frontend rag-backend rag-api

docker compose ps
docker inspect rag-api --format='{{.State.Health.Status}}'

docker system info
docker system df
```

Container names are fixed by `container_name`: `rag-frontend`, `rag-backend`,
`rag-api`, and `rag-ollama` for the optional Ollama service.

---

## 5. Ollama Integration

### 5.1 Host Ollama

```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve
curl http://localhost:11434/api/tags

ollama pull qwen3.5:9b
ollama pull qwen3.5:4b
ollama list
```

### 5.2 From Inside a Container

```bash
docker compose exec rag-api curl -s http://host.docker.internal:11434/api/tags

curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3.5:4b", "prompt": "Hello", "stream": false}'
```

Ollama logs appear in the terminal running `ollama serve`; for the containerized
variant use `docker compose --profile with-ollama logs -f ollama`.

### 5.3 Model Management

```bash
ollama pull qwen3.5:9b
ollama pull qwen3.5:4b
ollama pull qwen3.6:27b     # optional high-end generation model

ollama rm old-model-name
ollama show qwen3.5:9b
```

Point the containers at a different model without editing code:

```bash
GENERATION_MODEL=qwen3.6:27b docker compose --env-file docker.env up -d rag-api backend
```

---

## 6. Data Management

### 6.1 Volumes and Mounts

Every application path is a **bind mount to a host directory**, so ordinary file
tools work. The only named volume is `ollama_data`, used by the optional Ollama
container.

| Host path | Container path | Contents |
|-----------|----------------|----------|
| `./lancedb` | `/app/lancedb` | Vectors and the native full-text index |
| `./index_store` | `/app/index_store` | Document overviews |
| `./shared_uploads` | `/app/shared_uploads` | Uploaded source documents |
| `./backend` | `/app/backend` | `chat_data.db` (shared by backend and rag-api) |

```bash
docker volume ls
docker system df -v

# Back up the host directories directly
tar czf backup/lancedb_backup.tar.gz lancedb/
tar czf backup/index_store_backup.tar.gz index_store/

# Only the containerized Ollama uses a named volume.
# The compose project name is the directory name, so it is localgpt_ollama_data.
docker run --rm -v localgpt_ollama_data:/data -v $(pwd)/backup:/backup \
  alpine tar czf /backup/ollama_models.tar.gz -C /data .

docker volume prune
```

### 6.2 Database Management

```bash
# sqlite3 is installed in both Python images
docker compose exec backend sqlite3 /app/backend/chat_data.db ".tables"

# Back up the database
cp backend/chat_data.db backup/chat_data_$(date +%Y%m%d).db

# Check LanceDB tables from the container
docker compose exec rag-api python -c "
import lancedb
db = lancedb.connect('/app/lancedb')
print('Tables:', db.table_names())
"
```

`backend` and `rag-api` both set `DB_PATH=/app/backend/chat_data.db` and mount the
same host directory, so they read and write one file.

### 6.3 File Management

```bash
docker compose exec rag-api ls -la /app/shared_uploads

docker cp local_file.pdf rag-api:/app/shared_uploads/
docker cp rag-api:/app/shared_uploads/file.pdf ./local_file.pdf

docker compose exec rag-api df -h
```

Because `shared_uploads/` is a bind mount, copying a file into `./shared_uploads`
on the host is equivalent and simpler.

---

## 7. Troubleshooting

### 7.1 Common Issues

#### Container Won't Start
```bash
docker version
lsof -i :3000 -i :8000 -i :8001
docker compose logs [service-name]
```

#### `backend` stays in `created`
That is `depends_on: rag-api: condition: service_healthy` doing its job. Watch
`docker compose logs -f rag-api` — on a cold start it is downloading and loading
the embedding and reranker models.

#### Ollama Connection Issues
```bash
curl http://localhost:11434/api/tags

pkill ollama
ollama serve

docker compose exec rag-api curl -s http://host.docker.internal:11434/api/tags
```

#### Chats return "Could not connect to the RAG API server"
The backend builds its URLs from `RAG_API_URL`, which must be
`http://rag-api:8001` inside compose (`localhost` there means the backend
container itself).
```bash
docker compose exec backend env | grep RAG_API_URL
docker compose exec backend curl -s http://rag-api:8001/health
```

#### Performance Issues
```bash
docker stats
docker compose ps

# Docker Desktop → Settings → Resources → Memory → 8GB+
```

### 7.2 Reset and Clean

```bash
# Stop everything
./start-docker.sh stop

# Clean containers and images
docker system prune -a

# Complete reset (⚠️ deletes indexes, uploads and chat history)
docker compose down
rm -rf lancedb/* index_store/* shared_uploads/* backend/chat_data.db
docker system prune -a
```

`docker compose down -v` only removes the named `ollama_data` volume — application
data lives in host directories and must be deleted explicitly.

### 7.3 Health Checks

```bash
curl -f http://localhost:3000 && echo "✅ Frontend OK"
curl -f http://localhost:8000/health && echo "✅ Backend OK"
curl -f http://localhost:8001/health && echo "✅ RAG API OK"
curl -f http://localhost:11434/api/tags && echo "✅ Ollama OK"

docker compose ps

# Test model loading inside the container
docker compose exec rag-api python -c "
from rag_system.factory import get_agent
agent = get_agent('default')
print('✅ RAG System initialized successfully')
"
```

These are the same endpoints the container health checks use: `curl -f /health`
for `backend` and `rag-api`, and busybox `wget -qO- http://localhost:3000` for
`frontend` (the alpine image has no curl).

---

## 8. Advanced Usage

### 8.1 Production Deployment

```bash
# docker.env already sets NODE_ENV=production
docker compose --env-file docker.env up --build -d

# All services already declare restart: unless-stopped
docker compose ps
```

There is no authentication and CORS is wide open on both APIs. Put a reverse proxy
in front and publish only what you need. Note the browser streams directly from
port 8001, so that port must be reachable by clients (or proxied) unless you turn
off "Stream phases" in the chat UI.

### 8.2 Scaling

`docker compose up -d --scale backend=2 --scale rag-api=2` **does not work with the
shipped compose file**: each service sets a fixed `container_name`
(`rag-backend`, `rag-api`) and publishes a fixed host port, both of which conflict
on the second replica. Remove `container_name` and the `ports:` mappings (or switch
to a random host port) first.

Even then, the RAG API is single-threaded and holds per-process state — the
resident agent, its in-memory chat history and the semantic cache — so replicas
need sticky sessions at minimum.

### 8.3 Security

```bash
docker scout cves rag-frontend
docker scout cves rag-backend
docker scout cves rag-api

docker compose build --no-cache --pull
```

---

## 9. Configuration

### 9.1 Environment Variables

`docker.env` is passed with `--env-file` and supplies both runtime environment and
compose-level substitution:

```bash
# Ollama on the host; extra_hosts makes this resolve on Linux too
OLLAMA_HOST=http://host.docker.internal:11434
# Containerized alternative: OLLAMA_HOST=http://ollama:11434

NODE_ENV=production
RAG_API_URL=http://rag-api:8001

# Browser-facing; inlined into the frontend bundle at build time
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_RAG_API_URL=http://localhost:8001

# Shared SQLite + vector store
DB_PATH=/app/backend/chat_data.db
LANCEDB_PATH=/app/lancedb

# Models
GENERATION_MODEL=qwen3.5:9b
ENRICHMENT_MODEL=qwen3.5:4b
EMBEDDING_MODEL=microsoft/harrier-oss-v1-0.6b
RERANKER_MODEL=Qwen/Qwen3-Reranker-4B
```

Values already exported in your shell win over `--env-file` — that is how
`start-docker.sh container` overrides `OLLAMA_HOST`.

Changing `EMBEDDING_MODEL` invalidates existing indexes: every table records the
embedding model that wrote it (and its vector width), and writing or querying it
with a different model fails with an explicit error. Rebuild your indexes after
switching.

### 9.2 Custom Configuration

```bash
cp docker.env docker.custom.env
nano docker.custom.env
docker compose --env-file docker.custom.env up -d

# Remember to rebuild if you changed a NEXT_PUBLIC_* value
docker compose --env-file docker.custom.env up -d --build frontend
```

---

## 10. Success Checklist

- ✅ All containers healthy: `docker compose ps`
- ✅ Ollama reachable: `curl http://localhost:11434/api/tags`
- ✅ Frontend loads: `curl http://localhost:3000`
- ✅ Backend responds: `curl http://localhost:8000/health`
- ✅ RAG API responds: `curl http://localhost:8001/health`
- ✅ You can create an index and chat with your documents

### What to Expect

- **First `up --build`** is slow: it installs the Python dependencies (torch,
  transformers, docling) and builds the Next.js bundle, and `rag-api` then
  downloads ~10GB of HuggingFace weights before it reports healthy
- **Restarting** an existing container is fast. The HuggingFace weights are
  downloaded at runtime into the container's writable layer, and no volume is
  mounted for them, so **recreating** `rag-api` (any `up --build`, `down` + `up`, or
  image change) downloads them again. Mount a cache directory and set `HF_HOME` if
  that matters to you.
- **One RAG request at a time** — the RAG API is single-threaded

---

**Happy Containerizing! 🐳**

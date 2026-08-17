# 🐳 Docker Troubleshooting Guide - LocalGPT

_Last updated: 2026-08-08_

This guide helps diagnose and fix Docker-related issues with LocalGPT's
containerized deployment.

---

## 🏁 Quick Health Check

### System Status Check
```bash
# Check Docker daemon
docker version

# Check Ollama status
curl http://localhost:11434/api/tags

# Check containers
./start-docker.sh status

# Test all endpoints
curl -f http://localhost:3000 && echo "✅ Frontend OK"
curl -f http://localhost:8000/health && echo "✅ Backend OK"
curl -f http://localhost:8001/health && echo "✅ RAG API OK"
curl -f http://localhost:11434/api/tags && echo "✅ Ollama OK"
```

### Expected Success Output
```
✅ Frontend OK
✅ Backend OK
✅ RAG API OK
✅ Ollama OK
```

### Know the startup order before you debug

```
rag-api (healthy: GET /health)  →  backend (healthy: GET /health)  →  frontend
```

`backend` has `depends_on: rag-api: condition: service_healthy`, and `rag-api` only
answers `/health` once the agent and the embedding model have loaded (the reranker
loads lazily, on the first reranked query).
On a cold start that is a multi-minute HuggingFace download. **`backend` sitting in
`created` is usually patience, not a bug** — confirm with
`docker compose logs -f rag-api`.

---

## 🚨 Common Issues & Solutions

### 1. Docker Daemon Issues

#### Problem: "Cannot connect to Docker daemon"
```
Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?
```

#### Solution A: Restart Docker Desktop (macOS/Windows)
```bash
# Quit Docker Desktop completely
# macOS: Click Docker icon → "Quit Docker Desktop"
# Windows: Right-click Docker icon → "Quit Docker Desktop"

# Wait for it to fully shut down
sleep 10

# Start Docker Desktop
open -a Docker  # macOS
# Windows: Click Docker Desktop from Start menu

# Wait for Docker to be ready (2-3 minutes)
docker version
```

#### Solution B: Linux Docker Service
```bash
sudo systemctl status docker
sudo systemctl restart docker
sudo systemctl enable docker
docker version
```

#### Solution C: Hard Reset
```bash
# Kill all Docker processes
sudo pkill -f docker

# Remove socket files
sudo rm -f /var/run/docker.sock
sudo rm -f "$HOME/.docker/run/docker.sock"  # macOS

# Restart Docker Desktop
open -a Docker  # macOS
```

### 2. Ollama Connection Issues

#### Problem: RAG API can't connect to Ollama
```
ConnectionError: Failed to connect to Ollama at http://host.docker.internal:11434
```

#### Solution A: Verify Ollama is Running
```bash
curl http://localhost:11434/api/tags

# If not running, start it
ollama serve

# Install the required models
ollama pull qwen3.5:9b
ollama pull qwen3.5:4b
```

`run_system.py` pulls these automatically, but `start-docker.sh` does not — with
host Ollama you must pull them yourself.

#### Solution B: Test from Container
```bash
docker compose exec rag-api curl -sv http://host.docker.internal:11434/api/tags

# Inspect the network. The compose project name is the directory name,
# so the network is localgpt_rag-network.
docker network ls
docker network inspect localgpt_rag-network
```

All services declare `extra_hosts: ["host.docker.internal:host-gateway"]`, so this
name resolves on Linux as well as macOS and Windows. If it still fails, check that
Ollama is listening on all interfaces rather than only the loopback:

```bash
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

#### Solution C: Use the containerized Ollama instead
```bash
./start-docker.sh container

docker compose --profile with-ollama exec ollama ollama pull qwen3.5:9b
docker compose --profile with-ollama exec ollama ollama pull qwen3.5:4b
```

This exports `OLLAMA_HOST=http://ollama:11434`, which wins over `docker.env`
because shell variables take precedence over `--env-file`.

#### Solution D: Point at a specific host address
```bash
# Edit docker.env rather than appending duplicates - the last value wins,
# but duplicate keys make the file confusing.
# macOS example:
OLLAMA_HOST=http://$(ipconfig getifaddr en0):11434 ./start-docker.sh
```

### 3. Backend / RAG API Wiring

#### Problem: every chat fails with 502 "Could not connect to the RAG API server"

The backend builds `/chat` and `/index` from `RAG_API_URL`. Inside compose that
must be `http://rag-api:8001` — `localhost` there refers to the backend container.
Chat failures come back as JSON error bodies: 502 when the RAG API is unreachable,
504 when the call times out.

```bash
docker compose exec backend env | grep RAG_API_URL
docker compose exec backend curl -s http://rag-api:8001/health
```

#### Problem: chat returns 504 "did not respond within 600s"

The RAG API is single-threaded, so a chat request queued behind an indexing run can
exceed the timeout. Watch `docker compose logs -f rag-api`, and raise the limits if
your hardware is genuinely slow:

```bash
RAG_API_TIMEOUT=1200 RAG_API_INDEX_TIMEOUT=7200 \
docker compose --env-file docker.env up -d backend
```

#### Problem: the frontend calls the wrong host

`NEXT_PUBLIC_API_URL` and `NEXT_PUBLIC_RAG_API_URL` are inlined by `next build`.
Setting them only in `environment:` cannot change an already-built image.

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000 \
NEXT_PUBLIC_RAG_API_URL=http://localhost:8001 \
docker compose --env-file docker.env up -d --build frontend
```

### 4. Container Build Failures

#### Problem: Frontend build fails
```
ERROR: Failed to build frontend container
```

#### Solution: Clean Build
```bash
./start-docker.sh stop

docker system prune -f
docker builder prune -f

docker compose build --no-cache frontend
docker compose up -d frontend

docker compose logs frontend
```

`Dockerfile.frontend` copies `package.json`, `package-lock.json`, `src/`,
`public/`, `next.config.ts`, `tsconfig.json`, `postcss.config.mjs` and
`eslint.config.mjs`. If you deleted one of those in a fork, the `COPY` fails.
`public/` is kept non-empty by `public/.gitkeep`.

#### Problem: Python package installation fails
```
ERROR: Could not install packages due to an EnvironmentError
```

#### Solution
```bash
# Both Python images install requirements-docker.txt
ls -la requirements-docker.txt

# Test locally
pip install -r requirements-docker.txt --dry-run

# Rebuild with an updated base image
docker compose build --no-cache --pull rag-api
```

### 5. Port Conflicts

#### Problem: "Port already in use"
```
Error starting userland proxy: listen tcp4 0.0.0.0:3000: bind: address already in use
```

#### Solution: Find and Kill Conflicting Processes
```bash
lsof -i :3000 -i :8000 -i :8001

# A direct (non-Docker) run of the same stack is the usual culprit
python run_system.py --stop

# Or by port
sudo kill -9 $(lsof -t -i:3000)
sudo kill -9 $(lsof -t -i:8000)
sudo kill -9 $(lsof -t -i:8001)

./start-docker.sh
```

### 6. Memory Issues

#### Problem: Containers crash due to OOM
```
Container killed due to memory limit
```

The `rag-api` container holds the embedding model and the reranker in memory for
the life of the process, and enabling late chunking loads a second copy of the
embedder during indexing.

#### Solution: Increase Docker Memory or shrink the models
```bash
docker stats --no-stream

# Docker Desktop → Settings → Resources → Memory → 8GB+

# Lighter configuration (rebuild indexes after changing the embedding model)
GENERATION_MODEL=qwen3.5:4b \
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B \
docker compose --env-file docker.env up -d rag-api backend

# Or switch the whole RAG API to the fast profile
RAG_CONFIG_MODE=fast docker compose --env-file docker.env up -d rag-api
```

#### Problem: System running slow
```bash
free -h  # Linux
vm_stat  # macOS

docker system prune -f
docker volume prune -f
```

### 7. Volume Mount Issues

#### Problem: Permission denied accessing files
```
Permission denied: /app/lancedb
```

#### Solution: Fix Permissions
```bash
mkdir -p lancedb index_store shared_uploads backend

chmod -R 755 lancedb index_store shared_uploads
chmod 664 backend/chat_data.db

ls -la lancedb/ shared_uploads/ backend/

sudo chown -R $USER:$USER lancedb shared_uploads index_store backend
```

#### Problem: Database file not found
```
No such file or directory: '/app/backend/chat_data.db'
```

`ChatDatabase` creates the parent directory and the file itself, so this normally
means the `./backend` bind mount is missing or `DB_PATH` points somewhere
unmounted.

```bash
docker compose exec backend env | grep DB_PATH        # expect /app/backend/chat_data.db
docker compose exec backend ls -la /app/backend

# Create it from the host if needed
python -c "
from backend.database import ChatDatabase
db = ChatDatabase()
print('Database initialized at', db.db_path)
"

./start-docker.sh stop
./start-docker.sh
```

#### Problem: Indexes exist in the sidebar but every answer says nothing was found

The SQLite rows and the LanceDB tables must match. Restoring one without the other,
or changing `LANCEDB_PATH`, leaves index records pointing at tables that do not
exist.

```bash
docker compose exec rag-api python -c "
import lancedb
print(lancedb.connect('/app/lancedb').table_names())
"
docker compose exec backend sqlite3 /app/backend/chat_data.db \
  "select id, name, vector_table_name from indexes;"
```

#### Problem: `changing the embedding model requires rebuilding the index`

Vector width is read from the loaded embedding model. Delete and rebuild the index
after changing `EMBEDDING_MODEL`.

---

## 🔍 Advanced Debugging

### Container-Level Debugging

#### Access Container Shells
```bash
# RAG API container (most issues happen here)
docker compose exec rag-api bash

# Check the wiring
docker compose exec rag-api env | grep -E "OLLAMA|MODEL|DB_PATH|LANCEDB|RAG_CONFIG"

# Test Python imports
docker compose exec rag-api python -c "
import sys
print('Python version:', sys.version)
from rag_system.factory import get_agent
print('✅ RAG system imports work')
"

# Backend container
docker compose exec backend bash
docker compose exec backend python -c "
from backend.database import ChatDatabase
print('✅ Database imports work')
"

# Frontend container (alpine -> sh, not bash)
docker compose exec frontend sh -c "node --version && npm --version"
```

#### Check Container Resources
```bash
docker stats

docker compose ps
docker inspect rag-api --format='{{.State.Health.Status}}'
docker inspect rag-api --format='{{json .State.Health}}' | head -c 2000

docker compose config
```

#### Network Debugging

The Python images are `python:3.11-slim` and the frontend is `node:20-alpine`;
**none of them ship `ping` or `nslookup`**. Use curl (installed in both Python
images) or busybox wget (in the alpine image):

```bash
# Backend ↔ RAG API by service name
docker compose exec backend curl -sv http://rag-api:8001/health
docker compose exec rag-api  curl -sv http://backend:8000/health

# Container → host Ollama
docker compose exec rag-api curl -sv http://host.docker.internal:11434/api/tags

# From the frontend (alpine)
docker compose exec frontend wget -qO- http://backend:8000/health

# If you really want ping/nslookup, install them in the running container first
docker compose exec rag-api sh -c "apt-get update && apt-get install -y iputils-ping dnsutils"
```

### Log Analysis

#### Container Logs
```bash
./start-docker.sh logs

docker compose logs -f rag-api
docker compose logs -f backend
docker compose logs -f frontend

docker compose logs rag-api 2>&1 | grep -i error
docker compose logs backend 2>&1 | grep -i "traceback\|error"

docker compose logs > docker-debug.log 2>&1
```

The RAG API logs each stage of a query, so `docker compose logs -f rag-api` while
you ask a question shows exactly where time is going: retrieval, reranking, context
expansion, pruning, synthesis, verification.

#### System Logs
```bash
# Docker daemon logs (Linux)
journalctl -u docker.service -f

# macOS: Check Console app for Docker logs
# Windows: Check Event Viewer
```

---

## 🧪 Testing & Validation

### Manual Container Testing

`test_docker_build.sh` automates exactly this — building each image and probing its
health endpoint:

```bash
./test_docker_build.sh
```

By hand:

```bash
# RAG API alone
docker build -f Dockerfile.rag-api -t test-rag-api .
docker run -d --name test-rag-api -p 8001:8001 \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  --add-host host.docker.internal:host-gateway test-rag-api
sleep 60      # models load before /health answers
curl http://localhost:8001/health
docker rm -f test-rag-api

# Backend alone
docker build -f Dockerfile.backend -t test-backend .
docker run -d --name test-backend -p 8000:8000 test-backend
sleep 15
curl http://localhost:8000/health
docker rm -f test-backend
```

### Integration Testing
```bash
./start-docker.sh -y

# Wait for the health-gated chain to come up
until curl -sf http://localhost:8000/health >/dev/null; do sleep 5; done

# Create a session
curl -X POST http://localhost:8000/sessions \
  -H "Content-Type: application/json" \
  -d '{"title": "Test Session"}'

# Create an index, upload a document, build it
IDX=$(curl -s -X POST http://localhost:8000/indexes \
  -H "Content-Type: application/json" \
  -d '{"name":"Smoke Test"}' | python3 -c "import sys,json;print(json.load(sys.stdin)['index_id'])")
curl -X POST "http://localhost:8000/indexes/$IDX/upload" -F "files=@test.pdf"
curl -X POST "http://localhost:8000/indexes/$IDX/build" \
  -H "Content-Type: application/json" -d '{"chunk_size": 512}'

./start-docker.sh stop
```

The upload form field must be named `files`.

---

## 🔄 Recovery Procedures

### Complete System Reset

#### Soft Reset
```bash
./start-docker.sh stop
docker system prune -f
./start-docker.sh
```

#### Hard Reset (⚠️ Deletes all data)
```bash
./start-docker.sh stop
docker compose down

# Application data lives in host directories, not volumes -
# `docker compose down -v` alone will NOT remove it.
rm -rf lancedb/* index_store/* shared_uploads/* backend/chat_data.db

docker system prune -a --volumes
./start-docker.sh
```

#### Selective Reset
```bash
# Just the chat/index database
./start-docker.sh stop
rm backend/chat_data.db
./start-docker.sh

# Just vector storage (indexes must be rebuilt; delete the DB rows too or they
# will point at tables that no longer exist)
./start-docker.sh stop
rm -rf lancedb/*
./start-docker.sh

# Just uploaded documents
rm -rf shared_uploads/*
```

---

## 📊 Performance Optimization

### Resource Monitoring
```bash
watch -n 5 'docker stats --no-stream'

docker system df
du -sh lancedb shared_uploads index_store backend

htop  # Linux
top   # macOS
```

### Performance Tuning
```bash
# Fast profile: vector-only retrieval, no reranking, decomposition or verification
RAG_CONFIG_MODE=fast docker compose --env-file docker.env up -d rag-api

# Smaller generation model
GENERATION_MODEL=qwen3.5:4b docker compose --env-file docker.env up -d rag-api backend

# Smaller embedding model (rebuild indexes afterwards)
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B docker compose --env-file docker.env up -d rag-api

# Faster indexing: turn off contextual enrichment in the index build options
# (it makes one LLM call per chunk)
```

Structural limits worth knowing before you tune:
- The RAG API is single-threaded — requests are serialised.
- Indexing is synchronous; `POST /indexes/{id}/build` stays open until it finishes.
- Model weights are downloaded into the container's writable layer, so recreating
  `rag-api` re-downloads them unless you mount a cache and set `HF_HOME`.

---

## 🆘 When All Else Fails

### Alternative Deployment Options

#### 1. Direct Development (No Docker)
```bash
./start-docker.sh stop
python run_system.py
```

#### 2. Minimal Docker (RAG API only)
```bash
docker build -f Dockerfile.rag-api -t rag-api .
docker run -p 8001:8001 \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  --add-host host.docker.internal:host-gateway rag-api

# Run the rest directly, from the repository root
python backend/server.py &
npm run dev
```

#### 3. Hybrid Approach
```bash
docker compose --env-file docker.env up -d rag-api

# Point the host backend at the container
RAG_API_URL=http://localhost:8001 python backend/server.py &
npm run dev
```

### Getting Help

#### Diagnostic Information to Collect
```bash
docker version
docker compose version
uname -a

docker compose ps
docker compose config

docker stats --no-stream
docker system df

docker compose logs > docker-errors.log 2>&1
```

#### Support Channels
1. **Check GitHub Issues**: Search existing issues for similar problems
2. **Documentation**: Review `Documentation/` and `DOCKER_README.md`
3. **Create Issue**: Include the diagnostic information above

---

## ✅ Success Checklist

Your Docker deployment is working correctly when:

- ✅ `docker version` shows Docker is running
- ✅ `curl http://localhost:11434/api/tags` shows Ollama is accessible
- ✅ `docker compose ps` shows all services healthy
- ✅ `curl -f http://localhost:8000/health` and `.../8001/health` both return 200
- ✅ You can access the frontend at http://localhost:3000
- ✅ You can create document indexes successfully
- ✅ You can chat with your documents
- ✅ No error messages in container logs

**If all boxes are checked, your Docker deployment is successful! 🎉**

---

**Still having issues?** Check `DOCKER_README.md` or create an issue with your diagnostic information.

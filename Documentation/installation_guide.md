# 📦 LocalGPT Installation Guide

_Last updated: 2026-08-08_

This guide provides step-by-step instructions for installing and setting up
LocalGPT using either Docker or a direct development install.

---

## 🎯 Installation Options

### Option 1: Docker Deployment 🐳
- **Best for**: reproducible environments, keeping Python dependencies isolated
- **Requirements**: Docker Desktop + Ollama (host or containerized)
- **Setup time**: ~10 minutes plus image build and model downloads

### Option 2: Direct Development 💻
- **Best for**: development, customization, debugging
- **Requirements**: Python + Node.js + Ollama
- **Setup time**: ~15 minutes plus model downloads

---

## 1. Prerequisites

### 1.1 System Requirements

#### **Minimum Requirements**
- **CPU**: 4 cores, 2.5GHz+
- **RAM**: 8GB (16GB recommended)
- **Storage**: 50GB free space
- **OS**: macOS 10.15+, Ubuntu 20.04+, Windows 10+ (WSL2)

#### **Recommended Requirements**
- **CPU**: 8+ cores, 3.0GHz+
- **RAM**: 32GB+ (for the larger generation models)
- **Storage**: 200GB+ SSD
- **GPU**: NVIDIA GPU with 8GB+ VRAM (optional; Apple Silicon uses MPS)

Disk budget for the defaults: `qwen3.5:9b` and `qwen3.5:4b` in Ollama, plus
`microsoft/harrier-oss-v1-0.6b` (~1.2GB) in the HuggingFace cache
(`~/.cache/huggingface`). Reranking is off by default, so no reranker weights
are fetched unless you switch it on (`Qwen/Qwen3-Reranker-4B`, ~7.5GB).

### 1.2 Common Dependencies

**Required for both approaches:**
- **Ollama**: model runtime (always required)
- **Git**: 2.30+ for cloning the repository

**Docker-specific:**
- **Docker Desktop**: 24.0+ with the Compose plugin

**Direct Development-specific:**
- **Python**: 3.10+ (3.11 recommended — the Docker images use `python:3.11-slim`)
- **Node.js**: 20+ with npm

---

## 2. Ollama Installation (Required for Both)

### 2.1 Install Ollama

#### **macOS/Linux:**
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Verify installation
ollama --version
```

#### **Windows:**
```bash
# Download from: https://ollama.ai/download
# Run the installer and follow setup wizard
```

### 2.2 Pull the Models

```bash
# Start Ollama server
ollama serve

# In another terminal, install the two default models
ollama pull qwen3.5:9b      # answer generation
ollama pull qwen3.5:4b      # routing, triage, decomposition, enrichment, verification

# Verify models are installed
ollama list

# Test Ollama
ollama run qwen3.5:4b "Hello, how are you?"
```

The embedding and reranker models are **not** Ollama models — they are downloaded
from HuggingFace automatically the first time the pipeline loads them.

**⚠️ Important**: Keep Ollama running (`ollama serve`) for the entire setup process.

---

## 3. 🐳 Docker Installation & Setup

### 3.1 Install Docker

#### **macOS:**
```bash
# Install Docker Desktop via Homebrew
brew install --cask docker

# Or download from: https://www.docker.com/products/docker-desktop/
# Start Docker Desktop from Applications

# Verify installation
docker --version
docker compose version
```

#### **Ubuntu/Debian:**
```bash
# Update system
sudo apt-get update

# Install Docker using convenience script
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Install Docker Compose V2
sudo apt-get install docker-compose-plugin

# Verify installation
docker --version
docker compose version
```

#### **Windows:**
1. Download Docker Desktop from https://www.docker.com/products/docker-desktop/
2. Run installer and enable WSL 2 integration
3. Restart computer and start Docker Desktop
4. Verify in PowerShell: `docker --version`

### 3.2 Clone and Start

```bash
# Clone repository
git clone https://github.com/PromtEngineer/localGPT.git
cd localGPT

# Verify Ollama is running on the host
curl http://localhost:11434/api/tags

# Start Docker containers (uses local Ollama)
./start-docker.sh

# Or, without host Ollama:
#   ./start-docker.sh container
# For scripts and CI, add -y so the script never prompts:
#   ./start-docker.sh local -y     (or: NONINTERACTIVE=1 ./start-docker.sh)

# Verify deployment
./start-docker.sh status
```

The first build compiles the frontend and installs the Python dependencies, and
`rag-api` loads the embedding and reranker models before it reports healthy. The
`backend` service has `depends_on: rag-api: service_healthy`, so it deliberately
waits. Expect several minutes on a cold start.

### 3.3 Test Docker Deployment

```bash
# Test all endpoints
curl -f http://localhost:3000 && echo "✅ Frontend OK"
curl -f http://localhost:8000/health && echo "✅ Backend OK"
curl -f http://localhost:8001/health && echo "✅ RAG API OK"
curl -f http://localhost:11434/api/tags && echo "✅ Ollama OK"

# Access the application
open http://localhost:3000
```

---

## 4. 💻 Direct Development Setup

### 4.1 Install Development Dependencies

#### **Python Setup:**
```bash
# Clone repository
git clone https://github.com/PromtEngineer/localGPT.git
cd localGPT

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Install Python dependencies
pip install -r requirements.txt

# Verify Python setup
python -c "import torch; print('✅ PyTorch OK')"
python -c "import transformers; print('✅ Transformers OK')"
python -c "import lancedb; print('✅ LanceDB OK')"
python -c "import docling; print('✅ Docling OK')"
```

`requirements.txt` at the repository root is the full install used by
`run_system.py`. Two variants exist for narrower cases:

| File | Purpose |
|------|---------|
| `requirements.txt` | Everything the local stack needs |
| `requirements-docker.txt` | Used by `Dockerfile.backend` and `Dockerfile.rag-api` (no macOS-only packages) |
| `rag_system/requirements.txt` | RAG core only; adds macOS `ocrmac` and `ibm-watsonx-ai` |
| `backend/requirements.txt` | Gateway only (`requests`, `python-dotenv`) |

#### **Node.js Setup:**
```bash
# Install Node.js dependencies
npm install

# Verify Node.js setup
node --version  # Should be 20+
npm --version
```

### 4.2 Start Direct Development

```bash
# Ensure Ollama is running
curl http://localhost:11434/api/tags

# Start all components with one command
python run_system.py

# Or start components manually, each from the repository root:
# Terminal 1: python -m rag_system.api_server
# Terminal 2: python backend/server.py
# Terminal 3: npm run dev
```

> Always run from the repository root. `backend/chat_data.db`, `lancedb/`,
> `index_store/` and `shared_uploads/` are resolved relative to the working
> directory, so `cd backend && python server.py` would create a second database at
> `backend/backend/chat_data.db`.

### 4.3 Test Direct Development

```bash
# Deep check: imports, config, LanceDB, embedding model, sample query
python system_health_check.py

# HTTP health check per service
python run_system.py --health

# Test endpoints
curl -f http://localhost:3000 && echo "✅ Frontend OK"
curl -f http://localhost:8000/health && echo "✅ Backend OK"
curl -f http://localhost:8001/health && echo "✅ RAG API OK"
```

---

## 5. Detailed Installation Steps

### 5.1 Repository Setup

```bash
# Clone repository
git clone https://github.com/PromtEngineer/localGPT.git
cd localGPT

# Check repository structure
ls -la

# These are created on demand, but you can pre-create them
mkdir -p lancedb index_store shared_uploads logs backend

# Set permissions
chmod -R 755 lancedb index_store shared_uploads
```

The SQLite file is created automatically the first time `ChatDatabase` is
constructed — you do not need to `touch` it.

### 5.2 Configuration

LocalGPT runs with no configuration file. Every variable below has a default that
matches the code. Override them in the shell, in a `.env` at the repository root
(`load_dotenv()` runs on import of `rag_system/main.py` and
`rag_system/factory.py`), or in `docker.env` for containers. `.env.example` ships
the same list.

#### **Environment Variables**

| Variable | Default | Read by |
|----------|---------|---------|
| `OLLAMA_HOST` | `http://localhost:11434` | `rag_system/main.py`, `backend/ollama_client.py` |
| `RAG_API_URL` | `http://localhost:8001` | `backend/server.py` |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | `src/lib/api.ts` (build-time) |
| `NEXT_PUBLIC_RAG_API_URL` | `http://localhost:8001` | `src/lib/api.ts` (build-time) |
| `DB_PATH` | `backend/chat_data.db` | `backend/database.py` |
| `LANCEDB_PATH` | `./lancedb` | `rag_system/main.py` (pipeline profiles), `backend/database.py`, `system_health_check.py` |
| `GENERATION_MODEL` | `qwen3.5:9b` | `rag_system/main.py`, `backend/server.py`, `run_system.py` |
| `ENRICHMENT_MODEL` | `qwen3.5:4b` | same |
| `EMBEDDING_MODEL` | `microsoft/harrier-oss-v1-0.6b` | `rag_system/main.py` |
| `RERANKER_MODEL` | `Qwen/Qwen3-Reranker-4B` (only loaded when reranking is switched on) | `rag_system/main.py` |
| `RAG_CONFIG_MODE` | `default` | `rag_system/api_server.py` (`default` or `fast`) |
| `RAG_API_TIMEOUT` | `600` | `backend/server.py` |
| `RAG_API_INDEX_TIMEOUT` | `3600` | `backend/server.py` |
| `LLM_BACKEND` | `ollama` | `rag_system/main.py` (`ollama` or `watsonx`) |
| `HF_TOKEN` | unset | HuggingFace downloads |

For Docker these are set in `docker.env` and passed with
`docker compose --env-file docker.env`:

```bash
OLLAMA_HOST=http://host.docker.internal:11434
NODE_ENV=production
RAG_API_URL=http://rag-api:8001
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_RAG_API_URL=http://localhost:8001
DB_PATH=/app/backend/chat_data.db
LANCEDB_PATH=/app/lancedb
GENERATION_MODEL=qwen3.5:9b
ENRICHMENT_MODEL=qwen3.5:4b
EMBEDDING_MODEL=microsoft/harrier-oss-v1-0.6b
RERANKER_MODEL=Qwen/Qwen3-Reranker-4B
```

`run_system.py` does **not** invent environment variables. It inherits your shell
environment unchanged and only adds `NODE_ENV=production` to the Python services
in `--mode prod`. If you want a non-default `OLLAMA_HOST` or `RAG_API_URL`, export
it before launching.

`NEXT_PUBLIC_*` values are inlined into the frontend bundle by `next build`, so
they are build-time settings. In Docker they are passed as build args in
`docker-compose.yml`; changing them means `docker compose build frontend`.

#### **Model Configuration**

| Role | Default | Documented options |
|------|---------|--------------------|
| Generation | `qwen3.5:9b` | `qwen3.6:27b` (high-end, ~17GB), `qwen3.5:4b` (light) |
| Enrichment / utility | `qwen3.5:4b` | `qwen3.5:2b` (light) |
| Embedding | `microsoft/harrier-oss-v1-0.6b` (MIT, 1024 dims) | `Qwen/Qwen3-Embedding-4B` (2560 dims, 32K context), `Qwen/Qwen3-Embedding-0.6B` (1024 dims) |
| Reranker (**off by default**) | `Qwen/Qwen3-Reranker-4B`, loaded lazily by the in-repo `QwenRerankerScorer` | `BAAI/bge-reranker-v2-m3`, `answerdotai/answerai-colbert-small-v1`, `Qwen/Qwen3-Reranker-0.6B` |

Notes:
- Embedding dimensions are read from the loaded model, never hardcoded.
  **Changing `EMBEDDING_MODEL` requires rebuilding every existing index** —
  appending vectors of a different width to a LanceDB table raises an explicit
  error telling you to re-index.
- If the reranker fails to load, the pipeline logs a warning and continues
  **without reranking**. There is no secondary reranker to fall back to.
- Any name containing `/` is treated as a HuggingFace model; anything else is
  treated as an Ollama tag, so an Ollama embedding model such as
  `nomic-embed-text` also works if you have pulled it.
- Vision / multimodal models are not wired into any pipeline. PDF parsing and OCR
  are handled by Docling.

### 5.3 Database Initialization

```bash
# Initialize SQLite database (also happens automatically at first use)
python -c "
from backend.database import ChatDatabase
db = ChatDatabase()
print('✅ Database initialized at', db.db_path)
"

# Verify database
sqlite3 backend/chat_data.db ".tables"
```

---

## 6. Verification & Testing

### 6.1 System Health Checks

#### **Comprehensive Health Check:**
```bash
# For Docker deployment
./start-docker.sh status
docker compose ps

# For Direct development
python system_health_check.py
python run_system.py --health

# Universal health check
curl -f http://localhost:3000 && echo "✅ Frontend OK"
curl -f http://localhost:8000/health && echo "✅ Backend OK"
curl -f http://localhost:8001/health && echo "✅ RAG API OK"
curl -f http://localhost:11434/api/tags && echo "✅ Ollama OK"
```

#### **RAG System Test:**
```bash
# Test RAG system initialization (factory.py is the single entry point)
python -c "
from rag_system.factory import get_agent
agent = get_agent('default')
print('✅ RAG System initialized successfully')
"

# Test embedding generation and report the real dimension
python -c "
from rag_system.factory import get_agent
agent = get_agent('default')
embedder = agent.retrieval_pipeline._get_text_embedder()
test_emb = embedder.create_embeddings(['Hello world'])
print(f'✅ Embedding generated: {test_emb.shape}')
"
```

### 6.2 Functional Testing

#### **Document Upload Test:**
1. Access http://localhost:3000
2. Click "Create New Index"
3. Upload a PDF document
4. Configure settings and build index
5. Test chat functionality

#### **API Testing:**
```bash
# Test session creation
curl -X POST http://localhost:8000/sessions \
  -H "Content-Type: application/json" \
  -d '{"title": "Test Session"}'

# Test models endpoints
curl http://localhost:8000/models
curl http://localhost:8001/models

# Test health endpoints
curl http://localhost:8000/health
curl http://localhost:8001/health
```

---

## 7. Troubleshooting Installation

### 7.1 Common Issues

#### **Ollama Issues:**
```bash
# Ollama not responding
curl http://localhost:11434/api/tags

# If it fails, restart Ollama
pkill ollama
ollama serve

# Reinstall models if needed
ollama pull qwen3.5:9b
ollama pull qwen3.5:4b
```

#### **Docker Issues:**
```bash
# Docker daemon not running
docker version

# Restart Docker Desktop (macOS/Windows)
# Or restart docker service (Linux)
sudo systemctl restart docker

# Clear Docker cache if build fails
docker system prune -f
```

#### **Python Issues:**
```bash
# Check Python version
python --version  # 3.10+ required, 3.11 recommended

# Check virtual environment
which python
pip list | grep torch

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

#### **Node.js Issues:**
```bash
# Check Node version
node --version  # Should be 20+

# Clear and reinstall
rm -rf node_modules package-lock.json
npm install
```

#### **Scanned PDFs produce no text:**
A PDF with no text layer is re-run through Docling's OCR pipeline, but only with an
engine that is actually installed. `rag_system/ingestion/document_converter.py`
probes, in order: `ocrmac` (macOS only), `easyocr`, `rapidocr_onnxruntime`,
`tesserocr`, then the `tesseract` binary. Install whichever suits your platform —
`pip install ocrmac` on macOS (it is listed in `rag_system/requirements.txt` but not
in the root `requirements.txt`), or `pip install easyocr` / `apt install tesseract-ocr`
elsewhere. At startup the converter prints the engine it chose (`OCR engine: …`) or
`No OCR engine available; using docling's default OCR settings.`

### 7.2 Performance Issues

#### **Memory Problems:**
```bash
# Check system memory
free -h  # Linux
vm_stat  # macOS

# For Docker: Increase memory allocation
# Docker Desktop → Settings → Resources → Memory → 8GB+

# Use lighter models
export GENERATION_MODEL=qwen3.5:4b
# (the default embedder is already the small one, ~1.2GB)
```

#### **Slow Performance:**
- Use SSD storage for `lancedb/` and `shared_uploads/`
- Switch to `RAG_CONFIG_MODE=fast` (vector-only retrieval, no reranking,
  no decomposition, no verification)
- Lower `indexing.embedding_batch_size` / `enrichment_batch_size` if you are
  swapping rather than compute-bound
- Remember the RAG API serialises requests — one query or indexing run at a time

---

## 8. Post-Installation Setup

### 8.1 Model Experiments

```bash
# Install additional generation models
ollama pull qwen3.6:27b            # highest quality, ~17GB
ollama pull qwen3.5:2b             # lightest utility model

# Try one for a single request
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Hello", "model": "qwen3.5:4b"}'
```

A per-request `model` is applied only for that request and then restored, and is
rejected with a warning if it is not valid for the active `LLM_BACKEND`.

### 8.2 Security Configuration

Neither server implements authentication, and both send
`Access-Control-Allow-Origin: *`. Do not expose ports 8000/8001 outside a trusted
network.

```bash
# Set proper file permissions
chmod 600 backend/chat_data.db    # Restrict database access
chmod 700 lancedb/                # Restrict vector DB access

# Configure firewall (production)
sudo ufw allow 3000/tcp           # Frontend
sudo ufw deny 8000/tcp            # Backend (internal only)
sudo ufw deny 8001/tcp            # RAG API (internal only)
```

### 8.3 Backup Setup

```bash
# Create backup script
cat > backup_system.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup databases and indexes
cp -r backend/chat_data.db "$BACKUP_DIR/"
cp -r lancedb "$BACKUP_DIR/"
cp -r index_store "$BACKUP_DIR/"
cp -r shared_uploads "$BACKUP_DIR/"

echo "Backup completed: $BACKUP_DIR"
EOF

chmod +x backup_system.sh
```

All four paths are plain host directories (bind-mounted into the containers), so
a file copy is a complete backup. Stop the services first so SQLite and LanceDB
are not mid-write.

---

## 9. Success Criteria

### 9.1 Installation Complete When:

- ✅ `python run_system.py --health` exits 0
- ✅ Frontend loads at http://localhost:3000
- ✅ `ollama list` shows `qwen3.5:9b` and `qwen3.5:4b`
- ✅ You can create a document index
- ✅ You can chat with uploaded documents
- ✅ No error messages in `logs/` or the terminal

### 9.2 Performance Expectations

Numbers depend heavily on hardware, model size and document length. On a machine
matching the recommended requirements, expect:

- First startup dominated by model downloads (Ollama tags + ~10GB of HuggingFace
  weights); subsequent startups load from cache
- Indexing dominated by contextual enrichment — it runs one LLM call per chunk, so
  turn it off (`enable_enrich: false`) for the fastest ingest
- Query latency dominated by generation; `RAG_CONFIG_MODE=fast` removes reranking,
  decomposition and verification

---

## 10. Next Steps

### 10.1 Getting Started

1. **Upload Documents**: Create your first index
2. **Explore Features**: Try different retrieval modes and models
3. **Customize**: Adjust chunk size, enrichment and verification per request
4. **Scale**: Add more documents and create multiple indexes

### 10.2 Additional Resources

- **Quick Start**: See `Documentation/quick_start.md`
- **Docker Usage**: See `Documentation/docker_usage.md`
- **Deployment**: See `Documentation/deployment_guide.md`
- **System Architecture**: See `Documentation/architecture_overview.md`
- **API Reference**: See `Documentation/api_reference.md`
- **WatsonX backend**: See `WATSONX_README.md`

---

**Congratulations! 🎉** Visit http://localhost:3000 to start chatting with your documents.

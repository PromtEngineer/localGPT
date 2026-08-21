# ⚡ Quick Start Guide - LocalGPT

_Get up and running in about 10 minutes (plus model download time)._

---

## 🚀 Choose Your Deployment Method

### Option 1: Docker Deployment 🐳

Best for: isolated environments, running the same stack everywhere.

### Option 2: Direct Development 💻

Best for: development, customization, debugging, faster iteration.

Both need Ollama. The Docker path runs Ollama on the host by default (better GPU
access), but can also run it as a container.

---

## 🐳 Docker Deployment

### Prerequisites
- Docker Desktop (or Docker Engine 24+ with the Compose plugin) installed and running
- 8GB+ RAM available
- Internet connection (first build downloads Python and Node packages; first query
  downloads the embedding model from HuggingFace; the reranker follows on first use)

### Step 1: Clone

```bash
git clone https://github.com/PromtEngineer/localGPT.git
cd localGPT

# Ensure Docker is running
docker version
```

### Step 2: Install Ollama Locally

**By default the containers talk to Ollama on the host:**

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama (in one terminal)
ollama serve

# Install models (in another terminal)
ollama pull qwen3.5:9b     # answer generation
ollama pull qwen3.5:4b     # routing, triage, enrichment, verification
```

Prefer not to install Ollama on the host? Skip this step and use
`./start-docker.sh container` below.

### Step 3: Start Docker Containers

```bash
# Start all containers against local Ollama
./start-docker.sh

# Or manually:
docker compose --env-file docker.env up --build -d
```

Containerized Ollama instead:

```bash
./start-docker.sh container
# Pull the models inside the container the first time
docker compose --profile with-ollama exec ollama ollama pull qwen3.5:9b
docker compose --profile with-ollama exec ollama ollama pull qwen3.5:4b
```

`./start-docker.sh` with no argument checks port 11434. If nothing is listening it
offers to switch to the containerized Ollama; pass `-y` (or set `NONINTERACTIVE=1`)
to accept that without a prompt in scripts.

### Step 4: Verify Deployment

```bash
# Check container status (backend waits for rag-api to report healthy)
docker compose ps

# Test endpoints
curl http://localhost:3000            # Frontend
curl http://localhost:8000/health     # Backend
curl http://localhost:8001/health     # RAG API
```

The `rag-api` container loads the agent and the embedding model at startup, so its
health check has a 120s start period and the first `docker compose up` can take
several minutes before `backend` starts. The reranker (~7.5 GB) is fetched lazily on
the first query that reranks — expect one slow first answer.

### Step 5: Access Application

Open your browser to: **http://localhost:3000**

---

## 💻 Direct Development

### Prerequisites
- Python 3.10+ (3.11 recommended)
- Node.js 20+ and npm
- 8GB+ RAM available

### Step 1: Clone and Install Dependencies

```bash
git clone https://github.com/PromtEngineer/localGPT.git
cd localGPT

# Install Python dependencies
pip install -r requirements.txt

# Install Node.js dependencies
npm install
```

### Step 2: Install and Configure Ollama

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama (in one terminal)
ollama serve

# Install models (in another terminal)
ollama pull qwen3.5:9b
ollama pull qwen3.5:4b
```

### Step 3: Start the System

```bash
# Start all components with one command
python run_system.py
```

`run_system.py` reuses an already-running Ollama, pulls any missing model, then
starts the RAG API, the backend and the frontend.

**Or start components manually in separate terminals — all from the repository root:**

```bash
# Terminal 1: RAG API
python -m rag_system.api_server

# Terminal 2: Backend
python backend/server.py

# Terminal 3: Frontend
npm run dev
```

> Do not `cd backend` first. The SQLite path defaults to `backend/chat_data.db`
> relative to the working directory, so running from inside `backend/` creates a
> second database at `backend/backend/chat_data.db`.

### Step 4: Verify Installation

```bash
# Loads the models and runs a sample query against the first LanceDB table
python system_health_check.py

# HTTP health check per service (exits non-zero if a required one is unhealthy)
python run_system.py --health

# Test endpoints
curl http://localhost:3000            # Frontend
curl http://localhost:8000/health     # Backend
curl http://localhost:8001/health     # RAG API
```

### Step 5: Access Application

Open your browser to: **http://localhost:3000**

---

## 🎯 First Use Guide

### 1. Create a Chat Session
- Click "New Chat" in the interface
- Give your session a descriptive name

### 2. Upload Documents
- Click "Create New Index"
- Upload PDF, DOCX, TXT, MD or HTML files
- Configure processing options:
  - **Chunk Size**: 512 (default)
  - **Embedding Model**: `microsoft/harrier-oss-v1-0.6b` (default)
  - **Enable Enrichment**: Yes
- Click "Build Index" and wait for processing

Indexing is synchronous: the request stays open until the pipeline finishes.

### 3. Start Chatting
- Select your built index
- Ask questions about your documents:
  - "What is this document about?"
  - "Summarize the key points"
  - "What are the main findings?"
  - "Compare the arguments in section 3 and 5"

The chat settings panel exposes the same knobs the API does: search type
(`hybrid` / `vector_only` / `fts_only`), retrieval_k, reranker top-k, context
window, decomposition, verification, Provence pruning, and "Stream phases".

> **Stream phases is on by default**, and that path streams straight from the RAG
> API to the browser. When the stream completes, the UI saves the finished turn
> through the gateway (`POST /sessions/{id}/messages/save`), so the conversation
> **is** written to the chat history database. Only direct (non-UI) stream
> consumers must save the turn themselves.

---

## 🔧 Management Commands

### Docker Commands

```bash
# Container management
./start-docker.sh                    # Start (local Ollama)
./start-docker.sh container          # Start (containerized Ollama)
./start-docker.sh stop               # Stop all containers
./start-docker.sh logs               # View logs
./start-docker.sh status             # Check status
./start-docker.sh help               # Usage

# Manual Docker Compose
docker compose ps                    # Check status
docker compose logs -f               # Follow logs
docker compose down                  # Stop containers
docker compose --env-file docker.env up --build -d   # Rebuild and start
```

### Direct Development Commands

```bash
# System management
python run_system.py                # Start all services
python run_system.py --mode prod    # `npm run build` then `next start`
python run_system.py --no-frontend  # Ollama + RAG API + backend only
python run_system.py --health       # HTTP health checks
python run_system.py --logs-only    # Tail logs/*.log from another shell
python run_system.py --stop         # Stop everything in logs/run_system.pid
python system_health_check.py       # Deep check: models, LanceDB, sample query

# Individual components (from the repository root)
python -m rag_system.api_server     # RAG API only
python backend/server.py            # Backend only
npm run dev                          # Frontend only

# Stop: Ctrl+C in the terminal running the services, or `python run_system.py --stop`
```

---

## 🆘 Quick Troubleshooting

### Docker Issues

**Containers not starting?**
```bash
# Check Docker daemon
docker version

# The backend will not start until rag-api reports healthy
docker compose ps
docker compose logs -f rag-api
```

**Port conflicts?**
```bash
# Check what's using ports
lsof -i :3000 -i :8000 -i :8001

# Stop conflicting processes
./start-docker.sh stop
```

### Direct Development Issues

**Import errors?**
```bash
# Check Python installation
python --version  # 3.10+ required, 3.11 recommended

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**Node.js errors?**
```bash
# Check Node version
node --version    # Should be 20+

# Reinstall dependencies
rm -rf node_modules package-lock.json
npm install
```

### Common Issues

**Ollama not responding?**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama
pkill ollama
ollama serve
```

**Out of memory?**
```bash
# Check memory usage
docker stats  # For Docker
htop          # For direct development

# Use lighter models
export GENERATION_MODEL=qwen3.5:4b
# (the default embedder is already the small one, ~1.2GB)
```

**Answers ignore my documents?**
Check that the session is linked to a built index — the backend only routes to the
RAG API when the session has one. Sending `"force_rag": true` on
`POST /sessions/{id}/messages` bypasses the router.

---

## 📊 System Verification

```bash
# Check all endpoints
curl -f http://localhost:3000 && echo "✅ Frontend OK"
curl -f http://localhost:8000/health && echo "✅ Backend OK"
curl -f http://localhost:8001/health && echo "✅ RAG API OK"
curl -f http://localhost:11434/api/tags && echo "✅ Ollama OK"

# For Docker: Check containers
docker compose ps
```

---

## 🎉 Success!

If you see:
- ✅ All services responding
- ✅ Frontend accessible at http://localhost:3000
- ✅ No error messages

You're ready to start using LocalGPT!

### What's Next?

1. **📚 Upload Documents**: Add files to create an index
2. **💬 Start Chatting**: Ask questions about your documents
3. **🔧 Customize**: Try `RAG_CONFIG_MODE=fast`, different models, other retrieval modes
4. **📖 Learn More**: Check the documentation below

### 📁 Key Files

```
localGPT/
├── 🐳 start-docker.sh           # Docker deployment script
├── 🏃 run_system.py             # Direct development launcher
├── 🩺 system_health_check.py    # Deep system verification
├── 🛠️ create_index_script.py    # Interactive / batch index creation
├── 📋 requirements.txt          # Python dependencies
├── 📦 package.json              # Node.js dependencies
├── ⚙️  .env.example              # Every environment variable with its default
├── 📁 Documentation/            # Complete documentation
├── 📁 rag_system/               # RAG API, agent, pipelines (config in main.py)
├── 📁 backend/                  # Gateway server + SQLite database
└── 📁 src/                      # Next.js frontend
```

### 📖 Additional Resources

- **🏗️ Architecture**: See `Documentation/architecture_overview.md`
- **🔧 Configuration**: See `Documentation/system_overview.md`
- **🚀 Deployment**: See `Documentation/deployment_guide.md`
- **🐳 Docker**: See `Documentation/docker_usage.md`
- **🐛 Troubleshooting**: See `DOCKER_TROUBLESHOOTING.md`

---

## 🛠️ Indexing Without the UI

### Built-in CLI

```bash
# Index a file or a whole directory with the 'default' profile
python -m rag_system.main index ./my_documents

# Speed-optimised profile
python -m rag_system.main index ./my_documents --mode fast

# Ask one question and print the JSON result
python -m rag_system.main chat "What are the key findings?" --mode default

# Start the RAG API (same as `python -m rag_system.api_server`)
python -m rag_system.main api --port 8001
```

`index` walks a directory for `.pdf`, `.docx`, `.html`, `.htm`, `.md` and `.txt`.
It writes into the profile's shared table (`text_pages_v4`), not into a per-index
table, so indexes built this way are not listed in the web UI.

### Interactive / Batch Script

For an index the web UI can see, use `create_index_script.py` — it creates the
database row, uploads the document records and writes to `text_pages_<index_id>`:

```bash
# Guided prompts
python create_index_script.py

# Write a template, edit it, then run it
python create_index_script.py --create-sample     # writes index_config.sample.json
python create_index_script.py --batch index_config.sample.json

# Use a custom pipeline config instead of PIPELINE_CONFIGS["default"]
python create_index_script.py --config my_pipeline.json
```

The batch file looks like this — replace the placeholder paths with your own
absolute paths:

```json
{
  "index_name": "Sample Batch Index",
  "index_description": "Example batch index configuration",
  "documents": [
    "/absolute/path/to/first.pdf",
    "/absolute/path/to/second.pdf"
  ],
  "processing": {
    "chunk_size": 512,
    "enable_enrich": true,
    "enable_latechunk": true,
    "enable_docling": true,
    "embedding_model": "microsoft/harrier-oss-v1-0.6b",
    "enrich_model": "qwen3.5:4b",
    "retrieval_mode": "hybrid",
    "window_size": 2
  }
}
```

Both paths:
- ✅ Parse documents with Docling (OCR fallback for scanned PDFs)
- ✅ Chunk, optionally enrich, and embed
- ✅ Write vectors plus a native full-text index into LanceDB
- ✅ Generate a document overview used by the query router

The script exits non-zero on failure and deletes the half-created index row.

---

**Happy RAG-ing! 🚀**

# ⚡ Quick Start Guide - RAG System

_Get up and running in 5 minutes!_

---

## 🚀 Choose Your Deployment Method

### Option 1: Docker Deployment (Production Ready) 🐳

Best for: Production deployments, isolated environments, easy scaling

### Option 2: Direct Development (Developer Friendly) 💻  

Best for: Development, customization, debugging, faster iteration

---

## 🐳 Docker Deployment

### Prerequisites
- Docker Desktop installed and running
- 8GB+ RAM available
- Internet connection

### Step 1: Clone and Setup

```bash
# Clone repository
git clone <your-repository-url>
cd rag_system_old

# Ensure Docker is running
docker version
```

### Step 2: Install Ollama Locally

**Even with Docker, Ollama runs locally for better performance:**

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama (in one terminal)
ollama serve

# Install models (in another terminal)
ollama pull qwen3:0.6b
ollama pull qwen3:8b
```

### Step 3: Start Docker Containers

```bash
# Start all containers
./start-docker.sh

# Or manually:
docker compose --env-file docker.env up --build -d
```

### Step 4: Verify Deployment

```bash
# Check container status
docker compose ps

# Test endpoints
curl http://localhost:3000      # Frontend
curl http://localhost:8000/health  # Backend  
curl http://localhost:8001/models  # RAG API
```

### Step 5: Access Application

Open your browser to: **http://localhost:3000**

---

## 💻 Direct Development

### Prerequisites
- Python 3.8+
- Node.js 16+ and npm
- 8GB+ RAM available

### Step 1: Clone and Install Dependencies

```bash
# Clone repository
git clone <your-repository-url>
cd rag_system_old

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
ollama pull qwen3:0.6b
ollama pull qwen3:8b
```

### Step 3: Start the System

```bash
# Start all components with one command
python run_system.py
```

**Or start components manually in separate terminals:**

```bash
# Terminal 1: RAG API
python -m rag_system.api_server

# Terminal 2: Backend
cd backend && python server.py

# Terminal 3: Frontend
npm run dev
```

### Step 4: Verify Installation

```bash
# Check system health
python system_health_check.py

# Test endpoints
curl http://localhost:3000      # Frontend
curl http://localhost:8000/health  # Backend
curl http://localhost:8001/models  # RAG API
```

### Step 5: Access Application

Open your browser to: **http://localhost:3000**

---

## 🎯 First Use Guide

### 1. Create a Chat Session
- Click "New Chat" in the interface
- Give your session a descriptive name

### 2. Upload Documents
- Click "Create New Index" button
- Upload PDF files from your computer
- Configure processing options:
  - **Chunk Size**: 512 (recommended)
  - **Embedding Model**: Qwen/Qwen3-Embedding-0.6B
  - **Enable Enrichment**: Yes
- Click "Build Index" and wait for processing

### 3. Start Chatting
- Select your built index
- Ask questions about your documents:
  - "What is this document about?"
  - "Summarize the key points"
  - "What are the main findings?"
  - "Compare the arguments in section 3 and 5"

---

## 🔧 Management Commands

### Docker Commands

```bash
# Container management
./start-docker.sh                    # Start all containers
./start-docker.sh stop              # Stop all containers
./start-docker.sh logs              # View logs
./start-docker.sh status            # Check status

# Manual Docker Compose
docker compose ps                    # Check status
docker compose logs -f              # Follow logs
docker compose down                 # Stop containers
docker compose up --build -d        # Rebuild and start
```

### Direct Development Commands

```bash
# System management
python run_system.py               # Start all services
python system_health_check.py      # Check system health

# Individual components
python -m rag_system.api_server    # RAG API only
cd backend && python server.py     # Backend only
npm run dev                         # Frontend only

# Stop: Press Ctrl+C in terminal running services
```

---

## 🆘 Quick Troubleshooting

### Docker Issues

**Containers not starting?**
```bash
# Check Docker daemon
docker version

# Restart Docker Desktop and try again
./start-docker.sh
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
python --version  # Should be 3.8+

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**Node.js errors?**
```bash
# Check Node version
node --version    # Should be 16+

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

# Recommended: 16GB+ RAM for optimal performance
```

---

## 📊 System Verification

Run this comprehensive check:

```bash
# Check all endpoints
curl -f http://localhost:3000 && echo "✅ Frontend OK"
curl -f http://localhost:8000/health && echo "✅ Backend OK"  
curl -f http://localhost:8001/models && echo "✅ RAG API OK"
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

1. **📚 Upload Documents**: Add your PDF files to create indexes
2. **💬 Start Chatting**: Ask questions about your documents
3. **🔧 Customize**: Explore different models and settings
4. **📖 Learn More**: Check the full documentation below

### 📁 Key Files

```
rag-system/
├── 🐳 start-docker.sh           # Docker deployment script
├── 🏃 run_system.py             # Direct development launcher
├── 🩺 system_health_check.py    # System verification
├── 📋 requirements.txt          # Python dependencies
├── 📦 package.json              # Node.js dependencies
├── 📁 Documentation/            # Complete documentation
└── 📁 rag_system/              # Core system code
```

### 📖 Additional Resources

- **🏗️ Architecture**: See `Documentation/architecture_overview.md`
- **🔧 Configuration**: See `Documentation/system_overview.md`  
- **🚀 Deployment**: See `Documentation/deployment_guide.md`
- **🐛 Troubleshooting**: See `DOCKER_TROUBLESHOOTING.md`

---

**Happy RAG-ing! 🚀** 

---

## 🛠️ Indexing Scripts

The repository includes several convenient scripts for document indexing:

### Simple Index Creation Script

For quick document indexing without the UI:

```bash
# Basic usage
./simple_create_index.sh "Index Name" "document.pdf"

# Multiple documents
./simple_create_index.sh "Research Papers" "paper1.pdf" "paper2.pdf" "notes.txt"

# Using wildcards
./simple_create_index.sh "Invoice Collection" ./invoices/*.pdf
```

**Supported file types**: PDF, TXT, DOCX, MD

### Batch Indexing Script

For processing large document collections:

```bash
# Using the Python batch indexing script
python demo_batch_indexing.py

# Or using the direct indexing script
python create_index_script.py
```

These scripts automatically:
- ✅ Check prerequisites (Ollama running, Python dependencies)
- ✅ Validate document formats
- ✅ Create database entries
- ✅ Process documents with the RAG pipeline
- ✅ Generate searchable indexes

--- 
# 📦 RAG System Installation Guide

_Last updated: 2025-01-07_

This guide provides step-by-step instructions for installing and setting up the RAG system using either Docker or direct development approaches.

---

## 🎯 Installation Options

### Option 1: Docker Deployment (Production Ready) 🐳
- **Best for**: Production environments, isolated setups, easy management
- **Requirements**: Docker Desktop + Local Ollama
- **Setup time**: ~10 minutes

### Option 2: Direct Development (Developer Friendly) 💻
- **Best for**: Development, customization, debugging
- **Requirements**: Python + Node.js + Ollama
- **Setup time**: ~15 minutes

---

## 1. Prerequisites

### 1.1 System Requirements

#### **Minimum Requirements**
- **CPU**: 4 cores, 2.5GHz+
- **RAM**: 8GB (16GB recommended)
- **Storage**: 50GB free space
- **OS**: macOS 10.15+, Ubuntu 20.04+, Windows 10+

#### **Recommended Requirements**
- **CPU**: 8+ cores, 3.0GHz+
- **RAM**: 32GB+ (for large models)
- **Storage**: 200GB+ SSD
- **GPU**: NVIDIA GPU with 8GB+ VRAM (optional)

### 1.2 Common Dependencies

**Required for both approaches:**
- **Ollama**: AI model runtime (always required)
- **Git**: 2.30+ for cloning repository

**Docker-specific:**
- **Docker Desktop**: 24.0+ with Docker Compose

**Direct Development-specific:**
- **Python**: 3.8+ 
- **Node.js**: 16+ with npm

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

### 2.2 Configure Ollama

```bash
# Start Ollama server
ollama serve

# In another terminal, install required models
ollama pull qwen3:0.6b      # Fast model (650MB)
ollama pull qwen3:8b        # High-quality model (4.7GB)

# Verify models are installed
ollama list

# Test Ollama
ollama run qwen3:0.6b "Hello, how are you?"
```

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

### 3.2 Clone and Setup RAG System

```bash
# Clone repository
git clone <your-repository-url>
cd rag_system_old

# Verify Ollama is running
curl http://localhost:11434/api/tags

# Start Docker containers
./start-docker.sh

# Wait for containers to start (2-3 minutes)
sleep 120

# Verify deployment
./start-docker.sh status
```

### 3.3 Test Docker Deployment

```bash
# Test all endpoints
curl -f http://localhost:3000 && echo "✅ Frontend OK"
curl -f http://localhost:8000/health && echo "✅ Backend OK"
curl -f http://localhost:8001/models && echo "✅ RAG API OK"
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
git clone https://github.com/your-org/rag-system.git
cd rag-system

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
```

#### **Node.js Setup:**
```bash
# Install Node.js dependencies
npm install

# Verify Node.js setup
node --version  # Should be 16+
npm --version
npm list --depth=0
```

### 4.2 Start Direct Development

```bash
# Ensure Ollama is running
curl http://localhost:11434/api/tags

# Start all components with one command
python run_system.py

# Or start components manually in separate terminals:
# Terminal 1: python -m rag_system.api_server
# Terminal 2: cd backend && python server.py  
# Terminal 3: npm run dev
```

### 4.3 Test Direct Development

```bash
# Check system health
python system_health_check.py

# Test endpoints
curl -f http://localhost:3000 && echo "✅ Frontend OK"
curl -f http://localhost:8000/health && echo "✅ Backend OK"
curl -f http://localhost:8001/models && echo "✅ RAG API OK"

# Access the application
open http://localhost:3000
```

---

## 5. Detailed Installation Steps

### 5.1 Repository Setup

```bash
# Clone repository
git clone https://github.com/your-org/rag-system.git
cd rag-system

# Check repository structure
ls -la

# Create required directories
mkdir -p lancedb index_store shared_uploads logs backend
touch backend/chat_data.db

# Set permissions
chmod -R 755 lancedb index_store shared_uploads
chmod 664 backend/chat_data.db
```

### 5.2 Configuration

#### **Environment Variables**
For Docker (automatic via `docker.env`):
```bash
OLLAMA_HOST=http://host.docker.internal:11434
NODE_ENV=production
RAG_API_URL=http://rag-api:8001
NEXT_PUBLIC_API_URL=http://localhost:8000
```

For Direct Development (set automatically by `run_system.py`):
```bash
OLLAMA_HOST=http://localhost:11434
RAG_API_URL=http://localhost:8001
NEXT_PUBLIC_API_URL=http://localhost:8000
```

#### **Model Configuration**
The system defaults to these models:
- **Embedding**: `Qwen/Qwen3-Embedding-0.6B` (1024 dimensions)
- **Generation**: `qwen3:0.6b` for fast responses, `qwen3:8b` for quality
- **Reranking**: Built-in cross-encoder

### 5.3 Database Initialization

```bash
# Initialize SQLite database
python -c "
from backend.database import ChatDatabase
db = ChatDatabase()
db.init_database()
print('✅ Database initialized')
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

# Universal health check
curl -f http://localhost:3000 && echo "✅ Frontend OK"
curl -f http://localhost:8000/health && echo "✅ Backend OK"
curl -f http://localhost:8001/models && echo "✅ RAG API OK"
curl -f http://localhost:11434/api/tags && echo "✅ Ollama OK"
```

#### **RAG System Test:**
```bash
# Test RAG system initialization
python -c "
from rag_system.main import get_agent
agent = get_agent('default')
print('✅ RAG System initialized successfully')
"

# Test embedding generation
python -c "
from rag_system.main import get_agent
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

# Test models endpoint
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

# If fails, restart Ollama
pkill ollama
ollama serve

# Reinstall models if needed
ollama pull qwen3:0.6b
ollama pull qwen3:8b
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
python --version  # Should be 3.8+

# Check virtual environment
which python
pip list | grep torch

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

#### **Node.js Issues:**
```bash
# Check Node version
node --version  # Should be 16+

# Clear and reinstall
rm -rf node_modules package-lock.json
npm install
```

### 7.2 Performance Issues

#### **Memory Problems:**
```bash
# Check system memory
free -h  # Linux
vm_stat  # macOS

# For Docker: Increase memory allocation
# Docker Desktop → Settings → Resources → Memory → 8GB+

# Use smaller models
ollama pull qwen3:0.6b  # Instead of qwen3:8b
```

#### **Slow Performance:**
- Use SSD storage for databases (`lancedb/`, `shared_uploads/`)
- Increase CPU cores if possible
- Close unnecessary applications
- Use smaller batch sizes in configuration

---

## 8. Post-Installation Setup

### 8.1 Model Optimization

```bash
# Install additional models (optional)
ollama pull nomic-embed-text        # Alternative embedding model
ollama pull llama3.1:8b            # Alternative generation model

# Test model switching
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Hello", "model": "qwen3:8b"}'
```

### 8.2 Security Configuration

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

---

## 9. Success Criteria

### 9.1 Installation Complete When:

- ✅ All health checks pass without errors
- ✅ Frontend loads at http://localhost:3000
- ✅ All models are installed and responding
- ✅ You can create document indexes
- ✅ You can chat with uploaded documents
- ✅ No error messages in logs/terminal

### 9.2 Performance Benchmarks

**Acceptable Performance:**
- System startup: < 5 minutes
- Index creation: < 2 minutes per 100MB document
- Query response: < 30 seconds
- Memory usage: < 8GB total

**Optimal Performance:**
- System startup: < 2 minutes
- Index creation: < 1 minute per 100MB document
- Query response: < 10 seconds
- Memory usage: < 4GB total

---

## 10. Next Steps

### 10.1 Getting Started

1. **Upload Documents**: Create your first index with PDF documents
2. **Explore Features**: Try different query types and models
3. **Customize**: Adjust model settings and chunk sizes
4. **Scale**: Add more documents and create multiple indexes

### 10.2 Additional Resources

- **Quick Start**: See `Documentation/quick_start.md`
- **Docker Usage**: See `Documentation/docker_usage.md`
- **System Architecture**: See `Documentation/architecture_overview.md`
- **API Reference**: See `Documentation/api_reference.md`

---

**Congratulations! 🎉** Your RAG system is now ready to use. Visit http://localhost:3000 to start chatting with your documents. 
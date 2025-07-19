# 🐳 LocalGPT Docker Deployment Guide

This guide covers running LocalGPT using Docker containers with local Ollama for optimal performance.

## 🚀 Quick Start

### Complete Setup (5 Minutes)
```bash
# 1. Install Ollama locally
curl -fsSL https://ollama.ai/install.sh | sh

# 2. Start Ollama server
ollama serve

# 3. Install required models (in another terminal)
ollama pull qwen3:0.6b
ollama pull qwen3:8b

# 4. Clone and start LocalGPT
git clone https://github.com/your-org/rag-system.git
cd rag-system
./start-docker.sh

# 5. Access the application
open http://localhost:3000
```

## 📋 Prerequisites

- **Docker Desktop** installed and running
- **Ollama** installed locally (required for best performance)
- **8GB+ RAM** (16GB recommended for larger models)
- **10GB+ free disk space**

## 🏗️ Architecture

### Current Setup (Local Ollama + Docker Containers)
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │────│    Backend      │────│    RAG API      │
│  (Container)    │    │  (Container)    │    │  (Container)    │
│   Port: 3000    │    │   Port: 8000    │    │   Port: 8001    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        │ API calls
                                                        ▼
                                               ┌─────────────────┐
                                               │     Ollama      │
                                               │ (Local/Host)    │
                                               │   Port: 11434   │
                                               └─────────────────┘
```

**Why Local Ollama?**
- ✅ Better performance (direct GPU access)
- ✅ Simpler setup (one less container)
- ✅ Easier model management
- ✅ More reliable connection

## 🛠️ Container Details

### Frontend Container (rag-frontend)
- **Image**: Custom Node.js 18 build
- **Port**: 3000
- **Purpose**: Next.js web interface
- **Health Check**: HTTP GET to /
- **Memory**: ~500MB

### Backend Container (rag-backend) 
- **Image**: Custom Python 3.11 build
- **Port**: 8000
- **Purpose**: Session management, chat history, API gateway
- **Health Check**: HTTP GET to /health
- **Memory**: ~300MB

### RAG API Container (rag-api)
- **Image**: Custom Python 3.11 build
- **Port**: 8001
- **Purpose**: Document indexing, retrieval, AI processing
- **Health Check**: HTTP GET to /models
- **Memory**: ~2GB (varies with model usage)

## 📂 Volume Mounts & Data

### Persistent Data
- `./lancedb/` → Vector database storage
- `./index_store/` → Document indexes and metadata
- `./shared_uploads/` → Uploaded document files
- `./backend/chat_data.db` → SQLite chat history database

### Shared Between Containers
All containers share access to document storage and databases through bind mounts.

## 🔧 Configuration

### Environment Variables (docker.env)
```bash
# Ollama Configuration
OLLAMA_HOST=http://host.docker.internal:11434

# Service Configuration  
NODE_ENV=production
RAG_API_URL=http://rag-api:8001
NEXT_PUBLIC_API_URL=http://localhost:8000

# Database Paths (inside containers)
DATABASE_PATH=/app/backend/chat_data.db
LANCEDB_PATH=/app/lancedb
UPLOADS_PATH=/app/shared_uploads
```

### Model Configuration
The system uses these models by default:
- **Embedding**: `Qwen/Qwen3-Embedding-0.6B` (1024 dimensions)
- **Generation**: `qwen3:0.6b` (fast) or `qwen3:8b` (high quality)
- **Reranking**: Built-in cross-encoder

## 🎯 Management Commands

### Start/Stop Services
```bash
# Start all services
./start-docker.sh

# Stop all services
./start-docker.sh stop

# Restart services
./start-docker.sh stop && ./start-docker.sh
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

# Rebuild specific service
docker compose build --no-cache rag-api
docker compose up -d rag-api
```

### Health Checks
```bash
# Test all endpoints
curl -f http://localhost:3000 && echo "✅ Frontend OK"
curl -f http://localhost:8000/health && echo "✅ Backend OK"
curl -f http://localhost:8001/models && echo "✅ RAG API OK"
curl -f http://localhost:11434/api/tags && echo "✅ Ollama OK"
```

## 🐞 Debugging

### Access Container Shells
```bash
# RAG API container (most debugging happens here)
docker compose exec rag-api bash

# Backend container
docker compose exec backend bash

# Frontend container
docker compose exec frontend sh
```

### Common Debug Commands
```bash
# Test RAG system initialization
docker compose exec rag-api python -c "
from rag_system.main import get_agent
agent = get_agent('default')
print('✅ RAG System OK')
"

# Test Ollama connection from container
docker compose exec rag-api curl http://host.docker.internal:11434/api/tags

# Check environment variables
docker compose exec rag-api env | grep OLLAMA

# View Python packages
docker compose exec rag-api pip list | grep -E "(torch|transformers|lancedb)"
```

### Resource Monitoring
```bash
# Monitor container resources
docker stats

# Check disk usage
docker system df
df -h ./lancedb ./shared_uploads

# Check memory usage by service
docker stats --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"
```

## 🚨 Troubleshooting

### Common Issues

#### Container Won't Start
```bash
# Check logs for specific error
docker compose logs [service-name]

# Rebuild from scratch
./start-docker.sh stop
docker system prune -f
./start-docker.sh

# Check for port conflicts
lsof -i :3000 -i :8000 -i :8001
```

#### Can't Connect to Ollama
```bash
# Verify Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama
pkill ollama
ollama serve

# Test from container
docker compose exec rag-api curl http://host.docker.internal:11434/api/tags
```

#### Memory Issues
```bash
# Check memory usage
docker stats --no-stream
free -h  # On host

# Increase Docker memory limit
# Docker Desktop → Settings → Resources → Memory → 8GB+

# Use smaller models
ollama pull qwen3:0.6b  # Instead of qwen3:8b
```

#### Frontend Build Errors
```bash
# Clean build
docker compose build --no-cache frontend
docker compose up -d frontend

# Check frontend logs
docker compose logs frontend
```

#### Database/Storage Issues
```bash
# Check file permissions
ls -la backend/chat_data.db
ls -la lancedb/

# Reset permissions
chmod 664 backend/chat_data.db
chmod -R 755 lancedb/ shared_uploads/

# Test database access
docker compose exec backend sqlite3 /app/backend/chat_data.db ".tables"
```

### Performance Issues

#### Slow Response Times
- Use faster models: `qwen3:0.6b` instead of `qwen3:8b`
- Increase Docker memory allocation
- Ensure SSD storage for databases
- Monitor with `docker stats`

#### High Memory Usage
- Reduce batch sizes in configuration
- Use smaller embedding models
- Clear unused Docker resources: `docker system prune`

### Complete Reset
```bash
# Nuclear option - reset everything
./start-docker.sh stop
docker system prune -a --volumes
rm -rf lancedb/* shared_uploads/* backend/chat_data.db
./start-docker.sh
```

## 🏆 Success Criteria

Your Docker deployment is successful when:

- ✅ `./start-docker.sh status` shows all containers healthy
- ✅ All health checks pass (see commands above)  
- ✅ You can access http://localhost:3000
- ✅ You can upload documents and create indexes
- ✅ You can chat with your documents
- ✅ No errors in container logs

### Performance Benchmarks

**Good Performance:**
- Container startup: < 2 minutes
- Index creation: < 2 min per 100MB document
- Query response: < 30 seconds
- Memory usage: < 4GB total containers

**Optimal Performance:**
- Container startup: < 1 minute
- Index creation: < 1 min per 100MB document  
- Query response: < 10 seconds
- Memory usage: < 2GB total containers

## 📚 Additional Resources

- **Detailed Troubleshooting**: See `DOCKER_TROUBLESHOOTING.md`
- **Complete Documentation**: See `Documentation/docker_usage.md`
- **System Architecture**: See `Documentation/architecture_overview.md`
- **Direct Development**: See main `README.md` for non-Docker setup

---

**Happy Dockerizing! 🐳** Need help? Check the troubleshooting guide or open an issue. 
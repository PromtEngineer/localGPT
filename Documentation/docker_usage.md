# 🐳 Docker Usage Guide - RAG System

_Last updated: 2025-01-07_

This guide provides practical Docker commands and procedures for running the RAG system in containerized environments with local Ollama.

---

## 📋 Prerequisites

### Required Setup
- Docker Desktop installed and running
- Ollama installed locally (even for Docker deployment)
- 8GB+ RAM available

### Architecture Overview
```
┌─────────────────────────────────────┐
│           Docker Containers        │
├─────────────────────────────────────┤
│ Frontend (Port 3000)               │
│ Backend (Port 8000)                │
│ RAG API (Port 8001)                │
└─────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────┐
│         Local System               │
├─────────────────────────────────────┤
│ Ollama Server (Port 11434)         │
└─────────────────────────────────────┘
```

---

## 1. Quick Start Commands

### Step 1: Clone and Setup

```bash
# Clone repository
git clone <your-repository-url>
cd rag_system_old

# Verify Docker is running
docker version
```

### Step 2: Install and Configure Ollama (Required)

**⚠️ Important**: Even with Docker, Ollama must be installed locally for optimal performance.

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama (in one terminal)
ollama serve

# Install required models (in another terminal)
ollama pull qwen3:0.6b      # Fast model (650MB)
ollama pull qwen3:8b        # High-quality model (4.7GB)

# Verify models are installed
ollama list

# Test Ollama connection
curl http://localhost:11434/api/tags
```

### Step 3: Start Docker Containers

```bash
# Start all containers
./start-docker.sh

# Stop all containers
./start-docker.sh stop

# View logs
./start-docker.sh logs

# Check status
./start-docker.sh status

# Restart containers
./start-docker.sh stop
./start-docker.sh
```

### 1.2 Service Access

Once running, access the system at:
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000  
- **RAG API**: http://localhost:8001
- **Ollama**: http://localhost:11434

---

## 2. Container Management

### 2.1 Using the Convenience Script

```bash
# Start all containers
./start-docker.sh

# Stop all containers
./start-docker.sh stop

# View logs
./start-docker.sh logs

# Check status
./start-docker.sh status

# Restart containers
./start-docker.sh stop
./start-docker.sh
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
docker compose up --build -d
```

### 2.3 Individual Service Management

```bash
# Start specific service
docker compose up -d frontend
docker compose up -d backend
docker compose up -d rag-api

# Restart specific service
docker compose restart rag-api

# Stop specific service
docker compose stop backend

# View specific service logs
docker compose logs -f rag-api
```

---

## 3. Development Workflow

### 3.1 Code Changes

```bash
# After frontend changes
docker compose restart frontend

# After backend changes  
docker compose restart backend

# After RAG system changes
docker compose restart rag-api

# Rebuild after dependency changes
docker compose build --no-cache rag-api
docker compose up -d rag-api
```

### 3.2 Debugging Containers

```bash
# Access container shell
docker compose exec frontend sh
docker compose exec backend bash
docker compose exec rag-api bash

# Run commands in container
docker compose exec rag-api python -c "from rag_system.main import get_agent; print('✅ RAG System OK')"
docker compose exec backend curl http://localhost:8000/health

# Check environment variables
docker compose exec rag-api env | grep OLLAMA
```

### 3.3 Development vs Production

```bash
# Development mode (if docker-compose.dev.yml exists)
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Production mode (default)
docker compose --env-file docker.env up -d
```

---

## 4. Logging & Monitoring

### 4.1 Log Management

```bash
# View all logs
docker compose logs

# View specific service logs
docker compose logs frontend
docker compose logs backend
docker compose logs rag-api

# Follow logs in real-time
docker compose logs -f

# View last N lines
docker compose logs --tail=100

# View logs with timestamps
docker compose logs -t

# Save logs to file
docker compose logs > system.log 2>&1

# View logs since specific time
docker compose logs --since=2h
docker compose logs --since=2025-01-01T00:00:00
```

### 4.2 System Monitoring

```bash
# Monitor resource usage
docker stats

# Monitor specific containers
docker stats rag-frontend rag-backend rag-api

# Check container health
docker compose ps

# System information
docker system info
docker system df
```

---

## 5. Ollama Integration

### 5.1 Ollama Setup

```bash
# Install Ollama (one-time setup)
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama server
ollama serve

# Check Ollama status
curl http://localhost:11434/api/tags

# Install models
ollama pull qwen3:0.6b      # Fast model
ollama pull qwen3:8b        # High-quality model

# List installed models
ollama list
```

### 5.2 Ollama Management

```bash
# Check model status from container
docker compose exec rag-api curl http://host.docker.internal:11434/api/tags

# Test Ollama connection
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3:0.6b", "prompt": "Hello", "stream": false}'

# Monitor Ollama logs (if running with logs)
# Ollama logs appear in the terminal where you ran 'ollama serve'
```

### 5.3 Model Management

```bash
# Update models
ollama pull qwen3:0.6b
ollama pull qwen3:8b

# Remove unused models
ollama rm old-model-name

# Check model information
ollama show qwen3:0.6b
```

---

## 6. Data Management

### 6.1 Volume Management

```bash
# List volumes
docker volume ls

# View volume usage
docker system df -v

# Backup volumes
docker run --rm -v rag_system_old_lancedb:/data -v $(pwd)/backup:/backup alpine tar czf /backup/lancedb_backup.tar.gz -C /data .

# Clean unused volumes
docker volume prune
```

### 6.2 Database Management

```bash
# Access SQLite database
docker compose exec backend sqlite3 /app/backend/chat_data.db

# Backup database
cp backend/chat_data.db backup/chat_data_$(date +%Y%m%d).db

# Check LanceDB tables from container
docker compose exec rag-api python -c "
import lancedb
db = lancedb.connect('/app/lancedb')
print('Tables:', db.table_names())
"
```

### 6.3 File Management

```bash
# Access shared files
docker compose exec rag-api ls -la /app/shared_uploads

# Copy files to/from containers
docker cp local_file.pdf rag-api:/app/shared_uploads/
docker cp rag-api:/app/shared_uploads/file.pdf ./local_file.pdf

# Check disk usage
docker compose exec rag-api df -h
```

---

## 7. Troubleshooting

### 7.1 Common Issues

#### Container Won't Start
```bash
# Check Docker daemon
docker version

# Check for port conflicts
lsof -i :3000 -i :8000 -i :8001

# Check container logs
docker compose logs [service-name]

# Restart Docker Desktop
# macOS/Windows: Restart Docker Desktop
# Linux: sudo systemctl restart docker
```

#### Ollama Connection Issues
```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama
pkill ollama
ollama serve

# Check from container
docker compose exec rag-api curl http://host.docker.internal:11434/api/tags
```

#### Performance Issues
```bash
# Check resource usage
docker stats

# Increase Docker memory (Docker Desktop Settings)
# Recommended: 8GB+ for Docker

# Check container health
docker compose ps
```

### 7.2 Reset and Clean

```bash
# Stop everything
./start-docker.sh stop

# Clean containers and images
docker system prune -a

# Clean volumes (⚠️ deletes data)
docker volume prune

# Complete reset (⚠️ deletes everything)
docker compose down -v
docker system prune -a --volumes
```

### 7.3 Health Checks

```bash
# Comprehensive health check
curl -f http://localhost:3000 && echo "✅ Frontend OK"
curl -f http://localhost:8000/health && echo "✅ Backend OK"
curl -f http://localhost:8001/models && echo "✅ RAG API OK"
curl -f http://localhost:11434/api/tags && echo "✅ Ollama OK"

# Check all container status
docker compose ps

# Test model loading
docker compose exec rag-api python -c "
from rag_system.main import get_agent
agent = get_agent('default')
print('✅ RAG System initialized successfully')
"
```

---

## 8. Advanced Usage

### 8.1 Production Deployment

```bash
# Use production environment
export NODE_ENV=production

# Start with resource limits
docker compose --env-file docker.env up -d

# Enable automatic restarts
docker update --restart unless-stopped $(docker ps -q)
```

### 8.2 Scaling

```bash
# Scale specific services
docker compose up -d --scale backend=2 --scale rag-api=2

# Use Docker Swarm for clustering
docker swarm init
docker stack deploy -c docker-compose.yml rag-system
```

### 8.3 Security

```bash
# Scan images for vulnerabilities
docker scout cves rag-frontend
docker scout cves rag-backend
docker scout cves rag-api

# Update base images
docker compose build --no-cache --pull
```

---

## 9. Configuration

### 9.1 Environment Variables

The system uses `docker.env` for configuration:

```bash
# Ollama configuration
OLLAMA_HOST=http://host.docker.internal:11434

# Service configuration
NODE_ENV=production
RAG_API_URL=http://rag-api:8001
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### 9.2 Custom Configuration

```bash
# Create custom environment file
cp docker.env docker.custom.env

# Edit custom configuration
nano docker.custom.env

# Use custom configuration
docker compose --env-file docker.custom.env up -d
```

---

## 10. Success Checklist

Your Docker deployment is successful when:

- ✅ All containers are running: `docker compose ps`
- ✅ Ollama is accessible: `curl http://localhost:11434/api/tags`
- ✅ Frontend loads: `curl http://localhost:3000`
- ✅ Backend responds: `curl http://localhost:8000/health`
- ✅ RAG API works: `curl http://localhost:8001/models`
- ✅ You can create indexes and chat with documents

### Performance Expectations

**Acceptable Performance:**
- Container startup: < 2 minutes
- Memory usage: < 4GB Docker containers + Ollama
- Response time: < 30 seconds for complex queries

**Optimal Performance:**
- Container startup: < 1 minute  
- Memory usage: < 2GB Docker containers + Ollama
- Response time: < 10 seconds for complex queries

---

**Happy Containerizing! 🐳** 
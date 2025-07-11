# 🐳 Docker Troubleshooting Guide - LocalGPT

_Last updated: 2025-01-07_

This guide helps diagnose and fix Docker-related issues with LocalGPT's containerized deployment.

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
curl -f http://localhost:8001/models && echo "✅ RAG API OK"
curl -f http://localhost:11434/api/tags && echo "✅ Ollama OK"
```

### Expected Success Output
```
✅ Frontend OK
✅ Backend OK
✅ RAG API OK
✅ Ollama OK
```

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
# Check Docker service status
sudo systemctl status docker

# Restart Docker service
sudo systemctl restart docker

# Enable auto-start
sudo systemctl enable docker

# Test connection
docker version
```

#### Solution C: Hard Reset
```bash
# Kill all Docker processes
sudo pkill -f docker

# Remove socket files
sudo rm -f /var/run/docker.sock
sudo rm -f /Users/prompt/.docker/run/docker.sock  # macOS

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
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not running, start it
ollama serve

# Install required models
ollama pull qwen3:0.6b
ollama pull qwen3:8b
```

#### Solution B: Test from Container
```bash
# Test Ollama connection from RAG API container
docker compose exec rag-api curl http://host.docker.internal:11434/api/tags

# If this fails, check Docker network settings
docker network ls
docker network inspect rag_system_old_default
```

#### Solution C: Alternative Ollama Host
```bash
# Edit docker.env to use different host
echo "OLLAMA_HOST=http://172.17.0.1:11434" >> docker.env

# Or use IP address
echo "OLLAMA_HOST=http://$(ipconfig getifaddr en0):11434" >> docker.env  # macOS
```

### 3. Container Build Failures

#### Problem: Frontend build fails
```
ERROR: Failed to build frontend container
```

#### Solution: Clean Build
```bash
# Stop containers
./start-docker.sh stop

# Clean Docker cache
docker system prune -f
docker builder prune -f

# Rebuild frontend only
docker compose build --no-cache frontend
docker compose up -d frontend

# Check logs
docker compose logs frontend
```

#### Problem: Python package installation fails
```
ERROR: Could not install packages due to an EnvironmentError
```

#### Solution: Update Dependencies
```bash
# Check requirements file exists
ls -la requirements-docker.txt

# Test package installation locally
pip install -r requirements-docker.txt --dry-run

# Rebuild with updated base image
docker compose build --no-cache --pull rag-api
```

### 4. Port Conflicts

#### Problem: "Port already in use"
```
Error starting userland proxy: listen tcp4 0.0.0.0:3000: bind: address already in use
```

#### Solution: Find and Kill Conflicting Processes
```bash
# Check what's using the ports
lsof -i :3000 -i :8000 -i :8001

# Kill specific processes
pkill -f "npm run dev"      # Frontend
pkill -f "server.py"        # Backend
pkill -f "api_server"       # RAG API

# Or kill by port
sudo kill -9 $(lsof -t -i:3000)
sudo kill -9 $(lsof -t -i:8000)
sudo kill -9 $(lsof -t -i:8001)

# Restart containers
./start-docker.sh
```

### 5. Memory Issues

#### Problem: Containers crash due to OOM (Out of Memory)
```
Container killed due to memory limit
```

#### Solution: Increase Docker Memory
```bash
# Check current memory usage
docker stats --no-stream

# Increase Docker Desktop memory allocation
# Docker Desktop → Settings → Resources → Memory → 8GB+

# Monitor memory usage
docker stats

# Use smaller models if needed
ollama pull qwen3:0.6b  # Instead of qwen3:8b
```

#### Problem: System running slow
```bash
# Check host memory
free -h  # Linux
vm_stat  # macOS

# Clean up Docker resources
docker system prune -f
docker volume prune -f
```

### 6. Volume Mount Issues

#### Problem: Permission denied accessing files
```
Permission denied: /app/lancedb
```

#### Solution: Fix Permissions
```bash
# Create directories if they don't exist
mkdir -p lancedb index_store shared_uploads backend

# Fix permissions
chmod -R 755 lancedb index_store shared_uploads
chmod 664 backend/chat_data.db

# Check ownership
ls -la lancedb/ shared_uploads/ backend/

# Reset permissions if needed
sudo chown -R $USER:$USER lancedb shared_uploads backend
```

#### Problem: Database file not found
```
No such file or directory: '/app/backend/chat_data.db'
```

#### Solution: Initialize Database
```bash
# Create empty database file
touch backend/chat_data.db

# Or initialize with schema
python -c "
from backend.database import ChatDatabase
db = ChatDatabase()
db.init_database()
print('Database initialized')
"

# Restart containers
./start-docker.sh stop
./start-docker.sh
```

---

## 🔍 Advanced Debugging

### Container-Level Debugging

#### Access Container Shells
```bash
# RAG API container (most issues happen here)
docker compose exec rag-api bash

# Check environment variables
docker compose exec rag-api env | grep -E "(OLLAMA|RAG|NODE)"

# Test Python imports
docker compose exec rag-api python -c "
import sys
print('Python version:', sys.version)
from rag_system.main import get_agent
print('✅ RAG system imports work')
"

# Backend container
docker compose exec backend bash
python -c "
from backend.database import ChatDatabase
print('✅ Database imports work')
"

# Frontend container  
docker compose exec frontend sh
npm --version
node --version
```

#### Check Container Resources
```bash
# Monitor real-time resource usage
docker stats

# Check individual container health
docker compose ps
docker inspect rag-api --format='{{.State.Health.Status}}'

# View container configurations
docker compose config
```

#### Network Debugging
```bash
# Check network connectivity
docker compose exec rag-api ping backend
docker compose exec backend ping rag-api
docker compose exec rag-api ping host.docker.internal

# Check DNS resolution
docker compose exec rag-api nslookup host.docker.internal

# Test HTTP connections
docker compose exec rag-api curl -v http://backend:8000/health
docker compose exec rag-api curl -v http://host.docker.internal:11434/api/tags
```

### Log Analysis

#### Container Logs
```bash
# View all logs
./start-docker.sh logs

# Follow specific service logs
docker compose logs -f rag-api
docker compose logs -f backend
docker compose logs -f frontend

# Search for errors
docker compose logs rag-api 2>&1 | grep -i error
docker compose logs backend 2>&1 | grep -i "traceback\|error"

# Save logs to file
docker compose logs > docker-debug.log 2>&1
```

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

#### Test Individual Containers
```bash
# Test RAG API alone
docker build -f Dockerfile.rag-api -t test-rag-api .
docker run --rm -p 8001:8001 -e OLLAMA_HOST=http://host.docker.internal:11434 test-rag-api &
sleep 30
curl http://localhost:8001/models
pkill -f test-rag-api

# Test Backend alone
docker build -f Dockerfile.backend -t test-backend .
docker run --rm -p 8000:8000 test-backend &
sleep 30
curl http://localhost:8000/health
pkill -f test-backend
```

#### Integration Testing
```bash
# Full system test
./start-docker.sh

# Wait for all services to be ready
sleep 60

# Test complete workflow
curl -X POST http://localhost:8000/sessions \
  -H "Content-Type: application/json" \
  -d '{"title": "Test Session"}'

# Test document upload (if you have a test PDF)
# curl -X POST http://localhost:8000/upload -F "file=@test.pdf"

# Clean up
./start-docker.sh stop
```

### Automated Testing Script

Create `test-docker-health.sh`:
```bash
#!/bin/bash
set -e

echo "🐳 Docker Health Test Starting..."

# Start containers
./start-docker.sh

# Wait for services
echo "⏳ Waiting for services to start..."
sleep 60

# Test endpoints
echo "🔍 Testing endpoints..."
curl -f http://localhost:3000 && echo "✅ Frontend OK" || echo "❌ Frontend FAIL"
curl -f http://localhost:8000/health && echo "✅ Backend OK" || echo "❌ Backend FAIL"  
curl -f http://localhost:8001/models && echo "✅ RAG API OK" || echo "❌ RAG API FAIL"
curl -f http://localhost:11434/api/tags && echo "✅ Ollama OK" || echo "❌ Ollama FAIL"

# Test container health
echo "🔍 Checking container health..."
docker compose ps

echo "🎉 Health test complete!"
```

---

## 🔄 Recovery Procedures

### Complete System Reset

#### Soft Reset
```bash
# Stop containers
./start-docker.sh stop

# Clean up Docker resources
docker system prune -f

# Restart containers
./start-docker.sh
```

#### Hard Reset (⚠️ Deletes all data)
```bash
# Stop everything
./start-docker.sh stop

# Remove all containers, images, and volumes
docker system prune -a --volumes

# Remove local data (CAUTION: This deletes all your documents and chat history)
rm -rf lancedb/* shared_uploads/* backend/chat_data.db

# Rebuild from scratch
./start-docker.sh
```

#### Selective Reset

Reset only specific components:
```bash
# Reset just the database
./start-docker.sh stop
rm backend/chat_data.db
./start-docker.sh

# Reset just vector storage
./start-docker.sh stop
rm -rf lancedb/*
./start-docker.sh

# Reset just uploaded documents
rm -rf shared_uploads/*
```

---

## 📊 Performance Optimization

### Resource Monitoring
```bash
# Monitor containers continuously
watch -n 5 'docker stats --no-stream'

# Check disk usage
docker system df
du -sh lancedb shared_uploads backend

# Monitor host resources
htop  # Linux
top   # macOS/Windows
```

### Performance Tuning
```bash
# Use smaller models for better performance
ollama pull qwen3:0.6b  # Instead of qwen3:8b

# Reduce Docker memory if needed
# Docker Desktop → Settings → Resources → Memory

# Clean up regularly
docker system prune -f
docker volume prune -f
```

---

## 🆘 When All Else Fails

### Alternative Deployment Options

#### 1. Direct Development (No Docker)
```bash
# Stop Docker containers
./start-docker.sh stop

# Use direct development instead
python run_system.py
```

#### 2. Minimal Docker (RAG API only)
```bash
# Run only RAG API in Docker
docker build -f Dockerfile.rag-api -t rag-api .
docker run -p 8001:8001 rag-api

# Run other components directly
cd backend && python server.py &
npm run dev
```

#### 3. Hybrid Approach
```bash
# Run some services in Docker, others directly
docker compose up -d rag-api
cd backend && python server.py &
npm run dev
```

### Getting Help

#### Diagnostic Information to Collect
```bash
# System information
docker version
docker compose version
uname -a

# Container information
docker compose ps
docker compose config

# Resource information
docker stats --no-stream
docker system df

# Error logs
docker compose logs > docker-errors.log 2>&1
```

#### Support Channels
1. **Check GitHub Issues**: Search existing issues for similar problems
2. **Documentation**: Review the complete documentation in `Documentation/`
3. **Create Issue**: Include diagnostic information above

---

## ✅ Success Checklist

Your Docker deployment is working correctly when:

- ✅ `docker version` shows Docker is running
- ✅ `curl http://localhost:11434/api/tags` shows Ollama is accessible
- ✅ `./start-docker.sh status` shows all containers healthy
- ✅ All health check URLs return 200 OK
- ✅ You can access the frontend at http://localhost:3000
- ✅ You can create document indexes successfully
- ✅ You can chat with your documents
- ✅ No error messages in container logs

**If all boxes are checked, your Docker deployment is successful! 🎉**

---

**Still having issues?** Check the main `DOCKER_README.md` or create an issue with your diagnostic information. 
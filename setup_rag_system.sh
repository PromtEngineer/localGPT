#!/bin/bash
# setup_rag_system.sh - Complete RAG System Setup Script
# This script handles Docker installation, system setup, and initial configuration

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
    error "This script should not be run as root (except for package installation steps)"
    exit 1
fi

echo "================================================================"
echo "🚀 RAG System Complete Setup Script"
echo "================================================================"
echo ""

# Step 1: System Requirements Check
log "Step 1: Checking system requirements..."

# Check OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
    info "Detected macOS"
elif [[ -f /etc/os-release ]]; then
    . /etc/os-release
    OS=$ID
    info "Detected Linux: $OS"
else
    error "Unsupported operating system"
    exit 1
fi

# Check available memory
MEMORY_GB=$(free -g 2>/dev/null | grep '^Mem:' | awk '{print $2}' || sysctl -n hw.memsize 2>/dev/null | awk '{print int($1/1024/1024/1024)}' || echo "unknown")
if [[ "$MEMORY_GB" != "unknown" && "$MEMORY_GB" -lt 8 ]]; then
    warn "System has ${MEMORY_GB}GB RAM. Recommended: 16GB+ for optimal performance"
else
    info "Memory check passed: ${MEMORY_GB}GB RAM"
fi

# Check available disk space
DISK_GB=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//' || echo "unknown")
if [[ "$DISK_GB" != "unknown" && "$DISK_GB" -lt 50 ]]; then
    warn "Available disk space: ${DISK_GB}GB. Recommended: 50GB+ free space"
else
    info "Disk space check passed: ${DISK_GB}GB available"
fi

# Step 2: Install Dependencies
log "Step 2: Installing system dependencies..."

# Install Git if not present
if ! command -v git &> /dev/null; then
    info "Installing Git..."
    case $OS in
        "macos")
            if command -v brew &> /dev/null; then
                brew install git
            else
                error "Git not found. Please install Git first or install Homebrew"
                exit 1
            fi
            ;;
        "ubuntu"|"debian")
            sudo apt-get update
            sudo apt-get install -y git
            ;;
        "centos"|"rhel"|"fedora")
            if command -v dnf &> /dev/null; then
                sudo dnf install -y git
            else
                sudo yum install -y git
            fi
            ;;
    esac
else
    info "Git is already installed: $(git --version)"
fi

# Install curl if not present
if ! command -v curl &> /dev/null; then
    info "Installing curl..."
    case $OS in
        "macos")
            # curl is usually pre-installed on macOS
            ;;
        "ubuntu"|"debian")
            sudo apt-get install -y curl
            ;;
        "centos"|"rhel"|"fedora")
            if command -v dnf &> /dev/null; then
                sudo dnf install -y curl
            else
                sudo yum install -y curl
            fi
            ;;
    esac
else
    info "curl is already installed"
fi

# Step 3: Install Docker
log "Step 3: Installing Docker..."

if command -v docker &> /dev/null; then
    info "Docker is already installed: $(docker --version)"
else
    info "Docker not found. Installing Docker..."
    
    case $OS in
        "macos")
            # Check if Homebrew is installed
            if ! command -v brew &> /dev/null; then
                info "Installing Homebrew..."
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            fi
            
            # Install Docker Desktop
            info "Installing Docker Desktop..."
            brew install --cask docker
            
            warn "Docker Desktop installed. Please:"
            warn "1. Start Docker Desktop from Applications"
            warn "2. Wait for Docker to start completely"
            warn "3. Run this script again"
            exit 0
            ;;
            
        "ubuntu"|"debian")
            # Update package index
            sudo apt-get update
            
            # Install dependencies
            sudo apt-get install -y \
                ca-certificates \
                curl \
                gnupg \
                lsb-release
            
            # Add Docker's official GPG key
            sudo mkdir -p /etc/apt/keyrings
            curl -fsSL https://download.docker.com/linux/$OS/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
            
            # Set up repository
            echo \
              "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/$OS \
              $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
            
            # Install Docker Engine
            sudo apt-get update
            sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
            
            # Add user to docker group
            sudo usermod -aG docker $USER
            
            # Start Docker service
            sudo systemctl enable docker
            sudo systemctl start docker
            
            info "Docker installed successfully!"
            warn "Please log out and log back in for group changes to take effect, then run this script again"
            warn "Or run: newgrp docker && $0"
            exit 0
            ;;
            
        "centos"|"rhel"|"fedora")
            # Install required packages
            if command -v dnf &> /dev/null; then
                sudo dnf install -y yum-utils
                sudo dnf config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
                sudo dnf install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
            else
                sudo yum install -y yum-utils
                sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
                sudo yum install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
            fi
            
            # Add user to docker group
            sudo usermod -aG docker $USER
            
            # Start Docker service
            sudo systemctl enable docker
            sudo systemctl start docker
            
            info "Docker installed successfully!"
            warn "Please log out and log back in for group changes to take effect, then run this script again"
            exit 0
            ;;
    esac
fi

# Verify Docker is working
if ! docker --version &> /dev/null; then
    error "Docker is not working properly. Please check Docker installation"
    exit 1
fi

if ! docker compose version &> /dev/null; then
    error "Docker Compose is not working properly. Please check Docker Compose installation"
    exit 1
fi

info "Docker verification passed: $(docker --version)"
info "Docker Compose verification passed: $(docker compose version)"

# Test Docker daemon
if ! docker ps &> /dev/null; then
    error "Cannot connect to Docker daemon. Please ensure Docker is running"
    exit 1
fi

# Step 4: Setup RAG System
log "Step 4: Setting up RAG System..."

# Create project directory structure
info "Creating directory structure..."
mkdir -p {lancedb,shared_uploads,logs,ollama_data}
mkdir -p index_store/{overviews,bm25,graph}
mkdir -p backups

# Set proper permissions
chmod 755 {lancedb,shared_uploads,logs,ollama_data}
chmod 755 index_store/{overviews,bm25,graph}
chmod 755 backups

# Create environment file
if [[ ! -f ".env" ]]; then
    info "Creating environment configuration..."
    cat > .env << 'EOF'
# System Configuration
NODE_ENV=production
LOG_LEVEL=info
DEBUG=false

# Service URLs
FRONTEND_URL=http://localhost:3000
BACKEND_URL=http://localhost:8000
RAG_API_URL=http://localhost:8001
OLLAMA_URL=http://localhost:11434

# Database Configuration
DATABASE_PATH=./backend/chat_data.db
LANCEDB_PATH=./lancedb
UPLOADS_PATH=./shared_uploads
INDEX_STORE_PATH=./index_store

# Model Configuration
DEFAULT_EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
# Default model names - updated to current versions
DEFAULT_GENERATION_MODEL=qwen3:8b
DEFAULT_RERANKER_MODEL=answerdotai/answerai-colbert-small-v1
DEFAULT_ENRICHMENT_MODEL=qwen3:0.6b

# Performance Configuration
MAX_CONCURRENT_REQUESTS=5
REQUEST_TIMEOUT=300
EMBEDDING_BATCH_SIZE=32
MAX_CONTEXT_LENGTH=4096

# Security Configuration
CORS_ORIGINS=http://localhost:3000
API_KEY_REQUIRED=false
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60

# Storage Configuration
MAX_FILE_SIZE=50MB
MAX_UPLOAD_FILES=10
CLEANUP_INTERVAL=3600
BACKUP_RETENTION_DAYS=30
EOF
    info "Environment file created: .env"
else
    info "Environment file already exists: .env"
fi

# Step 5: Build and Start Services
log "Step 5: Building and starting services..."

info "Building Docker containers (this may take 10-15 minutes)..."
docker compose build --no-cache

info "Starting services..."
docker compose up -d

# Wait for services to start
info "Waiting for services to initialize..."
sleep 30

# Check service status
info "Checking service status..."
docker compose ps

# Step 6: Install AI Models
log "Step 6: Installing AI models..."

# Wait for Ollama to be ready
info "Waiting for Ollama to be ready..."
max_attempts=30
attempt=0
while ! docker compose exec ollama ollama list &> /dev/null; do
    if [ $attempt -ge $max_attempts ]; then
        error "Ollama failed to start after $max_attempts attempts"
        exit 1
    fi
    info "Waiting for Ollama... (attempt $((attempt+1))/$max_attempts)"
    sleep 10
    ((attempt++))
done

# Download Ollama models
info "Downloading required Ollama models..."
docker compose exec ollama ollama pull qwen3:8b
docker compose exec ollama ollama pull qwen3:0.6b

info "Verifying model installation..."
docker compose exec ollama ollama list

# Step 7: System Verification
log "Step 7: Verifying system installation..."

# Check service health
info "Checking service health..."
services=("frontend:3000" "backend:8000" "rag-api:8001" "ollama:11434")
for service in "${services[@]}"; do
    name="${service%:*}"
    port="${service#*:}"
    
    if curl -s -f "http://localhost:$port" &> /dev/null || curl -s -f "http://localhost:$port/health" &> /dev/null || curl -s -f "http://localhost:$port/api/tags" &> /dev/null || curl -s -f "http://localhost:$port/models" &> /dev/null; then
        info "✅ $name service is healthy"
    else
        warn "⚠️ $name service may not be ready yet"
    fi
done

# Step 8: Create Helper Scripts
log "Step 8: Creating helper scripts..."

# Create start script
cat > start_rag_system.sh << 'EOF'
#!/bin/bash
# Start RAG System
echo "Starting RAG System..."
docker compose up -d
echo "RAG System started. Access at: http://localhost:3000"
EOF
chmod +x start_rag_system.sh

# Create stop script
cat > stop_rag_system.sh << 'EOF'
#!/bin/bash
# Stop RAG System
echo "Stopping RAG System..."
docker compose down
echo "RAG System stopped."
EOF
chmod +x stop_rag_system.sh

# Create status script
cat > status_rag_system.sh << 'EOF'
#!/bin/bash
# Check RAG System Status
echo "=== RAG System Status ==="
docker compose ps
echo ""
echo "=== Service Health ==="
curl -s -f http://localhost:3000 && echo "✅ Frontend: OK" || echo "❌ Frontend: FAIL"
curl -s -f http://localhost:8000/health && echo "✅ Backend: OK" || echo "❌ Backend: FAIL"
curl -s -f http://localhost:8001/models && echo "✅ RAG API: OK" || echo "❌ RAG API: FAIL"
curl -s -f http://localhost:11434/api/tags && echo "✅ Ollama: OK" || echo "❌ Ollama: FAIL"
EOF
chmod +x status_rag_system.sh

# Create backup script
cat > backup_rag_system.sh << 'EOF'
#!/bin/bash
# Backup RAG System Data
BACKUP_DIR="./backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "Creating backup in $BACKUP_DIR..."

# Stop services
docker compose down

# Backup data
cp -r ./backend/chat_data.db "$BACKUP_DIR/" 2>/dev/null || true
cp -r ./lancedb "$BACKUP_DIR/" 2>/dev/null || true
cp -r ./shared_uploads "$BACKUP_DIR/" 2>/dev/null || true
cp -r ./index_store "$BACKUP_DIR/" 2>/dev/null || true

# Backup configuration
cp .env "$BACKUP_DIR/"
cp docker-compose.yml "$BACKUP_DIR/"

# Restart services
docker compose up -d

echo "Backup completed: $BACKUP_DIR"
EOF
chmod +x backup_rag_system.sh

# Create update script
cat > update_rag_system.sh << 'EOF'
#!/bin/bash
# Update RAG System
echo "Updating RAG System..."

# Backup first
./backup_rag_system.sh

# Pull latest changes
git pull origin main

# Rebuild containers
docker compose build --no-cache

# Restart services
docker compose up -d

echo "Update completed!"
EOF
chmod +x update_rag_system.sh

info "Helper scripts created:"
info "  - start_rag_system.sh: Start the system"
info "  - stop_rag_system.sh: Stop the system"
info "  - status_rag_system.sh: Check system status"
info "  - backup_rag_system.sh: Backup system data"
info "  - update_rag_system.sh: Update the system"

# Step 9: Final Setup
log "Step 9: Final setup and verification..."

# Create initial database if it doesn't exist
if [[ ! -f "./backend/chat_data.db" ]]; then
    info "Creating initial database..."
    docker compose exec backend python -c "
import sqlite3
conn = sqlite3.connect('/app/backend/chat_data.db')
conn.execute('CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, title TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)')
conn.execute('CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY, session_id TEXT, content TEXT, role TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)')
conn.execute('CREATE TABLE IF NOT EXISTS indexes (id TEXT PRIMARY KEY, name TEXT, metadata TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)')
conn.execute('CREATE TABLE IF NOT EXISTS session_indexes (session_id TEXT, index_id TEXT, PRIMARY KEY (session_id, index_id))')
conn.commit()
conn.close()
print('Database initialized')
" 2>/dev/null || warn "Database initialization may have failed"
fi

# Final health check
info "Performing final health check..."
sleep 10
./status_rag_system.sh

echo ""
echo "================================================================"
echo "🎉 RAG System Setup Complete!"
echo "================================================================"
echo ""
echo "✅ System Status:"
echo "   - Frontend: http://localhost:3000"
echo "   - Backend API: http://localhost:8000"
echo "   - RAG API: http://localhost:8001"
echo "   - Ollama: http://localhost:11434"
echo ""
echo "📚 Documentation:"
echo "   - System Overview: Documentation/system_overview.md"
echo "   - Deployment Guide: Documentation/deployment_guide.md"
echo "   - Docker Usage: Documentation/docker_usage.md"
echo "   - Installation Guide: Documentation/installation_guide.md"
echo ""
echo "🔧 Helper Scripts:"
echo "   - Start system: ./start_rag_system.sh"
echo "   - Stop system: ./stop_rag_system.sh"
echo "   - Check status: ./status_rag_system.sh"
echo "   - Backup data: ./backup_rag_system.sh"
echo "   - Update system: ./update_rag_system.sh"
echo ""
echo "🚀 Next Steps:"
echo "   1. Open http://localhost:3000 in your browser"
echo "   2. Create a new chat session"
echo "   3. Upload some PDF documents"
echo "   4. Start asking questions about your documents!"
echo ""
echo "📋 System Information:"
echo "   - OS: $OS"
echo "   - Memory: ${MEMORY_GB}GB"
echo "   - Disk Space: ${DISK_GB}GB available"
echo "   - Docker: $(docker --version)"
echo "   - Docker Compose: $(docker compose version)"
echo ""
echo "For support and troubleshooting, check the documentation in the"
echo "Documentation/ folder or run ./status_rag_system.sh to check system health."
echo "" 
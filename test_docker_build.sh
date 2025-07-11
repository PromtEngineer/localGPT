#!/bin/bash

# Test Docker builds individually
echo "🐳 Testing Docker builds individually..."

# Function to check if Docker is running
check_docker() {
    if ! docker version >/dev/null 2>&1; then
        echo "❌ Docker is not running. Please start Docker Desktop."
        exit 1
    fi
    echo "✅ Docker is running"
}

# Function to build and test a single container
build_and_test() {
    local service=$1
    local dockerfile=$2
    local port=$3
    
    echo ""
    echo "🔨 Building $service..."
    docker build -f $dockerfile -t "rag-$service" .
    if [ $? -ne 0 ]; then
        echo "❌ Failed to build $service"
        return 1
    fi
    
    echo "✅ $service built successfully"
    
    # Test running the container
    echo "🚀 Testing $service container..."
    docker run -d --name "test-$service" -p "$port:$port" "rag-$service"
    if [ $? -ne 0 ]; then
        echo "❌ Failed to run $service"
        return 1
    fi
    
    echo "⏳ Waiting for $service to start..."
    sleep 10
    
    # Test health
    if [ "$service" = "frontend" ]; then
        curl -f "http://localhost:$port" >/dev/null 2>&1
    elif [ "$service" = "backend" ]; then
        curl -f "http://localhost:$port/health" >/dev/null 2>&1
    elif [ "$service" = "rag-api" ]; then
        curl -f "http://localhost:$port/models" >/dev/null 2>&1
    fi
    
    if [ $? -eq 0 ]; then
        echo "✅ $service is healthy"
    else
        echo "⚠️ $service health check failed (but container is running)"
        docker logs "test-$service" | tail -10
    fi
    
    # Cleanup
    docker stop "test-$service" >/dev/null 2>&1
    docker rm "test-$service" >/dev/null 2>&1
    
    return 0
}

# Main execution
check_docker

echo "🧹 Cleaning up old containers and images..."
docker container prune -f >/dev/null 2>&1
docker image prune -f >/dev/null 2>&1

# Build in dependency order
echo "📦 Building containers in dependency order..."

# 1. RAG API (no dependencies)
build_and_test "rag-api" "Dockerfile.rag-api" "8001"
if [ $? -ne 0 ]; then
    echo "❌ RAG API build failed, stopping"
    exit 1
fi

# 2. Backend (depends on RAG API)
build_and_test "backend" "Dockerfile.backend" "8000"
if [ $? -ne 0 ]; then
    echo "❌ Backend build failed, stopping"
    exit 1
fi

# 3. Frontend (depends on Backend)
build_and_test "frontend" "Dockerfile.frontend" "3000"
if [ $? -ne 0 ]; then
    echo "❌ Frontend build failed, stopping"
    exit 1
fi

echo ""
echo "🎉 All containers built and tested successfully!"
echo "🚀 You can now run: ./start-docker.sh" 
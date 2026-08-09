#!/bin/bash

# LocalGPT Docker Startup Script
# This script provides easy options for running LocalGPT in Docker

set -e

echo "🐳 LocalGPT Docker Deployment"
echo "============================"

# Non-interactive mode: never prompt. Enable with -y/--yes or NONINTERACTIVE=1.
ASSUME_YES="${NONINTERACTIVE:-}"
ARGS=()
for arg in "$@"; do
    case "$arg" in
        -y|--yes|--assume-yes)
            ASSUME_YES=1
            ;;
        *)
            ARGS+=("$arg")
            ;;
    esac
done
set -- "${ARGS[@]}"

# Function to check if local Ollama is running
check_local_ollama() {
    if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo "✅ Local Ollama detected on port 11434"
        return 0
    else
        echo "❌ No local Ollama detected on port 11434"
        return 1
    fi
}

# Function to start with local Ollama
start_with_local_ollama() {
    echo "🚀 Starting LocalGPT containers (using local Ollama)..."
    echo "📝 Note: Make sure your local Ollama is running on port 11434"
    
    # Use the docker.env file for configuration
    docker compose --env-file docker.env up --build -d
    
    echo ""
    echo "🎉 LocalGPT is starting up!"
    echo "📱 Frontend: http://localhost:3000"
    echo "🔧 Backend API: http://localhost:8000"
    echo "🧠 RAG API: http://localhost:8001"
    echo "🤖 Ollama: http://localhost:11434 (local)"
    echo ""
    echo "📊 Check container status: docker compose ps"
    echo "📝 View logs: docker compose logs -f"
    echo "🛑 Stop services: docker compose down"
}

# Function to start with containerized Ollama
start_with_container_ollama() {
    echo "🚀 Starting LocalGPT containers (including Ollama container)..."
    
    # Set environment variable for containerized Ollama.
    # The shell environment takes precedence over --env-file, so this wins over docker.env.
    export OLLAMA_HOST=http://ollama:11434

    # Start all services including Ollama
    docker compose --env-file docker.env --profile with-ollama up --build -d
    
    echo ""
    echo "🎉 LocalGPT is starting up!"
    echo "📱 Frontend: http://localhost:3000"
    echo "🔧 Backend API: http://localhost:8000"
    echo "🧠 RAG API: http://localhost:8001"
    echo "🤖 Ollama: http://localhost:11434 (containerized)"
    echo ""
    echo "⏳ Note: First startup may take longer as Ollama container initializes"
    echo "📊 Check container status: docker compose --profile with-ollama ps"
    echo "📝 View logs: docker compose --profile with-ollama logs -f"
    echo "🛑 Stop services: docker compose --profile with-ollama down"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [option] [-y|--yes]"
    echo ""
    echo "Options:"
    echo "  local     - Use local Ollama instance (default)"
    echo "  container - Use containerized Ollama"
    echo "  stop      - Stop all containers"
    echo "  logs      - Show container logs"
    echo "  status    - Show container status"
    echo "  help      - Show this help message"
    echo ""
    echo "Flags:"
    echo "  -y, --yes - Never prompt; fall back to containerized Ollama when no local"
    echo "              Ollama is detected. Also enabled with NONINTERACTIVE=1."
    echo ""
    echo "Examples:"
    echo "  $0 local      # Use local Ollama (recommended)"
    echo "  $0 container  # Use containerized Ollama"
    echo "  $0 local -y   # Scripted/CI use - no prompts"
    echo "  $0 stop       # Stop all services"
}

# Function to stop containers
stop_containers() {
    echo "🛑 Stopping LocalGPT containers..."
    docker compose down
    docker compose --profile with-ollama down 2>/dev/null || true
    echo "✅ All containers stopped"
}

# Function to show logs
show_logs() {
    echo "📝 Showing container logs (Ctrl+C to exit)..."
    if docker compose ps | grep -q "rag-ollama"; then
        docker compose --profile with-ollama logs -f
    else
        docker compose logs -f
    fi
}

# Function to show status
show_status() {
    echo "📊 Container Status:"
    docker compose ps
    echo ""
    echo "🐳 All Docker containers:"
    docker ps | grep -E "(rag-|CONTAINER)" || echo "No LocalGPT containers running"
}

# Main script logic
case "${1:-local}" in
    "local")
        if check_local_ollama; then
            start_with_local_ollama
        else
            echo ""
            echo "⚠️  No local Ollama detected. Options:"
            echo "1. Start local Ollama: 'ollama serve'"
            echo "2. Use containerized Ollama: '$0 container'"
            echo ""
            if [ -n "$ASSUME_YES" ]; then
                echo "▶️  Non-interactive mode: starting containerized Ollama"
                start_with_container_ollama
            elif [ -t 0 ]; then
                read -p "Start with containerized Ollama instead? (y/N): " -n 1 -r
                echo
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    start_with_container_ollama
                else
                    echo "❌ Cancelled. Please start local Ollama or use '$0 container'"
                    exit 1
                fi
            else
                echo "❌ No TTY for the confirmation prompt."
                echo "   Re-run with '$0 local --yes' (or NONINTERACTIVE=1 $0) to use containerized Ollama,"
                echo "   or '$0 container' to select it explicitly."
                exit 1
            fi
        fi
        ;;
    "container")
        start_with_container_ollama
        ;;
    "stop")
        stop_containers
        ;;
    "logs")
        show_logs
        ;;
    "status")
        show_status
        ;;
    "help"|"-h"|"--help")
        show_usage
        ;;
    *)
        echo "❌ Unknown option: $1"
        echo ""
        show_usage
        exit 1
        ;;
esac 
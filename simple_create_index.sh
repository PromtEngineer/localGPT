#!/bin/bash

# Simple Index Creation Script for LocalGPT RAG System
# Usage: ./simple_create_index.sh "Index Name" "path/to/document.pdf" [additional_files...]

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Status messages go to stderr so stdout can carry data (e.g. validated paths)
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1" >&2
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" >&2
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" >&2
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."

    # Check Python
    if ! command_exists python3; then
        print_error "Python 3 is required but not installed."
        exit 1
    fi

    # Check if we're in the right directory
    if [ ! -f "run_system.py" ] || [ ! -d "rag_system" ]; then
        print_error "This script must be run from the LocalGPT project root directory."
        exit 1
    fi

    # Check if Ollama is running
    if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        print_error "Ollama is not running. Please start Ollama first:"
        echo "  ollama serve" >&2
        exit 1
    fi

    print_success "Prerequisites check passed"
}

# Function to validate documents.
# Prints one valid path per line on stdout; logs go to stderr.
validate_documents() {
    local documents=("$@")
    local valid_count=0

    print_status "Validating documents..."

    for doc in "${documents[@]}"; do
        if [ -f "$doc" ]; then
            # Check file extension
            case "${doc##*.}" in
                pdf|txt|docx|md|html|htm)
                    printf '%s\n' "$doc"
                    valid_count=$((valid_count + 1))
                    print_status "✓ Valid document: $doc"
                    ;;
                *)
                    print_warning "Unsupported file type: $doc (skipping)"
                    ;;
            esac
        else
            print_warning "File not found: $doc (skipping)"
        fi
    done

    if [ "$valid_count" -eq 0 ]; then
        print_error "No valid documents found."
        exit 1
    fi
}

# Function to create the index.
# The index name and document paths are passed to Python as argv — never
# interpolated into the source — so names with quotes or code are safe.
create_index() {
    python3 - "$@" << 'EOF'
import os
import sys

sys.path.insert(0, os.getcwd())

from rag_system.main import PIPELINE_CONFIGS
from rag_system.pipelines.indexing_pipeline import IndexingPipeline
from rag_system.utils.ollama_client import OllamaClient
from backend.database import ChatDatabase


def main():
    index_name = sys.argv[1]
    documents = [os.path.abspath(d) for d in sys.argv[2:] if os.path.isfile(d)]
    if not documents:
        print("❌ No valid documents to process")
        sys.exit(1)

    db = ChatDatabase()
    index_id = db.create_index(
        name=index_name,
        description="Created with simple_create_index.sh",
        metadata={
            "chunk_size": 512,
            "chunk_overlap": 64,
            "enable_enrich": True,
            "enable_latechunk": True,
            "retrieval_mode": "hybrid",
            "created_by": "simple_create_index.sh",
        },
    )

    for doc_path in documents:
        db.add_document_to_index(index_id, os.path.basename(doc_path), doc_path)

    config = PIPELINE_CONFIGS.get("default", {})
    ollama_config = {
        "generation_model": "qwen3:8b",
        "enrichment_model": "qwen3:8b",
    }
    pipeline = IndexingPipeline(config, OllamaClient(), ollama_config)
    pipeline.process_documents(documents)

    print(f"✅ Index '{index_name}' created successfully!")
    print(f"Index ID: {index_id}")
    print(f"Processed {len(documents)} documents")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error creating index: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
EOF
}

# Function to show usage
show_usage() {
    echo "Usage: $0 \"Index Name\" \"path/to/document.pdf\" [additional_files...]"
    echo ""
    echo "Examples:"
    echo "  $0 \"My Documents\" \"document.pdf\""
    echo "  $0 \"Research Papers\" \"paper1.pdf\" \"paper2.pdf\" \"notes.txt\""
    echo "  $0 \"Invoice Collection\" ./invoices/*.pdf"
    echo ""
    echo "Supported file types: PDF, TXT, DOCX, MD, HTML"
}

# Main script
main() {
    # Check arguments
    if [ $# -lt 2 ]; then
        print_error "Insufficient arguments provided."
        show_usage
        exit 1
    fi

    local index_name="$1"
    shift
    local documents=("$@")

    # Check prerequisites
    check_prerequisites

    # Validate documents (newline-separated so paths with spaces survive)
    local valid_documents=()
    while IFS= read -r line; do
        [ -n "$line" ] && valid_documents+=("$line")
    done < <(validate_documents "${documents[@]}")

    if [ ${#valid_documents[@]} -eq 0 ]; then
        print_error "No valid documents to process."
        exit 1
    fi

    # Create the index
    print_status "Starting index creation process..."
    create_index "$index_name" "${valid_documents[@]}"

    print_success "Index creation completed!"
    print_status "You can now use the index in the LocalGPT interface."
}

# Run main function with all arguments
main "$@"

#!/usr/bin/env python3
"""
Interactive Index Creation Script for LocalGPT RAG System

This script provides a user-friendly interface for creating document indexes
using the LocalGPT RAG system. It supports both single documents and batch
processing of multiple documents.

Usage:
    python create_index_script.py
    python create_index_script.py --batch index_config.json
    python create_index_script.py --config custom_pipeline_config.json
    python create_index_script.py --create-sample
"""

import copy
import os
import sys
import json
import argparse
from typing import List, Optional
from pathlib import Path

# Add the project root to the path so we can import rag_system modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from rag_system.main import PIPELINE_CONFIGS, OLLAMA_CONFIG
    from rag_system.factory import get_agent
    from rag_system.pipelines.indexing_pipeline import IndexingPipeline
    from rag_system.utils.ollama_client import OllamaClient
    from backend.database import ChatDatabase
except ImportError as e:
    print(f"❌ Error importing required modules: {e}")
    print("Please ensure you're running this script from the project root directory.")
    sys.exit(1)


class IndexCreator:
    """Interactive index creation utility."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the index creator with optional custom configuration."""
        self.db = ChatDatabase()
        self.base_config = self._load_config(config_path)
        self.ollama_config = OLLAMA_CONFIG
        self.ollama_client = OllamaClient(host=OLLAMA_CONFIG["host"])

    def _load_config(self, config_path: Optional[str] = None) -> dict:
        """Load configuration from file or use default."""
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️  Error loading config from {config_path}: {e}")
                print("Using default configuration...")

        return PIPELINE_CONFIGS.get("default", {})

    def _build_pipeline(self, index_id: str, processing: dict) -> IndexingPipeline:
        """Build a pipeline that writes into this index's own tables."""
        config = copy.deepcopy(self.base_config)
        table_name = f"text_pages_{index_id}"

        config.setdefault("storage", {})["text_table_name"] = table_name
        # The pipeline reads whichever of these two keys is present.
        retrievers = config.get("retrievers")
        if retrievers is None:
            retrievers = config.setdefault("retrieval", {})
        retrievers.setdefault("dense", {})["lancedb_table_name"] = table_name
        retrievers.setdefault("latechunk", {})["enabled"] = bool(processing.get("enable_latechunk", False))

        config["chunker_mode"] = "docling" if processing.get("enable_docling", True) else "legacy"
        config.setdefault("chunking", {})["chunk_size"] = int(processing.get("chunk_size", 512))
        config.setdefault("contextual_enricher", {}).update({
            "enabled": bool(processing.get("enable_enrich", True)),
            "window_size": int(processing.get("window_size", 2)),
        })
        config["overview_path"] = f"index_store/overviews/{index_id}.jsonl"

        if processing.get("embedding_model"):
            config["embedding_model_name"] = processing["embedding_model"]
        if processing.get("enrich_model"):
            config["enrich_model"] = processing["enrich_model"]

        return IndexingPipeline(config, self.ollama_client, self.ollama_config)

    def get_user_input(self, prompt: str, default: str = "") -> str:
        """Get user input with optional default value."""
        if default:
            user_input = input(f"{prompt} [{default}]: ").strip()
            return user_input if user_input else default
        return input(f"{prompt}: ").strip()
    
    def select_documents(self) -> List[str]:
        """Interactive document selection."""
        print("\n📁 Document Selection")
        print("=" * 50)
        
        documents = []
        
        while True:
            print("\nOptions:")
            print("1. Add a single document")
            print("2. Add all documents from a directory")
            print("3. Finish and proceed with selected documents")
            print("4. Show selected documents")
            
            choice = self.get_user_input("Select an option (1-4)", "1")
            
            if choice == "1":
                doc_path = self.get_user_input("Enter document path")
                if os.path.exists(doc_path):
                    documents.append(os.path.abspath(doc_path))
                    print(f"✅ Added: {doc_path}")
                else:
                    print(f"❌ File not found: {doc_path}")
            
            elif choice == "2":
                dir_path = self.get_user_input("Enter directory path")
                if os.path.isdir(dir_path):
                    supported_extensions = ['.pdf', '.txt', '.docx', '.md', '.html', '.htm']
                    found_docs = []
                    
                    for ext in supported_extensions:
                        found_docs.extend(Path(dir_path).glob(f"*{ext}"))
                        found_docs.extend(Path(dir_path).glob(f"**/*{ext}"))
                    
                    if found_docs:
                        print(f"Found {len(found_docs)} documents:")
                        for doc in found_docs:
                            print(f"  - {doc}")
                        
                        if self.get_user_input("Add all these documents? (y/n)", "y").lower() == 'y':
                            documents.extend([str(doc.absolute()) for doc in found_docs])
                            print(f"✅ Added {len(found_docs)} documents")
                    else:
                        print("❌ No supported documents found in directory")
                else:
                    print(f"❌ Directory not found: {dir_path}")
            
            elif choice == "3":
                if documents:
                    break
                else:
                    print("❌ No documents selected. Please add at least one document.")
            
            elif choice == "4":
                if documents:
                    print(f"\n📄 Selected documents ({len(documents)}):")
                    for i, doc in enumerate(documents, 1):
                        print(f"  {i}. {doc}")
                else:
                    print("No documents selected yet.")
            
            else:
                print("Invalid choice. Please select 1-4.")
        
        return documents
    
    def configure_processing(self) -> dict:
        """Interactive processing configuration."""
        print("\n⚙️  Processing Configuration")
        print("=" * 50)
        
        print("Configure how documents will be processed:")
        
        # Basic settings
        chunk_size = int(self.get_user_input("Chunk size (tokens)", "512"))

        # Advanced settings
        print("\nAdvanced options:")
        enable_enrich = self.get_user_input("Enable contextual enrichment? (y/n)", "y").lower() == 'y'
        enable_latechunk = self.get_user_input("Enable late chunking? (y/n)", "y").lower() == 'y'
        enable_docling = self.get_user_input("Enable Docling chunking? (y/n)", "y").lower() == 'y'

        # Model selection
        print("\nModel Configuration:")
        default_embedding = self.base_config.get("embedding_model_name", "")
        embedding_model = self.get_user_input("Embedding model", default_embedding)
        enrich_model = self.get_user_input(
            "Enrichment model", self.ollama_config.get("enrichment_model", "")
        )

        return {
            "chunk_size": chunk_size,
            "enable_enrich": enable_enrich,
            "enable_latechunk": enable_latechunk,
            "enable_docling": enable_docling,
            "embedding_model": embedding_model,
            "enrich_model": enrich_model,
            "retrieval_mode": "hybrid",
            "window_size": 2
        }

    def create_index_interactive(self) -> bool:
        """Run the interactive index creation process."""
        print("🚀 LocalGPT Index Creation Tool")
        print("=" * 50)
        
        # Get index details
        index_name = self.get_user_input("Enter index name")
        index_description = self.get_user_input("Enter index description (optional)")
        
        # Select documents
        documents = self.select_documents()
        
        # Configure processing
        processing_config = self.configure_processing()
        
        # Confirm creation
        print("\n📋 Index Summary")
        print("=" * 50)
        print(f"Name: {index_name}")
        print(f"Description: {index_description or 'None'}")
        print(f"Documents: {len(documents)}")
        print(f"Chunk size: {processing_config['chunk_size']}")
        print(f"Enrichment: {'Enabled' if processing_config['enable_enrich'] else 'Disabled'}")
        print(f"Embedding model: {processing_config['embedding_model']}")
        
        if self.get_user_input("\nProceed with index creation? (y/n)", "y").lower() != 'y':
            print("❌ Index creation cancelled.")
            return False

        # Create the index
        index_id = None
        try:
            print("\n🔥 Creating index...")

            # Create index record in database
            index_id = self.db.create_index(
                name=index_name,
                description=index_description,
                metadata=processing_config
            )

            # Add documents to index
            for doc_path in documents:
                filename = os.path.basename(doc_path)
                self.db.add_document_to_index(index_id, filename, doc_path)

            # Process documents through pipeline
            print("📚 Processing documents...")
            self._build_pipeline(index_id, processing_config).run(documents)

            print(f"\n✅ Index '{index_name}' created successfully!")
            print(f"Index ID: {index_id}")
            print(f"Processed {len(documents)} documents")

            # Test the index
            if self.get_user_input("\nTest the index with a sample query? (y/n)", "y").lower() == 'y':
                self.test_index(index_id)

            return True

        except Exception as e:
            print(f"❌ Error creating index: {e}")
            import traceback
            traceback.print_exc()
            if index_id:
                print(f"🧹 Removing incomplete index record {index_id}")
                self.db.delete_index(index_id)
            return False

    def test_index(self, index_id: str) -> None:
        """Test the created index with a sample query."""
        try:
            print("\n🧪 Testing Index")
            print("=" * 50)
            
            # Get agent for testing
            agent = get_agent("default")
            
            # Test query
            test_query = self.get_user_input("Enter a test query", "What is this document about?")
            
            print(f"\nProcessing query: {test_query}")
            response = agent.run(test_query, table_name=f"text_pages_{index_id}")
            
            print(f"\n🤖 Response:")
            print(response)
            
        except Exception as e:
            print(f"❌ Error testing index: {e}")
    
    def batch_create_from_config(self, config_file: str) -> bool:
        """Create index from batch configuration file."""
        index_id = None
        try:
            with open(config_file, 'r') as f:
                batch_config = json.load(f)

            index_name = batch_config.get("index_name", "Batch Index")
            index_description = batch_config.get("index_description", "")
            documents = batch_config.get("documents", [])
            processing_config = batch_config.get("processing", {})

            if not documents:
                print("❌ No documents specified in batch configuration")
                return False

            # Validate documents exist
            valid_documents = []
            for doc_path in documents:
                if os.path.exists(doc_path):
                    valid_documents.append(os.path.abspath(doc_path))
                else:
                    print(f"⚠️  Document not found: {doc_path}")

            if not valid_documents:
                print("❌ No valid documents found")
                return False

            print(f"🚀 Creating batch index: {index_name}")
            print(f"📄 Processing {len(valid_documents)} documents...")

            # Create index
            index_id = self.db.create_index(
                name=index_name,
                description=index_description,
                metadata=processing_config
            )

            # Add documents
            for doc_path in valid_documents:
                filename = os.path.basename(doc_path)
                self.db.add_document_to_index(index_id, filename, doc_path)

            # Process documents
            self._build_pipeline(index_id, processing_config).run(valid_documents)

            print(f"✅ Batch index '{index_name}' created successfully!")
            print(f"Index ID: {index_id}")
            return True

        except Exception as e:
            print(f"❌ Error creating batch index: {e}")
            import traceback
            traceback.print_exc()
            if index_id:
                print(f"🧹 Removing incomplete index record {index_id}")
                self.db.delete_index(index_id)
            return False


SAMPLE_CONFIG_FILENAME = "index_config.sample.json"


def create_sample_batch_config():
    """Create a sample batch configuration file."""
    sample_config = {
        "index_name": "Sample Batch Index",
        "index_description": "Example batch index configuration",
        "documents": [
            "/absolute/path/to/first.pdf",
            "/absolute/path/to/second.pdf"
        ],
        "processing": {
            "chunk_size": 512,
            "enable_enrich": True,
            "enable_latechunk": True,
            "enable_docling": True,
            "embedding_model": PIPELINE_CONFIGS["default"]["embedding_model_name"],
            "enrich_model": OLLAMA_CONFIG["enrichment_model"],
            "retrieval_mode": "hybrid",
            "window_size": 2
        }
    }

    with open(SAMPLE_CONFIG_FILENAME, "w") as f:
        json.dump(sample_config, f, indent=2)

    print(f"📄 Sample batch configuration created: {SAMPLE_CONFIG_FILENAME}")


def main() -> int:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="LocalGPT Index Creation Tool")
    parser.add_argument("--batch", help="Batch configuration file", type=str)
    parser.add_argument("--config", help="Custom pipeline configuration file", type=str)
    parser.add_argument("--create-sample", action="store_true", help="Create sample batch config")

    args = parser.parse_args()

    if args.create_sample:
        create_sample_batch_config()
        return 0

    try:
        creator = IndexCreator(config_path=args.config)

        if args.batch:
            ok = creator.batch_create_from_config(args.batch)
        else:
            ok = creator.create_index_interactive()

        return 0 if ok else 1

    except KeyboardInterrupt:
        print("\n\n❌ Operation cancelled by user.")
        return 130
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

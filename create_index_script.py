#!/usr/bin/env python3
"""
Interactive Index Creation Script for LocalGPT RAG System

This script provides a user-friendly interface for creating document indexes
using the LocalGPT RAG system. It supports both single documents and batch
processing of multiple documents.

Usage:
    python create_index_script.py
    python create_index_script.py --batch
    python create_index_script.py --config custom_config.json
"""

import os
import sys
import json
import argparse
from typing import List, Optional
from pathlib import Path

# Add the project root to the path so we can import rag_system modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from rag_system.main import PIPELINE_CONFIGS, get_agent
    from rag_system.pipelines.indexing_pipeline import IndexingPipeline
    from rag_system.utils.ollama_client import OllamaClient
    from backend.database import ChatDatabase
except ImportError as e:
    print(f"❌ Error importing required modules: {e}")
    print("Please ensure you're running this script from the project root directory.")
    sys.exit(1)


class IndexCreator:
    """Interactive index creation utility for LocalGPT RAG system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the index creator with optional custom configuration.
        
        Args:
            config_path: Optional path to custom configuration file. If not provided,
                        uses default configuration.
        """
        self.db = ChatDatabase()
        self.config = self._load_config(config_path)
        
        # Initialize Ollama client
        self.ollama_client = OllamaClient()
        self.ollama_config = {
            "generation_model": "qwen3:0.6b",
            "embedding_model": "qwen3:0.6b"
        }
        
        # Initialize indexing pipeline
        self.pipeline = IndexingPipeline(
            self.config, 
            self.ollama_client, 
            self.ollama_config
        )
    
    def _load_config(self, config_path: Optional[str] = None) -> dict:
        """
        Load configuration from file or use default.
        
        Args:
            config_path: Optional path to configuration file.
            
        Returns:
            Dictionary containing configuration settings.
        """
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️  Error loading config from {config_path}: {e}")
                print("Using default configuration...")
        
        return PIPELINE_CONFIGS.get("default", {})
    
    def get_user_input(self, prompt: str, default: str = "") -> str:
        """
        Get user input with optional default value.
        
        Args:
            prompt: The prompt message to display to the user.
            default: Default value to use if user provides no input.
            
        Returns:
            User input string or default value if no input provided.
        """
        if default:
            user_input = input(f"{prompt} [{default}]: ").strip()
            return user_input if user_input else default
        return input(f"{prompt}: ").strip()
    
    def select_documents(self) -> List[str]:
        """
        Interactive document selection interface.
        
        Provides options to add single documents, entire directories,
        and manage the selected document list.
        
        Returns:
            List of absolute paths to selected documents.
        """
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
        """
        Interactive processing configuration interface.
        
        Allows users to configure document processing parameters including
        chunk size, overlap, enrichment options, and model selection.
        
        Returns:
            Dictionary containing processing configuration settings.
        """
        print("\n⚙️  Processing Configuration")
        print("=" * 50)
        
        print("Configure how documents will be processed:")
        
        # Basic settings
        chunk_size = int(self.get_user_input("Chunk size", "512"))
        chunk_overlap = int(self.get_user_input("Chunk overlap", "64"))
        
        # Advanced settings
        print("\nAdvanced options:")
        enable_enrich = self.get_user_input("Enable contextual enrichment? (y/n)", "y").lower() == 'y'
        enable_latechunk = self.get_user_input("Enable late chunking? (y/n)", "y").lower() == 'y'
        enable_docling = self.get_user_input("Enable Docling chunking? (y/n)", "y").lower() == 'y'
        
        # Model selection
        print("\nModel Configuration:")
        embedding_model = self.get_user_input("Embedding model", "Qwen/Qwen3-Embedding-0.6B")
        generation_model = self.get_user_input("Generation model", "qwen3:0.6b")
        
        return {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "enable_enrich": enable_enrich,
            "enable_latechunk": enable_latechunk,
            "enable_docling": enable_docling,
            "embedding_model": embedding_model,
            "generation_model": generation_model,
            "retrieval_mode": "hybrid",
            "window_size": 2
        }
    
    def create_index_interactive(self) -> None:
        """
        Run the interactive index creation process.
        
        Guides the user through the complete index creation workflow including
        naming, document selection, configuration, and processing.
        """
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
            return
        
        # Create the index
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
            self.pipeline.process_documents(documents)
            
            print(f"\n✅ Index '{index_name}' created successfully!")
            print(f"Index ID: {index_id}")
            print(f"Processed {len(documents)} documents")
            
            # Test the index
            if self.get_user_input("\nTest the index with a sample query? (y/n)", "y").lower() == 'y':
                self.test_index(index_id)
                
        except Exception as e:
            print(f"❌ Error creating index: {e}")
            import traceback
            traceback.print_exc()
    
    def test_index(self, index_id: str) -> None:
        """
        Test the created index with a sample query.
        
        Args:
            index_id: The ID of the index to test.
        """
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
    
    def batch_create_from_config(self, config_file: str) -> None:
        """
        Create index from batch configuration file.
        
        Processes a JSON configuration file containing index settings and
        document paths for automated index creation.
        
        Args:
            config_file: Path to the batch configuration JSON file.
        """
        try:
            with open(config_file, 'r') as f:
                batch_config = json.load(f)
            
            index_name = batch_config.get("index_name", "Batch Index")
            index_description = batch_config.get("index_description", "")
            documents = batch_config.get("documents", [])
            processing_config = batch_config.get("processing", {})
            
            if not documents:
                print("❌ No documents specified in batch configuration")
                return
            
            # Validate documents exist
            valid_documents = []
            for doc_path in documents:
                if os.path.exists(doc_path):
                    valid_documents.append(doc_path)
                else:
                    print(f"⚠️  Document not found: {doc_path}")
            
            if not valid_documents:
                print("❌ No valid documents found")
                return
            
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
            self.pipeline.process_documents(valid_documents)
            
            print(f"✅ Batch index '{index_name}' created successfully!")
            print(f"Index ID: {index_id}")
            
        except Exception as e:
            print(f"❌ Error creating batch index: {e}")
            import traceback
            traceback.print_exc()


def create_sample_batch_config():
    """
    Create a sample batch configuration file.
    
    Generates a JSON configuration file with example settings that can be
    used as a template for batch index creation.
    """
    sample_config = {
        "index_name": "Sample Batch Index",
        "index_description": "Example batch index configuration",
        "documents": [
            "./rag_system/documents/invoice_1039.pdf",
            "./rag_system/documents/invoice_1041.pdf"
        ],
        "processing": {
            "chunk_size": 512,
            "chunk_overlap": 64,
            "enable_enrich": True,
            "enable_latechunk": True,
            "enable_docling": True,
            "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
            "generation_model": "qwen3:0.6b",
            "retrieval_mode": "hybrid",
            "window_size": 2
        }
    }
    
    with open("batch_indexing_config.json", "w") as f:
        json.dump(sample_config, f, indent=2)
    
    print("📄 Sample batch configuration created: batch_indexing_config.json")


def main():
    """
    Main entry point for the script.
    
    Parses command line arguments and executes the appropriate index creation
    workflow based on the provided options.
    """
    parser = argparse.ArgumentParser(description="LocalGPT Index Creation Tool")
    parser.add_argument("--batch", help="Batch configuration file", type=str)
    parser.add_argument("--config", help="Custom pipeline configuration file", type=str)
    parser.add_argument("--create-sample", action="store_true", help="Create sample batch config")
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_batch_config()
        return
    
    try:
        creator = IndexCreator(config_path=args.config)
        
        if args.batch:
            creator.batch_create_from_config(args.batch)
        else:
            creator.create_index_interactive()
            
    except KeyboardInterrupt:
        print("\n\n❌ Operation cancelled by user.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
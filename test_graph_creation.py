#!/usr/bin/env python3
"""
Simple test script to verify knowledge graph creation during indexing.
"""
import tempfile
import os
import shutil
from rag_system.main import PIPELINE_CONFIGS
from rag_system.pipelines.indexing_pipeline import IndexingPipeline
from rag_system.utils.ollama_client import OllamaClient

def test_graph_creation():
    """Test knowledge graph creation during indexing."""
    print("🧪 Testing knowledge graph creation during indexing...")
    
    test_doc = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    test_doc.write('Apple Inc. was founded by Steve Jobs. Microsoft was founded by Bill Gates. Tesla was founded by Elon Musk.')
    test_doc.close()
    
    test_cache_dir = "./test_graph_creation_cache"
    
    try:
        config = PIPELINE_CONFIGS['default'].copy()
        config['retrieval']['graph']['enabled'] = True
        config['retrieval']['graph']['working_dir'] = test_cache_dir
        
        ollama_client = OllamaClient()
        ollama_config = {
            'generation_model': 'qwen2.5:0.5b',  # Can be any model user has configured
            'embedding_model': 'nomic-embed-text',  # Can be any embedding model user has configured
            'embedding_dim': 768
        }
        
        print("📄 Creating IndexingPipeline with graph enabled...")
        pipeline = IndexingPipeline(config, ollama_client, ollama_config)
        
        print("🔧 Running indexing pipeline...")
        pipeline.run([test_doc.name])
        
        print("✅ Graph creation test completed successfully!")
        
        if os.path.exists(test_cache_dir):
            files = os.listdir(test_cache_dir)
            print(f"📁 Graph cache directory contains: {files}")
        
    except Exception as e:
        print(f"❌ Graph creation test failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        if os.path.exists(test_doc.name):
            os.unlink(test_doc.name)
        if os.path.exists(test_cache_dir):
            shutil.rmtree(test_cache_dir, ignore_errors=True)
        if os.path.exists("./index_store"):
            shutil.rmtree("./index_store", ignore_errors=True)

if __name__ == "__main__":
    test_graph_creation()

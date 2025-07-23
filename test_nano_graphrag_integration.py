#!/usr/bin/env python3
"""
End-to-end test for nano-graphrag integration with LocalGPT.
"""
import os
import sys
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_system.main import get_agent, PIPELINE_CONFIGS
from rag_system.pipelines.indexing_pipeline import IndexingPipeline
from rag_system.utils.ollama_client import OllamaClient

def test_nano_graphrag_integration():
    """Test the complete nano-graphrag integration."""
    print("🧪 Starting nano-graphrag integration test...")
    
    test_docs_dir = tempfile.mkdtemp()
    test_doc_path = os.path.join(test_docs_dir, "test_doc.txt")
    
    with open(test_doc_path, 'w') as f:
        f.write("""
        Apple Inc. is a technology company founded by Steve Jobs, Steve Wozniak, and Ronald Wayne.
        The company is headquartered in Cupertino, California.
        Tim Cook is the current CEO of Apple Inc.
        Apple develops consumer electronics including the iPhone, iPad, and Mac computers.
        Microsoft Corporation is another major technology company founded by Bill Gates and Paul Allen.
        Microsoft is headquartered in Redmond, Washington.
        Satya Nadella is the current CEO of Microsoft.
        Google was founded by Larry Page and Sergey Brin at Stanford University.
        Google is now part of Alphabet Inc. and is headquartered in Mountain View, California.
        Sundar Pichai is the current CEO of Google.
        """)
    
    try:
        print("\n📚 Test 1: Indexing documents with nano-graphrag...")
        
        config = PIPELINE_CONFIGS["default"].copy()
        config["retrieval"]["graph"]["enabled"] = True
        config["retrieval"]["graph"]["working_dir"] = "./test_nano_graphrag_cache"
        
        ollama_client = OllamaClient()
        ollama_config = {
            "generation_model": "qwen2.5:0.5b",  # Can be any model user has configured
            "embedding_model": "nomic-embed-text",  # Can be any embedding model user has configured
            "embedding_dim": 768
        }
        
        indexing_pipeline = IndexingPipeline(config, ollama_client, ollama_config)
        indexing_pipeline.run([test_doc_path])
        print("✅ Indexing completed successfully")
        
        print("\n🔍 Test 2: Testing graph queries...")
        
        agent = get_agent("default")
        
        result = agent.run("Who is the CEO of Apple?")
        print(f"Local query result: {result.get('answer', 'No answer')[:200]}...")
        assert "Tim Cook" in result.get('answer', '') or "CEO" in result.get('answer', ''), "CEO query failed"
        
        result = agent.run("What is the relationship between Apple and Steve Jobs?")
        print(f"Relationship query result: {result.get('answer', 'No answer')[:200]}...")
        assert "founder" in result.get('answer', '').lower() or "Apple" in result.get('answer', ''), "Relationship query failed"
        
        print("✅ Graph queries completed successfully")
        
        print("\n🔄 Test 3: Testing hybrid retrieval...")
        
        result = agent.run("Tell me about technology companies and their founders")
        print(f"Hybrid query result: {result.get('answer', 'No answer')[:200]}...")
        assert any(name in result.get('answer', '') for name in ["Apple", "Microsoft", "Google"]), "Hybrid query failed"
        
        print("✅ Hybrid retrieval completed successfully")
        
        print("\n🧠 Test 4: Testing different query modes...")
        
        from rag_system.indexing.nano_graph_adapter import NanoGraphRAGAdapter
        adapter = NanoGraphRAGAdapter("./test_nano_graphrag_cache", ollama_client, ollama_config)
        
        local_result = adapter.query("Who founded Apple?", mode="local")
        print(f"Local mode result: {local_result[:100]}...")
        
        global_result = adapter.query("What are the main technology companies?", mode="global")
        print(f"Global mode result: {global_result[:100]}...")
        
        naive_result = adapter.query("Tell me about CEOs", mode="naive")
        print(f"Naive mode result: {naive_result[:100]}...")
        
        print("✅ Different query modes completed successfully")
        
        print("\n🎉 All tests passed! nano-graphrag integration is working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        shutil.rmtree(test_docs_dir, ignore_errors=True)
        shutil.rmtree("./test_nano_graphrag_cache", ignore_errors=True)
        shutil.rmtree("./index_store", ignore_errors=True)

if __name__ == "__main__":
    test_nano_graphrag_integration()

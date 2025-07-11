#!/usr/bin/env python3
"""
System Health Check for RAG System
Quick validation of configurations, models, and data access.
"""

import sys
import traceback
from pathlib import Path

def print_status(message, success=None):
    """Print status with emoji"""
    if success is True:
        print(f"✅ {message}")
    elif success is False:
        print(f"❌ {message}")
    else:
        print(f"🔍 {message}")

def check_imports():
    """Test basic imports"""
    print_status("Testing basic imports...")
    try:
        from rag_system.main import get_agent, EXTERNAL_MODELS, OLLAMA_CONFIG, PIPELINE_CONFIGS
        print_status("Basic imports successful", True)
        return True
    except Exception as e:
        print_status(f"Import failed: {e}", False)
        return False

def check_configurations():
    """Validate configurations"""
    print_status("Checking configurations...")
    try:
        from rag_system.main import EXTERNAL_MODELS, OLLAMA_CONFIG, PIPELINE_CONFIGS
        
        print(f"📊 External Models: {EXTERNAL_MODELS}")
        print(f"📊 Ollama Config: {OLLAMA_CONFIG}")
        print(f"📊 Pipeline Configs: {PIPELINE_CONFIGS}")
        
        # Check for common model dimension issues
        embedding_model = EXTERNAL_MODELS.get("embedding_model", "Unknown")
        if "bge-small" in embedding_model:
            print_status(f"Embedding model: {embedding_model} (384 dims)", True)
        elif "Qwen3-Embedding" in embedding_model:
            print_status(f"Embedding model: {embedding_model} (1024 dims) - Check data compatibility!", None)
        else:
            print_status(f"Embedding model: {embedding_model} - Verify dimensions!", None)
            
        print_status("Configuration check completed", True)
        return True
    except Exception as e:
        print_status(f"Configuration check failed: {e}", False)
        return False

def check_agent_initialization():
    """Test agent initialization"""
    print_status("Testing agent initialization...")
    try:
        from rag_system.main import get_agent
        agent = get_agent('default')
        print_status("Agent initialization successful", True)
        return agent
    except Exception as e:
        print_status(f"Agent initialization failed: {e}", False)
        traceback.print_exc()
        return None

def check_embedding_model(agent):
    """Test embedding model"""
    print_status("Testing embedding model...")
    try:
        embedder = agent.retrieval_pipeline._get_text_embedder()
        test_emb = embedder.create_embeddings(['test'])
        
        model_name = getattr(embedder.model, 'name_or_path', 'Unknown')
        dimensions = test_emb.shape[1]
        
        print_status(f"Embedding model: {model_name}", True)
        print_status(f"Vector dimension: {dimensions}", True)
        
        # Warn about dimension compatibility
        if dimensions == 384:
            print_status("Using 384-dim embeddings (bge-small compatible)", True)
        elif dimensions == 1024:
            print_status("Using 1024-dim embeddings (Qwen3 compatible) - Ensure data compatibility!", None)
        
        return True
    except Exception as e:
        print_status(f"Embedding model test failed: {e}", False)
        return False

def check_database_access():
    """Test database access"""
    print_status("Testing database access...")
    try:
        import lancedb
        db = lancedb.connect('./lancedb')
        tables = db.table_names()
        
        print_status(f"LanceDB connected - {len(tables)} tables available", True)
        if tables:
            print("📋 Available tables:")
            for table in tables[:5]:  # Show first 5 tables
                print(f"   - {table}")
            if len(tables) > 5:
                print(f"   ... and {len(tables) - 5} more")
        else:
            print_status("No tables found - may need to index documents first", None)
            
        return True
    except Exception as e:
        print_status(f"Database access failed: {e}", False)
        return False

def check_sample_query(agent):
    """Test a sample query if tables exist"""
    print_status("Testing sample query...")
    try:
        import lancedb
        db = lancedb.connect('./lancedb')
        tables = db.table_names()
        
        if not tables:
            print_status("No tables available for query test", None)
            return True
            
        # Use first available table
        table_name = tables[0]
        print_status(f"Testing query on table: {table_name}")
        
        result = agent.run('what is this document about?', table_name=table_name)
        
        if result and 'answer' in result:
            print_status("Sample query successful", True)
            print(f"📝 Answer preview: {result['answer'][:100]}...")
            print(f"📊 Found {len(result.get('source_documents', []))} source documents")
        else:
            print_status("Query returned empty result", None)
            
        return True
    except Exception as e:
        print_status(f"Sample query failed: {e}", False)
        return False

def main():
    """Run complete system health check"""
    print("🏥 RAG System Health Check")
    print("=" * 50)
    
    checks_passed = 0
    total_checks = 6
    
    # Basic checks
    if check_imports():
        checks_passed += 1
    
    if check_configurations():
        checks_passed += 1
    
    if check_database_access():
        checks_passed += 1
    
    # Agent-dependent checks
    agent = check_agent_initialization()
    if agent:
        checks_passed += 1
        
        if check_embedding_model(agent):
            checks_passed += 1
            
        if check_sample_query(agent):
            checks_passed += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"🏥 Health Check Complete: {checks_passed}/{total_checks} checks passed")
    
    if checks_passed == total_checks:
        print_status("System is healthy! 🎉", True)
        return 0
    elif checks_passed >= total_checks - 1:
        print_status("System mostly healthy with minor issues", None)
        return 0
    else:
        print_status("System has significant issues that need attention", False)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 
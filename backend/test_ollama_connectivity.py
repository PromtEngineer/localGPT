#!/usr/bin/env python3

import os
import sys

def check_ollama_connectivity():
    """Test Ollama connectivity from within Docker container"""
    print("🧪 Testing Ollama Connectivity")
    print("=" * 40)
    
    ollama_host = os.getenv('OLLAMA_HOST', 'Not set')
    print(f"OLLAMA_HOST environment variable: {ollama_host}")
    
    try:
        from ollama_client import OllamaClient
        client = OllamaClient()
        print(f"OllamaClient base_url: {client.base_url}")
        
        is_running = client.is_ollama_running()
        print(f"Ollama running: {is_running}")
        
        if is_running:
            models = client.list_models()
            print(f"Available models: {models}")
            print("✅ Ollama connectivity test passed!")
            return True
        else:
            print("❌ Ollama connectivity test failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Ollama connectivity: {e}")
        return False

if __name__ == "__main__":
    success = check_ollama_connectivity()
    sys.exit(0 if success else 1)

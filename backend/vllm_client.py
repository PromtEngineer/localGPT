import requests
import json
import os
from typing import List, Dict, Optional

class VLLMClient:
    def __init__(self, base_url: Optional[str] = None):
        if base_url is None:
            base_url = os.getenv("VLLM_HOST", "http://localhost:8000")
        self.base_url = base_url
        self.api_url = f"{base_url}/v1"
    
    def is_vllm_running(self) -> bool:
        """Check if vLLM server is running"""
        try:
            response = requests.get(f"{self.api_url}/models", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def list_models(self) -> List[str]:
        """Get list of available models"""
        try:
            response = requests.get(f"{self.api_url}/models")
            if response.status_code == 200:
                models = response.json().get("data", [])
                return [model["id"] for model in models]
            return []
        except requests.exceptions.RequestException as e:
            print(f"Error fetching models: {e}")
            return []
    
    def pull_model(self, model_name: str) -> bool:
        """vLLM doesn't support pulling models - models must be pre-loaded"""
        print(f"vLLM doesn't support pulling models. Model {model_name} must be pre-loaded when starting the server.")
        return False
    
    def chat(self, message: str, model: str = "qwen3:8b", conversation_history: Optional[List[Dict]] = None, enable_thinking: bool = True) -> str:
        """Send a chat message to vLLM using OpenAI-compatible API"""
        if conversation_history is None:
            conversation_history = []
        
        messages = conversation_history + [{"role": "user", "content": message}]
        
        try:
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                "temperature": 0.7,
                "max_tokens": 2048
            }
            
            response = requests.post(
                f"{self.api_url}/chat/completions",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                return f"Error: {response.status_code} - {response.text}"
                
        except requests.exceptions.RequestException as e:
            return f"Connection error: {e}"
    
    def chat_stream(self, message: str, model: str = "qwen3:8b", conversation_history: Optional[List[Dict]] = None, enable_thinking: bool = True):
        """Stream chat response from vLLM"""
        if conversation_history is None:
            conversation_history = []
        
        messages = conversation_history + [{"role": "user", "content": message}]
        
        try:
            payload = {
                "model": model,
                "messages": messages,
                "stream": True,
                "temperature": 0.7,
                "max_tokens": 2048
            }
            
            response = requests.post(
                f"{self.api_url}/chat/completions",
                json=payload,
                stream=True,
                timeout=60
            )
            
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            line_str = line.decode('utf-8')
                            if line_str.startswith('data: '):
                                data_str = line_str[6:]
                                if data_str.strip() == '[DONE]':
                                    break
                                data = json.loads(data_str)
                                if "choices" in data and len(data["choices"]) > 0:
                                    delta = data["choices"][0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        yield content
                        except json.JSONDecodeError:
                            continue
            else:
                yield f"Error: {response.status_code} - {response.text}"
                
        except requests.exceptions.RequestException as e:
            yield f"Connection error: {e}"

def main():
    """Test the vLLM client"""
    client = VLLMClient()
    
    if not client.is_vllm_running():
        print("❌ vLLM is not running. Please start vLLM server first.")
        print("Install: pip install vllm")
        print("Run: python -m vllm.entrypoints.openai.api_server --model <model_name>")
        return
    
    print("✅ vLLM is running!")
    
    # List available models
    models = client.list_models()
    print(f"Available models: {models}")
    
    if models:
        model_name = models[0]
        print(f"\n🤖 Testing chat with model: {model_name}")
        response = client.chat("Hello! Can you tell me a short joke?", model_name)
        print(f"AI: {response}")
    else:
        print("❌ No models available")

if __name__ == "__main__":
    main()

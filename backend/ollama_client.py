import requests
import json
import os
import re
from typing import List, Dict, Optional

DEFAULT_GENERATION_MODEL = os.getenv("GENERATION_MODEL", "qwen3.5:9b")

# Context-window sizing: identical scheme to rag_system/utils/ollama_client.py
# (duplicated because the gateway is a standalone stdlib-only service).
_NUM_CTX_BUCKETS = (8192, 16384, 32768)
_OUTPUT_HEADROOM_TOKENS = 2048


def _num_ctx_for(char_count: int) -> int:
    pinned = os.getenv("OLLAMA_NUM_CTX")
    if pinned:
        try:
            return max(2048, int(pinned))
        except ValueError:
            pass
    try:
        max_ctx = int(os.getenv("OLLAMA_NUM_CTX_MAX", "32768"))
    except ValueError:
        max_ctx = 32768
    estimated = char_count // 3 + _OUTPUT_HEADROOM_TOKENS
    for bucket in _NUM_CTX_BUCKETS:
        if estimated <= bucket <= max_ctx:
            return bucket
    return max_ctx


class OllamaClient:
    def __init__(self, base_url: Optional[str] = None):
        if base_url is None:
            base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
    
    def is_ollama_running(self) -> bool:
        """Check if Ollama server is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def list_models(self) -> List[str]:
        """Get list of available models"""
        try:
            response = requests.get(f"{self.api_url}/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [model["name"] for model in models]
            return []
        except requests.exceptions.RequestException as e:
            print(f"Error fetching models: {e}")
            return []
    
    def pull_model(self, model_name: str) -> bool:
        """Pull a model if not available"""
        try:
            response = requests.post(
                f"{self.api_url}/pull",
                json={"name": model_name},
                stream=True
            )
            
            if response.status_code == 200:
                print(f"Pulling model {model_name}...")
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        if "status" in data:
                            print(f"Status: {data['status']}")
                        if data.get("status") == "success":
                            return True
                return True
            return False
        except requests.exceptions.RequestException as e:
            print(f"Error pulling model: {e}")
            return False
    
    def chat(self, message: str, model: str = None, conversation_history: List[Dict] = None, enable_thinking: bool = True) -> str:
        """Send a chat message to Ollama.

        Raises requests.exceptions.RequestException (Timeout, ConnectionError,
        HTTPError) on failure — the gateway maps those to 504/502 instead of
        embedding error text in a 200 OK answer.
        """
        if model is None:
            model = DEFAULT_GENERATION_MODEL
        if conversation_history is None:
            conversation_history = []

        # Add user message to conversation
        messages = conversation_history + [{"role": "user", "content": message}]

        try:
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
            }

            # Multiple approaches to disable thinking tokens
            if not enable_thinking:
                payload.update({
                    "think": False,  # Native Ollama parameter
                    "options": {
                        "think": False,
                        "thinking": False,
                        "temperature": 0.7,
                        "top_p": 0.9
                    }
                })
            else:
                payload["think"] = True

            # Size the context window to the conversation: Ollama front-truncates
            # (silently drops the OLDEST messages/system prompt) when the request
            # exceeds its server-side slot, so request a window that fits. Same
            # scheme as rag_system/utils/ollama_client.py: bucketed, env-capped.
            payload.setdefault("options", {}).setdefault(
                "num_ctx",
                _num_ctx_for(sum(len(m.get("content") or "") for m in messages)),
            )
            
            response = requests.post(
                f"{self.api_url}/chat",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result["message"]["content"]

                # Additional cleanup: remove any thinking tokens that might slip through
                if not enable_thinking:
                    # Remove common thinking token patterns
                    response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL | re.IGNORECASE)
                    response_text = re.sub(r'<thinking>.*?</thinking>', '', response_text, flags=re.DOTALL | re.IGNORECASE)
                    response_text = response_text.strip()

                return response_text
            else:
                # Not answer text: raise so the gateway maps it to a 502.
                response.raise_for_status()

        except requests.exceptions.RequestException:
            # Timeout / ConnectionError / HTTPError propagate to the caller.
            raise
    
def main():
    """Test the Ollama client"""
    client = OllamaClient()
    
    # Check if Ollama is running
    if not client.is_ollama_running():
        print("❌ Ollama is not running. Please start Ollama first.")
        print("Install: https://ollama.ai")
        print("Run: ollama serve")
        return
    
    print("✅ Ollama is running!")
    
    # List available models
    models = client.list_models()
    print(f"Available models: {models}")
    
    # Try to use the configured generation model, pull if needed
    model_name = DEFAULT_GENERATION_MODEL
    if model_name not in models:
        print(f"Model {model_name} not found. Pulling...")
        if client.pull_model(model_name):
            print(f"✅ Model {model_name} pulled successfully!")
        else:
            print(f"❌ Failed to pull model {model_name}")
            return
    
    # Test chat
    print("\n🤖 Testing chat...")
    response = client.chat("Hello! Can you tell me a short joke?", model_name)
    print(f"AI: {response}")

if __name__ == "__main__":
    main()    
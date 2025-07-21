import requests
import json
from typing import List, Dict, Any
import base64
from io import BytesIO
from PIL import Image
import httpx, asyncio

class VLLMClient:
    """
    An enhanced client for vLLM that handles both text and embedding generation.
    """
    def __init__(self, host: str = "http://localhost:8000"):
        self.host = host
        self.api_url = f"{host}/v1"

    def generate_embedding(self, model: str, text: str) -> List[float]:
        try:
            response = requests.post(
                f"{self.api_url}/embeddings",
                json={"model": model, "input": text}
            )
            response.raise_for_status()
            data = response.json()
            return data["data"][0]["embedding"]
        except requests.exceptions.RequestException as e:
            print(f"Error generating embedding: {e}")
            return []

    def generate_completion(
        self,
        model: str,
        prompt: str,
        *,
        format: str = "",
        images: List[Image.Image] | None = None,
        enable_thinking: bool | None = None,
    ) -> Dict[str, Any]:
        """Generate completion using OpenAI-compatible API"""
        try:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "temperature": 0.7,
                "max_tokens": 2048
            }
            
            if format == "json":
                payload["response_format"] = {"type": "json_object"}

            response = requests.post(
                f"{self.api_url}/chat/completions",
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            return {
                "response": result["choices"][0]["message"]["content"],
                "done": True
            }

        except requests.exceptions.RequestException as e:
            print(f"Error generating completion: {e}")
            return {}

    async def generate_completion_async(
        self,
        model: str,
        prompt: str,
        *,
        format: str = "",
        images: List[Image.Image] | None = None,
        enable_thinking: bool | None = None,
        timeout: int = 60,
    ) -> Dict[str, Any]:
        """Asynchronous version of generate_completion using httpx."""
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "temperature": 0.7,
            "max_tokens": 2048
        }
        
        if format == "json":
            payload["response_format"] = {"type": "json_object"}

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(f"{self.api_url}/chat/completions", json=payload)
                resp.raise_for_status()
                result = resp.json()
                return {
                    "response": result["choices"][0]["message"]["content"],
                    "done": True
                }
        except (httpx.HTTPError, asyncio.CancelledError) as e:
            print(f"Async vLLM completion error: {e}")
            return {}

    def stream_completion(
        self,
        model: str,
        prompt: str,
        *,
        images: List[Image.Image] | None = None,
        enable_thinking: bool | None = None,
    ):
        """Generator that yields partial response strings as they arrive."""
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "temperature": 0.7,
            "max_tokens": 2048
        }

        with requests.post(f"{self.api_url}/chat/completions", json=payload, stream=True) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                try:
                    line_str = raw_line.decode('utf-8')
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

if __name__ == '__main__':
    print("vLLM client for RAG system with OpenAI-compatible API support.")
    try:
        client = VLLMClient()
        
        completion_response = client.generate_completion(
            model="qwen3:8b",
            prompt="What is the capital of France?"
        )
        
        if completion_response and 'response' in completion_response:
            print("\n--- Completion Test Response ---")
            print(completion_response['response'])
        else:
            print("\nFailed to get completion response. Is vLLM server running?")

    except Exception as e:
        print(f"An error occurred: {e}")

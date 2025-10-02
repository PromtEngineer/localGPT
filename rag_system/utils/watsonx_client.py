import json
from typing import List, Dict, Any, Optional
import base64
from io import BytesIO
from PIL import Image


class WatsonXClient:
    """
    A client for IBM Watson X AI that provides similar interface to OllamaClient
    for seamless integration with the RAG system.
    """
    def __init__(
        self,
        api_key: str,
        project_id: str,
        url: str = "https://us-south.ml.cloud.ibm.com",
    ):
        """
        Initialize the Watson X client.
        
        Args:
            api_key: IBM Cloud API key for authentication
            project_id: Watson X project ID
            url: Watson X service URL (default: us-south region)
        """
        self.api_key = api_key
        self.project_id = project_id
        self.url = url
        
        try:
            from ibm_watsonx_ai import APIClient
            from ibm_watsonx_ai import Credentials
            from ibm_watsonx_ai.foundation_models import ModelInference
            from ibm_watsonx_ai.foundation_models.schema import TextGenParameters
        except ImportError:
            raise ImportError(
                "ibm-watsonx-ai package is required. "
                "Install it with: pip install ibm-watsonx-ai"
            )
        
        self._APIClient = APIClient
        self._Credentials = Credentials
        self._ModelInference = ModelInference
        self._TextGenParameters = TextGenParameters
        
        self.credentials = self._Credentials(
            api_key=self.api_key,
            url=self.url
        )
        
        self.client = self._APIClient(self.credentials)
        self.client.set.default_project(self.project_id)

    def _extract_available_model_ids(self, specs: Dict[str, Any]) -> List[str]:
        "Return model_id entries that are currently available."
        model_ids: List[str] = []
        for resource in (specs or {}).get("resources", []):
            model_id = resource.get("model_id") or resource.get("id")
            if not model_id:
                continue
            lifecycle = resource.get("lifecycle") or []
            if lifecycle:
                states = {item.get("id") for item in lifecycle if isinstance(item, dict)}
                if states and "available" not in states:
                    continue
            model_ids.append(model_id)
        return model_ids

    def list_generation_models(self) -> List[str]:
        "List Watson X models suitable for text generation/chat."
        try:
            fm = self.client.foundation_models
            specs = fm.get_text_generation_model_specs()
            models = set(self._extract_available_model_ids(specs))
            try:
                chat_specs = fm.get_chat_model_specs()
                models.update(self._extract_available_model_ids(chat_specs))
            except Exception:
                pass
            return sorted(models)
        except Exception as exc:
            print(f"Error listing Watson X generation models: {exc}")
            return []

    def list_embedding_models(self) -> List[str]:
        "List Watson X embedding model identifiers."
        try:
            fm = self.client.foundation_models
            specs = fm.get_embeddings_model_specs()
            return sorted(set(self._extract_available_model_ids(specs)))
        except Exception as exc:
            print(f"Error listing Watson X embedding models: {exc}")
            return []

    def _image_to_base64(self, image: Image.Image) -> str:
        """Converts a Pillow Image to a base64 string."""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def generate_embedding(self, model: str, text: str) -> List[float]:
        """
        Generate embeddings using Watson X embedding models.
        Note: This requires using Watson X embedding models through the embeddings API.
        """
        try:
            from ibm_watsonx_ai.foundation_models import Embeddings
            
            embedding_model = Embeddings(
                model_id=model,
                credentials=self.credentials,
                project_id=self.project_id
            )
            
            result = embedding_model.embed_query(text)
            return result if isinstance(result, list) else []
            
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return []

    def generate_completion(
        self,
        model: str,
        prompt: str,
        *,
        format: str = "",
        images: Optional[List[Image.Image]] = None,
        enable_thinking: Optional[bool] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generates a completion using Watson X foundation models.
        
        Args:
            model: The name/ID of the Watson X model (e.g., 'ibm/granite-13b-chat-v2')
            prompt: The text prompt for the model
            format: The format for the response (e.g., "json")
            images: List of Pillow Image objects (for multimodal models)
            enable_thinking: Optional flag (not used in Watson X, kept for compatibility)
            **kwargs: Additional parameters for text generation
        
        Returns:
            Dictionary with response in Ollama-compatible format
        """
        try:
            gen_params = {}
            
            if kwargs.get('max_tokens'):
                gen_params['max_new_tokens'] = kwargs['max_tokens']
            if kwargs.get('temperature'):
                gen_params['temperature'] = kwargs['temperature']
            if kwargs.get('top_p'):
                gen_params['top_p'] = kwargs['top_p']
            if kwargs.get('top_k'):
                gen_params['top_k'] = kwargs['top_k']
            
            parameters = self._TextGenParameters(**gen_params) if gen_params else None
            
            model_inference = self._ModelInference(
                model_id=model,
                credentials=self.credentials,
                project_id=self.project_id,
                params=parameters
            )
            
            if images:
                print("Warning: Image support in Watson X may vary by model")
                result = model_inference.generate(prompt=prompt)
            else:
                result = model_inference.generate(prompt=prompt)
            
            generated_text = ""
            if isinstance(result, dict):
                generated_text = result.get('results', [{}])[0].get('generated_text', '')
            else:
                generated_text = str(result)
            
            return {
                'response': generated_text,
                'model': model,
                'done': True
            }
            
        except Exception as e:
            print(f"Error generating completion: {e}")
            return {'response': '', 'error': str(e)}

    async def generate_completion_async(
        self,
        model: str,
        prompt: str,
        *,
        format: str = "",
        images: Optional[List[Image.Image]] = None,
        enable_thinking: Optional[bool] = None,
        timeout: int = 60,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Asynchronous version of generate_completion.
        
        Note: IBM Watson X SDK may not have native async support,
        so this is a wrapper around the sync version.
        """
        import asyncio
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.generate_completion(
                model, prompt, format=format, images=images,
                enable_thinking=enable_thinking, **kwargs
            )
        )

    def stream_completion(
        self,
        model: str,
        prompt: str,
        *,
        images: Optional[List[Image.Image]] = None,
        enable_thinking: Optional[bool] = None,
        **kwargs
    ):
        """
        Generator that yields partial response strings as they arrive.
        
        Note: Watson X streaming support depends on the SDK version and model.
        """
        try:
            gen_params = {}
            if kwargs.get('max_tokens'):
                gen_params['max_new_tokens'] = kwargs['max_tokens']
            if kwargs.get('temperature'):
                gen_params['temperature'] = kwargs['temperature']
                
            parameters = self._TextGenParameters(**gen_params) if gen_params else None
            
            model_inference = self._ModelInference(
                model_id=model,
                credentials=self.credentials,
                project_id=self.project_id,
                params=parameters
            )
            
            try:
                for chunk in model_inference.generate_text_stream(prompt=prompt):
                    if chunk:
                        yield chunk
            except AttributeError:
                result = model_inference.generate(prompt=prompt)
                generated_text = ""
                if isinstance(result, dict):
                    generated_text = result.get('results', [{}])[0].get('generated_text', '')
                else:
                    generated_text = str(result)
                yield generated_text
                
        except Exception as e:
            print(f"Error in stream_completion: {e}")
            yield ""


if __name__ == '__main__':
    print("Watson X Client for IBM watsonx.ai integration")
    print("This client provides Ollama-compatible interface for Watson X granite models")
    print("\nTo use this client, you need:")
    print("1. IBM Cloud API key")
    print("2. Watson X project ID")
    print("3. ibm-watsonx-ai package installed")
    print("\nExample usage:")
    print("""
    from rag_system.utils.watsonx_client import WatsonXClient
    
    client = WatsonXClient(
        api_key="your-api-key",
        project_id="your-project-id"
    )
    
    response = client.generate_completion(
        model="ibm/granite-13b-chat-v2",
        prompt="What is AI?"
    )
    print(response['response'])
    """)

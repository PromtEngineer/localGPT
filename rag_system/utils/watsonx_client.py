import json
from typing import List, Dict, Any, Optional
import base64
from io import BytesIO
from PIL import Image

# The token-usage tracker lives with the Ollama client because that is where the
# counts originate; watsonx imports it only to report its own (absent) counts as
# zeros so the per-query summary stays well-formed on either backend.
from rag_system.utils.ollama_client import record_llm_usage


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

    def _image_to_base64(self, image: Image.Image) -> str:
        """Converts a Pillow Image to a base64 string."""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

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

            # Ollama-style options dict (interface parity): only `temperature`
            # maps onto TextGenParameters; the rest (num_ctx, …) are
            # Ollama-specific and accepted-and-ignored.
            options = kwargs.pop('options', None)
            if options and options.get('temperature') is not None:
                kwargs.setdefault('temperature', options['temperature'])

            if kwargs.get('max_tokens'):
                gen_params['max_new_tokens'] = kwargs['max_tokens']
            # `is not None` so an explicit temperature=0 (the deterministic
            # pin) is applied instead of dropped as falsy.
            if kwargs.get('temperature') is not None:
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
            
            generated_text = ""
            if isinstance(result, dict):
                generated_text = result.get('results', [{}])[0].get('generated_text', '')
            else:
                generated_text = str(result)
            
            # roadmap 4.5: the RAG system's token tracker reads Ollama's
            # `prompt_eval_count` / `eval_count`. watsonx's SDK does not surface
            # comparable per-call counts through this code path, so report zeros
            # rather than omitting the keys — a watsonx run then shows an honest
            # "0 tokens counted" instead of silently looking like a cache hit.
            payload = {
                'response': generated_text,
                'model': model,
                'done': True,
                'prompt_eval_count': 0,
                'eval_count': 0,
            }
            record_llm_usage(payload)
            return payload

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
        options: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Asynchronous version of generate_completion.

        Note: IBM Watson X SDK may not have native async support,
        so this is a wrapper around the sync version.

        *options* mirrors OllamaClient's; the sync method translates
        `temperature` into watsonx generation params and ignores the rest.
        """
        import asyncio

        # Inside a coroutine a loop is always running; get_event_loop() is
        # deprecated here and can raise when no loop has been set.
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.generate_completion(
                model, prompt, format=format, images=images,
                enable_thinking=enable_thinking, options=options, **kwargs
            )
        )

    def stream_completion(
        self,
        model: str,
        prompt: str,
        *,
        images: Optional[List[Image.Image]] = None,
        enable_thinking: Optional[bool] = None,
        stats: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Generator that yields partial response strings as they arrive.

        Note: Watson X streaming support depends on the SDK version and model.

        *stats* mirrors ``OllamaClient.stream_completion`` (roadmap 4.5). watsonx
        exposes no per-stream token counts here, so it is filled with zeros.
        """
        if stats is not None:
            stats.update({'prompt_eval_count': 0, 'eval_count': 0, 'done': True})
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
            # Yielding "" made a failure look like a valid empty content
            # chunk; log and stop the iteration instead.
            print(f"Error in stream_completion: {e}")
            return


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

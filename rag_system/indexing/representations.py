from typing import List, Dict, Any, Protocol
import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch
import os

# We keep the protocol to ensure a consistent interface
class EmbeddingModel(Protocol):
    def create_embeddings(self, texts: List[str]) -> np.ndarray: ...

# Global cache for models - use dict to cache by model name
_MODEL_CACHE = {}

# --- New Ollama Embedder ---
class QwenEmbedder(EmbeddingModel):
    """
    An embedding model that uses a local Hugging Face transformer model.
    """
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B"):
        self.model_name = model_name
        # Auto-select the best available device: CUDA > MPS > CPU
        if torch.cuda.is_available():
            self.device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        # Use model-specific cache
        if model_name not in _MODEL_CACHE:
            print(f"Initializing HF Embedder with model '{model_name}' on device '{self.device}'. (first load)")
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
            model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device != "cpu" else None,
            ).to(self.device).eval()
            _MODEL_CACHE[model_name] = (tokenizer, model)
            print(f"QwenEmbedder weights loaded and cached for {model_name}.")
        else:
            print(f"Reusing cached QwenEmbedder weights for {model_name}.")
        
        self.tokenizer, self.model = _MODEL_CACHE[model_name]

    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        print(f"Generating {len(texts)} embeddings with {self.model_name} model...")
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden = outputs.last_hidden_state  # [B, seq, dim]
            # Pool via last valid token per sequence (recommended for Qwen3)
            seq_len = inputs["attention_mask"].sum(dim=1) - 1  # index of last token
            batch_indices = torch.arange(last_hidden.size(0), device=self.device)
            embeddings = last_hidden[batch_indices, seq_len]
        
        # Convert to numpy and validate
        embeddings_np = embeddings.cpu().numpy()
        
        # Check for NaN or infinite values
        if np.isnan(embeddings_np).any():
            print(f"⚠️ Warning: NaN values detected in embeddings from {self.model_name}")
            # Replace NaN values with zeros
            embeddings_np = np.nan_to_num(embeddings_np, nan=0.0, posinf=0.0, neginf=0.0)
            print(f"🔄 Replaced NaN values with zeros")
        
        if np.isinf(embeddings_np).any():
            print(f"⚠️ Warning: Infinite values detected in embeddings from {self.model_name}")
            # Replace infinite values with zeros
            embeddings_np = np.nan_to_num(embeddings_np, nan=0.0, posinf=0.0, neginf=0.0)
            print(f"🔄 Replaced infinite values with zeros")
        
        return embeddings_np

class EmbeddingGenerator:
    def __init__(self, embedding_model: EmbeddingModel, batch_size: int = 50):
        self.model = embedding_model
        self.batch_size = batch_size

    def generate(self, chunks: List[Dict[str, Any]]) -> List[np.ndarray]:
        """Generate embeddings for all chunks using batch processing"""
        texts_to_embed = [chunk['text'] for chunk in chunks]
        if not texts_to_embed: 
            return []
        
        from rag_system.utils.batch_processor import BatchProcessor, estimate_memory_usage
        
        memory_mb = estimate_memory_usage(chunks)
        print(f"Estimated memory usage for {len(chunks)} chunks: {memory_mb:.1f}MB")
        
        batch_processor = BatchProcessor(batch_size=self.batch_size)
        
        def process_text_batch(text_batch):
            if not text_batch:
                return []
            batch_embeddings = self.model.create_embeddings(text_batch)
            return [embedding for embedding in batch_embeddings]
        
        all_embeddings = batch_processor.process_in_batches(
            texts_to_embed,
            process_text_batch,
            "Embedding Generation"
        )
        
        return all_embeddings

class OllamaEmbedder(EmbeddingModel):
    """Call Ollama's /api/embeddings endpoint for each text."""
    def __init__(self, model_name: str, host: str | None = None, timeout: int = 60):
        self.model_name = model_name
        self.host = (host or os.getenv("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        self.timeout = timeout

    def _embed_single(self, text: str):
        import requests, numpy as np, json
        payload = {"model": self.model_name, "prompt": text}
        r = requests.post(f"{self.host}/api/embeddings", json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        # Ollama may return {"embedding": [...]} or {"data": [...]} depending on version
        vec = data.get("embedding") or data.get("data")
        if vec is None:
            raise ValueError("Unexpected Ollama embeddings response format")
        return np.array(vec, dtype="float32")

    def create_embeddings(self, texts: List[str]):
        import numpy as np
        vectors = [self._embed_single(t) for t in texts]
        embeddings_np = np.vstack(vectors)
        
        # Check for NaN or infinite values
        if np.isnan(embeddings_np).any():
            print(f"⚠️ Warning: NaN values detected in Ollama embeddings from {self.model_name}")
            # Replace NaN values with zeros
            embeddings_np = np.nan_to_num(embeddings_np, nan=0.0, posinf=0.0, neginf=0.0)
            print(f"🔄 Replaced NaN values with zeros")
        
        if np.isinf(embeddings_np).any():
            print(f"⚠️ Warning: Infinite values detected in Ollama embeddings from {self.model_name}")
            # Replace infinite values with zeros
            embeddings_np = np.nan_to_num(embeddings_np, nan=0.0, posinf=0.0, neginf=0.0)
            print(f"🔄 Replaced infinite values with zeros")
        
        return embeddings_np

def select_embedder(model_name: str, ollama_host: str | None = None):
    """Return appropriate EmbeddingModel implementation for the given name."""
    if "/" in model_name or model_name.startswith("http"):
        # Treat as HF model path
        return QwenEmbedder(model_name=model_name)
    # Otherwise assume it's an Ollama tag
    return OllamaEmbedder(model_name=model_name, host=ollama_host)

if __name__ == '__main__':
    print("representations.py cleaned up.")
    try:
        qwen_embedder = QwenEmbedder()
        emb_gen = EmbeddingGenerator(embedding_model=qwen_embedder)
        
        sample_chunks = [{'text': 'Hello world'}, {'text': 'This is a test'}]
        embeddings = emb_gen.generate(sample_chunks)
        
        print(f"\nSuccessfully generated {len(embeddings)} embeddings.")
        print(f"Shape of first embedding: {embeddings[0].shape}")

    except Exception as e:
        print(f"\nAn error occurred during the QwenEmbedder test: {e}")
        print("Please ensure you have an internet connection for model downloads.")
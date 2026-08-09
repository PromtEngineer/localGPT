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

# ---------------------------------------------------------------------------
# Query-side instruction prefix
# ---------------------------------------------------------------------------
# Instruction-tuned decoder embedders (Qwen3-Embedding, microsoft/harrier-oss-v1)
# are trained with an instruction on the QUERY side only. Both model cards use
# the identical wire format and the identical MS-MARCO-style retrieval task
# string, and both state explicitly that documents must be embedded WITHOUT any
# instruction.
#
#   query    -> "Instruct: {task}\nQuery: {text}"
#   document -> "{text}"                            (unchanged, always)
#
# The asymmetry is what keeps this change index-compatible: an index built
# before the prefix existed stays valid, because nothing on the document side
# moves. Only the query vector changes.
#
# An embedder instance carries the instruction; it does not decide per call.
# The indexing pipeline builds its embedder without one, the retrieval pipeline
# builds its embedder with one, and neither can leak into the other.

QUERY_PROMPT_TEMPLATE = "Instruct: {instruction}\nQuery: {text}"

# The official retrieval task description used by both model families.
DEFAULT_RETRIEVAL_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)

# Model-name fragments whose families are instruction-tuned in this format.
_INSTRUCTION_TUNED_FAMILIES = ("qwen3-embedding", "harrier")


def default_query_instruction(model_name: str) -> str:
    """The retrieval instruction a model family expects, or "" when it wants none.

    Returning "" (not None) is deliberate: "" means "this model takes no
    instruction", which is a decision, whereas None means "nobody decided yet"
    and is what callers pass to ask for this default.
    """
    name = (model_name or "").lower()
    if any(fragment in name for fragment in _INSTRUCTION_TUNED_FAMILIES):
        return DEFAULT_RETRIEVAL_INSTRUCTION
    return ""


def apply_query_instruction(texts: List[str], instruction: str | None) -> List[str]:
    """Prefix every text with the instruction block, or return them untouched."""
    if not instruction:
        return texts
    return [QUERY_PROMPT_TEMPLATE.format(instruction=instruction, text=t) for t in texts]

# --- New Ollama Embedder ---
class QwenEmbedder(EmbeddingModel):
    """
    An embedding model that uses a local Hugging Face transformer model.
    """
    MAX_TOKENS = 8192

    def __init__(self, model_name: str | None = None, query_instruction: str | None = None):
        if not model_name:
            from rag_system.main import EXTERNAL_MODELS
            model_name = EXTERNAL_MODELS["embedding_model"]
        self.model_name = model_name
        # "" / None => this instance embeds raw text (the document side).
        # A non-empty string => this instance is a QUERY embedder and prefixes
        # every text it is given with the instruction block.
        self.query_instruction = query_instruction or ""
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
        # Some tokenizers report a sentinel model_max_length; clamp it so that
        # truncation=True actually truncates.
        reported = getattr(self.tokenizer, "model_max_length", None)
        self.max_length = min(reported, self.MAX_TOKENS) if isinstance(reported, int) and reported > 0 else self.MAX_TOKENS

    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        print(f"Generating {len(texts)} embeddings with {self.model_name} model...")
        texts = apply_query_instruction(texts, self.query_instruction)
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden = outputs.last_hidden_state  # [B, seq, dim]
            # Last-token pooling (recommended for Qwen3-Embedding). The tokenizer
            # pads on the left, in which case the final column is the last real
            # token for every row; handle right padding too for safety.
            attention_mask = inputs["attention_mask"]
            left_padded = bool(attention_mask[:, -1].min().item())
            if left_padded:
                embeddings = last_hidden[:, -1]
            else:
                seq_len = attention_mask.sum(dim=1) - 1  # index of last token
                batch_indices = torch.arange(last_hidden.size(0), device=last_hidden.device)
                embeddings = last_hidden[batch_indices, seq_len]

        # Convert to numpy and validate
        embeddings_np = embeddings.float().cpu().numpy()
        
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
    def __init__(self, model_name: str, host: str | None = None, timeout: int = 60,
                 query_instruction: str | None = None):
        self.model_name = model_name
        self.host = (host or os.getenv("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        self.timeout = timeout
        # Same contract as QwenEmbedder: set on the query-side instance only.
        self.query_instruction = query_instruction or ""

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
        texts = apply_query_instruction(texts, self.query_instruction)
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

def select_embedder(model_name: str, ollama_host: str | None = None,
                    query_instruction: str | None = None):
    """Return appropriate EmbeddingModel implementation for the given name.

    ``query_instruction`` is the query-side instruction prefix. Leave it unset
    (the default) for document-side embedders — that is what keeps indexes
    stable across this change. Callers on the query path pass the instruction
    explicitly; see ``RetrievalPipeline._get_text_embedder``.
    """
    if "/" in model_name or model_name.startswith("http"):
        # Treat as HF model path
        return QwenEmbedder(model_name=model_name, query_instruction=query_instruction)
    # Otherwise assume it's an Ollama tag
    return OllamaEmbedder(model_name=model_name, host=ollama_host,
                          query_instruction=query_instruction)

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
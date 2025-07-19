from typing import List, Dict, Any, Optional
import re
from transformers import AutoTokenizer

class MarkdownRecursiveChunker:
    """
    A recursive chunker that splits Markdown text based on its semantic structure
    and embeds document-level metadata into each chunk.
    """

    def __init__(self, max_chunk_size: int = 1500, min_chunk_size: int = 200, tokenizer_model: str = "Qwen/Qwen3-Embedding-0.6B"):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.split_priority = ["\n## ", "\n### ", "\n#### ", "```", "\n\n"]
        
        repo_id = tokenizer_model
        if "/" not in tokenizer_model and not tokenizer_model.startswith("Qwen/"):
            repo_id = {
                "qwen3-embedding-0.6b": "Qwen/Qwen3-Embedding-0.6B",
            }.get(tokenizer_model.lower(), tokenizer_model)
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
        except Exception as e:
            print(f"Warning: Failed to load tokenizer {repo_id}: {e}")
            print("Falling back to character-based approximation (4 chars ≈ 1 token)")
            self.tokenizer = None

    def _token_len(self, text: str) -> int:
        """Get token count for text using the tokenizer."""
        if self.tokenizer is not None:
            return len(self.tokenizer.tokenize(text))
        else:
            return max(1, len(text) // 4)
    
    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        final_chunks = []
        chunks_to_process = [text]
        
        for sep in separators:
            new_chunks = []
            for chunk in chunks_to_process:
                if self._token_len(chunk) > self.max_chunk_size:
                    sub_chunks = re.split(f'({sep})', chunk)
                    combined = []
                    i = 0
                    while i < len(sub_chunks):
                        if i + 1 < len(sub_chunks) and sub_chunks[i+1] == sep:
                            combined.append(sub_chunks[i+1] + sub_chunks[i+2])
                            i += 3
                        else:
                            if sub_chunks[i]:
                                combined.append(sub_chunks[i])
                            i += 1
                    new_chunks.extend(combined)
                else:
                    new_chunks.append(chunk)
            chunks_to_process = new_chunks
        
        final_chunks = []
        for chunk in chunks_to_process:
            if self._token_len(chunk) > self.max_chunk_size:
                words = chunk.split()
                current_chunk = ""
                for word in words:
                    test_chunk = current_chunk + " " + word if current_chunk else word
                    if self._token_len(test_chunk) <= self.max_chunk_size:
                        current_chunk = test_chunk
                    else:
                        if current_chunk:
                            final_chunks.append(current_chunk)
                        current_chunk = word
                if current_chunk:
                    final_chunks.append(current_chunk)
            else:
                final_chunks.append(chunk)

        return final_chunks

    def chunk(self, text: str, document_id: str, document_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Chunks the Markdown text and injects metadata.

        Args:
            text: The Markdown text to chunk.
            document_id: The identifier for the source document.
            document_metadata: A dictionary of metadata for the source document.

        Returns:
            A list of dictionaries, where each dictionary is a chunk with metadata.
        """
        if not text:
            return []

        raw_chunks = self._split_text(text, self.split_priority)
        
        merged_chunks_text = []
        current_chunk = ""
        for chunk_text in raw_chunks:
            test_chunk = current_chunk + chunk_text if current_chunk else chunk_text
            if not current_chunk or self._token_len(test_chunk) <= self.max_chunk_size:
                current_chunk = test_chunk
            elif self._token_len(current_chunk) < self.min_chunk_size:
                 current_chunk = test_chunk
            else:
                merged_chunks_text.append(current_chunk)
                current_chunk = chunk_text
        if current_chunk:
            merged_chunks_text.append(current_chunk)

        final_chunks = []
        for i, chunk_text in enumerate(merged_chunks_text):
            # Combine document-level metadata with chunk-specific metadata
            combined_metadata = (document_metadata or {}).copy()
            combined_metadata.update({
                "document_id": document_id,
                "chunk_number": i,
            })
            
            final_chunks.append({
                "chunk_id": f"{document_id}_{i}", # Create a more unique ID
                "text": chunk_text.strip(),
                "metadata": combined_metadata
            })

        return final_chunks

def create_contextual_window(all_chunks: List[Dict[str, Any]], chunk_index: int, window_size: int = 1) -> str:
    if not (0 <= chunk_index < len(all_chunks)):
        raise ValueError("chunk_index is out of bounds.")
    start = max(0, chunk_index - window_size)
    end = min(len(all_chunks), chunk_index + window_size + 1)
    context_chunks = all_chunks[start:end]
    return " ".join([chunk['text'] for chunk in context_chunks])

if __name__ == '__main__':
    print("chunking.py updated to include document metadata in each chunk.")
    
    sample_markdown = "# Doc Title\n\nContent paragraph."
    doc_meta = {"title": "My Awesome Document", "author": "Jane Doe", "year": 2024}
    
    chunker = MarkdownRecursiveChunker()
    chunks = chunker.chunk(
        text=sample_markdown, 
        document_id="doc456", 
        document_metadata=doc_meta
    )
    
    print(f"\n--- Created {len(chunks)} chunk(s) ---")
    for chunk in chunks:
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Text: '{chunk['text']}'")
        print(f"Metadata: {chunk['metadata']}")
        print("-" * 20)

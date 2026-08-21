from typing import List, Dict, Any, Optional
import re
from transformers import AutoTokenizer

class MarkdownRecursiveChunker:
    """
    A recursive chunker that splits Markdown text based on its semantic structure
    and embeds document-level metadata into each chunk.
    """

    def __init__(self, max_chunk_size: int = 1500, min_chunk_size: int = 200, tokenizer_model: str | None = None):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.split_priority = ["\n## ", "\n### ", "\n#### ", "```", "\n\n"]

        if not tokenizer_model:
            from rag_system.main import EXTERNAL_MODELS
            tokenizer_model = EXTERNAL_MODELS["embedding_model"]

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, trust_remote_code=True)
        except Exception as e:
            print(f"Warning: Failed to load tokenizer {tokenizer_model}: {e}")
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
                    # re.split with a capture group interleaves segments and
                    # separators: [seg0, sep, seg1, sep, seg2, ...]. Reattach
                    # each separator to the segment that FOLLOWS it, keeping
                    # every segment. (The previous loop advanced 3 positions
                    # after consuming 2, silently dropping seg0 and every
                    # other body segment of any document large enough to
                    # split — real corpora lost ~half their text.)
                    sub_chunks = re.split(f'({re.escape(sep)})', chunk)
                    combined = []
                    if sub_chunks and sub_chunks[0]:
                        combined.append(sub_chunks[0])
                    for j in range(1, len(sub_chunks) - 1, 2):
                        piece = sub_chunks[j] + sub_chunks[j + 1]
                        if piece:
                            combined.append(piece)
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
            else:
                # An undersized current_chunk is emitted as-is when merging it
                # forward would push the result past max_chunk_size — a merge
                # must never overshoot the token budget.
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

    def _doc_id(chunk: Dict[str, Any]):
        metadata = chunk.get("metadata") or {}
        return metadata.get("document_id", chunk.get("document_id"))

    # The flat list spans many documents; clamp the window to chunks of the
    # same document so one document's context never leaks into another's.
    doc_id = _doc_id(all_chunks[chunk_index])
    start = chunk_index
    while start > 0 and chunk_index - (start - 1) <= window_size and _doc_id(all_chunks[start - 1]) == doc_id:
        start -= 1
    end = chunk_index + 1
    while end < len(all_chunks) and end - chunk_index <= window_size and _doc_id(all_chunks[end]) == doc_id:
        end += 1
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

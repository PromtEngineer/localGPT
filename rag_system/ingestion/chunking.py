from typing import List, Dict, Any, Optional
import re

class MarkdownRecursiveChunker:
    """
    A recursive chunker that splits Markdown text based on its semantic structure
    and embeds document-level metadata into each chunk.
    """

    def __init__(self, max_chunk_size: int = 1500, min_chunk_size: int = 200):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.split_priority = ["\n## ", "\n### ", "\n#### ", "```", "\n\n"]

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        final_chunks = []
        chunks_to_process = [text]
        
        for sep in separators:
            new_chunks = []
            for chunk in chunks_to_process:
                if len(chunk) > self.max_chunk_size:
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
            if len(chunk) > self.max_chunk_size:
                final_chunks.extend([chunk[i:i+self.max_chunk_size] for i in range(0, len(chunk), self.max_chunk_size)])
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
            if not current_chunk or len(current_chunk) + len(chunk_text) <= self.max_chunk_size:
                current_chunk += chunk_text
            elif len(current_chunk) < self.min_chunk_size:
                 current_chunk += chunk_text
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
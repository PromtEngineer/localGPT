from typing import List, Dict, Any
from rag_system.utils.ollama_client import OllamaClient
from rag_system.ingestion.chunking import create_contextual_window
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the structured prompt templates, adapted from the example
SYSTEM_PROMPT = "You are an expert at summarizing and providing context for document sections based on their local surroundings."

LOCAL_CONTEXT_PROMPT_TEMPLATE = """<local_context>
{local_context_text}
</local_context>"""

CHUNK_PROMPT_TEMPLATE = """Here is the specific chunk we want to situate within the local context provided:
<chunk>
{chunk_content}
</chunk>

Based *only* on the local context provided, give a very short (2-5 sentence) context summary to situate this specific chunk. 
Focus on the chunk's topic and its relation to the immediately surrounding text shown in the local context. 
Focus on the the overall theme of the context, make sure to include topics, concepts, and other relevant information.
Answer *only* with the succinct context and nothing else."""

class ContextualEnricher:
    """
    Enriches chunks with a prepended summary of their surrounding context using Ollama,
    while preserving the original text.
    """
    def __init__(self, llm_client: OllamaClient, llm_model: str, batch_size: int = 10):
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.batch_size = batch_size
        logger.info(f"Initialized ContextualEnricher with Ollama model '{self.llm_model}' (batch_size={batch_size}).")

    def _generate_summary(self, local_context_text: str, chunk_text: str) -> str:
        """Generates a contextual summary using a structured, multi-part prompt."""
        # Combine the templates to form the final content for the HumanMessage equivalent
        human_prompt_content = (
            f"{LOCAL_CONTEXT_PROMPT_TEMPLATE.format(local_context_text=local_context_text)}\n\n"
            f"{CHUNK_PROMPT_TEMPLATE.format(chunk_content=chunk_text)}"
        )

        try:
            # Although we don't use LangChain's message objects, we can simulate the
            # System + Human message structure in the single prompt for the Ollama client.
            # A common way is to provide the system prompt and then the user's request.
            full_prompt = f"{SYSTEM_PROMPT}\n\n{human_prompt_content}"
            
            response = self.llm_client.generate_completion(self.llm_model, full_prompt, enable_thinking=False)
            summary_raw = response.get('response', '').strip()

            # --- Sanitize the summary to remove chain-of-thought markers ---
            # Many Qwen models wrap reasoning in <think>...</think> or similar tags.
            cleaned = re.sub(r'<think[^>]*>.*?</think>', '', summary_raw, flags=re.IGNORECASE | re.DOTALL)
            # Remove any assistant role tags that may appear
            cleaned = re.sub(r'<assistant[^>]*>|</assistant>', '', cleaned, flags=re.IGNORECASE)
            # If the model used an explicit "Answer:" delimiter keep only the part after it
            if 'Answer:' in cleaned:
                cleaned = cleaned.split('Answer:', 1)[1]

            # Take the first non-empty line to avoid leftover blank lines
            summary = next((ln.strip() for ln in cleaned.splitlines() if ln.strip()), '')

            # Fallback to raw if cleaning removed everything
            if not summary:
                summary = summary_raw

            if not summary or len(summary) < 5:
                logger.warning("Generated context summary is too short or empty. Skipping enrichment for this chunk.")
                return ""
            
            return summary

        except Exception as e:
            logger.error(f"LLM invocation failed during contextualization: {e}", exc_info=True)
            return "" # Gracefully fail by returning no summary

    def enrich_chunks(self, chunks: List[Dict[str, Any]], window_size: int = 1) -> List[Dict[str, Any]]:
        if not chunks:
            return []

        logger.info(f"Enriching {len(chunks)} chunks with contextual summaries (window_size={window_size}) using Ollama...")
        
        # Import batch processor
        from rag_system.utils.batch_processor import BatchProcessor, estimate_memory_usage
        
        # Estimate memory usage
        memory_mb = estimate_memory_usage(chunks)
        logger.info(f"Estimated memory usage for contextual enrichment: {memory_mb:.1f}MB")
        
        # Use batch processing for better performance and progress tracking
        batch_processor = BatchProcessor(batch_size=self.batch_size)
        
        def process_chunk_batch(chunk_indices):
            """Process a batch of chunk indices for contextual enrichment"""
            batch_results = []
            for i in chunk_indices:
                chunk = chunks[i]
                try:
                    local_context_text = create_contextual_window(chunks, chunk_index=i, window_size=window_size)
                    
                    # The summary is generated based on the original, unmodified text
                    original_text = chunk['text']
                    summary = self._generate_summary(local_context_text, original_text)
                    
                    new_chunk = chunk.copy()
                    
                    # Ensure metadata is a dictionary
                    if 'metadata' not in new_chunk or not isinstance(new_chunk['metadata'], dict):
                        new_chunk['metadata'] = {}

                    # Store original text and summary in metadata
                    new_chunk['metadata']['original_text'] = original_text
                    new_chunk['metadata']['contextual_summary'] = "N/A"

                    # Prepend the context summary ONLY if it was successfully generated
                    if summary:
                        new_chunk['text'] = f"Context: {summary}\n\n---\n\n{original_text}"
                        new_chunk['metadata']['contextual_summary'] = summary
                    
                    batch_results.append(new_chunk)
                    
                except Exception as e:
                    logger.error(f"Error enriching chunk {i}: {e}")
                    # Return original chunk if enrichment fails
                    batch_results.append(chunk)
                    
            return batch_results
        
        # Create list of chunk indices for batch processing
        chunk_indices = list(range(len(chunks)))
        
        # Process chunks in batches
        enriched_chunks = batch_processor.process_in_batches(
            chunk_indices,
            process_chunk_batch,
            "Contextual Enrichment"
        )
        
        return enriched_chunks
    
    def enrich_chunks_sequential(self, chunks: List[Dict[str, Any]], window_size: int = 1) -> List[Dict[str, Any]]:
        """Sequential enrichment method (legacy) - kept for comparison"""
        if not chunks:
            return []

        logger.info(f"Enriching {len(chunks)} chunks sequentially (window_size={window_size})...")
        enriched_chunks = []
        
        for i, chunk in enumerate(chunks):
            local_context_text = create_contextual_window(chunks, chunk_index=i, window_size=window_size)
            
            # The summary is generated based on the original, unmodified text
            original_text = chunk['text']
            summary = self._generate_summary(local_context_text, original_text)
            
            new_chunk = chunk.copy()
            
            # Ensure metadata is a dictionary
            if 'metadata' not in new_chunk or not isinstance(new_chunk['metadata'], dict):
                new_chunk['metadata'] = {}

            # Store original text and summary in metadata
            new_chunk['metadata']['original_text'] = original_text
            new_chunk['metadata']['contextual_summary'] = "N/A"

            # Prepend the context summary ONLY if it was successfully generated
            if summary:
                new_chunk['text'] = f"Context: {summary}\n\n---\n\n{original_text}"
                new_chunk['metadata']['contextual_summary'] = summary
            
            enriched_chunks.append(new_chunk)
            
            if (i + 1) % 10 == 0 or i == len(chunks) - 1:
                logger.info(f"  ...processed {i+1}/{len(chunks)} chunks.")
            
        return enriched_chunks
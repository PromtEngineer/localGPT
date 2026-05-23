from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from rag_system.ingestion.chunking import create_contextual_window
import logging
import re

# Chunks shorter than this are headers/footers unlikely to benefit from context.
_MIN_ENRICH_CHARS = 100

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
    Enriches chunks with a prepended summary of their surrounding context.
    Works with any LLM backend (Ollama, Anthropic, OpenAI, Groq) as long as
    the client implements generate_completion(model, prompt, **kwargs) -> {"response": str}.
    """
    def __init__(self, llm_client, llm_model: str, batch_size: int = 10,
                 timeout: int = 90, max_workers: int = 4):
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.batch_size = batch_size
        self.timeout = timeout
        self.max_workers = max_workers
        provider = type(llm_client).__name__
        logger.info(f"Initialized ContextualEnricher with {provider} model '{self.llm_model}' (batch_size={batch_size}, timeout={timeout}s, max_workers={max_workers}).")

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
            
            response = self.llm_client.generate_completion(
                self.llm_model,
                full_prompt,
                enable_thinking=False,
                timeout=self.timeout,
            )
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

    def _enrich_one(self, i: int, chunks: List[Dict[str, Any]], window_size: int) -> tuple[int, Dict[str, Any]]:
        """Enrich a single chunk. Returns (index, result) so callers can reassemble order."""
        chunk = chunks[i]
        original_text = chunk.get('text', '')

        # Skip trivial chunks — headers, footers, page numbers, etc.
        if len(original_text) < _MIN_ENRICH_CHARS:
            return i, chunk

        try:
            local_context_text = create_contextual_window(chunks, chunk_index=i, window_size=window_size)
            summary = self._generate_summary(local_context_text, original_text)

            new_chunk = chunk.copy()
            if not isinstance(new_chunk.get('metadata'), dict):
                new_chunk['metadata'] = {}
            new_chunk['metadata']['original_text'] = original_text
            new_chunk['metadata']['contextual_summary'] = summary or 'N/A'
            if summary:
                new_chunk['text'] = f"Context: {summary}\n\n---\n\n{original_text}"
            return i, new_chunk

        except Exception as e:
            logger.error(f"Error enriching chunk {i}: {e}")
            return i, chunk

    def enrich_chunks(self, chunks: List[Dict[str, Any]], window_size: int = 1) -> List[Dict[str, Any]]:
        if not chunks:
            return []

        skippable = sum(1 for c in chunks if len(c.get('text', '')) < _MIN_ENRICH_CHARS)
        to_enrich = len(chunks) - skippable
        logger.info(
            f"Enriching {to_enrich}/{len(chunks)} chunks (skipping {skippable} trivial) "
            f"window_size={window_size} workers={self.max_workers}"
        )

        enriched: List[Dict[str, Any]] = [None] * len(chunks)  # type: ignore[list-item]

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(self._enrich_one, i, chunks, window_size): i for i in range(len(chunks))}
            for future in as_completed(futures):
                try:
                    idx, result = future.result()
                    enriched[idx] = result
                except Exception as e:
                    orig_i = futures[future]
                    logger.error(f"Unexpected enrichment error at index {orig_i}: {e}")
                    enriched[orig_i] = chunks[orig_i]

        # Guard: if any slot is still None (shouldn't happen), fall back to original.
        return [e if e is not None else chunks[i] for i, e in enumerate(enriched)]
    

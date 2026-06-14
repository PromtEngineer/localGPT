import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from rag_system.ingestion.chunking import create_contextual_window

# Chunks shorter than this are headers/footers unlikely to benefit from context.
_MIN_ENRICH_CHARS = 100

logger = logging.getLogger(__name__)

# Contextual-retrieval prompts, following Anthropic's published pattern:
# situate the chunk within the OVERALL DOCUMENT (not just its neighbours),
# explicitly for the purpose of improving search retrieval of the chunk.
SYSTEM_PROMPT = "You situate document chunks within their source document to improve search retrieval."

DOCUMENT_CONTEXT_PROMPT_TEMPLATE = """Here is an excerpt from the beginning of the document this chunk belongs to:
<document_excerpt>
{document_excerpt}
</document_excerpt>"""

LOCAL_CONTEXT_PROMPT_TEMPLATE = """Here is the text immediately surrounding the chunk:
<local_context>
{local_context_text}
</local_context>"""

CHUNK_PROMPT_TEMPLATE = """Here is the chunk we want to situate:
<chunk>
{chunk_content}
</chunk>

Give a short succinct context (1-3 sentences) to situate this chunk within the overall document, for the purposes of improving search retrieval of the chunk. Name the document's subject and any entities, identifiers, project names, or section topics that connect this chunk to it. Answer only with the succinct context and nothing else."""


class ContextualEnricher:
    """
    Enriches chunks with a prepended summary of their surrounding context.
    Works with any LLM backend (Ollama, Anthropic, OpenAI, Groq) as long as
    the client implements generate_completion(model, prompt, **kwargs) -> {"response": str}.
    """

    def __init__(
        self,
        llm_client,
        llm_model: str,
        batch_size: int = 10,
        timeout: int = 90,
        max_workers: int = 4,
    ):
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.batch_size = batch_size
        self.timeout = timeout
        self.max_workers = max_workers
        provider = type(llm_client).__name__
        logger.info(
            f"Initialized ContextualEnricher with {provider} model '{self.llm_model}' (batch_size={batch_size}, timeout={timeout}s, max_workers={max_workers})."
        )

    def _generate_summary(
        self, local_context_text: str, chunk_text: str, document_excerpt: str = ""
    ) -> str:
        """Generates a contextual summary using a structured, multi-part prompt."""
        # Combine the templates to form the final content for the HumanMessage equivalent
        parts = []
        if document_excerpt:
            parts.append(
                DOCUMENT_CONTEXT_PROMPT_TEMPLATE.format(
                    document_excerpt=document_excerpt
                )
            )
        parts.append(
            LOCAL_CONTEXT_PROMPT_TEMPLATE.format(local_context_text=local_context_text)
        )
        parts.append(CHUNK_PROMPT_TEMPLATE.format(chunk_content=chunk_text))
        human_prompt_content = "\n\n".join(parts)

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
            summary_raw = response.get("response", "").strip()

            # --- Sanitize the summary to remove chain-of-thought markers ---
            # Many Qwen models wrap reasoning in <think>...</think> or similar tags.
            cleaned = re.sub(
                r"<think[^>]*>.*?</think>",
                "",
                summary_raw,
                flags=re.IGNORECASE | re.DOTALL,
            )
            # Remove any assistant role tags that may appear
            cleaned = re.sub(
                r"<assistant[^>]*>|</assistant>", "", cleaned, flags=re.IGNORECASE
            )
            # If the model used an explicit "Answer:" delimiter keep only the part after it
            if "Answer:" in cleaned:
                cleaned = cleaned.split("Answer:", 1)[1]

            # Take the first non-empty line to avoid leftover blank lines
            summary = next(
                (ln.strip() for ln in cleaned.splitlines() if ln.strip()), ""
            )

            # Fallback to raw if cleaning removed everything
            if not summary:
                summary = summary_raw

            if not summary or len(summary) < 5:
                logger.warning(
                    "Generated context summary is too short or empty. Skipping enrichment for this chunk."
                )
                return ""

            return summary

        except Exception as e:
            logger.error(
                f"LLM invocation failed during contextualization: {e}", exc_info=True
            )
            return ""  # Gracefully fail by returning no summary

    @staticmethod
    def _document_excerpt(chunks: List[Dict[str, Any]], cap: int = 1500) -> str:
        """Head of the document, used to ground every chunk's context in what
        the document actually is (title, subject, project identifiers)."""
        head = " ".join(c.get("text", "") for c in chunks[:3])
        return head[:cap]

    def _enrich_one(
        self,
        i: int,
        chunks: List[Dict[str, Any]],
        window_size: int,
        document_excerpt: str = "",
    ) -> tuple[int, Dict[str, Any]]:
        """Enrich a single chunk. Returns (index, result) so callers can reassemble order."""
        chunk = chunks[i]
        original_text = chunk.get("text", "")

        # Skip trivial chunks — headers, footers, page numbers, etc.
        if len(original_text) < _MIN_ENRICH_CHARS:
            return i, chunk

        try:
            local_context_text = create_contextual_window(
                chunks, chunk_index=i, window_size=window_size
            )
            summary = self._generate_summary(
                local_context_text, original_text, document_excerpt
            )

            new_chunk = chunk.copy()
            if not isinstance(new_chunk.get("metadata"), dict):
                new_chunk["metadata"] = {}
            new_chunk["metadata"]["original_text"] = original_text
            new_chunk["metadata"]["contextual_summary"] = summary or "N/A"
            if summary:
                new_chunk["text"] = f"Context: {summary}\n\n---\n\n{original_text}"
            return i, new_chunk

        except Exception as e:
            logger.error(f"Error enriching chunk {i}: {e}")
            return i, chunk

    def enrich_chunks(
        self, chunks: List[Dict[str, Any]], window_size: int = 1
    ) -> List[Dict[str, Any]]:
        if not chunks:
            return []

        skippable = sum(1 for c in chunks if len(c.get("text", "")) < _MIN_ENRICH_CHARS)
        to_enrich = len(chunks) - skippable
        logger.info(
            f"Enriching {to_enrich}/{len(chunks)} chunks (skipping {skippable} trivial) "
            f"window_size={window_size} workers={self.max_workers}"
        )

        enriched: List[Dict[str, Any]] = [None] * len(chunks)  # type: ignore[list-item]
        document_excerpt = self._document_excerpt(chunks)

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(
                    self._enrich_one, i, chunks, window_size, document_excerpt
                ): i
                for i in range(len(chunks))
            }
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

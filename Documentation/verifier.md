# Answer verifier

`rag_system/agent/verifier.py` optionally asks the configured generation provider whether a completed document-RAG answer is grounded in the retrieved context.

The agent runs it after answer synthesis when the request's `verify` flag is true, or when the active pipeline configuration enables verification and the request does not override it. It receives the query, a context string assembled from retrieved rows, and the answer. Context is capped at 4,000 characters before the prompt is sent.

The expected JSON fields are:

```json
{
  "verdict": "SUPPORTED",
  "is_grounded": true,
  "reasoning": "short explanation",
  "confidence_score": 100
}
```

`verdict` may be `SUPPORTED`, `NOT_SUPPORTED`, or `NEEDS_CLARIFICATION`. Invalid JSON fails closed to `NOT_SUPPORTED`, `is_grounded=false`, and confidence `0`. A low-confidence/non-grounded result can add a warning to the answer; a parser failure does not replace the answer.

This is a model-based secondary judgment, not a formal proof and not a substitute for citation-level entailment. The verifier currently uses the same generation model selected for the agent rather than an independently administered model.

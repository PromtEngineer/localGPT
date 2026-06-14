import json

from rag_system.utils.ollama_client import OllamaClient


class VerificationResult:
    def __init__(
        self, is_grounded: bool, reasoning: str, verdict: str, confidence_score: int
    ):
        self.is_grounded = is_grounded
        self.reasoning = reasoning
        self.verdict = verdict
        self.confidence_score = confidence_score


class Verifier:
    """
    Verifies if a generated answer is grounded in the provided context using Ollama.
    """

    def __init__(self, llm_client: OllamaClient, llm_model: str):
        self.llm_client = llm_client
        self.llm_model = llm_model
        print(f"Initialized Verifier with Ollama model '{self.llm_model}'.")

    # Synchronous verify() method removed – async version is used everywhere.

    # --- Async wrapper ------------------------------------------------
    async def verify_async(
        self,
        query: str,
        context: str,
        answer: str,
        model_override: str | None = None,
    ) -> VerificationResult:
        """Async variant that calls the Ollama client asynchronously."""
        prompt = f"""
        You are an automated fact-checker. Determine whether the ANSWER is fully supported by the CONTEXT and output a single line of JSON.

        # EXAMPLES

        <QUERY>
        What color is the sky?
        </QUERY>
        <CONTEXT>
        During the day, the sky appears blue due to Rayleigh scattering.
        </CONTEXT>
        <ANSWER>
        The sky is blue during the day.
        </ANSWER>
        <OUTPUT>
        {{"verdict": "SUPPORTED", "is_grounded": true, "reasoning": "The context explicitly supports that the sky is blue during the day.", "confidence_score": 100}}
        </OUTPUT>

        <QUERY>
        Where are apples and oranges grown?
        </QUERY>
        <CONTEXT>
        Apples are grown in orchards.
        </CONTEXT>
        <ANSWER>
        Apples are grown in orchards and oranges are grown in groves.
        </ANSWER>
        <OUTPUT>
        {{"verdict": "NOT_SUPPORTED", "is_grounded": false, "reasoning": "The context mentions orchards, but not oranges or groves.", "confidence_score": 80}}
        </OUTPUT>

        <QUERY>
        How long is the process?
        </QUERY>
        <CONTEXT>
        The first step takes 3 days. The second step takes 5 days.
        </CONTEXT>
        <ANSWER>
        The process takes 3 days.
        </ANSWER>
        <OUTPUT>
        {{"verdict": "NEEDS_CLARIFICATION", "is_grounded": false, "reasoning": "The answer omits the 5 days required for the second step.", "confidence_score": 70}}
        </OUTPUT>

        # TASK

        <QUERY>
        "{query}"
        </QUERY>
        <CONTEXT>
        """
        prompt += context[:4000]  # Clamp to avoid huge prompts
        prompt += """
        </CONTEXT>
        <ANSWER>
        """
        prompt += answer
        prompt += """
        </ANSWER>
        <OUTPUT>
        """
        resp = await self.llm_client.generate_completion_async(
            model_override or self.llm_model, prompt, format="json"
        )
        try:
            data = json.loads(resp.get("response", "{}"))
            return VerificationResult(
                is_grounded=data.get("is_grounded", False),
                reasoning=data.get("reasoning", "async parse error"),
                verdict=data.get("verdict", "NOT_SUPPORTED"),
                confidence_score=data.get("confidence_score", 0),
            )
        except (json.JSONDecodeError, AttributeError):
            return VerificationResult(False, "Failed async parse", "NOT_SUPPORTED", 0)

    # --- Reflection scoring (0-2 scale, sync) -------------------------
    # Used by the self-reflection loop (rag_system.agent.reflection): a 0-2
    # score is coarse enough to be stable across local models yet enough to
    # gate a bounded retry. Any LLM/parse failure scores 0 (fail-safe: the
    # loop will try to improve, bounded by max_loops).

    def _reflection_score(self, prompt: str, model_override: str | None) -> int:
        try:
            # Greedy decoding (temperature=0) keeps the 0-2 score stable across
            # identical calls — the reflection loop was thrashing on run-to-run
            # variance. Thinking off also makes each score fast (no CoT tokens).
            resp = self.llm_client.generate_completion(
                model_override or self.llm_model,
                prompt,
                format="json",
                temperature=0.0,
                enable_thinking=False,
            )
            raw = json.loads(resp.get("response", "{}")).get("score", 0)
            return max(0, min(2, int(raw)))
        except (json.JSONDecodeError, AttributeError, TypeError, ValueError):
            return 0

    def score_context_relevance(
        self, query: str, context: str, model_override: str | None = None
    ) -> int:
        """Rate the retrieved context for the query: 0=irrelevant, 1=partial, 2=high."""
        prompt = (
            "Rate how relevant the CONTEXT is for answering the QUERY on a 0-2 "
            "scale:\n0 = not relevant, 1 = partially relevant, 2 = highly relevant.\n"
            'Reply with JSON only: {"score": <0|1|2>}.\n\n'
            f"QUERY:\n{query}\n\nCONTEXT:\n{context[:4000]}"
        )
        return self._reflection_score(prompt, model_override)

    def score_response_groundedness(
        self, query: str, context: str, answer: str, model_override: str | None = None
    ) -> int:
        """Rate the answer against its context: 0=unsupported, 1=partial, 2=grounded."""
        prompt = (
            "Rate how well the ANSWER is grounded in the CONTEXT on a 0-2 scale:\n"
            "0 = unsupported or hallucinated, 1 = partially supported, "
            "2 = fully supported.\n"
            'Reply with JSON only: {"score": <0|1|2>}.\n\n'
            f"QUERY:\n{query}\n\nCONTEXT:\n{context[:4000]}\n\nANSWER:\n{answer[:2000]}"
        )
        return self._reflection_score(prompt, model_override)

import asyncio
import json
import os
import re
from threading import Lock
from typing import List, Optional

from rag_system.utils.ollama_client import OllamaClient

# Serialises the first (heavy) load of a local verifier model, the same way the
# reranker and Provence loads are serialised in retrieval_pipeline.py.
_local_verifier_lock: Lock = Lock()


class VerificationResult:
    def __init__(self, is_grounded: bool, reasoning: str, verdict: str, confidence_score: int):
        self.is_grounded = is_grounded
        self.reasoning = reasoning
        self.verdict = verdict
        self.confidence_score = confidence_score


class VerifierModelUnavailable(RuntimeError):
    """Raised when `VERIFIER_MODEL` names something that cannot be loaded."""


# Models checked for suitability on 2026-08-09 (roadmap 2.4). Reported to the
# user verbatim when a configured verifier fails to load, so the failure names
# what was actually verified instead of hand-waving.
VERIFIER_AVAILABILITY_NOTES = """\
Checked on 2026-08-09 (HuggingFace Hub API):
  * ThinknCheck (arXiv 2604.01652, UPenn)     — NO PUBLIC WEIGHTS. The paper is
    real (1B, 78.1 BAcc on LLMAggreFact) but a Hub search for "thinkncheck"
    returns zero models and the paper links no release. Cannot be wired.
  * ibm-granite/granite-guardian-3.3-8b       — exists, Apache-2.0, but 8B /
    ~16 GB. Far over the "small local verifier" budget this seam is for.
  * ibm-granite/granite-guardian-hap-38m      — exists, 38M, Apache-2.0, but it
    is a hate/abuse/profanity RoBERTa classifier. Wrong task: it does not score
    answer-vs-evidence entailment at all.
Verified working (<2 GB, no trust_remote_code):
  * lytang/MiniCheck-DeBERTa-v3-Large  (MIT, 1.74 GB)  <- smoke-tested default
  * lytang/MiniCheck-RoBERTa-Large     (MIT)
  * MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli (MIT, 369 MB, generic NLI)
Needs trust_remote_code (opt in with VERIFIER_TRUST_REMOTE_CODE=1):
  * vectara/hallucination_evaluation_model (HHEM-2.1-open, Apache-2.0, 438 MB)
"""

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


class LocalNLIVerifier:
    """Answer-vs-evidence scoring with a local sequence-classification model.

    The seam roadmap item 2.4 asks for. Any HuggingFace model that scores a
    (premise, hypothesis) pair works: MiniCheck's grounded-claim checkers, a
    generic MNLI cross-encoder, or Vectara's HHEM. The answer is split into
    sentences, each is scored against the retrieved evidence as the premise, and
    the **minimum** is taken — one unsupported sentence makes the answer
    ungrounded, which is the semantics the binary judge already uses.

    Note on ``[Confidence: N%]``: this number is a model output, not a
    calibrated probability of correctness. It is UX, and `Documentation/
    verifier.md` says so. Swapping the LLM prompt for an NLI model changes
    where the number comes from; it does not make it calibrated.
    """

    def __init__(self, model_name: str, threshold: float = 0.5,
                 trust_remote_code: Optional[bool] = None):
        self.model_name = model_name
        self.threshold = threshold
        if trust_remote_code is None:
            trust_remote_code = os.getenv("VERIFIER_TRUST_REMOTE_CODE", "") == "1"
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as e:  # pragma: no cover - transformers is a hard dep
            raise VerifierModelUnavailable(
                f"transformers/torch are required to load VERIFIER_MODEL: {e}")

        self._torch = torch
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=trust_remote_code)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, trust_remote_code=trust_remote_code)
        except Exception as e:
            raise VerifierModelUnavailable(
                f"Could not load VERIFIER_MODEL='{model_name}': {e}\n\n"
                f"{VERIFIER_AVAILABILITY_NOTES}"
                "Set VERIFIER_MODEL to one of the verified names above, unset it to "
                "use the default LLM-prompt verifier, or add "
                "VERIFIER_TRUST_REMOTE_CODE=1 if the model ships custom code."
            ) from e

        self.model.eval()
        self.device = ("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self._supported_index = self._resolve_supported_index()
        print(f"✅ Local verifier '{model_name}' loaded on {self.device} "
              f"(supported label index {self._supported_index}).")

    def _resolve_supported_index(self) -> int:
        """Which logit means "the evidence supports this"."""
        id2label = getattr(self.model.config, "id2label", None) or {}
        for idx, label in id2label.items():
            if str(label).lower() in {"entailment", "consistent", "supported", "1", "true"}:
                return int(idx)
        # Binary checkers (MiniCheck) label their classes "0"/"1": 1 = supported.
        return int(self.model.config.num_labels) - 1

    def score(self, evidence: str, answer: str) -> float:
        sentences = [s.strip() for s in _SENTENCE_SPLIT.split(answer or "") if s.strip()]
        if not sentences:
            return 0.0
        torch = self._torch
        scores: List[float] = []
        with torch.no_grad():
            for sentence in sentences:
                inputs = self.tokenizer(evidence, sentence, return_tensors="pt",
                                        truncation=True, max_length=self.tokenizer.model_max_length
                                        if self.tokenizer.model_max_length < 100000 else 2048)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                logits = self.model(**inputs).logits[0]
                if logits.numel() == 1:
                    probability = torch.sigmoid(logits)[0]
                else:
                    probability = torch.softmax(logits, dim=-1)[self._supported_index]
                scores.append(float(probability))
        # Weakest link: one unsupported sentence makes the answer ungrounded.
        return min(scores)

    def verify(self, query: str, context: str, answer: str) -> VerificationResult:
        probability = self.score(context, answer)
        grounded = probability >= self.threshold
        return VerificationResult(
            is_grounded=grounded,
            reasoning=(f"{self.model_name}: weakest answer sentence scored "
                       f"{probability:.3f} against the retrieved evidence "
                       f"(threshold {self.threshold})."),
            verdict="SUPPORTED" if grounded else "NOT_SUPPORTED",
            confidence_score=int(round(probability * 100)),
        )


class Verifier:
    """
    Verifies if a generated answer is grounded in the provided context.

    Two backends, same interface:

    * **default** — an LLM prompt on the utility model (below). This is what
      ships; nothing changes unless you opt in.
    * **local NLI/verifier model** — set ``VERIFIER_MODEL`` (or
      ``verification.model`` in the pipeline config) to a HuggingFace model name.
      Loaded lazily on first use through ``LocalNLIVerifier``, so naming a model
      costs nothing until a query is actually verified. A model that cannot be
      loaded raises with the list of names that were checked, rather than
      silently degrading to the LLM prompt — a verifier that quietly is not the
      verifier you configured is worse than an error.

    Roadmap item 2.4. Availability findings are in ``VERIFIER_AVAILABILITY_NOTES``.
    """

    def __init__(self, llm_client: OllamaClient, llm_model: str,
                 model_name: Optional[str] = None, threshold: float = 0.5):
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.local_model_name = model_name or os.getenv("VERIFIER_MODEL") or None
        self.local_threshold = threshold
        self._local: Optional[LocalNLIVerifier] = None
        if self.local_model_name:
            print(f"Initialized Verifier with local model '{self.local_model_name}' "
                  f"(loaded on first use); LLM fallback model '{self.llm_model}'.")
        else:
            print(f"Initialized Verifier with Ollama model '{self.llm_model}'.")

    def _get_local(self) -> Optional[LocalNLIVerifier]:
        if not self.local_model_name:
            return None
        if self._local is None:
            with _local_verifier_lock:
                if self._local is None:
                    self._local = LocalNLIVerifier(self.local_model_name,
                                                   self.local_threshold)
        return self._local

    # Synchronous verify() method removed – async version is used everywhere.

    # --- Async wrapper ------------------------------------------------
    async def verify_async(self, query: str, context: str, answer: str) -> VerificationResult:
        """Async variant that calls the Ollama client asynchronously."""
        local = self._get_local()
        if local is not None:
            # transformers is blocking; keep the event loop responsive.
            return await asyncio.to_thread(local.verify, query, context, answer)

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
        resp = await self.llm_client.generate_completion_async(self.llm_model, prompt, format="json")
        try:
            data = json.loads(resp.get("response", "{}"))
            return VerificationResult(
                is_grounded=data.get("is_grounded", False),
                reasoning=data.get("reasoning", "async parse error"),
                verdict=data.get("verdict", "NOT_SUPPORTED"),
                confidence_score=data.get('confidence_score', 0)
            )
        except (json.JSONDecodeError, AttributeError):
            return VerificationResult(False, "Failed async parse", "NOT_SUPPORTED", 0)

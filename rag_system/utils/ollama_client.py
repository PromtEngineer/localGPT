import requests
import json
import os
from typing import List, Dict, Any, Optional
import base64
import contextlib
import contextvars
import threading
from io import BytesIO
from PIL import Image
import httpx, asyncio

# ---------------------------------------------------------------------------
# Context-window sizing (fixes silent front-truncation)
# ---------------------------------------------------------------------------
# Ollama's server-side window (OLLAMA_CONTEXT_LENGTH, default 4096) is split
# across parallel slots, and a request that exceeds its slot is FRONT-truncated
# with no error — measured on this machine as an effective 8194-token ceiling
# that dropped the top-ranked context chunks from synthesis prompts. The fix is
# to size the window per request: `options.num_ctx` overrides the server value.
#
# Sizing is bucketed (8k/16k/32k) rather than exact, and per-model MONOTONIC:
# once a model has been granted a window it never shrinks back. Changing
# num_ctx between calls forces Ollama to reallocate the KV cache (measured:
# the smoke suite went 363s → 924s when the agent's small triage calls and
# large synthesis calls alternated buckets; a fixed window ran it in 259s), so
# the window may only ratchet upward — at most two transitions per model per
# process. The estimate deliberately overshoots (~3 chars/token vs the ~4
# typical for English) because undershooting reintroduces the truncation this
# exists to prevent.
#
# Env knobs: OLLAMA_NUM_CTX pins an exact value (skips estimation);
# OLLAMA_NUM_CTX_MAX caps the bucket (default 32768 — raise it on machines
# with memory to spare, at the cost of a larger KV cache).

_NUM_CTX_BUCKETS = (8192, 16384, 32768)
_OUTPUT_HEADROOM_TOKENS = 2048
_MODEL_CTX_HIGH_WATER: Dict[str, int] = {}
_MODEL_CTX_LOCK = threading.Lock()


def _num_ctx_for(char_count: int) -> int:
    pinned = os.getenv("OLLAMA_NUM_CTX")
    if pinned:
        try:
            return max(2048, int(pinned))
        except ValueError:
            pass
    try:
        max_ctx = int(os.getenv("OLLAMA_NUM_CTX_MAX", "32768"))
    except ValueError:
        max_ctx = 32768
    estimated = char_count // 3 + _OUTPUT_HEADROOM_TOKENS
    for bucket in _NUM_CTX_BUCKETS:
        if estimated <= bucket <= max_ctx:
            return bucket
    return max_ctx


def _apply_num_ctx(payload: Dict[str, Any]) -> None:
    """Set options.num_ctx from the prompt size, preserving caller options.

    Per-model monotonic: the requested window never shrinks below the largest
    one this process has already used for the model (see module comment).
    """
    options = payload.setdefault("options", {})
    if "num_ctx" in options:
        return
    needed = _num_ctx_for(len(payload.get("prompt") or ""))
    model = str(payload.get("model") or "")
    with _MODEL_CTX_LOCK:
        granted = max(needed, _MODEL_CTX_HIGH_WATER.get(model, 0))
        _MODEL_CTX_HIGH_WATER[model] = granted
    options["num_ctx"] = granted


def _warn_if_truncated(payload: Dict[str, Any], final_response: Dict[str, Any]) -> None:
    """Detect a filled context window — the signature of front-truncation.

    Best-effort diagnostics only; never raises.
    """
    try:
        num_ctx = int(payload.get("options", {}).get("num_ctx") or 0)
        used = int(final_response.get("prompt_eval_count") or 0)
        prompt_chars = len(payload.get("prompt") or "")
        if num_ctx and used >= num_ctx - 16:
            print(
                f"⚠️ Ollama likely FRONT-TRUNCATED this prompt: "
                f"prompt_eval_count={used} filled num_ctx={num_ctx} "
                f"(model={payload.get('model')}). Raise OLLAMA_NUM_CTX_MAX or "
                f"shrink the prompt — the beginning of the prompt was dropped."
            )
        elif used and prompt_chars and used < prompt_chars // 6:
            # Slot-proof check: the server divides num_ctx across its parallel
            # slots, so a request can be truncated far BELOW the window we
            # asked for and the filled-window check above stays silent
            # (measured: 94k-token prompt, num_ctx 32768, served 16386).
            # English text runs ~3.5–4.5 chars/token, so evaluating fewer than
            # chars/6 tokens means part of the prompt never reached the model.
            print(
                f"⚠️ Ollama FRONT-TRUNCATED this prompt below the requested window: "
                f"prompt_eval_count={used} for a {prompt_chars}-char prompt "
                f"(num_ctx={num_ctx}, model={payload.get('model')}). The served "
                f"per-request window is num_ctx divided by the server's parallel "
                f"slots — set OLLAMA_NUM_PARALLEL=1 or shrink the prompt."
            )
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Per-query token tracking (roadmap item 4.5)
# ---------------------------------------------------------------------------
# Ollama returns ``prompt_eval_count`` (input tokens) and ``eval_count`` (output
# tokens) on the final object of every /api/generate response, streaming or not.
# They are free — we already parse that object — so the only work is routing
# them somewhere useful.
#
# The routing is a ``ContextVar`` rather than a client attribute because the one
# ``OllamaClient`` instance is shared by the agent, the retrieval pipeline, the
# verifier and the decomposer. A context variable attributes each call to
# whichever request is on the stack, and `asyncio.to_thread` / `await` propagate
# it for free. The one place that does NOT propagate it is a raw
# ``ThreadPoolExecutor.submit`` — the agent's parallel sub-query fan-out — which
# copies the context explicitly (see ``rag_system/agent/loop.py``).
#
# Nothing here can fail a request: every record path is best-effort.

_CURRENT_TRACKER: contextvars.ContextVar[Optional["TokenUsageTracker"]] = contextvars.ContextVar(
    "localgpt_token_tracker", default=None
)
_CURRENT_STAGE: contextvars.ContextVar[str] = contextvars.ContextVar(
    "localgpt_token_stage", default="other"
)


class TokenUsageTracker:
    """Aggregates LLM token counts for one user query, bucketed by stage.

    Stages are the agent's own pipeline phases (``triage``, ``decomposition``,
    ``synthesis``, ``verification``, …). A bucket only appears once a call has
    been attributed to it, so an absent key means "that stage made no LLM call",
    not "zero tokens".
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_stage: Dict[str, Dict[str, int]] = {}

    def record(self, stage: str, prompt_tokens: int, output_tokens: int) -> None:
        with self._lock:
            bucket = self._by_stage.setdefault(
                stage, {"prompt_tokens": 0, "output_tokens": 0, "calls": 0}
            )
            bucket["prompt_tokens"] += int(prompt_tokens or 0)
            bucket["output_tokens"] += int(output_tokens or 0)
            bucket["calls"] += 1

    def as_dict(self) -> Dict[str, Any]:
        with self._lock:
            by_stage = {k: dict(v) for k, v in self._by_stage.items()}
        total = {
            "prompt_tokens": sum(b["prompt_tokens"] for b in by_stage.values()),
            "output_tokens": sum(b["output_tokens"] for b in by_stage.values()),
            "calls": sum(b["calls"] for b in by_stage.values()),
        }
        total["total_tokens"] = total["prompt_tokens"] + total["output_tokens"]
        return {"by_stage": by_stage, "total": total}


@contextlib.contextmanager
def track_token_usage(tracker: Optional[TokenUsageTracker]):
    """Bind *tracker* as the sink for LLM token counts inside this block."""
    token = _CURRENT_TRACKER.set(tracker)
    try:
        yield tracker
    finally:
        _CURRENT_TRACKER.reset(token)


@contextlib.contextmanager
def token_stage(name: str):
    """Attribute every LLM call made inside this block to stage *name*."""
    token = _CURRENT_STAGE.set(name)
    try:
        yield
    finally:
        _CURRENT_STAGE.reset(token)


def record_llm_usage(payload: Dict[str, Any]) -> None:
    """Record one completed LLM call from a raw Ollama-shaped response object.

    A no-op when no tracker is bound, or when the backend reports neither count
    (watsonx). Never raises.
    """
    tracker = _CURRENT_TRACKER.get()
    if tracker is None or not isinstance(payload, dict):
        return
    if "prompt_eval_count" not in payload and "eval_count" not in payload:
        return
    try:
        tracker.record(
            _CURRENT_STAGE.get(),
            payload.get("prompt_eval_count") or 0,
            payload.get("eval_count") or 0,
        )
    except Exception:
        pass


class OllamaClient:
    """
    An enhanced client for Ollama that now handles image data for VLM models.
    """
    def __init__(self, host: str = "http://localhost:11434"):
        self.host = host
        self.api_url = f"{host}/api"
        # (Connection check remains the same)

    def _image_to_base64(self, image: Image.Image) -> str:
        """Converts a Pillow Image to a base64 string."""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def generate_completion(
        self,
        model: str,
        prompt: str,
        *,
        format: str = "",
        images: List[Image.Image] | None = None,
        enable_thinking: bool | None = None,
        options: Optional[Dict[str, Any]] = None,
        timeout: int = 60,
    ) -> Dict[str, Any]:
        """
        Generates a completion, now with optional support for images.

        Args:
            model: The name of the generation model (e.g., 'llava', 'qwen-vl').
            prompt: The text prompt for the model.
            format: The format for the response, e.g., "json".
            images: A list of Pillow Image objects to send to the VLM.
            enable_thinking: Optional flag to disable chain-of-thought for Qwen models.
            options: Ollama options dict (e.g. {"temperature": 0}); an explicit
                num_ctx here wins over the automatic sizing.
            timeout: Seconds before the HTTP call gives up (same default the
                async variant uses) — a hung Ollama must not stall index
                builds, which call this once per chunk.
        """
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False
            }
            if format:
                payload["format"] = format
            if options:
                payload["options"] = dict(options)
            
            if images:
                payload["images"] = [self._image_to_base64(img) for img in images]

            # Thinking models put JSON into the `thinking` field and leave
            # `response` empty when format=json, so default thinking off there.
            # `think` is the top-level knob /api/generate actually honors
            # (chat_template_kwargs is silently ignored by the generate API).
            if enable_thinking is None and format == "json":
                enable_thinking = False
            if enable_thinking is not None:
                payload["think"] = enable_thinking

            _apply_num_ctx(payload)

            response = requests.post(
                f"{self.api_url}/generate",
                json=payload,
                timeout=timeout
            )
            response.raise_for_status()
            response_lines = response.text.strip().split('\n')
            final_response = json.loads(response_lines[-1])
            # roadmap 4.5: `prompt_eval_count` / `eval_count` ride along on this
            # object already; hand them to the per-query tracker if one is bound.
            record_llm_usage(final_response)
            _warn_if_truncated(payload, final_response)
            return final_response

        except requests.exceptions.RequestException as e:
            print(f"Error generating completion: {e}")
            return {}

    # -------------------------------------------------------------
    # Async variant – uses httpx so the caller can await multiple
    # LLM calls concurrently (triage, verification, etc.).
    # -------------------------------------------------------------
    async def generate_completion_async(
        self,
        model: str,
        prompt: str,
        *,
        format: str = "",
        images: List[Image.Image] | None = None,
        enable_thinking: bool | None = None,
        timeout: int = 60,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Asynchronous version of generate_completion using httpx.

        *options* is an Ollama options dict (e.g. {"temperature": 0}); an
        explicit num_ctx here wins over the automatic sizing.
        """

        payload = {"model": model, "prompt": prompt, "stream": False}
        if format:
            payload["format"] = format
        if options:
            # Caller-supplied sampling options (e.g. temperature). Merged
            # before _apply_num_ctx so an explicit num_ctx here wins.
            payload["options"] = dict(options)
        if images:
            payload["images"] = [self._image_to_base64(img) for img in images]

        if enable_thinking is None and format == "json":
            enable_thinking = False
        if enable_thinking is not None:
            payload["think"] = enable_thinking

        _apply_num_ctx(payload)

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(f"{self.api_url}/generate", json=payload)
                resp.raise_for_status()
                final_response = json.loads(resp.text.strip().split("\n")[-1])
                record_llm_usage(final_response)
                _warn_if_truncated(payload, final_response)
                return final_response
        except asyncio.CancelledError:
            # Cancellation is control flow, not an LLM failure — never swallow
            # it into the {} fail-open below (that fail-open is documented
            # behaviour for httpx errors only).
            raise
        except httpx.HTTPError as e:
            print(f"Async Ollama completion error: {e}")
            return {}

    # -------------------------------------------------------------
    # Streaming variant – yields token chunks in real time
    # -------------------------------------------------------------
    def stream_completion(
        self,
        model: str,
        prompt: str,
        *,
        images: List[Image.Image] | None = None,
        enable_thinking: bool | None = None,
        stats: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        """Generator that yields partial *response* strings as they arrive.

        Example:

            for tok in client.stream_completion("qwen2", "Hello"):
                print(tok, end="", flush=True)

        A streaming generator cannot return a value, so pass a dict as *stats*
        to receive the final Ollama object (``prompt_eval_count``,
        ``eval_count``, timings) once the stream completes. Token counts are
        also handed to the per-query tracker automatically (roadmap 4.5), which
        is what the agent uses — *stats* exists for direct callers.
        """
        payload: Dict[str, Any] = {"model": model, "prompt": prompt, "stream": True}
        if images:
            payload["images"] = [self._image_to_base64(img) for img in images]
        if enable_thinking is not None:
            payload["think"] = enable_thinking
        if options:
            # Caller-supplied sampling options (e.g. temperature). Merged
            # before _apply_num_ctx so an explicit num_ctx here wins.
            payload["options"] = dict(options)

        _apply_num_ctx(payload)

        with requests.post(f"{self.api_url}/generate", json=payload, stream=True) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    # Keep-alive newline
                    continue
                try:
                    data = json.loads(raw_line.decode())
                except json.JSONDecodeError:
                    continue
                # The Ollama streaming API sends objects like {"response":"Hi","done":false}
                chunk = data.get("response", "")
                if chunk:
                    yield chunk
                if data.get("done"):
                    # The final object carries the token counts for the whole
                    # stream (roadmap 4.5).
                    if stats is not None:
                        stats.update(data)
                    record_llm_usage(data)
                    _warn_if_truncated(payload, data)
                    break

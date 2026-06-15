"""Cloud LLM clients for contextual enrichment during indexing.

Each client implements the same duck-type interface as OllamaClient:
    generate_completion(model, prompt, *, enable_thinking=None, timeout=60, **kwargs)
    -> {"response": "<text>"}

API keys are resolved: explicit arg > environment variable. They are never logged or stored.
"""

import os
from typing import Any, Dict

import requests


class AnthropicEnricher:
    """Calls the Anthropic Messages API."""

    _BASE_URL = "https://api.anthropic.com/v1/messages"
    _API_VERSION = "2023-06-01"

    def __init__(self, api_key: str):
        self._api_key = api_key
        if not self._api_key:
            raise ValueError(
                "Anthropic API key is required. Set ANTHROPIC_API_KEY or pass it in the UI."
            )

    def generate_completion(
        self,
        model: str,
        prompt: str,
        *,
        enable_thinking=None,
        timeout: int = 60,
        **_,
    ) -> Dict[str, Any]:
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": self._API_VERSION,
            "content-type": "application/json",
        }
        payload = {
            "model": model,
            "max_tokens": 300,
            "messages": [{"role": "user", "content": prompt}],
        }
        try:
            resp = requests.post(
                self._BASE_URL, json=payload, headers=headers, timeout=timeout
            )
            resp.raise_for_status()
            data = resp.json()
            text = (data.get("content") or [{}])[0].get("text", "")
            return {"response": text}
        except requests.exceptions.RequestException as e:
            print(f"Anthropic API error: {e}")
            return {}


class OpenAICompatibleEnricher:
    """Calls OpenAI or any OpenAI-compatible endpoint (Groq, Together, etc.)."""

    _OPENAI_BASE = "https://api.openai.com/v1"
    _GROQ_BASE = "https://api.groq.com/openai/v1"

    def __init__(self, api_key: str, base_url: str | None = None):
        self._api_key = api_key
        self._base_url = (base_url or self._OPENAI_BASE).rstrip("/")
        if not self._api_key:
            raise ValueError(
                "API key is required. Set OPENAI_API_KEY / GROQ_API_KEY or pass it in the UI."
            )

    def generate_completion(
        self,
        model: str,
        prompt: str,
        *,
        enable_thinking=None,
        timeout: int = 60,
        **_,
    ) -> Dict[str, Any]:
        url = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "content-type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 300,
        }
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            return {"response": text}
        except (requests.exceptions.RequestException, KeyError, IndexError) as e:
            print(f"OpenAI-compatible API error ({self._base_url}): {e}")
            return {}


def create_enrichment_client(
    provider: str,
    api_key: str | None = None,
    ollama_client=None,
    policy=None,
    audit=None,
):
    """Return the right LLM client for the given enrichment provider.

    provider : "ollama" | "anthropic" | "openai" | "groq"
    api_key  : optional — falls back to env vars per provider
    ollama_client : returned unchanged for "ollama" provider; also used as the
        local fallback when the egress policy blocks cloud enrichment
    policy : optional data-egress policy (see rag_system.utils.data_policy)
    audit  : optional callback recording policy actions (counts only, no values)

    Every cloud provider is wrapped in a PolicyGuardedEnricher so document text
    is scanned for secrets/PII before any request leaves the machine.
    """
    provider = (provider or "ollama").lower()

    cloud: object | None = None
    if provider == "anthropic":
        key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        assert key is not None  # env fallback default ("") guarantees a str
        cloud = AnthropicEnricher(api_key=key)
    elif provider == "openai":
        key = api_key or os.getenv("OPENAI_API_KEY", "")
        assert key is not None  # env fallback default ("") guarantees a str
        cloud = OpenAICompatibleEnricher(api_key=key)
    elif provider == "groq":
        key = api_key or os.getenv("GROQ_API_KEY", "")
        assert key is not None  # env fallback default ("") guarantees a str
        cloud = OpenAICompatibleEnricher(
            api_key=key,
            base_url=OpenAICompatibleEnricher._GROQ_BASE,
        )

    if cloud is not None:
        from rag_system.utils.data_policy import PolicyGuardedEnricher

        return PolicyGuardedEnricher(
            cloud,
            local_fallback=ollama_client,
            policy=policy,
            audit=audit,
            provider=provider,
        )

    # "ollama" or anything unrecognised → use local Ollama
    return ollama_client

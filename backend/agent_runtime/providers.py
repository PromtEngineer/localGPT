from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import Any, Protocol

import requests


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class AssistantTurn:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class ModelCapability:
    id: str
    provider: str
    generation: bool
    embedding: bool
    tools: bool | None = None
    vision: bool | None = None
    context_length: int | None = None


class ChatProvider(Protocol):
    async def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float,
        max_tokens: int | None,
    ) -> AssistantTurn: ...

    async def embed(self, *, model: str, inputs: list[str]) -> list[list[float]]: ...


class OllamaProvider:
    def __init__(self, base_url: str | None = None, timeout: float = 300) -> None:
        self.base_url = (base_url or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")).rstrip("/")
        self.timeout = timeout

    async def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> AssistantTurn:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
            "think": False,
        }
        if tools:
            payload["tools"] = tools
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens

        def request() -> dict[str, Any]:
            response = requests.post(
                f"{self.base_url}/api/chat", json=payload, timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()

        data = await asyncio.to_thread(request)
        message = data.get("message") or {}
        calls: list[ToolCall] = []
        for position, raw_call in enumerate(message.get("tool_calls") or []):
            function = raw_call.get("function") or {}
            arguments = function.get("arguments") or {}
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            calls.append(
                ToolCall(
                    id=str(raw_call.get("id") or f"call-{position}"),
                    name=str(function.get("name") or ""),
                    arguments=dict(arguments),
                )
            )
        return AssistantTurn(
            content=str(message.get("content") or ""),
            tool_calls=calls,
            input_tokens=data.get("prompt_eval_count"),
            output_tokens=data.get("eval_count"),
        )

    async def embed(self, *, model: str, inputs: list[str]) -> list[list[float]]:
        def request() -> list[list[float]]:
            response = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": model, "input": inputs},
                timeout=self.timeout,
            )
            response.raise_for_status()
            return list(response.json().get("embeddings") or [])

        return await asyncio.to_thread(request)

    async def discover_models(self) -> list[ModelCapability]:
        def request() -> list[tuple[dict[str, Any], dict[str, Any]]]:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            rows = list(response.json().get("models") or [])[:100]
            output = []
            for item in rows:
                name = str(item.get("name") or item.get("model") or "")
                shown: dict[str, Any] = {}
                try:
                    detail = requests.post(
                        f"{self.base_url}/api/show",
                        json={"model": name},
                        timeout=10,
                    )
                    detail.raise_for_status()
                    shown = detail.json()
                except requests.RequestException:
                    pass
                output.append((item, shown))
            return output

        raw_models = await asyncio.to_thread(request)
        results: list[ModelCapability] = []
        for item, shown in raw_models:
            model_id = str(item.get("name") or item.get("model") or "")
            details = item.get("details") or {}
            family = str(details.get("family") or "").lower()
            families = {str(value).lower() for value in details.get("families") or []}
            capabilities = {str(value).lower() for value in shown.get("capabilities") or []}
            embedding = "embedding" in capabilities or any(
                marker in model_id.lower() or marker == family or marker in families
                for marker in ("embed", "embedding", "bert", "nomic-bert")
            )
            results.append(
                ModelCapability(
                    id=model_id,
                    provider="ollama",
                    generation="completion" in capabilities or not embedding,
                    embedding=embedding,
                    tools="tools" in capabilities if capabilities else None,
                    vision="vision" in capabilities if capabilities else None,
                    context_length=(shown.get("model_info") or {}).get(
                        f"{family}.context_length"
                    ),
                )
            )
        return results


class OpenAICompatibleProvider:
    """Thin adapter for llama.cpp, LM Studio, vLLM and compatible servers."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 300,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> AssistantTurn:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        def request() -> dict[str, Any]:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()

        data = await asyncio.to_thread(request)
        message = data["choices"][0]["message"]
        calls = []
        for raw_call in message.get("tool_calls") or []:
            function = raw_call.get("function") or {}
            arguments = function.get("arguments") or {}
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            calls.append(
                ToolCall(
                    id=str(raw_call.get("id") or "call"),
                    name=str(function.get("name") or ""),
                    arguments=dict(arguments),
                )
            )
        usage = data.get("usage") or {}
        return AssistantTurn(
            content=str(message.get("content") or ""),
            tool_calls=calls,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
        )

    async def discover_models(self) -> list[ModelCapability]:
        def request() -> list[dict[str, Any]]:
            response = requests.get(
                f"{self.base_url}/models", headers=self._headers(), timeout=10
            )
            response.raise_for_status()
            return list(response.json().get("data") or [])

        rows = await asyncio.to_thread(request)
        return [
            ModelCapability(
                id=str(item.get("id") or ""),
                provider="openai-compatible",
                generation=True,
                embedding=False,
            )
            for item in rows
            if item.get("id")
        ]

    async def embed(self, *, model: str, inputs: list[str]) -> list[list[float]]:
        def request() -> list[list[float]]:
            response = requests.post(
                f"{self.base_url}/embeddings",
                headers=self._headers(),
                json={"model": model, "input": inputs},
                timeout=self.timeout,
            )
            response.raise_for_status()
            rows = sorted(response.json().get("data") or [], key=lambda item: item.get("index", 0))
            return [list(item["embedding"]) for item in rows]

        return await asyncio.to_thread(request)


def configured_provider() -> ChatProvider:
    provider = os.getenv("LOCALGPT_INFERENCE_PROVIDER", "ollama").lower()
    if provider == "openai-compatible":
        return OpenAICompatibleProvider(
            os.environ["OPENAI_API_BASE"], os.getenv("OPENAI_API_KEY")
        )
    return OllamaProvider()

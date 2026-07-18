from __future__ import annotations

import json
import re
from typing import Any

import httpx
from openai import OpenAI

from .config import Config, load_config

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Local reasoning models may emit <think>...</think> inline; never let it leak downstream."""
    return _THINK_RE.sub("", text).strip()


def extract_json(raw: str) -> Any:
    raw = strip_thinking(raw)
    m = _FENCE_RE.search(raw)
    if m:
        raw = m.group(1)
    # fall back to the first balanced object/array in the string
    for opener, closer in (("{", "}"), ("[", "]")):
        start = raw.find(opener)
        if start == -1:
            continue
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(raw)):
            c = raw[i]
            if esc:
                esc = False
                continue
            if c == "\\":
                esc = True
            elif c == '"' and not esc:
                in_str = not in_str
            elif not in_str:
                if c == opener:
                    depth += 1
                elif c == closer:
                    depth -= 1
                    if depth == 0:
                        return json.loads(raw[start : i + 1])
        break
    raise ValueError(f"no parseable JSON in response: {raw[:200]!r}")


class LLM:
    """Thin client for any OpenAI-compatible local endpoint (ollama, vllm, mlx, llama.cpp)."""

    def __init__(self, role: str = "orchestrator", cfg: Config | None = None):
        self.cfg = cfg or load_config()
        self.role = role
        self.model = getattr(self.cfg.models, role, None) or self.cfg.models.orchestrator
        self.client = OpenAI(
            base_url=self.cfg.serving.base_url,
            api_key=self.cfg.serving.api_key,
            timeout=self.cfg.serving.timeout_s,
        )
        self._resolve_model()

    def _resolve_model(self) -> None:
        """If the configured model isn't served, fall back to a prefix match, then orchestrator."""
        try:
            served = [m.id for m in self.client.models.list().data]
        except Exception:
            return  # server not up yet; let the first real call raise a clear error
        if self.model in served:
            return
        base = self.model.split(":")[0]
        for m in served:
            if m.split(":")[0] == base or m.startswith(base):
                self.model = m
                return
        orch = self.cfg.models.orchestrator
        if self.role != "orchestrator" and orch in served:
            self.model = orch

    def chat(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        json_mode: bool = False,
        reasoning: str | None = None,
    ):
        """max_tokens must budget for reasoning models' thinking tokens: a starved budget
        yields finish_reason=length with EMPTY content. Pass reasoning="none" to disable
        thinking (ollama reasoning_effort) for trivial calls."""
        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if tools:
            kwargs["tools"] = tools
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if reasoning:
            kwargs["extra_body"] = {"reasoning_effort": reasoning}
        return self.client.chat.completions.create(**kwargs)

    def text(self, messages: list[dict], **kw) -> str:
        resp = self.chat(messages, **kw)
        return strip_thinking(resp.choices[0].message.content or "")

    def json(self, messages: list[dict], retries: int = 2, **kw) -> Any:
        last: Exception | None = None
        msgs = list(messages)
        for _ in range(retries + 1):
            raw = self.text(msgs, json_mode=True, **kw)
            try:
                return extract_json(raw)
            except (ValueError, json.JSONDecodeError) as e:
                last = e
                msgs = msgs + [
                    {"role": "assistant", "content": raw[:2000]},
                    {"role": "user", "content": "That was not valid JSON. Respond with ONLY the JSON object, no prose."},
                ]
        raise ValueError(f"model failed to produce JSON after {retries + 1} attempts: {last}")


def served_models(cfg: Config | None = None) -> list[str]:
    cfg = cfg or load_config()
    try:
        r = httpx.get(cfg.serving.base_url.rstrip("/").removesuffix("/v1") + "/api/tags", timeout=5)
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def served_context(cfg: Config | None = None) -> list[dict]:
    """Loaded models via /api/ps with the context length actually being served — a silent
    16K regression is visible here. Called lazily per request, never at import/startup."""
    cfg = cfg or load_config()
    try:
        r = httpx.get(cfg.serving.base_url.rstrip("/").removesuffix("/v1") + "/api/ps", timeout=5)
        return [
            {"name": m.get("name"), "context_length": m.get("context_length")}
            for m in r.json().get("models", [])
        ]
    except Exception:
        return []

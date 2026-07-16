from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any

from backend.agent_runtime.providers import ChatProvider
from backend.agent_runtime.tools import ToolContext, ToolRegistry


@dataclass(frozen=True, slots=True)
class AgentRequest:
    model: str
    messages: list[dict[str, Any]]
    allowed_tools: set[str] | None = None
    temperature: float = 0.2
    max_tokens: int | None = None
    max_iterations: int = 8
    max_tool_calls: int = 12
    max_elapsed_seconds: float = 300
    max_total_tokens: int = 100_000


@dataclass(frozen=True, slots=True)
class AgentResult:
    content: str
    messages: list[dict[str, Any]]
    tool_calls_executed: int
    iterations: int
    input_tokens: int = 0
    output_tokens: int = 0


class AgentBudgetExceeded(RuntimeError):
    pass


class AgentExecutor:
    """Bounded provider-neutral tool loop."""

    def __init__(self, *, provider: ChatProvider, tools: ToolRegistry) -> None:
        self.provider = provider
        self.tools = tools

    async def execute(
        self, request: AgentRequest, context: ToolContext
    ) -> AgentResult:
        messages = [dict(message) for message in request.messages]
        started = time.monotonic()
        tool_call_count = 0
        total_input_tokens = 0
        total_output_tokens = 0

        for iteration in range(1, request.max_iterations + 1):
            cancel_check = context.metadata.get("cancel_requested")
            if callable(cancel_check) and cancel_check():
                raise asyncio.CancelledError("Agent run was cancelled")
            if time.monotonic() - started > request.max_elapsed_seconds:
                raise AgentBudgetExceeded("Agent elapsed-time budget exceeded")
            turn = await self.provider.complete(
                model=request.model,
                messages=messages,
                tools=self.tools.definitions(request.allowed_tools),
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
            total_input_tokens += turn.input_tokens or 0
            total_output_tokens += turn.output_tokens or 0
            if total_input_tokens + total_output_tokens > request.max_total_tokens:
                raise AgentBudgetExceeded("Agent total-token budget exceeded")
            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": turn.content,
            }
            if turn.tool_calls:
                assistant_message["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": call.arguments,
                        },
                    }
                    for call in turn.tool_calls
                ]
            messages.append(assistant_message)

            if not turn.tool_calls:
                return AgentResult(
                    content=turn.content,
                    messages=messages,
                    tool_calls_executed=tool_call_count,
                    iterations=iteration,
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                )

            for call in turn.tool_calls:
                tool_call_count += 1
                if tool_call_count > request.max_tool_calls:
                    raise AgentBudgetExceeded("Agent tool-call budget exceeded")
                result = await self.tools.execute(
                    call.name, call.arguments, context=context
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": json.dumps(result, default=str),
                    }
                )
                checkpoint = context.metadata.get("checkpoint")
                if callable(checkpoint):
                    checkpoint(
                        {
                            "messages": messages,
                            "iteration": iteration,
                            "tool_calls_executed": tool_call_count,
                        }
                    )

        raise AgentBudgetExceeded("Agent iteration budget exceeded")

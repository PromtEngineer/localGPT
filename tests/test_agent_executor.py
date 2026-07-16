import asyncio

from backend.agent_runtime.executor import AgentExecutor, AgentRequest
from backend.agent_runtime.providers import AssistantTurn, ToolCall
from backend.agent_runtime.tools import ToolContext, ToolRegistry, ToolSpec


class ScriptedProvider:
    def __init__(self) -> None:
        self.turn = 0

    async def complete(self, **_kwargs: object) -> AssistantTurn:
        self.turn += 1
        if self.turn == 1:
            return AssistantTurn(
                content="",
                tool_calls=[ToolCall(id="call-1", name="lookup", arguments={"q": "x"})],
            )
        return AssistantTurn(content="grounded answer", tool_calls=[])


def test_agent_executes_a_tool_and_returns_the_followup_answer() -> None:
    registry = ToolRegistry()

    async def lookup(arguments: dict[str, object], _context: ToolContext) -> object:
        return {"fact": arguments["q"]}

    registry.register(
        ToolSpec(
            name="lookup",
            description="Lookup a fact",
            input_schema={
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
            handler=lookup,
        )
    )
    executor = AgentExecutor(provider=ScriptedProvider(), tools=registry)

    result = asyncio.run(
        executor.execute(
            AgentRequest(model="test", messages=[{"role": "user", "content": "ask"}]),
            ToolContext(run_id="run-1"),
        )
    )

    assert result.content == "grounded answer"
    assert result.tool_calls_executed == 1

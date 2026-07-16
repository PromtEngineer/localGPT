import asyncio

import pytest

from backend.agent_runtime.tools import ToolInputError, ToolRegistry, ToolSpec


def test_tool_inputs_are_validated_before_execution() -> None:
    registry = ToolRegistry()
    called = False

    async def add(arguments: dict[str, object], _context: object) -> dict[str, int]:
        nonlocal called
        called = True
        return {"value": int(arguments["left"]) + int(arguments["right"])}

    registry.register(
        ToolSpec(
            name="add",
            description="Add two integers",
            input_schema={
                "type": "object",
                "properties": {
                    "left": {"type": "integer"},
                    "right": {"type": "integer"},
                },
                "required": ["left", "right"],
                "additionalProperties": False,
            },
            handler=add,
        )
    )

    with pytest.raises(ToolInputError):
        asyncio.run(registry.execute("add", {"left": 1}, context=None))

    assert called is False

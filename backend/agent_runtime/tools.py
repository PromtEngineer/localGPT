from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import jsonschema


class ToolError(RuntimeError):
    pass


class ToolInputError(ToolError):
    pass


class ToolPermissionError(ToolError):
    pass


ToolHandler = Callable[[dict[str, Any], "ToolContext"], Awaitable[Any]]


@dataclass(slots=True)
class ToolContext:
    run_id: str
    session_id: str | None = None
    permissions: set[str] = field(default_factory=set)
    approved_tools: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)
    emit: Callable[[str, dict[str, Any]], None] | None = None


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler
    required_permissions: frozenset[str] = frozenset()
    approval_required: bool = False
    side_effects: bool = False
    timeout_seconds: float = 30.0
    max_output_bytes: int = 1_048_576

    def public_definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }

    def public_descriptor(self) -> dict[str, Any]:
        return {
            **self.public_definition(),
            "policy": {
                "required_permissions": sorted(self.required_permissions),
                "approval_required": self.approval_required,
                "side_effects": self.side_effects,
                "timeout_seconds": self.timeout_seconds,
                "max_output_bytes": self.max_output_bytes,
            },
        }


class ToolMiddleware(Protocol):
    async def before(
        self, spec: ToolSpec, arguments: dict[str, Any], context: ToolContext
    ) -> dict[str, Any]: ...

    async def after(
        self, spec: ToolSpec, result: Any, context: ToolContext
    ) -> Any: ...


class ToolRegistry:
    """Explicit, schema-validated and policy-controlled tool registry."""

    def __init__(self, middleware: list[ToolMiddleware] | None = None) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self._middleware = middleware or []

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"Tool already registered: {spec.name}")
        jsonschema.Draft202012Validator.check_schema(spec.input_schema)
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolInputError(f"Unknown tool: {name}") from exc

    def definitions(self, allowed: set[str] | None = None) -> list[dict[str, Any]]:
        names = sorted(self._tools if allowed is None else self._tools.keys() & allowed)
        return [self._tools[name].public_definition() for name in names]

    def names(self) -> list[str]:
        return sorted(self._tools)

    def descriptors(self) -> list[dict[str, Any]]:
        return [self._tools[name].public_descriptor() for name in sorted(self._tools)]

    async def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        context: ToolContext | None,
    ) -> Any:
        spec = self.get(name)
        resolved_context = context or ToolContext(run_id="standalone")
        try:
            jsonschema.validate(arguments, spec.input_schema)
        except jsonschema.ValidationError as exc:
            raise ToolInputError(
                f"Invalid input for {name}: {exc.message}"
            ) from exc

        missing = spec.required_permissions - resolved_context.permissions
        if missing:
            raise ToolPermissionError(
                f"Tool {name} requires permissions: {', '.join(sorted(missing))}"
            )
        if spec.approval_required and name not in resolved_context.approved_tools:
            raise ToolPermissionError(f"Tool {name} requires explicit approval")

        transformed = dict(arguments)
        for middleware in self._middleware:
            transformed = await middleware.before(spec, transformed, resolved_context)

        if resolved_context.emit:
            resolved_context.emit(
                "tool.started", {"tool": name, "arguments": transformed}
            )
        try:
            result = await asyncio.wait_for(
                spec.handler(transformed, resolved_context),
                timeout=spec.timeout_seconds,
            )
        except TimeoutError as exc:
            if resolved_context.emit:
                resolved_context.emit(
                    "tool.failed",
                    {"tool": name, "error": f"timed out after {spec.timeout_seconds:g}s"},
                )
            raise ToolError(
                f"Tool {name} timed out after {spec.timeout_seconds:g}s"
            ) from exc
        except Exception as exc:
            if resolved_context.emit:
                resolved_context.emit(
                    "tool.failed", {"tool": name, "error": str(exc)}
                )
            raise

        for middleware in reversed(self._middleware):
            result = await middleware.after(spec, result, resolved_context)

        serialized = json.dumps(result, default=str).encode("utf-8")
        if len(serialized) > spec.max_output_bytes:
            raise ToolError(
                f"Tool {name} output exceeded {spec.max_output_bytes} bytes"
            )
        if resolved_context.emit:
            resolved_context.emit("tool.completed", {"tool": name, "result": result})
        return result

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from backend.agent_runtime.tools import ToolContext, ToolRegistry, ToolSpec


@dataclass(frozen=True, slots=True)
class MCPConnector:
    id: str
    url: str
    authorization_token: str | None = None
    allowed_tools: frozenset[str] | None = None
    permissions: frozenset[str] = frozenset({"mcp:call"})
    approval_required: bool = True
    timeout_seconds: float = 60


class MCPConnectorRegistry:
    def __init__(self, connectors: dict[str, MCPConnector] | None = None) -> None:
        self.connectors = connectors or self._from_environment()

    @staticmethod
    def _from_environment() -> dict[str, MCPConnector]:
        loaded = json.loads(os.getenv("LOCALGPT_MCP_CONNECTORS_JSON", "{}"))
        connectors: dict[str, MCPConnector] = {}
        for connector_id, config in loaded.items():
            token = config.get("authorization_token")
            token_env = config.get("authorization_token_env")
            if token_env:
                token = os.getenv(str(token_env))
            allowed = config.get("allowed_tools")
            connectors[connector_id] = MCPConnector(
                id=connector_id,
                url=str(config["url"]),
                authorization_token=str(token) if token else None,
                allowed_tools=frozenset(allowed) if allowed is not None else None,
                permissions=frozenset(config.get("permissions", ["mcp:call"])),
                approval_required=bool(config.get("approval_required", True)),
                timeout_seconds=float(config.get("timeout_seconds", 60)),
            )
        return connectors


class MCPClient:
    """Official MCP SDK adapter for configured Streamable HTTP servers."""

    def __init__(self, connector: MCPConnector) -> None:
        self.connector = connector

    async def _session(self):
        try:
            import httpx
            from mcp import ClientSession
            from mcp.client.streamable_http import streamable_http_client
        except ImportError as exc:
            raise RuntimeError(
                "MCP connectors require the optional mcp>=1.27,<2 dependency"
            ) from exc
        headers = {}
        if self.connector.authorization_token:
            headers["Authorization"] = f"Bearer {self.connector.authorization_token}"
        http_client = httpx.AsyncClient(
            headers=headers,
            timeout=self.connector.timeout_seconds,
            follow_redirects=False,
        )
        transport = streamable_http_client(
            self.connector.url, http_client=http_client
        )
        return ClientSession, http_client, transport

    async def list_tools(self) -> list[dict[str, Any]]:
        ClientSession, http_client, transport = await self._session()
        async with http_client:
            async with transport as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    response = await session.list_tools()
                    tools = []
                    for tool in response.tools:
                        if (
                            self.connector.allowed_tools is not None
                            and tool.name not in self.connector.allowed_tools
                        ):
                            continue
                        tools.append(
                            {
                                "name": tool.name,
                                "description": tool.description or "",
                                "input_schema": tool.inputSchema,
                            }
                        )
                    return tools

    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if (
            self.connector.allowed_tools is not None
            and name not in self.connector.allowed_tools
        ):
            raise PermissionError(f"MCP tool is not allowlisted: {name}")
        ClientSession, http_client, transport = await self._session()
        async with http_client:
            async with transport as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(name, arguments=arguments)
                    if hasattr(result, "model_dump"):
                        return result.model_dump(mode="json")
                    return {"content": str(result)}


def _safe_tool_name(connector_id: str, tool_name: str) -> str:
    def normalize(value: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_-]", "_", value)

    return f"mcp__{normalize(connector_id)}__{normalize(tool_name)}"


async def register_mcp_tools(
    registry: ToolRegistry, connectors: MCPConnectorRegistry
) -> list[str]:
    registered: list[str] = []
    for connector in connectors.connectors.values():
        client = MCPClient(connector)
        for remote_tool in await client.list_tools():
            public_name = _safe_tool_name(connector.id, remote_tool["name"])

            async def call(
                arguments: dict[str, Any],
                _context: ToolContext,
                *,
                selected_client: MCPClient = client,
                selected_name: str = remote_tool["name"],
            ) -> dict[str, Any]:
                return await selected_client.call(selected_name, arguments)

            registry.register(
                ToolSpec(
                    name=public_name,
                    description=(
                        f"MCP connector {connector.id}: "
                        f"{remote_tool['description']}"
                    ),
                    input_schema=remote_tool["input_schema"],
                    handler=call,
                    required_permissions=connector.permissions,
                    approval_required=connector.approval_required,
                    side_effects=True,
                    timeout_seconds=connector.timeout_seconds,
                )
            )
            registered.append(public_name)
    return registered

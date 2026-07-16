from __future__ import annotations

import asyncio
import ipaddress
import os
import socket
from dataclasses import asdict
from typing import Any
from urllib.parse import urljoin, urlsplit

import requests

from backend.agent_runtime.artifacts import ArtifactStore
from backend.agent_runtime.tools import ToolContext, ToolRegistry, ToolSpec


class UnsafeURL(ValueError):
    pass


def _is_public_address(address: str) -> bool:
    parsed = ipaddress.ip_address(address)
    return not any(
        (
            parsed.is_private,
            parsed.is_loopback,
            parsed.is_link_local,
            parsed.is_multicast,
            parsed.is_reserved,
            parsed.is_unspecified,
        )
    )


def validate_public_url(url: str) -> str:
    """Reject non-HTTP URLs and any destination resolving off the public internet."""
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise UnsafeURL("Only public HTTP and HTTPS URLs are allowed")
    if parsed.username or parsed.password:
        raise UnsafeURL("Credentials in web-fetch URLs are not allowed")
    hostname = parsed.hostname.rstrip(".").lower()
    if hostname == "localhost" or hostname.endswith(".localhost"):
        raise UnsafeURL("Loopback destinations are not allowed")
    try:
        addresses = {str(ipaddress.ip_address(hostname))}
    except ValueError:
        try:
            addresses = {
                result[4][0]
                for result in socket.getaddrinfo(
                    hostname, parsed.port or (443 if parsed.scheme == "https" else 80)
                )
            }
        except socket.gaierror as exc:
            raise UnsafeURL(f"Could not resolve web-fetch host: {hostname}") from exc
    if not addresses or not all(_is_public_address(address) for address in addresses):
        raise UnsafeURL("Private, loopback, link-local, and reserved networks are blocked")
    return url


def fetch_public_url(
    url: str,
    *,
    max_bytes: int = 2 * 1024 * 1024,
    timeout: float = 15,
    max_redirects: int = 3,
) -> dict[str, Any]:
    current = validate_public_url(url)
    headers = {"User-Agent": "LocalGPT-Fetch/1.0", "Accept": "text/*,application/json"}
    for _ in range(max_redirects + 1):
        response = requests.get(
            current,
            headers=headers,
            timeout=timeout,
            allow_redirects=False,
            stream=True,
        )
        if response.is_redirect or response.is_permanent_redirect:
            location = response.headers.get("Location")
            if not location:
                raise UnsafeURL("Redirect response omitted Location")
            current = validate_public_url(urljoin(current, location))
            continue
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").split(";", 1)[0]
        if not (
            content_type.startswith("text/")
            or content_type in {"application/json", "application/xml"}
        ):
            raise UnsafeURL(f"Unsupported web content type: {content_type or 'unknown'}")
        chunks: list[bytes] = []
        size = 0
        for chunk in response.iter_content(64 * 1024):
            size += len(chunk)
            if size > max_bytes:
                raise UnsafeURL(f"Web response exceeds {max_bytes} bytes")
            chunks.append(chunk)
        return {
            "url": current,
            "content_type": content_type,
            "content": b"".join(chunks).decode(response.encoding or "utf-8", errors="replace"),
            "trust": "untrusted_external_content",
        }
    raise UnsafeURL(f"Web fetch exceeded {max_redirects} redirects")


class RagRetrievalClient:
    def __init__(self, base_url: str | None = None, token: str | None = None) -> None:
        self.base_url = (base_url or os.getenv("RAG_API_URL", "http://127.0.0.1:8001")).rstrip("/")
        self.token = token if token is not None else os.getenv("LOCALGPT_API_TOKEN")

    async def search(
        self,
        query: str,
        *,
        session_id: str,
        retrieval_k: int = 8,
        search_type: str = "hybrid",
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        def request() -> dict[str, Any]:
            response = requests.post(
                f"{self.base_url}/chat",
                headers=headers,
                json={
                    "query": query,
                    "session_id": session_id,
                    "force_rag": True,
                    "retrieval_k": retrieval_k,
                    "search_type": search_type,
                    **(options or {}),
                },
                timeout=300,
            )
            response.raise_for_status()
            return response.json()

        result = await asyncio.to_thread(request)
        sources = []
        for rank, source in enumerate(result.get("source_documents") or [], start=1):
            metadata = source.get("metadata") or {}
            sources.append(
                {
                    "rank": rank,
                    "chunk_id": source.get("chunk_id"),
                    "document_id": source.get("document_id")
                    or metadata.get("document_id"),
                    "chunk_index": source.get("chunk_index")
                    or metadata.get("chunk_index"),
                    "page": metadata.get("page") or metadata.get("page_number"),
                    "text": source.get("text"),
                    "score": source.get("score"),
                }
            )
        return {"answer": result.get("answer", ""), "citations": sources}


def register_core_tools(
    registry: ToolRegistry,
    *,
    artifacts: ArtifactStore,
    retrieval: RagRetrievalClient | None = None,
) -> None:
    retrieval_client = retrieval or RagRetrievalClient()

    async def search_knowledge(
        arguments: dict[str, Any], context: ToolContext
    ) -> dict[str, Any]:
        if not context.session_id:
            raise ValueError("Knowledge search requires a session")
        return await retrieval_client.search(
            arguments["query"],
            session_id=context.session_id,
            retrieval_k=arguments.get("top_k", 8),
            search_type=arguments.get("search_type", "hybrid"),
            options={
                key: value
                for key, value in arguments.items()
                if key not in {"query", "top_k", "search_type"}
            },
        )

    async def list_artifacts(
        _arguments: dict[str, Any], context: ToolContext
    ) -> dict[str, Any]:
        return {
            "artifacts": [
                asdict(artifact)
                for artifact in artifacts.list(session_id=context.session_id)
            ]
        }

    async def read_artifact(
        arguments: dict[str, Any], context: ToolContext
    ) -> dict[str, Any]:
        artifact = artifacts.get(arguments["artifact_id"])
        if artifact is None or artifact.session_id != context.session_id:
            raise KeyError("Artifact not found")
        content = artifacts.read_bytes(artifact.id)
        return {
            "artifact": asdict(artifact),
            "content": content.decode("utf-8", errors="replace"),
        }

    async def web_fetch(
        arguments: dict[str, Any], _context: ToolContext
    ) -> dict[str, Any]:
        return await asyncio.to_thread(fetch_public_url, arguments["url"])

    async def web_search(
        arguments: dict[str, Any], _context: ToolContext
    ) -> dict[str, Any]:
        api_key = os.getenv("BRAVE_SEARCH_API_KEY")
        if not api_key:
            raise RuntimeError("BRAVE_SEARCH_API_KEY is not configured")

        def search() -> dict[str, Any]:
            response = requests.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers={"X-Subscription-Token": api_key, "Accept": "application/json"},
                params={"q": arguments["query"], "count": arguments.get("count", 5)},
                timeout=15,
            )
            response.raise_for_status()
            rows = response.json().get("web", {}).get("results", [])
            return {
                "results": [
                    {
                        "title": row.get("title"),
                        "url": row.get("url"),
                        "description": row.get("description"),
                        "trust": "untrusted_external_content",
                    }
                    for row in rows
                ]
            }

        return await asyncio.to_thread(search)

    registry.register(
        ToolSpec(
            name="search_knowledge",
            description="Search documents linked to the current LocalGPT session.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 1},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 30},
                    "search_type": {
                        "type": "string",
                        "enum": ["hybrid", "dense", "lexical"],
                    },
                    "query_decompose": {"type": "boolean"},
                    "compose_sub_answers": {"type": "boolean"},
                    "ai_rerank": {"type": "boolean"},
                    "context_expand": {"type": "boolean"},
                    "verify": {"type": "boolean"},
                    "context_window_size": {"type": "integer", "minimum": 0, "maximum": 20},
                    "reranker_top_k": {"type": "integer", "minimum": 1, "maximum": 100},
                    "dense_weight": {"type": "number", "minimum": 0, "maximum": 1},
                    "provence_prune": {"type": "boolean"},
                    "provence_threshold": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            handler=search_knowledge,
            required_permissions=frozenset({"knowledge:read"}),
            timeout_seconds=300,
        )
    )
    registry.register(
        ToolSpec(
            name="list_artifacts",
            description="List artifacts owned by the current session.",
            input_schema={"type": "object", "additionalProperties": False},
            handler=list_artifacts,
            required_permissions=frozenset({"artifact:read"}),
        )
    )
    registry.register(
        ToolSpec(
            name="read_artifact",
            description="Read a text artifact by identifier.",
            input_schema={
                "type": "object",
                "properties": {"artifact_id": {"type": "string"}},
                "required": ["artifact_id"],
                "additionalProperties": False,
            },
            handler=read_artifact,
            required_permissions=frozenset({"artifact:read"}),
        )
    )
    registry.register(
        ToolSpec(
            name="web_fetch",
            description="Fetch bounded text content from a public URL.",
            input_schema={
                "type": "object",
                "properties": {"url": {"type": "string", "format": "uri"}},
                "required": ["url"],
                "additionalProperties": False,
            },
            handler=web_fetch,
            required_permissions=frozenset({"network:public"}),
        )
    )
    registry.register(
        ToolSpec(
            name="web_search",
            description="Search the public web using the configured search provider.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 1},
                    "count": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            handler=web_search,
            required_permissions=frozenset({"network:public"}),
        )
    )

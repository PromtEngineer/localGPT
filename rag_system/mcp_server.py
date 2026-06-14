#!/usr/bin/env python3
"""Model Context Protocol (stdio) server exposing LocalGPT's indexes as tools.

Lets an MCP client (Claude Desktop / Claude Code) list the local indexes and
ask questions against them through the unified FastAPI backend,
so all of LocalGPT's retrieval, multi-collection search, metadata filtering,
and citation logic applies unchanged. Nothing leaves the machine.

Zero dependencies (stdlib only): newline-delimited JSON-RPC 2.0 over stdio,
protocol version 2025-03-26. Requires the backend on port 8000; its URL is
configurable via BACKEND_URL.

Register with Claude Code (from the repo root):
    claude mcp add localgpt -- python -m rag_system.mcp_server
or add the .mcp.json in the repo root. See the module docstring in
.mcp.json / README for the Claude Desktop config.
"""

import json
import os
import sys
import urllib.error
import urllib.request

PROTOCOL_VERSION = "2025-03-26"
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
SERVER_INFO = {"name": "localgpt", "version": "1.0.0"}


def _log(msg: str) -> None:
    # stderr only — stdout is reserved for MCP messages
    print(f"[localgpt-mcp] {msg}", file=sys.stderr, flush=True)


def _http_json(method: str, url: str, payload: dict | None = None, timeout: int = 600):
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _strip_uuid_prefix(name: str) -> str:
    import re

    return re.sub(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_",
        "",
        str(name or ""),
        flags=re.IGNORECASE,
    )


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------

TOOLS = [
    {
        "name": "list_indexes",
        "description": (
            "List the document indexes available in LocalGPT. Returns each "
            "index's id, name, document count, and (if defined) its metadata "
            "filter schema. Call this first to discover what can be queried."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "ask_index",
        "description": (
            "Ask a question against a LocalGPT index. Performs hybrid "
            "retrieval and returns a synthesized answer with the source "
            "documents it drew from. Optionally restrict retrieval with "
            "typed metadata filters (see the index's schema from "
            'list_indexes), e.g. {"project": "Antapaccay", "year": '
            '{">=": 2020}}.'
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "index": {
                    "type": "string",
                    "description": "Index id (prefix accepted) or exact name, from list_indexes.",
                },
                "question": {
                    "type": "string",
                    "description": "The question to answer.",
                },
                "filters": {
                    "type": "object",
                    "description": "Optional metadata filters, validated against the index schema.",
                },
            },
            "required": ["index", "question"],
        },
    },
]


def _resolve_index(index_ref: str) -> dict:
    """Find an index by id-prefix or exact (case-insensitive) name."""
    data = _http_json("GET", f"{BACKEND_URL}/indexes")
    indexes = data.get("indexes", [])
    ref = (index_ref or "").strip()
    for ix in indexes:
        if ix.get("id") == ref:
            return ix
    for ix in indexes:
        if (ix.get("id") or "").startswith(ref) or (
            ix.get("name") or ""
        ).lower() == ref.lower():
            return ix
    raise ValueError(
        f"No index matches '{index_ref}'. Available: "
        + ", ".join(ix.get("name") or ix.get("id", "?") for ix in indexes)
        or "(none)"
    )


def _tool_list_indexes(_args: dict) -> str:
    data = _http_json("GET", f"{BACKEND_URL}/indexes")
    lines = []
    for ix in data.get("indexes", []):
        meta = ix.get("metadata") or {}
        schema = meta.get("metadata_schema")
        fields = (
            ", ".join(f"{f['name']}:{f['type']}" for f in schema) if schema else "—"
        )
        lines.append(
            f"• {ix.get('name')}  (id {ix.get('id', '')[:8]}, "
            f"{len(ix.get('documents') or [])} docs, status {meta.get('status', '?')})\n"
            f"    filter fields: {fields}"
        )
    return "Available indexes:\n" + "\n".join(lines) if lines else "No indexes found."


def _tool_ask_index(args: dict) -> str:
    index_ref = args.get("index")
    question = args.get("question")
    if not index_ref or not question:
        raise ValueError("'index' and 'question' are both required")
    ix = _resolve_index(index_ref)
    table = ix.get("vector_table_name")
    if not table:
        raise ValueError(
            f"Index '{ix.get('name')}' has no vector table (not built yet?)"
        )

    payload = {"query": question, "table_name": table, "force_rag": True}
    if isinstance(args.get("filters"), dict) and args["filters"]:
        payload["filters"] = args["filters"]

    result = _http_json("POST", f"{BACKEND_URL}/rag/chat", payload)
    if "error" in result:
        raise RuntimeError(result["error"])

    answer = result.get("answer", "(no answer)")
    sources = result.get("source_documents", [])
    seen, src_lines = set(), []
    for d in sources:
        name = _strip_uuid_prefix(d.get("document_id"))
        if name not in seen:
            seen.add(name)
            src_lines.append(f"  - {name}")
    src_block = ("\n\nSources:\n" + "\n".join(src_lines)) if src_lines else ""
    return f"{answer}{src_block}"


TOOL_FUNCS = {"list_indexes": _tool_list_indexes, "ask_index": _tool_ask_index}


# --------------------------------------------------------------------------
# JSON-RPC / MCP plumbing
# --------------------------------------------------------------------------


def _send(msg: dict) -> None:
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def _result(req_id, result) -> None:
    _send({"jsonrpc": "2.0", "id": req_id, "result": result})


def _error(req_id, code: int, message: str) -> None:
    _send({"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}})


def _handle(msg: dict) -> None:
    method = msg.get("method")
    req_id = msg.get("id")

    if method == "initialize":
        _result(
            req_id,
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": SERVER_INFO,
                "instructions": "Query LocalGPT's local document indexes. Call list_indexes, then ask_index.",
            },
        )
    elif method == "notifications/initialized":
        return  # notification: no response
    elif method == "tools/list":
        _result(req_id, {"tools": TOOLS})
    elif method == "tools/call":
        params = msg.get("params") or {}
        name = params.get("name")
        func = TOOL_FUNCS.get(name)
        if not func:
            _error(req_id, -32601, f"Unknown tool: {name}")
            return
        try:
            text = func(params.get("arguments") or {})
            _result(
                req_id, {"content": [{"type": "text", "text": text}], "isError": False}
            )
        except urllib.error.URLError as e:
            _result(
                req_id,
                {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Could not reach LocalGPT ({e}). Is it running on {BACKEND_URL}?",
                        }
                    ],
                    "isError": True,
                },
            )
        except Exception as e:
            _result(
                req_id,
                {
                    "content": [{"type": "text", "text": f"Tool error: {e}"}],
                    "isError": True,
                },
            )
    elif req_id is not None:
        _error(req_id, -32601, f"Method not found: {method}")
    # else: unknown notification → ignore


def main() -> None:
    _log(f"started (backend={BACKEND_URL})")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            _error(None, -32700, "Parse error")
            continue
        try:
            _handle(msg)
        except Exception as e:  # never let one bad message kill the loop
            _log(f"handler crash: {e}")
            if msg.get("id") is not None:
                _error(msg.get("id"), -32603, f"Internal error: {e}")


if __name__ == "__main__":
    main()

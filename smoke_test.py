#!/usr/bin/env python3
"""
LocalGPT smoke test.

Runs against a live app (all services must be started first).
Checks connectivity, key API contracts, and an end-to-end chat round-trip.

Usage:
    python smoke_test.py [--timeout SECONDS] [--fast] [--no-color]

Exit code: 0 = all checks passed/skipped, 1 = one or more hard failures.
"""
from __future__ import annotations

import argparse
import sys
import textwrap
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

try:
    import requests
except ImportError:
    sys.exit("requests is not installed. Run: pip install requests")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BACKEND  = "http://localhost:8000"
RAG_API  = "http://localhost:8001"
OLLAMA   = "http://localhost:11434"
FRONTEND = "http://localhost:3000"
DEFAULT_GENERATION_MODEL = "qwen3:8b"

# ---------------------------------------------------------------------------
# Result state
# ---------------------------------------------------------------------------
class Status(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"


@dataclass
class Result:
    label:  str
    status: Status
    detail: str = ""


@dataclass
class RunState:
    """All mutable state for one test run — no module-level globals."""
    results:          list[Result]       = field(default_factory=list)
    session_id:       str | None         = None
    first_model:      str | None         = None
    health_cache:     dict | None        = None
    llm_timeout:      int                = 120
    use_color:        bool               = True
    fast:             bool               = False

    # connectivity flags set in section 1
    c_ollama:   bool = True
    c_rag:      bool = True
    c_backend:  bool = True
    c_frontend: bool = True


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
_COLORS = {
    Status.PASS: "\033[32m",
    Status.FAIL: "\033[31m",
    Status.SKIP: "\033[33m",
}
_RESET = "\033[0m"
_BOLD  = "\033[1m"
_DIM   = "\033[2m"


def _colored(text: str, code: str, state: RunState) -> str:
    return f"{code}{text}{_RESET}" if state.use_color else text


def _print_result(r: Result, state: RunState) -> None:
    color = _COLORS[r.status]
    mark  = _colored(r.status.value, color, state)
    label = r.label
    yellow = "\033[33m"
    extra = f"  {_colored('→ ' + r.detail, yellow, state)}" if r.detail else ""
    print(f"  [{mark}] {label}{extra}")


def _section(title: str, state: RunState) -> None:
    t = f"{_BOLD}{title}{_RESET}" if state.use_color else title
    print(f"\n{t}")


def _record(label: str, status: Status, detail: str, state: RunState) -> bool:
    r = Result(label, status, detail)
    state.results.append(r)
    _print_result(r, state)
    return status == Status.PASS


def _check(label: str, fn: Callable[[], tuple[bool, str]], state: RunState) -> bool:
    try:
        passed, detail = fn()
        status = Status.PASS if passed else Status.FAIL
    except Exception as exc:
        # Shorten noisy connection-refused messages
        msg = str(exc)
        if "Connection refused" in msg or "NewConnectionError" in msg:
            detail = "connection refused"
        elif "Max retries exceeded" in msg:
            detail = "connection refused"
        else:
            detail = msg[:120]
        status = Status.FAIL
        passed = False
    return _record(label, status, detail if status == Status.FAIL else (detail or ""), state)


def _skip(label: str, reason: str, state: RunState) -> bool:
    return _record(label, Status.SKIP, reason, state)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
def _get(url: str, timeout: int = 10) -> requests.Response:
    return requests.get(url, timeout=timeout)


def _post(url: str, payload: dict, timeout: int = 10) -> requests.Response:
    return requests.post(url, json=payload, timeout=timeout)


def _delete(url: str, timeout: int = 10) -> requests.Response:
    return requests.delete(url, timeout=timeout)


# ---------------------------------------------------------------------------
# Cached health fetch
# ---------------------------------------------------------------------------
def _health(state: RunState) -> dict:
    if state.health_cache is None:
        state.health_cache = _get(f"{BACKEND}/health").json()
    return state.health_cache


def _pick_model(state: RunState) -> str | None:
    if state.first_model:
        return state.first_model
    try:
        models = _health(state).get("available_models") or []
        if models:
            state.first_model = DEFAULT_GENERATION_MODEL if DEFAULT_GENERATION_MODEL in models else models[0]
            return state.first_model
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Section 1 — Connectivity
# ---------------------------------------------------------------------------
def _section_connectivity(state: RunState) -> None:
    _section("1. Connectivity", state)

    def chk_ollama():
        r = _get(f"{OLLAMA}/api/tags")
        return r.status_code == 200, "" if r.status_code == 200 else f"HTTP {r.status_code}"

    def chk_rag():
        r = _get(f"{RAG_API}/models")
        return r.status_code == 200, "" if r.status_code == 200 else f"HTTP {r.status_code}"

    def chk_backend():
        r = _get(f"{BACKEND}/health")
        return r.status_code == 200, "" if r.status_code == 200 else f"HTTP {r.status_code}"

    def chk_frontend():
        r = _get(f"{FRONTEND}/", timeout=15)
        return r.status_code == 200, "" if r.status_code == 200 else f"HTTP {r.status_code}"

    state.c_ollama   = _check("Ollama reachable    (11434)", chk_ollama,   state)
    state.c_rag      = _check("RAG API reachable   (8001)",  chk_rag,      state)
    state.c_backend  = _check("Backend reachable   (8000)",  chk_backend,  state)
    state.c_frontend = _check("Frontend reachable  (3000)",  chk_frontend, state)


# ---------------------------------------------------------------------------
# Section 2 — Backend /health content
# ---------------------------------------------------------------------------
def _section_health(state: RunState) -> None:
    _section("2. Backend /health", state)

    checks = [
        ("status == 'ok'",         lambda d: (d.get("status") == "ok",            f"status={d.get('status')!r}")),
        ("ollama_running == true",  lambda d: (bool(d.get("ollama_running")),      "ollama_running is false")),
        ("rag_system_available",    lambda d: (bool(d.get("rag_system_available")),"rag_system_available is false")),
        ("≥1 Ollama model present", lambda d: (len(d.get("available_models") or []) > 0,
                                               f"{len(d.get('available_models') or [])} model(s)")),
        ("database_stats present",  lambda d: ("database_stats" in d,             "database_stats key missing")),
    ]

    if not state.c_backend:
        for label, _ in checks:
            _skip(label, "backend offline", state)
        return

    try:
        data = _health(state)
    except Exception as exc:
        for label, _ in checks:
            _skip(label, f"health fetch failed: {exc}", state)
        return

    for label, fn in checks:
        ok, detail = fn(data)
        _record(label, Status.PASS if ok else Status.FAIL, "" if ok else detail, state)


# ---------------------------------------------------------------------------
# Section 3 — Models endpoints
# ---------------------------------------------------------------------------
def _section_models(state: RunState) -> None:
    _section("3. Models endpoints", state)

    def chk_backend_models():
        data = _get(f"{BACKEND}/models").json()
        ok = isinstance(data.get("generation_models"), list) and \
             isinstance(data.get("embedding_models"), list)
        if ok:
            g, e = len(data["generation_models"]), len(data["embedding_models"])
            return ok, f"{g} generation, {e} embedding"
        return ok, "unexpected response shape"

    def chk_rag_models():
        data = _get(f"{RAG_API}/models").json()
        ok = isinstance(data.get("generation_models"), list) and \
             isinstance(data.get("embedding_models"), list)
        return ok, "" if ok else "unexpected response shape"

    if state.c_backend:
        _check("Backend  /models shape", chk_backend_models, state)
    else:
        _skip("Backend  /models shape", "backend offline", state)

    if state.c_rag:
        _check("RAG API  /models shape", chk_rag_models, state)
    else:
        _skip("RAG API  /models shape", "RAG API offline", state)


# ---------------------------------------------------------------------------
# Section 4 — Sessions CRUD
# ---------------------------------------------------------------------------
def _section_sessions(state: RunState) -> None:
    _section("4. Sessions CRUD", state)

    if not state.c_backend:
        for label in ("POST /sessions", "GET /sessions/{id}",
                      "GET /sessions (list contains)", "POST /sessions/{id}/rename",
                      "DELETE /sessions/{id}", "GET /sessions/{id} → 404"):
            _skip(label, "backend offline", state)
        return

    # Create
    def chk_create():
        r = _post(f"{BACKEND}/sessions", {"title": "smoke-test", "model": "smoke"})
        ok = r.status_code == 200 and "session_id" in r.json()
        if ok:
            state.session_id = r.json()["session_id"]
        return ok, state.session_id or f"HTTP {r.status_code}: {r.text[:80]}"

    created = _check("POST /sessions", chk_create, state)

    # Dependent checks
    if not created or not state.session_id:
        for label in ("GET /sessions/{id}", "GET /sessions (list contains)",
                      "POST /sessions/{id}/rename",
                      "DELETE /sessions/{id}", "GET /sessions/{id} → 404"):
            _skip(label, "session creation failed", state)
        return

    sid = state.session_id

    def chk_get():
        r = _get(f"{BACKEND}/sessions/{sid}")
        body = r.json() if r.status_code == 200 else {}
        ok = r.status_code == 200 and body.get("session", {}).get("id") == sid
        return ok, "" if ok else f"HTTP {r.status_code}"

    def chk_list():
        data = _get(f"{BACKEND}/sessions").json()
        ids = [s["id"] for s in data.get("sessions", [])]
        ok = sid in ids
        return ok, "" if ok else "session not found in list"

    def chk_rename():
        r = _post(f"{BACKEND}/sessions/{sid}/rename", {"title": "smoke-renamed"})
        if r.status_code != 200:
            return False, f"HTTP {r.status_code}"
        title = r.json().get("session", {}).get("title", "")
        ok = title == "smoke-renamed"
        return ok, "" if ok else f"title={title!r}"

    def chk_delete():
        r = _delete(f"{BACKEND}/sessions/{sid}")
        return r.status_code == 200, "" if r.status_code == 200 else f"HTTP {r.status_code}"

    def chk_gone():
        r = _get(f"{BACKEND}/sessions/{sid}")
        ok = r.status_code == 404
        return ok, "" if ok else f"expected 404, got {r.status_code}"

    _check("GET /sessions/{id}",            chk_get,    state)
    _check("GET /sessions (list contains)", chk_list,   state)
    _check("POST /sessions/{id}/rename",    chk_rename, state)
    _check("DELETE /sessions/{id}",         chk_delete, state)
    _check("GET /sessions/{id} → 404",      chk_gone,   state)


# ---------------------------------------------------------------------------
# Section 5 — Index CRUD
# ---------------------------------------------------------------------------
def _section_indexes(state: RunState) -> None:
    _section("5. Indexes CRUD", state)

    if not state.c_backend:
        for label in ("GET /indexes", "POST /indexes",
                      "GET /indexes/{id}", "DELETE /indexes/{id}"):
            _skip(label, "backend offline", state)
        return

    def chk_list():
        r = _get(f"{BACKEND}/indexes")
        ok = r.status_code == 200 and "indexes" in r.json()
        return ok, "" if ok else f"HTTP {r.status_code}"

    _check("GET /indexes", chk_list, state)

    # Create a throw-away index
    index_id: list[str] = []  # mutable box for closure

    def chk_create():
        r = _post(f"{BACKEND}/indexes", {"name": "smoke-index"})
        ok = r.status_code == 200
        if ok:
            data = r.json()
            idx = data.get("id") or data.get("index_id") or (data.get("index") or {}).get("id")
            if idx:
                index_id.append(idx)
                return True, idx
            return False, "no id in response"
        return False, f"HTTP {r.status_code}: {r.text[:80]}"

    created = _check("POST /indexes", chk_create, state)

    if not created or not index_id:
        _skip("GET /indexes/{id}",    "index creation failed", state)
        _skip("DELETE /indexes/{id}", "index creation failed", state)
        return

    iid = index_id[0]

    def chk_get():
        r = _get(f"{BACKEND}/indexes/{iid}")
        ok = r.status_code == 200
        if ok:
            data = r.json()
            returned_id = data.get("id") or data.get("index_id") or (data.get("index") or {}).get("id")
            ok = returned_id == iid
        return ok, "" if ok else f"HTTP {r.status_code}"

    def chk_delete():
        r = _delete(f"{BACKEND}/indexes/{iid}")
        return r.status_code == 200, "" if r.status_code == 200 else f"HTTP {r.status_code}"

    _check("GET /indexes/{id}",    chk_get,    state)
    _check("DELETE /indexes/{id}", chk_delete, state)


# ---------------------------------------------------------------------------
# Section 6 — End-to-end LLM chat
# ---------------------------------------------------------------------------
PING_PROMPT = "Reply with exactly the word PONG and nothing else."


def _section_chat(state: RunState) -> None:
    _section("6. End-to-end LLM chat", state)

    if state.fast:
        for label in ("Backend  /chat round-trip",
                      "RAG API  /chat round-trip",
                      "Session  /messages roundtrip"):
            _skip(label, "--fast mode", state)
        return

    if not state.c_ollama:
        for label in ("Backend  /chat round-trip",
                      "RAG API  /chat round-trip",
                      "Session  /messages roundtrip"):
            _skip(label, "Ollama offline", state)
        return

    model = _pick_model(state)
    if not model:
        for label in ("Backend  /chat round-trip",
                      "RAG API  /chat round-trip",
                      "Session  /messages roundtrip"):
            _skip(label, "no Ollama model installed", state)
        return

    def chk_backend_chat():
        if not state.c_backend:
            return False, "backend offline"
        r = _post(f"{BACKEND}/chat",
                  {"message": PING_PROMPT, "model": model},
                  timeout=state.llm_timeout)
        ok = r.status_code == 200 and bool(r.json().get("response"))
        if ok:
            snippet = r.json()["response"][:60].replace("\n", " ")
            return True, f"model={model!r} reply={snippet!r}"
        return False, f"HTTP {r.status_code}: {r.text[:120]}"

    def chk_rag_chat():
        if not state.c_rag:
            return False, "RAG API offline"
        r = _post(f"{RAG_API}/chat",
                  {"query": PING_PROMPT, "model": model},
                  timeout=state.llm_timeout)
        if r.status_code != 200:
            return False, f"HTTP {r.status_code}: {r.text[:120]}"
        data = r.json()
        reply = data.get("answer") or data.get("response") or data.get("result") or ""
        ok = bool(reply)
        return ok, f"reply={str(reply)[:60]!r}" if ok else "empty answer in response"

    def chk_session_roundtrip():
        if not state.c_backend:
            return False, "backend offline"
        r = _post(f"{BACKEND}/sessions", {"title": "smoke-roundtrip", "model": model})
        if r.status_code != 200:
            return False, f"session create HTTP {r.status_code}"
        sid = r.json()["session_id"]
        try:
            r = _post(f"{BACKEND}/sessions/{sid}/messages",
                      {"message": PING_PROMPT, "model": model},
                      timeout=state.llm_timeout)
            if r.status_code != 200:
                return False, f"messages HTTP {r.status_code}: {r.text[:120]}"
            data = r.json()
            reply = data.get("response") or data.get("answer") or data.get("message") or ""
            ok = bool(reply)
            snippet = str(reply)[:60].replace("\n", " ")
            return ok, f"reply={snippet!r}" if ok else "empty reply"
        finally:
            try:
                _delete(f"{BACKEND}/sessions/{sid}", timeout=5)
            except Exception:
                pass

    _check("Backend  /chat round-trip",    chk_backend_chat,      state)
    _check("RAG API  /chat round-trip",    chk_rag_chat,          state)
    _check("Session  /messages roundtrip", chk_session_roundtrip, state)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run(llm_timeout: int = 120, fast: bool = False, use_color: bool = True) -> int:
    state = RunState(llm_timeout=llm_timeout, fast=fast, use_color=use_color)
    t0 = time.monotonic()

    _section_connectivity(state)
    _section_health(state)
    _section_models(state)
    _section_sessions(state)
    _section_indexes(state)
    _section_chat(state)

    elapsed = time.monotonic() - t0

    # ── Summary ──────────────────────────────────────────────────────────────
    total   = len(state.results)
    passed  = sum(1 for r in state.results if r.status == Status.PASS)
    skipped = sum(1 for r in state.results if r.status == Status.SKIP)
    failed  = sum(1 for r in state.results if r.status == Status.FAIL)

    bar = "─" * 50
    print(f"\n{bar}")
    elapsed_str = f"  ({elapsed:.1f}s)"

    if failed == 0:
        msg = f"All {passed} checks passed, {skipped} skipped.{elapsed_str}"
        print(_colored(msg, _BOLD + "\033[32m", state))
        return 0

    print(_colored(f"{failed} check(s) FAILED, {passed} passed, {skipped} skipped.{elapsed_str}",
                   _BOLD + "\033[31m", state))
    print("\nFailed checks:")
    for r in state.results:
        if r.status == Status.FAIL:
            extra = f": {r.detail}" if r.detail else ""
            red = "\033[31m"
            print(f"  {_colored('✗', red, state)} {r.label}{extra}")
    return 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="LocalGPT smoke test")
    parser.add_argument("--timeout", type=int, default=120,
                        help="Seconds to wait for an LLM response (default: 120)")
    parser.add_argument("--fast", action="store_true",
                        help="Skip LLM round-trip checks (connectivity + CRUD only)")
    parser.add_argument("--no-color", action="store_true",
                        help="Disable ANSI color output (useful for CI logs)")
    args = parser.parse_args()

    use_color = not args.no_color
    bold = _BOLD if use_color else ""
    reset = _RESET if use_color else ""

    mode = "fast (no LLM)" if args.fast else f"full (LLM timeout {args.timeout}s)"
    print(textwrap.dedent(f"""\
        {bold}LocalGPT Smoke Test{reset}  [{mode}]
        Backend:  {BACKEND}
        RAG API:  {RAG_API}
        Ollama:   {OLLAMA}
        Frontend: {FRONTEND}
    """))

    sys.exit(run(llm_timeout=args.timeout, fast=args.fast, use_color=use_color))


if __name__ == "__main__":
    main()

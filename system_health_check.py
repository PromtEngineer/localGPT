#!/usr/bin/env python3
"""Probe LocalGPT service health and configured storage paths."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import requests

from localgpt_runtime import env_path


def _headers() -> dict[str, str]:
    token = os.environ.get("LOCALGPT_API_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def probe(name: str, url: str, headers: dict[str, str] | None = None) -> bool:
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        print(f"✅ {name}: {url}")
        return True
    except requests.RequestException as exc:
        print(f"❌ {name}: {exc}")
        return False


def storage_check(name: str, path: Path, directory: bool = True) -> bool:
    try:
        target = path if directory else path.parent
        target.mkdir(parents=True, exist_ok=True)
        if not os.access(target, os.R_OK | os.W_OK):
            raise PermissionError(f"not readable/writable: {target}")
        print(f"✅ {name}: {path}")
        return True
    except OSError as exc:
        print(f"❌ {name}: {exc}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-ollama", action="store_true")
    args = parser.parse_args()

    backend_port = os.environ.get("LOCALGPT_BACKEND_PORT", "8000")
    checks = [
        probe("backend", f"http://127.0.0.1:{backend_port}/health", _headers()),
        probe("RAG API", "http://127.0.0.1:8001/health", _headers()),
        storage_check("SQLite", env_path("LOCALGPT_DB_PATH", "data/chat_data.db"), False),
        storage_check("uploads", env_path("LOCALGPT_UPLOAD_DIR", "shared_uploads")),
        storage_check("LanceDB", env_path("LANCEDB_PATH", "lancedb")),
        storage_check(
            "overviews", env_path("LOCALGPT_OVERVIEW_DIR", "index_store/overviews")
        ),
    ]
    if not args.skip_ollama and os.environ.get("LLM_BACKEND", "ollama").lower() == "ollama":
        ollama_host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
        checks.append(probe("Ollama", f"{ollama_host}/api/tags"))

    passed = sum(checks)
    print(f"{passed}/{len(checks)} health checks passed")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())

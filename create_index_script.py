#!/usr/bin/env python3
"""Create a LocalGPT index through the public backend API."""

from __future__ import annotations

import argparse
import json
import os
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import requests

from localgpt_runtime import SUPPORTED_UPLOAD_EXTENSIONS


def _headers(token: str | None, json_body: bool = False) -> dict[str, str]:
    headers: dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if json_body:
        headers["Content-Type"] = "application/json"
    return headers


def _response_json(response: requests.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"LocalGPT returned HTTP {response.status_code}: {response.text[:500]}"
        ) from exc
    if not response.ok:
        raise RuntimeError(payload.get("error") or f"HTTP {response.status_code}")
    return payload


def load_options(config_path: str | None) -> dict[str, Any]:
    if not config_path:
        return {}
    with Path(config_path).open(encoding="utf-8") as config_file:
        payload = json.load(config_file)
    if not isinstance(payload, dict):
        raise ValueError("The build config must be a JSON object")
    return payload.get("processing_options", payload)


def validate_files(file_names: list[str]) -> list[Path]:
    files = [Path(name).expanduser().resolve() for name in file_names]
    if not files:
        raise ValueError("At least one document is required")
    for path in files:
        if not path.is_file():
            raise ValueError(f"Document does not exist: {path}")
        if path.suffix.lower() not in SUPPORTED_UPLOAD_EXTENSIONS:
            raise ValueError(f"Unsupported document type: {path.suffix}")
    return files


def create_index(
    *,
    api_url: str,
    token: str | None,
    name: str,
    description: str,
    files: list[Path],
    options: dict[str, Any],
) -> dict[str, Any]:
    base_url = api_url.rstrip("/")
    created = _response_json(
        requests.post(
            f"{base_url}/indexes",
            headers=_headers(token, json_body=True),
            json={"name": name, "description": description},
            timeout=30,
        )
    )
    index_id = str(created["index_id"])
    try:
        with ExitStack() as stack:
            multipart = [
                ("files", (path.name, stack.enter_context(path.open("rb"))))
                for path in files
            ]
            _response_json(
                requests.post(
                    f"{base_url}/indexes/{index_id}/upload",
                    headers=_headers(token),
                    files=multipart,
                    timeout=300,
                )
            )
        build = _response_json(
            requests.post(
                f"{base_url}/indexes/{index_id}/build",
                headers=_headers(token, json_body=True),
                json=options,
                timeout=900,
            )
        )
        return {"index_id": index_id, "build": build}
    except Exception:
        requests.delete(
            f"{base_url}/indexes/{index_id}",
            headers=_headers(token),
            timeout=30,
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", help="Documents to upload")
    parser.add_argument("--name", help="Index name")
    parser.add_argument("--description", default="")
    parser.add_argument("--config", help="JSON build options")
    parser.add_argument(
        "--api-url",
        default=os.environ.get("LOCALGPT_BACKEND_URL", "http://127.0.0.1:8000"),
    )
    parser.add_argument("--token", default=os.environ.get("LOCALGPT_API_TOKEN"))
    args = parser.parse_args()

    name = args.name or input("Index name: ").strip()
    if not name:
        parser.error("--name is required")
    file_names = args.files
    if not file_names:
        entered = input("Document paths (comma-separated): ").strip()
        file_names = [item.strip() for item in entered.split(",") if item.strip()]

    try:
        result = create_index(
            api_url=args.api_url,
            token=args.token,
            name=name,
            description=args.description,
            files=validate_files(file_names),
            options=load_options(args.config),
        )
    except (OSError, ValueError, RuntimeError, requests.RequestException) as exc:
        parser.exit(1, f"Index creation failed: {exc}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

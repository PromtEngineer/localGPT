#!/usr/bin/env python3
"""Build one or more indexes from JSON through the LocalGPT backend API."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import requests

from create_index_script import create_index, validate_files


def _index_specs(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(payload.get("indexes"), list):
        return payload["indexes"]
    return [payload]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="batch_indexing_config.json")
    parser.add_argument(
        "--api-url",
        default=os.environ.get("LOCALGPT_BACKEND_URL", "http://127.0.0.1:8000"),
    )
    parser.add_argument("--token", default=os.environ.get("LOCALGPT_API_TOKEN"))
    args = parser.parse_args()

    try:
        config_path = Path(args.config).expanduser().resolve()
        with config_path.open(encoding="utf-8") as config_file:
            payload = json.load(config_file)
        if not isinstance(payload, dict):
            raise ValueError("Batch config must be a JSON object")

        results = []
        for number, spec in enumerate(_index_specs(payload), start=1):
            if not isinstance(spec, dict):
                raise ValueError(f"Index entry {number} must be an object")
            name = spec.get("name") or spec.get("index_name")
            if not name:
                raise ValueError(f"Index entry {number} has no name")
            raw_documents = spec.get("documents") or []
            documents = [
                str((config_path.parent / document).resolve())
                if not Path(document).is_absolute()
                else document
                for document in raw_documents
            ]
            options = spec.get("processing_options") or spec.get("processing") or {}
            result = create_index(
                api_url=args.api_url,
                token=args.token,
                name=str(name),
                description=str(spec.get("description") or spec.get("index_description") or ""),
                files=validate_files(documents),
                options=options,
            )
            results.append(result)
            print(f"Built {name}: {result['index_id']}")
    except (OSError, ValueError, RuntimeError, requests.RequestException) as exc:
        parser.exit(1, f"Batch indexing failed: {exc}\n")

    print(json.dumps({"indexes": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

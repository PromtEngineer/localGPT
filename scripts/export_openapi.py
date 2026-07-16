#!/usr/bin/env python3
"""Export the authoritative FastAPI schema for client generation and review."""

from __future__ import annotations

import json
from pathlib import Path

from backend.api import app


OUTPUT = Path("Documentation/openapi.json")


def main() -> None:
    OUTPUT.write_text(
        json.dumps(app.openapi(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()

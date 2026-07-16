"""ASGI entry point for the LocalGPT backend.

Run with ``python -m backend.server`` for local development or point an ASGI
process manager at ``backend.server:app`` in production.
"""

from __future__ import annotations

import os

import uvicorn

from backend import api

app = api.app


def main() -> None:
    uvicorn.run(
        "backend.server:app",
        host=os.getenv("BACKEND_HOST", os.getenv("LOCALGPT_BACKEND_HOST", "127.0.0.1")),
        port=int(os.getenv("BACKEND_PORT", os.getenv("LOCALGPT_BACKEND_PORT", "8000"))),
        reload=False,
        access_log=False,
    )


if __name__ == "__main__":
    main()

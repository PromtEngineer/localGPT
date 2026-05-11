"""
Persistent conversion worker.

Stays alive between files so Docling ML models (loaded once on first call)
are reused across every file in an indexing run instead of being reloaded
from disk for each document.

Protocol (newline-delimited JSON over stdin/stdout):
  Request  -> {"file_path": "...", "document_id": "...", "config": {...}}
  Response <- {"chunks": [...]}  |  {"error": "...", "traceback": "..."}

The process exits cleanly when stdin reaches EOF (parent closed the pipe).
"""
from __future__ import annotations

import json
import sys
import traceback


def main() -> None:
    # Import here so the heavy Docling/model imports happen once per process.
    from rag_system.pipelines.indexing_pipeline import convert_and_chunk_document

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            req = json.loads(raw)
        except json.JSONDecodeError as e:
            _respond({"error": f"Bad JSON in request: {e}"})
            continue

        try:
            result = convert_and_chunk_document(
                req["file_path"],
                req["document_id"],
                req["config"],
            )
        except Exception:
            result = {"error": traceback.format_exc()}

        _respond(result)


def _respond(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, default=str) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()

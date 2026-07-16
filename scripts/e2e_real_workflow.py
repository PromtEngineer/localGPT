#!/usr/bin/env python3
"""Run a real LocalGPT upload -> embed -> index -> retrieve workflow.

The services and Ollama models must already be running. The script fails if it
does not receive an actual embedding vector, a grounded answer, citations, and
replayable durable-run events. Resources are deleted in a finally block.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import requests


def expect(response: requests.Response) -> dict[str, Any]:
    if not response.ok:
        raise RuntimeError(f"{response.request.method} {response.url}: {response.status_code} {response.text}")
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--document", type=Path, required=True)
    parser.add_argument("--generation-model", default="qwen3:0.6b")
    parser.add_argument("--embedding-model", default="qwen3-embedding:0.6b")
    parser.add_argument("--expected", default="AURORA-17")
    parser.add_argument("--token")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {args.token}"} if args.token else {}
    session_id = None
    index_id = None
    report: dict[str, Any] = {}
    try:
        health = expect(requests.get(f"{base}/health", headers=headers, timeout=30))
        report["health"] = health["status"]

        embedding = expect(
            requests.post(
                f"{base}/v1/embeddings",
                headers=headers,
                json={"model": args.embedding_model, "input": "embedding smoke test"},
                timeout=300,
            )
        )
        vector = embedding["data"][0]["embedding"]
        if len(vector) < 32:
            raise AssertionError("Embedding provider returned an implausibly small vector")
        report["embedding_dimensions"] = len(vector)

        session = expect(
            requests.post(
                f"{base}/sessions",
                headers=headers,
                json={"title": "Real E2E", "model": args.generation_model},
                timeout=30,
            )
        )
        session_id = session["session_id"]

        index = expect(
            requests.post(
                f"{base}/indexes",
                headers=headers,
                json={
                    "name": f"real-e2e-{int(time.time())}",
                    "description": "Ephemeral real workflow index",
                    "options": {
                        "embedding_model": args.embedding_model,
                        "enable_enrich": False,
                        "enable_docling_chunk": False,
                        "enable_latechunk": False,
                        "retrieval_mode": "hybrid",
                        "chunk_size": 256,
                        "chunk_overlap": 32,
                    },
                },
                timeout=30,
            )
        )
        index_id = index["index_id"]

        with args.document.open("rb") as handle:
            uploaded = expect(
                requests.post(
                    f"{base}/indexes/{index_id}/upload",
                    headers=headers,
                    files={"files": (args.document.name, handle, "text/plain")},
                    timeout=60,
                )
            )
        report["artifact_id"] = uploaded["uploaded_files"][0]["artifact_id"]

        built = expect(
            requests.post(
                f"{base}/indexes/{index_id}/build",
                headers={**headers, "Content-Type": "application/json"},
                json={},
                timeout=1200,
            )
        )
        if built.get("chunks_indexed", 0) < 1:
            raise AssertionError(f"No chunks indexed: {built}")
        report["chunks_indexed"] = built["chunks_indexed"]
        report["index_run_id"] = built["run_id"]

        expect(
            requests.post(
                f"{base}/sessions/{session_id}/indexes/{index_id}",
                headers=headers,
                timeout=30,
            )
        )

        run = expect(
            requests.post(
                f"{base}/v1/runs",
                headers=headers,
                json={
                    "session_id": session_id,
                    "message": "What is the calibration phrase for the Borealis instrument?",
                    "model": args.generation_model,
                    "force_rag": True,
                    "retrieval_k": 5,
                    "search_type": "hybrid",
                },
                timeout=30,
            )
        )
        run_id = run["id"]
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            current = expect(requests.get(f"{base}/v1/runs/{run_id}", headers=headers, timeout=30))
            if current["status"] in {"completed", "failed", "cancelled"}:
                break
            time.sleep(0.2)
        if current["status"] != "completed":
            raise AssertionError(f"Query run did not complete: {current}")
        answer = current["result"]["content"]
        citations = current["result"]["citations"]
        if args.expected.lower() not in answer.lower():
            raise AssertionError(f"Grounded answer omitted {args.expected!r}: {answer}")
        if not citations or not any(args.expected.lower() in str(item.get("text", "")).lower() for item in citations):
            raise AssertionError(f"Expected fact was not present in citations: {citations}")
        report["answer"] = answer
        report["citation_count"] = len(citations)
        report["query_run_id"] = run_id

        replay = requests.get(
            f"{base}/v1/runs/{run_id}/events",
            headers={**headers, "Last-Event-ID": "0"},
            timeout=60,
        )
        replay.raise_for_status()
        if "event: run.started" not in replay.text or "event: run.completed" not in replay.text:
            raise AssertionError(f"Durable event replay incomplete: {replay.text}")
        report["event_replay"] = True

        artifacts = expect(
            requests.get(f"{base}/v1/artifacts", headers=headers, params={"index_id": index_id}, timeout=30)
        )
        if not artifacts["artifacts"]:
            raise AssertionError("Uploaded artifact was not persisted")
        report["artifact_count"] = len(artifacts["artifacts"])
        print(json.dumps(report, indent=2))
    finally:
        if index_id:
            response = requests.delete(f"{base}/indexes/{index_id}", headers=headers, timeout=120)
            report["index_cleanup"] = response.ok
        if session_id:
            response = requests.delete(f"{base}/sessions/{session_id}", headers=headers, timeout=30)
            report["session_cleanup"] = response.ok


if __name__ == "__main__":
    main()

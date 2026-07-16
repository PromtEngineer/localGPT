#!/usr/bin/env python3
"""Evaluate grounded answers and citation recall for an indexed session."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--dataset", type=Path, default=Path("evals/grounded_retrieval.json"))
    parser.add_argument("--model", default="qwen3:0.6b")
    parser.add_argument("--token")
    args = parser.parse_args()
    headers = {"Authorization": f"Bearer {args.token}"} if args.token else {}
    rows = json.loads(args.dataset.read_text(encoding="utf-8"))
    results = []
    for case in rows:
        response = requests.post(
            f"{args.base_url.rstrip('/')}/v1/runs",
            headers=headers,
            json={
                "session_id": args.session_id,
                "message": case["question"],
                "model": args.model,
                "force_rag": True,
            },
            timeout=30,
        )
        response.raise_for_status()
        run_id = response.json()["id"]
        for _ in range(3000):
            run_response = requests.get(
                f"{args.base_url.rstrip('/')}/v1/runs/{run_id}",
                headers=headers,
                timeout=30,
            )
            run_response.raise_for_status()
            run = run_response.json()
            if run["status"] in {"completed", "failed", "cancelled"}:
                break
            time.sleep(0.2)
        result = run.get("result") or {}
        answer = str(result.get("content", ""))
        citation_text = "\n".join(str(item.get("text", "")) for item in result.get("citations", []))
        answer_hit = all(term.lower() in answer.lower() for term in case["expected_terms"])
        citation_hit = all(term.lower() in citation_text.lower() for term in case["citation_terms"])
        results.append({**case, "run_id": run_id, "answer_hit": answer_hit, "citation_hit": citation_hit})
    summary = {
        "cases": len(results),
        "answer_accuracy": sum(item["answer_hit"] for item in results) / max(1, len(results)),
        "citation_recall": sum(item["citation_hit"] for item in results) / max(1, len(results)),
        "results": results,
    }
    print(json.dumps(summary, indent=2))
    if summary["answer_accuracy"] < 1 or summary["citation_recall"] < 1:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

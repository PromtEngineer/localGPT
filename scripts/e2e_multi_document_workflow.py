#!/usr/bin/env python3
"""Adversarial real-model retrieval test over a heterogeneous document corpus."""

from __future__ import annotations

import argparse
import contextlib
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import requests


CORPUS = {
    "01_authoritative_calibration.md": """# Borealis Calibration Standard

Revision: 2026-06-14. Status: CURRENT AND AUTHORITATIVE.

The current calibration phrase for the Borealis instrument is **AURORA-17**.
This revision supersedes every earlier calibration memo. Operators must use the
current phrase during the preflight handshake and record the successful check.
The phrase is an identifier, not a password, and must be reproduced exactly.
""",
    "02_archived_calibration.md": """# Archived Borealis Calibration Memo

Revision: 2023-01-08. Status: ARCHIVED — DO NOT USE.

The former Borealis calibration phrase was POLARIS-09. This historical phrase
was retired and must not be used for current operations. Consult the current
calibration standard for the active value. This file remains only for audits.
""",
    "03_custodian.txt": """Borealis responsibility register

The designated custodian of the Borealis instrument is Dr. Mira Chen. She owns
the calibration log, approves maintenance access, and reviews handoff records.
The backup contact is Jordan Ellis, but the backup is not the designated
custodian. Escalations about ownership should name Dr. Mira Chen.
""",
    "04_maintenance.html": """<html><body><h1>Borealis maintenance schedule</h1>
<p>The instrument must be recalibrated every 42 days during normal service.</p>
<p>An impact event or sensor replacement triggers an additional immediate
recalibration, but it does not change the normal forty-two-day interval.</p>
</body></html>""",
    "05_asset_register.csv": """asset,serial,location,status
Borealis,BRL-2048,Lab North,active
Zephyr,ZPH-7781,Lab East,active
Orion,ORN-1102,Archive,retired
""",
    "06_operating_limits.json": json.dumps(
        {
            "instrument": "Borealis",
            "temperature_c": {"minimum": -20, "maximum": 55},
            "humidity_percent_max": 70,
            "note": "Operating limits do not define calibration identifiers.",
        },
        indent=2,
    ),
    "07_rollout_notice.eml": """From: operations@example.test
To: laboratory@example.test
Subject: Borealis revision rollout
Message-ID: <borealis-rollout@example.test>
Content-Type: text/plain; charset=utf-8

The current calibration standard takes effect on 2026-06-20. Archive older
printed memos and confirm that operators can locate the authoritative record.
""",
    "08_zephyr_calibration.md": """# Zephyr calibration card

The current calibration phrase for the Zephyr instrument is ZEPHYR-88. Zephyr
is a different instrument from Borealis. Values in this card must never be
substituted into Borealis procedures, even when both devices share a lab.
""",
    "09_orion_retirement.txt": """Orion retirement note

The Orion instrument was retired in 2024. Its historical handshake label was
ORION-END. This document does not describe Borealis or Zephyr calibration and
must not be used to answer questions about active equipment.
""",
    "10_glossary.html": """<html><body><h1>Operations glossary</h1>
<p>Custodian: the person accountable for records and controlled access.</p>
<p>Calibration phrase: an exact identifier used in a preflight handshake.</p>
<p>Recalibration interval: the normal elapsed time between scheduled checks.</p>
</body></html>""",
    "11_incident_history.csv": """date,instrument,event,resolution
2026-01-04,Borealis,power interruption,inspection completed
2026-02-19,Zephyr,sensor drift,recalibrated
2026-03-11,Borealis,case impact,immediate recalibration completed
""",
    "12_source_precedence.md": """# Source precedence policy

When records disagree, a document marked CURRENT AND AUTHORITATIVE takes
precedence over a document marked ARCHIVED or historical. Instrument names are
strict scope boundaries: a value for Zephyr or Orion cannot answer a Borealis
question. Answers should preserve exact identifiers, names, and intervals.
""",
}


CASES = [
    {
        "name": "lexical_current_vs_archived",
        "question": "What is the current calibration phrase for the Borealis instrument?",
        "search_type": "lexical",
        "retrieval_k": 4,
        "expected_terms": ["AURORA-17"],
        "expected_sources": ["01_authoritative_calibration.md"],
        "forbidden_terms": ["POLARIS-09", "ZEPHYR-88"],
        "forbidden_sources": [
            "02_archived_calibration.md",
            "08_zephyr_calibration.md",
        ],
    },
    {
        "name": "dense_paraphrase_custodian",
        "question": "Who has primary accountability for Borealis records and controlled access?",
        "search_type": "dense",
        "retrieval_k": 4,
        "expected_terms": ["Mira Chen"],
        "expected_sources": ["03_custodian.txt"],
        "forbidden_terms": ["POLARIS-09", "ZEPHYR-88"],
        "forbidden_sources": ["08_zephyr_calibration.md"],
    },
    {
        "name": "hybrid_numeric_schedule",
        "question": "Under normal service, how frequently should Borealis be recalibrated?",
        "search_type": "hybrid",
        "retrieval_k": 4,
        "expected_terms": ["42"],
        "expected_sources": ["04_maintenance.html"],
        "forbidden_terms": ["AURORA-17", "ZEPHYR-88", "Zephyr calibration card"],
        "forbidden_sources": ["08_zephyr_calibration.md"],
    },
    {
        "name": "multi_document_synthesis",
        "question": (
            "Give the current Borealis calibration phrase, its designated "
            "custodian, and the normal recalibration interval."
        ),
        "search_type": "hybrid",
        "retrieval_k": 6,
        "query_decompose": True,
        "max_citations": 15,
        "expected_terms": ["AURORA-17", "Mira Chen", "42"],
        "expected_sources": [
            "01_authoritative_calibration.md",
            "03_custodian.txt",
            "04_maintenance.html",
        ],
        "forbidden_terms": ["POLARIS-09", "ZEPHYR-88"],
        "forbidden_sources": ["08_zephyr_calibration.md"],
    },
    {
        "name": "instrument_scope_distractor",
        "question": "What is the current calibration phrase for Zephyr?",
        "search_type": "hybrid",
        "retrieval_k": 4,
        "expected_terms": ["ZEPHYR-88"],
        "expected_sources": ["08_zephyr_calibration.md"],
        "forbidden_terms": ["POLARIS-09", "AURORA-17", "Borealis calibration memo"],
        "forbidden_sources": [
            "01_authoritative_calibration.md",
            "02_archived_calibration.md",
            "03_custodian.txt",
            "04_maintenance.html",
        ],
    },
]


def expect(response: requests.Response) -> dict[str, Any]:
    if not response.ok:
        raise RuntimeError(
            f"{response.request.method} {response.url}: "
            f"{response.status_code} {response.text}"
        )
    return response.json()


def wait_for_run(base: str, headers: dict[str, str], run_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        run = expect(
            requests.get(f"{base}/v1/runs/{run_id}", headers=headers, timeout=30)
        )
        if run["status"] in {"completed", "failed", "cancelled"}:
            return run
        time.sleep(0.2)
    raise TimeoutError(f"Run did not finish: {run_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--generation-model", default="qwen3:0.6b")
    parser.add_argument("--embedding-model", default="qwen3-embedding:0.6b")
    parser.add_argument("--token")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {args.token}"} if args.token else {}
    session_id: str | None = None
    index_id: str | None = None
    report: dict[str, Any] = {"documents": len(CORPUS), "cases": []}

    with tempfile.TemporaryDirectory(prefix="localgpt-multidoc-") as directory:
        corpus_root = Path(directory)
        for filename, content in CORPUS.items():
            (corpus_root / filename).write_text(content, encoding="utf-8")

        try:
            session = expect(
                requests.post(
                    f"{base}/sessions",
                    headers=headers,
                    json={"title": "Multi-document E2E", "model": args.generation_model},
                    timeout=30,
                )
            )
            session_id = session["session_id"]
            index = expect(
                requests.post(
                    f"{base}/indexes",
                    headers=headers,
                    json={
                        "name": f"multi-document-e2e-{int(time.time())}",
                        "description": "Heterogeneous adversarial retrieval corpus",
                        "options": {
                            "embedding_model": args.embedding_model,
                            "enable_enrich": False,
                            "enable_docling_chunk": False,
                            "enable_latechunk": False,
                            "retrieval_mode": "hybrid",
                            "chunk_size": 96,
                            "chunk_overlap": 16,
                        },
                    },
                    timeout=30,
                )
            )
            index_id = index["index_id"]

            with contextlib.ExitStack() as stack:
                files = []
                for filename in CORPUS:
                    handle = stack.enter_context((corpus_root / filename).open("rb"))
                    files.append(("files", (filename, handle)))
                uploaded = expect(
                    requests.post(
                        f"{base}/indexes/{index_id}/upload",
                        headers=headers,
                        files=files,
                        timeout=120,
                    )
                )
            if len(uploaded["uploaded_files"]) != len(CORPUS):
                raise AssertionError(f"Upload count mismatch: {uploaded}")

            index_detail = expect(
                requests.get(f"{base}/indexes/{index_id}", headers=headers, timeout=30)
            )["index"]
            if len(index_detail["documents"]) != len(CORPUS):
                raise AssertionError("Index metadata did not retain all documents")

            built = expect(
                requests.post(
                    f"{base}/indexes/{index_id}/build",
                    headers={**headers, "Content-Type": "application/json"},
                    json={},
                    timeout=1800,
                )
            )
            report["chunks_indexed"] = built.get("chunks_indexed")
            report["index_run_id"] = built.get("run_id")
            if int(report["chunks_indexed"] or 0) < len(CORPUS):
                raise AssertionError(f"Expected at least one chunk per document: {built}")

            expect(
                requests.post(
                    f"{base}/sessions/{session_id}/indexes/{index_id}",
                    headers=headers,
                    timeout=30,
                )
            )

            for case in CASES:
                submitted = expect(
                    requests.post(
                        f"{base}/v1/runs",
                        headers=headers,
                        json={
                            "session_id": session_id,
                            "message": case["question"],
                            "model": args.generation_model,
                            "force_rag": True,
                            "retrieval_k": case["retrieval_k"],
                            "search_type": case["search_type"],
                            "context_window_size": 0,
                            **(
                                {"query_decompose": True}
                                if case.get("query_decompose")
                                else {}
                            ),
                        },
                        timeout=30,
                    )
                )
                run = wait_for_run(base, headers, submitted["id"])
                if run["status"] != "completed":
                    raise AssertionError(f"Case failed: {case['name']}: {run}")
                result = run["result"]
                answer = str(result["content"])
                citations = result["citations"]
                citation_docs = [
                    str(item.get("document_id") or "") for item in citations
                ]
                missing_terms = [
                    term
                    for term in case["expected_terms"]
                    if term.lower() not in answer.lower()
                ]
                missing_sources = [
                    source
                    for source in case["expected_sources"]
                    if not any(document.endswith(source) for document in citation_docs)
                ]
                distractor_terms = [
                    term
                    for term in case["forbidden_terms"]
                    if term.lower() in answer.lower()
                ]
                distractor_sources = [
                    source
                    for source in case.get("forbidden_sources", [])
                    if any(document.endswith(source) for document in citation_docs)
                ]
                expected_ranks = {
                    source: next(
                        (
                            rank
                            for rank, document in enumerate(citation_docs, start=1)
                            if document.endswith(source)
                        ),
                        None,
                    )
                    for source in case["expected_sources"]
                }
                rank_limit = (
                    3
                    if len(case["expected_sources"]) == 1
                    else case.get("max_citations", case["retrieval_k"])
                )
                poorly_ranked = [
                    source
                    for source, rank in expected_ranks.items()
                    if rank is None or rank > rank_limit
                ]
                case_report = {
                    "name": case["name"],
                    "search_type": case["search_type"],
                    "run_id": run["id"],
                    "answer": answer,
                    "citation_count": len(citations),
                    "citation_documents": citation_docs,
                    "expected_source_ranks": expected_ranks,
                    "missing_terms": missing_terms,
                    "missing_sources": missing_sources,
                    "distractor_terms": distractor_terms,
                    "distractor_sources": distractor_sources,
                    "poorly_ranked_sources": poorly_ranked,
                }
                report["cases"].append(case_report)
                if (
                    missing_terms
                    or missing_sources
                    or distractor_terms
                    or distractor_sources
                    or poorly_ranked
                    or len(citations)
                    > case.get("max_citations", case["retrieval_k"])
                ):
                    raise AssertionError(json.dumps(case_report, indent=2))

            artifacts = expect(
                requests.get(
                    f"{base}/v1/artifacts",
                    headers=headers,
                    params={"index_id": index_id},
                    timeout=30,
                )
            )["artifacts"]
            report["artifact_count"] = len(artifacts)
            if len(artifacts) != len(CORPUS):
                raise AssertionError("Artifact count did not match document count")
            report["passed"] = True
            print(json.dumps(report, indent=2))
        finally:
            if index_id:
                requests.delete(
                    f"{base}/indexes/{index_id}", headers=headers, timeout=120
                )
            if session_id:
                requests.delete(
                    f"{base}/sessions/{session_id}", headers=headers, timeout=30
                )


if __name__ == "__main__":
    main()

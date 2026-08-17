"""Scripted end-to-end smoke test (Phase 0.4).

Starts the two Python services as child processes against a throwaway SQLite
database and a throwaway LanceDB directory, drives them over HTTP exactly the
way the browser does, and asserts on the answers. Nothing touches the developer's
real ``backend/chat_data.db`` or ``./lancedb``.

  1. POST :8000/indexes                       create an index row
  2. POST :8000/indexes/<id>/upload           upload the Atlas-7 planted-fact PDF
  3. POST :8000/indexes/<id>/build            build it (delegates to :8001/index)
  4. POST :8000/sessions                      create a session
  5. POST :8000/sessions/<sid>/indexes/<iid>  link them
  6. POST :8000/sessions/<sid>/messages       4 planted-fact questions, non-streaming
       asserts: planted fact present in the answer, source_documents non-empty,
                a [Confidence: N%] tag on the answer, message_count == 2 * turns
  7. POST :8000/sessions/<sid>/messages/save  persist a streamed turn with steps+sources
  8. GET  :8000/sessions/<sid>                assert both round-trip out of SQLite

Teardown (children killed, temp dirs removed) runs even when an assertion fails
or the run is interrupted. Exit code 0 = every assertion passed, 1 = at least one
failed or the services never came up.

Pre-flight: if :8000 or :8001 is already accepting connections (e.g. the
developer's own stack), the run aborts before spawning anything — otherwise the
health checks would 200 against that pre-existing service and the smoke run
would POST uploads/builds/messages into the REAL ``backend/chat_data.db`` and
``./lancedb``.

    .venv/bin/python eval/smoke_e2e.py
"""

import argparse
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time

import requests

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(EVAL_DIR, ".."))
RUN_STARTED_AT = time.time()
TEST_PDF = os.path.join(EVAL_DIR, "corpora", "atlas7_service_manual.pdf")

BACKEND = "http://localhost:8000"
RAG_API = "http://localhost:8001"

EMBEDDER = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")

# (question, substring that must appear in the answer)
QUESTIONS = [
    ("What pressure does the brew boiler operate at during extraction?", "9.2"),
    ("Which sensor part should be replaced when error code E11 appears?", "TS-71"),
    ("How long is the Atlas-7 parts warranty?", "36"),
    ("Where is the serial number engraved?", "drip tray"),
]

CONFIDENCE_RE = re.compile(r"\[Confidence:\s*(\d+)%\]")

# The gateway stores uploads under <repo>/shared_uploads/, outside the temp dir
# it knows nothing about, so teardown has to clean them up explicitly.
UPLOADED_PATHS: list = []


class Results:
    """Collects one pass/fail line per assertion so the report is complete."""

    def __init__(self):
        self.rows = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.rows.append((name, bool(ok), detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  — ' + detail) if detail else ''}")
        return bool(ok)

    @property
    def failed(self):
        return [r for r in self.rows if not r[1]]

    def report(self) -> int:
        print("\n" + "=" * 72)
        print(f"{len(self.rows) - len(self.failed)}/{len(self.rows)} assertions passed")
        for name, ok, detail in self.rows:
            if not ok:
                print(f"  FAILED: {name} — {detail}")
        print("=" * 72)
        return 1 if self.failed else 0


def port_accepting_connections(port: int, host: str = "localhost") -> bool:
    """True when something is already listening on this port."""
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


def wait_for(url: str, timeout: float, proc: subprocess.Popen, log_path: str) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            print(f"    process exited early with code {proc.returncode}; see {log_path}")
            return False
        try:
            if requests.get(url, timeout=3).status_code == 200:
                # A 200 while the child is dead means the port belongs to someone
                # else's service — never declare that healthy.
                if proc.poll() is None:
                    return True
                print(f"    health 200 but the child exited with code "
                      f"{proc.returncode} — something else owns this port")
                return False
        except requests.RequestException:
            pass
        time.sleep(1.0)
    return False


def start_services(env: dict, log_dir: str, timeout: float):
    """Start the RAG API then the gateway; return (procs, log_paths, ok)."""
    procs, logs = [], {}

    for name, command, health in (
        ("rag-api", [sys.executable, "-m", "rag_system.api_server"], f"{RAG_API}/health"),
        ("backend", [sys.executable, "backend/server.py"], f"{BACKEND}/health"),
    ):
        log_path = os.path.join(log_dir, f"{name}.log")
        logs[name] = log_path
        handle = open(log_path, "w", encoding="utf-8")
        print(f"  starting {name} → {log_path}")
        proc = subprocess.Popen(
            command, cwd=REPO_ROOT, env=env,
            stdout=handle, stderr=subprocess.STDOUT,
            start_new_session=True,  # own process group, so teardown kills children too
        )
        procs.append((name, proc, handle))
        if not wait_for(health, timeout, proc, log_path):
            print(f"  ❌ {name} did not become healthy within {timeout:.0f}s")
            return procs, logs, False
        print(f"  {name} healthy")

    return procs, logs, True


def stop_services(procs) -> None:
    for name, proc, handle in reversed(procs):
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    proc.kill()
                proc.wait(timeout=10)
        print(f"  stopped {name} (exit {proc.returncode})")
        handle.close()


def run_smoke(results: Results, timeout: float) -> None:
    print("\n--- 1..3  index the planted-fact PDF over HTTP")
    resp = requests.post(f"{BACKEND}/indexes",
                         json={"name": "smoke-atlas7", "description": "Phase 0 smoke"},
                         timeout=60)
    resp.raise_for_status()
    index_id = resp.json()["index_id"]
    print(f"  index_id {index_id}")

    with open(TEST_PDF, "rb") as fh:
        upload = requests.post(f"{BACKEND}/indexes/{index_id}/upload",
                               files={"files": ("atlas7_service_manual.pdf", fh, "application/pdf")},
                               timeout=120)
    upload.raise_for_status()
    uploaded = upload.json().get("uploaded_files", [])
    UPLOADED_PATHS.extend(f["stored_path"] for f in uploaded if f.get("stored_path"))
    results.check("upload accepted the PDF", len(uploaded) == 1,
                  json.dumps(upload.json())[:160])

    build = requests.post(f"{BACKEND}/indexes/{index_id}/build",
                          json={"enable_enrich": False, "chunk_size": 512,
                                "embedding_model": EMBEDDER},
                          timeout=timeout)
    build_body = build.json()
    results.check("index build returned 200 with no error",
                  build.status_code == 200 and "error" not in build_body,
                  f"status={build.status_code} body={json.dumps(build_body)[:220]}")

    print("\n--- 4..5  session + link")
    session_id = requests.post(f"{BACKEND}/sessions", json={"title": "smoke"},
                               timeout=30).json()["session_id"]
    link = requests.post(f"{BACKEND}/sessions/{session_id}/indexes/{index_id}", timeout=30)
    results.check("index linked to session", link.status_code == 200, link.text[:160])
    print(f"  session_id {session_id}")

    print("\n--- 6  four planted-fact questions (non-streaming, force_rag)")
    for turn, (question, expected) in enumerate(QUESTIONS, start=1):
        label = f"q{turn}"
        chat = requests.post(f"{BACKEND}/sessions/{session_id}/messages",
                             json={"message": question, "force_rag": True, "verify": True},
                             timeout=timeout)
        if chat.status_code != 200:
            results.check(f"{label}: chat returned 200", False,
                          f"status={chat.status_code} body={chat.text[:200]}")
            continue
        body = chat.json()
        answer = body.get("response", "") or ""
        sources = body.get("source_documents") or []
        message_count = (body.get("session") or {}).get("message_count")

        results.check(f"{label}: planted fact '{expected}' in answer",
                      expected.lower() in answer.lower(),
                      f"{question!r} -> {answer[:200]!r}")
        results.check(f"{label}: source_documents non-empty",
                      len(sources) > 0, f"{len(sources)} sources")
        confidence = CONFIDENCE_RE.search(answer)
        results.check(f"{label}: [Confidence: N%] tag present",
                      confidence is not None,
                      f"tag={confidence.group(0) if confidence else 'absent'}")
        results.check(f"{label}: message_count == {2 * turn}",
                      message_count == 2 * turn, f"got {message_count}")

    print("\n--- 7..8  streamed-turn persistence round-trip")
    saved_user = "What is the descaling interval?"
    saved_answer = "Descaling must be performed every 60 days when water hardness exceeds 120 ppm."
    saved_sources = [{
        "chunk_id": "smoke-chunk-1",
        "text": "Descaling must be performed every 60 days when water hardness exceeds 120 ppm.",
        "document_id": "atlas7_service_manual.pdf",
        "chunk_index": 0,
        "score": 0.42,
    }]
    saved_steps = [
        {"type": "retrieval_started", "data": {"mode": "hybrid"}},
        {"type": "rerank_done", "data": {"count": 10}},
    ]
    save = requests.post(f"{BACKEND}/sessions/{session_id}/messages/save",
                         json={"user_message": saved_user, "assistant_message": saved_answer,
                               "source_documents": saved_sources, "steps": saved_steps},
                         timeout=60)
    results.check("messages/save returned 200", save.status_code == 200, save.text[:200])

    fetched = requests.get(f"{BACKEND}/sessions/{session_id}", timeout=60).json()
    messages = fetched.get("messages", [])
    assistant = next((m for m in messages
                      if m.get("sender") == "assistant" and m.get("content") == saved_answer), None)
    results.check("saved assistant message round-trips out of SQLite",
                  assistant is not None, f"{len(messages)} messages in session")

    metadata = (assistant or {}).get("metadata") or {}
    round_tripped_sources = metadata.get("source_documents") or []
    results.check("saved source_documents round-trip in metadata",
                  len(round_tripped_sources) == 1
                  and round_tripped_sources[0].get("chunk_id") == "smoke-chunk-1",
                  json.dumps(round_tripped_sources)[:200])
    round_tripped_steps = metadata.get("steps") or []
    results.check("saved steps round-trip in metadata",
                  [s.get("type") for s in round_tripped_steps]
                  == ["retrieval_started", "rerank_done"],
                  json.dumps(round_tripped_steps)[:200])

    expected_total = 2 * len(QUESTIONS) + 2
    results.check(f"final message_count == {expected_total}",
                  (fetched.get("session") or {}).get("message_count") == expected_total,
                  f"got {(fetched.get('session') or {}).get('message_count')}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--timeout", type=float, default=900.0,
                        help="per-request / per-service-startup timeout in seconds")
    parser.add_argument("--keep-temp", action="store_true", help="do not delete the temp dir")
    args = parser.parse_args()

    if not os.path.exists(TEST_PDF):
        print(f"missing test PDF: {TEST_PDF}")
        return 1

    # Pre-flight, BEFORE any child is spawned: if either port is already serving,
    # the health checks would pass against that pre-existing service and the run
    # would drive uploads/builds/messages into the developer's REAL
    # backend/chat_data.db and ./lancedb.
    for name, port in (("backend", 8000), ("rag-api", 8001)):
        if port_accepting_connections(port):
            print(f"❌ port {port} is already accepting connections — is your own "
                  f"{name} stack running? The smoke run would POST into IT, not "
                  f"into its throwaway children. Stop whatever listens on :{port} "
                  f"and re-run.")
            return 1

    temp_root = tempfile.mkdtemp(prefix="localgpt-smoke-")
    log_dir = os.path.join(temp_root, "logs")
    os.makedirs(log_dir, exist_ok=True)

    env = os.environ.copy()
    env.update({
        "EMBEDDING_MODEL": EMBEDDER,
        "DB_PATH": os.path.join(temp_root, "smoke_chat.db"),
        "LANCEDB_PATH": os.path.join(temp_root, "lancedb"),
        "RAG_API_URL": RAG_API,
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })

    print("localGPT end-to-end smoke")
    print(f"  embedder  {EMBEDDER}")
    print(f"  temp dir  {temp_root}")

    results = Results()
    procs = []
    started = time.time()
    try:
        procs, logs, ok = start_services(env, log_dir, args.timeout)
        if not ok:
            results.check("both services became healthy", False,
                          f"logs in {log_dir}")
        else:
            results.check("both services became healthy", True)
            run_smoke(results, args.timeout)
    except Exception as e:  # noqa: BLE001 — a crash here is a smoke failure, not a traceback
        results.check("smoke run completed without raising", False, f"{type(e).__name__}: {e}")
    finally:
        print("\n--- teardown")
        stop_services(procs)
        for path in UPLOADED_PATHS:
            try:
                os.remove(path)
                print(f"  removed upload {path}")
            except OSError as e:
                print(f"  could not remove upload {path}: {e}")
        # The overview builder writes to <repo cwd>/index_store/overviews/ (not
        # env-redirectable), so delete any overview files created during this run.
        overview_dir = os.path.join(REPO_ROOT, "index_store", "overviews")
        if os.path.isdir(overview_dir):
            for name in os.listdir(overview_dir):
                p = os.path.join(overview_dir, name)
                try:
                    if os.path.getmtime(p) >= RUN_STARTED_AT:
                        os.remove(p)
                        print(f"  removed leaked overview {p}")
                except OSError:
                    pass
        if args.keep_temp:
            print(f"  kept {temp_root}")
        else:
            shutil.rmtree(temp_root, ignore_errors=True)
            print(f"  removed {temp_root}")
        print(f"  wall clock {time.time() - started:.1f}s")

    return results.report()


if __name__ == "__main__":
    sys.exit(main())

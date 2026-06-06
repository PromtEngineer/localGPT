# LocalGPT Upgrade Implementation Plan

_Created: 2026-06-06_

This is the canonical plan for moving LocalGPT from its current split-service,
partially verified state to a release-ready application. A task is complete only
when its behavior is covered by a meaningful test and the relevant release check
passes.

## Review Baseline

Verified on 2026-06-06:

- `npm run lint:ui` passed.
- `npx tsc --noEmit` passed.
- `npm run lint` passed.
- `npm run build` passed.
- `.venv/bin/python -m pytest -q` passed: 28 tests.
- The system `python` still reports 22 passed and 6 errors because it does not
  have the `lancedb` package installed and imports the local `lancedb/` data
  directory as a namespace module. Project checks must use `.venv`.
- Direct ranking verification showed BM25-heavy and vector-heavy fusion settings
  return the same chunks in the same order; the existing fusion test does not
  assert that rankings change.
- `ruff` reported 987 errors.
- `black --check` would reformat 44 files and cannot parse
  `backend/server_old.py`.
- Docker Compose validation was not run because Docker is not installed in the
  review environment.

## P0: Index Integrity and Secret Safety

Target: no build may report a healthy index without validating its table, and no
provider secret may be persisted.

1. Redact `enrichApiKey` before writing job options to SQLite or shared in-memory
   job state. Store secrets only in an ephemeral, job-scoped secret store; resumed
   cloud jobs must request a key again or use a server environment variable.
2. Validate the existing LanceDB table on the incremental early-return path where
   every source file is unchanged.
3. Repair the LanceDB test/runtime environment and pin a compatible package
   version. Add a startup dependency check that fails with an actionable message.
4. Replace the current concatenation-based hybrid merge with score normalization
   and fusion by stable chunk identity. Assert that changing fusion weights changes
   ordering on controlled fixtures.
5. Make `search_type` (`vector_only`, `fts_only`, `hybrid`) and `dense_weight`
   affect actual retrieval behavior.
6. Centralize the LanceDB path and use it for creation, health checks, deletion,
   and maintenance.

Exit criteria:

- A build cannot become `completed` if its table is missing, empty, unreadable, or
  dimensionally incompatible.
- Persisted job rows never contain API keys.
- Hybrid, vector-only, and FTS-only tests pass and verify ranking semantics.

## P1: Request Isolation and API Reliability

Target: concurrent sessions cannot change one another's models, tables, or
retrieval settings.

1. Introduce request-scoped `ChatRequest` and `RetrievalOptions` objects.
2. Stop mutating the module-global RAG agent, retrieval config, retriever fusion
   config, and storage table name during requests.
3. Move chat and indexing orchestration behind Python service classes callable by
   FastAPI.
4. Replace blocking `requests` calls inside async endpoints with an async client or
   a bounded worker call, with connect/read timeouts and typed upstream errors.
5. Split cheap liveness from dependency readiness; keep container liveness free of
   Ollama model enumeration.
6. Use `try/finally` in metrics middleware so active-request counts recover after
   exceptions.
7. Propagate RAG/LLM failures as HTTP errors instead of saving error text as
   assistant messages.

Exit criteria:

- A concurrency test proves two sessions can use different indexes and settings
  without leakage.
- Every upstream network call has an explicit timeout.
- Liveness responds without Ollama or LanceDB access.

## P2: Consolidate the API Servers

Target: Next.js communicates with one FastAPI service; port 8001 is removed.

1. Inventory each route and callback in `rag_system/api_server.py`.
2. Extract pure indexing, retrieval, routing, and progress services without HTTP
   dependencies.
3. Add FastAPI routes that call those services directly while preserving current
   response and SSE contracts.
4. Switch the frontend and backend callers route by route, beginning with health,
   then index build/progress, then chat/streaming.
5. Run compatibility tests during a temporary dual-path period.
6. Remove `rag_system/api_server.py`, `api_server_with_progress.py`, port 8001
   configuration, and obsolete startup logic after parity is proven.

Exit criteria:

- One process owns API routing and lifecycle.
- No localhost HTTP hop exists between backend and RAG code.
- Index creation, resume, cancellation, chat, and streaming smoke tests pass.

## P3: Database, Upload, and Multi-Index Correctness

1. Route every SQLite connection through one helper that enables WAL, busy timeout,
   row factory, and `PRAGMA foreign_keys=ON`.
2. Add migrations and uniqueness constraints for session/index links and job-file
   identity; verify cascades with tests.
3. Stream uploads to temporary files, sanitize filenames, enforce per-file and
   request limits, validate target session/index existence, and clean up partial
   writes.
4. Define the product behavior for multiple indexes per session. Either retrieve
   across all linked tables with result fusion or enforce one active index in both
   schema and UI.
5. Add authentication or a local-only deployment guard for destructive maintenance
   and upload endpoints when binding beyond loopback.

## P4: CI, Cleanup, and Release Evidence

1. Remove or quarantine obsolete files such as `backend/server_old.py`; stop
   linting dead code as production code.
2. Establish a formatting baseline, then make `ruff`, `black`, and `mypy` pass
   locally and in CI.
3. Convert script-like tests that return booleans into real assertions and mark
   live-service tests explicitly.
4. Add index lifecycle integration tests: upload, build, health, unchanged rebuild,
   crash recovery, resume, delete, and orphan cleanup.
5. Qualify the README privacy claim: cloud enrichment sends document content to the
   selected provider.
6. Validate Docker Compose and startup scripts on Linux and macOS before release.

## Tracking Rules

- **Done**: behavior is implemented, tested, and its release check passes.
- **Partial**: implementation exists but behavior or integration is incomplete.
- **Blocked**: work cannot proceed without an external decision or dependency.
- **Unverified**: implementation exists but the required check has not been run.

`RELEASE_CHECKLIST.md` is the release gate. `Documentation/improvement_plan.md`
remains the feature roadmap and should link here for execution order.

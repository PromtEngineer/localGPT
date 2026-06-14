# LocalGPT Release Checklist

This file is the canonical release readiness checklist for LocalGPT. It consolidates the major delivery, documentation, CI, and maintenance items needed for a production release.

## 1. Documentation
- ✅ `README.md` reflects current runtime requirements (Python 3.11+, Node.js 18+).
- ✅ `CONTRIBUTING.md` matches the same runtime requirements.
- ✅ `.env.example` exists and is referenced in `README.md`.
- ✅ `RELEASE_CHECKLIST.md` is referenced from `README.md`.
- ✅ Core tracking docs are aligned:
  - `FEATURE_11_PHASE_1_SUMMARY.md`
  - `FEATURE_11_VALIDATION_CHECKLIST.md`
  - `MAINTENANCE_SUMMARY.md`
  - `MAINTENANCE_DELIVERY_CHECKLIST.md`
  - `Documentation/improvement_plan.md`
  - `Documentation/ui_manual_test_cases.md`

## 2. CI and Build Validation
- ✅ Frontend validation is configured in GitHub Actions:
  - `npm install`
  - `npm run lint:ui`
  - `npm run lint`
  - `npm run build`
- ✅ Backend integration tests for maintenance and job tracking APIs are present in `test_backend_api_contract.py`.
- ⚠️ Python linting is configured in CI (`ruff`, `black`, `mypy`), but the current codebase still contains failures that must be fixed before final merge.

## 3. Deployment and Environment
- ✅ `.env.example` is provided for runtime onboarding.
- ⚠️ Docker documentation should be reviewed for exact environment consistency after final dependency updates.
- ⚠️ Shell/script startup validation should be completed against the final runtime environment.
- ⚠️ Cloud provider key persistence should be audited in code and logs before release.

## 4. Job Tracking and Maintenance
- ✅ Job persistence and pipeline integration are documented.
- ✅ Maintenance tooling is delivered and documented.
- ✅ Frontend job progress/resume workflows and the maintenance health-report panel are present in the UI; manual validation steps are documented in `Documentation/ui_manual_test_cases.md` but the live pass remains to be run.
- ✅ The all-unchanged incremental path validation and post-build table validation are implemented and covered by backend contract tests.

## 5. Known gaps and deferred items
- ✅ Per-file progress display, resume/repair actions, and the maintenance health-report panel are implemented in `IndexPicker.tsx`; manual test cases for all three flows are documented in `Documentation/ui_manual_test_cases.md` (no headless-browser harness in CI yet, so these remain manual).
- ✅ FastAPI owns chat, SSE, and index execution in-process; the standalone port-8001 RAG API (`rag_system/api_server.py`, `api_server_with_progress.py`, `Dockerfile.rag-api`) has been deleted and all startup, frontend, MCP, eval, and Docker paths run through port 8000.
- ✅ Fusion weights and search modes affect retrieval behavior and have behavioral regression coverage.
- ✅ Generation model, embedding/fusion, Provence, table selection, and late-chunk settings are request-scoped; the global RAG-agent lock has been removed.
- ⚠️ Stream and sanitize uploads instead of reading up to 500 MB into memory.
- ⚠️ Enable SQLite foreign-key enforcement on every connection.
- ⚠️ Complete destructive-mode, fault-injection, and concurrency validation for maintenance operations.
- ✅ Add a central release checklist reference in the primary documentation index if one exists.

## 6. Release readiness signoff
- ⚠️ `pip install -r requirements.txt` should be run and verified in the target environment.
- ⚠️ `npm ci` should be run and verified.
- ✅ `npm run lint:ui` is configured and currently passing in workflow validation.
- ✅ `npm run build` is configured and currently passing in workflow validation.
- ⚠️ `python -m pytest test_hybrid_retrieval.py test_incremental_indexing.py test_logging_utils.py -v --tb=short` remains a recommended verification step.
- ⚠️ Review the high-level architecture and confirm service boundaries in `Documentation/system_overview.md`.
- ⚠️ Confirm `RELEASE_CHECKLIST.md` remains up to date before merge.

## Review Snapshot: 2026-06-13
- ✅ `npm run lint:ui`
- ✅ `npm run build`
- ✅ `.venv/bin/python -m pytest -q` (77 passed)
- ✅ Live index crash/restart/resume completed with 4/4 files and a healthy 108-row LanceDB table.
- ✅ Retrieval evaluation gate (100%)
- ✅ Live parallel mixed-model/mixed-search RAG requests returned HTTP 200 without global serialization
- ✅ Legacy live connectivity scripts are no longer misreported as pytest passes; run them explicitly when the required services are available.
- ⚠️ Project commands should be validated with a clean target environment; the current environment still imports local `lancedb/` under system Python.
- ⚠️ `ruff check rag_system/ backend/` currently reports lint failures.
- ⚠️ `black --check rag_system/ backend/` currently reports formatting issues.
- ⚠️ Docker Compose validation is pending due to environment availability.

See `Documentation/upgrade_implementation_plan.md` for the prioritized implementation sequence.

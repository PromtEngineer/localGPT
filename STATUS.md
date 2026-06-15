# LocalGPT — Project Status

**The single source of truth for delivery status, release readiness, and open
work.** It replaces the former scattered trackers (release checklist, Feature 10/11
summaries and checklists, upgrade/improvement plans). How-to and reference docs
live under [`Documentation/`](Documentation/) and in the `*_QUICK_START.md` files.

_Last reconciled against the code + gates: 2026-06-14._

---

## TL;DR

LocalGPT is a local-first RAG app: one unified FastAPI backend (port 8000) with
the RAG runtime in-process, a Next.js frontend (3000), LanceDB + SQLite + Ollama.
All automated gates are green and **no blocking code/test work remains**. The
only items left are an on-demand **manual browser pass** and a release-time
**Docker / clean-environment check on the target host** — both accepted gates
(see Decisions), not open code work.

## Verification snapshot (2026-06-14)

| Gate | Result |
|------|--------|
| Python tests (`pytest -q`) | ✅ 121 passed |
| Retrieval evaluation gate (`rag_eval.py gate`) | ✅ 100% |
| ruff / black / mypy (`rag_system/` + `backend/`) | ✅ clean (enforced in CI, pinned versions) |
| UI tests (`npm run test:ui`) | ✅ 11 passed (render + request-contract smokes) |
| `npx tsc --noEmit` / `npm run lint` / `npm run lint:ui` | ✅ clean |
| `npm run build` | ✅ passes |
| Live disposable index lifecycle (create→upload→preflight→build→SSE→diagnostics→RAG chat→delete) | ✅ |
| Live crash/restart/resume (4/4 files, healthy LanceDB table) | ✅ |
| Live parallel mixed-model/mixed-search RAG | ✅ HTTP 200, no global serialization |

---

## Delivered ✅

### Core architecture
- Unified FastAPI backend on **8000** with the RAG runtime in-process
  (`chat_runtime`, `indexing_runtime`); the standalone port-**8001** RAG API and
  its Docker service were **retired/deleted**.
- **Request-scoped** retrieval knobs (generation model, embedding/fusion,
  Provence, table, late-chunk); the global RAG-agent lock is gone, enabling
  concurrent chats.
- SQLite metadata + LanceDB vectors; CORS allow-list with credentials auto-off;
  loopback-default bind; `/maintenance/*` loopback-guarded.

### Retrieval / RAG
- Multi-collection retrieval (per-collection embedder, RRF fallback, cross-index
  rerank), hybrid dense + native FTS, AI reranking, contextual expansion,
  Provence pruning, query decomposition, late chunking.
- Typed metadata schemas + validated SQL filters.
- Agentic plan-and-execute mode (opt-in); MCP stdio server; hermetic eval gate.

### NVIDIA-blueprint-inspired features (opt-in, local-first)
- **Per-stage timing + TTFT** on chat responses (`LOCALGPT_TIMINGS`), incl. a
  generation span; works on streaming and non-streaming paths.
- **Two-axis self-reflection** (context-relevance + answer-groundedness, 0–2,
  bounded retry) with deterministic (temperature-0, thinking-off) scoring,
  best-answer fallback, budgeted regeneration, and a split judge/answer model.
- **Standalone multi-turn query rewrite** for retrieval.
- All three are surfaced in the chat **Settings → Reflection & Multi-turn**
  (toggles + reflection model, max loops, relevance/groundedness thresholds,
  defaults sourced from the backend), with a per-answer metrics footer.

### Persistent indexing jobs (Feature 11)
- Crash-recoverable, resumable indexing with per-stage tracking; SSE progress;
  resume/repair controls; post-build table validation; all-unchanged incremental
  path. Live kill/restart/resume validated. Reference:
  [`Documentation/persistent_indexing_jobs.md`](Documentation/persistent_indexing_jobs.md).

### Maintenance tools (Feature 10)
- Repair stuck builds, orphan-file/table sweeps (dry-run by default), broken-index
  cleanup, vacuum, health report, diagnostics export. Reference:
  [`Documentation/maintenance_tools.md`](Documentation/maintenance_tools.md),
  [`MAINTENANCE_QUICK_START.md`](MAINTENANCE_QUICK_START.md).
- Index deletion has owned-artifact removal, **shared-file preservation**,
  **rollback** (artifact failure keeps the DB row), **concurrent-delete** safety,
  and idempotency — all with regression coverage.

### Quality / security
- Python lint+type gates **enforced** in CI (ruff/black/mypy, pinned); frontend
  lint+tsc+`test:ui` in CI.
- `enrichApiKey` is stripped before any DB write/log/response (verified);
  metadata-filter SQL is allow-listed + escaped; upload filenames sanitized,
  size/type-limited, partial batches rolled back.
- SQLite foreign keys enabled on every **writer** connection (core DB, job
  persistence, maintenance, incremental indexer, PDF store). Centralized
  `row_factory`.

### Recently closed (2026-06-14 review)
- **Job-timeline timezones unified to UTC** — `backend/database.py` job-timestamp
  writes now match `job_persistence`/`maintenance` (UTC); fixes negative/out-of-order
  stage durations.
- **Diagnostics export reads its JSON body** (was silently-ignored query params).
- **`vacuum-database` gained a `dry_run` preview** (fragmentation / reclaimable
  space / would-vacuum, no DB change).
- **Total upload-batch size cap** — `_save_uploads` now enforces a 2 GB
  whole-batch limit (on top of the 500 MB per-file limit), with the same
  mid-batch rollback. Tested.
- **Maintenance fault/concurrency coverage** — added tests for LanceDB
  orphan-table **execute-mode** cleanup (drops orphans, preserves known tables)
  and **concurrent maintenance sweeps** (4 sweeps on one SQLite DB, no
  errors/corruption under WAL + busy-timeout).

---

## Open / remaining

**None blocking.** Every code/test item is closed (above). The two items below
are accepted gates that fundamentally require a human tester or a target host —
automated coverage stands in for the logic; the irreducible step is recorded as
a decision below.

## Decisions / accepted (not open bugs)

- **Manual browser acceptance → accepted manual gate.** The resume/repair and
  maintenance-report flows are covered by component render smokes
  (`npm run test:ui`), backend contract tests, and the live index lifecycle +
  crash/resume runs. The only uncovered step is a human clicking through
  `Documentation/ui_manual_test_cases.md`; no headless-browser harness in CI and
  the Browser-Use bridge isn't available here, so this is run on demand by a
  person, not blocking.
- **Clean-environment / Docker validation → deferred to the target host.**
  `docker` isn't available in this environment, so a fresh `pip install`/`npm ci`/
  `docker compose` run can't be done here; it's a release-time check on the
  deployment machine. Compose files and requirements are in the repo and
  syntactically maintained.
- **Small-model synthesis fallback → accepted quirk.** `qwen3:0.6b` once appended
  the "could not find that information" fallback after a correct answer;
  `qwen3:8b` was clean. The recommended synthesis models don't exhibit it; a
  narrow strip-the-exact-sentence guard can be added on request if tiny synthesis
  models are used heavily.
- **Orphan cleanup → run on demand.** A 2026-06-14 dry-run found ~11 orphan
  uploads and ~10 orphan LanceDB tables. Tooling exists
  (`/maintenance/remove-orphan-files`, `/maintenance/remove-orphan-tables`, both
  now with verified execute-mode + concurrency tests); execution is left to an
  explicit, confirmed run since it deletes real files/tables.

---

## Tracking rules

- **Done ✅** — implemented, tested, and its gate passes.
- **Open ⚠️** — implementation exists but a behavior/integration/validation step
  remains.
- **Decision** — reviewed and deliberately not changed (rationale recorded above).

## Reference docs (not status)

Architecture & guides live under [`Documentation/`](Documentation/):
`system_overview.md`, `architecture_overview.md`, `api_reference.md`,
`indexing_pipeline.md`, `retrieval_pipeline.md`, `maintenance_tools.md`,
`persistent_indexing_jobs.md`, `ui_manual_test_cases.md`, install/deploy/docker
guides. Usage quick-starts: `MAINTENANCE_QUICK_START.md`,
`PERSISTENT_JOBS_QUICK_START.md`.

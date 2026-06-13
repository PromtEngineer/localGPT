# UI Manual Test Cases — Index Build Monitoring & Maintenance

This document covers manual (and, where noted, end-to-end) test cases for the
three frontend flows tracked under "Complete UI integration":

1. Per-file indexing stage display
2. Resume / repair actions
3. Maintenance health report display

All three live in `src/components/IndexPicker.tsx` and are reachable from the
chat UI's "Select an index" picker (the `…` row menu and the header buttons).

## Setup

- Start the unified backend (`http://localhost:8000`) and the frontend
  (`npm run dev`, `http://localhost:3000`).
- Have at least one index with several documents so a rebuild produces
  multiple per-file rows, and ideally one index in a `warning`/`unhealthy`
  state (e.g. with a missing source file) for the diagnostics cases.

## 1. Per-file indexing stage display

| # | Steps | Expected result |
|---|-------|-----------------|
| 1.1 | Open the index picker, select an index with files, choose "Rebuild changed only" or "Force rebuild" from its row menu. | A progress section appears with the job stage, a progress bar, and the overall percentage (`buildJob.stage` / `buildJob.progress`). |
| 1.2 | While the build runs, click "Show file details". | A panel lists each file with its filename, current stage (e.g. `Stage: embedding`), and a color-coded status badge (`pending` / `processing` / `done` / `failed` / `skipped`). |
| 1.3 | Wait for at least one file to finish, and (if possible) make one fail, e.g. by pointing the index at an unreadable file. | Completed files show "Chunks generated: N"; failed files show a red "Error: …" line (`formatFileDetails`). |
| 1.4 | Toggle "Show file details" closed and back open while the build is still running. | The panel hides/shows instantly without an extra fetch — file data comes from the polled `buildJob`, not a separate request. |
| 1.5 | Let the build finish. | The progress section and file-details panel disappear, a completion summary alert appears (`formatBuildSummary`), and the index list / health badges refresh. |

## 2. Resume / repair actions

| # | Steps | Expected result |
|---|-------|-----------------|
| 2.1 | Start a background build, then stop the backend process mid-run, restart it, and reopen the picker. | `_recover_stale_index_builds()` runs on startup and flips the crashed job to `paused`; the picker shows the paused job with a "Build paused. Resume to continue indexing." notice. |
| 2.2 | With the `paused` job visible, click "Resume rebuild". | `handleResumeBuild` calls `POST /index-jobs/{job_id}/resume` (→ `JobProgressTracker.mark_job_resuming`); the status moves to `queued`/`running`, the message updates to "Resuming build…", and previously-completed files are **not** reprocessed. |
| 2.3 | While a build is active, click "Cancel rebuild". | A cancel request is sent; the job message updates immediately and the status eventually reports `cancelled`. |
| 2.4 | Open the row menu (`…`) for any index and choose "Run diagnostics". | An alert summarizes health, file/vector counts, recommended action, and any errors/warnings (`formatDiagnostics`). |
| 2.5 | Choose "Diagnose + repair" on an index flagged `warning` or `unhealthy`. | If `can_repair` is true, a confirm dialog proposes a rebuild (incremental or `force_rebuild`, based on `recommended_action`); accepting runs `rebuildInBackground` and refreshes diagnostics on completion. If `can_repair` is false, only the diagnostic details are shown. |
| 2.6 | Click directly on an index name while it is `unhealthy`. | `handleOpenIndex` blocks the open, explains why, and offers to run diagnose + repair (or tells you to fix the source files first when repair isn't possible). |

## 3. Maintenance health report display

| # | Steps | Expected result |
|---|-------|-----------------|
| 3.1 | Open the index picker and click "Maintenance report" in the header (next to "Refresh health"). | A panel opens below the search bar, first showing "Loading maintenance health report…", then the report body (`GET /maintenance/index-health`). |
| 3.2 | Inspect the panel header. | Shows "Maintenance health report" and an "As of `<timestamp>`" label from the report's `timestamp` field. |
| 3.3 | Inspect the summary line. | Shows aggregate counts in the form "`X` healthy · `Y` warning · `Z` unhealthy (`N` total)" from `report.summary`. |
| 3.4 | Inspect the per-index rows. | Each row shows the index name, a color-coded health badge (green / yellow / red for healthy / warning / unhealthy), document count, metadata status, and — when present — the latest job's status and error. |
| 3.5 | Click "Maintenance report" again. | The panel collapses without re-fetching (`showHealthReport` toggles; the cached report stays in state for instant re-open). |
| 3.6 | Stop the backend (or temporarily disable `maintenance_tools`) and click "Maintenance report". | An alert reports the fetch failure ("Maintenance health report error: …") and the panel does not stay open in a stuck loading state. |

## End-to-end coverage notes

- Sections 1 and 2 share ground with the Feature #11 crash/resume scenario in
  `FEATURE_11_VALIDATION_CHECKLIST.md`; the automated counterpart for the
  resume-classification logic lives in
  `test_backend_api_contract.py::test_job_progress_tracker_mark_job_resuming_identifies_incomplete_files`
  and `test_mark_job_resuming_reports_missing_job`.
- Section 3 has no automated UI test yet — there is no headless-browser
  harness in CI (`frontend` job runs `lint`/`build` only). Treat the manual
  checks above as the acceptance criteria for `getMaintenanceHealthReport` /
  the "Maintenance report" panel until one exists.
- `chatAPI` surfaces fetch failures via `alert()`/`console.warn()`, which are
  easy to miss in a recording — keep the browser console and Network tab open
  while running these flows.

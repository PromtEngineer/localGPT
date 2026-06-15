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
| Python tests (`pytest -q`) | ✅ 131 passed |
| Retrieval evaluation gate (`rag_eval.py gate`) | ✅ 100% |
| ruff / black / mypy (`rag_system/` + `backend/`) | ✅ clean (enforced in CI, pinned versions) |
| UI tests (`npm run test:ui`) | ✅ 11 passed (render + request-contract smokes) |
| `npx tsc --noEmit` / `npm run lint` / `npm run lint:ui` | ✅ clean |
| `npm run build` | ✅ passes |
| Live disposable index lifecycle (create→upload→preflight→build→SSE→diagnostics→RAG chat→delete) | ✅ |
| Live crash/restart/resume (4/4 files, healthy LanceDB table) | ✅ |
| Live parallel mixed-model/mixed-search RAG | ✅ HTTP 200, no global serialization |
| Docker compose static validation + `pip check` | ✅ both compose files valid; deps consistent |

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

### Closed in the 2nd audit round (2026-06-14)
- **Job `finished_at`/in-memory `updated_at` were still local** — the UTC fix in
  `database.py` was undermined by `server.py` writing these `index_jobs` fields
  with `datetime.now()`. All `index_jobs` timestamp writers now use
  `_utc_now_iso()`, so the whole timeline (created/updated/finished + persistence
  stages) is naive-UTC.
- **`reflection_max_loops` is clamped** to a hard ceiling (5) — an unbounded
  request value was a resource-exhaustion lever (each loop is a full
  retrieval+generation). Tested.
- **Reflection metadata gained `converged`** — when `max_loops` is hit the
  reported relevance/groundedness describe the last-evaluated (possibly
  pre-final-regeneration) answer; `converged` now flags whether the thresholds
  were actually met. Tested.
- **Docker/clean-env: static portion validated** — both compose files parse with
  valid services + present Dockerfiles, and `pip check` reports a consistent
  dependency graph. (A fresh-host `docker compose up` / clean install still needs
  the deployment machine — see Decisions.)

### Closed in the 3rd audit round (frontend + indexing pipeline)
- **Frontend: cross-session state clobber** — the post-`complete` session refresh
  ran on a 100 ms `setTimeout` with a stale `activeSessionId`; switching sessions
  in that window overwrote the now-current session with the just-finished one.
  Now guarded by the stream's abort signal.
- **Frontend: regenerate could spawn a second session** — `sendMessage` read the
  `sessionId` prop, which lags behind a brand-new chat; regenerating before it
  propagated created a duplicate session. Now falls back to `currentSession?.id`.
- **Frontend: step-handler crash on mixed events** — `sub_query_result` /
  `final_answer` indexed a fixed 8-step array that `direct_answer` replaces with
  one element; guarded against a malformed/mixed stream.
- **Indexing: NoneType deref when the docling worker restart fails** —
  `_convert_via_worker` now raises a clean per-file `RuntimeError` instead of
  `AttributeError` on `None.stdin`, so the batch keeps going. Tested.
- **Indexing: duplicate vectors on resume** — the pre-append delete was gated on
  `incremental and not force_reindex`, so a forced/non-incremental build that
  crashed between the LanceDB append and the SQLite stage commit would re-append
  on resume. Storage is now always delete-then-insert (idempotent; the delete
  no-ops when nothing's present).
- **Indexing: negative stage durations** — `complete_stage`/`fail_stage` now
  clamp `duration` to ≥ 0 (clock skew between connections could make it negative).

### Closed in the 4th audit round (retrieval / agent / security)
- **Security: unbounded chat knobs (local DoS)** — `retrieval_k` /
  `reranker_top_k` / `context_window_size` reached LanceDB unclamped, so a
  request like `retrieval_k=10_000_000` could OOM the host. Now clamped
  (≤500 / ≤200 / ≤10) and coerced; the `query` is length-capped. Tested.
- **Security: diagnostics `output_path` traversal** — the request-supplied
  `output_path` was `mkdir`'d unvalidated (arbitrary directory write / `..`
  traversal, loopback-guarded). Now contained to the project root. Tested.
- **Agent: verifier could crash a successful request** — a non-int
  `confidence_score` or any non-timeout verifier exception propagated and
  discarded an already-computed answer. Confidence is now coerced+clamped and
  all verifier failures degrade to "skip annotation."
- **Agent: hallucinated answer when every sub-query fails** — the
  compose-from-sub-answers branch ran with empty evidence; now returns the
  "could not find" fallback when there are no sub-answer sources.
- **Agent: lost-update on conversation history** — history was a request-start
  snapshot mutated locally then written back whole, so two concurrent
  same-session chats clobbered each other's turns. Both write sites now
  append under the lock to the latest stored history.

The audit also confirmed safe: metadata-filter SQL allow-listing/escaping, the
model string (Ollama doesn't auto-pull → no SSRF), `table_name` (a table
lookup, not a SQL identifier), the MCP server, the `/maintenance/*` loopback
guard, that per-request retrieval overrides do NOT leak into shared pipeline
state, and that decomposition parallelism is bounded (≤10 sub-queries, pool of
≤3).

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
- **Clean-environment / Docker validation → static done, runtime deferred to the
  target host.** Compose files are validated (parse, services, present
  Dockerfiles) and `pip check` confirms a consistent dependency graph. `docker`
  isn't available here, so the actual `docker compose up` + fresh
  `pip install`/`npm ci` is a release-time check on the deployment machine.
- **Small-model synthesis fallback → accepted quirk.** `qwen3:0.6b` once appended
  the "could not find that information" fallback after a correct answer;
  `qwen3:8b` was clean. The recommended synthesis models don't exhibit it; a
  narrow strip-the-exact-sentence guard can be added on request if tiny synthesis
  models are used heavily.
- **Within-collection base/late-chunk dedup → fixed ✅ (0930e0d).** A collection's
  base and `_lc` legs can both return the same chunk, double-counting one passage
  in RRF / reranking / synthesis. Added `_dedup_within_collection()` keyed on
  `(_source_table, chunk_id)` — deliberately the pair, since `chunk_id` is only
  unique within a collection (`<document_id>_<chunk_index>`), so the same
  `chunk_id` from two indexes is distinct content and is kept. Validated against
  the retrieval eval gate (still 100%) plus `RetrievalDedupTests` and the existing
  `MultiCollectionRetrievalTests` cross-index guard. Two related items stay
  deferred (same "fix deliberately with eval coverage" bucket): the multi-collection
  routing cache ignores conversation history, and a mixed int-vs-float schema
  across collections can silently narrow a list filter.
- **Overview written before storage → minor, transient.** A file's overview
  (triage/routing data) is appended before its embed/store stages, so a file that
  fails at storage can briefly have an overview but zero vectors. This only
  persists while the file is in a **failed** build state (user-visible; the file
  is retried on the next run, which re-creates its vectors). Fixing it "properly"
  means reordering the tracked stages (overview after storage), which is invasive
  for the crash/resume path — deferred as a deliberate, separately-tested change.
- **Orphan cleanup → run on demand.** A 2026-06-14 dry-run found ~11 orphan
  uploads and ~10 orphan LanceDB tables. Tooling exists
  (`/maintenance/remove-orphan-files`, `/maintenance/remove-orphan-tables`, both
  now with verified execute-mode + concurrency tests); execution is left to an
  explicit, confirmed run since it deletes real files/tables.

---

## External pattern review — Brev "build-an-agent" workshop + NVIDIA report-gen blog

Two sessions independently reviewed `brevdev/workshop-build-an-agent` and the
NVIDIA "report generator agent on OpenRouter" blog for ideas worth importing.
Both converged on the same thesis; this is the reconciled, canonical decision.

**Decision: treat them as a pattern catalogue, not a framework migration.**
LocalGPT already implements ~70% of what they teach (ReAct/agentic loop,
agentic RAG, query rewrite, 2-axis reflection/verifier, and an e2e eval harness
in [`rag_eval.py`](rag_eval.py) that already scores groundedness, context
relevancy, judge-accuracy, and citation validity). The NVIDIA/cloud/GPU stack
(LangGraph, DeepAgents, FAISS, NIM/Nemotron, OpenRouter, Tavily, GRPO/NeMo) is a
**poor compatibility match** and is **not adopted** — it conflicts with the
local-first, Ollama/LanceDB/in-process runtime and the pinned deps
(`torch==2.4.1`, `transformers==4.51.0`). **Licensing caveat:** the workshop repo
has no visible LICENSE — borrow architecture/ideas, do **not** copy code.

**Adopt selectively (candidates, in priority order):**

1. **`feat/eval-suite`** *(High compat / Low risk)* — **DELIVERED ✅** (merged
   from branch `feat/eval-suite`, 3 commits). Extended the existing harness
   (no RAGAS/LangChain): question **categories + difficulty** with per-group
   recall/chunk-hit breakdowns; per-case **failure attribution** (first-cause,
   retrieval + e2e) dumped to `results-*-cases.jsonl`; a **helpfulness** judge
   axis; opt-in **per-index regression baselines** (`--save-baseline` /
   `--compare-baseline --tolerance`, exit 1 on a real drop); and a
   **config-comparison** command (`compare`, dense-weight sweep tabulated with
   per-column winners). Fixture gate stayed 100% throughout; suite 131 → 144;
   no app/runtime code touched. *Deferred to `feat/data-policy`:* the
   prompt-injection/safety eval set (better owned alongside the policy layer).
2. **`feat/data-policy`** *(High compat / Medium risk).* A provider-neutral
   PII/secret classifier + policy decision (local / cloud / block / confirm)
   before any external request. **Reframed:** this governs an egress that
   **already exists today** — the optional `enrichApiKey` cloud-enrichment path —
   so it is retroactive governance, not future-proofing. Fail-closed; index-level
   policies; audit records; no-cloud-by-default fixtures.
3. **`feat/tool-events`** *(High compat / Medium risk).* A generic SSE
   tool/activity event contract (index selection, rewrite, retrieve, rerank,
   reflect, verify, optional web/skill) over the existing `event_callback` in
   [`chat_runtime.py`](rag_system/chat_runtime.py). Surface as a compact
   expandable activity view — not the workshop's drag-and-drop agent builder.
4. **`feat/research-mode` — split by data source.** A long-form report workflow
   (plan sections → retrieve evidence → dedupe → draft → verify citations →
   compile), request-gated like timings/reflect/rewrite, reusing the reflection
   budgeting (max-loops/regen clamps). **Local-only variant can ship early**
   (no external network → no policy dependency); the **web-augmented variant is
   gated behind `feat/data-policy`**, web disabled by default, with `[Local]` vs
   `[Web]` citation labelling.
5. **`feat/session-skills`** *(prototype carefully).* Markdown "skill packs"
   (report writing, contract compare, incident analysis, etc.) implemented as
   **prompt modules only — never executable plugins.** Allowlisted directory,
   metadata schema, size limits, versioning, explicit user selection; skill text
   is treated as privileged instructions and arbitrary uploaded Markdown is never
   loaded as a skill.

**Do not adopt:** LangGraph/DeepAgents migration, FAISS (we use LanceDB),
NIM/Nemotron/OpenRouter/Tavily as defaults, GRPO training + NeMo synthetic data,
and the drag-and-drop agent builder (product-fit mismatch).

**Contingent — not on the roadmap** (revisit only if the prerequisite exists):
human-approval/checkpoint for side-effect tools (only once such tools exist);
Docker/OpenShell kernel sandboxing (only if shell/file-exec tools are added — we
don't execute untrusted code today); MCP **client** for external tools (defer
until governance lands — note we are already an MCP **server** via
[`mcp_server.py`](rag_system/mcp_server.py), a different trust direction).
Persistent agentic-query checkpoints are **low value** — `job_persistence.py`
already checkpoints indexing in SQLite and retrieval queries are short-lived.

**Status:** analysis only — no LocalGPT files changed for this review. Each
candidate above should land as an independent worktree behind its own gate; do
not combine policy + web + skills + tool-exec into one migration.

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

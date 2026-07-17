# Status — overnight build, July 16, 2026

## What exists now

- **DESIGN.md** — the researched system design (July 2026 landscape).
- **RESULTS.md** — full validation. Single-session headline: 53/60 = 88.3% agentic across
  4 domains (research 93.3, financial 86.7, legal 100, health-unseen 73.3); tri-hybrid+rerank
  retrieval 84–97% hop recall. Read its variance note before comparing any two scores:
  determinism is per-session, cross-session drift is ~1 question/domain, so only same-session
  A/B counts and differences ≤2 questions at n=15 are noise.
- **ui/mock.html** — mock workbench UI (sessions, multi-source scope, upload/ingest, tool
  trace, page evidence) on real corpus numbers.
- **Three datasets** (`data/raw/`): research_papers (25), financial_docs (24), legal_docs (25)
  — all multimodal, manifest.json each, all publicly sourced.
- **Three QA benchmarks** (`data/benchmarks/`): 15 verified multi-hop questions each with
  gold doc/page/modality hops (absolute PDF pages, visually verified).
- **Working pipeline** (`src/marag/`): ingest → hybrid index (LanceDB dense+FTS) → visual
  late-interaction index (ColModernVBERT) → RRF fusion → rerank → router → single-shot +
  iterative search agent (hybrid_search/grep/read_doc/list_docs/list_tables/sql tools) →
  LLM-judge + citation-grounding eval. Unit tests in `tests/`.

## How to run

```bash
# serving (dedicated server — the Ollama app caps context at 16K):
OLLAMA_HOST=127.0.0.1:11435 OLLAMA_CONTEXT_LENGTH=65536 ollama serve &

uv run marag status
uv run marag agent "Which grew data-center revenue faster, NVIDIA or AMD?" financial_docs
uv run marag eval-retrieval financial_docs
uv run marag eval-answers financial_docs --mode agentic
```

## Operational gotchas (learned the hard way — see RESULTS.md bug table)

1. One GPU-heavy process at a time on Apple Silicon (MPS deadlocks otherwise).
2. Reasoning models + small max_tokens = empty output. Use `LLM.chat(reasoning="none")`
   for trivial calls, ≥3K budgets otherwise.
3. Always verify `ollama ps` CONTEXT column — the app default (16K) silently truncates
   agent loops.
4. `HF_HUB_OFFLINE=1` + cached snapshots keeps everything running through HF outages.

## Workbench UI + API (`marag serve`)

`marag serve` runs a FastAPI backend + wired web UI at http://127.0.0.1:8000 covering the
three things asked for — multiple chat sessions (SQLite-persisted), source selection across
indices with live scope, and upload/create that runs ingestion — plus a streaming agent tool
trace and citation→page-evidence panel. Verified end-to-end in the browser against real data
(sessions persist and reload, `[doc pN]` opens the real rendered page + Docling tables, SSE
answer stream). `src/marag/server/`; static reference design in `ui/mock.html`.

## Formats &amp; document-level ops (July 17)

- **Beyond PDF**: ingest now dispatches by extension. DOCX/PPTX/HTML/MD go through Docling
  (markdown + tables, split into ~800-token pseudo-pages; no visual channel — these have no
  canonical layout). CSV/XLSX/Parquet register their **full sheets into DuckDB** as `sql`
  views plus one searchable profile card per sheet (columns, types, ranges, sample rows, and
  the exact view name) — so spreadsheets plug straight into the numbers-via-sql path rather
  than being embedded as raw grids. Verified end-to-end on a CSV+XLSX+MD corpus: the agent
  answered a cross-file question via sql (Q4 cloud+ads = $5,620M, APAC 28% YoY — both exact).
- **Whole-document ops**: `summarize_doc(doc_id)` — cached map-reduce over every chunk (utility
  model maps section notes, orchestrator reduces). This is the document-level primitive top-k
  retrieval can't provide. `marag summarize <ds> <doc>` / exposed as an agent tool; the router
  sends "summarize/overview" requests to the agentic path.
- **Multi-index retrieval (now real)**: with several sources in scope the agent searches ALL
  of them — `search_multi` pools each index's candidates and reranks across the combined set;
  grep/read/summarize/view_page resolve each doc to its owning source; `sql` exposes every
  in-scope source's tables in one flat namespace (globally-unique view names), so a single
  query can join a financial table and a health table. Verified: `hlt=33, fin=18` from one
  cross-source query. The API `/api/ask` passes the full scope; `/api/resolve/{doc}` maps a
  citation to its source so evidence opens correctly from any index.
- **Still fuzzy**: spreadsheet merged-cell headers rely on pandas' read (fine for clean data
  files); list_tables caps at 120 rows so a giant source can crowd out others in the unfiltered
  listing (per-doc list_tables is unaffected).

## Opt-in capabilities (off by default; default path byte-identical to validated)

- **Dense vision reader** — `models.vision` + `agent.view_page_auto_zoom` + `view_page_thinking`
  route `view_page` to a dense VLM (e.g. gemma4:31b) that locates a figure, re-renders that
  region at 500 dpi, and reads it with thinking. Config: `configs/gemma4_vision.yaml`.
- **numbers_via_sql verifier** — when a table-quantitative answer never queried the tables, one
  correction loop makes the agent read each figure from its exact row via sql (targets the
  fin_q01 segment-vs-category confusion). Config: `configs/numbers_sql.yaml`.

Both are being settled by same-session A/Bs (runs/vision_ab.log, runs/numbers_ab.log); verdicts
land in RESULTS.md.

## Immediate next steps

Ranked list in RESULTS.md §Next steps.

## Not yet built (from DESIGN.md)

Verifier loop beyond numbers grounding (§8.5), code-execution sandbox for the analyst
(§8.4 — Docker on this Mac, macOS 15 has no Apple `container`), deep-research orchestrator with
parallel subagents (§8.2), true multi-index scope merge in the API (currently first-source),
MCP server packaging of the retrieval harness, deferred parsing, audio ingestion,
Langfuse/Phoenix tracing.

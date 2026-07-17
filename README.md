# Multimodal Agentic RAG (local-first)

A fully-local agentic multimodal RAG system: plans its own retrieval pipeline per query, reasons over documents (text, tables, charts, scans, audio), executes analysis code in a sandbox, steps back to gather more context when evidence is thin, and fans out parallel search subagents from an orchestrator — all on open-weight models.

**Start here: [DESIGN.md](DESIGN.md)** — the full system design (July 2026), grounded in a primary-source research sweep of the current model, retrieval, and infrastructure landscape.

## Design at a glance

- **Models:** Qwen3.5/3.6 multimodal backbone (orchestrator + subagents share one server), tomoro-colqwen3 visual retriever, Qwen3-VL-Embedding/Reranker, MiniCPM-V perception sidecar — all Apache 2.0.
- **Index:** three fused channels (BM25 + dense + visual late-interaction, RRF k=60) in LanceDB (Mac) or Qdrant (NVIDIA), plus DuckDB for extracted tables.
- **Agent plane:** router (direct / single-shot / iterative / deep) → orchestrator with plan + memory files → parallel search subagents → sandboxed analyst → grounded verifier.
- **Interface boundary:** one MCP retrieval-harness server (`hybrid_search`, `grep`, `find_files`, `read`, `sql`, `view_page`) so the agent framework stays swappable.

## Quickstart

```bash
uv sync
ollama pull qwen3.6:35b-a3b && ollama pull qwen3.5:9b   # or edit configs/default.yaml for your server

marag status                          # corpus / index / model overview
marag ingest research_papers          # parse → chunks + page images + tables
marag index research_papers           # dense + FTS index (LanceDB)
marag index-visual research_papers    # late-interaction page-image index

marag search "ColPali ViDoRe results" research_papers
marag ask "..." research_papers       # single-shot RAG
marag agent "..." research_papers     # iterative agent (search/grep/read/sql)
marag answer "..." research_papers    # routed: router picks the mode

marag eval-retrieval research_papers  # hit rates per channel vs QA benchmark
marag eval-answers research_papers --mode agentic

marag serve                           # local workbench UI at http://127.0.0.1:8000
```

## Workbench UI (`marag serve`)

A local web workbench backs the whole pipeline: multiple chat sessions (persisted in SQLite),
source selection across indices with live doc/page/chunk scope, drag-and-drop upload that runs
the ingest pipeline, an agent answer that **streams its tool trace** (search → grep → read →
sql → view_page) as it works, and clickable `[doc pN]` citations that open the real rendered
page plus its extracted tables in an evidence panel. `ui/mock.html` is the static design
reference; the wired app is served at `/`. API is plain REST + SSE (`GET /api/sources`,
`POST /api/ask` streaming, `GET /api/page/...png`, …) so any frontend can drive it.

Datasets live in `data/raw/<dataset>/` with a `manifest.json`; QA benchmarks in `data/benchmarks/<dataset>.json`; eval outputs in `runs/`.

## Build phases

| Phase | Scope |
|---|---|
| 0 | Serving bring-up (llama-swap + MLX / vLLM), tool-call smoke tests |
| 1 | Classic hybrid RAG + private benchmark + tracing (the baseline) |
| 2 | Multimodal channels: page-image index, table/figure dual indexing, audio |
| 3 | MCP harness + router + orchestrator/subagents + verifier |
| 4 | Sandboxed analyst, deferred parsing, CI eval gate, hardening |

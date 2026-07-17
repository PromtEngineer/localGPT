# Status — overnight build, July 16, 2026

## What exists now

- **DESIGN.md** — the researched system design (July 2026 landscape).
- **RESULTS.md** — first full validation results. Headline: agentic 100%/80%/73.3% vs
  single-shot 33.3%/46.7%/13.3%; tri-hybrid+rerank retrieval 97%/86.7%/84.4% hop recall.
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

## Immediate next steps

Ranked list in RESULTS.md §Next steps. Biggest win first: the `view_page` VLM tool —
page PNGs already exist under `data/processed/*/pages/`, qwen3.6:35b-a3b accepts images,
and figure_read questions are the dominant failure class.

## Not yet built (from DESIGN.md)

Verifier loop (§8.5), code-execution sandbox for the analyst (§8.4 — Docker on this Mac,
macOS 15 has no Apple `container`), deep-research orchestrator with parallel subagents (§8.2),
Docling parsing upgrade, MCP server packaging of the retrieval harness, deferred parsing,
audio ingestion, Langfuse/Phoenix tracing.

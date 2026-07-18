# Graph RAG for this system — research & verdict (July 17, 2026)

Research-only (no implementation). Three parallel research passes: framework landscape,
measured evidence vs strong baselines, local-first construction recipes. Sources inline.

## Verdict

**A full GraphRAG pipeline: no.** Nothing in the 2025–2026 literature predicts a
community-summarization pipeline (Microsoft-GraphRAG-style) would improve a system that is
already agentic multi-hop at 88.3% — pipeline GraphRAG and LightRAG *lose* to strong dense
retrieval on QA ([HippoRAG 2 tables](https://arxiv.org/abs/2502.14802)), indexing costs
~10M tokens per 1K passages, and Microsoft itself retreated from LLM-extracted graphs
([LazyGraphRAG](https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/)
deletes 99.9% of that cost — and never shipped open-source:
[graphrag#1490](https://github.com/microsoft/graphrag/discussions/1490)).

**A narrow graph layer as an agent tool: qualified yes, for two specific capabilities** —
but not for the current benchmark, where the expected gain is ~1 question (inside our
measured noise band).

## The evidence, honestly weighted

- Against **naive RAG**: graphs win modestly (+1–3 F1 multi-hop), and lose on simple facts
  (−11.6 pts, [GraphRAG-Bench](https://arxiv.org/abs/2506.05690)). Irrelevant to us.
- Against **strong dense retrieval**: only *passage-anchored* graphs (HippoRAG 2: entities +
  triples + synonym edges with passages as first-class nodes, PPR retrieval) win, mainly on
  entity-alias-heavy corpora (2Wiki Recall@5 76.5→90.4). Structure-heavy methods lose.
- Against **agentic retrieval** (the only comparison that matters for us): two direct 2026
  studies, opposite conclusions, both below our model class.
  [RAGSearch](https://arxiv.org/abs/2604.09666) (Apr 2026): graph backends still help agentic
  search at 3B–7B, but the gap **shrinks monotonically with model scale** (14.70→9.75 EM,
  3B→7B). [LogicalRAG](https://arxiv.org/abs/2605.27123) (May 2026): an agentic *hybrid*
  baseline beats the best graph method outright (0.807 vs 0.683) — "graph advantages are
  diminished in agentic RAG systems." No published result shows graph gains on top of a
  frontier-class agent.
- Reported GraphRAG gains are systematically **inflated** by LLM-judge win-rate methodology
  ([unbiased re-evaluation](https://arxiv.org/abs/2506.06331)) and shrink further on messy
  real corpora ([WildGraphBench](https://arxiv.org/abs/2602.02053)).

**Mapped to our residual failures:** of the current single-session misses, only one
(fin_q01 — Data-Center-market-platform vs Compute-&-Networking-segment, a typed-entity
disambiguation) is graph-shaped. The health misses are chart-reading physics; a graph cannot
help. So on the existing 60 questions, graph RAG buys ≤1 question ≈ noise.

**Where the replicated wins actually are (both untested in our benchmark):**
1. **Corpus-level sensemaking/aggregate questions** ("across all filings, which companies…",
   theme synthesis) — +13 pts in [GraphRAG-Bench](https://arxiv.org/abs/2506.05690); the one
   question class top-k retrieval structurally cannot answer and our per-doc `summarize_doc`
   only partially covers.
2. **Entity-alias/bridge recall robustness** — the graph as a canonical-entity/alias index
   (the fin_q01 class, generalized).

## If/when we build it — the recipe (all permissive, all local)

- **Pattern:** HippoRAG-2-style passage-anchored graph exposed as **agent tools**, NOT a
  replacement pipeline and NOT community summaries (they are the cost driver, kill
  incremental ingest, and their value vs our agent is unproven). Skipping them makes adding
  one document a cheap set-union + ANN-synonym-edge append.
- **Extraction:** per-domain compact schemas; qwen3.6/gemma4 via **Ollama structured outputs**
  (grammar-constrained decoding — malformed-JSON risk is largely solved:
  [docs](https://docs.ollama.com/capabilities/structured-outputs)); non-thinking mode;
  ~1K-token chunks; ≤1 gleaning round; optional [GLiNER2](https://github.com/fastino-ai/GLiNER2)
  (Apache-2.0, CPU) prepass for typed mentions. Known risk: best published local extraction
  used a 70B model; relation fidelity is where ~30B models degrade, and graph errors are
  silent — extraction quality must be audited before trusting retrieval built on it.
- **Entity resolution** (the "NVIDIA" = "the Company" problem): deterministic per-document
  alias table first (10-K boilerplate resolves mechanically), then embedding-blocked candidate
  pairs (our existing embedder), then LLM adjudication + cluster-merge
  ([KGGen](https://arxiv.org/html/2502.09956v1)).
- **Storage:** NetworkX projection (BSD; ~10⁴–10⁵ nodes at our scale) persisted as parquet
  edge lists; [LadybugDB](https://github.com/LadybugDB/ladybug) (MIT fork of the abandoned
  Kuzu — Kuzu was archived Oct 2025) if we want durable Cypher+FTS+vector in one file.
  FalkorDB (SSPL) and Memgraph (BSL) fail the license constraint; Neo4j CE is GPLv3 (server).
  DuckPGQ is pinned to DuckDB 1.4.4 — trial-only.
- **Agent exposure:** typed tools — `graph_entity_lookup` (alias/fuzzy; defeats the measured
  typo failure mode), `graph_neighborhood` (≤2 hops, provenance IDs), `graph_path`,
  `graph_ppr_retrieve` (returns chunk IDs into our existing reranker). Raw text2cypher only
  as a guarded fallback — even Claude-class models hit 0.71 avg on Cypher-tool agent evals
  ([Neo4j](https://neo4j.com/blog/developer/evaluating-graph-retrieval-in-mcp-agentic-systems/)).
  Same lesson as view_page/numbers-via-sql: enforce the path, don't request it.
- **Multimodal provenance — our unfair advantage:** every triple inherits Docling's
  (doc_id, page, bbox); figures and tables become first-class nodes with `EVIDENCED_BY`
  edges, and `Table` nodes carry their DuckDB view name so the agent can hop graph→SQL.
  For us this is a metadata join, not research ([MegaRAG](https://arxiv.org/html/2512.20626v2)
  and [docling-graph](https://github.com/docling-project/docling-graph) validate the pattern).
- **Frameworks if not building ourselves:** [LightRAG](https://github.com/HKUDS/LightRAG)
  (MIT, most active, native Ollama, per-role models) or vendored
  [HippoRAG 2](https://github.com/OSU-NLP-Group/HippoRAG) (MIT, cheapest, tool-shaped);
  [Cognee](https://github.com/topoteretes/cognee) (Apache) if we wanted a memory platform.
  Stale/excluded: nano-graphrag, fast-graphrag, MiniRAG (dead releases); RAGFlow (no ARM64).

## Cost at our scale (~100 docs, ~5K chunks)

HippoRAG-2-style: ~2 structured calls/chunk ≈ 10K extraction calls → one overnight run on
this hardware; per-query cost ~1K tokens (vs 331K/query for GraphRAG-global). Incremental
per-doc ingest thereafter. That is affordable *if* the capability is wanted.

## Decision gate (our eval ethos)

Build only against a falsifiable test, in this order:
1. Author a **corpus-level aggregate benchmark section** (~10 questions the current system
   should struggle with) — measure the baseline *first*; if the agent + summarize_doc already
   scores well, the strongest remaining case for graphs collapses.
2. Same-session A/B with the graph tools on: (a) that new section, (b) fin_q01-class
   disambiguation probes, (c) the existing 60 (regression guard), with token costs reported.

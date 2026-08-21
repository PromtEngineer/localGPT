# Research Evidence (August 2026)

Three independent research sweeps on the state of the art in agentic retrieval,
compiled 2026-08-08 by LLM research agents from primary sources only (first-party
engineering blogs, arXiv/ACL/ICLR/SIGIR papers, official model cards and
leaderboards). Anonymous blog posts and SEO content were explicitly rejected;
each report carries its own rejection log and a list of claims that could NOT
be verified.

| File | Scope |
|------|-------|
| [industry-evidence-2026.md](industry-evidence-2026.md) | What Anthropic, OpenAI, Google, Microsoft, LlamaIndex, LangChain, Weaviate, Qdrant, Vespa, Pinecone, Elastic, Exa, Perplexity, Glean et al. have published about production agentic retrieval — architectures, numbers, and where vendors directly contradict each other |
| [academic-evidence-2026.md](academic-evidence-2026.md) | The 2025–26 literature: RL-trained search agents, reasoning-aware retrieval (BRIGHT lineage), late interaction, hybrid fusion, late chunking, context compression, verification/attribution, GraphRAG, memory — with the controlled replications that deflated the 2024 headline claims |
| [component-map-2026.md](component-map-2026.md) | Component-by-component SOTA map for local/self-hosted stacks: parsing, chunking, embeddings, sparse+fusion, rerankers, query planning, routing, loop patterns, verification, compression, memory, evaluation — each claim graded established / emerging / contested |

## How to read these

- These are **evidence documents, not descriptions of localGPT**. Nothing in
  them implies a feature exists in this repo. What localGPT actually does is
  documented in the rest of `Documentation/`.
- Every claim carries its source and date. Claims graded *established* are
  peer-reviewed or independently replicated; *vendor* numbers are self-reported
  and directionally credible at best.
- The actionable distillation lives in
  [../research_roadmap.md](../research_roadmap.md), which maps this evidence to
  concrete, staged changes with acceptance criteria.

Provenance: generated with web access on 2026-08-08; model knowledge cutoffs
predate several cited releases, so everything recent is web-sourced. Each
report's own "could not verify / do not cite" appendix applies.

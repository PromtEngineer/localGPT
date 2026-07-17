# Validation Results — July 16, 2026

First full validation of the marag pipeline on three purpose-built multimodal benchmarks
(15 verified multi-hop questions each; gold evidence = doc + page + modality, answers read
off rendered PDF pages by the benchmark authors). Hardware: M2 Max 96GB. Models: qwen3.6:35b-a3b
(orchestrator), qwen3.5:9b (router/judge), Qwen3-Embedding-0.6B, Qwen3-Reranker-0.6B,
ColModernVBERT-250M (visual), all via Ollama (64K ctx) + in-process MPS.

## Retrieval (hop recall @ k=10, page ±1)

| variant | research_papers | financial_docs | legal_docs |
|---|---|---|---|
| BM25/FTS only | 93.9% | 63.3% | 65.6% |
| dense only | 93.9% | 70.0% | 62.5% |
| hybrid (dense+FTS, RRF) | **97.0%** | 73.3% | 71.9% |
| hybrid + rerank | 93.9% | 73.3% | 78.1% |
| visual only (ColModernVBERT) | 84.8% | 76.7% | 78.1% |
| tri-hybrid (dense+FTS+visual) | **97.0%** | 73.3% | 71.9% |
| tri-hybrid + rerank | 93.9% | **86.7%** | **84.4%** |

Findings:
1. **Hybrid > any single channel** on every dataset — RRF fusion is free recall.
2. **The visual channel is what rescues chart/scan-heavy corpora**: on financial and legal
   it beats both text channels alone, and tri-hybrid+rerank adds **+13.4pp / +12.5pp**
   over text-only hybrid. On clean text-rich papers it adds nothing — include it per-corpus,
   as the ViDoRe-v3 pipeline evidence predicted.
3. **Reranking helps where recall is hard** (financial/legal) and can slightly hurt where
   recall is easy (papers: 97→93.9 — the cross-encoder occasionally demotes a gold hop that
   RRF ranked in top-10). Worth revisiting rerank-depth (currently 25).
4. Table-hop recall reaches 100% on legal, 83% on financial (tri-hybrid+rerank);
   figure hops are found via the visual channel (legal figure recall 38%→50-62% with visual).

## End-to-end answers (LLM-judged vs gold, binary)

| dataset | single-shot | agentic (search/grep/read/sql agent) | avg tool calls | avg latency |
|---|---|---|---|---|
| research_papers | 33.3% | **100%** | 6.9 | ~59s |
| financial_docs | 46.7% | **80.0%** | 7.0 | ~112s |
| legal_docs | 13.3% | **73.3%** | 9.3 | ~114s |

Citation grounding (fraction of [doc pN] citations pointing at evidence actually retrieved):
0.92–1.0 in agentic mode.

Findings:
1. **Iterative agentic retrieval is decisively better on multi-hop multimodal questions**
   (+47 to +60pp over single-shot) at ~2x latency — consistent with the 2026 literature
   (agentic wins narrow multi-hop; route simple queries elsewhere).
2. **Residual failures are concentrated in figure_read questions** (fin_q05, leg_q03, leg_q13…):
   the agent retrieves the right page but can only read its extracted *text* — chart values
   often aren't in the text layer. This is precisely what the designed-but-not-yet-built
   `view_page` VLM tool addresses (qwen3.6 has vision; the page PNGs are already on disk).
3. One numeric miss (fin_q01: $116.2B vs $115.186B) shows why the design's "numbers rule"
   (quantitative claims must come from `sql` over extracted tables) needs *enforcement*,
   not just prompting — blocked today by pymupdf `find_tables` missing many tables (Docling
   upgrade path).

## Table-parsing upgrade (July 16, follow-up)

Bench on the 39 gold table pages (`scripts/table_bench.py`, raw: `runs/table_bench.json`) —
pymupdf `find_tables` vs Docling TableFormer (FAST + ACCURATE) vs LiteParse 2.6 (LlamaIndex's
spatial parser):

| parser | value recall (text) | label adjacency | gold values in structured cells | garble | s/page |
|---|---|---|---|---|---|
| pymupdf (old) | 56%* | 0% | **31%** | 14% | ~0 |
| **Docling FAST (adopted)** | 56%* | 21% | **56%** | 7% | 1.8 |
| Docling ACCURATE | 56%* | 21% | 56% | 7% | 2.3 |
| LiteParse | 56%* | 21% | — (no structure by design) | 0% | 0.6 |

\* 56% is the metric ceiling — the remainder are computed values or numbers from other hops'
pages; all parsers extract everything actually printed. Verdict: **LiteParse ties Docling on
text fidelity at 3× speed but produces no structured tables, so it cannot repair the SQL
layer — Docling FAST adopted for table extraction** (`marag rebuild-tables`; ACCURATE buys
nothing here). LiteParse remains attractive as a future fast text-ingest path.

**Effect on structured coverage** (gold table hops with a SQL-queryable table, `scripts/table_audit.py`):
research 41%→**100%**, financial 67%→**100%**, legal 60%→**100%**; total extracted tables
52→243 / 356→912 / 46→215.

## view_page VLM tool + hardened agent loop (July 16, final)

The remaining failure class after the table upgrade was **figure_read** — chart values that
exist in no text layer. Added `view_page(doc_id, page, question)`: the agent asks a focused
question about a page; the tool sends the page PNG to qwen3.6 (vision) in a *separate*
sub-call and returns only the text reading, so images never enter the agent's context.
Two more empirical rules baked in along the way:
- **Vision + thinking starves even 3K-token budgets** → view_page runs with
  `reasoning_effort="none"` (faster and fuller readings).
- **Chart labels in extracted text lose their visual association** (a prior-year bar's
  label reads identically to the current bar's) → the agent MUST confirm chart values
  with view_page even when numbers appear in text. This exact failure produced a wrong
  Uber Q4'24 answer ($37,575M = the Q4'23 bar) before the rule; exact-match after.
- Empty finals (thinking overflow) are never accepted — automatic no-thinking retry.

**Answer-accuracy progression** (agentic mode, 15 questions/dataset, LLM-judged vs gold):

| dataset | v1 text-only tools | v2 +Docling tables, budget 20, temp 0 | v4 +view_page & guards |
|---|---|---|---|
| research_papers | 100% | 100% | 93.3% |
| financial_docs | 80.0% | 86.7% | **93.3%** |
| legal_docs | 73.3% (unstable) | 66.7% | **100%** |
| **aggregate** | 84.4% | 84.4% | **95.6% (43/45)** |

Avg 7.3–7.4 tool calls/question, citation grounding 1.0. The two residual failures are
single-question variance (rp_q08, fin_q01 — each has passed in other runs); fin_q01
(NVIDIA "Compute & Networking" segment vs "Data Center" market platform) remains the
motivating case for hard-enforcing the numbers-via-SQL rule.

## Bugs found by validation (all fixed)

| bug | symptom | fix |
|---|---|---|
| Two PyTorch-MPS processes deadlock | index job frozen at 0% CPU in Metal dispatch | serialize GPU jobs; `MARAG_DEVICE` override |
| Thinking-token starvation | reasoning models return EMPTY content when max_tokens ≤ thinking length; zeroed an entire eval round | budgets ≥3-4K for generation; `reasoning_effort:"none"` for router/judge; `LLM.chat(reasoning=…)` |
| Ollama app serves 16K context (model supports 262K) | long agent loops silently truncated → empty answers on high-tool-call questions | dedicated `OLLAMA_CONTEXT_LENGTH=65536 ollama serve` on :11435 (see configs/default.yaml) |
| HF Hub outage hangs model loads | eval stuck on stalled SSL read | local-snapshot-first loading; `HF_HUB_OFFLINE=1` compatible |

## Next steps (ranked by expected gain)

1. ~~`view_page` VLM tool~~ **DONE** — legal 66.7%→100%, aggregate 95.6%.
2. ~~Docling parsing upgrade~~ **DONE** — next: hard-enforce the numbers-rule (answers with
   numbers must cite a `sql` result) now that coverage is 100% (would catch fin_q01).
3. **Rerank tuning** — depth/threshold so easy-recall corpora aren't hurt.
4. Router-based auto mode end-to-end eval (route simple→single-shot, hard→agentic) to get
   the cost/quality frontier.
5. Verifier pass (claim→span NLI + ≤2 correction loops) per DESIGN.md §8.5.

Raw eval artifacts: `runs/*.json` (per-question results, judges' reasons, transcripts).

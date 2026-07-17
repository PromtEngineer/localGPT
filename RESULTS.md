# Validation Results — July 16, 2026

**Headline: 95.6% agentic accuracy (43/45) on the three development domains, and 80% on a
fourth unseen domain run with zero code changes — the system generalizes.** See the
Generalization test section for the transfer evidence.

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

## Generalization test — unseen domain (health_docs, July 16)

To test whether the system generalizes or was merely tuned to its three development
corpora, a **fourth domain the pipeline had never seen** was added and run with **zero
code or prompt changes**: 24 public-health PDFs (14 CDC MMWR/surveillance, 5 WHO reports,
5 FDA drug labels; influenza/COVID/RSV + vaccination theme) with 15 verified multi-hop
questions (93% carrying a table/figure hop). It is *harder* than the originals: FDA label
tables have an unusable text layer (Mounjaro's Table 1 extracts with no values at all),
so some answers are reachable only through vision.

| metric | health_docs (unseen) | 3 development domains |
|---|---|---|
| retrieval hop recall (tri-hybrid+rerank) | **97.0%** | 84.4–97.0% |
| full-question retrieval recall | **93.3%** | 66.7–93.3% |
| Docling table coverage of gold hops | **100%** (19/19) | 100% |
| single-shot answers | 33.3% | 13.3–46.7% |
| **agentic answers** | **80.0%** | 93.3–100% |
| citation grounding | 1.0 | 1.0 |

Everything upstream of generation transferred **without any tuning** — retrieval landed at
the top of the range seen on the development domains, and Docling hit 100% table coverage
on a document class (FDA labels, WHO reports) it had never been tested on.

Notable: **the visual channel alone (93.9% hop recall) beat every text configuration** on
this corpus — the strongest evidence yet for the design's multimodal bet, on the domain
where the text layer is worst.

Agentic accuracy at 80% (12/15) sits below the development domains but above where those
domains started (financial 80%, legal 73.3% at the same stage). All three failures are
**figure_read** or chart-derived: reading a peak off an epidemic curve (hlt_q10), a
multi-panel WHO slope chart (hlt_q14), and a TB-report figure (hlt_q15). The benchmark
author independently hit the same wall — two of their first-pass figure readings were
wrong at normal render and needed 400–500 dpi crops. Our page images are rendered at
150 dpi, which is very likely the binding constraint.

**Verdict: the system generalizes.** Retrieval, parsing, and the agent loop transfer
cleanly to an unseen domain; the gap is confined to fine-grained chart reading (see the
next section — it is a model limit, not a domain-fit problem).

### Postmortem: the high-DPI hypothesis was wrong (negative result)

Hypothesis: the health failures were a *resolution* problem — page images render at 150 dpi
while the benchmark author needed 400–500 dpi crops to read the same figures. Test case
hlt_q10: FIGURE 1 of hlt014 p5, peak of the 2024–25 RSV curve for infants 0–7 months
(gold ≈0.75/1,000; the pooled 2018–20 curve peaks ≈1.33).

Built `view_page(..., region=)` — re-rasterizes the page or a region from the source PDF at
up to 500 dpi (upscaling the stored PNG adds no information). Then measured:

| test | blue 2024-25 peak | black 2018-20 peak |
|---|---|---|
| **ground truth** (read off the axis by hand) | **~0.72–0.75** | **~1.33** |
| qwen3.6, stored 150-dpi full page | 0.95 ✗ | 1.3 ✓ |
| qwen3.6, ~305-dpi region zoom | 0.9 ✗ | 1.3 ✓ |
| qwen3.6, **tight 500-dpi panel crop, series named in the prompt** | 0.9 ✗ | 1.3 ✓ |
| MiniCPM-V 8B (dedicated doc-VLM), same 500-dpi crop | **1.6 ✗✗** | 1.3 ✓ |

Resolution is not the binding constraint: every VLM reads the *black* line correctly and
misreads the *thick blue* one at every DPI, because the panel carries four similar blue-ish
elements (solid series, dashed CI, dotted CI, shaded band). It is a **series-discrimination
limit**, and it is general — a dedicated document-VLM did worse than the generalist.

End-to-end effect: **80.0% → 73.3%** on health_docs (worse; hlt_q10 degraded from
"wrong value" to "curves swapped"). The change was reverted: full-page `view_page` again
uses the validated stored-PNG path; the `region` zoom remains available as opt-in (it costs
nothing unused) but is **not** credited with any measured gain.

**What this means for the design:** local VLMs reliably read *discrete, labeled* visual
values (bar labels, pie percentages, table cells, patent drawings — the wins that took
legal 66.7%→100%), but not *interpolated* values off dense multi-series line charts. That is
today's ceiling for the multimodal path, and the honest next moves are (a) treat such reads
as approximate and surface uncertainty rather than assert a number, or (b) retrieve the
underlying data table when one exists, not more pixels.

## Bugs found by validation (all fixed)

| bug | symptom | fix |
|---|---|---|
| Two PyTorch-MPS processes deadlock | index job frozen at 0% CPU in Metal dispatch | serialize GPU jobs; `MARAG_DEVICE` override |
| Thinking-token starvation | reasoning models return EMPTY content when max_tokens ≤ thinking length; zeroed an entire eval round | budgets ≥3-4K for generation; `reasoning_effort:"none"` for router/judge; `LLM.chat(reasoning=…)` |
| Ollama app serves 16K context (model supports 262K) | long agent loops silently truncated → empty answers on high-tool-call questions | dedicated `OLLAMA_CONTEXT_LENGTH=65536 ollama serve` on :11435 (see configs/default.yaml) |
| HF Hub outage hangs model loads | eval stuck on stalled SSL read | local-snapshot-first loading; `HF_HUB_OFFLINE=1` compatible |

## Next steps (ranked by expected gain)

1. ~~`view_page` VLM tool~~ **DONE** — legal 66.7%→100%, aggregate 95.6%.
2. ~~High-DPI figure crops~~ **TRIED, REVERTED** — negative result, see postmortem above.
   The replacement idea: when a chart read is uncertain, prefer the underlying data table
   (`sql`) or surface the uncertainty; don't assert an interpolated number.
3. ~~Docling parsing upgrade~~ **DONE** — next: hard-enforce the numbers-rule (answers with
   numbers must cite a `sql` result) now that coverage is 100% (would catch fin_q01).
4. **Rerank tuning** — depth/threshold so easy-recall corpora aren't hurt.
5. **Bigger VLM for chart reads** — the only untested lever on the chart ceiling: route
   `view_page` to a larger vision model (Qwen3.5-122B-A10B 4-bit fits this 96GB box) and
   re-measure hlt_q10/q14 before believing it.
4. Router-based auto mode end-to-end eval (route simple→single-shot, hard→agentic) to get
   the cost/quality frontier.
5. Verifier pass (claim→span NLI + ≤2 correction loops) per DESIGN.md §8.5.

Raw eval artifacts: `runs/*.json` (per-question results, judges' reasons, transcripts).

# Validation Results — July 16-17, 2026

**Headline: 88.3% agentic accuracy (53/60) across four domains of verified multi-hop
multimodal questions — single-session, validated config — including a domain the system had
never seen, run with zero code changes.**

Single-session baseline (July 17, all four evals in one Ollama session, validated config):

| dataset | agentic | failures |
|---|---|---|
| research_papers | 93.3% (14/15) | rp_q13 |
| financial_docs | 86.7% (13/15) | fin_q01, fin_q05 |
| legal_docs | 100% (15/15) | — |
| health_docs (unseen) | 73.3% (11/15) | hlt_q08, hlt_q10, hlt_q14, hlt_q15 |
| **aggregate** | **53/60 = 88.3%** | |

Prior drafts quoted 95.6% on the three development domains; that figure was assembled across
several sessions and is ~2 questions optimistic versus this single-session measurement. The
remaining failures concentrate in chart/figure reads (the known VLM ceiling) plus one
segment-vs-category confusion (fin_q01) that numbers-via-SQL enforcement should catch.

## Read this before comparing any two numbers below

Measured on July 17 after a config change appeared to move a score by 2 questions:

- **Within one Ollama session the pipeline is deterministic.** Three consecutive runs of an
  identical config on financial_docs returned byte-identical results (12/15, same three
  failures, zero flaky questions). Both judges — at temperature 0.2 and 0.0 — regrade fixed
  answers with zero verdict flips.
- **Across sessions it drifts by ~1 question.** The same code scored financial_docs 93.3% in
  one session and 86.7% in another; between them the server was restarted and two extra models
  were pulled. Different resident-model state changes batching/KV-cache, and at temperature 0
  a single flipped low-probability token cascades through a 10-20 step agent loop.

**Therefore: only same-session A/B comparisons are evidence.** At n=15 per domain one question
is 6.7%, so a 1-2 question difference measured across sessions means nothing. Historical
numbers in this document were gathered across several sessions and carry ±1 question of
uncertainty each; the four-domain table is a single-session baseline. Earlier claims that
rested on cross-session 2-question deltas (notably the Gemma 4 vision verdict) are labelled
as inconclusive rather than settled.

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

### The chart-reading experiment (dense VLM, zoom, thinking) — capability up, score flat

Round 2 of the chart investigation, after the DPI-only postmortem below. Hypothesis: the
orchestrator qwen3.6:35b-a3b is a **MoE with only ~3B active parameters**, so its vision head
runs on a fraction of the compute — try a **dense** reader. Gemma 4 31B (dense, vision+tools,
Apache 2.0) was routed to `view_page` via a new `models.vision` slot while qwen3.6 kept driving
the agent loop.

**On the isolated failing read** (hlt014 p5 FIGURE 1; truth: blue ≈0.75, black ≈1.33) three
factors proved *jointly* necessary — this is why the DPI-only test below looked like a dead end:

| reader | render | thinking | blue peak | black peak |
|---|---|---|---|---|
| qwen3.6 (MoE) | any dpi, any crop | either | 0.9–0.95 ✗ | 1.3 ✓ |
| gemma4 31B (dense) | 150-dpi full page | on | 1.1 ✗ | 1.4 ✗ |
| gemma4 31B (dense) | 500-dpi corner crop | off | 0.9 ✗ | 1.3 ✓ |
| **gemma4 31B (dense)** | **500-dpi corner crop** | **on** | **0.7–0.8 ✓** | **1.3 ✓** |

Resolution only pays off in a model that can use it; thinking is what performs the
tick-to-value interpolation ("halfway between the 0.6 and 0.8 tick marks"); and the crop must be
a *corner* region — halves hit the pixel cap at ~305 dpi and read wrong.

**Two harness lessons on the way:**
1. *Prompting the agent to do the two-step didn't work.* Told the locate→zoom sequence was
   mandatory, the agent still called `view_page` once at full page (hlt_q10) or zoomed to
   `right`/`bottom-right` when the panel was `top-left`. Perfect vision, wrong window.
   The fix was mechanical: `view_page` now runs locate→zoom *inside the tool* (cheap no-think
   full-page pass → parse its `REGION:` hint → re-render that region at 500 dpi → read with
   thinking). Same lesson as the numbers-via-SQL rule: **enforce the path, don't request it.**
2. *The locate step must be biased toward the tightest region*, or it names a half and loses
   the resolution it just went to get.

**End-to-end (first pass): INCONCLUSIVE — superseded below.** The original comparison ran
across server sessions (53/60 with gemma4 vs 55/60 with the default), which the variance work
above showed is worth ~1 question per domain of drift on its own. The measured difference never
exceeded the measurement error. **A proper same-session A/B (July 17) settled it — see
"Settled by same-session A/Bs" below: the gemma4 vision arm recovers hlt_q10 (health
80.0%→86.7%) with financial identical, at ~1.5× latency.**

Two things that are solid regardless:
- Chart readings still fluctuate ±0.1 between calls even when correct (the black curve read
  1.2 / 1.3 / 1.4 on separate calls), so a strict binary judge cannot bank the capability.
- Auto-zoom plausibly **crops away context** on reads that already worked — tables, patent
  drawings, dense text — which is where legal's 100% came from. Plausible, not established.

**Reverted to the validated default.** The recipe is documented here rather than shipped as
dormant config: route `view_page` to a dense vision model, render a *corner* region at 500 dpi
from the source PDF, read with thinking and a ≥6K token budget, and run locate→zoom inside the
tool rather than asking the agent to do it.

**The durable finding:** local VLM chart reading went from *systematically wrong* to
*approximately right with noise*. Neither state supports asserting an exact interpolated value.
The design's own rule already covers it — prefer the underlying table via `sql`, or surface the
uncertainty. Precision on dense multi-series line charts remains the honest ceiling.

### Postmortem: the DPI-only hypothesis was wrong (negative result)

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

## Settled by same-session A/Bs (July 17)

Both open opt-ins were settled the only way the variance protocol allows: back-to-back runs
in one Ollama session, judge pinned (temp 0, reasoning none), verdicts from run pairs sorted
by mtime.

### Dense-vision view_page (configs/gemma4_vision.yaml) — ADOPTED as the vision opt-in

| arm | financial_docs | health_docs |
|---|---|---|
| A — default (qwen3.6 does vision) | 93.3% (fails: fin_q01) | 80.0% (fails: hlt_q10, q14, q15) |
| B — gemma4 vision + auto-zoom + thinking | 93.3% (fails: fin_q01) | **86.7%** (fails: hlt_q14, q15) |

The gemma4 arm **recovers hlt_q10** — the dense multi-series epidemic curve that drove the
entire chart-reading investigation — exactly as the isolated-read experiments predicted, and
degrades nothing. +1 question at n=15 is inside the formal noise band, but it is the *specific*
question the mechanism targets, recovered with the failure mode the mechanism fixes, at zero
cost elsewhere. Cost: ~1.5× per-question latency when `view_page` fires. Verdict: the earlier
cross-session "worse end-to-end" reading was variance, as suspected. The recipe works; it stays
opt-in only because of the latency price.

### numbers_via_sql verifier (configs/numbers_sql.yaml) — NOT ADOPTED (honest negative)

| arm | financial_docs | sql calls | avg tools/q |
|---|---|---|---|
| A — default | 93.3% (fails: fin_q01) | 6 | 7.1 |
| B — numbers_via_sql | 93.3% (fails: fin_q01) | 41 | 10.0 |

The verifier *fired* exactly as designed — sql usage went 6→41 calls — and fin_q01 **still
fails**: the agent queries the tables and pulls **$47,405M from the Compute & Networking
segment row** instead of $47,525M from the Data Center market-platform row. The error was
never a grounding problem (reading numbers from prose/pixels); it is a **disambiguation
problem** (two near-identical revenue concepts in the same 10-K, $120M apart). Forcing the
sql path cannot fix a wrong-row choice, and it costs +41% tool calls. Grounding enforcement
and semantic disambiguation are different failures needing different mechanisms — a candidate
for the DESIGN §8.5 claim-level verifier, not for path enforcement.

## Bugs found by validation (all fixed)

| bug | symptom | fix |
|---|---|---|
| **Tool description contradicted the implementation** | after reverting auto-zoom, `view_page`'s description still advertised "it automatically finds the relevant part and re-reads it zoomed — one call is enough". The agent believed the description, stopped zooming, and financial_docs lost a question (80% → 86.7% when the honest description was restored, same session). **A tool description is an executable interface contract, not documentation** — reverting behavior without reverting the contract silently degrades the agent | restore description and implementation together; treat prompt/description edits as code changes |
| Eval discarded the tool sequence | a wrong answer looked like a model failure; the transcript showed the agent had zoomed to `right` when the panel was `top-left` — a wrong *path*, not a wrong model. Cost a full diagnostic cycle | `runs/*.json` now persists each question's tool calls |
| Judge ran at temperature 0.2 | a stochastic grader would have added silent noise (it happened not to flip any verdict, but the exposure was real) | judge pinned to temperature 0.0 |
| Two PyTorch-MPS processes deadlock | index job frozen at 0% CPU in Metal dispatch | serialize GPU jobs; `MARAG_DEVICE` override |
| Thinking-token starvation | reasoning models return EMPTY content when max_tokens ≤ thinking length; zeroed an entire eval round | budgets ≥3-4K for generation; `reasoning_effort:"none"` for router/judge; `LLM.chat(reasoning=…)` |
| Ollama app serves 16K context (model supports 262K) | long agent loops silently truncated → empty answers on high-tool-call questions | dedicated `OLLAMA_CONTEXT_LENGTH=262144 ollama serve` on :11435 — serve at the model's max, not Ollama's default; all three LLMs fit 100% GPU at 262K (see configs/default.yaml) |
| HF Hub outage hangs model loads | eval stuck on stalled SSL read | local-snapshot-first loading; `HF_HUB_OFFLINE=1` compatible |

## Next steps (ranked by expected gain)

1. ~~`view_page` VLM tool~~ **DONE** — legal 66.7%→100%, aggregate 95.6%.
2. ~~High-DPI figure crops~~ **TRIED, REVERTED** — negative result, see postmortem above.
   The replacement idea: when a chart read is uncertain, prefer the underlying data table
   (`sql`) or surface the uncertainty; don't assert an interpolated number.
3. ~~Docling parsing upgrade~~ **DONE** — ~~next: hard-enforce the numbers-rule~~ **TRIED,
   NOT ADOPTED**: the same-session A/B showed fin_q01 is a wrong-row disambiguation error,
   not a grounding error; sql enforcement fired (6→41 calls) and didn't fix it. See
   "Settled by same-session A/Bs".
4. **Rerank tuning** — depth/threshold so easy-recall corpora aren't hurt.
5. ~~Bigger/denser VLM for chart reads~~ **VALIDATED (Gemma 4 31B dense), opt-in** — the
   same-session A/B recovered hlt_q10 (health 80.0→86.7%) with nothing degraded; stays
   opt-in for the ~1.5× latency. The earlier cross-session "worse end-to-end" was variance.
   Untested remainder: Qwen3.5-122B-A10B 4-bit as the vision model.
6. **Wire the mock UI to the backend** — `ui/mock.html` mocks sessions, multi-source scope,
   upload/ingest stages, the tool trace and page evidence against real corpus numbers.
4. Router-based auto mode end-to-end eval (route simple→single-shot, hard→agentic) to get
   the cost/quality frontier.
5. Verifier pass (claim→span NLI + ≤2 correction loops) per DESIGN.md §8.5.

Raw eval artifacts: `runs/*.json` (per-question results, judges' reasons, transcripts).

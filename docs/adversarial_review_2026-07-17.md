# Independent adversarial review — July 17, 2026

Two independent review passes (methodology/design red-team + implementation code review),
run read-only at commit b722924. Condensed findings; severity as assessed by the reviewers.
[CONFIRMED] = reproduced by execution during review. Remediation tracking below.

## Part 1 — Measurement methodology & design

### C1 (critical) — Judge never validated; demonstrable misgrades
- DESIGN §10 promised judge calibration vs ~30 hand-labeled examples; never built.
  scripts/judge_variance.py measures self-consistency only (precision, not validity).
- Confirmed misgrade on the graph decision gate: fin_agg_q07 named ≥3/5 rubric themes
  (gold says PASS) but was FAILED for including an extra theme — violating the gold's
  rubric and the judge's own "extra correct detail is fine" rule. financial_agg is
  plausibly 4/7, not 3/7. fin_agg_q06's stated reasoning also wrong (real error coexists).
- JUDGE_SYSTEM conflicts with agg golds ("all gold items must be present" vs "Accept
  also: …"); model answers truncated at 3000 chars into the judge — aggregate
  enumerations are exactly the answers that blow that budget.
- "Information missing = incorrect" rule punishes the design's own mandated
  surface-uncertainty behavior on chart reads (rp_q13 failed for declining to assert).
- Fix: hand-label stored runs/ answers → publish judge–human agreement; checklist
  grading for rubric/multi-part; move accept-bands into gold text as single source.

### C2 (critical) — Graph-RAG verdict predates its own decision gate; gate result unintegrated
- docs/graphrag_research.md verdict committed before financial_agg/research_agg ran;
  the unfavorable result (42.9% financial) never folded back.
- Gate confounded: agg runs averaged 20.4 tool calls vs max_tool_calls 20 — measures
  budget starvation as much as architecture (and the budget itself leaks, see Part 2 #5).
- Fix: rerun agg at budget 40–60 with exhaustion flags + closed-book control; rewrite
  verdict from data.

### C3 (critical) — Benchmark circularity & uncontrolled contamination
- Golds verified by regex over the same pages.jsonl the agent's grep reads — correctness
  is relative to the extraction, not the documents. Pixel-only content invisible to both.
- Aggregate questions are grep-shaped (authored by grep); research_agg 80% shows "agent
  can drive grep," not sensemaking; the one sensemaking question/corpus was operationalized
  as keyword co-occurrence and misgraded (C1).
- Corpora are famous public documents inside the models' training window; no closed-book
  control run exists.
- Fix: closed-book session (tools disabled); author some golds from page renders only;
  human-authored abstractive cross-doc questions.

### M1 (major) — Variance protocol understated; selectively enforced
- Own artifacts show a 2-question cross-session envelope (financial 12/13/14 of 15 within
  ~13h; health 11/12/13), not "~1 question." 88.3% = one session-draw; Wilson 95% CI on
  53/60 ≈ [78%, 94%].
- Vision opt-in adopted on a +1 delta the protocol defines as noise. Stronger unexploited
  evidence: hlt_q10 failed in ALL 10 default-config runs, passed only in the vision arm.
  Legal (the flagged at-risk domain for auto-zoom) never included in the vision A/B.
- Fix: 3–5 session replicates incl. legal; report per-question pass counts.

### M2 (major) — No run provenance; "same-session" untested proxy
- runs/*.json record no config/models/session/git sha; arm attribution by mtime sort.
- LLM role fallback silently substitutes orchestrator when a model is missing — the
  pinned judge could silently become the system-under-test, unrecorded.
- No A/A control with mid-session model swap ever run.
- Fix: stamp cfg + resolved models + ollama ps + sha into every report; run A/A control.

### M3 (major) — Single-LLM rejection not yet earned
- One domain, one pair; config differs in TWO variables (model + vision flags);
  judge family-bias (Qwen judging Qwen-vs-Gemma) unprobed; chain unfinished. Direct
  wrong-number failures look genuine → direction probably right, magnitude unproven.
- Fix: finish 4 domains with vision flags equalized; regrade arm-B fails with second
  judge + human tiebreak.

### M4 (major) — Health chart questions grade an admitted model ceiling with
inconsistent tolerances (notes say ±10%, gold text tighter, judge allows last-digit
rounding). hlt_q15 failed inside the notes' tolerance. "(unseen)" label on 73.3% stale —
the genuine zero-changes run was 80.0%.

### M5 (major) — README describes unbuilt features present-tense (sandbox, parallel
subagents); "citation grounding" only checks cited pages ∈ touched pages (rename:
citation-page consistency); DESIGN still says "pre-implementation."

### Minor: untuned retrieval constants (rrf_k=60 etc.; rerank hurts research 97→93.9,
never tuned); recall reported @k=10 but agent searches k=8; multimodal bet validated on
much weaker stack than designed (250M visual retriever vs designed 8B — headroom unflagged);
prompt injection unaddressed in DESIGN §11 (see Part 2); "hardware-agnostic" is
endpoint-agnostic (MPS-specific ops are load-bearing); single test file vs promised CI gate.

## Part 2 — Implementation (file:line at commit b722924)

1. [CRITICAL][CONFIRMED] sql tool arbitrary local file read — tools.py:256-268 regex
   guard doesn't block DuckDB reader fns: read_text('/Users/…/.ssh/id_rsa'), read_csv,
   glob('/Users/**') all execute. Model-authored query + injected doc text = local file
   exfiltration into answers + sessions.db. Fix: view allowlist or block reader fns +
   `SET enable_external_access=false` / disabled_filesystems.
2. [HIGH] Indirect prompt injection unmitigated — search_agent.py:14-33/:120: tool
   results appended raw, no untrusted-data delimiting, SYSTEM never marks corpus text as
   data-not-instructions. Multiplier for #1/#6. Fix: delimit + SYSTEM clause.
3. [HIGH][CONFIRMED] Upload path traversal + unbounded read — app.py:143-159:
   `raw / file.filename` escapes via `..` (→ arbitrary file write, unauthenticated);
   whole body read into RAM. Fix: basename only, byte cap, extension allowlist.
4. [HIGH][CONFIRMED] Multi-index doc_id collision — tools.py:65-70 last-writer-wins
   doc2ds + :225-228 CREATE VIEW collision swallowed by bare except: sql returns source
   A's numbers while read_doc/view_page serve source B for the same id. Fix: namespace by
   dataset, detect collisions at init, surface view errors.
5. [MEDIUM][CONFIRMED] Tool budget leaks — search_agent.py:73/:109-116: per-round check,
   parallel calls all execute (27 recorded vs cap 20 in shipped agg traces; reproduced).
   Nudge/correction rounds compound. Fix: enforce cap inside dispatch loop.
6. [MEDIUM][CONFIRMED] grep ReDoS — tools.py:101-103: model-authored regex, no timeout;
   (a+)+$ blowup measured. Fix: timeout / pattern cap.
7. [MEDIUM] sql DoS — tools.py:263 fetchdf() materializes full result before head(50);
   cross-joins pass guard. Fix: wrap with LIMIT, statement timeout, memory_limit.
8. [MEDIUM][CONFIRMED] Empty-sheet page/view divergence — formats.py:79/:116-128:
   catalog page=j+1 skips over empty sheets while pages.jsonl renumbers sequentially →
   evidence panel finds no table for a cited page. Fix: single running counter.
9. [MEDIUM] summarize_doc stale cache after force re-ingest — tools.py:160-162; ingest
   never unlinks summary.md. Fix: content-hash invalidation or unlink on re-parse.
10. [MEDIUM] search_multi chunk-id collision — hybrid.py:161-193: by_id keyed by
    dataset-independent chunk id; colliding doc_ids drop evidence/mis-attribute dataset;
    vis:: id formats differ between search and search_multi. Fix: key by (dataset,id).
11. [LOW-MED] doc_id f-string interpolation into WHERE — store.py:54,63; tools.py:251
    (model-callable). Fix: parameterize.
12. [LOW-MED] Server shares one Retriever/embedder across request threads — app.py:21-27,
    :188-219: violates own single-MPS rule; VisualIndex._cache unlocked. Fix: inference lock.
13. [LOW] fitz handles never closed (tools.py:72,291-292); _build_duckdb non-atomic
    (pipeline.py:95-97 unlink-then-rebuild). Fix: context managers; temp+rename.
14. [LOW] Path traversal via dataset/doc_id in /api/page,/api/evidence (bounded to fixed
    leaf names). Fix: validate params, assert resolved path under processed/.
15. [LOW][CONFIRMED] sql guard false-positives ("WHERE note='please update'" rejected).
    Subsumed by #1's allowlist.
16. [LOW] lru_cached config ignores env changes mid-process; served context length
    documented-but-unverified at startup (16K regression would be silent); message
    history unbounded across rounds. Fix: startup context assert; history trim.

## Reviewers' bottom line

Empirical culture better than average (same-session discipline, published negatives,
bug tables) — but: 88.3% survives only as "high-80s ±8pp, self-authored text-layer-
verified benchmark, uncalibrated judge"; vision opt-in directionally credible (hlt_q10
0-for-10 default vs pass with vision) yet overclaimed from one pair skipping legal;
single-LLM rejection probably right, not yet earned; graph deferral weakest (verdict
predates gate; gate unusable due to budget confound). The corrosive root: an unvalidated
9B binary judge underwrites every claim.

## Remediation queue (agreed priority)

P0 (safety, before any further serve/eval): sql external-access lockdown (#1), upload
hardening (#3), injection delimiting + SYSTEM clause (#2), budget cap fix (#5) — note #5
also invalidates "avg 20.4 calls" as evidence the agg budget was truly 20.
P1 (measurement validity): judge calibration vs hand labels; run-artifact provenance;
JUDGE_SYSTEM/gold-rubric reconciliation.
P2 (redo gated conclusions): agg rerun at budget 40–60 + closed-book control; finish
single-LLM A/B (flags equalized); vision replicates incl. legal.
P3: remaining mediums/lows (collisions, ReDoS/DoS caps, stale caches, atomicity,
traversal, README honesty pass).

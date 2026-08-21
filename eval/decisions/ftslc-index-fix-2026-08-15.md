# FTS index on late-chunk tables (arm K) — 2026-08-15

**Verdict: ADOPTED.** Bug fix restoring designed hybrid retrieval on the
late-chunk leg; measured E2E-neutral on all 5 benches, Sonnet-panel-confirmed
zero regressions. Plus one methodological finding: the 4b screen judge was
nondeterministic; now pinned to temperature 0.

## The bug

`IndexingPipeline` created an FTS index only on the base text table; the
late-chunk block indexed vectors into `<table>_lc` and never created one.
Consequence: **every** `_lc` table in existence lacked an FTS index — verified
across all scratch indexes, including the 683-row rfc table — so the hybrid
retriever's FTS leg failed on `_lc` and silently degraded to dense-only
(retrievers.py graceful-degradation path, added 2026-08-12, masked it). The
chip (task_c7eaedfb) guessed "tiny corpora"; the truth is the lc leg has been
dense-only everywhere since late-chunking landed.

## The fix

- `indexing_pipeline.py`: after the late-chunk indexing loop, create
  `text_idx` on the lc table (same guard/naming as the base-table block).
- Existing tables need no re-ingest: `create_fts_index` on the 5 scratch
  `_lc` tables in place; smoke-verified the FTS leg now returns rows on
  `hr_product_v1_lc` and "FTS leg failed" no longer appears in arm-K logs.

## Measurement (arm K: identical code, index present vs absent)

Full 120-row rerun vs the arm-I/HEAD answers (0 errors):

- Mechanical: substring rfc 10→9, docs 17→16, others flat (noise floor);
  cited-documents changed on only **5/96** authored rows; many answers
  byte-identical (hr 20/24, atlas7 19/24); rfc mean wall 74s→59s.
- Expected, and explains the flatness: lc rows carry the same text as base
  rows, so the restored lexical leg mostly re-finds what base FTS already
  found and RRF fusion barely moves.
- Sonnet panel (3 blind voters, 18 cells = 9 genuinely-differing down-flipped
  rows × both arms): **every row judges identically across arms** (8/9 pass
  both, docs_d17 fail both), 2 split votes, 0 cell flips. Zero regressions.

## Judge-noise finding (4b screen)

The 4b coarse screen initially reported authored 69→59 — but 11 of its 24
verdict flips (including all four hr flips) were on **byte-identical
answers**: the judge itself sampled at default temperature. `eval/judge.py`
now passes `options={"temperature": 0}` on the Ollama path. Prior 4b screen
numbers remain valid only as coarse ordering, never row-level evidence —
which is already their documented status.

## Status

- Fix + judge determinism committed. Rebuild is NOT required for existing
  user indexes, but `create_fts_index("text", use_tantivy=False)` on `_lc`
  tables is the one-liner migration; new indexes get it automatically.
- Artifacts: `rfc_e2e_answers_k.jsonl`, `authored_e2e_answers_ftslc.jsonl`,
  `judged4b_{ftslc,rfc_k}.jsonl`, `ftslc_panel_prompts.jsonl`,
  `votes_ftslc_{1,2,3}.jsonl` in the session scratchpad.

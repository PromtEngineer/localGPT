# Plan: retrieval-robustness & index-lifecycle hardening (from the PwC-search review)

**Date:** 2026-08-25 · **Status:** planned, not started · **Source:** review of
https://huggingface.co/blog/pwc-search against the current codebase.

Three workstreams, ordered by value/effort. None touches a synthesis or decomposer
prompt, so none triggers the frozen-prompt full-bench revalidation gate; each has its
own cheaper gate stated below. Explicitly excluded: Matryoshka dimension truncation —
solves a storage/latency problem this app does not have (corpora are hundreds of
chunks; the reranker, not vector search, is the latency bottleneck).

---

## WS1 — Symmetric graceful degradation in hybrid retrieval  (small, do first)

**Problem.** `MultiVectorRetriever.retrieve()` (rag_system/retrieval/retrievers.py)
degrades to dense-only when the FTS leg fails, but the dense leg is unwrapped:
`vec_df = vec_future.result()` propagates, the outer `except` swallows the whole
retrieval, and a transient embedder failure (OOM, corrupted model file, MPS hiccup)
returns zero results even though the FTS leg had answers. Asymmetric by our own
stated design ("the two legs are redundant by design").

**Change.**
1. In the hybrid branch, wrap `vec_future.result()` in try/except mirroring the FTS
   handler: log `⚠️ Dense leg failed (…); continuing FTS-only.`, set `vec_df = None`.
2. If BOTH legs fail, keep today's behavior (propagate to the outer handler) — full
   failure must stay loud.
3. `vector_only` mode still propagates (caller asked for exactly that leg), mirroring
   the existing `fts_only` rule.
4. Same treatment in the document-scoped twin `_search_within_documents()`
   (retrieval_pipeline.py): it already tracks `leg_ran`/`leg_failed` per leg —
   verify the dense leg there really degrades (it appears to; add the test anyway).

**Gate.** Fault-injection unit test: monkeypatch the embedder to raise; assert hybrid
retrieve() returns FTS-leg results and logs the degradation; assert `vector_only`
still raises; assert both-legs-down still fails loudly. Plus one live smoke on the
rfc index. No bench run needed — happy path is untouched (assert byte-identical
results on a normal query before/after as the cheap proof).

**Effort:** ~1 hour. **Risk:** near zero.

---

## WS2 — Full embedding contract in table markers  (medium)

**Problem.** Table markers (schema metadata + JSON sidecar, embedders.py) record
`{embedding_model, normalized}`. Missing: model **revision**, vector **dimensions**,
and the **query-instruction version** (the retriever applies a query-side instruction
prefix; a table built under a different instruction regime silently degrades). The
PwC design pins all of these and validates at query time.

**Change.**
1. Extend the marker dict: `{embedding_model, model_revision, dims, normalized,
   query_instruction_sha, marker_version: 2}`.
   - `model_revision`: HF commit hash when resolvable from the local cache
     (huggingface_hub `try_to_load_from_cache` / repo revision), else "unknown" —
     never block indexing on it.
   - `query_instruction_sha`: sha256 of the exact instruction string in force at
     build time (empty-string hash when none).
2. Writer: `table_schema_metadata()` + `_write_sidecar()` gain the new fields.
3. Reader/validator (`_check_table_identity` in retrievers.py):
   - v2 marker: assert model name (as today), assert dims against the live embedder's
     output width (hard error — this catches the same-width-swap class properly),
     WARN on revision mismatch, WARN once on instruction-sha mismatch.
   - v1 marker (all existing tables): keep today's behavior exactly; no forced
     rebuilds. New builds write v2.
4. Table version does NOT bump (text_pages_v4 stays) — v1 markers stay readable.

**Gate.** Unit tests: v2 round-trip; v1 tolerance; dims-mismatch raises
EmbedderMismatchError; revision/instruction mismatches warn not raise. Then one
fresh index build + 24-row rfc E2E smoke to prove no behavior change on the
happy path.

**Effort:** ~half a day. **Risk:** low; the only sharp edge is being strict on new
fields for OLD tables — the v1 branch above is the guard.

---

## WS3 — Content-hash incremental indexing  (larger, product feature)

**Problem.** `IndexingPipeline.run()` processes every file it is given, every time:
re-running on a folder re-parses, re-enriches (LLM cost), and re-embeds unchanged
documents, and appends duplicate chunks (`document_id` = basename; nothing dedups
against existing rows). The PwC design hashes content and touches only deltas.

**Change (three steps, each shippable alone).**
1. **Skip unchanged** — before parsing, compute sha256 of file bytes; store
   `content_sha` per document in a small per-index manifest
   (`<index>/documents.json`, alongside the existing overview sidecars; the lancedb
   table stays the single source of truth for chunks — the manifest is a cache and
   is rebuilt from scratch if missing). If sha matches, skip the file entirely
   (no parse, no enrich, no embed). Log `↷ unchanged, skipped`.
2. **Replace changed** — when sha differs for a known `document_id`: delete that
   document's rows (`tbl.delete(f"document_id = '…'")` — also from the `_lc` table
   when late chunking is on, and its overview record), then index the new version.
   This also fixes today's silent duplicate-append on re-index.
3. **Report deletions** — files present in the manifest but absent from the input
   set are reported (`ℹ️ N indexed document(s) not in this batch`) but NOT deleted:
   `run()` is called with single files by the API path, so absence ≠ removal.
   An explicit `remove_documents(ids)` API handles true deletion.

**Ordering note.** Cross-ref extraction and enrichment operate per-document on
`doc_chunks_map`, so skipping a file wholesale is safe — but `annotate_chunks`
resolves references against `_existing_document_ids`, which now includes skipped
docs; verify resolution still sees them (it reads the table, so it does).

**Gate.**
- Unit: same file twice → second run skips, chunk count unchanged, chunk_ids unique;
  modified file → old rows gone, new rows present; manifest deleted → full re-index,
  no duplicates.
- Bench-grade proof: rebuild the rfc index, run once, touch nothing, run again —
  assert the table row-set is IDENTICAL (ids + text) to a single run. Then the
  24-row rfc E2E smoke at 4b-det to confirm no retrieval drift.
- Cost proof (the point of the feature): time and LLM-call counts for a no-op re-run
  before/after — expect ~zero enrichment calls after.

**Effort:** ~1 day incl. tests. **Risk:** medium — deletion is the sharp edge;
mitigations: delete scoped to exact `document_id` match, manifest is regenerable,
and step 1 alone (pure skip, no deletes) can ship first.

---

## Sequencing

WS1 → WS2 → WS3 step 1 → WS3 steps 2–3. Each lands as its own commit with its gate
results in the message. Nothing here alters ranking, prompts, or defaults; the only
externally visible behavior changes are failure-path (WS1) and re-index cost (WS3).

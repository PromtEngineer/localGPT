# Phase 4.1 + 4.5 — full-document escalation and per-query token tracking — implemented 2026-08-09

**Status of each item, stated up front:**

| Item | Ships as | Default |
|---|---|---|
| 4.1 full-document escalation | flag-gated code path, **unmeasured** | **OFF** |
| 4.5 per-query token tracking | always-on observability | **ON** |

4.1 has **not** been benchmarked against the gold set. It is off, and it should
stay off until someone runs the A/B described in §6. Nothing in this document
claims it improves answers; it claims only that it fires when it is supposed to
and produces the document it says it produces.

No file in `Documentation/` was edited this wave. Proposed documentation diffs
are in §7, to be applied at the adoption gate.

---

## 1. What shipped

### 4.1 — full-document escalation (off by default)

New files:

* **`rag_system/retrieval/document_fetch.py`** — reassembles one document from
  its indexed chunks. Filters the LanceDB text table by `document_id`, orders by
  `chunk_index`, prefers `metadata.metadata.original_text` over the top-level
  `text` column (which carries the contextual-enrichment `Context: …` preamble
  when enrichment is on), and truncates to a token budget. Returns `None`
  — never raises — when the table cannot be opened, the document has no rows, or
  no row carries a usable `chunk_index`. **Order is the point**: a document whose
  chunks cannot be ordered is not escalated at all rather than escalated
  scrambled (DOS-RAG).
* **`rag_system/agent/escalation.py`** — `EscalatingRetrievalPipeline`, the
  trigger and the wiring.

**Token counting is `len(text) // 4`**, stated as such in the module docstring
and surfaced everywhere as `approx_tokens`. No tokenizer is loaded. The budget is
a context-window guard, not an accounting figure, and a 4-chars-per-token
estimate runs slightly conservative on English prose.

**Trigger.** After candidate selection completes — that is, *after* the
evidence-sufficiency retry (§5 of `design_rationale.md`) has had its one attempt
— the same signal the retry uses is read again: the reranker's calibrated
probability when there is one, else the dense contrast score
`(cos_top − cos_background) / (1 − cos_background)`. If it is still below
threshold, the top-ranked chunk's whole document is reassembled and appended to
the synthesis context as:

```
––––– FULL DOCUMENT (escalated): <name> –––––
<document text, in chunk order>
––––– END FULL DOCUMENT –––––
```

The threshold defaults to the retry's own `min_top_score` (0.12) — escalation is
what happens when the retry already ran and the evidence is *still* weak, so the
two are judged against the same bar unless `min_evidence` overrides it. Where the
retry has no signal (`fts_only`, legacy unnormalized tables), escalation has no
signal either and does not fire — same rule, deliberately.

**Bounds.** One document per user query (`max_documents`, enforced by a locked
per-request budget so a decomposed query's parallel sub-queries cannot each
escalate), one token budget, no loop, no LLM call of its own.

**Citations are untouched.** `source_documents` is exactly what it would have
been without escalation. The escalated block is reading material appended to the
synthesis prompt, not a source.

**Event.** `document_escalation` is emitted through the existing
`event_callback`, so it lands in the SSE stream next to `retrieval_retry`.
Payload: `document_id`, `document_name`, `chunks_used`, `chunks_total`,
`approx_tokens`, `truncated`, `signal`, `score`, `threshold`, `token_budget` —
never the document text. `/chat` (non-streaming, no events) gets the same
payloads as a `document_escalation` list on the result.

### 4.5 — per-query token tracking (on by default)

* **`rag_system/utils/ollama_client.py`** — `generate_completion`,
  `generate_completion_async` and `stream_completion` now hand Ollama's
  `prompt_eval_count` / `eval_count` to a `TokenUsageTracker` bound for the
  duration of one user query. `stream_completion` also takes an optional
  `stats` dict out-parameter, since a generator cannot return the final object.
* **`rag_system/agent/loop.py`** — one tracker per `Agent.run()`, with stage
  labels around triage, decomposition, synthesis and verification. The result
  payload (and therefore the SSE `complete` event and the `/chat` response body)
  carries:

```json
"token_usage": {
  "by_stage": {"synthesis": {"prompt_tokens": 634, "output_tokens": 430, "calls": 1}},
  "total": {"prompt_tokens": 634, "output_tokens": 430, "calls": 1, "total_tokens": 1064}
}
```

* **`rag_system/utils/watsonx_client.py`** — returns `prompt_eval_count: 0` /
  `eval_count: 0` and records the call, so a watsonx run reports an honest
  "N calls, 0 tokens counted" rather than looking like a cache hit.
* **`rag_system/api_server.py`** — **no change was needed.** It already returns
  the agent's result dict verbatim from `/chat` and passes it as the `complete`
  event's data, so `token_usage` flows through both endpoints for free. This was
  verified over HTTP (§4).

---

## 2. Honest limitations of the token accounting

* **The retry's reformulation call is counted under `synthesis`.** It is an
  enrichment-model call made *inside* `RetrievalPipeline.run()`, which the agent
  wraps as one "synthesis" stage. Splitting it would require editing
  `retrieval_pipeline.py`, which this wave does not own.
* **There is no `escalation` stage bucket**, because escalation makes no LLM call
  of its own by design — it enlarges the synthesis prompt. Its cost shows up as
  a larger `synthesis.prompt_tokens`, and the size of the appended block is in
  the `document_escalation` event's `approx_tokens`. A stage that never fires
  would have been a lie; an empty bucket is not emitted.
* **`by_stage` omits stages that made no call.** An absent key means "no LLM call
  in that stage", not "zero tokens".
* **Only Ollama reports real counts.** watsonx reports zeros (see above).
* **Embedding calls are not counted.** Ollama's embedding endpoint is not routed
  through these three methods, and the shipped embedder is in-process
  (harrier-oss-v1) so there is no token count to read.

---

## 3. The ownership question — `retrieval_pipeline.py` was NOT touched

**This wave made zero edits to `rag_system/pipelines/retrieval_pipeline.py`.**
(The file does carry uncommitted changes in the working tree — they belong to
another workstream, not to 4.1/4.5. Nothing below touched it.)

This is worth flagging because the brief anticipated a small additive change
there and it turned out to be avoidable. The problem: escalation has to happen
*between* candidate selection and synthesis, and both live inside
`RetrievalPipeline.run()`, which returns only `{"answer", "source_documents"}`.
Exposing the contrast signal on `retrieve_candidates()`'s return value would not
have been enough — `run()` does not pass it out, and by the time the agent sees
the result the answer has already been generated. Escalating from the agent loop
after `run()` returns would have meant a **second** synthesis pass: two
generations, two token streams into the UI.

Instead, `Agent` now constructs `EscalatingRetrievalPipeline` — a subclass that
overrides the two methods `run()` already calls in sequence:

* `retrieve_candidates()` → `super()`, then read the final post-retry evidence
  score and remember the top-ranked chunk's `document_id`;
* `_synthesize_final_answer()` → append the document block to `facts`, then
  `super()`.

Result: single generation pass, zero edits to the other workstream's file. The
handoff between the two hooks is a `threading.local`, so the parallel sub-query
fan-out cannot cross-wire one sub-query's document into another's synthesis.

**What the gate must know:** this couples escalation to two method names in
`retrieval_pipeline.py` — the public `retrieve_candidates()` and the private
`_synthesize_final_answer(query, facts, *, event_callback=None)`. If either is
renamed or its signature changes, escalation stops firing. The constructor logs
a warning when `retrieve_candidates` is absent, and `_plan_escalation` is wrapped
so a failure there can never break retrieval — but a rename would be a silent
loss of the feature, not a crash. If the pipeline is being restructured anyway,
promoting these to a documented seam (e.g. an `extra_context` hook before
synthesis) is the cleaner long-term shape.

---

## 4. Verification

All of this ran on this machine on 2026-08-09. Nothing below is estimated.

### 4a. Static checks

* `python -m py_compile` on every touched Python file: clean.
* `npx tsc --noEmit` (`src/` was touched): exit 0, no output.

### 4b. `eval/smoke_e2e.py`

```
$ .venv/bin/python eval/smoke_e2e.py
...
  wall clock 261.2s

========================================================================
25/25 assertions passed
========================================================================
```

Unchanged from the pre-change baseline, as expected: escalation is off, and
token tracking adds a key nobody asserts on.

### 4c. Scratch test — escalation ON against a throwaway Atlas-7 index

Not added to `eval/` (owned by another workstream this wave). The script built a
throwaway LanceDB index from `eval/corpora/atlas7_service_manual.pdf`
(docling chunker, 100-token chunks → 5 chunks), then drove the agent and a live
RAG API on port 8077. Verbatim output:

```
--- index has 5 chunks for atlas7_service_manual.pdf ---
PASS  4.1b fetch returns a document
PASS  4.1b all chunks reassembled  |  used=5 total=5 rows=5
PASS  4.1b chunk order is ascending  |  indices=[0, 1, 2, 3, 4]
PASS  4.1b not truncated at a large budget
PASS  4.1b document text follows chunk_index order
PASS  4.1b truncates to the token budget  |  approx_tokens=300 budget=300 truncated=True
PASS  4.1b truncated text is a prefix of the full text
PASS  4.1a escalation is OFF by default  |  events=['analyze', 'retrieval_done', 'retrieval_retry', 'retrieval_started', 'token']
PASS  4.1a no escalation payload in the default result
PASS  4.1a escalation fires on a weak query  |  events=['analyze', 'document_escalation', 'retrieval_done', 'retrieval_retry', 'retrieval_started', 'token']
PASS  4.1a event carries name/score/threshold  |  {"document_id": "atlas7_service_manual.pdf", "document_name": "atlas7_service_manual.pdf", "chunks_used": 5, "chunks_total": 5, "approx_tokens": 330, "truncated": false, "signal": "dense_contrast", "score": 0.0431, "threshold": 0.12, "token_budget": 6000}
PASS  4.1a score is below the threshold  |  0.0431 < 0.12
PASS  4.1a within the token budget  |  approx_tokens=330
PASS  4.1a exactly one document escalated  |  count=1
PASS  4.1a citations survive escalation  |  sources=5
PASS  4.1a strong query does not escalate  |  events=['analyze', 'retrieval_done', 'retrieval_started', 'token']
PASS  4.5 token_usage present in the agent result  |  {"by_stage": {"synthesis": {"prompt_tokens": 1084, "output_tokens": 3551, "calls": 2}}, "total": {"prompt_tokens": 1084, "output_tokens": 3551, "calls": 2, "total_tokens": 4635}}
PASS  4.5 has by_stage + total
PASS  4.5 non-zero totals  |  {"prompt_tokens": 1084, "output_tokens": 3551, "calls": 2, "total_tokens": 4635}
PASS  4.5 synthesis stage attributed  |  stages=['synthesis']
    token_usage = {"by_stage": {"synthesis": {"prompt_tokens": 1084, "output_tokens": 3551, "calls": 2}}, "total": {"prompt_tokens": 1084, "output_tokens": 3551, "calls": 2, "total_tokens": 4635}}
PASS  4.5c RAG API started
PASS  4.5c /chat returned 200
PASS  4.5c token_usage in the /chat response body
PASS  4.5c totals are non-zero  |  {"by_stage": {"synthesis": {"prompt_tokens": 1192, "output_tokens": 861, "calls": 1}}, "total": {"prompt_tokens": 1192, "output_tokens": 861, "calls": 1, "total_tokens": 2053}}
    /chat token_usage = {"by_stage": {"synthesis": {"prompt_tokens": 1192, "output_tokens": 861, "calls": 1}}, "total": {"prompt_tokens": 1192, "output_tokens": 861, "calls": 1, "total_tokens": 2053}}

ALL PASSED
```

What each group establishes:

* **(a) it fires on a weak query.** `"summarize the overall approach and its
  implications"` against a coffee-machine service manual scored
  `dense_contrast = 0.0431` against a `0.12` threshold *after* the retry had
  already run and failed to improve it (`retrieval_retry` is in the event list
  on both the off and on runs). With the flag off no `document_escalation`
  event and no result key; with it on, exactly one. A well-matched query
  ("What pressure does the brew boiler operate at during extraction?") scored
  above threshold and did **not** escalate — the trigger discriminates, it does
  not just fire on everything.
* **(b) in-order and inside the budget.** Reassembly used all 5 chunks with
  `chunk_indices == [0,1,2,3,4]`, and each chunk's text was located in the
  assembled string strictly after the previous chunk's — so the output is
  genuinely in `chunk_index` order, not merely composed of the right pieces. At
  a deliberately tiny 300-token budget it truncated to exactly `approx_tokens=300`,
  set `truncated=True`, and the truncated text is a prefix of the untruncated
  text.
* **(c) `token_usage` in a real `/chat` response.** Over HTTP against a RAG API
  started as a subprocess: `1192` prompt / `861` output tokens.

Two things this run also confirms about §2's caveats, visible in the numbers:
`calls: 2` on the escalated in-process run is the retry's reformulation call
*plus* the synthesis call, both billed to `synthesis`; and there is no
`escalation` bucket, because escalation made no LLM call of its own.

**What this does NOT establish:** whether the escalated answer is *better*. No
answer-quality comparison was run. See §6.

---

## 5. Proposed config keys (for `rag_system/main.py`, at the adoption gate)

`main.py` was not touched. Every key below is read through `config.get()` with
the default baked into `DEFAULT_DOCUMENT_ESCALATION` in
`rag_system/agent/escalation.py`, so the code behaves identically whether or not
the profiles declare them. Adding them makes the flag discoverable and
togglable per profile:

```python
# in PIPELINE_CONFIGS["default"]["retrieval"], next to "retry":

            # Full-document escalation (roadmap 4.1). OFF until benchmarked.
            # When the evidence-sufficiency retry above has already run and the
            # evidence is STILL below threshold, reassemble the top-ranked
            # chunk's whole document in chunk_index order and append it to the
            # synthesis context, capped at token_budget. One document, no loop.
            # min_evidence defaults to retry.min_top_score when omitted.
            "document_escalation": {
                "enabled": False,
                "max_documents": 1,
                "token_budget": 6000
            }
```

```python
# in PIPELINE_CONFIGS["fast"]["retrieval"], next to "retry":

            # Off in `fast` for the same reason the retry is: this profile
            # exists to avoid extra work, and escalation only inflates the
            # synthesis prompt.
            "document_escalation": {"enabled": False}
```

Optional fourth key, not declared above because the fallback is the better
default: `"min_evidence": <float>` overrides the trigger threshold
independently of `retry.min_top_score`.

No environment variable was added — the brief did not ask for one and a flag
that is off pending measurement does not need a second way to turn it on.

---

## 6. What must be measured before 4.1 is switched on

The flag exists so this can be answered with numbers rather than intuition:

1. **Answer quality on weak-evidence queries.** Run the gold set with
   `document_escalation.enabled` false/true and judge only the subset where the
   trigger actually fires (on most queries the run is byte-identical, so a
   whole-set average would drown the effect). The relevant comparison is
   end-to-end answer correctness, not retrieval metrics — escalation changes no
   retrieval output, only the synthesis prompt.
2. **The cost.** `token_usage.by_stage.synthesis.prompt_tokens` with and without,
   plus wall-clock. A 6000-token budget is a large prompt increase on a local
   model, and long-context degradation ("lost in the middle") is a real risk on
   a small generation model.
3. **The budget itself.** 6000 is a guess. It should be tuned against the
   generation model's context window and the corpus's document lengths.
4. **Whether the threshold should differ from the retry's.** Inheriting
   `min_top_score` is a defensible default, not a measured one.

Until (1) shows a win on the fire-subset, the honest status is: implemented,
bounded, unmeasured, off.

---

## 7. Proposed documentation diffs (NOT applied — apply at adoption)

Docs describe shipped behaviour, and 4.1 is off, so **the only doc change that
should land before the escalation A/B is the 4.5 one.**

### 7a. `Documentation/design_rationale.md` — new section, ship now (4.5 is on)

Insert after §5 (evidence-sufficiency retry):

```diff
+## 5a. Per-query token accounting
+
+**What ships.** Every Ollama completion the agent makes — streaming or not —
+reports `prompt_eval_count` and `eval_count` on its final object. Those are
+aggregated per user query, bucketed by pipeline stage (`triage`,
+`decomposition`, `synthesis`, `verification`), and returned as `token_usage` on
+the `/chat` response body and in the SSE `complete` event. On by default: it
+costs one dict update per LLM call and adds no request.
+
+The aggregation point is a `ContextVar` in `rag_system/utils/ollama_client.py`
+rather than an argument threaded through every call site, because one
+`OllamaClient` is shared by the agent, the retrieval pipeline, the verifier and
+the decomposer. `await` and `asyncio.to_thread` propagate it; the agent's
+parallel sub-query `ThreadPoolExecutor` copies it explicitly.
+
+Two honest gaps: the retry's reformulation call is billed to `synthesis`,
+because it happens inside `RetrievalPipeline.run()` which the agent labels as
+one stage; and watsonx reports zeros, because the SDK path in use surfaces no
+per-call counts.
```

### 7b. `Documentation/api_reference.md` — ship now

Add `token_usage` to the documented `/chat` response body and to the
`complete` SSE event's data, with the shape shown in §1 above, and the note that
an absent stage key means "no LLM call in that stage".

### 7c. `Documentation/design_rationale.md` — 4.1, hold until measured

When (and only when) the §6 A/B shows a win:

```diff
+## 5b. Full-document escalation
+
+**What ships.** `retrieval.document_escalation` — <default to be set by the
+A/B>. When the evidence-sufficiency retry (§5) has run and the signal is still
+below threshold, the top-ranked chunk's document is reassembled from the index
+in `chunk_index` order and appended to the synthesis context as one delimited
+block, capped at `token_budget` and at one document per query. Chunk citations
+are unchanged. Surfaces as a `document_escalation` SSE event.
+
+**Why in-order.** DOS-RAG: a document handed to the model in its original order
+beats the same text ranked by similarity. A document whose chunks carry no
+usable `chunk_index` is therefore not escalated at all.
+
+**Why bounded.** PEA-CAE's escalate-don't-pre-decide, without the loop:
+search volume correlates only weakly with answer quality, so the escalation is
+one document, once, with no agency over what to read next.
```

### 7d. `Documentation/research_roadmap.md` — at adoption

Mark 4.5 done and 4.1 as implemented-but-gated in the Phase 4 table.

---

## 8. Backlog created by this wave

* **UI does not display token counts.** `token_usage` reaches the browser on the
  `complete` event and is persisted into the turn's steps snapshot via the
  existing `saveStreamedTurn` path (`src/components/ui/session-chat.tsx`, final
  step's `details.token_usage`). Rendering it as a compact
  "· 1.2k in / 340 out" line needs a change in
  `src/components/ui/conversation-page.tsx`, which renders the cascade — out of
  scope for a "minimal `src/`" wave.
* **The non-streaming gateway path drops `token_usage`.** The browser streams
  straight from the RAG API (`RAG_API_BASE_URL/chat/stream`), so the SSE
  `complete` event carries it. But `backend/server.py`'s `_query_rag_api`
  extracts only `answer` and `source_documents` from the RAG API's `/chat`
  response, so the non-streaming fallback loses it. One line in `backend/`
  (not owned this wave) would fix it.
* **Pre-existing off-by-one in the step cascade.** `session-chat.tsx` addresses
  steps positionally (`steps[5]`, `steps[6]`, `steps[7]`) in the
  `sub_query_result`, `sub_query_token`, `final_answer` and `token` handlers.
  Those indices were correct before the `retry` step was inserted at index 3 and
  are now one short — `sub_query_result` writes sub-answers into the
  "Expanding context window" step. **Not introduced by this wave and not fixed by
  it**; the new `document_escalation` handler deliberately uses `findIndex` and
  adds no array entry, precisely so it does not shift these further. Worth a
  dedicated fix that converts every positional access to `findIndex`.

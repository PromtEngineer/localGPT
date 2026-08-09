# Phase 2.3 — cheapen gateway routing — shipped 2026-08-09

Replaces the backend gateway's per-message LLM routing call with a deterministic
gate. Every number on this page was produced by a command run on this machine on
2026-08-09; the commands are at the bottom so each one can be re-run.

**Scope:** `backend/server.py` (routing only), `backend/test_gateway_routing.py`
(new). Docs touched: `Documentation/architecture_overview.md` §1/§2.2/§3,
`Documentation/system_overview.md` §2.1 layer 1 (+ two consequential phrases),
`Documentation/api_reference.md` §1.3, `backend/README.md`. Nothing under
`rag_system/**` was modified — the agent-side triage is untouched and remains the
system's single LLM routing layer.

---

## 1. What was there

`ChatHandler._should_use_rag(message, idx_ids)`, on the non-streaming path only
(`POST /sessions/<id>/messages`):

1. No linked indexes → direct LLM.
2. Otherwise `_load_document_overviews(idx_ids)` read up to 40 overview
   paragraphs off disk, `_route_using_overviews` pasted them into a prompt and
   asked the **enrichment model** (`qwen3.5:4b`) for `USE_RAG` or `DIRECT_LLM`.
   An unparseable reply defaulted to RAG.
3. If either step raised, `_simple_pattern_routing` decided by keyword and
   length.

The pattern fallback was worse than improvement_plan §2.3 records. It matched
its greeting list with `pattern in message_lower` — **substring**, not word —
so `'hi'` matched *w**hi**ch*, *t**hi**s* and *mac**hi**ne*, and `'ok'` matches
*b**ook***. Re-running the deleted function verbatim on eight Atlas-7 questions
(script in §5) routes **7 of 8 to direct LLM**:

```
DIRECT  Which test procedure applies after replacing the pump?          <- matched ['hi', 'test']
DIRECT  What is the test point voltage on the control board?            <- matched ['test']
DIRECT  Where is the serial number engraved?                            <- matched []
DIRECT  What does this manual say about the steam boiler?               <- matched ['hi']
DIRECT  How long is the Atlas-7 parts warranty?                         <- matched []
DIRECT  Which sensor part should be replaced when error code E11 appears? <- matched ['hi']
RAG     What pressure does the brew boiler operate at during extraction? <- matched []
DIRECT  Who manufactures the machine?                                   <- matched ['hi']
```

(`Where is the serial number engraved?` and `How long is the Atlas-7 parts
warranty?` matched no greeting at all — they went direct for the *other*
reason: no `rag_indicator` keyword and under the 40-character question rule.)
This is why the function is deleted rather than patched.

## 2. What ships now

Two module-level functions in `backend/server.py` (module-level so they are
unit-testable without an HTTP request):

```python
should_use_rag(message, idx_ids, force_rag=False) -> bool
is_smalltalk_or_meta(message) -> bool
```

The cascade is retrieval-first — *escalate, don't pre-decide*:

| Condition | Route |
|---|---|
| `force_rag` | RAG (unchanged) |
| No linked indexes | direct LLM (unchanged) |
| Whole message matches the smalltalk allowlist (≤ 6 words) | direct LLM |
| Whole message matches the assistant-meta allowlist | direct LLM |
| Everything else | **RAG** |

Both allowlists are anchored, whole-message regexes:

* **Smalltalk** — the message must consist *only* of allowlisted phrases
  (greetings, thanks, farewells, acknowledgements) plus inert filler
  (`there`, `again`, `so much`, …), **and** contain at least one *core*
  phrase, **and** be at most `SMALLTALK_MAX_WORDS = 6` words. `"hello"`,
  `"thanks!"`, `"got it, thanks"` match. `"Hello, what is the brew boiler
  pressure?"` does not.
* **Assistant-meta** — a tight list of self-referential questions
  (`who are you`, `what model are you`, `which model do you use`,
  `are you an AI`, `who built you`, `tell me about yourself`), anchored so
  `"Who are the authors of the service manual?"` and `"What model number is
  the pressure sensor?"` fall through to RAG.

Deleted entirely: `_should_use_rag`, `_route_using_overviews`,
`_load_document_overviews` (its only caller was the router) and
`_simple_pattern_routing` — 212 lines. `used_rag` is still returned on
`POST /sessions/<id>/messages`, and `force_rag` behaviour is unchanged
(it also still goes into the payload forwarded to the RAG API).

**Why over-sending to RAG is safe.** `Agent._triage_query_async` in
`rag_system/agent/loop.py` runs on every request the gateway forwards and can
still return `direct_answer` without retrieving. A false "use RAG" therefore
costs one triage call on a model that would have been called anyway; a false
"answer directly" costs an unanswerable question. The gate is biased
accordingly. This is why the gateway can afford a gate this crude — it is not
the decision-maker, it is a smalltalk filter in front of the decision-maker.

**Evidence** (`Documentation/research/`): pre-retrieval LLM routing is the
weakest measured pattern of 2026 — four ML approaches failed because "the need
for augmentation cannot be determined from the query alone"; fixed-hybrid beat
rule-based adaptive routing; cheap discriminative gates match LLM routers at
~zero cost. Roadmap item 2.3.

## 3. Measurements

### 3.1 Latency of the routing decision — 20 sample messages

Same 20 messages, same machine, same session overviews (the Atlas-7 service
manual index), warm-up call excluded, Ollama warm.

| | old (`_should_use_rag`, enrichment-model call) | new (`should_use_rag`) |
|---|---|---|
| total, 20 messages | 15012.258 ms | 0.040 ms |
| mean | **750.613 ms** | **0.002 ms** |
| median | 755.507 ms | 0.002 ms |
| min / max | 729.089 / 760.750 ms | 0.001 / 0.006 ms |

**Saving: ~750.6 ms per non-streaming message** (mean 750.613 → 0.002 ms), plus
one fewer `qwen3.5:4b` generation and one fewer overview file read per message.
The old path's cost was flat across message types — even `"hello"` paid the
full ~750 ms — because the LLM call happened before any decision.

Caveats, stated plainly: this is one machine, one warm Ollama, one small
overview set (1 overview paragraph). A larger corpus makes the *old* number
worse (up to 40 overviews in the prompt), never better. The new number is
independent of corpus size.

### 3.2 Decision agreement

On those same 20 messages the new gate reproduced the old LLM router's decision
**20/20** (13 RAG, 7 direct): all 13 document questions RAG, and `hello`,
`hi there`, `thanks!`, `thank you so much`, `bye`, `who are you?`,
`what model are you` direct. Not a benchmark — 20 messages on one corpus — but
it is the same behaviour at 1/375,000th the cost.

### 3.3 Unit tests — `backend/test_gateway_routing.py`

155/155 checks pass (exit 0). Sections: planted-fact questions route RAG;
smalltalk and assistant-meta route direct; **messages containing "test" /
"check" route RAG** (the old defect); `force_rag` honoured in every
combination; sessions with no indexes always route direct; the gate's source
contains no LLM/network call and the four deleted methods are gone from
`ChatHandler`.

### 3.4 End-to-end smoke — `eval/smoke_e2e.py`

**25/25 assertions passed**, exit 0, wall clock 296.8 s (`.venv/bin/python eval/smoke_e2e.py`,
2026-08-09, after the change). All four planted facts (`9.2`, `TS-71`, `36`,
`drip tray`) came back with non-empty `source_documents` and a
`[Confidence: N%]` tag, and the streamed-turn save/round-trip assertions held.

Stated honestly: **smoke sends `force_rag: True`** (`eval/smoke_e2e.py:188`), so
it exercises the gate's `force_rag` branch and the RAG forward path, not the
discriminative branch. §3.5 covers that gap.

### 3.5 HTTP-level gate verification (the branch smoke does not reach)

Backend started against a throwaway SQLite file (`DB_PATH=…/gate_db.sqlite`)
with `RAG_API_URL` deliberately pointed at a dead port, so a forwarded request
is visible as an error from the RAG API rather than an answer. A session was
linked to an index row; `used_rag` in the response is the gate's decision:

```
-- session WITH index linked --
  used_rag=True   'What pressure does the brew boiler operate at during ex'  -> Error from RAG API (501)…   [forwarded]
  used_rag=True   'Which test procedure applies after replacing the pump?'   -> Error from RAG API (501)…   [forwarded]
  used_rag=False  'hello'         -> Hello! How can I help you today? …
  used_rag=False  'thanks!'       -> You're very welcome! …
  used_rag=False  'who are you?'  -> I'm **Qwen3.5**, the latest large language model …

-- force_rag on smalltalk --
  used_rag=True   'hello' + force_rag -> Error from RAG API (501)…   [forwarded]

-- session with NO index --
  used_rag=False  'What pressure does the brew boiler operate at during ex'  -> (direct answer)
  used_rag=False  'hello'                                                    -> (direct answer)
```

The "test" question — the old defect — is forwarded to RAG over real HTTP, and
`used_rag` is still present and correct in every response body.

### 3.6 Worst-case regex cost

The smalltalk regex is an alternation under a `*` quantifier, so it was checked
for catastrophic backtracking: `"hi " × 200`, `"thanks so much and " × 50`,
`"who are you " × 40`, a 5 KB tail after a meta prefix and a 20 KB blob all
resolve in 0.004–0.011 ms. The ≤ 6-word cap runs before the smalltalk regex, and
the meta regex is anchored at both ends, so neither can be driven into a blowup.

---

## 4. Proposed `improvement_plan.md` rows

*(This agent does not edit `improvement_plan.md` or `research_roadmap.md`.)*

**Add to §0 Landed:**

| Area | Change | Verify at |
|------|--------|-----------|
| Routing | **Roadmap 2.3 — gateway routing is a deterministic gate.** The per-message enrichment-model router and the `_simple_pattern_routing` keyword/length fallback are deleted; `should_use_rag()` routes on `force_rag` → linked indexes → a whole-message smalltalk/assistant-meta allowlist → RAG. ~750 ms/message saved; agent triage is now the only LLM routing layer | `backend/server.py::should_use_rag`, `backend/test_gateway_routing.py` (155/155), `eval/decisions/phase2-gateway.md` |

**Remove from §2 Routing / triage (Open):**

| ID | Item | Why it closes |
|----|------|---------------|
| 2.3 | Retire the keyword fallback | `_simple_pattern_routing` no longer exists; the "test" misroute is covered by a regression test |

**Amend §2 item 2.1** ("Embed and cache document overviews — *both routers* make
an LLM call per query"): only one router does now. The item still stands for the
agent-side router, but its rationale should say so.

**Observation, not a claim of ownership — §2 item 2.4** ("Make `force_rag` mean
one thing: on the gateway it selects the RAG route but is not forwarded"). In
the current tree it *is* forwarded: `handle_session_chat` sets
`options["force_rag"] = True` and `_handle_rag_query` does `payload.update(options)`.
That row looks stale against the code today; whoever owns it should re-verify.

---

## 5. Commands

```bash
# unit tests (no HTTP, no LLM)
.venv/bin/python backend/test_gateway_routing.py

# end-to-end smoke (starts both services against throwaway stores)
.venv/bin/python eval/smoke_e2e.py

# HTTP-level gate check (§3.5): backend only, throwaway DB, RAG API pointed at a
# dead port so a forwarded request is unmistakable
DB_PATH=/tmp/gate_db.sqlite RAG_API_URL=http://localhost:8899 \
  .venv/bin/python backend/server.py &
# then: create a session, POST /indexes, link them, and POST messages with and
# without force_rag, reading `used_rag` out of each response.

# latency: the old path was timed before the refactor by calling
# ChatHandler._should_use_rag on 20 messages with the Atlas-7 overviews linked;
# the new path times server.should_use_rag on the identical list.
# Harness: eval/decisions/phase2-gateway-bench.py (see below)
```

The benchmark harness is 60 lines: it imports `backend/server.py`, times
`should_use_rag(message, idx_ids)` over the 20 messages listed in §3.1 with
`perf_counter`, and reports total/mean/median/min/max. To reproduce the "old"
column, check out the pre-change `backend/server.py`, instantiate
`ChatHandler.__new__(ChatHandler)` with an `OllamaClient`, and call
`_should_use_rag(message, ["<index_id>"])` from the repo root so the relative
overview paths resolve.

---

## 6. Stale elsewhere — not owned by this change

`Documentation/triage_system.md` still documents the deleted gateway router in
detail (`_should_use_rag`, `_load_document_overviews`, `_route_using_overviews`,
`_simple_pattern_routing`, with line numbers) at lines 4, 36, 39–40, 42, 82 and
101. That file is owned by another agent in this phase and was deliberately not
touched here. Every gateway-routing statement in it is now false and needs the
§2 cascade above.


---

**Gate resolution (2026-08-09):** §6's flag about `Documentation/triage_system.md` is
resolved — the file was rewritten at the validation gate; every gateway-routing
statement now describes `should_use_rag()`.

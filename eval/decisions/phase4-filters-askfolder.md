# Phase 4 items 4.4 and 4.6 — metadata filter DSL and ephemeral "ask a folder"

Date: 2026-08-09
Status: **implemented; 4.4 is inert unless a caller passes `filters`, 4.6 is a new
CLI subcommand. Neither is on any gold-set metric, and neither can be.**
Owner of this wave: `rag_system/retrieval/`, `rag_system/api_server.py`, the
filters plumbing in `rag_system/agent/loop.py` and
`rag_system/pipelines/retrieval_pipeline.py`, and the **CLI section** of
`rag_system/main.py`. The `PIPELINE_CONFIGS` blocks, `Documentation/**` and
`backend/server.py` were deliberately **not** edited — the config keys and diffs
they need are *proposed* at the bottom of this file and belong to the adoption
gate.

Read the honesty section first: **there is no retrieval-quality evidence in this
document, because neither feature is a retrieval-quality change.** 4.4 changes
*what the caller is allowed to ask for*; when no filter is supplied it does
nothing at all, and that "nothing at all" is the strongest claim here — it is
proven byte-for-byte below. 4.6 is packaging around the pipeline that already
ships. What follows is a mechanism proof and a non-regression proof, not a
benchmark.

---

## 1. What shipped

### 4.4 Metadata filter DSL

**New module** — `rag_system/retrieval/filters.py`.

JSON, not a filter string. The roadmap's source surface is
`semantic_search(filters="field=value, field in (a,b)")`, a *string* that has to
be parsed; a parser is exactly the component an injection attack aims at. A JSON
object arrives already parsed, so the only work left is validation — and
validation is the whole security story:

```json
{"document_id": "07_nda.pdf"}
{"document_id": {"in": ["07_nda.pdf", "03_ip_certification.pdf"]}}
{"document_name": {"contains": "nda"}, "chunk_index": {"gte": 0, "lte": 4}}
```

| field | column | operators |
|-------|--------|-----------|
| `document_id` | `document_id` | `eq`, `in`, `contains` |
| `document_name` | `document_id` (substring) | `contains` only |
| `chunk_id` | `chunk_id` | `eq`, `in` |
| `chunk_index` | `chunk_index` | `eq`, `in`, `gt`, `gte`, `lt`, `lte` |

Top-level keys are ANDed. A bare scalar is shorthand for `eq`, which is the
roadmap's `field=value`. There is no OR, no NOT, no nesting: nothing has asked
for them, and every operator is another string that ends up inside a SQL
predicate.

`document_name` deserves its own line. It is **not** a column. Document ids are
the file's basename for anything indexed by the CLI and `<uuid>_<basename>` for
anything uploaded through the UI, so matching a name means substring-matching
the id — hence `contains` and *only* `contains`. Offering `eq` there would
silently miss every UI-uploaded document, which is a trap rather than a feature.

Security, following `rag_system/retrieval/document_fetch.py`'s precedent
(it refuses ids containing quotes/backslashes rather than escaping them):

* **Refuse, don't escape.** A string value containing `'`, `"`, `\`, `;`,
  backtick, or any control character raises `FilterError`. Nothing is repaired.
* The one exception is the `LIKE` metacharacters `%` and `_`, which *are*
  escaped, with an explicit `ESCAPE '\'` clause, so that `contains` means
  substring literally — otherwise `contains: "01_acquisition"` would match
  `01Xacquisition` and the filter would be quietly wider than it reads.
* Types are checked (`bool` is rejected for an int field even though Python
  says `isinstance(True, int)`), strings are capped at 256 characters, IN-lists
  at 256 items, integers at the int32 range the column actually holds.
* **Fail loud.** Unknown field, unknown operator, wrong type, empty IN-list and
  the **empty object `{}`** all raise. `{}` in particular: a client bug that
  sends an empty filter must not be indistinguishable from an unfiltered
  search. Omitting the key entirely is how you ask for no filter.

Compilation is deterministic — fields are emitted in a fixed canonical order and
operators sorted, so the same filter object always produces the same
where-clause regardless of JSON key order. No LLM anywhere. (The roadmap's
LLM filter-*extraction* — natural language to filter — is explicitly "later";
this is the deterministic layer it would target.)

**Retriever** — `rag_system/retrieval/retrievers.py`.

`MultiVectorRetriever.retrieve()` gains a **keyword-only** `where=None`. It is
applied as a `prefilter=True` `.where()` on **both** legs — the vector search
and the BM25/FTS search — so a hybrid query cannot leak an excluded chunk in
through the lexical side. Prefiltering rather than post-filtering is the point:
post-filtering returns "up to k, minus whatever the filter removed", so a filter
for a rare document would come back empty while the document sat in the table.

One deliberate behaviour change on the filtered path only: when `where` is set
and the search raises, the exception is **re-raised** instead of being swallowed
into `return []`. A filtered search that failed and a filtered search that
matched nothing are different answers, and only one of them is safe to show.

**Pipeline** — `rag_system/pipelines/retrieval_pipeline.py`.

The compiled filter travels as a **thread-local scope** (`filter_scope()` /
`active_filter()`) opened by `run()`, plus a keyword-only `filters=` on
`retrieve_candidates()` for direct callers such as the eval harness. The
thread-local is not decoration: `EscalatingRetrievalPipeline` (roadmap 4.1)
overrides `retrieve_candidates` with a fixed four-argument signature and calls
`super()` **positionally**, so a new parameter that `run()` had to pass would
break it. A scope is invisible to that subclass, and the agent's parallel
sub-query fan-out enters `run()` *inside* each worker thread, so each worker
sets its own — verified with a concurrency test below.

Every path that reaches LanceDB is narrowed, not just the first stage:

* `_first_stage` — the main table and the late-chunk table.
* `_search_within_documents` — ANDs the filter into the internal
  `document_id IN (…)` clause, so the **cross-reference hop (4.2)** cannot be
  used to reach a document the caller filtered out, and the **overview
  prefilter's restrict mode (4.3)** stays inside the filter.
* `_get_surrounding_chunks_lancedb` — context expansion. Without this, a
  `chunk_index <= 0` filter would still pull chunk 1 back in as a neighbour and
  "nothing that fails the filter reaches synthesis" would be false.

When a filter is active, `retrieve_candidates` adds
`result["filters"] = {"spec": …, "where": …}` and emits a `filters_applied`
event. When there is no filter the result dict is unchanged — no new key.

**Agent** — `rag_system/agent/loop.py` (plumbing only).

`Agent.run(..., filters=…)` keyword-only; compiled once per user query and
handed to every retrieval that query performs, including each parallel
sub-query. Two non-obvious consequences, both deliberate:

* **The semantic cache is filter-aware.** "What does the NDA say" and "what do
  all ten documents say" have near-identical embeddings and different right
  answers, so a cache entry now records its filter signature and only matches a
  request with the same one. Both sides are `None` on the unfiltered path.
* **A filter skips triage**, the way `force_rag` does. Someone who filtered to a
  named document has already decided the question is about the documents;
  letting triage answer from general knowledge would ignore the filter and look,
  from the outside, exactly like a filter that matched nothing.

**API** — `rag_system/api_server.py`.

`/chat` and `/chat/stream` accept an optional `filters` object. It is compiled
in `_parse_chat_request`, so an invalid filter is a **400 before any retrieval
work happens**, and `filters` is only added to the `Agent.run` kwargs when
present — an unfiltered request reaches the agent with exactly the arguments it
always did.

**CLI** — `python -m rag_system.main chat "<q>" --filters '<json>'`, same DSL,
same validation (bad JSON or an invalid filter exits 2 with the message).

*Unrelated one-line fix in the same file*: `python -m rag_system.api_server
--port N` previously **accepted `--port` and ignored it**, silently listening on
8001. It now parses it. This was found by having a test talk to the wrong
server; leaving it would have been leaving a trap.

### 4.6 Ephemeral "ask a folder"

**New module** — `rag_system/ask_folder.py`. **CLI** — `rag_system/main.py`:

```
python -m rag_system.main ask <folder> "<question>" ["<question>" ...]
        [--mode {fast,default}] [--interactive] [--agent] [--filters JSON] [--keep]
```

Index the folder's supported files into a throwaway LanceDB table, answer, delete
everything. No parallel pipeline: `IndexingPipeline` builds the index and the
standard `Agent` object answers — via `agent.retrieval_pipeline.run()` by
default (the roadmap's "same pipeline, **no agent loop**") or via `agent.run(…,
force_rag=True)` under `--agent`, which adds decomposition and verification.

Profile is `fast` per the roadmap, with two further switch-offs on top:
contextual enrichment (an LLM call per chunk) and **document overviews** (an LLM
call per document). Overviews only feed the agent's triage router and the
default-off overview prefilter, and neither runs here — the answer path skips
triage because someone who typed `ask <folder>` has already decided the question
is about the folder.

Cleanup is the feature, so it is arranged to be simple enough to be obviously
correct: **everything the run writes lives under one `tempfile.mkdtemp`
directory.** `storage.lancedb_uri`, `storage.db_path` and `overview_path` all
point inside it, so a single `rmtree` in a `finally` is the entire teardown, and
the directory is printed at the start and its removal reported at the end.
`--keep` skips deletion and says so loudly.

`SIGTERM` is temporarily rebound to raise `SystemExit` so the `finally` actually
runs. This is not hypothetical: the first end-to-end run of this feature was
killed by a 2-minute harness timeout (`SIGTERM`) mid-synthesis and **leaked its
temp directory**, because Python's default SIGTERM disposition exits without
unwinding. `SIGINT` already unwound. `SIGKILL` cannot be caught and will still
leak a `localgpt-ask-*` directory under `$TMPDIR`; that is a property of
`kill -9`, and it is documented rather than papered over.

---

## 2. Verification

Everything below was executed on this machine today. Scratch scripts live in the
session scratchpad (not in `eval/`); paths are given so the runs can be redone.

### 2.1 `py_compile`

```
$ .venv/bin/python -m py_compile rag_system/main.py rag_system/ask_folder.py \
    rag_system/retrieval/filters.py rag_system/retrieval/retrievers.py \
    rag_system/pipelines/retrieval_pipeline.py rag_system/agent/loop.py \
    rag_system/api_server.py
py_compile OK (7 files)
```

`npx tsc --noEmit` was **not** run: no file under `src/` was touched. The
frontend sends no `filters` and is unaffected.

### 2.2 No-filter behaviour is byte-identical to pre-change

The claim that matters most, so it was measured rather than argued.

A pre-change copy of the tree was reconstructed in a scratch directory:
`retrievers.py` straight from `git HEAD` (it carried no working-tree changes
before this wave), and `retrieval_pipeline.py` as the current file with **this
wave's edits reversed one by one, every reversal asserting that its target was
present** (`scratchpad/make_baseline.py`; it also asserts the result mentions
none of `filter_scope`, `compile_filters`, `active_filter`, `filter_where`, and
then compiles it). `git HEAD` is not a usable baseline for
`retrieval_pipeline.py` because wave 1's cross-reference/overview-prefilter work
is in the working tree and not in HEAD.

Both trees then ran the same five queries through
`RetrievalPipeline.retrieve_candidates()` on the existing eval index
`eval/.eval_indexes/microsoft__harrier-oss-v1-0.6b/acq`, table `eval_acq`
(13 chunks, 10 PDFs), dumping per-candidate `chunk_id`, `document_id`,
`chunk_index`, `score`, `_distance`, `bm25`, text length and a text hash. The
evidence-sufficiency retry was forced off, because it calls an LLM to reformulate
and the two arms would not be comparable otherwise.

```
$ cmp candidates_baseline.json candidates_after.json
IDENTICAL: candidates_baseline.json == candidates_after.json
$ md5 -q candidates_baseline.json candidates_after.json
63e4edbcf805bdf87498c6c26fb36541
63e4edbcf805bdf87498c6c26fb36541
$ wc -c candidates_baseline.json candidates_after.json
   41602 candidates_baseline.json
   41602 candidates_after.json
```

**Caveat, stated plainly**: this is five queries on one 13-chunk index with the
retry off, not the gold set. It proves the no-filter code path is unchanged; the
code-level argument is what generalises it — `compile_filters(None)` returns
`None`, the scope is never entered, `active_filter()` returns `None`,
`where=None` makes the retriever's `_filtered()` helper the identity, and
`combine(x, None)` returns `x`. Every changed expression reduces to the original.

### 2.3 Filter DSL unit tests — 36/36

`scratchpad/test_filters_unit.py`. Full verbatim output is in the session log;
the compiled forms and a sample of the refusals:

```
  OK   document_id scalar eq         -> document_id = '07_nda.pdf'
  OK   document_id in-list           -> document_id IN ('07_nda.pdf', '03_ip_certification.pdf')
  OK   document_name contains        -> document_id LIKE '%01\_acquisition%' ESCAPE '\'
  OK   chunk_index range             -> chunk_index >= 0 AND chunk_index <= 4
  OK   multi-field AND               -> document_id IN ('a.pdf', 'b.pdf') AND chunk_index <= 0
  key order A -> document_id = 'x.pdf' AND chunk_index >= 1 AND chunk_index <= 3
  key order B -> document_id = 'x.pdf' AND chunk_index >= 1 AND chunk_index <= 3
  OK   identical regardless of JSON key order

  OK   single quote in value -> FilterError: filters.document_id.eq contains a forbidden character ("'"); quoting characters are refused, not escaped.
  OK   semicolon / stacked statement -> FilterError: filters.document_id.eq contains a forbidden character (';'); ...
  OK   backslash -> FilterError: filters.document_id.eq contains a forbidden character ('\\'); ...
  OK   unknown field -> FilterError: Unsupported filter field(s): page. Supported: document_id (eq, in, contains); document_name (contains); chunk_id (eq, in); chunk_index (eq, in, gt, gte, lt, lte).
  OK   bool as chunk_index -> FilterError: filters.chunk_index.eq must be an integer, got bool.
  OK   empty object -> FilterError: filters is empty. Omit the field entirely to search without a filter; ...

36 passed, 0 failed
```

### 2.4 Filtered retrieval on the real eval index — 37/37

`scratchpad/test_filters_pipeline.py`, same `eval_acq` index. Verbatim:

```
== 1. unfiltered baseline (what the filter has to change) ==
  hybrid      13 chunks over 10 documents: [...all ten PDFs...]
  vector_only 13 chunks over 10 documents: [...all ten PDFs...]
  fts_only    10 chunks over 10 documents: [...all ten PDFs...]
  [PASS] hybrid: unfiltered result has no 'filters' key  — keys=['documents', 'first_stage', 'query_used', 'retry']

== 2. document_id equality restricts BOTH legs ==
  [PASS] hybrid: only 07_nda.pdf returned  — 1 chunk(s), documents=['07_nda.pdf']
  [PASS] vector_only: only 07_nda.pdf returned  — 1 chunk(s), documents=['07_nda.pdf']
  [PASS] fts_only: only 07_nda.pdf returned  — 1 chunk(s), documents=['07_nda.pdf']
  [PASS] hybrid: result carries the applied where-clause  — {'spec': {'document_id': '07_nda.pdf'}, 'where': "document_id = '07_nda.pdf'"}

== 2b. the raw retriever, one leg at a time ==
  [PASS] MultiVectorRetriever fts_only where=... → only the NDA  — 1 row(s) filtered vs 10 unfiltered over 10 documents
  [PASS] MultiVectorRetriever vector_only where=... → only the NDA  — 1 row(s) filtered vs 13 unfiltered over 10 documents

== 3. other operators ==
  [PASS] IN-list returns exactly those two documents  — ['03_ip_certification.pdf', '07_nda.pdf']
  [PASS] document_name contains 'nda' → 07_nda.pdf  — ['07_nda.pdf']
  [PASS] underscore is literal, not a wildcard  — ['01_acquisition_agreement.pdf']
  [PASS] chunk_index <= 0 keeps only first chunks  — chunk_index values [0]
  [PASS] two fields AND together  — [('01_acquisition_agreement.pdf', 1)]

== 4. a filter that matches nothing does NOT fall back ==
  [PASS] empty result, not an unfiltered one  — 0 chunk(s)

== 5. invalid filters fail loud at the pipeline boundary ==
  [PASS] unknown field rejected / unknown operator rejected / empty object rejected /
         wrong type rejected / injection: quote rejected / injection: stacked rejected /
         injection: backslash rejected            (all FilterError, messages as in §2.3)

== 5b. the table is intact after every injection attempt ==
  [PASS] eval_acq still has 13 rows  — 13 rows

== 6. pipeline.run(filters=...) validates before doing any work ==
  [PASS] run() rejects an invalid filter  — FilterError: Unsupported filter field(s): bogus. ...

== 7. internally-scoped searches are narrowed too (crossref hop / restrict mode) ==
  [PASS] _search_within_documents honours the active filter  — scoped=[0] unscoped=[0, 1]

== 8. concurrent queries with different filters do not cross-wire ==
  [PASS] thread A saw only its own filter  — ['07_nda.pdf']
  [PASS] thread B saw only its own filter  — ['03_ip_certification.pdf']
  [PASS] unfiltered thread was unaffected  — [...all ten PDFs...]
  [PASS] no filter leaked into the next unfiltered query  — keys=['documents', 'first_stage', 'query_used', 'retry']

37/37 assertions passed
```

Note on §2 and §2b: with `retrieval_k=20` against a 13-chunk index, "1 chunk"
*is* the whole of `07_nda.pdf` — the NDA is a single chunk. The
`1 filtered vs 13 unfiltered` comparison is what makes the assertion
non-vacuous.

### 2.5 HTTP end-to-end — `/chat` and `/chat/stream`

`scratchpad/test_filters_http.py` starts the RAG API as a child process against
the same `eval_acq` index (`LANCEDB_PATH`) and a throwaway chat DB, then drives
it over HTTP. **21/21 assertions passed.** The nine malformed/injecting filters
were each sent to *both* endpoints:

```
  rag-api healthy on 8011

== malformed filters are 400, on both endpoints ==
  [PASS] /chat unknown field → 400  — status=400 body={ "error": "Invalid filters: Unsupported filter field(s): page. Supported: document_id (eq, in, contains); document_name (contains); chunk_id (eq, in); chunk_index (eq, in, gt, g…
  [PASS] /chat unknown operator → 400  — status=400 body={ "error": "Invalid filters: filters.document_id does not support the 'regex' operator. Supported: eq, in, contains." }
  [PASS] /chat empty object → 400  — status=400 body={ "error": "Invalid filters: filters is empty. Omit the field entirely to search without a filter; an empty filter object is refused so that a client bug cannot look like an unfi…
  [PASS] /chat wrong type → 400  — status=400 body={ "error": "Invalid filters: filters.chunk_index.eq must be an integer, got str." }
  [PASS] /chat not an object → 400  — status=400 body={ "error": "Invalid filters: filters must be a JSON object, got str. Supported fields: …" }
  [PASS] /chat injection: quote → 400  — status=400 body={ "error": "Invalid filters: filters.document_id.eq contains a forbidden character (\"'\"); quoting characters are refused, not escaped." }
  [PASS] /chat injection: stacked statement → 400  — status=400 body={ "error": "Invalid filters: filters.document_id.eq contains a forbidden character (';'); quoting characters are refused, not escaped." }
  [PASS] /chat injection: backslash → 400  — status=400 body={ "error": "Invalid filters: filters.document_id.eq contains a forbidden character ('\\\\'); quoting characters are refused, not escaped." }
  [PASS] /chat injection: LIKE wildcard smuggling via contains → 400  — status=400 body={ "error": "Invalid filters: filters.document_name.contains contains a forbidden character (\"'\"); quoting characters are refused, not escaped." }
  … the same nine, verbatim, against /chat/stream: all [PASS] 400.

== a valid filter restricts a real answer ==
  [PASS] /chat with a valid filter returns 200  — 200
  [PASS] every cited chunk is from the filtered document  — 1 source(s) from ['07_nda.pdf'] in 59.9s

  answer: Answer:
According to the Mutual Non-Disclosure Agreement ("NDA") entered into as of October 1, 2024, by TechCorp Industries, Inc. (located at 500 Technology Drive, San Francisco, CA 94105) and StartupXYZ LLC (located at 123 Innovation Way, Palo Alto, CA 94301), the confidentiality obligations are explicitly defined in Section 2 of the document ("OBLIGATIONS"). Each Party agrees to adhere to the fo…

== the same question unfiltered cites other documents ==
  [PASS] unfiltered answer spans more than one document  — ['01_acquisition_agreement.pdf', '07_nda.pdf', '10_closing_checklist.pdf']

21/21 assertions passed
```

The last pair is the one that matters: same question, same index, `retrieval_k=3`
— filtered cites one document, unfiltered cites three.

### 2.6 4.6 end-to-end on `eval/corpora/acquisition/` (10 PDFs)

One real question, one real answer, exit code 0.

```
$ .venv/bin/python -m rag_system.main ask eval/corpora/acquisition \
    "What is the total purchase price of the acquisition and how is it broken down?"

📂 ask: 10 file(s) from /Users/prompt/videos/localgpt_08082026/localGPT/eval/corpora/acquisition
🗑️  ephemeral index: table 'ask_d74a23299d65' in /var/folders/…/T/localgpt-ask-jf_scljx (profile 'fast')
⏱️  indexed in 9.4s
```

Answer (tail; the corpus's planted numbers are `$45,000,000` original,
`$43,330,000` adjusted, and a `$39,630,000` total at closing):

```
…t Adjustment ($175,000); Revenue Recognition Impact (Implied value adjustment at 15x) ($1,275,000).
*   Adjusted Purchase Price: **$43,330,000**.

Section 6 of the Financial Adjustments Memo provides a revised payment structure reflecting these
adjustments ("As revised from Document: Acquisition Agreement Section 2.2"):
*   (a) Cash at closing: **$28,330,000** (adjusted);
*   (b) Stock consideration: **$10,000,000**; and
*   (c) Earnout payments: **$5,000,000**.

Additionally, a Contingent Liability Reserve of **$1,300,000** was recommended to be held in escrow
per Exhibit C - Earnout Terms. …

The **Closing Checklist** indicates a "Total at Closing" of **$39,630,000**, which corresponds to the
sum of Cash ($28,330,000) + Escrow deposit ($1,300,000) + Stock issuance ($10,000,000). The $5,000,000
earnout payments are treated as future contingent consideration rather than immediate cash required at
closing.
```

Citations and teardown:

```
📎 Sources (452.7s):
  [1] 05_financial_adjustments.pdf#0 (score 0.5329)
      FINANCIAL ADJUSTMENTS MEMO FINANCIAL ADJUSTMENTS MEMORANDUM To: Deal Team From: Finance Department Date: December 23, 2024 Re: Purchase Price Adjustments - StartupXYZ Acquisition Following our review in connection with t…
  [2] 01_acquisition_agreement.pdf#0 (score 0.5259)
      ACQUISITION AGREEMENT ACQUISITION AGREEMENT This Acquisition Agreement ("Agreement") is entered into as of January 15, 2025, by and between TechCorp Industries, Inc. ("Buyer") and StartupXYZ LLC ("Seller"). ARTICLE I - D…
  [3] 10_closing_checklist.pdf#0 (score 0.5086)
      CLOSING CHECKLIST Acquisition of StartupXYZ LLC by TechCorp Industries, Inc. Closing Date: March 1, 2025 Closing Location: Wilson & Partners LLP, San Francisco I. PRE-CLOSING CONDITIONS A. Regulatory [X] HSR Filing submi…
  [4] 02_due_diligence_report.pdf#0 (score 0.4959)
      DUE DILIGENCE REPORT CONFIDENTIAL DUE DILIGENCE REPORT Prepared for: TechCorp Industries, Inc. Subject: StartupXYZ LLC Date: December 20, 2024 Prepared by: Morrison & Associates, LLP EXECUTIVE SUMMARY This report summari…
  [5] 04_risk_assessment.pdf#0 (score 0.4910)
      RISK ASSESSMENT MEMO CONFIDENTIAL RISK ASSESSMENT MEMORANDUM To: TechCorp Board of Directors From: Corporate Development Team Date: December 22, 2024 Re: Risk Assessment - StartupXYZ Acquisition This memo summarizes key …
  … and 5 more source chunk(s)

🧹 Removed the ephemeral index (/var/folders/…/T/localgpt-ask-jf_scljx).
⏱️  total 462.1s
```

**Cleanup, before and after.** Both listings were taken with the same command;
`lancedb/` and `index_store/` are identical:

```
before                                          after
--- lancedb/ ---                                --- lancedb/ ---
text_pages_82b2f5a9-….lance                     text_pages_82b2f5a9-….lance
--- index_store/ ---                            --- index_store/ ---
overviews                                       overviews
index_store/overviews:                          index_store/overviews:
2fb7a91a-….jsonl                                2fb7a91a-….jsonl
66ac9551-….jsonl                                66ac9551-….jsonl
82b2f5a9-….jsonl                                82b2f5a9-….jsonl
```

`diff` reports **no difference** in those two sections. No `ask_*` table, no new
overview JSONL, no `.vectors.npz`.

**The one leak, reported because it happened.** An *earlier* attempt at this same
run was killed by a 2-minute harness timeout and left
`$TMPDIR/localgpt-ask-syr5n_ob` behind (156K, containing only an empty `lancedb`
directory). That is what prompted the SIGTERM handler described in §1. It was
then verified:

```
$ .venv/bin/python -m rag_system.main ask eval/corpora/acquisition "test" &  # then SIGTERM mid-index
temp dir: /var/folders/…/T/localgpt-ask-_smy40vj
exists before SIGTERM: yes
exit=143
removed after SIGTERM
🧹 Removed the ephemeral index (/var/folders/…/T/localgpt-ask-_smy40vj).
⏱️  total 13.3s
```

The stale `localgpt-ask-syr5n_ob` from the pre-fix run was deleted by hand.
`ls -d $TMPDIR/localgpt-*` now lists nothing but the running smoke test's own
directory.

### 2.7 The other `ask` branches, on a two-file scratch folder

The run above only exercises the default path, so the three remaining branches
were run on a two-document scratch folder (`widget_spec.md`: torque 47 Nm,
service interval 900 hours, bearing BR-2210; `pricing.md`: $3,150/unit, 12%
above 50 units). All four exited as intended and all three indexing runs cleaned
up. Output below is filtered to the mode's own lines (the full multi-line answers
are longer than what the filter kept).

```
########## A: --agent ##########
⏱️  indexed in 3.5s
❓ What is the service interval and the list price of the Widget X-9?
💬 Answer: … The list price of the Widget X-9 is 3,150 dollars per unit, and volume orders
   above 50 units receive a 12 percent discount. Regarding maintenance requirements, its
   service interval is specified as 900 operating hours.
📎 Sources (69.4s):  [1] pricing.md#0 (score 0.6490)   [2] widget_spec.md#0 (score 0.6070)
🧹 Removed the ephemeral index (…/localgpt-ask-2v0ldoxa).
⏱️  total 72.9s
exitA=0

########## B: --filters '{"document_id": "pricing.md"}' ##########
⏱️  indexed in 3.1s
🔎 Metadata filter (prefilter, both legs): document_id = 'pricing.md'
🔎 Metadata filter applied: document_id = 'pricing.md' → 1 candidate(s).
📎 Sources (67.7s):  [1] pricing.md#0 (score 0.5035)          ← widget_spec.md excluded
🧹 Removed the ephemeral index (…/localgpt-ask-i9ewqvun).
exitB=0

########## C: --interactive (two questions piped on stdin, then a blank line) ##########
❓ What is the replacement bearing part number?
💬 … the replacement bearing part number for the Widget X-9 is BR-2210.
❓ What discount applies above 50 units?
💬 … volume orders for the Widget X-9 that exceed **50 units** receive a **12 percent discount**.
❓ Ask a follow-up (blank line to finish):
🧹 Removed the ephemeral index (…/localgpt-ask-4jzkp3tg).
⏱️  total 141.4s
exitC=0

########## D: bad filter ##########
❌ Invalid --filters: Unsupported filter field(s): page. Supported: document_id (eq, in,
   contains); document_name (contains); chunk_id (eq, in); chunk_index (eq, in, gt, gte, lt, lte).
exitD=2
```

B is the interesting one: the filter is honoured on an index that was created
seconds earlier, and `widget_spec.md` — which contains the torque and bearing
facts and would otherwise be candidate 2 — never enters the context.

Both filters (4.4) and ask (4.6) therefore compose, and 4.4 works against an
ephemeral table as well as a persistent one.

**After all of the above**: `ls -d $TMPDIR/localgpt-*` lists nothing, `lancedb/`
holds only the single pre-existing `text_pages_82b2f5a9-….lance`, and
`index_store/overviews/` holds only the same three pre-existing `.jsonl` files.

### 2.8 Smoke test

```
$ .venv/bin/python eval/smoke_e2e.py
  [PASS] both services became healthy
  [PASS] upload accepted the PDF
  [PASS] index build returned 200 with no error
  [PASS] index linked to session
  [PASS] q1: planted fact '9.2' in answer        [PASS] q1: source_documents non-empty
  [PASS] q1: [Confidence: N%] tag present — tag=[Confidence: 100%]
  [PASS] q1: message_count == 2  — got 2
  [PASS] q2: planted fact 'TS-71' in answer      [PASS] q2: source_documents non-empty
  [PASS] q2: [Confidence: N%] tag present — tag=[Confidence: 100%]
  [PASS] q2: message_count == 4  — got 4
  [PASS] q3: planted fact '36' in answer         [PASS] q3: source_documents non-empty
  [PASS] q3: [Confidence: N%] tag present — tag=[Confidence: 100%]
  [PASS] q3: message_count == 6  — got 6
  [PASS] q4: planted fact 'drip tray' in answer  [PASS] q4: source_documents non-empty
  [PASS] q4: [Confidence: N%] tag present — tag=[Confidence: 100%]
  [PASS] q4: message_count == 8  — got 8
  [PASS] messages/save returned 200
  [PASS] saved assistant message round-trips out of SQLite  — 10 messages in session
  [PASS] saved source_documents round-trip in metadata
  [PASS] saved steps round-trip in metadata
  [PASS] final message_count == 10  — got 10
  wall clock 280.6s
25/25 assertions passed
exit=0
```

(The four `[PASS] qN: …` lines are reflowed two-per-line here purely to fit;
the wording is verbatim.)

---

## 3. Limitations — read before enabling anything

* **Only real columns are filterable.** The LanceDB text table has six columns
  (`vector`, `text`, `chunk_id`, `document_id`, `chunk_index`, `metadata`) and
  `metadata` is a **JSON string**. The roadmap names "document name, page,
  date"; page and date live inside that JSON string and are therefore **not
  filterable**. They were not faked with a `metadata LIKE '%…%'` substring
  match, which would look like it worked and quietly be wrong (`"page": 3`
  matches `"page": 31`, and any user text containing the phrase matches too).
  Fixing this needs page/date promoted to real columns — a schema change and a
  re-index. See the backlog.
* **`document_name` is a substring of `document_id`.** For UI-uploaded
  documents the id is `<uuid>_<name>`, so `contains: "report"` also matches a
  document whose *uuid* happens to contain "report" (unlikely) and matches every
  document whose name contains it (intended). There is no exact name match.
* **No OR / NOT / nesting.** `{"document_id": {"in": [...]}}` covers the common
  disjunction. Anything else is a new feature, not a config.
* **No gold-set number exists for 4.4 and one cannot be produced today**: every
  gold row is scored against an unfiltered corpus, so a filter can only ever
  make recall worse there. Measuring filters needs gold rows that *carry* a
  filter and an expected in-scope answer. Backlog.
* **4.6 is slow on document-sized chunks.** The one real run below spent
  **452.7s of its 462.1s in synthesis** — the `fast` profile's `retrieval_k=10`
  against ten single-chunk documents hands the generation model ten whole
  documents. Indexing was 9.4s. This is not a regression (it is the shipped
  pipeline behaving normally at this chunk size) but "fast profile" reads like a
  promise this mode does not keep.
* **4.6 has no eval coverage at all.** It is a CLI wrapper; the harness does not
  drive CLI subcommands.
* **`SIGKILL` still leaks the temp directory.** Nothing can be done about that
  from inside the process.
* **`ask --agent` barely differs from the default on the `fast` profile.** `fast`
  has `query_decomposition` and `verification` off, so on that profile the flag
  buys only the agent's history/caching bookkeeping — run A in §2.7 produces no
  `[Confidence: N%]` tag for exactly that reason. `--agent --mode default` is
  where the flag means something, and that combination has **not** been run.
* **`--interactive` was verified with piped stdin, not a TTY.** The prompt string
  and the pipeline's log lines interleave on one line in that mode, which is
  cosmetic but ugly; on a real terminal it reads normally.

---

## 4. Proposed config keys (for `rag_system/main.py`, at the adoption gate)

**None are required.** 4.4 has no flag by design: no `filters` argument means
byte-identical behaviour, so there is nothing to switch off, and a flag would
only create a state where a caller's explicit filter is silently ignored — the
exact failure this feature must never have. 4.6 is a CLI subcommand, opt-in by
existing.

Two keys are *available* if the gate wants them discoverable rather than
hard-coded in `rag_system/ask_folder.py`:

```python
    # Optional. "ask" (roadmap 4.6) currently hard-codes these on top of the
    # `fast` profile; a profile block would let a user change them.
    "ask": {
        "profile": "fast",       # profile the ephemeral index is built with
        "enrichment": False,     # LLM call per chunk
        "overviews": False       # LLM call per document; nothing here reads them
    }
```

If the gate would rather cap filter surface centrally, the limits currently
living as module constants in `rag_system/retrieval/filters.py`
(`_MAX_STRING_LENGTH = 256`, `_MAX_IN_ITEMS = 256`) are the candidates. They are
guard rails, not tuning knobs, which is why they are constants today.

## 5. Proposed gateway diff (`backend/server.py`, **not applied** — not my file)

The RAG API accepts `filters` today; the Node/Python gateway at
`backend/server.py` drops unknown keys, so a browser client cannot yet reach it.
Two changes:

```python
# CHAT_OPTIONS, next to "retrieval_mode":
    "filters": (dict, ()),
```

`normalize_options` calls `caster(data[key])`; `dict({...})` copies a dict and
raises `TypeError`/`ValueError` on anything else, which the existing
`except (TypeError, ValueError)` branch already passes through unchanged — so a
non-object `filters` still reaches the RAG API and is rejected there with the
400 that carries the real message. No validation logic is duplicated in the
gateway, deliberately: one validator, one error text.

```python
# should_use_rag(...), alongside the force_rag short-circuit:
    if options.get("filters"):
        return True   # an explicit filter is a statement that this is a document question
```

(the second is the gateway-level twin of the agent-side rule in §1).

## 6. Proposed documentation diffs (none applied this wave)

* **`Documentation/retrieval_pipeline.md`**
  * New section *Metadata filters*: the JSON DSL table (field → column →
    operators), that both legs are prefiltered, that internal
    document-scoped searches and context expansion are narrowed too, the
    `filters_applied` SSE event and the `result["filters"]` shape, and the
    explicit statement that an absent `filters` key changes nothing.
  * A line under the retry: the filter is re-applied to the reformulated query
    because it is read from the scope, not passed per call.
* **`Documentation/api.md` / whichever file documents `/chat`** — the optional
  `filters` body field, the 400 contract, and the fact that a filter implies
  `force_rag`.
* **`Documentation/design_rationale.md`**
  * §4 or a new short section: why the filter language is JSON rather than the
    source project's filter *string* (a parser is the attack surface), and why
    values are refused rather than escaped (`document_fetch.py`'s precedent).
  * Why `document_name` is `contains`-only.
  * Why the filter travels as a thread-local scope rather than a parameter (the
    4.1 subclass's positional `super()` call) — this is a real coupling and the
    next person to touch `retrieve_candidates` needs to know it exists.
* **`Documentation/research_roadmap.md`** — Phase 4 rows 4.4 and 4.6: mark
  implemented, pointing here; note that 4.4 covers `document_id` /
  `document_name` / `chunk_index` / `chunk_id` but **not** page or date, which
  the row currently promises.
* **`README.md` / CLI docs** — the `ask` subcommand and `chat --filters`.

## 7. Couplings the next person needs to know

1. **`retrieve_candidates`'s positional signature is load-bearing.**
   `EscalatingRetrievalPipeline` (roadmap 4.1) overrides it and calls `super()`
   with four positional arguments. That is why `filters` is keyword-only and why
   `run()` uses a thread-local scope instead of passing it down. Anyone adding
   another parameter must do the same, or update
   `rag_system/agent/escalation.py` in the same change.
2. **`retrieve_candidates`'s body moved into `_retrieve_candidates_filtered`,
   and `run()`'s second half into `_run_after_candidates`.** Both are pure
   extractions with the filter scope wrapped around them; no logic moved between
   them. A merge with another change to either method will conflict textually
   and should be resolved by keeping the split.
3. **`_post_candidates` is still the single tail hook** for every
   `retrieve_candidates` exit path (wave 1's contract). Nothing here changed it.
4. **The semantic cache entry gained a `"filters"` key.** Entries written by an
   older build have no such key, so `dict.get` returns `None`, which matches an
   unfiltered request — the intended behaviour, and the reason the comparison is
   `.get(...) != signature` rather than a lookup that would `KeyError`.
5. **`rag_system/api_server.py`'s `__main__` now parses `--port`.** Anything that
   started it with `--port` and relied on getting 8001 anyway will now get the
   port it asked for. Nothing in the repo does; `eval/smoke_e2e.py` starts it
   without `--port`.

## 8. Backlog

1. **Gold rows that carry filters.** ~6 rows of the form
   *(query, filters, expected)* where the expected answer is inside the filtered
   scope and a plausible distractor is outside it. Without these, 4.4 has no
   number and cannot get one. The harness would need `retrieve_candidates(...,
   filters=…)`, which is already the public signature.
2. **Promote `page` (and any date) to real LanceDB columns** so the roadmap's
   full 4.4 surface is reachable. Schema change + re-index; the DSL grows two
   rows in its field table and nothing else.
3. **A harm check for filters + the 4.2 hop.** The hop is currently narrowed by
   the filter, which is right, but it means a filtered query can silently lose
   the hop's benefit. Worth a line in the docs once 4.2 is benchmarked at all.
4. **LLM filter extraction** (the roadmap's "later"): natural language →
   this DSL, on the enrichment model, with the compiled where-clause shown to
   the user before it runs. The deterministic layer it needs now exists; the
   measured ~0.999 F1 claim in the roadmap is for easy/medium translations and
   should be re-measured on this DSL before anything ships.
5. **`ask --agent --mode default`** has never been run; it is the only untested
   combination of the new CLI flags. Cheap to cover once someone has 10 minutes
   of generation budget.
6. **Reconsider `ask`'s default `retrieval_k`.** Ten whole documents into one
   synthesis prompt is what made the run below take 7.7 minutes; a lower `k`
   (or chunking that produces more, smaller chunks) is the fix, but it is a
   quality/latency trade and should be measured, not guessed.

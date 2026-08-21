# localGPT shakedown on unseen real documents — 23 IETF RFCs (QUIC / HTTP-3)

Run 2026-08-12 / 2026-08-13 on this Mac. Repo `/Users/prompt/videos/localgpt_08082026/localGPT`,
branch `rearchitect/evidence-gated-aug-2026`, interpreter `.venv/bin/python`,
Ollama at `localhost:11434` (generation `qwen3.5:9b`, enrichment + judge
`qwen3.5:4b`, embedder `microsoft/harrier-oss-v1-0.6b`). Everything local.

Every number below came from a command executed in this session. Ollama is a
shared local service; the only wall-clock figure offered as evidence is the
indexing time, and even that is "on a machine also running the eval harness".

> **Read this first.** §1-§9 are the original run, kept verbatim as the record.
> Finding (1) below was validated and **fixed by the gate on 2026-08-13**, and
> everything downstream of it was re-measured from scratch. The current numbers
> and the current verdict are in **[Post-fix re-run](#post-fix-re-run-2026-08-13)**,
> after §8. Finding (2) is unchanged by the fix.

**Short answer to "does the setup work on unseen real documents?" — no, not yet.
Two things break before retrieval quality is even the question:**

1. **The chunker silently discards ~48% of the corpus.** Any document over
   ~10,000 markdown tokens loses roughly half its text before it is ever
   embedded. 10 of 24 mechanically verified gold rows have their answer in *no*
   indexed chunk as a result. This is a bug in shipped code, it affects
   `Documentation/design_rationale.md` in the existing `docs` corpus too, and it
   is the single largest finding here.
2. **The cross-reference extractor resolves 0 of 731 references** on real
   documents. Not "few" — zero. The extraction half of roadmap 4.2 is inert on
   any corpus whose filenames are not literal substrings of its own prose.

Retrieval, measured only over the rows whose content actually reached the index,
is respectable (recall@10 0.929, nDCG@10 0.601). End-to-end answer quality is
poor (4/24 judged pass), and the dominant failure mode is the system saying
"there is no mention of that in the provided text" — which is what you would
expect from a half-indexed corpus, and is at least an honest failure rather than
a hallucinated one.

---

## 1. Corpus

`eval/corpora/rfc/` — **23 documents, 1,511,267 bytes (1.44 MiB)**, downloaded
verbatim from `https://www.rfc-editor.org/rfc/rfcNNNN.txt`. Reproducible with
`.venv/bin/python eval/corpora/rfc/download.py`; `--check` re-verifies sizes and
the link graph. Full per-file table, rationale and exclusions:
`/Users/prompt/videos/localgpt_08082026/localGPT/eval/corpora/rfc/MANIFEST.md`.

The cluster is the QUIC / HTTP-3 family plus what it is defined against:

* QUIC core — 8999, 9000, 9001, 9002, 9221, 9369, 9308, 9312
* HTTP over QUIC — 9114, 9204, 9218, 9220, 9297, 9298, 9412, and the HTTP/2
  counterparts 8336, 8441 that those documents defer their semantics to
* shared normative dependencies — 2119, 8174, 8126, 6066, 7301

Selection rule, enforced mechanically by `download.py --check`: **every document
references or is referenced by at least two others in the set.** Measured:
**110 directed intra-corpus `RFC NNNN` references**; the lowest-degree document
(RFC 6066) touches 4 others; RFC 9000 is cited by 14 of the other 22.

RFC 5234 (ABNF) was in the first draft and was removed when `--check` reported
it at **degree 0** — the family's ABNF references all route through RFC 9110 /
9112, which are excluded for budget. RFC 8126 replaced it.

Excluded for the ~1.5 MB budget, and this is a real limitation rather than a
neutral trim: RFC 9110 (502,941 B — the most-cited document in the HTTP/3
sub-cluster), RFC 8446 (337,736 B — RFC 9001's principal TLS dependency),
9113, 6455, 7541, 9111, 9112. The 9001↔8446 cross-reference pair the task
suggested is therefore *not* in the gold set; the TLS-side crossref rows anchor
on 7301 and 6066 instead.

---

## 2. Gold set

`eval/goldset/rfc.jsonl` — **24 rows, hand-authored**, same schema as
`eval/goldset/acquisition.jsonl` plus the same `expected_sources` /
`anchor_doc` / `multi_document` / `requires_crossref` fields. Answer-bearing
anchors are catalogued in `eval/corpora/rfc/rfc.facts.json` (26 facts).

Gate: `eval/verify_rfc_goldset.py`, modelled on `eval/verify_crossref_goldset.py`.
Output, verbatim:

```
/Users/prompt/videos/localgpt_08082026/localGPT/eval/goldset/rfc.jsonl: 24 rows, 26 expected strings, 23 source documents
  expected_in_source         26/26
  no_verbatim_leak           26/26
  unique_to_source           26/26
  fact_id_resolves           26/26
  crossref_flag_consistent   24/24
  multi_document_consistent  24/24
  requires_crossref=true     10
  multi_document=true        2
  question_type              {'factoid': 16, 'negative': 3, 'procedural': 3, 'comparative': 2}
  difficulty                 {'easy': 9, 'hard': 15}
  documents referenced       16/23

all row-level checks passed.
```

Notes on the gates:

* Comparison is whitespace-normalised and case-insensitive, because RFCs are
  hard-wrapped at 72 columns and a sentence-length anchor necessarily spans a
  line break. `run_eval.py` scores chunk relevance with exactly the same
  normalisation (`run_eval.norm`), so a string that passes here is a string the
  metric can match.
* **`unique_to_source` passed 26/26, so nothing had to be exempted.** The
  mechanism for RFC boilerplate exists in the script (`UNIQUENESS_EXEMPT`, empty
  and documented as such) because the obvious anchors — the BCP 14 sentence, the
  bare phrase "MUST NOT" — occur in all 23 documents and were deliberately
  avoided during authoring. Every anchor that shipped carries a hex constant, a
  codepoint, a named parameter or a distinctively worded restriction.
* The 10 `requires_crossref=true` rows each state document A's premise and ask
  for a fact whose text exists only in document B — e.g. RFC 9002's deferral of
  the anti-amplification limit to RFC 9000 §8.1 (`rfc_q15`), RFC 9412's deferral
  of ORIGIN payload semantics to RFC 8336 (`rfc_q22`), RFC 9220's reuse of RFC
  8441's `:protocol` pseudo-header (`rfc_q21`).
* 2 rows are `multi_document` with `match: "all"` (`rfc_q18` needs both QUIC v1
  and v2 initial salts; `rfc_q24` needs both HTTP/3's requirement and RFC 9000's
  parameter definition).
* 16 of the 23 documents carry at least one anchor. The other 7 (2119, 8174,
  9308, 9204's siblings etc.) are present as distractors and reference targets.

**Gate 2 (reachability after chunking) FAILS for 10 of 24 rows.** That is not a
gold-set defect — see §4 — and the rows are deliberately left in place so the
gate keeps reporting the loss.

---

## 3. Indexing with product defaults

`build_product_index.py` — `PIPELINE_CONFIGS["default"]` unchanged except for
storage location (scratch; `eval/.eval_indexes/` and `index_store/` untouched)
and `chunk_size = 512`. Confirmed active at run time: contextual enrichment ON
(window 1, `qwen3.5:4b`), document overviews ON + embedded sidecar, late
chunking ON, `extract_crossrefs` ON, docling chunker.

| Stage | Wall clock |
|---|---:|
| Document processing + chunking + 23 document overviews | 201.8 s |
| Cross-reference extraction | 0.26 s |
| Contextual enrichment, 387 chunks | 1763.8 s |
| Embedding generation | 52.4 s |
| Late-chunk embedding & indexing | 65.6 s |
| Overview embedding (23 docs → `.vectors.npz`) | 1.5 s |
| **Total** | **2089.2 s (34.8 min)** |

**387 chunks** from 23 files. Enrichment is 84% of the build and runs at 0.22
chunks/s on the shared Ollama.

**No `Ollama likely FRONT-TRUNCATED` warning occurred** — checked in the index
build log, the E2E run log and the judge log. Count is 0 in all three.

### Cross-reference extraction on real documents — the headline

Printed by the pipeline itself, identical in the harness-config build and the
product-defaults build:

```
🔗 Cross-references: 731 reference(s) in 260 chunk(s); 0 resolved to 0 document(s).
```

**0 of 731 resolved.** `crossref_diagnostic.py` re-runs `annotate_chunks`
unmodified over the same 387 chunk texts under three document-naming schemes
(the schemes are passed as different `known_documents` id sets — a public
argument; nothing in `rag_system/` was patched):

| Naming scheme | refs | of which `document` mentions | resolved | documents linked |
|---|---:|---:|---:|---:|
| shipped `RFC 9000 - QUIC A UDP-Based ....txt` | 731 | 0 | **0** | 0 |
| `RFC 9000.txt` | 822 | 91 | **91** | 18 |
| `QUIC A UDP-Based ... - RFC 9000.txt` (citation word order) | 766 | 35 | **35** | 11 |

Three distinct reasons, all measured:

1. **The document-mention pass requires the whole normalised filename to appear
   in the text.** Under the shipped naming, **0 of 23** document names occur
   anywhere in the corpus (0 occurrences). Under `RFC 9000.txt`, 20 of 23 names
   occur, 178 times. RFCs cite each other as `RFC 9000` or `[RFC9000]`, never as
   `RFC 9000 - QUIC A UDP-Based Multiplexed and Secure Transport`. The
   numeric-prefix-stripping alias in `CrossRefExtractor.__init__` does not help,
   because these filenames start with `RFC`, not with digits.
2. **96% of what the extractor does find is unusable.** The 731 references break
   down as **703 `section` + 28 `exhibit` + 0 `document`**. A `section` ref is
   recorded as a bare `"section 8.1"` with `target_doc: None`, because no
   document is named `section 8 1`. In an RFC corpus that is the wrong shape
   entirely: of **1,713** `Section N` occurrences, **355 are explicitly
   qualified** — `Section 8.1 of [QUIC-TRANSPORT]` — and the extractor's
   `_SECTION_RE` captures the section number and **throws away the `[…]`
   qualifier, which is precisely the hop target.** Those 355 are the real
   cross-reference graph of this corpus and the extractor cannot see any of it.
3. **The qualifiers are symbolic, not numeric.** 52 distinct tags appear in
   those qualified references; the most common are `[QUIC-TRANSPORT]` (74),
   `[HTTP]` (44), `[QUIC-TLS]` (39), `[QUIC]` (20), `[RFC9000]` (17). Resolving
   them means reading each document's own References section to map
   `[QUIC-TRANSPORT] → RFC 9000`. Filename matching cannot do it in principle,
   only the 17 literal `[RFC9000]`-style tags are reachable that way.

The `MAX_CROSSREFS_PER_CHUNK = 8` cap is a minor contributor, not the cause: 8
chunks hit the cap, and 745 label+section references were found before it
against 731 kept. It would matter more under a naming scheme that produced
document mentions, since the document pass runs last and gets the leftovers.

**Reading:** this is not "the extractor scored low", it is "the extractor is
inert on any corpus we did not name for it". The index-time half of roadmap 4.2
has, until now, only ever been run on `eval/corpora/acquisition/`, whose
filenames (`05_escrow_agreement.pdf`) were written to match the prose ("the
Escrow Agreement"). The corpus and the extractor were co-designed. On documents
that were not, it produces nothing to hop on.

---

## 4. The chunker discards half the corpus

*(Fixed by the gate on 2026-08-13 — see [P1](#p1-the-chunker-fix-verified-independently).
This section records the pre-fix state and the diagnosis.)*

This was found by gate 2 and is the reason most other numbers on this page are
depressed.

`run_eval.py --coverage-only` on the 387-chunk index:

```
⚠️  10 gold row(s) whose expected text is in NO indexed chunk:
     rfc_q02, rfc_q09, rfc_q10, rfc_q11, rfc_q13,
     rfc_q14, rfc_q15, rfc_q17, rfc_q18, rfc_q24
```

Cause, isolated in `chunker_loss_repro.py`, measured three ways:

* **The LanceDB table holds 52% of the corpus's whitespace-normalised
  characters** (790,786 of 1,511,267 raw bytes' worth). Per document the
  retention splits cleanly in two: documents under ~10,000 markdown tokens
  retain ~100%, documents over it retain 45-57%. RFC 9114 retains 0.45, RFC 9000
  0.50, RFC 9297 1.02 (the >1 is the one-sentence overlap duplicating text).
* **The conversion step is innocent.** `DocumentConverter.convert_to_markdown`
  on RFC 9114 returns 139,294 normalised characters against 139,286 in the
  source — ratio 1.00, and the missing gold string is present in its output.
* **The loss is in `MarkdownRecursiveChunker._split_text`**
  (`rag_system/ingestion/chunking.py`). `re.split(f'({sep})', chunk)` yields
  `[t0, sep, t1, sep, t2, sep, t3]`; the re-combining loop advances `i += 3`
  after a match, which lands on a separator rather than the next body segment.
  It emits `[sep+t1, sep, sep+t3]`: `t0` is never emitted and `t2` is replaced
  by a bare separator. Synthetic repro — 8 paragraphs in, `max_chunk_size=200`
  tokens: **segments 1,3,5,7 kept; segments 0,2,4,6 lost; 0.50 of characters
  survive.**

It only fires when a chunk exceeds `max_chunk_size`, and `DoclingChunker`
constructs its internal `MarkdownRecursiveChunker` with `max_chunk_size=10_000`.
That is why no previous corpus exposed it: **12 of 23 RFCs cross the threshold,
against 1 of 13 files in `Documentation/`.**

**The pre-existing `docs` corpus is affected**, in exactly one file:
`Documentation/design_rationale.md`, 13,285 tokens, **0.49 retained**. Every
other Documentation file is under the threshold at ~1.00. So the `docs` and
`mixed` numbers in `eval/BASELINE.md` were measured against a corpus that is
missing half of one of its thirteen files. Small, but it means the baseline will
move when this is fixed, and it should not be treated as a regression.

Nothing was edited to establish any of this — the repro imports the shipped
classes and calls them.

---

## 5. Retrieval

Harness configuration (`run_eval.py` machinery, driven from
`run_rfc_eval.py`, which injects the `rfc` corpus and redirects the index root
into scratch without editing anything under `eval/`): enrichment / overviews /
late chunking / context expansion OFF, reranker OFF, `k = 20`,
`chunk_size = 512`, embedder `microsoft/harrier-oss-v1-0.6b`, hybrid search.
`--retry off` for the comparison cells, per the determinism protocol.

The `final == first_stage` invariant held on all 24 queries (rerank off, hop
off), so the first-stage numbers are the whole story.

| Slice | n | recall@5 | recall@10 | recall@20 | nDCG@10 | 1st ms mean |
|---|---:|---:|---:|---:|---:|---:|
| **all rows** | 24 | 0.417 | 0.542 | 0.583 | **0.392** | 282 |
| `requires_crossref=true` | 10 | 0.500 | 0.600 | 0.600 | 0.518 | 341 |
| control (`=false`) | 14 | 0.357 | 0.500 | 0.571 | 0.303 | 239 |
| **reachable rows only** (gate 2 pass) | 14 | 0.714 | **0.929** | **1.000** | **0.601** | 276 |
| ├ `requires_crossref=true` | 6 | 0.833 | 1.000 | 1.000 | 0.696 | 325 |
| └ control | 8 | 0.625 | 0.875 | 1.000 | 0.529 | 240 |
| unreachable rows (chunker loss) | 10 | 0.000 | 0.000 | 0.000 | 0.100 | 289 |

The 0.100 nDCG on the unreachable slice is entirely `rfc_q18`, a `match: "all"`
row that scores nDCG 1.000 on the one of its two anchors that survived chunking
while scoring recall 0 — the coverage-vs-ranking split `eval/README.md` warns
about, showing up for real.

By dimension (all 24 rows, pooled): easy `recall@10` 0.556 / nDCG 0.380, hard
0.533 / 0.400; factoid 0.625 / 0.361, procedural 0.667 / 0.667, negative 0.333 /
0.210, comparative 0.000 / 0.500.

With `--retry on` (the shipped profile) the numbers are slightly *worse* —
all-rows recall@10 0.500 vs 0.542, nDCG@10 0.355 vs 0.392 — and the retry fires
on 6/24 queries (`q01, q03, q04, q05, q17, q19`) with 5 rewrites kept. Per-query
latency rises from 282 ms to 1480 ms. On 24 rows this is a diagnostic, not a
verdict on the retry; but it is the opposite sign to what the retry is for, and
the rewrites are nondeterministic, which is why the comparison cells above use
`--retry off`.

### Unseen-real vs authored-synthetic

Against `eval/BASELINE.md` § *Phase 4 baseline* (same embedder, same `k`, same
chunk size, reranker off):

| Corpus | n | chunks | recall@5 | recall@10 | recall@20 | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| `acq` (10 synthetic M&A PDFs) | 24 | 13 | 0.958 | 1.000 | 1.000 | 0.810 |
| `acq+docs` | 48 | 373 | 0.917 | 0.958 | 1.000 | 0.738 |
| `acq+docs`, crossref slice | 11 | 373 | 1.000 | 1.000 | 1.000 | 0.748 |
| **`rfc`, all rows** | 24 | 387 | 0.417 | 0.542 | 0.583 | 0.392 |
| **`rfc`, reachable rows** | 14 | 387 | 0.714 | 0.929 | 1.000 | 0.601 |
| **`rfc`, reachable crossref** | 6 | 387 | 0.833 | 1.000 | 1.000 | 0.696 |

The honest comparison is the reachable-rows line, since the all-rows line is
mostly measuring the chunker bug. On that line:

* **Recall is comparable.** 0.929 @10 and 1.000 @20 against `acq+docs`'s 0.958
  and 1.000, on a corpus with a similar chunk count (387 vs 373) but 5.7x the
  text and genuine same-domain distractors rather than topically disjoint ones.
* **Ranking is meaningfully worse.** nDCG@10 0.601 vs 0.738. The answer-bearing
  chunk is usually retrieved but sits at rank 3-5 instead of 1-2. That is the
  expected effect of same-domain distractors: 23 documents that all discuss
  connection IDs, frame types and settings compete far harder than localGPT
  documentation competes with an M&A deal room.
* **The crossref slice is again not the weak one** — 1.000 recall@10 and 0.696
  nDCG against the control's 0.875 / 0.529. This reproduces the `acq` finding
  and for the same reason: a crossref query that names document A *and* asks
  about B's subject matter hands the hybrid FTS leg lexical signal from both
  ends. It is a property of how honest crossref questions are worded, not
  evidence that pointer-following is solved. Note the control slice here is
  genuinely harder than on `acq` (0.529 vs 0.796), which is the same-domain
  distractor effect.

---

## 6. End-to-end judged answers

All 24 gold questions through the in-process `Agent` on the product-defaults
index, `force_rag=True`, `agent._query_cache.clear()` before every query.
Answers recorded verbatim in `results/rfc_e2e_answers.jsonl`.

Per-query wall clock: mean 88.7 s, median 59.4 s, min 41.0 s, max 465.0 s
(`rfc_q21`), total 2130 s. Zero exceptions; zero truncation warnings.

Judged with `eval/judge.py`'s `GroundednessJudge` on `qwen3.5:4b`, prompt v1,
**k = 5 votes per row**, with the mapping the task specifies:
`EVIDENCE = the system answer`, `ANSWER = the gold answer`,
`QUESTION = the gold query`.

| Slice | n | majority pass (≥3/5) | unanimous pass | unanimous fail | mean k/5 | cited an expected source |
|---|---:|---:|---:|---:|---:|---:|
| all | 24 | **4** | 2 | 17 | 0.92 | 24 / 24 |
| single-doc | 14 | **3** | 1 | 11 | 0.86 | 14 / 14 |
| `requires_crossref` | 10 | **1** | 1 | 6 | 1.00 | 10 / 10 |
| — reachable rows | 14 | 3 | — | — | 1.21 | 14 / 14 |
| — unreachable rows | 10 | 1 | — | — | 0.50 | 10 / 10 |

`k_true` histogram: `{0: 17, 1: 1, 2: 2, 3: 1, 4: 1, 5: 2}` — the judge is
mostly unanimous, in the negative.

Per row:

| id | k/5 | reachable | crossref | first 100 chars of the system answer |
|---|---:|---|---|---|
| rfc_q01 | 0/5 | yes | | "there is **no mention of an `ack_delay_exponent` parameter**…" |
| rfc_q02 | 0/5 | no | | "a QUIC endpoint is generally free to advertise any value…there is **no expli…**" |
| rfc_q03 | 0/5 | yes | | "there is **no single specific error code** defined solely for…" |
| rfc_q04 | **3/5** | yes | | "if a QUIC sender has not yet measured any RTT samples…" |
| rfc_q05 | **4/5** | yes | | "The sender is permitted to send up to two probe datagrams…" |
| rfc_q06 | 0/5 | yes | | "Based on RFC 9297 … and RFC 9003 (the specification for…" |
| rfc_q07 | **5/5** | yes | | "the version field value for long headers is…" |
| rfc_q08 | 0/5 | yes | | "there is **no specific numerical range**…" |
| rfc_q09 | 0/5 | no | | "**no**, a … DoQ client should not open its initial connection using UDP port 53…" |
| rfc_q10 | 0/5 | no | | "there is **no specific numeric value or fixed amount** of flow-control credit…" |
| rfc_q11 | 0/5 | no | | "the receiving endpoint treats the missing parameter as having a value of **0**" |
| rfc_q12 | 0/5 | yes | | "there are **no specific header field names**…" |
| rfc_q13 | 0/5 | no | | "the default applied is considered low … between **0 and 6**" |
| rfc_q14 | 0/5 | no | | "there is **no mention** of a \"latency spin bit\"…" |
| rfc_q15 | **5/5** | no | Y | "if no additional data … (three times) …" |
| rfc_q16 | 0/5 | yes | Y | "Based on **RFC 9000 (QUIC)** and its associated transport parameters…" |
| rfc_q17 | 0/5 | no | Y | "there is no specific mention of the **width** (in bits)…" |
| rfc_q18 | 0/5 | no | Y | "The provided context does not specify two distinct salts…" |
| rfc_q19 | 0/5 | yes | Y | "there is **no mention** of a specific error code or numeric value…" |
| rfc_q20 | 2/5 | yes | Y | "mandates two additional demands beyond expert review…" |
| rfc_q21 | 2/5 | yes | Y | "first defined in **RFC 8441**, published in September 2018…" |
| rfc_q22 | 0/5 | yes | Y | "there is no \"HTTP/3 ORIGIN extension\" that utilizes a corresponding HTTP/2 frame…" |
| rfc_q23 | 1/5 | yes | Y | "there are no direct facts linking Connect-UDP tunnels to a specific capsule type…" |
| rfc_q24 | 0/5 | no | Y | "QUIC does not have a single fixed transport parameter…" |

Two observations that are not judge artefacts:

* **Citations are perfect and answers are not.** All 24 answers cited at least
  one chunk from a document that genuinely holds the answer. The retrieval stage
  got the endpoint to the right document 24/24 times, and the synthesis stage
  still concluded "not stated here" 15 times. That gap is the chunker: the right
  document was retrieved with the wrong half of its text in it.
* **The system fails safe, not loudly.** The dominant failure is an explicit
  "there is no mention of X in the provided text", not a fabricated value. One
  clear hallucination: `rfc_q06` cites "RFC 9003", which does not exist.
  `rfc_q13` states a wrong range (0-6, default "low" instead of 0-7, default 3).

### Judge-suspect rows — for a Sonnet-voter rerun, not adjudicated here

Flagged mechanically by `judge_e2e.py` (verdict/reason inconsistency, or a split
vote), the known `qwen3.5:4b` failure mode from
`eval/decisions/phase4-escalation-rerun.md` §6:

| row | k/5 | why it is suspect |
|---|---:|---|
| `rfc_q04` | 3/5 | split vote, no majority worth trusting |
| `rfc_q15` | 5/5 | vote 3's reason is phrased as a rejection while the verdict is `true` |
| `rfc_q20` | 2/5 | split vote **and** vote 5 votes `true` with a reason my screen reads as negative |
| `rfc_q21` | 2/5 | split vote **and** vote 1 votes `true` with a reason my screen reads as negative |

A second, separate mechanical screen (`results/orientation_flags.txt`) — rows
the judge failed even though the system answer contains ≥60% of the gold
answer's distinctive tokens (hex codes, 3+-digit numbers, snake_case
identifiers, ALLCAPS names): **`rfc_q02`, `rfc_q09`, `rfc_q21`, `rfc_q22`,
`rfc_q23`.** These are candidates for the same rerun. I have not decided any of
them. Two are worth the gate's attention specifically because the *orientation*
may be doing the work rather than the judge:

* `rfc_q09` — the system answer states "clients MUST establish a QUIC connection
  to UDP port 853" and that port 53 should not be used, hedged with "unless
  there is a specific mutual agreement". Every one of the 5 votes rejected the
  gold answer's phrase "explicitly forbidden" as unsupported by that hedge.
* `rfc_q02` — the system answer asserts the *opposite* of the gold ("no explicit
  floor"), so the 0/5 looks correct; it is on the screen only because it repeats
  the parameter name. The screen over-flags by construction and is offered as a
  filter, not a finding.

**Caveat on the judging orientation, stated because it changes how the 4/24
should be read.** With `EVIDENCE = system answer` and `ANSWER = gold answer`,
the question being asked is "does the system's answer contain everything the
gold answer asserts?" — a recall check on the system answer, not a groundedness
check. A system answer that is correct but states less than the gold answer
fails. This is the mapping the task specified and I ran it as specified, but the
4/24 is a lower bound on correctness, not an estimate of it.

---

## 7. Anomalies and things that did not go as expected

1. **387 chunks for 1.44 MiB.** Noticed as "suspiciously few" before gate 2 ran;
   it was the first symptom of §4.
2. **`--retry on` measured worse than `--retry off`** on this corpus (nDCG@10
   0.355 vs 0.392, recall@10 0.500 vs 0.542) while costing 5x the latency. 24
   rows; a diagnostic only.
3. **`rfc_q21` took 465 s** — 5x the median. It is the row whose answer needed
   RFC 8441; the agent decomposed it and ran several sub-queries.
4. **No `FRONT-TRUNCATED` warnings at all**, across the 34.8-minute enrichment
   build (387 enrichment calls), the 24-query E2E run and 120 judge calls.
5. **`rfc_q06`'s answer cites a non-existent "RFC 9003".** The only outright
   fabrication in 24 answers.
6. The gold set passed all six row-level gates on the first run, with no string
   needing a uniqueness exemption — which is worth recording because RFC prose
   is far more repetitive than the synthetic corpora, and I expected to need
   exemptions.

## 8. Verdict, and the three weakest spots

*(Pre-fix verdict, superseded by [P6](#p6-revised-verdict).)*

**Does the setup work on unseen real documents? Not today.** The retrieval stage
is fine — on the content that reaches the index it finds the right document
every time and the right chunk 93% of the time at k=10. Everything downstream is
undermined by content that never got indexed, and the one Phase-4 mechanism this
corpus was built to exercise does not fire at all.

The three weakest spots, named:

1. **`MarkdownRecursiveChunker._split_text` drops ~half of any document over
   10,000 tokens.** Highest severity, cheapest fix, and it invalidates the
   `docs`/`mixed` baseline slightly (`design_rationale.md` at 0.49 retention).
   A background task has been filed with the repro and the measurement. Nothing
   else on this page should be re-measured until it is fixed.
2. **Cross-reference resolution is filename-shaped and real corpora are not.**
   0/731 resolved. The fix that would actually pay here is not a better filename
   heuristic — it is keeping the `[…]` qualifier that `_SECTION_RE` currently
   discards (355 explicit cross-document section references in this corpus) and
   resolving symbolic reference tags through each document's own References
   section. Until then, treat the index-time half of roadmap 4.2 as
   demonstrated only on the corpus it was co-designed with.
3. **Synthesis gives up rather than degrading.** 15 of 24 answers are "the
   provided text does not mention this", including on rows where retrieval
   ranked the answer-bearing document first. Some of that is (1); some of it is
   a 9b model being asked to answer from chunks that carry an
   enrichment-generated preamble plus dense RFC prose. Worth re-running once (1)
   is fixed, because right now the two causes cannot be separated.

Honourable mention, not in the top three because it is a measurement property
rather than a defect: **the `requires_crossref` slice is once again the *easier*
slice** (reachable rows: recall@10 1.000 vs 0.875, nDCG 0.696 vs 0.529). Two
corpora now agree. A crossref gold row that names document A and asks about B's
subject matter is not a hard retrieval problem, because it leaks lexical signal
from both ends. If Phase 4.2 needs a discriminative first-stage metric, the gold
set that provides it has to ask about the *pointer* and nothing about the
target's content — and that gold set still does not exist.

---

# Post-fix re-run (2026-08-13)

The gate validated the §4 finding and fixed
`MarkdownRecursiveChunker._split_text` in `rag_system/ingestion/chunking.py`
(separator now reattached to the segment that follows it; `seg0` emitted).
Everything in §3-§6 was re-measured from scratch against the fixed chunker:
both scratch indexes deleted and rebuilt, gold set and corpus untouched. The
pre-fix outputs are preserved under `prefix_results/` and as
`results/*.prefix.jsonl`.

I did not edit `rag_system/` — the diff is the gate's. My only change to my own
tooling was adding resume support to `judge_e2e.py` after the judging process
was killed at row 17 by a harness timeout; the 17 completed rows were kept and
the remaining 7 judged on restart, same model, same prompt, same k.

## P1. The chunker fix, verified independently

`chunker_loss_repro.py` re-run unchanged:

| Measurement | pre-fix | post-fix |
|---|---:|---:|
| Synthetic repro: segments kept (8 paragraphs, `max_chunk_size=200`) | `[1,3,5,7]` | **`[0,1,2,3,4,5,6,7]`** |
| Synthetic repro: character ratio | 0.50 | **1.00** |
| `rfc` corpus, characters retained through convert → chunk | **0.52** | **1.02** |
| RFC 9000 (93,395 tokens) | 0.50 | 1.02 |
| RFC 9114 (38,505 tokens) | 0.45 | 1.02 |
| `Documentation/design_rationale.md` (13,285 tokens) | 0.49 | 1.09 |

The >1.00 ratios are the chunker's one-sentence overlap duplicating text at
chunk boundaries, which was always there; the 12 RFCs over the 10,000-token
threshold now behave exactly like the 11 under it.

## P2. Index rebuild and gate 2

| | pre-fix | post-fix |
|---|---:|---:|
| Harness-config index (no LLM) | 387 chunks, 106.2 s | **683 chunks, 142.4 s** |
| Product-defaults index | 387 chunks, 2089.2 s (34.8 min) | **683 chunks, 3096.7 s (51.6 min)** |
| — document processing + 23 overviews | 201.8 s | 163.5 s |
| — contextual enrichment | 1763.8 s | 2764.2 s |
| — embedding / late-chunk / overview-embed | 52.4 / 65.6 / 1.5 s | 91.4 / 71.9 / 1.3 s |
| **Gate 2 — gold rows reachable in an indexed chunk** | **14 / 24** | **24 / 24** |

`gold coverage 24/24 rows reachable in the index`, `coverage_failures` empty.
The 76% index-time increase is 76% more chunks to enrich, at the same
0.25 chunks/s. Ollama is shared with nothing else during the build, but it is a
local service and this is one measurement on one machine.

**No `Ollama likely FRONT-TRUNCATED` warning in any post-fix log** — index
build, E2E run, judge run. Count 0 in all three, as before.

## P3. Cross-reference extraction — unchanged, and now measured on the whole corpus

```
🔗 Cross-references: 1403 reference(s) in 485 chunk(s); 0 resolved to 0 document(s).
```

| | pre-fix (387 chunks) | post-fix (683 chunks) |
|---|---:|---:|
| references extracted | 731 | **1403** |
| of which `section` / `exhibit` / `document` | 703 / 28 / 0 | **1334 / 69 / 0** |
| **resolved** | **0** | **0** |
| documents linked | 0 | 0 |
| resolved under `RFC 9000.txt` naming | 91 (18 docs) | 124 (20 docs) |
| resolved under `<Title> - RFC 9000.txt` naming | 35 (11 docs) | 53 (12 docs) |
| document names occurring anywhere in corpus text (shipped naming) | 0 / 23 | **0 / 23** |
| chunks hitting `MAX_CROSSREFS_PER_CHUNK` | 8 | 23 |

The finding is unchanged and is now measured against the complete corpus rather
than half of it: **0 of 1403.** Every word of §3's diagnosis stands, including
the 355 `Section N of [TAG]` qualified cross-document references whose `[TAG]`
`_SECTION_RE` discards, and the 52 symbolic reference tags
(`[QUIC-TRANSPORT]`, `[QUIC-TLS]`, …) that no filename heuristic can resolve.

## P4. Retrieval, before → after

Identical configuration both times: `run_eval.py` machinery, enrichment /
overviews / late chunking / context expansion OFF, reranker OFF, `k = 20`,
`chunk_size = 512`, `--retry off`, `microsoft/harrier-oss-v1-0.6b`, hybrid
search. `final == first_stage` invariant held on all 24 queries in both runs.

| Slice | n | recall@5 | recall@10 | recall@20 | nDCG@10 |
|---|---:|---|---|---|---|
| **all rows** | 24 | 0.417 → **0.750** | 0.542 → **0.833** | 0.583 → **0.958** | 0.392 → **0.659** |
| `requires_crossref=true` | 10 | 0.500 → **0.600** | 0.600 → **0.800** | 0.600 → **1.000** | 0.518 → **0.719** |
| control (`=false`) | 14 | 0.357 → **0.857** | 0.500 → **0.857** | 0.571 → **0.929** | 0.303 → **0.616** |
| rows unreachable pre-fix | 10 | 0.000 → **0.700** | 0.000 → **0.800** | 0.000 → **1.000** | 0.100 → **0.723** |

"Reachable-only" is no longer a separate cut — all 24 rows pass gate 2, so that
slice *is* the corpus. The right pre-fix comparison for the honest reader is
therefore the pre-fix reachable-only line, and even against that the fix is a
clear win on ranking:

| | n | recall@5 | recall@10 | recall@20 | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| pre-fix, reachable rows only | 14 | 0.714 | 0.929 | 1.000 | 0.601 |
| **post-fix, all rows** | 24 | 0.750 | 0.833 | 0.958 | **0.659** |

recall@10 dips (0.929 → 0.833) because the post-fix figure is over 10 harder
rows that previously could not be scored at all, not because anything regressed;
recall@20 is 0.958 with the two misses being `rfc_q19` (recall 0 at @5/@10, 1 at
@20) and `rfc_q18` (a `match: "all"` row that needs both salts).

By dimension, post-fix: procedural 1.000 recall@10 / 1.000 nDCG, factoid 0.875 /
0.634, negative 0.667 / 0.421, comparative 0.500 / 0.702; easy 0.889 / 0.623,
hard 0.800 / 0.680.

Against the authored-synthetic baseline (`eval/BASELINE.md`, same embedder, same
`k`, reranker off):

| Corpus | n | chunks | recall@5 | recall@10 | recall@20 | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| `acq+docs` | 48 | 373 | 0.917 | 0.958 | 1.000 | 0.738 |
| `acq+docs`, crossref slice | 11 | 373 | 1.000 | 1.000 | 1.000 | 0.748 |
| **`rfc`, post-fix, all rows** | 24 | 683 | 0.750 | 0.833 | 0.958 | **0.659** |
| **`rfc`, post-fix, crossref slice** | 10 | 683 | 0.600 | 0.800 | 1.000 | **0.719** |

Unseen-real is still measurably harder than authored-synthetic — recall@10
0.833 vs 0.958, nDCG@10 0.659 vs 0.738 — but the gap is now a retrieval gap of
the size you would expect from same-domain distractors, not the collapse the
pre-fix numbers showed. The crossref slice is within 0.03 nDCG of the `acq`
crossref slice.

With `--retry on` the retry now fires on **1/24** queries (was 6/24) — better
first-pass evidence means less to retry — and scores slightly below `--retry
off` (nDCG@10 0.617 vs 0.659, recall@10 0.792 vs 0.833), the same sign as
pre-fix. 24 rows; a diagnostic, not a verdict on the retry.

## P5. End-to-end judged answers, before → after

Same protocol both times: in-process `Agent`, product defaults, `force_rag=True`,
`_query_cache.clear()` per query, `qwen3.5:4b` judge, prompt v1, **k = 5**,
`EVIDENCE = system answer`, `ANSWER = gold answer`, verifier suffix stripped by
`judge.py` as before. Zero exceptions in either run.

| Slice | n | majority pass (≥3/5) | unanimous pass | unanimous fail | mean k/5 | cited an expected source |
|---|---:|---|---|---|---|---|
| all | 24 | 4 → **3** | 2 → 2 | 17 → 19 | 0.92 → 0.67 | 24/24 → **24/24** |
| single-doc | 14 | 3 → **0** | 1 → 0 | 11 → 13 | 0.86 → 0.14 | 14/14 → 14/14 |
| `requires_crossref` | 10 | 1 → **3** | 1 → 2 | 6 → 6 | 1.00 → 1.40 | 10/10 → 10/10 |

`k_true` histogram: `{0:17, 1:1, 2:2, 3:1, 4:1, 5:2}` → `{0:19, 1:1, 2:1, 3:1, 5:2}`.
Per-query wall clock: mean 88.7 → 83.5 s, median 59.4 → 58.6 s, max 465.0 →
222.2 s, total 2130 → 2003 s.

Per row:

| id | pre | post | Δ | crossref | was unreachable pre-fix |
|---|---:|---:|---:|---|---|
| rfc_q01 | 0/5 | 0/5 | 0 | | |
| rfc_q02 | 0/5 | 0/5 | 0 | | yes |
| rfc_q03 | 0/5 | 0/5 | 0 | | |
| rfc_q04 | 3/5 | 0/5 | **−3** | | |
| rfc_q05 | 4/5 | 0/5 | **−4** | | |
| rfc_q06 | 0/5 | 0/5 | 0 | | |
| rfc_q07 | 5/5 | 2/5 | **−3** | | |
| rfc_q08 | 0/5 | 0/5 | 0 | | |
| rfc_q09 | 0/5 | 0/5 | 0 | | yes |
| rfc_q10 | 0/5 | 0/5 | 0 | | yes |
| rfc_q11 | 0/5 | 0/5 | 0 | | yes |
| rfc_q12 | 0/5 | 0/5 | 0 | | |
| rfc_q13 | 0/5 | 0/5 | 0 | | yes |
| rfc_q14 | 0/5 | 0/5 | 0 | | yes |
| rfc_q15 | 5/5 | 5/5 | 0 | Y | yes |
| rfc_q16 | 0/5 | 0/5 | 0 | Y | |
| rfc_q17 | 0/5 | 0/5 | 0 | Y | yes |
| rfc_q18 | 0/5 | 0/5 | 0 | Y | yes |
| rfc_q19 | 0/5 | 0/5 | 0 | Y | |
| rfc_q20 | 2/5 | 1/5 | −1 | Y | |
| rfc_q21 | 2/5 | 3/5 | **+1** | Y | |
| rfc_q22 | 0/5 | 0/5 | 0 | Y | |
| rfc_q23 | 1/5 | 5/5 | **+4** | Y | |
| rfc_q24 | 0/5 | 0/5 | 0 | Y | yes |

**4 → 3 is not a real difference at n = 24 with a 4b judge, and I am not
claiming one.** Three things in this table *are* worth reading, and all three are
checkable against the recorded answers:

1. **Refusals halved.** Answers that open by saying the corpus does not contain
   the fact: **11/24 → 6/24** (`q01, q03, q08, q17, q23` all stopped refusing;
   `q10, q12, q14, q18, q19, q22` still refuse). That is the chunker fix showing
   up in synthesis, and it is the effect the fix was supposed to have.
2. **The crossref slice improved and the single-doc slice did not.** crossref
   1/10 → 3/10 (mean k 1.00 → 1.40); single-doc 3/14 → 0/14 (mean k 0.86 →
   0.14). `rfc_q23` is the cleanest single win in the whole exercise: pre-fix it
   answered "no direct facts linking Connect-UDP tunnels to a specific capsule
   type" and attributed the capsule to a non-existent "RFC 9076"; post-fix it
   answers "Capsule Type 0x00 (the DATAGRAM Capsule) … Section 3.5", 5/5.
3. **Three single-doc rows genuinely got worse, and it is not a judge flip.**
   Reading the recorded answers:
   * `rfc_q04` (3/5 → 0/5): pre-fix the answer stated "**SHOULD be set to 333
     milliseconds**". Post-fix it cites RFC 9002 §5.3 (`smoothed_rtt`/`rttvar`)
     and says RTT is treated as infinite until an acknowledgement arrives —
     never giving 333 ms. More indexed text pulled synthesis to a different,
     also-relevant part of the same document. All 5 judge reasons say exactly
     that.
   * `rfc_q05` (4/5 → 0/5): post-fix the answer says "one or two probe
     datagrams", quoting RFC 9002's §6.2 overview sentence, and drops the "MUST
     send at least one ack-eliciting packet" clause the pre-fix answer quoted.
     The judge rejects the gold answer's "ack-eliciting" and "full-sized" on
     that basis.
   * `rfc_q07` (5/5 → 2/5): the answer still gives the right value,
     **0x6b3343cf**, but adds an unsupported explanation of how the constant was
     derived; 3 of 5 votes reject on the explanation, not the value. The
     product's own verifier flagged it too — the answer carries
     `[Confidence: 90%] [Warning: Low confidence. Groundedness: False]`, which
     `judge.py` strips before judging.

**Hallucinated citations persist and did not improve.** Answers citing RFC
numbers absent from the corpus: 8/24 pre-fix, 9/24 post-fix. Some are real
documents the corpus text itself references (6298 for TCP's RTO, 6455, 7540) and
are fair; others are plain misattributions — pre-fix `rfc_q23` credited the
DATAGRAM capsule to "RFC 9076", post-fix `rfc_q06` credits the QUIC datagram
extension to "RFC 9287" instead of RFC 9221. Every answer in both runs carries
the verifier suffix, so the verifier is running; it is not preventing these.

### Judge-suspect rows — for a Sonnet-voter rerun, not adjudicated here

| run | verdict/reason inconsistency or split vote | orientation screen (judged fail, ≥60% of gold's distinctive tokens present) |
|---|---|---|
| pre-fix | `rfc_q04`, `rfc_q15`, `rfc_q20`, `rfc_q21` | `rfc_q02`, `rfc_q09`, `rfc_q21`, `rfc_q22`, `rfc_q23` |
| **post-fix** | **`rfc_q07`, `rfc_q13`, `rfc_q21`** | **`rfc_q02`, `rfc_q07`, `rfc_q16`, `rfc_q22`** |

Post-fix detail: `rfc_q07` 2/5 and `rfc_q21` 3/5 are split votes with no
majority worth trusting; `rfc_q13` has one vote whose reason my screen reads as
positive against a `false` verdict. `rfc_q07` appears on both lists — it is the
single strongest candidate for a stronger voter, since the disputed content is
an added explanation rather than the answer itself. I have adjudicated none of
them.

The orientation caveat from §6 still applies unchanged and matters more now that
refusals have halved: with `EVIDENCE = system answer` and `ANSWER = gold`, an
answer that is correct but says *less* than the gold answer, or *more*, fails.
`rfc_q05` and `rfc_q07` are both of that shape. **3/24 is a lower bound on
correctness, not an estimate of it.**

## P6. Revised verdict

**Retrieval on unseen real documents now works; answer synthesis and
cross-reference extraction do not.**

The chunker fix moved the corpus from half-indexed to fully indexed
(gate 2 14/24 → 24/24, character retention 0.52 → 1.02) and retrieval moved with
it: recall@20 0.583 → 0.958, nDCG@10 0.392 → 0.659, and the crossref slice to
1.000 recall@20 / 0.719 nDCG — within 0.03 nDCG of the synthetic `acq` corpus it
was designed against. Unseen-real remains harder than authored-synthetic
(nDCG@10 0.659 vs 0.738), which is the honest size of the same-domain-distractor
penalty and is a normal number, not a failure.

The two weakest spots, revised:

1. **Cross-reference extraction is still inert: 0 of 1403 references resolved.**
   Unchanged by the fix and now measured on the complete corpus. The index-time
   half of roadmap 4.2 produces nothing to hop on for any corpus whose filenames
   are not literal substrings of its own prose. The payoff is not a better
   filename heuristic but keeping the `[TAG]` qualifier `_SECTION_RE` currently
   discards (355 qualified cross-document section references here) and resolving
   symbolic tags through each document's References section.
2. **Synthesis is now the bottleneck, and it is unstable.** 3/24 judged pass
   under a strict orientation, refusals down but not gone (6/24), hallucinated
   RFC attributions flat (9/24 answers), and three single-doc rows that
   regressed because *more* correct retrieved text moved the answer to a
   different, also-relevant part of the right document (`rfc_q04`, `rfc_q05`,
   `rfc_q07`). That last pattern is the one worth chasing: it says the failure
   is context selection and answer composition, not retrieval.

Third, unchanged from §8 and still true: **the `requires_crossref` slice is not
the hard slice.** Post-fix it beats its own control on nDCG (0.719 vs 0.616) and
matches it on recall, and it is the only slice whose E2E pass count improved
(1/10 → 3/10). Two corpora now agree. A first-stage metric that discriminates
for Phase 4.2 needs a gold set that asks about the *pointer* and nothing about
the target's content, and that gold set still does not exist.

---

## 9. Files produced

New files under the repo (for the gate to review and commit):

| Path | What |
|---|---|
| `eval/corpora/rfc/*.txt` | the 23 downloaded RFCs, unmodified |
| `eval/corpora/rfc/download.py` | reproducible download + link-graph check |
| `eval/corpora/rfc/MANIFEST.md` | per-file manifest, selection rationale, exclusions, naming note |
| `eval/corpora/rfc/rfc.facts.json` | 26 answer-bearing anchors (kept inside `rfc/` so `verify_facts.py`'s `corpora/*.facts.json` glob does not pick it up and no existing gate output changes) |
| `eval/goldset/rfc.jsonl` | 24 verified gold rows |
| `eval/verify_rfc_goldset.py` | the row-level gate |

Nothing under `rag_system/`, `backend/`, `Documentation/` or any pre-existing
`eval/` file was modified.

Scratch (`.../scratchpad/rfc_shakedown/`):

| Path | What |
|---|---|
| `build_rfc_goldset.py` | authors the gold set + facts sidecar |
| `run_rfc_eval.py` | injects the `rfc` corpus into `run_eval.py` without editing it |
| `build_product_index.py` | product-defaults index build |
| `run_e2e.py`, `judge_e2e.py` | E2E answers and k=5 judging |
| `crossref_diagnostic.py`, `chunker_loss_repro.py`, `slice_results.py` | the three diagnostics |
| `compare_e2e.py` | before/after table for the judged E2E pass |
| `results/rfc_retrieval_retryoff_postfix.json`, `..._retryon_postfix.json` | **post-fix** retrieval runs |
| `results/rfc_e2e_answers.jsonl`, `results/rfc_e2e_judged.jsonl`, `results/rfc_e2e_judge_summary.json` | **post-fix** answers, all 120 votes, summary |
| `results/*.prefix.jsonl`, `prefix_results/` | the complete pre-fix outputs, preserved unmodified |
| `results/crossref_diagnostic.json`, `results/chunker_loss.json` | diagnostics, re-run post-fix |
| `results/orientation_flags.txt`, `results/orientation_flags_postfix.txt` | the orientation screen, both runs |
| `product_index/` | the throwaway product-defaults index (683 chunks post-fix, overviews, latechunk) |

### Reproducing

```bash
cd /Users/prompt/videos/localgpt_08082026/localGPT
SCR=/private/tmp/claude-501/-Users-prompt-videos-localgpt-08082026/4d62420b-7ab2-4be1-90f2-708d7bae9146/scratchpad/rfc_shakedown

.venv/bin/python eval/corpora/rfc/download.py --check      # corpus + link graph
.venv/bin/python eval/verify_rfc_goldset.py                # gold-set gate
.venv/bin/python $SCR/run_rfc_eval.py --corpus rfc --coverage-only   # gate 2
.venv/bin/python $SCR/run_rfc_eval.py --corpus rfc --retry off \
  --json-out $SCR/results/rfc_retrieval_retryoff.json      # retrieval
.venv/bin/python $SCR/slice_results.py $SCR/results/rfc_retrieval_retryoff.json
.venv/bin/python $SCR/crossref_diagnostic.py
.venv/bin/python $SCR/chunker_loss_repro.py
.venv/bin/python $SCR/build_product_index.py               # ~35 min
.venv/bin/python $SCR/run_e2e.py                           # ~36 min
.venv/bin/python $SCR/judge_e2e.py                         # ~3 min
```

---

## Gate validation and final judged numbers (2026-08-13)

The chunker data-loss claim was independently reproduced at the gate (RFC 9000
retention 49.12% pre-fix; alternating-section synthetic repro) before the fix
(commit 7d71051) was written; the fix is property-tested lossless. The gold-set
gates were re-run at the gate (all pass). Crossref inertness was independently
reproduced (0 resolved on real RFC text). Retrieval was re-run live at the
gate: recall@20 0.958 exactly matches; recall@5/@10 and nDCG@10 reproduced
within one query of the agent's numbers (0.708/0.792/0.617 vs
0.750/0.833/0.659) — small tie-break variance, direction unchanged.

**Final E2E answer quality (Sonnet subagent panel, 3 voters, the validated
judge — supersedes the 4b numbers above):** **5/24 pass** — single-doc 1/14,
requires_crossref 4/10. All 24 rows unanimous across voters; agreement with
the 4b k=5 majorities on 22/24, and both disagreements are rows the 4b flagged
judge-suspect (`rfc_q07`, `rfc_q20` — Sonnet rules both grounded). Per-row
votes in `rfc-shakedown-sonnet-panel.json`.

The bottleneck is therefore answer synthesis on dense unseen technical text,
not retrieval: the right documents are retrieved (recall@20 0.958) and cited
(24/24), but the 9b generation model frequently answers from its own prior
instead of the supplied snippets — e.g. `rfc_q01` asserts a default
`ack_delay_exponent` of -1 and fabricates a supporting quote attributed to
"RFC 9002 Section 13.4" while the correct value (3) sat in the retrieved text.
The verifier correctly marked most of these low-confidence, but the answer
text still leads with the wrong claim.

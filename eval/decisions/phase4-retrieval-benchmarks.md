# Phase 4 items 4.2 and 4.3 — the retrieval benchmark matrix, on rebuilt indexes

Date: 2026-08-09
Status: **measured.** Both mechanisms now fire and both move numbers. Neither
result is a clean win. A PROPOSED call is recorded per item at the end; **the
adoption gate makes the final call, not this document.**

Scope of this wave: `eval/.eval_indexes/**` (rebuilds), `eval/results/**`,
`eval/run_eval.py` (one additive flag, §3.1), `eval/BASELINE.md`, this file.
Nothing under `rag_system/`, `Documentation/`, `src/` or `backend/` was
modified. One `rag_system` design finding is recorded in §2.5 with a proposed
diff and deliberately **not** applied.

Prior art this builds on, and does not repeat:
[`phase4-crossref-prefilter.md`](phase4-crossref-prefilter.md) (what shipped,
plus the *Gate correction (2026-08-09)* appendix that fixed the resolver) and
[`phase4-eval-final-metric.md`](phase4-eval-final-metric.md) (the final-list
metric, and the first A/B — a zero-hop negative result on indexes built before
the resolver fix).

**Determinism protocol, applied to every run below:** `--retry off` (the
evidence-sufficiency retry is an LLM reformulation and is nondeterministic),
reranker off, no empty-string env vars, no `Documentation/` edit between arms.
With those settings the query path makes **no LLM call at all**, so each pair of
arms differs only in the flag under test.

**Latency is not reported and no performance claim is made.** A second agent was
running answer-quality benchmarks against the same Ollama instance throughout;
every wall-clock number in this wave is contended and meaningless.

---

## 1. Index rebuild and re-baseline

### 1.1 Why

`rag_system/indexing/crossref.py` was changed at the gate (numeric-prefix-stripped
filename aliases, so `08_regulatory_approval.pdf` also registers as
`"regulatory approval"`). Cross-references are stamped into chunk metadata **at
index time**, so the existing `acq` / `acq_plus_docs` eval indexes — built before
the extractor existed, and certainly before the fix — carried none of it. Both
were deleted and rebuilt:

```
rm -rf eval/.eval_indexes/microsoft__harrier-oss-v1-0.6b/acq \
       eval/.eval_indexes/microsoft__harrier-oss-v1-0.6b/acq_plus_docs

.venv/bin/python eval/run_eval.py --corpus acq --retry off --crossref-hop off \
  --overview-prefilter off \
  --json-out eval/results/phase4_w3_rebuild_acq_hop_off.json
.venv/bin/python eval/run_eval.py --corpus acq+docs --retry off --crossref-hop off \
  --overview-prefilter off \
  --json-out eval/results/phase4_w3_rebuild_acqdocs_hop_off.json
```

Build-time log lines, verbatim:

```
acq       🔗 Cross-references: 68 reference(s) in 11 chunk(s); 34 resolved to 9 document(s).
acq+docs  🔗 Cross-references: 215 reference(s) in 70 chunk(s); 93 resolved to 21 document(s).
```

### 1.2 Verification, read straight out of LanceDB

Not from the build log — from the built table, decoding the `metadata` column
and reading `["metadata"]["crossrefs"]` (`VectorIndexer` stores
`json.dumps(chunk)` there, so the real metadata is nested one level down).

```
LanceDB connection established at: eval/.eval_indexes/microsoft__harrier-oss-v1-0.6b/acq
eval_acq: 13 chunks
chunks carrying crossrefs: 11   total refs: 68   resolved: 34   distinct target documents: 9

resolved edges (source_doc -> target_doc via ref):
  01_acquisition_agreement.pdf  --[document: due diligence report]-->  02_due_diligence_report.pdf
  02_due_diligence_report.pdf   --[document: acquisition agreement]-->  01_acquisition_agreement.pdf
  02_due_diligence_report.pdf   --[document: customer consents]-->     09_customer_consents.pdf
  02_due_diligence_report.pdf   --[document: financial adjustments]--> 05_financial_adjustments.pdf
  02_due_diligence_report.pdf   --[document: regulatory approval]-->   08_regulatory_approval.pdf
  03_ip_certification.pdf       --[document: acquisition agreement]--> 01_acquisition_agreement.pdf
  03_ip_certification.pdf       --[document: legal opinion]-->         06_legal_opinion.pdf
  03_ip_certification.pdf       --[document: risk assessment]-->       04_risk_assessment.pdf
  04_risk_assessment.pdf        --[document: acquisition agreement]--> 01_acquisition_agreement.pdf
  04_risk_assessment.pdf        --[document: closing checklist]-->     10_closing_checklist.pdf
  04_risk_assessment.pdf        --[document: due diligence report]-->  02_due_diligence_report.pdf
  04_risk_assessment.pdf        --[document: financial adjustments]--> 05_financial_adjustments.pdf
  04_risk_assessment.pdf        --[document: regulatory approval]-->   08_regulatory_approval.pdf
  05_financial_adjustments.pdf  --[document: acquisition agreement]--> 01_acquisition_agreement.pdf
  05_financial_adjustments.pdf  --[document: closing checklist]-->     10_closing_checklist.pdf
  05_financial_adjustments.pdf  --[document: due diligence report]-->  02_due_diligence_report.pdf
  05_financial_adjustments.pdf  --[document: risk assessment]-->       04_risk_assessment.pdf
  06_legal_opinion.pdf          --[document: acquisition agreement]--> 01_acquisition_agreement.pdf
  06_legal_opinion.pdf          --[document: due diligence report]-->  02_due_diligence_report.pdf
  06_legal_opinion.pdf          --[document: ip certification]-->      03_ip_certification.pdf
  06_legal_opinion.pdf          --[document: regulatory approval]-->   08_regulatory_approval.pdf
  07_nda.pdf                    --[document: acquisition agreement]--> 01_acquisition_agreement.pdf
  07_nda.pdf                    --[document: due diligence report]-->  02_due_diligence_report.pdf
  07_nda.pdf                    --[document: ip certification]-->      03_ip_certification.pdf
  08_regulatory_approval.pdf    --[document: acquisition agreement]--> 01_acquisition_agreement.pdf
  08_regulatory_approval.pdf    --[document: closing checklist]-->     10_closing_checklist.pdf
  08_regulatory_approval.pdf    --[document: due diligence report]-->  02_due_diligence_report.pdf
  08_regulatory_approval.pdf    --[document: risk assessment]-->       04_risk_assessment.pdf
  09_customer_consents.pdf      --[document: acquisition agreement]--> 01_acquisition_agreement.pdf
  09_customer_consents.pdf      --[document: due diligence report]-->  02_due_diligence_report.pdf
  10_closing_checklist.pdf      --[document: acquisition agreement]--> 01_acquisition_agreement.pdf
  10_closing_checklist.pdf      --[document: due diligence report]-->  02_due_diligence_report.pdf
  10_closing_checklist.pdf      --[document: financial adjustments]--> 05_financial_adjustments.pdf
  10_closing_checklist.pdf      --[document: regulatory approval]-->   08_regulatory_approval.pdf

unresolved refs: {('exhibit','exhibit a'): 4, ('exhibit','schedule 3'): 5,
  ('exhibit','exhibit b'): 4, ('exhibit','exhibit c'): 6, ('exhibit','schedule 1'): 5,
  ('exhibit','schedule 2'): 4, ('section','section 1.5'): 1, ('section','section 2.2'): 2,
  ('section','section 4.1'): 1, ('section','section 1.1'): 1, ('section','section 4'): 1}

self-resolution check: NONE (good)
```

**34 resolved references, 9 of 10 documents linked, no self-edges** — exactly what
the gate predicted from corpus text. `07_nda.pdf` is the one document nothing
points *at*; it points at three others. The 34 unresolved refs are the Exhibits
and Schedules, which are sections *inside* `01_acquisition_agreement.pdf` rather
than separate files, plus the bare `section N` forms — correctly left `null`,
unchanged by the fix and not fixable by a filename-based resolver.

`acq+docs` shows the same 34 acquisition edges plus 59 more from
`Documentation/*.md` title mentions (93 resolved / 21 documents).

### 1.3 Re-baseline — no drift

Gold coverage on both rebuilt indexes: **24/24** (`acq`) and **48/48**
(`acq+docs`), `coverage_failures` empty. Chunk counts unchanged (13 / 373).

| corpus | slice | n | R@5 | R@10 | R@20 | nDCG@10 (1st) | previous retry-off figure |
|---|---|---|---|---|---|---|---|
| `acq` | all | 24 | 0.958 | 1.000 | 1.000 | **0.8101** | 0.8101 ✅ |
| `acq` | `requires_crossref=true` | 11 | 1.000 | 1.000 | 1.000 | **0.7477** | 0.7477 ✅ |
| `acq` | control (`=false`) | 13 | 0.923 | 1.000 | 1.000 | **0.8628** | 0.8628 ✅ |
| `acq+docs` | all | 48 | 0.854 | 0.896 | 0.958 | **0.7194** | 0.7194 ✅ |
| `acq+docs` | `requires_crossref=true` | 11 | 1.000 | 1.000 | 1.000 | **0.7477** | 0.7477 ✅ |
| `acq+docs` | control (`=false`) | 13 | 0.769 | 0.769 | 0.846 | **0.7731** | 0.7731 ✅ |

**Zero drift to four decimals on every cell.** The crossref-slice first-stage
nDCG@10 baseline of 0.748 reproduces exactly. That is the expected result and it
is worth stating why: cross-reference extraction writes only chunk *metadata* —
it moves neither the `text` column nor a bit of any vector — so a rebuild that
adds 34 resolved references cannot change a first-stage ranking. It is also the
control that says the rebuild introduced nothing else.

The `final == first_stage` invariant passed on both rebuild runs (24/24 and
48/48 queries, chunk-id order and both metric families).

---

## 2. Item 4.2 — the cross-reference hop A/B

Twelve runs: `{acq, acq+docs}` × `k ∈ {3, 5, 20}` × `{hop off, hop on}`, all
`--retry off`, all on the rebuilt indexes. Command shape:

```
.venv/bin/python eval/run_eval.py --corpus <acq|acq+docs> --retry off \
  --crossref-hop <off|on> --overview-prefilter off --k <3|5|20> \
  --json-out eval/results/phase4_w3_42_<tag>_k<k>_hop_<arm>.json
```

### 2.1 The full matrix

`hop q` = queries on which the hop fired. `chunks` = chunks it appended in
total. `hit_src` = queries where a hopped chunk came from a document listed in
the gold row's `expected_sources` (document-level hop precision). `rel` =
queries where a hopped chunk actually contains the gold `expected` text
(text-level hop precision).

| corpus | k | arm | slice | n | R@5 | R@10 | R@20 | nDCG@10 (1st) | R@5 (fin) | R@10 (fin) | R@20 (fin) | **nDCG@10 (fin)** | hop q | chunks | hit_src | rel |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| acq | 20 | off | all | 24 | 0.958 | 1.000 | 1.000 | 0.8101 | 0.958 | 1.000 | 1.000 | **0.8101** | 0 | 0 | 0 | 0 |
| acq | 20 | **on** | all | 24 | 0.958 | 1.000 | 1.000 | 0.8101 | 0.958 | 1.000 | 1.000 | **0.8101** | **0** | 0 | 0 | 0 |
| acq | 20 | off | xref | 11 | 1.000 | 1.000 | 1.000 | 0.7477 | 1.000 | 1.000 | 1.000 | **0.7477** | 0 | 0 | 0 | 0 |
| acq | 20 | **on** | xref | 11 | 1.000 | 1.000 | 1.000 | 0.7477 | 1.000 | 1.000 | 1.000 | **0.7477** | **0** | 0 | 0 | 0 |
| acq | 20 | off | control | 13 | 0.923 | 1.000 | 1.000 | 0.8628 | 0.923 | 1.000 | 1.000 | **0.8628** | 0 | 0 | 0 | 0 |
| acq | 20 | **on** | control | 13 | 0.923 | 1.000 | 1.000 | 0.8628 | 0.923 | 1.000 | 1.000 | **0.8628** | **0** | 0 | 0 | 0 |
| acq | 5 | off | all | 24 | 0.917 | 0.917 | 0.917 | 0.7875 | 0.917 | 0.917 | 0.917 | **0.7875** | 0 | 0 | 0 | 0 |
| acq | 5 | **on** | all | 24 | 0.917 | 0.917 | 0.917 | 0.7875 | 0.917 | **0.958** | **0.958** | **0.8024** | **24** | 34 | 1 | 1 |
| acq | 5 | off | xref | 11 | 1.000 | 1.000 | 1.000 | 0.7358 | 1.000 | 1.000 | 1.000 | **0.7358** | 0 | 0 | 0 | 0 |
| acq | 5 | **on** | xref | 11 | 1.000 | 1.000 | 1.000 | 0.7358 | 1.000 | 1.000 | 1.000 | **0.7358** | **11** | 16 | **0** | **0** |
| acq | 5 | off | control | 13 | 0.846 | 0.846 | 0.846 | 0.8313 | 0.846 | 0.846 | 0.846 | **0.8313** | 0 | 0 | 0 | 0 |
| acq | 5 | **on** | control | 13 | 0.846 | 0.846 | 0.846 | 0.8313 | 0.846 | **0.923** | **0.923** | **0.8587** | **13** | 18 | 1 | 1 |
| acq | 3 | off | all | 24 | 0.792 | 0.792 | 0.792 | 0.8036 | 0.792 | 0.792 | 0.792 | **0.8036** | 0 | 0 | 0 | 0 |
| acq | 3 | **on** | all | 24 | 0.792 | 0.792 | 0.792 | 0.8036 | **0.875** | **0.875** | **0.875** | **0.8394** | **24** | 37 | 2 | 2 |
| acq | 3 | off | xref | 11 | 0.909 | 0.909 | 0.909 | 0.7868 | 0.909 | 0.909 | 0.909 | **0.7868** | 0 | 0 | 0 | 0 |
| acq | 3 | **on** | xref | 11 | 0.909 | 0.909 | 0.909 | 0.7868 | 0.909 | 0.909 | 0.909 | **0.7868** | **11** | 18 | **0** | **0** |
| acq | 3 | off | control | 13 | 0.692 | 0.692 | 0.692 | 0.8178 | 0.692 | 0.692 | 0.692 | **0.8178** | 0 | 0 | 0 | 0 |
| acq | 3 | **on** | control | 13 | 0.692 | 0.692 | 0.692 | 0.8178 | **0.846** | **0.846** | **0.846** | **0.8840** | **13** | 19 | 2 | 2 |
| acq+docs | 20 | off | all | 48 | 0.854 | 0.896 | 0.958 | 0.7194 | 0.854 | 0.896 | 0.958 | **0.7194** | 0 | 0 | 0 | 0 |
| acq+docs | 20 | **on** | all | 48 | 0.854 | 0.896 | 0.958 | 0.7194 | 0.854 | 0.896 | 0.958 | **0.7194** | **14** | 30 | **0** | **0** |
| acq+docs | 20 | off | xref | 11 | 1.000 | 1.000 | 1.000 | 0.7477 | 1.000 | 1.000 | 1.000 | **0.7477** | 0 | 0 | 0 | 0 |
| acq+docs | 20 | **on** | xref | 11 | 1.000 | 1.000 | 1.000 | 0.7477 | 1.000 | 1.000 | 1.000 | **0.7477** | **0** | 0 | 0 | 0 |
| acq+docs | 20 | off | control | 13 | 0.769 | 0.769 | 0.846 | 0.7731 | 0.769 | 0.769 | 0.846 | **0.7731** | 0 | 0 | 0 | 0 |
| acq+docs | 20 | **on** | control | 13 | 0.769 | 0.769 | 0.846 | 0.7731 | 0.769 | 0.769 | 0.846 | **0.7731** | **8** | 12 | **0** | **0** |
| acq+docs | 5 | off | all | 48 | 0.854 | 0.854 | 0.854 | 0.6996 | 0.854 | 0.854 | 0.854 | **0.6996** | 0 | 0 | 0 | 0 |
| acq+docs | 5 | **on** | all | 48 | 0.854 | 0.854 | 0.854 | 0.6996 | 0.854 | **0.917** | **0.917** | **0.7104** | **33** | 60 | 1 | 3 |
| acq+docs | 5 | off | xref | 11 | 1.000 | 1.000 | 1.000 | 0.7477 | 1.000 | 1.000 | 1.000 | **0.7477** | 0 | 0 | 0 | 0 |
| acq+docs | 5 | **on** | xref | 11 | 1.000 | 1.000 | 1.000 | 0.7477 | 1.000 | 1.000 | 1.000 | **0.7477** | **11** | 13 | **0** | **0** |
| acq+docs | 5 | off | control | 13 | 0.769 | 0.769 | 0.769 | 0.7577 | 0.769 | 0.769 | 0.769 | **0.7577** | 0 | 0 | 0 | 0 |
| acq+docs | 5 | **on** | control | 13 | 0.769 | 0.769 | 0.769 | 0.7577 | 0.769 | **0.846** | **0.846** | **0.7851** | **12** | 17 | 1 | 1 |
| acq+docs | 3 | off | all | 48 | 0.708 | 0.708 | 0.708 | 0.6446 | 0.708 | 0.708 | 0.708 | **0.6446** | 0 | 0 | 0 | 0 |
| acq+docs | 3 | **on** | all | 48 | 0.708 | 0.708 | 0.708 | 0.6446 | **0.771** | **0.792** | **0.792** | **0.6671** | **33** | 64 | 1 | 4 |
| acq+docs | 3 | off | xref | 11 | 0.909 | 0.909 | 0.909 | 0.7413 | 0.909 | 0.909 | 0.909 | **0.7413** | 0 | 0 | 0 | 0 |
| acq+docs | 3 | **on** | xref | 11 | 0.909 | 0.909 | 0.909 | 0.7413 | 0.909 | 0.909 | 0.909 | **0.7413** | **11** | 16 | **0** | **0** |
| acq+docs | 3 | off | control | 13 | 0.692 | 0.692 | 0.692 | 0.7408 | 0.692 | 0.692 | 0.692 | **0.7408** | 0 | 0 | 0 | 0 |
| acq+docs | 3 | **on** | control | 13 | 0.692 | 0.692 | 0.692 | 0.7408 | **0.769** | **0.769** | **0.769** | **0.7740** | **12** | 18 | 1 | 1 |

### 2.2 First stage is identical across every arm

Checked per query, not in aggregate — `recall@{5,10,20}`, `ndcg10_first_stage`
and the candidate count, hop-off vs hop-on:

```
acq      k=20: first-stage recall+nDCG+candidate-count identical on 24/24 queries: True
acq      k=5 : first-stage recall+nDCG+candidate-count identical on 24/24 queries: True
acq      k=3 : first-stage recall+nDCG+candidate-count identical on 24/24 queries: True
acq+docs k=20: first-stage recall+nDCG+candidate-count identical on 48/48 queries: True
acq+docs k=5 : first-stage recall+nDCG+candidate-count identical on 48/48 queries: True
acq+docs k=3 : first-stage recall+nDCG+candidate-count identical on 48/48 queries: True
```

That is the contract (`_crossref_hop` only ever appends to `documents`) verified
on 216 query pairs, and it means every difference in the *final* columns is the
hop and nothing else.

### 2.3 What actually happened

**The hop now fires — a lot.** 0 → 24/24 queries on `acq` at k=3 and k=5, and
33/48 on `acq+docs`. The resolver fix is what changed; nothing else did.

**At k=20 on `acq` it still fires zero times, and that is correct.** `acq` is 13
chunks; k=20 sweeps the entire corpus, so all ten documents are already
`represented` and the "not already a candidate" guard suppresses every target.
There is nothing to fetch. This is the structural constraint
`phase4-crossref-prefilter.md` names, now confirmed on an index that *has*
resolvable references: the hop is a candidate-selection mechanism and cannot act
on a corpus smaller than the candidate budget.

**At k=20 on `acq+docs` it fires 14 times and moves nothing** (0.7194 → 0.7194,
recall bit-identical). Hopped chunks are appended, so at k=20 they land at ranks
21+ and cannot enter nDCG@10 by construction — and recall@20 is likewise a
prefix that ends before them. `hit_expected_source = 0` on all 14.

**At small k it moves final recall — but not on the slice it was built for.**
Every gain is in the `requires_crossref=false` control slice:

| corpus | k | slice | R@10 (fin) off → on | nDCG@10 (fin) off → on |
|---|---|---|---|---|
| acq | 5 | **xref (n=11)** | 1.000 → 1.000 (**+0.000**) | 0.7358 → 0.7358 (**+0.0000**) |
| acq | 5 | control (n=13) | 0.846 → 0.923 (+0.077) | 0.8313 → 0.8587 (+0.0274) |
| acq | 3 | **xref (n=11)** | 0.909 → 0.909 (**+0.000**) | 0.7868 → 0.7868 (**+0.0000**) |
| acq | 3 | control (n=13) | 0.692 → 0.846 (+0.154) | 0.8178 → 0.8840 (+0.0662) |
| acq+docs | 5 | **xref (n=11)** | 1.000 → 1.000 (**+0.000**) | 0.7477 → 0.7477 (**+0.0000**) |
| acq+docs | 5 | control (n=13) | 0.769 → 0.846 (+0.077) | 0.7577 → 0.7851 (+0.0274) |
| acq+docs | 3 | **xref (n=11)** | 0.909 → 0.909 (**+0.000**) | 0.7413 → 0.7413 (**+0.0000**) |
| acq+docs | 3 | control (n=13) | 0.692 → 0.769 (+0.077) | 0.7408 → 0.7740 (+0.0332) |

**On the `requires_crossref` slice the hop changes nothing at any k on either
corpus, and `hit_expected_source` is 0/11 in every single cell.** Not one hop on
a cross-reference query landed in a gold source document.

Two reasons, both true at once:

1. The slice is at ceiling before the hop runs (recall 0.909–1.000 at every k —
   this is the finding `BASELINE.md` already records: "the crossref slice is
   **not** weak"). A mechanism that only *adds* candidates cannot help a query
   whose answer is already retrieved.
2. The hop picks the wrong target. See §2.5.

**The hop never hurt anything.** Recall never fell, nDCG@10 (final) never fell,
in any of the 36 rows above. Appending cannot demote.

### 2.4 The honest control: is the hop better than just retrieving more chunks?

The hop's gains all come from making the final list longer. The fair comparison
is therefore not off-vs-on at equal `k`, but off-vs-on at **equal final list
length**. Mean final list size and final recall/nDCG:

```
acq      k=3 hop ON : mean final list = 4.54 chunks | R@10f=0.875 nDCG@10f=0.8394
acq      k=5 hop ON : mean final list = 6.42 chunks | R@10f=0.958 nDCG@10f=0.8024
acq      k=3 hop off: mean final list = 3.00 chunks | R@10f=0.792 nDCG@10f=0.8036
acq      k=4 hop off: mean final list = 4.00 chunks | R@10f=0.875 nDCG@10f=0.7802
acq      k=5 hop off: mean final list = 5.00 chunks | R@10f=0.917 nDCG@10f=0.7875
acq      k=6 hop off: mean final list = 6.00 chunks | R@10f=0.958 nDCG@10f=0.8115

acq+docs k=3 hop ON : mean final list = 4.33 chunks | R@10f=0.792 nDCG@10f=0.6671
acq+docs k=5 hop ON : mean final list = 6.25 chunks | R@10f=0.917 nDCG@10f=0.7104
acq+docs k=3 hop off: mean final list = 3.00 chunks | R@10f=0.708 nDCG@10f=0.6446
acq+docs k=4 hop off: mean final list = 4.00 chunks | R@10f=0.812 nDCG@10f=0.6858
acq+docs k=5 hop off: mean final list = 5.00 chunks | R@10f=0.854 nDCG@10f=0.6996
acq+docs k=6 hop off: mean final list = 6.00 chunks | R@10f=0.875 nDCG@10f=0.7103
```

Budget-matched, four cells:

| corpus | hop arm | list size | R@10 (fin) | nearest hop-off arm | list size | R@10 (fin) | verdict |
|---|---|---|---|---|---|---|---|
| acq | k=3 on | 4.54 | 0.875 | k=4 off | 4.00 | 0.875 | **tie, hop costs 0.54 more chunks** |
| acq | k=5 on | 6.42 | 0.958 | k=6 off | 6.00 | 0.958 | **tie, hop costs 0.42 more chunks** (and nDCG 0.8024 vs 0.8115 — hop slightly worse) |
| acq+docs | k=3 on | 4.33 | 0.792 | k=4 off | 4.00 | 0.812 | **plain k=4 wins**, with fewer chunks |
| acq+docs | k=5 on | 6.25 | 0.917 | k=6 off | 6.00 | 0.875 | **hop wins**, +0.042 recall for +0.25 chunks |

One cell out of four where the hop beats simply raising `k`. That is the single
most important number in this document: on this corpus and this gold set, **the
cross-reference hop is, to a first approximation, an expensive way of retrieving
one more chunk.** It costs an extra filtered vector search per query; raising
`k` costs nothing.

### 2.5 Mechanism finding — the hop reliably picks the hub, not the target

Per-query hop targets, `acq`, k=3, hop on (`hit_src` / `rel` are the precision
flags; `xref` is `requires_crossref`):

```
id         xref   anchor -> hop target                                                    || expected_sources
acq_q01    False  05_financial_adjustments.pdf --[risk assessment]-->    04_risk_assessment.pdf     || ['01_acquisition_agreement.pdf']
acq_q02    False  07_nda.pdf                   --[due diligence report]->02_due_diligence_report.pdf|| ['07_nda.pdf']
acq_q03    False  03_ip_certification.pdf      --[risk assessment]-->    04_risk_assessment.pdf     || ['03_ip_certification.pdf']
acq_q04    False  06_legal_opinion.pdf         --[due diligence report]->02_due_diligence_report.pdf|| ['02_due_diligence_report.pdf']   hit_src=True rel=True
acq_q05    False  07_nda.pdf                   --[due diligence report]->02_due_diligence_report.pdf|| ['07_nda.pdf', '07_nda.pdf']
acq_q06    False  07_nda.pdf                   --[due diligence report]->02_due_diligence_report.pdf|| ['07_nda.pdf']
acq_q07    False  08_regulatory_approval.pdf   --[acquisition agreement]>01_acquisition_agreement.pdf|| ['08_regulatory_approval.pdf']
acq_q08    False  06_legal_opinion.pdf         --[acquisition agreement]>01_acquisition_agreement.pdf|| ['06_legal_opinion.pdf']
acq_q09    False  05_financial_adjustments.pdf --[acquisition agreement]>01_acquisition_agreement.pdf|| ['02_due_diligence_report.pdf', '05_financial_adjustments.pdf']
acq_q10    False  09_customer_consents.pdf     --[acquisition agreement]>01_acquisition_agreement.pdf|| ['04_risk_assessment.pdf', '09_customer_consents.pdf']
acq_q11    False  05_financial_adjustments.pdf --[acquisition agreement]>01_acquisition_agreement.pdf|| ['05_financial_adjustments.pdf', '10_closing_checklist.pdf']
acq_q12    False  05_financial_adjustments.pdf --[due diligence report]->02_due_diligence_report.pdf|| ['02_due_diligence_report.pdf']   hit_src=True rel=True
acq_q13    True   01_acquisition_agreement.pdf --[due diligence report]->02_due_diligence_report.pdf|| ['08_regulatory_approval.pdf']
acq_q14    True   04_risk_assessment.pdf       --[acquisition agreement]>01_acquisition_agreement.pdf|| ['04_risk_assessment.pdf']
acq_q15    True   05_financial_adjustments.pdf --[due diligence report]->02_due_diligence_report.pdf|| ['05_financial_adjustments.pdf']
acq_q16    True   03_ip_certification.pdf      --[risk assessment]-->    04_risk_assessment.pdf     || ['03_ip_certification.pdf']
acq_q17    True   07_nda.pdf                   --[acquisition agreement]>01_acquisition_agreement.pdf|| ['02_due_diligence_report.pdf']
acq_q18    True   04_risk_assessment.pdf       --[acquisition agreement]>01_acquisition_agreement.pdf|| ['03_ip_certification.pdf']
acq_q19    True   06_legal_opinion.pdf         --[acquisition agreement]>01_acquisition_agreement.pdf|| ['09_customer_consents.pdf'] ×3
acq_q20    True   08_regulatory_approval.pdf   --[acquisition agreement]>01_acquisition_agreement.pdf|| ['08_regulatory_approval.pdf']
acq_q21    True   10_closing_checklist.pdf     --[acquisition agreement]>01_acquisition_agreement.pdf|| ['05_financial_adjustments.pdf']
acq_q22    True   08_regulatory_approval.pdf   --[acquisition agreement]>01_acquisition_agreement.pdf|| ['10_closing_checklist.pdf']
acq_q23    True   01_acquisition_agreement.pdf --[due diligence report]->02_due_diligence_report.pdf|| ['05_financial_adjustments.pdf', '08_regulatory_approval.pdf']
acq_q24    False  08_regulatory_approval.pdf   --[acquisition agreement]>01_acquisition_agreement.pdf|| ['08_regulatory_approval.pdf']
```

Target concentration across the on-arms:

```
acq k=3      (24 targets):  13 × 01_acquisition_agreement.pdf,  8 × 02_due_diligence_report.pdf,  3 × 04_risk_assessment.pdf
acq k=5      (24 targets):  10 × 01_acquisition_agreement.pdf,  8 × 02_due_diligence_report.pdf,  3 × 08_regulatory_approval.pdf, 1 each × 3 others
acq+docs k=3 (33 targets):  10 × 01_acquisition_agreement.pdf,  9 × 02_due_diligence_report.pdf,  9 × retrieval_pipeline.md, 2 × 04_risk_assessment.pdf, 1 each × 3 others
acq+docs k=5 (33 targets):  10 × 02_due_diligence_report.pdf,   7 × retrieval_pipeline.md,        6 × 01_acquisition_agreement.pdf, 3 × verifier.md, …
```

**21 of 24 hops on `acq` k=3 go to one of two documents.** The reason is in
`rag_system/pipelines/retrieval_pipeline.py::_crossref_hop`: candidate targets
are collected by scanning the top-3 candidates in rank order and, within each,
the chunk's `crossrefs` list *in order of first appearance in the text*; the list
is then truncated with `targets = targets[:max_hops]` and `max_hops` is 1. So the
winner is whichever document the top candidate happens to mention first — and in
a deal room every document opens by naming the master agreement. Query relevance
never enters the target choice at any point.

**This is a design finding, not a crash, and it is deliberately not fixed here**
(`rag_system/` is outside this wave's ownership). The proposed diff, for the
gate:

```python
# rag_system/pipelines/retrieval_pipeline.py, in _crossref_hop, replacing
#     targets = targets[:max_hops]

        # Order candidate targets by evidence that the query is about them,
        # not by where the reference happens to sit in the chunk text. Two
        # signals, both free: how many of the top-3 candidates point at the
        # same document (agreement), and how highly ranked the referring
        # candidate was. Without this, `max_hops=1` on a corpus with a hub
        # document sends every query to the hub — measured 21/24 on `acq`.
        votes: Dict[str, int] = {}
        for t in targets:
            votes[t["target_doc"]] = votes.get(t["target_doc"], 0) + 1
        targets.sort(key=lambda t: (-votes[t["target_doc"]], t["from_rank"]))
        targets = targets[:max_hops]
```

Honest caveat on that diff: it is **untested and unmeasured**, it would not have
rescued a single `requires_crossref` row here (those rows are already at recall
ceiling, so there is nothing for a better-chosen hop to add), and on this corpus
the hub is also the most-voted-for document, so it may well change nothing. It is
recorded because the current selection rule is indefensible on inspection, not
because there is evidence it costs recall.

A second, cheaper option the gate may prefer: raise `max_hops` from 1 to 2–3, so
the hub does not crowd out the specific reference. That trades precision for
context length and needs its own arm.

---

## 3. Item 4.3 — the overview prefilter

### 3.1 Making it measurable: one additive change to `eval/run_eval.py`

Before this wave, 4.3 could not be measured at all. The prefilter scores the
query against `index_store/overviews/<id>.vectors.npz`, a sidecar written at the
end of an index build by `IndexingPipeline` step 4 — and only when
`overview.enabled` is true. The eval harness hard-set `cfg["overview"] =
{"enabled": False}` (one LLM call per document, same reason enrichment is off),
so no eval index has ever had a sidecar, and `RetrievalPipeline._overview_vectors`
correctly degraded to a one-line "no embedded overviews were found" no-op.

The change, all inside `eval/run_eval.py`, all additive, default unchanged:

* **`--overviews {off,on}`**, default `off`. `on` sets
  `cfg["overview"] = {"enabled": True, "embed": True}`.
* **`cfg["overview_path"]` is redirected into the corpus's own eval index
  directory** (`eval/.eval_indexes/<embedder>/<corpus>_ov/overviews.jsonl`)
  rather than the repo's shared `index_store/overviews/`. The sidecar is then
  owned by the index, deleted with it, and cannot be read by the wrong corpus.
  `RetrievalPipeline._overview_vectors` resolution rule 2 (`config["overview_path"]`
  with `.jsonl` → `.vectors.npz`) picks it up with no pipeline change.
* **An overview build gets its own index directory** (`<corpus>_ov`). Sharing one
  would make every alternation between `--overviews off` and `--overviews on` a
  full re-index, and would delete an index another process is reading.
* **`index_fingerprint` gains `"overviews": True` — only when true.** Adding the
  key unconditionally would have invalidated every index cached before the flag
  existed and forced needless rebuilds of `mixed`, `docs`, `atlas7`, `hr`.
* The run header and `run.overviews` in the results JSON report the state.

Nothing else changed; `py_compile` clean.

**Verification that this does not alter what is being measured.** An
overview-enabled build must be chunk-for-chunk identical to a normal one —
`OverviewBuilder.build_and_store` only appends a JSONL line. Checked against the
`acq_plus_docs` index built in §1:

```
chunks: 373 373
chunk_id sets identical: True
text identical         : True
vector shapes          : (373, 1024) (373, 1024)
max abs vector delta   : 0.0
```

And the sidecar really is produced, in the right place:

```
🔗 Cross-references: 215 reference(s) in 70 chunk(s); 93 resolved to 21 document(s).
🧭 Embedded 23 document overview(s) → /…/eval/.eval_indexes/microsoft__harrier-oss-v1-0.6b/acq_plus_docs_ov/overviews.vectors.npz

$ ls eval/.eval_indexes/microsoft__harrier-oss-v1-0.6b/acq_plus_docs_ov/
eval_acq_plus_docs.built.json  eval_acq_plus_docs.lance  overviews.jsonl  overviews.vectors.npz
```

`index_store/overviews/` was untouched (its three files all predate this wave).

### 3.2 The arms

```
for m in off boost restrict; do
  .venv/bin/python eval/run_eval.py --corpus acq+docs --overviews on --retry off \
    --crossref-hop off --overview-prefilter $m \
    --json-out eval/results/phase4_w3_43_acqdocs_ov_$m.json
  .venv/bin/python eval/run_eval.py --corpus mixed --overviews on --retry off \
    --crossref-hop off --overview-prefilter $m \
    --json-out eval/results/phase4_w3_43_mixed_ov_$m.json
done
```

`top_documents = 5` (the profile default), k = 20, 23 overviews on `acq+docs`,
15 on `mixed`. The prefilter loaded and fired on every query — verbatim, from
the `mixed` run log:

```
🧭 Overview prefilter: 15 document overview(s) loaded from /…/mixed_ov/overviews.vectors.npz.
🧭 Overview prefilter selected 5 document(s): atlas7_service_manual.pdf, design_rationale.md, quick_start.md, northwind_leave_policy.pdf, prompt_inventory.md
🧭 Overview prefilter selected 5 document(s): atlas7_service_manual.pdf, quick_start.md, architecture_overview.md, prompt_inventory.md, northwind_leave_policy.pdf
…
$ grep -c "Overview prefilter selected" <log>
72
```

### 3.3 Results

| corpus | arm | slice | n | R@5 | R@10 | R@20 | nDCG@10 (1st) | R@5 (fin) | R@10 (fin) | R@20 (fin) | nDCG@10 (fin) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| acq+docs | off | all | 48 | 0.854 | 0.896 | 0.958 | **0.7194** | 0.854 | 0.896 | 0.958 | 0.7194 |
| acq+docs | **boost** | all | 48 | **0.812** | **0.917** | 0.958 | **0.7017** | 0.812 | 0.917 | 0.958 | 0.7017 |
| acq+docs | **restrict** | all | 48 | **0.812** | **0.854** | **0.896** | **0.6951** | 0.812 | 0.854 | 0.896 | 0.6951 |
| acq+docs | off | `requires_crossref=true` | 11 | 1.000 | 1.000 | 1.000 | 0.7477 | 1.000 | 1.000 | 1.000 | 0.7477 |
| acq+docs | boost | `requires_crossref=true` | 11 | 1.000 | 1.000 | 1.000 | **0.7261** | 1.000 | 1.000 | 1.000 | 0.7261 |
| acq+docs | restrict | `requires_crossref=true` | 11 | 1.000 | 1.000 | 1.000 | 0.7477 | 1.000 | 1.000 | 1.000 | 0.7477 |
| acq+docs | off | control (`=false`) | 13 | 0.769 | 0.769 | 0.846 | 0.7731 | 0.769 | 0.769 | 0.846 | 0.7731 |
| acq+docs | boost | control (`=false`) | 13 | **0.846** | **0.846** | 0.846 | **0.8790** | 0.846 | 0.846 | 0.846 | 0.8790 |
| acq+docs | restrict | control (`=false`) | 13 | **0.846** | **0.846** | **0.923** | **0.8480** | 0.846 | 0.846 | 0.923 | 0.8480 |
| acq+docs | off | **multi_document=true** | 4 | 0.750 | 0.750 | 0.750 | 0.7803 | 0.750 | 0.750 | 0.750 | 0.7803 |
| acq+docs | **boost** | **multi_document=true** | 4 | 0.750 | 0.750 | 0.750 | **0.8926** | 0.750 | 0.750 | 0.750 | 0.8926 |
| acq+docs | **restrict** | **multi_document=true** | 4 | 0.750 | 0.750 | **1.000** | **0.7394** | 0.750 | 0.750 | 1.000 | 0.7394 |
| mixed | off | all | 72 | 0.944 | 0.972 | 1.000 | **0.8873** | 0.944 | 0.972 | 1.000 | 0.8873 |
| mixed | **boost** | all | 72 | **0.889** | 0.972 | 1.000 | **0.8662** | 0.889 | 0.972 | 1.000 | 0.8662 |
| mixed | **restrict** | all | 72 | **0.889** | **0.931** | **0.944** | **0.8740** | 0.889 | 0.931 | 0.944 | 0.8740 |

`mixed` has no `multi_document` rows (the key lives only on the `acq` gold set),
so that slice is reported for `acq+docs` only. `multi_document` is a **top-level
gold-row key, not a `dimensions` key**, so `run_eval.py`'s `by_dimension` does
not surface it; the four rows above were sliced by joining the results JSON back
to `eval/goldset/acquisition.jsonl`.

The `final == first_stage` invariant passed on all six arms (hop off, rerank
off), so every 4.3 number above is a pure first-stage effect.

### 3.4 The harm check — `restrict` hides the answer document

`recall@20` is where a document the prefilter excluded shows up as a total miss.
Per-query transitions against the `off` arm:

```
acq+docs boost   : lost@20 []                                          gained@20 []
acq+docs restrict: lost@20 [docs_d04, docs_d08, docs_d12, docs_d13]    gained@20 [acq_q09]
mixed    boost   : lost@20 []                                          gained@20 []
mixed    restrict: lost@20 [docs_d12, docs_d13, docs_d20, docs_d24]    gained@20 []
```

**`restrict` costs four queries their answer entirely on each corpus** — the
answer-bearing chunk is not merely demoted, it is unreachable. `boost` loses
nothing at @20 on either corpus, exactly as designed (it reorders, it never
drops).

A worked example, `docs_d12` *"How do I change the embedding model?"*. The gold
string `"Changing the embedding model requires re-indexing"` lives in
`deployment_guide.md`, `retrieval_pipeline.md` and `system_overview.md`. The
prefilter's top-5 for that query was:

```
🧭 Overview prefilter selected 5 document(s): quick_start.md, indexing_pipeline.md,
   installation_guide.md, triage_system.md, architecture_overview.md
```

None of the three documents that contain the answer. Under `boost` those
documents merely get no second RRF leg and the query still scores; under
`restrict` the LanceDB `document_id IN (…)` prefilter removes them and recall
goes 1 → 0. The overviews are not wrong — a question about *changing a model*
does read like a setup question — they are just a 120-token summary standing in
for a 30-chunk document.

At @5 both modes cost the same four documentation queries and rescue two
(`acq+docs`: lost `docs_d04, docs_d08, docs_d10, docs_d12`, gained `acq_q04,
docs_d17`; `mixed`: lost `docs_d10, docs_d12, docs_d20, docs_d24`, gained none).

### 3.5 Reading the 4.3 result

`boost` is **strongly corpus-dependent, and the split is legible**:

* On `acq+docs` it is a real gain where the corpus is *heterogeneous* — the
  `acq` control slice goes 0.769 → 0.846 recall@10 and **0.7731 → 0.8790
  nDCG@10 (+0.106)**, and the `multi_document` slice **0.7803 → 0.8926
  (+0.112)**. When ten M&A PDFs sit in a table with thirteen localGPT
  documentation files, a document-level prior is genuinely informative.
* On `mixed` — atlas7 + hr + docs, where twelve of the fifteen documents are
  localGPT documentation whose overviews all say roughly "a technical document
  about the localGPT RAG system" — it is a **loss**: recall@5 0.944 → 0.889 and
  nDCG@10 0.8873 → 0.8662. The prior is uninformative and the RRF second leg is
  just noise promoting the wrong documentation file.
* On the whole `acq+docs` set the two effects cancel into a small net loss
  (nDCG@10 0.7194 → 0.7017, recall@5 0.854 → 0.812, recall@10 0.896 → 0.917).

`restrict` is worse than `boost` on every aggregate on both corpora **and** it is
the only arm that destroys recall@20. Its one apparent win — `multi_document`
recall@20 0.750 → 1.000 (n=4, i.e. one query) — is not worth the four queries it
kills on the same run.

The `requires_crossref` slice is untouched by `restrict` (0.7477 → 0.7477) and
mildly hurt by `boost` (0.7477 → 0.7261). 4.3 was never aimed at that slice.

---

## 4. Caveats, stated plainly

* **n = 11** on the `requires_crossref` slice and **n = 4** on
  `multi_document`. One query is 0.09 and 0.25 of those figures respectively.
  The `multi_document` "0.750 → 1.000" in §3.3 is literally one query changing.
  Neither slice can support a fine-grained call.
* **`acq` is 13 chunks.** At the shipped `k = 20` no candidate-selection change
  can move a metric on it, which is why the 4.2 evidence lives at k=3 and k=5 —
  values the product does not use. The k=20 rows are the ones that describe
  shipped behaviour, and there the hop is inert.
* **Retrieval metrics only, no judge.** Nothing here says whether an answer
  improved. A hop that puts the right document into the context without changing
  its rank is invisible to this harness — the "which document gets cited" gap
  `BASELINE.md` names. For 4.2 in particular, the *product* question is whether
  the hopped chunk gets cited, and it is unanswered.
* **The `requires_crossref` slice is at recall ceiling before either mechanism
  runs.** This is the deepest limitation: the gold set was built to *expose* the
  cross-reference failure and does not reproduce it (BASELINE.md § "The crossref
  slice is **not** weak" gives three reasons). A mechanism that adds candidates
  cannot be shown to help queries that already retrieve their answer. **4.2
  remains under-tested rather than disproven.**
* **Overview text is LLM-generated and is not reproducible across rebuilds.**
  Within this wave all three arms of each 4.3 comparison read the *same* sidecar,
  so the comparison is exact; a future rebuild will produce different overviews
  and could move the 4.3 numbers without any code change. The `_ov` index
  directories should be preserved, not regenerated, if these numbers are to be
  compared against.
* **`acq+docs` full-set figures are lower than `BASELINE.md`'s Phase 4 baseline
  row** (0.854 / 0.896 / 0.958 vs 0.917 / 0.958 / 1.000). That is `--retry off`
  versus the baseline row's `--retry profile`, not a regression — same caveat as
  `phase4-eval-final-metric.md` §5.
* **No latency claim.** A concurrent agent shared the Ollama instance.

---

## 5. PROPOSED calls — for the gate, which decides

### 4.2 cross-reference hop — **PROPOSE: HOLD** (do not adopt, do not reject)

| evidence | |
|---|---|
| Resolver fix works | 0 → 34 resolved refs on `acq`, 9/10 documents linked, no self-edges (§1.2) |
| Hop fires now | 0 → 24/24 queries at k=3/k=5 on `acq`, 33/48 on `acq+docs` (§2.1) |
| At the shipped k=20 | inert on `acq` (0 hops — corpus smaller than the candidate budget) and null on `acq+docs` (14 hops, every metric bit-identical, `hit_expected_source = 0`) (§2.1) |
| On the slice it exists for | **0/11 hops hit a gold source document, at every k, on both corpora; not one metric moved** (§2.3) |
| Where it does gain | only the `requires_crossref=false` control slice, at k=3/k=5 (§2.3) |
| Budget-matched | beats simply raising `k` in **1 of 4** cells; ties or loses in the other three (§2.4) |
| Harm | none measured anywhere — it only appends, and recall never fell (§2.1) |
| Target selection | 21/24 hops on `acq` go to one of two hub documents, because targets are ordered by text position and `max_hops=1` (§2.5) |

One-line justification: **the mechanism is now demonstrably alive and
demonstrably harmless, but it produced no gain on the slice it was built for and
does not beat raising `k` at equal context budget — so there is no evidence for
turning it on, and no evidence for tearing it out.** Keep
`indexing.extract_crossrefs` on (free, metadata-only, verified inert on the
chunk index) and keep `retrieval.crossref_hop.enabled = False`. What would
settle it: a judged end-to-end arm on the 11 `requires_crossref` rows, hop on vs
off, measuring *which document gets cited* — the retrieval metric is at ceiling
and cannot answer it. Secondarily, a stricter reference-only gold set whose
queries name the pointer and nothing about the target's content.

### 4.3 overview prefilter, `boost` mode — **PROPOSE: HOLD**

| evidence | |
|---|---|
| Now measurable | yes, via `--overviews on` (§3.1); sidecar produced, chunk index bit-identical |
| Heterogeneous corpus (`acq` slice of `acq+docs`) | control nDCG@10 **0.7731 → 0.8790 (+0.106)**, recall@10 0.769 → 0.846; `multi_document` nDCG@10 **0.7803 → 0.8926 (+0.112)** |
| Homogeneous corpus (`mixed`) | recall@5 **0.944 → 0.889**, nDCG@10 **0.8873 → 0.8662** — a loss |
| Whole `acq+docs` set | recall@5 0.854 → 0.812, nDCG@10 0.7194 → 0.7017 — a small net loss |
| Harm at @20 | **none** — nothing lost at recall@20 on either corpus |

One-line justification: **`boost` helps exactly where document overviews carry
signal (a genuinely mixed collection) and hurts where they do not (fifteen
documents about the same system), and localGPT cannot know which one a user's
index is — so it should not ship on by default on a two-corpus split.** It is the
best candidate here for a per-index opt-in, and the natural next measurement is
`top_documents` sensitivity (5 of 15 documents is a third of `mixed`; 5 of 23 is
a fifth of `acq+docs`) plus a third, deliberately heterogeneous corpus.

### 4.3 overview prefilter, `restrict` mode — **PROPOSE: REJECT**

| evidence | |
|---|---|
| `acq+docs` | recall@10 0.896 → 0.854, recall@20 **0.958 → 0.896**, nDCG@10 0.7194 → 0.6951 |
| `mixed` | recall@10 0.972 → 0.931, recall@20 **1.000 → 0.944**, nDCG@10 0.8873 → 0.8740 |
| Harm | **4 queries per corpus lose the answer document entirely** (recall@20 1 → 0), with a worked trace for `docs_d12` (§3.4) |
| Any win? | one query's worth of `multi_document` recall@20, n=4 |

One-line justification: **it is worse than `boost` on every aggregate on both
corpora and it is the only arm that makes an answer unreachable — a 120-token
LLM summary is not a safe basis for excluding a document from search.** Keep the
code (it costs nothing switched off and `boost` shares its machinery) and never
default it on.

---

## 6. Files touched

* `eval/.eval_indexes/microsoft__harrier-oss-v1-0.6b/{acq,acq_plus_docs}` —
  deleted and rebuilt (resolver fix).
* `eval/.eval_indexes/microsoft__harrier-oss-v1-0.6b/{acq_plus_docs_ov,mixed_ov}` —
  new, overview-enabled builds for the 4.3 arms. Preserve them if the 4.3
  numbers are to be compared against (§4, overview reproducibility).
* `eval/results/phase4_w3_*.json` (22 runs) and their `.log` files.
* `eval/run_eval.py` — `--overviews {off,on}` and its plumbing only (§3.1).
* `eval/BASELINE.md` — rebuilt-index baselines, dated.
* `eval/decisions/phase4-retrieval-benchmarks.md` — this file.

Nothing under `rag_system/`, `Documentation/`, `src/` or `backend/`.

## 7. Reproducing every number above

```bash
cd /path/to/localGPT

# 1. rebuild (the resolver fix is index-time)
rm -rf eval/.eval_indexes/microsoft__harrier-oss-v1-0.6b/acq \
       eval/.eval_indexes/microsoft__harrier-oss-v1-0.6b/acq_plus_docs
.venv/bin/python eval/run_eval.py --corpus acq --retry off --crossref-hop off \
  --overview-prefilter off \
  --json-out eval/results/phase4_w3_rebuild_acq_hop_off.json
.venv/bin/python eval/run_eval.py --corpus acq+docs --retry off --crossref-hop off \
  --overview-prefilter off \
  --json-out eval/results/phase4_w3_rebuild_acqdocs_hop_off.json

# 2. the 4.2 matrix (k=20 hop-off arms are the two rebuild runs above)
.venv/bin/python eval/run_eval.py --corpus acq --retry off --crossref-hop on \
  --overview-prefilter off \
  --json-out eval/results/phase4_w3_42_acq_k20_hop_on.json
.venv/bin/python eval/run_eval.py --corpus acq+docs --retry off --crossref-hop on \
  --overview-prefilter off \
  --json-out eval/results/phase4_w3_42_acqdocs_k20_hop_on.json
for k in 5 3; do for arm in off on; do
  .venv/bin/python eval/run_eval.py --corpus acq --retry off --crossref-hop $arm \
    --overview-prefilter off --k $k \
    --json-out eval/results/phase4_w3_42_acq_k${k}_hop_${arm}.json
  .venv/bin/python eval/run_eval.py --corpus acq+docs --retry off --crossref-hop $arm \
    --overview-prefilter off --k $k \
    --json-out eval/results/phase4_w3_42_acqdocs_k${k}_hop_${arm}.json
done; done

# 2b. the budget-matched hop-off controls (§2.4)
for k in 4 6; do
  .venv/bin/python eval/run_eval.py --corpus acq --retry off --crossref-hop off \
    --overview-prefilter off --k $k \
    --json-out eval/results/phase4_w3_42_acq_k${k}_hop_off.json
  .venv/bin/python eval/run_eval.py --corpus acq+docs --retry off --crossref-hop off \
    --overview-prefilter off --k $k \
    --json-out eval/results/phase4_w3_42_acqdocs_k${k}_hop_off.json
done

# 3. the 4.3 arms (first run of each corpus builds the _ov index: 23 / 15 LLM calls)
for m in off boost restrict; do
  .venv/bin/python eval/run_eval.py --corpus acq+docs --overviews on --retry off \
    --crossref-hop off --overview-prefilter $m \
    --json-out eval/results/phase4_w3_43_acqdocs_ov_${m}.json
  .venv/bin/python eval/run_eval.py --corpus mixed --overviews on --retry off \
    --crossref-hop off --overview-prefilter $m \
    --json-out eval/results/phase4_w3_43_mixed_ov_${m}.json
done
```

The crossref verification in §1.2, the budget-matched table in §2.4, the hop
target listing in §2.5 and the `multi_document` / harm slices in §3.3–§3.4 are
derived from those results JSONs and the built LanceDB tables; they are not
produced by `run_eval.py` itself.

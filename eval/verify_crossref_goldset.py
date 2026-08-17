"""Row-level verification for the hand-authored `acquisition` gold set.

`eval/corpora/verify_facts.py` is gate 1 for the *facts* sidecar and
`run_eval.py --coverage-only` is gate 2 for reachability after chunking. Neither
checks the properties that make the cross-reference rows mean anything, so this
does:

  1. every ``expected`` string occurs in the document named in ``expected_sources``
  2. the query does not contain any of its ``expected`` strings verbatim
     (whitespace-normalised, case-insensitive) — no answer leak
  3. every ``expected`` string occurs in **exactly one** of the ten documents, so
     "the answer lives in a different document" is a checkable claim
  4. ``fact_ids`` resolve to ``acquisition.facts.json`` with the same text and source
  5. ``requires_crossref`` is exactly
     ``anchor_doc is not None and any(source != anchor_doc)``
  6. ``multi_document`` is exactly ``len(set(expected_sources)) > 1``
  7. every ``anchor_doc`` names one of the ten documents (``expected_sources``
     entries already fail check 1 when they name no document)

    .venv/bin/python eval/verify_crossref_goldset.py

The tallies it prints are the ones quoted in BASELINE.md § "Phase 4 baseline".
"""

import json
import os
import sys
from collections import Counter

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(EVAL_DIR, "corpora", "acquisition")
SIDECAR = os.path.join(EVAL_DIR, "corpora", "acquisition.facts.json")
GOLD = os.path.join(EVAL_DIR, "goldset", "acquisition.jsonl")


def norm(text: str) -> str:
    return " ".join((text or "").split()).lower()


def document_texts() -> dict:
    import pymupdf

    texts = {}
    for name in sorted(os.listdir(CORPUS_DIR)):
        if not name.endswith(".pdf"):
            continue
        doc = pymupdf.open(os.path.join(CORPUS_DIR, name))
        try:
            texts[name] = norm(" ".join(page.get_text() for page in doc))
        finally:
            doc.close()
    return texts


def main() -> int:
    texts = document_texts()
    with open(SIDECAR, "r", encoding="utf-8") as fh:
        facts = {f["id"]: f for f in json.load(fh)["facts"]}
    with open(GOLD, "r", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]

    problems = []
    tally = Counter()
    n_expected = 0

    for row in rows:
        sources = row["expected_sources"]
        if len(sources) != len(row["expected"]) or len(sources) != len(row["fact_ids"]):
            problems.append(f"{row['id']}: expected / expected_sources / fact_ids length mismatch")
            continue

        for text, source, fact_id in zip(row["expected"], sources, row["fact_ids"]):
            n_expected += 1

            if source in texts and norm(text) in texts[source]:
                tally["expected_in_source"] += 1
            else:
                problems.append(f"{row['id']}: {text!r} not found in {source}")

            if norm(text) not in norm(row["query"]):
                tally["no_verbatim_leak"] += 1
            else:
                problems.append(f"{row['id']}: query leaks the expected string {text!r}")

            holders = [d for d, t in texts.items() if norm(text) in t]
            if holders == [source]:
                tally["unique_to_source"] += 1
            else:
                problems.append(f"{row['id']}: {text!r} occurs in {holders}, not only {source}")

            fact = facts.get(fact_id)
            if fact and norm(fact["expected"]) == norm(text) and fact["source"] == source:
                tally["fact_id_resolves"] += 1
            else:
                problems.append(f"{row['id']}: fact id {fact_id!r} does not match the sidecar")

        anchor = row["anchor_doc"]
        if anchor is not None and anchor not in texts:
            problems.append(f"{row['id']}: anchor_doc {anchor!r} is not a corpus document")
        expected_crossref = anchor is not None and any(s != anchor for s in sources)
        if row["dimensions"]["requires_crossref"] is expected_crossref:
            tally["crossref_flag_consistent"] += 1
        else:
            problems.append(f"{row['id']}: requires_crossref should be {expected_crossref}")

        if row["multi_document"] is (len(set(sources)) > 1):
            tally["multi_document_consistent"] += 1
        else:
            problems.append(f"{row['id']}: multi_document flag is wrong")

    print(f"{GOLD}: {len(rows)} rows, {n_expected} expected strings, "
          f"{len(texts)} source documents")
    for key in ("expected_in_source", "no_verbatim_leak", "unique_to_source",
                "fact_id_resolves"):
        print(f"  {key:<26} {tally[key]}/{n_expected}")
    for key in ("crossref_flag_consistent", "multi_document_consistent"):
        print(f"  {key:<26} {tally[key]}/{len(rows)}")

    crossref = [r for r in rows if r["dimensions"]["requires_crossref"]]
    multi = [r for r in rows if r["multi_document"]]
    print(f"  requires_crossref=true     {len(crossref)}")
    print(f"  multi_document=true        {len(multi)}")
    print(f"  question_type              "
          f"{dict(Counter(r['dimensions']['question_type'] for r in rows))}")
    print(f"  difficulty                 "
          f"{dict(Counter(r['dimensions']['difficulty'] for r in rows))}")

    if problems:
        print(f"\n{len(problems)} PROBLEM(S):")
        for problem in problems:
            print("  " + problem)
        return 1
    print("\nall row-level checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

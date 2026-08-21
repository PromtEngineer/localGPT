"""Row-level verification for the hand-authored `rfc` gold set.

The same gate `eval/verify_crossref_goldset.py` applies to `acquisition`, ported
to a corpus of 23 real IETF RFCs. It checks the properties that make the
cross-reference rows mean anything, on the actual downloaded text:

  1. every ``expected`` string occurs in the document named in ``expected_sources``
  2. the query does not contain any of its ``expected`` strings verbatim
     (whitespace-normalised, case-insensitive) — no answer leak
  3. every ``expected`` string occurs in **exactly one** of the 23 documents, so
     "the answer lives in a different document" is a checkable claim.
     ``UNIQUENESS_EXEMPT`` below records the strings where that is impossible in
     an RFC corpus, with the reason; those are counted separately, never silently.
  4. ``fact_ids`` resolve to ``eval/corpora/rfc/rfc.facts.json`` with the same
     text and source
  5. ``requires_crossref`` is exactly
     ``anchor_doc is not None and any(source != anchor_doc)``
  6. ``multi_document`` is exactly ``len(set(expected_sources)) > 1``
  7. every ``anchor_doc`` and ``expected_sources`` entry names a file that exists

Comparison is on whitespace-normalised, lowercased text, because RFCs are
hard-wrapped at 72 columns: a sentence-length anchor necessarily spans a line
break. ``eval/run_eval.py`` scores chunk relevance with exactly the same
normalisation (``run_eval.norm``), so a string that passes here is a string the
metric can match.

    .venv/bin/python eval/verify_rfc_goldset.py
"""

import json
import os
import sys
from collections import Counter

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(EVAL_DIR, "corpora", "rfc")
SIDECAR = os.path.join(CORPUS_DIR, "rfc.facts.json")
GOLD = os.path.join(EVAL_DIR, "goldset", "rfc.jsonl")

# Strings that cannot be unique to one document in this corpus, with the reason.
# Nothing is exempted for convenience: each entry is a phrase the RFC series
# repeats by construction. An exempt string is still required to be present in
# its named source (check 1) — only check 3 is relaxed, and the count is
# reported separately so the tally can never hide behind a total.
UNIQUENESS_EXEMPT = {
    # (none needed today — kept as the documented mechanism, since RFC
    # boilerplate such as the BCP 14 sentence appears in all 23 documents and any
    # future row anchored on one would have to be recorded here.)
}


def norm(text: str) -> str:
    return " ".join((text or "").split()).lower()


def document_texts() -> dict:
    texts = {}
    for name in sorted(os.listdir(CORPUS_DIR)):
        if not name.endswith(".txt"):
            continue
        with open(os.path.join(CORPUS_DIR, name), "r", encoding="utf-8",
                  errors="replace") as fh:
            texts[name] = norm(fh.read())
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
            elif text in UNIQUENESS_EXEMPT:
                tally["uniqueness_exempt"] += 1
            else:
                problems.append(
                    f"{row['id']}: {text!r} occurs in {holders}, not only {source} "
                    f"(add it to UNIQUENESS_EXEMPT with a reason if that is unavoidable)")

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

    unused = sorted(set(facts) - {f for r in rows for f in r["fact_ids"]})

    print(f"{GOLD}: {len(rows)} rows, {n_expected} expected strings, "
          f"{len(texts)} source documents")
    for key in ("expected_in_source", "no_verbatim_leak", "unique_to_source",
                "fact_id_resolves"):
        print(f"  {key:<26} {tally[key]}/{n_expected}")
    if tally["uniqueness_exempt"]:
        print(f"  {'uniqueness_exempt':<26} {tally['uniqueness_exempt']} "
              f"(recorded in UNIQUENESS_EXEMPT, not counted as unique)")
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
    print(f"  documents referenced       "
          f"{len({s for r in rows for s in r['expected_sources']})}/{len(texts)}")
    if unused:
        print(f"  sidecar facts unused by any row: {len(unused)}")

    if problems:
        print(f"\n{len(problems)} PROBLEM(S):")
        for problem in problems:
            print("  " + problem)
        return 1
    print("\nall row-level checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Apply the human verification pass to the raw generated queries.

Every one of the 72 generated (query, anchor) pairs was read against its source
document by hand. The outcome of that pass is recorded here, per row, so the
committed gold set is auditable rather than "trust me":

  accepted   the model's question was answerable from the anchor, unambiguous,
             and did not simply restate the answer — used verbatim.
  rescued    the model ignored the JSON key and returned {"question": ...} or a
             truncated string; the question was hand-written from the anchor
             (the raw model output is kept in the row for audit).
  rewritten  the question was wrong, vague, or leaked the whole expected string
             into the query (which would hand the lexical leg a free win).
  discarded  not answerable from the source; dropped entirely.

    .venv/bin/python eval/finalize_goldset.py

Reads eval/goldset/_generated/<corpus>.raw.jsonl, writes eval/goldset/<corpus>.jsonl.
"""

import json
import os
import sys

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(EVAL_DIR, "goldset", "_generated")
OUT_DIR = os.path.join(EVAL_DIR, "goldset")
CORPORA_DIR = os.path.join(EVAL_DIR, "corpora")

SIDECARS = {
    "atlas7": "atlas7_service_manual.facts.json",
    "hr": "northwind_leave_policy.facts.json",
    "docs": "repo_docs.facts.json",
}

# id -> (verdict, replacement query or None, replacement fact_ids or None, reason)
EDITS = {
    # ---- Atlas-7 -----------------------------------------------------------
    "atlas7_a02": ("rewritten", "What pressure is the steam boiler held at?", None,
                   "model wrapped its output in literal quote characters"),
    "atlas7_a03": ("rescued", "What temperature does the PID hold the brew water at?", None,
                   "model returned {\"question\": ...}; question taken from that payload"),
    "atlas7_a05": ("rewritten", "What is the water hardness threshold for this machine?", None,
                   "'What is the hardness threshold?' had no anchor to the document at all"),
    "atlas7_a08": ("rewritten", "Who makes the Atlas-7, and where are they based?", None,
                   "generated question asked who manufactures the manufacturer"),
    "atlas7_a11": ("rescued", "How tightly is the brew water temperature controlled?", None,
                   "model returned {\"question\": ...}"),
    "atlas7_a16": ("rewritten", "The machine is not registering any water flow at all - what should I check?", None,
                   "'stops delivering water' was ambiguous between the E42 and E57 procedures"),
    "atlas7_a19": ("rewritten", "Which needs doing more often on this machine: descaling, or replacing the group head gasket?", None,
                   "generated comparative leaked the descaling interval into the question"),
    "atlas7_a20": ("rescued", "Which error code points to a failed temperature sensor, and which one points to excess steam pressure?", None,
                   "model returned a truncated {\"question\": ...} payload"),
    "atlas7_a21": ("rewritten", "What kind of descaling product will void the warranty?", None,
                   "generated question contained the expected answer text verbatim"),
    "atlas7_a22": ("rewritten", "Will a warranty claim be accepted without a serial number, and where do I find it?",
                   ["atlas_serial_location"],
                   "generated question restated '120 ppm'; re-anchored to the serial-number requirement"),
    "atlas7_a23": ("rewritten", "Is there a limit on how far the brew temperature may drift before it is out of spec?", None,
                   "reworded so it is not a near-duplicate of a11's phrasing"),

    # ---- HR handbook -------------------------------------------------------
    "hr_h08": ("rewritten", "What is the identifier and revision number of the leave policy?", None,
               "generated question quoted 'PPL-204 revision 4', i.e. the expected string"),
    "hr_h09": ("rewritten", "Which department owns this policy, and where is it based?", None,
               "generated question asked who owns the department, inverting the fact"),
    "hr_h12": ("rewritten", "After the initial full-pay period of a long illness ends, what proportion of salary continues and for how long?", None,
               "generated question invented a 'one month / next two months' schedule the policy does not state"),
    "hr_h14": ("rewritten", "How far in advance do I have to file a leave request, and where do I file it?", None,
               "generated question contained 'Kestrel HR portal', part of the expected string"),
    "hr_h15": ("rewritten", "At what point during a sickness absence do I have to produce a doctor's note?", None,
               "generated question contained '4 consecutive working days', the expected string"),
    "hr_h16": ("rewritten", "What do I need to qualify for an extended unpaid break, how much notice must I give, and who signs it off?", None,
               "'a leave away from work' was too vague to be answerable by the sabbatical section specifically"),
    "hr_h18": ("rescued", "How does annual leave entitlement differ between employees below Grade 7 and those at Grade 7 or above?", None,
               "model returned a truncated {\"question\": ...} payload"),
    "hr_h19": ("rewritten", "How does sick pay in the first three months of an absence compare with the months that follow?", None,
               "generated question referenced 'after 20 weeks', which is past the end of the stated schedule"),

    # ---- Repo documentation ------------------------------------------------
    "docs_d07": ("rewritten", "How many extra vectors does turning on late chunking write?", None,
                 "generated question compared against 'early chunking', a term the docs never use"),
    "docs_d08": ("rewritten", "What does the enricher do when the model returns an almost-empty summary?", None,
                 "generated question restated 'shorter than 5 characters'"),
    "docs_d09": ("rewritten", "How large is the per-retriever cache that stores previously embedded queries?", None,
                 "generated question invented 'for each search engine'"),
    "docs_d10": ("rewritten", "How many model calls does knowledge-graph extraction spend on each chunk?", None,
                 "'each processed unit of information' was too vague to be answerable"),
    "docs_d11": ("rewritten", "What confidence value is interpreted as a failed parse rather than a real score?", None,
                 "tightened so the answer is the value, not the behaviour"),
    "docs_d14": ("rescued", "Why do documents indexed from the command line end up with a different chunk size than ones indexed through the HTTP API?", None,
                 "model returned a truncated, off-topic {\"question\": ...} payload"),
    "docs_d15": ("rewritten", "How are plain-text uploads processed differently from PDFs and Word files?", None,
                 "generated question presupposed a 'standard processing pipeline' the docs do not name"),
    "docs_d16": ("rewritten", "Does this project build an approximate-nearest-neighbour index, and what does that mean for how a vector query executes?", None,
                 "generated question contained both expected strings"),
    "docs_d17": ("rewritten", "Which model handles routing and verification, and how much extra work does verification add per query?", None,
                 "generated question was about swapping models, which the anchors do not cover"),
    "docs_d18": ("rewritten", "Does the indexer overlap chunks or parallelise the work across workers?", None,
                 "generated question asked about 'distributed systems features', not answerable from the anchors"),
    "docs_d19": ("rewritten", "Can I tune how much the keyword leg counts versus the vector leg when the two are combined?", None,
                 "generated question contained the expected string 'weighted linear blend'"),
    "docs_d20": ("rewritten", "Does the generated answer contain markers pointing at the passage each claim came from?", None,
                 "generated question contained the expected string 'inline citation marker'"),
    "docs_d21": ("rewritten", "Does the agent do any pattern matching on the query before it asks a model to route it?", None,
                 "generated question contained the expected string 'regex or keyword stage'"),
    "docs_d24": ("rescued", "Why does a vector-dimension mismatch raise an error instead of just rebuilding the table?", None,
                 "model returned {\"question\": ...}"),
}


def load_facts(corpus: str) -> dict:
    with open(os.path.join(CORPORA_DIR, SIDECARS[corpus]), "r", encoding="utf-8") as fh:
        return {f["id"]: f for f in json.load(fh)["facts"]}


def main() -> int:
    tally = {"accepted": 0, "rescued": 0, "rewritten": 0, "discarded": 0}
    for corpus in sorted(SIDECARS):
        facts = load_facts(corpus)
        raw_path = os.path.join(RAW_DIR, f"{corpus}.raw.jsonl")
        if not os.path.exists(raw_path):
            print(f"missing {raw_path}; run eval/build_goldset.py first")
            return 1

        out_rows = []
        with open(raw_path, "r", encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                verdict, new_query, new_fact_ids, reason = EDITS.get(
                    row["id"], ("accepted", None, None, None))
                tally[verdict] += 1
                if verdict == "discarded":
                    continue

                fact_ids = new_fact_ids or row["fact_ids"]
                anchors = [facts[fid] for fid in fact_ids]
                out_rows.append({
                    "id": row["id"],
                    "corpus": corpus,
                    "query": new_query or row["query"],
                    "expected": [a["expected"] for a in anchors],
                    "match": row["match"],
                    "fact_ids": fact_ids,
                    "answer": " ".join(a["summary"] for a in anchors),
                    "dimensions": {**row["dimensions"], "topic": anchors[0]["topic"]},
                    "verification": {
                        "verdict": verdict,
                        "reason": reason,
                        "generated_query": row["query"] or row.get("raw_response"),
                        "generator_model": row["generator_model"],
                    },
                })

        out_path = os.path.join(OUT_DIR, f"{corpus}.jsonl")
        with open(out_path, "w", encoding="utf-8") as fh:
            for row in sorted(out_rows, key=lambda r: r["id"]):
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{corpus}: wrote {len(out_rows)} rows -> {out_path}")

    total = sum(tally.values())
    print(f"\nverification pass over {total} generated pairs: " +
          ", ".join(f"{k}={v}" for k, v in tally.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())

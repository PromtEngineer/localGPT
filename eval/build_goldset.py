"""Reverse-generate the gold query set from planted facts, one query per dimension tuple.

Structured, not freeform: the dimension table below is hand-authored
(anchor fact(s) x question-type x difficulty), and the *only* thing the LLM does
is phrase a natural-language question for a tuple whose answer is already fixed.
That keeps the gold label (the answer-bearing substring) independent of the model
that wrote the question.

    .venv/bin/python eval/build_goldset.py --corpus all --out eval/goldset/_generated

Writes ``<out>/<corpus>.raw.jsonl``. The *committed* gold set lives at
``eval/goldset/<corpus>.jsonl`` and is the human-verified edit of that raw file —
see eval/README.md. This script is a one-shot generator, not part of the eval run.

Gold relevance is defined by answer-bearing text, never by chunk id, so the gold
set survives re-chunking, a different chunk size, or an embedder swap.
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag_system.utils.ollama_client import OllamaClient  # noqa: E402

CORPORA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpora")

QUESTION_TYPES = ("factoid", "procedural", "comparative", "negative")
DIFFICULTIES = ("easy", "hard")

TYPE_GUIDANCE = {
    "factoid": "Ask for one specific value, name, part number or figure.",
    "procedural": "Ask how to do something, or what steps to take. Phrase it the way an operator would.",
    "comparative": "Ask a question that can only be answered by using BOTH source snippets together — a difference, a contrast, or a combined total.",
    "negative": (
        "Ask about a restriction, exclusion, limit, threshold or condition that "
        "invalidates something. The answer must still be stated in the snippet — "
        "this is a negatively-framed question about the document, NOT a question "
        "the document cannot answer."
    ),
}

DIFFICULTY_GUIDANCE = {
    "easy": "Use the document's own vocabulary. A keyword search would plausibly find it.",
    "hard": "Paraphrase. Avoid reusing the snippet's distinctive nouns and numbers, so that only a semantic match finds it.",
}

PROMPT = """You write evaluation questions for a document-retrieval benchmark.

SOURCE SNIPPET(S), verbatim from {doc_label}:
{snippets}

Write ONE natural question that a real user of this document would ask, whose
answer is contained in the snippet(s) above.

Question type: {qtype}. {type_guidance}
Difficulty: {difficulty}. {difficulty_guidance}

Hard rules:
- The question must be answerable using ONLY the snippet(s) above.
- Do NOT include the answer in the question.
- One sentence. No preamble, no quotes around it.
- Do not mention "the snippet", "the document above" or "the text".

Respond with JSON only: {{"query": "<the question>"}}
"""

# --- Dimension table -------------------------------------------------------
# (tuple_id, question_type, difficulty, [anchor fact ids], match)
# match "any": retrieving one anchor-bearing chunk counts as a hit.
# match "all": every anchor must appear in the retrieved set (comparatives).
DIMENSIONS = {
    "atlas7": [
        ("a01", "factoid", "easy", ["atlas_brew_pressure"], "any"),
        ("a02", "factoid", "easy", ["atlas_steam_pressure"], "any"),
        ("a03", "factoid", "easy", ["atlas_brew_temperature"], "any"),
        ("a04", "factoid", "easy", ["atlas_pump_rating"], "any"),
        ("a05", "factoid", "easy", ["atlas_water_hardness"], "any"),
        ("a06", "factoid", "easy", ["atlas_gasket_part"], "any"),
        ("a07", "factoid", "easy", ["atlas_warranty_length"], "any"),
        ("a08", "factoid", "easy", ["atlas_manufacturer"], "any"),
        ("a09", "factoid", "easy", ["atlas_model_revision"], "any"),
        ("a10", "factoid", "easy", ["atlas_serial_location"], "any"),
        ("a11", "factoid", "hard", ["atlas_temperature_tolerance"], "any"),
        ("a12", "factoid", "hard", ["atlas_e11_part"], "any"),
        ("a13", "procedural", "easy", ["atlas_e42_prime", "atlas_e42_procedure"], "all"),
        ("a14", "procedural", "easy", ["atlas_backflush"], "any"),
        ("a15", "procedural", "hard", ["atlas_descale_interval", "atlas_water_hardness"], "all"),
        ("a16", "procedural", "hard", ["atlas_e57"], "any"),
        ("a17", "procedural", "hard", ["atlas_e23"], "any"),
        ("a18", "comparative", "easy", ["atlas_brew_pressure", "atlas_steam_pressure"], "all"),
        ("a19", "comparative", "hard", ["atlas_descale_interval", "atlas_gasket_interval"], "all"),
        ("a20", "comparative", "hard", ["atlas_e11", "atlas_e23"], "all"),
        ("a21", "negative", "easy", ["atlas_warranty_void"], "any"),
        ("a22", "negative", "easy", ["atlas_water_hardness"], "any"),
        ("a23", "negative", "hard", ["atlas_temperature_tolerance"], "any"),
        ("a24", "negative", "hard", ["atlas_e23"], "any"),
    ],
    "hr": [
        ("h01", "factoid", "easy", ["hr_annual_below_g7"], "any"),
        ("h02", "factoid", "easy", ["hr_annual_g7_plus"], "any"),
        ("h03", "factoid", "easy", ["hr_sick_full_pay"], "any"),
        ("h04", "factoid", "easy", ["hr_parental_length"], "any"),
        ("h05", "factoid", "easy", ["hr_bereavement"], "any"),
        ("h06", "factoid", "easy", ["hr_jury_duty"], "any"),
        ("h07", "factoid", "easy", ["hr_public_holiday_count"], "any"),
        ("h08", "factoid", "easy", ["hr_policy_id"], "any"),
        ("h09", "factoid", "easy", ["hr_policy_owner"], "any"),
        ("h10", "factoid", "easy", ["hr_sabbatical_length"], "any"),
        ("h11", "factoid", "hard", ["hr_carryover_expiry"], "any"),
        ("h12", "factoid", "hard", ["hr_sick_reduced_pay"], "any"),
        ("h13", "factoid", "hard", ["hr_public_holiday_pay"], "any"),
        ("h14", "procedural", "easy", ["hr_request_notice"], "any"),
        ("h15", "procedural", "easy", ["hr_medical_certificate"], "any"),
        ("h16", "procedural", "hard", ["hr_sabbatical_eligibility", "hr_sabbatical_notice", "hr_sabbatical_approver"], "all"),
        ("h17", "procedural", "hard", ["hr_director_approval"], "any"),
        ("h18", "comparative", "easy", ["hr_annual_below_g7", "hr_annual_g7_plus"], "all"),
        ("h19", "comparative", "hard", ["hr_sick_full_pay", "hr_sick_reduced_pay"], "all"),
        ("h20", "comparative", "hard", ["hr_carryover_cap", "hr_carryover_expiry"], "all"),
        ("h21", "negative", "easy", ["hr_contractors_excluded"], "any"),
        ("h22", "negative", "easy", ["hr_resignation_payout"], "any"),
        ("h23", "negative", "hard", ["hr_parental_deadline"], "any"),
        ("h24", "negative", "hard", ["hr_parental_blocks"], "any"),
    ],
    "docs": [
        ("d01", "factoid", "easy", ["docs_provence_model"], "any"),
        ("d02", "factoid", "easy", ["docs_verifier_context_clamp"], "any"),
        ("d03", "factoid", "easy", ["docs_triage_overview_cap"], "any"),
        ("d04", "factoid", "easy", ["docs_embedding_dimensions"], "any"),
        ("d05", "factoid", "easy", ["docs_overview_truncation"], "any"),
        ("d06", "factoid", "easy", ["docs_direct_answer_length"], "any"),
        ("d07", "factoid", "easy", ["docs_latechunk_cost"], "any"),
        ("d08", "factoid", "easy", ["docs_enrichment_short_summary"], "any"),
        ("d09", "factoid", "hard", ["docs_query_embed_cache"], "any"),
        ("d10", "factoid", "hard", ["docs_graph_two_llm_calls"], "any"),
        ("d11", "factoid", "hard", ["docs_verifier_zero_score"], "any"),
        ("d12", "procedural", "easy", ["docs_reindex_required"], "any"),
        ("d13", "procedural", "easy", ["docs_pruning_off_by_default"], "any"),
        ("d14", "procedural", "hard", ["docs_chunk_size_layering"], "any"),
        ("d15", "procedural", "hard", ["docs_txt_bypasses_docling"], "any"),
        ("d16", "comparative", "easy", ["docs_no_ann_index", "docs_brute_force_vector"], "all"),
        ("d17", "comparative", "hard", ["docs_verifier_cost", "docs_triage_utility_model"], "all"),
        ("d18", "comparative", "hard", ["docs_no_overlap_knob", "docs_indexing_sequential"], "all"),
        ("d19", "negative", "easy", ["docs_no_weighted_blend"], "any"),
        ("d20", "negative", "easy", ["docs_no_citation_markers"], "any"),
        ("d21", "negative", "easy", ["docs_triage_no_regex"], "any"),
        ("d22", "negative", "hard", ["docs_triage_no_switch"], "any"),
        ("d23", "negative", "hard", ["docs_expansion_filtered_out"], "any"),
        ("d24", "negative", "hard", ["docs_dimension_mismatch_raises"], "any"),
    ],
}

SIDECARS = {
    "atlas7": "atlas7_service_manual.facts.json",
    "hr": "northwind_leave_policy.facts.json",
    "docs": "repo_docs.facts.json",
}

DOC_LABELS = {
    "atlas7": "the Atlas-7 espresso machine service manual",
    "hr": "the Northwind Robotics leave and absence policy handbook",
    "docs": "the localGPT project's developer documentation",
}


def load_facts(corpus: str) -> dict:
    with open(os.path.join(CORPORA_DIR, SIDECARS[corpus]), "r", encoding="utf-8") as fh:
        return {f["id"]: f for f in json.load(fh)["facts"]}


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def generate(corpus: str, client: OllamaClient, model: str) -> list:
    facts = load_facts(corpus)
    rows = []
    for tuple_id, qtype, difficulty, fact_ids, match in DIMENSIONS[corpus]:
        anchors = [facts[fid] for fid in fact_ids]
        snippets = "\n".join(f'{i + 1}. "{a["expected"]}" — {a["summary"]}' for i, a in enumerate(anchors))
        prompt = PROMPT.format(
            doc_label=DOC_LABELS[corpus],
            snippets=snippets,
            qtype=qtype,
            type_guidance=TYPE_GUIDANCE[qtype],
            difficulty=difficulty,
            difficulty_guidance=DIFFICULTY_GUIDANCE[difficulty],
        )
        resp = client.generate_completion(model=model, prompt=prompt, format="json")
        raw = strip_think(resp.get("response", "") or "")
        try:
            query = json.loads(raw).get("query", "").strip()
        except json.JSONDecodeError:
            query = ""
        rows.append({
            "id": f"{corpus}_{tuple_id}",
            "corpus": corpus,
            "query": query,
            "expected": [a["expected"] for a in anchors],
            "match": match,
            "fact_ids": fact_ids,
            "dimensions": {"topic": anchors[0]["topic"], "question_type": qtype, "difficulty": difficulty},
            "generator_model": model,
            "raw_response": raw if not query else None,
        })
        status = "ok " if query else "FAIL"
        print(f"[{status}] {corpus}_{tuple_id} ({qtype}/{difficulty}): {query or raw[:120]!r}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="all", choices=["all", *sorted(DIMENSIONS)])
    parser.add_argument("--model", default=os.getenv("ENRICHMENT_MODEL", "qwen3.5:4b"))
    parser.add_argument("--host", default=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    parser.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "goldset", "_generated"))
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    client = OllamaClient(host=args.host)
    corpora = sorted(DIMENSIONS) if args.corpus == "all" else [args.corpus]

    for corpus in corpora:
        rows = generate(corpus, client, args.model)
        path = os.path.join(args.out, f"{corpus}.raw.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\nWrote {len(rows)} rows to {path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

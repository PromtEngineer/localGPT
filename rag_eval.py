#!/usr/bin/env python3
"""RAG evaluation harness for LocalGPT.

Methodology follows Anthropic's RAG cookbook: build a golden Q&A set from the
index's own chunks, then measure (a) whether retrieval surfaces the right
document and (b) whether the end-to-end answer contains the reference facts,
judged by the local LLM. Retrieval-quality bugs (broken score columns, dead
fusion, ignored search settings) are invisible to unit tests but show up here
as a measurable drop.

Usage (from the repo root, with the venv python):
  # 1. Generate a golden set from an index's chunks (needs Ollama)
  python rag_eval.py generate --index <index-id> --n 20

  # 2. Score retrieval only — fast, no servers needed
  python rag_eval.py run --index <index-id> --mode retrieval

  # 3. Score end-to-end answers through the live RAG API (slow)
  python rag_eval.py run --index <index-id> --mode e2e
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GENERATION_MODEL = os.getenv("EVAL_MODEL", "qwen3:8b")
RAG_API = os.getenv("RAG_API_URL", "http://127.0.0.1:8001").rstrip("/")
EVAL_DIR = "evals"

QUESTION_PROMPT = """You will receive a passage from a document. Write ONE specific, factual question that this passage clearly answers, and the answer.

Rules:
- The question must be answerable from this passage alone.
- Prefer questions about concrete facts: names, numbers, dates, equipment, recommendations.
- Do not mention "the passage" or "the document" in the question.
- Reply with JSON only: {{"question": "...", "answer": "..."}}

PASSAGE (from {doc_name}):
{passage}
"""

JUDGE_PROMPT = """Compare a candidate answer against a reference answer.

REFERENCE ANSWER: {reference}

CANDIDATE ANSWER: {candidate}

Does the candidate contain the key information of the reference (same facts, numbers, names — wording may differ)? Reply with JSON only: {{"correct": true}} or {{"correct": false}}.
"""


def resolve_index(index_id: str) -> tuple[str, str, str]:
    """Return (full_index_id, vector_table_name, embedding_model)."""
    conn = sqlite3.connect(os.path.join("backend", "chat_data.db"))
    row = conn.execute(
        "SELECT id, vector_table_name, metadata FROM indexes WHERE id LIKE ?",
        (f"{index_id}%",),
    ).fetchone()
    if not row:
        raise SystemExit(f"No index found matching '{index_id}'")
    meta = json.loads(row[2] or "{}")
    model = meta.get("embedding_model") or "Qwen/Qwen3-Embedding-0.6B"
    return row[0], row[1], model


def eval_path(full_index_id: str) -> str:
    os.makedirs(EVAL_DIR, exist_ok=True)
    return os.path.join(EVAL_DIR, f"{full_index_id[:8]}.jsonl")


def _load_chunks(table_name: str):
    import lancedb

    tbl = lancedb.connect("./lancedb").open_table(table_name)
    df = tbl.to_pandas()[["chunk_id", "document_id", "text", "metadata"]]
    # Prefer the original text when chunks were enriched
    def _orig(row):
        try:
            return json.loads(row["metadata"] or "{}").get("original_text") or row["text"]
        except Exception:
            return row["text"]
    df["text"] = df.apply(_orig, axis=1)
    return df[df["text"].str.len() > 300]


def cmd_generate(args):
    from rag_system.utils.ollama_client import OllamaClient

    full_id, table, _ = resolve_index(args.index)
    df = _load_chunks(table)
    if df.empty:
        raise SystemExit("No usable chunks in the table")

    # Round-robin across documents so the set covers the corpus, not one file
    by_doc = [g.sample(frac=1, random_state=42) for _, g in df.groupby("document_id")]
    picked, i = [], 0
    while len(picked) < args.n and any(len(g) > i for g in by_doc):
        for g in by_doc:
            if len(g) > i and len(picked) < args.n:
                picked.append(g.iloc[i])
        i += 1

    client = OllamaClient()
    out = eval_path(full_id)
    written = 0
    with open(out, "w", encoding="utf-8") as f:
        for row in picked:
            prompt = QUESTION_PROMPT.format(
                doc_name=row["document_id"], passage=row["text"][:4000]
            )
            resp = client.generate_completion(
                GENERATION_MODEL, prompt, format="json", enable_thinking=False, timeout=120
            )
            try:
                qa = json.loads(resp.get("response", ""))
                question, answer = qa["question"].strip(), str(qa["answer"]).strip()
            except (json.JSONDecodeError, KeyError, AttributeError):
                print(f"  skipped (bad generation) chunk={row['chunk_id'][:40]}")
                continue
            if not question or not answer:
                continue
            # Self-referential questions can't be retrieved against — the
            # generator was told not to write them, but it sometimes does
            if re.search(r"\b(the|this) (passage|document|text|excerpt)\b", question, re.I):
                print(f"  skipped (self-referential): {question[:70]}")
                continue
            f.write(json.dumps({
                "question": question,
                "reference_answer": answer,
                "expected_doc": row["document_id"],
                "chunk_id": row["chunk_id"],
            }, ensure_ascii=False) + "\n")
            written += 1
            print(f"  [{written}/{args.n}] {question[:80]}")
    print(f"\nWrote {written} golden questions to {out}")


def _load_eval_set(full_id: str):
    path = eval_path(full_id)
    if not os.path.exists(path):
        raise SystemExit(f"No eval set at {path} — run `generate` first")
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def _doc_rank(expected_doc: str, docs: list) -> int | None:
    """1-based rank of the first source doc matching expected_doc."""
    seen, rank = set(), 0
    for d in docs:
        doc_id = d.get("document_id")
        if doc_id in seen:
            continue
        seen.add(doc_id)
        rank += 1
        if doc_id == expected_doc:
            return rank
    return None


def _chunk_hit(expected_chunk: str, docs: list) -> bool:
    """Whether the exact golden chunk made it into the result set.

    Doc-level hits can mask the real failure mode: the right document is
    found via some chunk, while the chunk that actually contains the answer
    never reaches the synthesis context.
    """
    return any(d.get("chunk_id") == expected_chunk for d in docs)


def _summarize(name: str, ranks: list, chunk_hits: list, k: int):
    n = len(ranks)
    hits5 = sum(1 for r in ranks if r is not None and r <= 5)
    hitsk = sum(1 for r in ranks if r is not None)
    mrr = sum(1.0 / r for r in ranks if r is not None) / n if n else 0.0
    chunks = sum(chunk_hits)
    print(f"\n=== {name} (n={n}) ===")
    print(f"doc hit@5:    {hits5}/{n} ({100*hits5/n:.0f}%)")
    print(f"doc hit@{k}:   {hitsk}/{n} ({100*hitsk/n:.0f}%)")
    print(f"MRR (doc):    {mrr:.3f}")
    print(f"chunk hit@{k}: {chunks}/{n} ({100*chunks/n:.0f}%)  ← the chunk that contains the answer")


def cmd_run_retrieval(args, full_id: str, table: str, embedding_model: str):
    from rag_system.indexing.embedders import LanceDBManager
    from rag_system.indexing.representations import select_embedder
    from rag_system.retrieval.retrievers import MultiVectorRetriever

    cases = _load_eval_set(full_id)
    retriever = MultiVectorRetriever(LanceDBManager("./lancedb"), select_embedder(embedding_model))
    ranks, chunk_hits = [], []
    for c in cases:
        docs = retriever.retrieve(c["question"], table_name=table, k=args.k)
        rank = _doc_rank(c["expected_doc"], docs)
        ranks.append(rank)
        chunk_hits.append(_chunk_hit(c["chunk_id"], docs))
        mark = "✓" if rank is not None else "✗"
        cmark = "chunk✓" if chunk_hits[-1] else "chunk✗"
        print(f"  {mark} rank={rank if rank else '-'} {cmark}  {c['question'][:70]}")
    _summarize(f"retrieval — {table}", ranks, chunk_hits, args.k)


def cmd_run_e2e(args, full_id: str, table: str, _model: str):
    import requests

    from rag_system.utils.ollama_client import OllamaClient

    cases = _load_eval_set(full_id)
    judge = OllamaClient()
    ranks, chunk_hits, correct = [], [], 0
    for c in cases:
        resp = requests.post(
            f"{RAG_API}/chat",
            json={
                "query": c["question"], "table_name": table, "force_rag": True,
                "retrieval_k": args.k, "reranker_top_k": args.k,
            },
            timeout=600,
        )
        resp.raise_for_status()
        data = resp.json()
        answer = data.get("answer", "")
        sources = data.get("source_documents", [])
        ranks.append(_doc_rank(c["expected_doc"], sources))
        chunk_hits.append(_chunk_hit(c["chunk_id"], sources))

        verdict = judge.generate_completion(
            GENERATION_MODEL,
            JUDGE_PROMPT.format(reference=c["reference_answer"], candidate=answer[:3000]),
            format="json", enable_thinking=False, timeout=120,
        )
        try:
            ok = bool(json.loads(verdict.get("response", "")).get("correct"))
        except (json.JSONDecodeError, AttributeError):
            ok = False
        correct += ok
        print(f"  {'✓' if ok else '✗'} {c['question'][:75]}")

    _summarize(f"e2e retrieval — {table}", ranks, chunk_hits, args.k)
    n = len(cases)
    print(f"answer accuracy (LLM judge): {correct}/{n} ({100*correct/n:.0f}%)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="generate a golden Q&A set from the index's chunks")
    g.add_argument("--index", required=True, help="index id (prefix ok)")
    g.add_argument("--n", type=int, default=20)

    r = sub.add_parser("run", help="score the index against its golden set")
    r.add_argument("--index", required=True, help="index id (prefix ok)")
    r.add_argument("--mode", choices=["retrieval", "e2e"], default="retrieval")
    r.add_argument("--k", type=int, default=20)

    args = p.parse_args()
    full_id, table, model = resolve_index(args.index)
    print(f"index {full_id[:8]} → table {table} (embedding: {model})")

    if args.cmd == "generate":
        cmd_generate(args)
    elif args.mode == "retrieval":
        cmd_run_retrieval(args, full_id, table, model)
    else:
        cmd_run_e2e(args, full_id, table, model)


if __name__ == "__main__":
    main()

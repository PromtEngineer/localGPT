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
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GENERATION_MODEL = os.getenv("EVAL_MODEL", "qwen3:8b")
BACKEND_API = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
EVAL_DIR = "evals"

QUESTION_PROMPT = """You will receive a passage from a document. Write ONE specific, factual question that this passage clearly answers, and the answer.

Rules:
- The question must be answerable from this passage alone.
- Prefer questions about concrete facts: names, numbers, dates, equipment, recommendations.
- The document is part of a collection of similar reports about DIFFERENT projects/sites. Make the question self-contained: include the project name, mine/site name, company, or document identifier (visible in the passage or the document name) so the question cannot be confused with another report. Questions like "What is the NPV of the project?" are useless; "What is the NPV of the Las Chispas base case?" is good.
- Do not mention "the passage" or "the document" in the question.
- Reply with JSON only: {{"question": "...", "answer": "..."}}

PASSAGE (from {doc_name}):
{passage}
"""

GROUNDEDNESS_PROMPT = """You will see retrieved source snippets and an answer generated from them.

SOURCES:
{sources}

ANSWER:
{answer}

Is every factual claim in the answer supported by the sources (no invented facts)? Reply with JSON only: {{"grounded": true}} or {{"grounded": false}}.
"""

RELEVANCY_PROMPT = """You will see a question and retrieved source snippets.

QUESTION: {question}

SOURCES:
{sources}

Do the sources contain information relevant to answering the question? Reply with JSON only: {{"relevant": true}} or {{"relevant": false}}.
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


def _summarize(name: str, ranks: list, chunk_hits: list, k: int) -> dict:
    """Print and return the metric set (returned dict feeds the CI gate)."""
    n = len(ranks)
    metrics = {"n": n}
    print(f"\n=== {name} (n={n}) ===")
    for kk in (1, 3, 5, 10, k):
        if kk > k:
            continue
        hits = sum(1 for r in ranks if r is not None and r <= kk)
        metrics[f"doc_recall@{kk}"] = hits / n if n else 0.0
        print(f"doc recall@{kk:<2}: {hits}/{n} ({100*hits/n:.0f}%)")
    mrr = sum(1.0 / r for r in ranks if r is not None) / n if n else 0.0
    metrics["mrr"] = mrr
    print(f"MRR (doc):     {mrr:.3f}")
    chunks = sum(chunk_hits)
    metrics[f"chunk_hit@{k}"] = chunks / n if n else 0.0
    metrics["chunk_hit"] = chunks / n if n else 0.0
    print(f"chunk hit@{k}:  {chunks}/{n} ({100*chunks/n:.0f}%)  ← the chunk that contains the answer")
    return metrics


def cmd_run_retrieval(args, full_id: str, table: str, embedding_model: str):
    from rag_system.indexing.embedders import LanceDBManager
    from rag_system.indexing.representations import select_embedder
    from rag_system.retrieval.retrievers import MultiVectorRetriever

    cases = _load_eval_set(full_id)
    retriever = MultiVectorRetriever(LanceDBManager("./lancedb"), select_embedder(embedding_model))
    fusion_override = None
    if args.dense_weight is not None:
        w = max(0.0, min(1.0, args.dense_weight))
        fusion_override = {"bm25_weight": 1.0 - w, "vec_weight": w}
    ranks, chunk_hits = [], []
    for c in cases:
        docs = retriever.retrieve(c["question"], table_name=table, k=args.k, fusion_override=fusion_override)
        rank = _doc_rank(c["expected_doc"], docs)
        ranks.append(rank)
        chunk_hits.append(_chunk_hit(c["chunk_id"], docs))
        mark = "✓" if rank is not None else "✗"
        cmark = "chunk✓" if chunk_hits[-1] else "chunk✗"
        print(f"  {mark} rank={rank if rank else '-'} {cmark}  {c['question'][:70]}")
    metrics = _summarize(f"retrieval — {table}", ranks, chunk_hits, args.k)
    _dump_results(full_id, "retrieval", metrics)


def _dump_results(full_id: str, mode: str, metrics: dict):
    os.makedirs(EVAL_DIR, exist_ok=True)
    path = os.path.join(EVAL_DIR, f"results-{full_id[:8]}-{mode}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"(metrics written to {path})")


def cmd_run_e2e(args, full_id: str, table: str, _model: str):
    import requests

    from rag_system.utils.ollama_client import OllamaClient

    import time as _time

    cases = _load_eval_set(full_id)
    judge = OllamaClient()
    ranks, chunk_hits, answers, latencies, source_sets, citation_ok = [], [], [], [], [], []

    # Phase 1: generate all answers. Judging is a separate pass so Ollama
    # doesn't swap models between every question when --model differs from
    # the judge model.
    for c in cases:
        payload = {
            "query": c["question"], "table_name": table, "force_rag": True,
            "retrieval_k": args.k, "reranker_top_k": args.reranker_top_k or args.k,
        }
        # Sweepable pipeline knobs — only sent when explicitly set
        if args.rerank is not None:
            payload["ai_rerank"] = args.rerank
        if args.window is not None:
            payload["context_window_size"] = args.window
        if args.dense_weight is not None:
            payload["dense_weight"] = args.dense_weight
        if args.model:
            payload["model"] = args.model
        _t0 = _time.time()
        resp = requests.post(f"{BACKEND_API}/rag/chat", json=payload, timeout=900)
        latencies.append(_time.time() - _t0)
        resp.raise_for_status()
        data = resp.json()
        answer = data.get("answer", "")
        answers.append(answer)
        sources = data.get("source_documents", [])
        source_sets.append(sources)
        ranks.append(_doc_rank(c["expected_doc"], sources))
        chunk_hits.append(_chunk_hit(c["chunk_id"], sources))
        # Citation validity: every [N] in the answer must reference a source
        # that exists (1-based, within range)
        cited = {int(m) for m in re.findall(r"\[(\d+)\]", answer)}
        citation_ok.append(bool(cited) and all(1 <= n <= len(sources) for n in cited))
        print(f"  answered ({latencies[-1]:.0f}s): {c['question'][:62]}")

    # Phase 2: judge all answers (accuracy, groundedness, context relevancy)
    def _judge(prompt, key):
        verdict = judge.generate_completion(
            GENERATION_MODEL, prompt, format="json", enable_thinking=False, timeout=120,
        )
        try:
            return bool(json.loads(verdict.get("response", "")).get(key))
        except (json.JSONDecodeError, AttributeError):
            return False

    correct = grounded = relevant = 0
    for c, answer, sources in zip(cases, answers, source_sets):
        src_text = "\n\n".join(
            f"[{i}] {s.get('text','')[:800]}" for i, s in enumerate(sources[:10], 1)
        )
        ok = _judge(JUDGE_PROMPT.format(reference=c["reference_answer"], candidate=answer[:3000]), "correct")
        correct += ok
        grounded += _judge(GROUNDEDNESS_PROMPT.format(sources=src_text, answer=answer[:3000]), "grounded")
        relevant += _judge(RELEVANCY_PROMPT.format(question=c["question"], sources=src_text), "relevant")
        print(f"  {'✓' if ok else '✗'} {c['question'][:75]}")

    metrics = _summarize(f"e2e retrieval — {table}", ranks, chunk_hits, args.k)
    n = len(cases)
    metrics.update({
        "answer_accuracy": correct / n,
        "groundedness": grounded / n,
        "context_relevancy": relevant / n,
        "citation_validity": sum(citation_ok) / n,
        "latency_avg_s": sum(latencies) / n,
        "latency_p95_s": sorted(latencies)[max(0, int(0.95 * n) - 1)],
    })
    print(f"answer accuracy (LLM judge): {correct}/{n} ({100*correct/n:.0f}%)")
    print(f"groundedness:                {grounded}/{n} ({100*grounded/n:.0f}%)")
    print(f"context relevancy:           {relevant}/{n} ({100*relevant/n:.0f}%)")
    print(f"citation validity:           {sum(citation_ok)}/{n} ({100*sum(citation_ok)/n:.0f}%)")
    print(f"latency avg/p95:             {metrics['latency_avg_s']:.0f}s / {metrics['latency_p95_s']:.0f}s")
    _dump_results(full_id, "e2e", metrics)


FIXTURES_DIR = os.path.join("tests", "eval_fixtures")


def cmd_gate(args):
    """CI gate: build the committed synthetic corpus into a temp index and
    fail (exit 1) if retrieval metrics fall below the committed thresholds.

    Deterministic and Ollama-free: real chunking + real embedder, stub LLM
    (overviews skip, enrichment off), retrieval-only scoring. Golden cases
    match by answer keyword, not chunk id, so chunker changes don't break it.
    """
    import shutil
    import tempfile

    import hashlib
    import numpy as np

    from rag_system.indexing.embedders import LanceDBManager
    from rag_system.retrieval.retrievers import MultiVectorRetriever

    corpus = sorted(
        str(f) for f in Path(os.path.join(FIXTURES_DIR, "corpus")).glob("*.txt")
    )
    golden = [json.loads(l) for l in open(os.path.join(FIXTURES_DIR, "golden.jsonl"), encoding="utf-8") if l.strip()]
    thresholds = json.load(open(os.path.join(FIXTURES_DIR, "thresholds.json"), encoding="utf-8"))
    if not corpus or not golden:
        raise SystemExit("eval fixtures missing")

    embedding_model = "fixture-hash-embedder"
    tmp = tempfile.mkdtemp(prefix="rag-gate-")
    try:
        import rag_system.pipelines.indexing_pipeline as indexing_module

        class _FixtureEmbedder:
            """Deterministic, dependency-free embeddings for the CI fixture."""

            dimensions = 256

            def create_embeddings(self, texts):
                vectors = []
                for text in texts:
                    vector = np.zeros(self.dimensions, dtype=np.float32)
                    for token in re.findall(r"[a-z0-9]+", text.lower()):
                        digest = hashlib.sha256(token.encode("utf-8")).digest()
                        slot = int.from_bytes(digest[:4], "big") % self.dimensions
                        vector[slot] += 1.0
                    norm = np.linalg.norm(vector)
                    if norm:
                        vector /= norm
                    vectors.append(vector)
                return np.vstack(vectors)

        class _NoLLM:  # overviews skip persisting on empty responses
            def generate_completion(self, *a, **k):
                return {"response": ""}
            def stream_completion(self, *a, **k):
                yield ""

        config = {
            "storage": {"lancedb_uri": tmp, "db_path": tmp, "text_table_name": "gate_corpus"},
            "db_path": os.path.join(tmp, "gate.sqlite3"),
            "index_store_path": os.path.join(tmp, "index_store"),
            "embedding_model_name": embedding_model,
            "contextual_enricher": {"enabled": False},
            "retrieval": {"late_chunking": {"enabled": False}},
            "overview_path": os.path.join(tmp, "ov.jsonl"),
            "indexing": {"embedding_batch_size": 16},
        }
        print(f"Building gate corpus ({len(corpus)} docs) ...")
        original_select_embedder = indexing_module.select_embedder
        indexing_module.select_embedder = lambda *_args, **_kwargs: _FixtureEmbedder()
        try:
            pipeline = indexing_module.IndexingPipeline(
                config,
                _NoLLM(),
                {"generation_model": "none", "enrichment_model": "none"},
            )
            # The gate favors isolation over the production worker optimization.
            pipeline._start_persistent_worker = lambda: setattr(pipeline, "_worker", None)
            def _convert_in_process(file_path, document_id):
                result = indexing_module.convert_and_chunk_document(file_path, document_id, config)
                if result.get("error"):
                    raise RuntimeError(result["error"])
                return result.get("chunks", [])
            pipeline._convert_and_chunk_file = _convert_in_process
            pipeline.run(corpus, index_id="gate-corpus", force_reindex=True)
        finally:
            indexing_module.select_embedder = original_select_embedder

        retriever = MultiVectorRetriever(LanceDBManager(tmp), _FixtureEmbedder())
        ranks, keyword_hits = [], []
        for case in golden:
            docs = retriever.retrieve(case["question"], table_name="gate_corpus", k=args.k)
            ranks.append(_doc_rank(case["expected_doc"], docs))
            kw = case["answer_keyword"].lower()
            keyword_hits.append(any(kw in (d.get("text") or "").lower() for d in docs))
        metrics = _summarize("CI gate — fixture corpus", ranks, keyword_hits, args.k)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n--- threshold check ---")
    failed = []
    for key, minimum in thresholds.items():
        actual = metrics.get(key)
        ok = actual is not None and actual >= minimum
        print(f"  {'PASS' if ok else 'FAIL'}  {key}: {actual if actual is not None else 'missing'} (min {minimum})")
        if not ok:
            failed.append(key)
    if failed:
        print(f"\nGATE FAILED: {', '.join(failed)} below threshold")
        raise SystemExit(1)
    print("\nGATE PASSED")


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
    # e2e sweep knobs (default: whatever the server config does)
    r.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=None,
                   help="--rerank / --no-rerank: toggle the AI reranker")
    r.add_argument("--window", type=int, default=None, help="context expansion window (0 disables)")
    r.add_argument("--dense-weight", type=float, default=None, help="vector weight in hybrid fusion (0..1)")
    r.add_argument("--reranker-top-k", type=int, default=None)
    r.add_argument("--model", default=None, help="synthesis model override (e.g. gpt-oss:20b)")

    g2 = sub.add_parser("gate", help="CI gate: score the committed fixture corpus against thresholds")
    g2.add_argument("--k", type=int, default=20)

    args = p.parse_args()
    if args.cmd == "gate":
        cmd_gate(args)
        return
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

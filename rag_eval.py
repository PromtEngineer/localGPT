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
- Also classify the question:
  - "category": one of "numeric" (asks for a number/amount/date), "entity" (asks for a name/place/org), or "factual" (anything else).
  - "difficulty": "easy" if the answer is stated verbatim and unambiguous, "medium" if it requires locating one specific detail among similar ones, "hard" if it needs combining or disambiguating details.
- Reply with JSON only: {{"question": "...", "answer": "...", "category": "...", "difficulty": "..."}}

PASSAGE (from {doc_name}):
{passage}
"""

# Accepted label vocabularies; anything else is normalized to the default.
_CATEGORIES = {"numeric", "entity", "factual"}
_DIFFICULTIES = {"easy", "medium", "hard"}


def _norm_label(value: object, allowed: set, default: str) -> str:
    """Coerce a model-supplied label into the accepted vocabulary."""
    v = str(value or "").strip().lower()
    return v if v in allowed else default

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

HELPFULNESS_PROMPT = """You will see a question and a candidate answer.

QUESTION: {question}

CANDIDATE ANSWER: {candidate}

Is the answer directly responsive and complete — does it actually answer what was asked, rather than deflecting, hedging, or saying it cannot find the information? Reply with JSON only: {{"helpful": true}} or {{"helpful": false}}.
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
                "category": _norm_label(qa.get("category"), _CATEGORIES, "factual"),
                "difficulty": _norm_label(qa.get("difficulty"), _DIFFICULTIES, "medium"),
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


def _classify_retrieval_failure(rank: "int | None", chunk_hit: bool, k: int) -> "str | None":
    """Human-readable reason a retrieval case underperformed, or None if clean.

    The two failure modes are distinct and worth separating: the expected
    document never surfaced at all, vs. the document surfaced but the chunk
    that actually holds the answer didn't reach the result set.
    """
    if rank is None:
        return "doc_not_retrieved"
    if rank > k:
        return f"doc_rank_beyond_k(rank={rank})"
    if not chunk_hit:
        return "answer_chunk_missed"
    return None


def _classify_e2e_failure(
    rank: "int | None", chunk_hit: bool, k: int,
    correct: bool, grounded: bool, relevant: bool, helpful: bool, citation_ok: bool,
) -> "str | None":
    """First-cause failure reason for an end-to-end case, or None if clean.

    Ordered by where the pipeline broke: a retrieval miss explains a wrong
    answer, so report it first rather than the downstream symptom.
    """
    retrieval = _classify_retrieval_failure(rank, chunk_hit, k)
    if retrieval is not None:
        return retrieval
    if not relevant:
        return "context_irrelevant"
    if not correct:
        return "answer_incorrect"
    if not grounded:
        return "answer_ungrounded"
    if not helpful:
        return "answer_unhelpful"
    if not citation_ok:
        return "bad_citation"
    return None


def _breakdown(cases: list, ranks: list, chunk_hits: list, key: str, k: int) -> dict:
    """Group cases by a label field (e.g. 'category') and report recall@k +
    chunk-hit per group, so a regression can be traced to a question type."""
    groups: dict = {}
    for case, rank, hit in zip(cases, ranks, chunk_hits, strict=False):
        groups.setdefault(case.get(key, "unknown"), []).append((rank, hit))
    out = {}
    for label, rows in sorted(groups.items()):
        n = len(rows)
        recall = sum(1 for r, _ in rows if r is not None and r <= k) / n if n else 0.0
        chunk = sum(1 for _, h in rows if h) / n if n else 0.0
        out[label] = {"n": n, f"recall@{k}": recall, "chunk_hit": chunk}
    return out


def _print_breakdown(title: str, groups: dict) -> None:
    if not groups or set(groups) == {"unknown"}:
        return  # nothing meaningful to break down (legacy set without labels)
    print(f"\n  by {title}:")
    for label, m in groups.items():
        kk = next(key for key in m if key.startswith("recall@"))
        print(f"    {label:<10} n={m['n']:<3} recall={100*m[kk]:.0f}%  chunk={100*m['chunk_hit']:.0f}%")


def _case_records(cases: list, ranks: list, chunk_hits: list, k: int) -> list:
    """Per-case rows (with failure reasons) for the dumped results file."""
    records = []
    for case, rank, hit in zip(cases, ranks, chunk_hits, strict=False):
        records.append({
            "question": case.get("question"),
            "expected_doc": case.get("expected_doc"),
            "category": case.get("category", "unknown"),
            "difficulty": case.get("difficulty", "unknown"),
            "rank": rank,
            "chunk_hit": hit,
            "failure": _classify_retrieval_failure(rank, hit, k),
        })
    return records


def _summarize(name: str, ranks: list, chunk_hits: list, k: int, quiet: bool = False) -> dict:
    """Print and return the metric set (returned dict feeds the CI gate).

    `quiet` suppresses the per-run printout — used when sweeping configs, where
    only the final comparison table matters.
    """
    n = len(ranks)
    metrics = {"n": n}
    if not quiet:
        print(f"\n=== {name} (n={n}) ===")
    for kk in (1, 3, 5, 10, k):
        if kk > k:
            continue
        hits = sum(1 for r in ranks if r is not None and r <= kk)
        metrics[f"doc_recall@{kk}"] = hits / n if n else 0.0
        if not quiet:
            print(f"doc recall@{kk:<2}: {hits}/{n} ({100*hits/n:.0f}%)")
    mrr = sum(1.0 / r for r in ranks if r is not None) / n if n else 0.0
    metrics["mrr"] = mrr
    chunks = sum(chunk_hits)
    metrics[f"chunk_hit@{k}"] = chunks / n if n else 0.0
    metrics["chunk_hit"] = chunks / n if n else 0.0
    if not quiet:
        print(f"MRR (doc):     {mrr:.3f}")
        print(f"chunk hit@{k}:  {chunks}/{n} ({100*chunks/n:.0f}%)  ← the chunk that contains the answer")
    return metrics


def _fusion_from_weight(dense_weight: "float | None") -> "dict | None":
    """Translate a 0..1 vector weight into the retriever's fusion override."""
    if dense_weight is None:
        return None
    w = max(0.0, min(1.0, dense_weight))
    return {"bm25_weight": 1.0 - w, "vec_weight": w}


def _score_retrieval(retriever, cases, table, k, fusion_override, verbose=True):
    """Run retrieval for every case; return (ranks, chunk_hits) in case order."""
    ranks, chunk_hits = [], []
    for c in cases:
        docs = retriever.retrieve(
            c["question"], table_name=table, k=k, fusion_override=fusion_override
        )
        rank = _doc_rank(c["expected_doc"], docs)
        ranks.append(rank)
        chunk_hits.append(_chunk_hit(c["chunk_id"], docs))
        if verbose:
            mark = "✓" if rank is not None else "✗"
            cmark = "chunk✓" if chunk_hits[-1] else "chunk✗"
            print(f"  {mark} rank={rank if rank else '-'} {cmark}  {c['question'][:70]}")
    return ranks, chunk_hits


def cmd_run_retrieval(args, full_id: str, table: str, embedding_model: str):
    from rag_system.indexing.embedders import LanceDBManager
    from rag_system.indexing.representations import select_embedder
    from rag_system.retrieval.retrievers import MultiVectorRetriever

    cases = _load_eval_set(full_id)
    retriever = MultiVectorRetriever(LanceDBManager("./lancedb"), select_embedder(embedding_model))
    fusion_override = _fusion_from_weight(args.dense_weight)
    ranks, chunk_hits = _score_retrieval(retriever, cases, table, args.k, fusion_override)
    metrics = _summarize(f"retrieval — {table}", ranks, chunk_hits, args.k)
    metrics["by_category"] = _breakdown(cases, ranks, chunk_hits, "category", args.k)
    metrics["by_difficulty"] = _breakdown(cases, ranks, chunk_hits, "difficulty", args.k)
    _print_breakdown("category", metrics["by_category"])
    _print_breakdown("difficulty", metrics["by_difficulty"])
    _dump_results(full_id, "retrieval", metrics)
    _dump_cases(full_id, "retrieval", _case_records(cases, ranks, chunk_hits, args.k))
    _handle_baseline(full_id, "retrieval", metrics, args)


_COMPARE_METRICS = ["doc_recall@1", "doc_recall@5", "mrr", "chunk_hit"]


def _best_per_metric(results: list, metric_keys: list) -> dict:
    """Label of the winning config for each metric (higher is better).

    `results` is a list of (label, metrics) pairs. Ties keep the first config,
    so an order like ascending dense-weight prefers the lower-weight winner.
    """
    best = {}
    for key in metric_keys:
        scored = [(label, m.get(key)) for label, m in results if m.get(key) is not None]
        if scored:
            best[key] = max(scored, key=lambda lv: lv[1])[0]
    return best


def _format_comparison(results: list, metric_keys: list) -> str:
    """Render a config x metric table, starring the winner in each column."""
    best = _best_per_metric(results, metric_keys)
    width = max([len("config")] + [len(label) for label, _ in results])
    header = "  " + "config".ljust(width) + "".join(f"  {k:>12}" for k in metric_keys)
    lines = [header, "  " + "-" * (len(header) - 2)]
    for label, m in results:
        cells = []
        for k in metric_keys:
            val = m.get(k)
            star = "*" if best.get(k) == label else " "
            cells.append(f"  {('%.3f' % val if val is not None else '-'):>11}{star}")
        lines.append("  " + label.ljust(width) + "".join(cells))
    return "\n".join(lines)


def cmd_compare(args, full_id: str, table: str, embedding_model: str):
    """Sweep one retrieval knob (dense-weight) and tabulate the metrics so the
    best setting for an index is a single fast, server-free run."""
    from rag_system.indexing.embedders import LanceDBManager
    from rag_system.indexing.representations import select_embedder
    from rag_system.retrieval.retrievers import MultiVectorRetriever

    cases = _load_eval_set(full_id)
    retriever = MultiVectorRetriever(LanceDBManager("./lancedb"), select_embedder(embedding_model))
    results = []
    for w in args.dense_weights:
        fusion = _fusion_from_weight(w)
        ranks, chunk_hits = _score_retrieval(retriever, cases, table, args.k, fusion, verbose=False)
        metrics = _summarize("", ranks, chunk_hits, args.k, quiet=True)
        results.append((f"dense={w:.2f}", metrics))
        print(f"  scored dense-weight {w:.2f}")
    print(f"\n=== config comparison — {table} (n={len(cases)}, k={args.k}) ===")
    print(_format_comparison(results, _COMPARE_METRICS))
    print("\n  (* = best in column)")


def _dump_results(full_id: str, mode: str, metrics: dict):
    os.makedirs(EVAL_DIR, exist_ok=True)
    path = os.path.join(EVAL_DIR, f"results-{full_id[:8]}-{mode}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"(metrics written to {path})")


def _dump_cases(full_id: str, mode: str, records: list):
    """Per-case records — the failure rows are what you read after a drop."""
    os.makedirs(EVAL_DIR, exist_ok=True)
    path = os.path.join(EVAL_DIR, f"results-{full_id[:8]}-{mode}-cases.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    failures = [r for r in records if r.get("failure")]
    if failures:
        print(f"\n  {len(failures)} failing case(s) → {path}")
        for r in failures[:10]:
            print(f"    ✗ [{r['failure']}] {(r.get('question') or '')[:64]}")


def _baseline_path(full_id: str, mode: str) -> str:
    return os.path.join(EVAL_DIR, f"baseline-{full_id[:8]}-{mode}.json")


def _scalar_metrics(metrics: dict) -> dict:
    """Flat numeric metrics only — drops n and the nested by_* breakdowns so a
    baseline compares like-for-like and never trips over a bool/dict."""
    out = {}
    for key, value in metrics.items():
        if key == "n" or isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            out[key] = value
    return out


def _compare_to_baseline(current: dict, baseline: dict, tolerance: float = 0.02) -> list:
    """Quality metrics that dropped below baseline by more than `tolerance`.

    All tracked quality metrics are higher-is-better, so a regression is a
    drop. Latency is reported informationally elsewhere, not gated here, since
    it's noisy and scale-incompatible with the [0,1] tolerance.
    """
    regressions = []
    cur, base = _scalar_metrics(current), _scalar_metrics(baseline)
    for key, base_val in base.items():
        if "latency" in key or key not in cur:
            continue
        delta = cur[key] - base_val
        if delta < -tolerance:
            regressions.append(
                {"metric": key, "baseline": base_val, "current": cur[key], "delta": delta}
            )
    return regressions


def _handle_baseline(full_id: str, mode: str, metrics: dict, args) -> None:
    """Save and/or compare a per-index baseline; exit 1 on a real regression so
    this doubles as a per-index CI check (separate from the fixture gate)."""
    if getattr(args, "save_baseline", False):
        os.makedirs(EVAL_DIR, exist_ok=True)
        path = _baseline_path(full_id, mode)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_scalar_metrics(metrics), f, indent=2)
        print(f"\n(baseline saved to {path})")
    if getattr(args, "compare_baseline", False):
        path = _baseline_path(full_id, mode)
        if not os.path.exists(path):
            raise SystemExit(f"No baseline at {path} — run once with --save-baseline first")
        baseline = json.load(open(path, encoding="utf-8"))
        regressions = _compare_to_baseline(metrics, baseline, args.tolerance)
        print(f"\n--- baseline comparison (tolerance {args.tolerance:+.2f}) ---")
        for key in sorted(_scalar_metrics(baseline)):
            cur = _scalar_metrics(metrics).get(key)
            base = baseline[key]
            if cur is None:
                continue
            d = cur - base
            flag = "REGRESSED" if any(r["metric"] == key for r in regressions) else "ok"
            print(f"  {flag:<10} {key}: {base:.3f} -> {cur:.3f} ({d:+.3f})")
        if regressions:
            names = ", ".join(r["metric"] for r in regressions)
            print(f"\nBASELINE REGRESSION: {names}")
            raise SystemExit(1)
        print("\nNo regressions vs baseline.")


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

    correct = grounded = relevant = helpful = 0
    e2e_cases = []
    for c, answer, sources, rank, chit, cite_ok in zip(
        cases, answers, source_sets, ranks, chunk_hits, citation_ok, strict=False
    ):
        src_text = "\n\n".join(
            f"[{i}] {s.get('text','')[:800]}" for i, s in enumerate(sources[:10], 1)
        )
        ok = _judge(JUDGE_PROMPT.format(reference=c["reference_answer"], candidate=answer[:3000]), "correct")
        is_grounded = _judge(GROUNDEDNESS_PROMPT.format(sources=src_text, answer=answer[:3000]), "grounded")
        is_relevant = _judge(RELEVANCY_PROMPT.format(question=c["question"], sources=src_text), "relevant")
        is_helpful = _judge(HELPFULNESS_PROMPT.format(question=c["question"], candidate=answer[:3000]), "helpful")
        correct += ok
        grounded += is_grounded
        relevant += is_relevant
        helpful += is_helpful
        e2e_cases.append({
            "question": c.get("question"),
            "expected_doc": c.get("expected_doc"),
            "category": c.get("category", "unknown"),
            "difficulty": c.get("difficulty", "unknown"),
            "rank": rank,
            "chunk_hit": chit,
            "failure": _classify_e2e_failure(
                rank, chit, args.k, ok, is_grounded, is_relevant, is_helpful, cite_ok
            ),
        })
        print(f"  {'✓' if ok else '✗'} {c['question'][:75]}")

    metrics = _summarize(f"e2e retrieval — {table}", ranks, chunk_hits, args.k)
    metrics["by_category"] = _breakdown(cases, ranks, chunk_hits, "category", args.k)
    metrics["by_difficulty"] = _breakdown(cases, ranks, chunk_hits, "difficulty", args.k)
    _print_breakdown("category", metrics["by_category"])
    _print_breakdown("difficulty", metrics["by_difficulty"])
    n = len(cases)
    metrics.update({
        "answer_accuracy": correct / n,
        "groundedness": grounded / n,
        "context_relevancy": relevant / n,
        "helpfulness": helpful / n,
        "citation_validity": sum(citation_ok) / n,
        "latency_avg_s": sum(latencies) / n,
        "latency_p95_s": sorted(latencies)[max(0, int(0.95 * n) - 1)],
    })
    print(f"answer accuracy (LLM judge): {correct}/{n} ({100*correct/n:.0f}%)")
    print(f"groundedness:                {grounded}/{n} ({100*grounded/n:.0f}%)")
    print(f"context relevancy:           {relevant}/{n} ({100*relevant/n:.0f}%)")
    print(f"helpfulness:                 {helpful}/{n} ({100*helpful/n:.0f}%)")
    print(f"citation validity:           {sum(citation_ok)}/{n} ({100*sum(citation_ok)/n:.0f}%)")
    print(f"latency avg/p95:             {metrics['latency_avg_s']:.0f}s / {metrics['latency_p95_s']:.0f}s")
    _dump_results(full_id, "e2e", metrics)
    _dump_cases(full_id, "e2e", e2e_cases)
    _handle_baseline(full_id, "e2e", metrics, args)


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
    # Per-index regression baselines (separate from the fixture CI gate)
    r.add_argument("--save-baseline", action="store_true",
                   help="record this run's metrics as the accepted baseline for the index+mode")
    r.add_argument("--compare-baseline", action="store_true",
                   help="compare this run against the saved baseline; exit 1 on a regression")
    r.add_argument("--tolerance", type=float, default=0.02,
                   help="allowed quality-metric drop before a baseline regression is flagged")

    c = sub.add_parser("compare", help="sweep a retrieval knob and tabulate metrics side by side")
    c.add_argument("--index", required=True, help="index id (prefix ok)")
    c.add_argument("--k", type=int, default=20)
    c.add_argument("--dense-weights", type=float, nargs="+", default=[0.0, 0.3, 0.5, 0.7, 1.0],
                   help="vector weights to compare (0..1); each is one column row")

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
    elif args.cmd == "compare":
        cmd_compare(args, full_id, table, model)
    elif args.mode == "retrieval":
        cmd_run_retrieval(args, full_id, table, model)
    else:
        cmd_run_e2e(args, full_id, table, model)


if __name__ == "__main__":
    main()

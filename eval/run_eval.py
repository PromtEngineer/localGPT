"""Retrieval metrics runner for the localGPT gold set (Phase 0.2).

In-process: no HTTP server is started. The harness builds a LanceDB index with
the repo's own ``IndexingPipeline`` and queries it with the repo's own
``RetrievalPipeline`` components (``MultiVectorRetriever`` for the first stage,
``_get_ai_reranker()`` for the cross-encoder), so the numbers describe the
shipped retrieval path — not a reimplementation of it.

What it deliberately does NOT do: answer synthesis, context expansion, Provence
pruning, late chunking, contextual enrichment. Those are downstream of the two
metrics the roadmap says matter (first-stage recall@k, post-rerank nDCG@10) and
each one adds an LLM round-trip or a nondeterministic step.

Metric definitions (binary relevance, answer-bearing text match):

  hit(chunk, query)  A gold row carries one or more ``expected`` substrings.
                     A chunk is relevant when its text contains any of them
                     (whitespace-normalised, case-insensitive).
  recall@k           match="any": 1.0 when at least one of the top-k chunks is
                     relevant. match="all" (comparatives): 1.0 only when every
                     expected substring appears somewhere in the top-k union.
                     Reported as the mean over queries — with one gold target
                     per query this is recall, hit-rate and success@k alike.
  nDCG@10            DCG over binary per-chunk relevance of the top 10, divided
                     by the IDCG of the *same candidate set* re-sorted ideally.
                     A query whose candidate set contains no relevant chunk
                     scores 0, so a first-stage miss is never hidden.

Gold relevance is text-based, never chunk-id-based, so the set survives
re-chunking and embedder swaps.

Usage (see eval/README.md):

    .venv/bin/python eval/run_eval.py --corpus all                     # shipped defaults
    EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B .venv/bin/python eval/run_eval.py --corpus all

The rerank stage follows the shipped "default" profile, which has reranking off
(eval/DECISIONS.md). Pass ``--reranker <model>`` to switch it on for a run.
"""

import argparse
import contextlib
import glob
import io
import json
import math
import os
import platform
import random
import shutil
import sys
import time
from datetime import datetime, timezone

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(EVAL_DIR, ".."))
sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rag_system.factory import _build_llm_client, get_pipeline_config  # noqa: E402
from rag_system.indexing.embedders import LanceDBManager  # noqa: E402
from rag_system.main import EXTERNAL_MODELS  # noqa: E402
from rag_system.pipelines.indexing_pipeline import IndexingPipeline  # noqa: E402
from rag_system.pipelines.retrieval_pipeline import RetrievalPipeline  # noqa: E402

SEED = 20260808
INDEX_ROOT = os.path.join(EVAL_DIR, ".eval_indexes")
RESULTS_DIR = os.path.join(EVAL_DIR, "results")
GOLDSET_DIR = os.path.join(EVAL_DIR, "goldset")

# Excluded from the docs corpus on purpose: these two files are the planning
# documents this harness is tracked in, so every eval-related edit would change
# the corpus and move the baseline. Nothing in the gold set anchors on them.
DOCS_EXCLUDE = {"improvement_plan.md", "research_roadmap.md"}

CORPORA = {
    "atlas7": {
        "label": "Atlas-7 service manual (planted-fact PDF)",
        "files": [os.path.join(EVAL_DIR, "corpora", "atlas7_service_manual.pdf")],
    },
    "hr": {
        "label": "Northwind leave policy (synthetic planted-fact PDF)",
        "files": [os.path.join(EVAL_DIR, "corpora", "northwind_leave_policy.pdf")],
    },
    "docs": {
        "label": "localGPT Documentation/*.md (real heterogeneous corpus)",
        "glob": os.path.join(REPO_ROOT, "Documentation", "*.md"),
    },
    # Every corpus in one table. The two planted-fact PDFs are only a handful of
    # chunks each, so in isolation k=20 sweeps the whole document and recall@k
    # saturates at 1.0 by construction. Here their queries have to beat the
    # documentation chunks as distractors, which is the number worth tracking.
    "mixed": {
        "label": "all three corpora in one table (planted-fact PDFs + docs as distractors)",
        "files": [
            os.path.join(EVAL_DIR, "corpora", "atlas7_service_manual.pdf"),
            os.path.join(EVAL_DIR, "corpora", "northwind_leave_policy.pdf"),
        ],
        "glob": os.path.join(REPO_ROOT, "Documentation", "*.md"),
        "goldset_of": ["atlas7", "hr", "docs"],
    },
}

RECALL_KS = (5, 10, 20)
NDCG_K = 10


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def norm(text: str) -> str:
    return " ".join((text or "").split()).lower()


def slug(name: str) -> str:
    return name.replace("/", "__").replace(":", "_")


def corpus_files(corpus: str) -> list:
    spec = CORPORA[corpus]
    files = list(spec.get("files", []))
    if "glob" in spec:
        files.extend(f for f in glob.glob(spec["glob"])
                     if os.path.basename(f) not in DOCS_EXCLUDE)
    return sorted(files)


def _read_gold_file(name: str) -> list:
    path = os.path.join(GOLDSET_DIR, f"{name}.jsonl")
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("//"):
                rows.append(json.loads(line))
    return rows


def load_goldset(corpus: str) -> list:
    names = CORPORA[corpus].get("goldset_of", [corpus])
    rows = [row for name in names for row in _read_gold_file(name)]
    return sorted(rows, key=lambda r: r["id"])


@contextlib.contextmanager
def captured(log_path: str, verbose: bool):
    """Send the pipeline's chatty stdout to a log file unless --verbose."""
    if verbose:
        yield
        return
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            yield
    finally:
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(buf.getvalue())


# --------------------------------------------------------------------------
# index build / reuse
# --------------------------------------------------------------------------

def build_config(corpus: str, embedder: str, reranker: str, db_path: str, table: str,
                 k: int, chunk_size: int, rerank_enabled: bool,
                 aggregate: str = "mean") -> dict:
    """The 'default' profile with every nondeterministic / LLM-dependent stage off."""
    cfg = get_pipeline_config("default")
    cfg["storage"]["lancedb_uri"] = db_path
    cfg["storage"]["text_table_name"] = table
    cfg["embedding_model_name"] = embedder
    cfg["chunker_mode"] = "docling"
    cfg["chunking"] = {"chunk_size": chunk_size}
    # Off for determinism and speed — every one of these is an LLM round-trip
    # per chunk, per document, or a second embedding pass.
    cfg["contextual_enricher"] = {"enabled": False, "window_size": 1}
    cfg["overview"] = {"enabled": False}
    cfg["retrieval"]["latechunk"] = {"enabled": False}
    cfg["retrieval"]["search_type"] = "hybrid"
    # `enabled: False` keeps the *agent's* decomposition branch out of it — this
    # harness never calls Agent.run(). Sub-queries, when --decompose is passed,
    # are handed straight to the rerank stage, which is where item 2.2 puts them.
    cfg["query_decomposition"] = {"enabled": False, "rerank_aggregate": aggregate}
    cfg["verification"] = {"enabled": False}
    cfg["context_window_size"] = 0
    cfg["retrieval_k"] = k
    cfg["reranker"] = {
        "enabled": rerank_enabled,
        "model_type": "cross-encoder",
        "strategy": "rerankers-lib",
        "model_name": reranker,
        "top_k": None,  # rank the whole candidate list; we truncate for nDCG@10
    }
    return cfg


def apply_retry_setting(cfg: dict, mode: str) -> dict:
    """Force the evidence-sufficiency retry on/off, or leave the profile's value.

    Unlike every other stage this harness disables, the retry is *conditional*:
    it only fires on queries whose first pass found weak evidence, so leaving it
    at the profile default is what measures the shipped stack. ``--no-retry``
    produces the control arm.
    """
    if mode == "profile":
        return cfg
    block = dict(cfg["retrieval"].get("retry") or {})
    block["enabled"] = (mode == "on")
    cfg["retrieval"]["retry"] = block
    return cfg


def index_fingerprint(corpus: str, embedder: str, chunk_size: int) -> dict:
    files = corpus_files(corpus)
    return {
        "corpus": corpus,
        "embedder": embedder,
        "chunk_size": chunk_size,
        "enrichment": False,
        "latechunk": False,
        # Vectors are L2-normalized at write time since the Phase 1 adoption
        # (eval/DECISIONS.md). Carrying it here invalidates every index built
        # before that, exactly once, instead of silently reusing them.
        "normalized": True,
        "files": [
            {"path": os.path.relpath(f, REPO_ROOT), "size": os.path.getsize(f),
             "mtime": round(os.path.getmtime(f), 3)}
            for f in files
        ],
    }


def ensure_index(corpus: str, embedder: str, chunk_size: int, cfg: dict, db_path: str,
                 table: str, force: bool, log_path: str, verbose: bool) -> dict:
    marker = os.path.join(db_path, f"{table}.built.json")
    fingerprint = index_fingerprint(corpus, embedder, chunk_size)

    if not force and os.path.exists(marker):
        with open(marker, "r", encoding="utf-8") as fh:
            existing = json.load(fh)
        if existing.get("fingerprint") == fingerprint:
            print(f"  reusing cached index  {db_path}::{table} "
                  f"({existing['chunks']} chunks, built {existing['built_at']})")
            return existing

    if os.path.isdir(db_path):
        shutil.rmtree(db_path)
    os.makedirs(db_path, exist_ok=True)

    files = corpus_files(corpus)
    print(f"  building index        {db_path}::{table} from {len(files)} file(s)…")
    llm_client, llm_config = _build_llm_client()
    t0 = time.time()
    with captured(log_path, verbose):
        pipeline = IndexingPipeline(cfg, llm_client, llm_config)
        pipeline.run(files)
    build_seconds = time.time() - t0

    with captured(log_path, verbose):
        tbl = LanceDBManager(db_path=db_path).get_table(table)
        chunks = len(tbl.to_pandas())

    record = {
        "fingerprint": fingerprint,
        "chunks": chunks,
        "build_seconds": round(build_seconds, 2),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(marker, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
    print(f"  built                 {chunks} chunks in {build_seconds:.1f}s")
    return record


def chunk_texts(db_path: str, table: str, log_path: str, verbose: bool) -> list:
    with captured(log_path, verbose):
        tbl = LanceDBManager(db_path=db_path).get_table(table)
        df = tbl.to_pandas()
    return [norm(t) for t in df["text"].tolist()]


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def relevance(chunk_text: str, expected: list) -> int:
    n = norm(chunk_text)
    return 1 if any(norm(e) in n for e in expected) else 0


def query_hit(ranked_texts: list, expected: list, match: str) -> int:
    joined = " || ".join(norm(t) for t in ranked_texts)
    if match == "all":
        return 1 if all(norm(e) in joined for e in expected) else 0
    return 1 if any(norm(e) in joined for e in expected) else 0


def ndcg_at_k(rels: list, k: int) -> float:
    top = rels[:k]
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(top))
    ideal = sorted(rels, reverse=True)[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return (dcg / idcg) if idcg > 0 else 0.0


SUBQUERY_CACHE = os.path.join(INDEX_ROOT, "_subqueries")


def decomposition_sub_queries(pipeline: RetrievalPipeline, gold: list, args,
                              log_path: str) -> dict:
    """Sub-queries per gold row for the roadmap-2.2 A/B, or ``{}`` when off.

    `QueryDecomposer` is an LLM call, so the result is cached per corpus and
    reused: the decomposition-on and decomposition-off arms must differ only in
    whether the sub-queries are *used*, never in what they are. The first stage
    never sees them — item 2.2's whole point is that they apply at reranking.
    """
    if not args.decompose:
        return {}
    os.makedirs(SUBQUERY_CACHE, exist_ok=True)
    path = os.path.join(SUBQUERY_CACHE, f"{args.corpus_being_run}.json")
    cache = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            cache = json.load(fh)

    from rag_system.retrieval.query_transformer import QueryDecomposer  # noqa: E402
    llm_client, llm_config = _build_llm_client()
    model = llm_config.get("enrichment_model") or llm_config["generation_model"]
    decomposer = QueryDecomposer(llm_client, model)

    missing = [row for row in gold if row["id"] not in cache]
    if missing:
        print(f"  decomposing            {len(missing)} query/queries with {model}…")
        for row in missing:
            with captured(log_path, args.verbose):
                cache[row["id"]] = decomposer.decompose(row["query"], [], max_sub_queries=10)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(cache, fh, indent=2)
    multi = sum(1 for v in cache.values() if len(v) > 1)
    print(f"  sub-queries            {multi}/{len(gold)} rows decomposed into >1 sub-query")
    return cache


# --------------------------------------------------------------------------
# main evaluation
# --------------------------------------------------------------------------

def rerank_settings(args) -> tuple:
    """(enabled, model_name) for this run.

    The shipped "default" profile turns reranking off (eval/DECISIONS.md), and a
    bare eval run must measure what ships. Naming a model with ``--reranker``
    switches it back on, which is what the A/B commands in eval/decisions/*.md
    do; ``--no-rerank`` always wins.
    """
    profile_default = bool(
        get_pipeline_config("default").get("reranker", {}).get("enabled", False)
    )
    enabled = (not args.no_rerank) and (args.reranker is not None or profile_default)
    return enabled, (args.reranker or EXTERNAL_MODELS["reranker_model"])


def evaluate_corpus(corpus: str, args, log_path: str) -> dict:
    embedder = args.embedder
    rerank_enabled, reranker_name = rerank_settings(args)
    db_path = os.path.join(INDEX_ROOT, slug(embedder), corpus)
    table = f"eval_{corpus}"

    args.corpus_being_run = corpus
    print(f"\n=== corpus: {corpus} — {CORPORA[corpus]['label']}")
    cfg = build_config(corpus, embedder, reranker_name, db_path, table,
                       args.k, args.chunk_size, rerank_enabled, args.aggregate)
    cfg = apply_retry_setting(cfg, args.retry)
    index_record = ensure_index(corpus, embedder, args.chunk_size, cfg, db_path, table,
                                args.force_reindex, log_path, args.verbose)

    gold = load_goldset(corpus)
    if not gold:
        names = CORPORA[corpus].get("goldset_of", [corpus])
        print(f"  no gold rows for {names} in {GOLDSET_DIR} — skipping queries")
        return {"corpus": corpus, "index": index_record, "queries": [], "coverage": []}

    # Gate 2 of gold verification: does the answer-bearing text survive
    # conversion + chunking into at least one indexed chunk? A gold row that
    # fails here is structurally unreachable and is reported, not hidden.
    all_chunks = chunk_texts(db_path, table, log_path, args.verbose)
    coverage = []
    for row in gold:
        missing = [e for e in row["expected"] if not any(norm(e) in c for c in all_chunks)]
        if missing:
            coverage.append({"id": row["id"], "missing": missing})
    if coverage:
        print(f"  ⚠️  {len(coverage)} gold row(s) whose expected text is in NO indexed chunk:")
        for c in coverage:
            print(f"       {c['id']}: {c['missing']}")
    else:
        print(f"  gold coverage         {len(gold)}/{len(gold)} rows reachable in the index")

    if args.coverage_only:
        return {"corpus": corpus, "index": index_record, "queries": [], "coverage": coverage}

    with captured(log_path, args.verbose):
        pipeline = RetrievalPipeline(cfg, *_build_llm_client())
        pipeline.retriever  # force lazy init outside the timed section

    decompose = decomposition_sub_queries(pipeline, gold, args, log_path)

    results = []
    for row in gold:
        query, expected, match = row["query"], row["expected"], row.get("match", "any")

        # Go through the pipeline's own candidate path (first stage + optional
        # rerank + the evidence-sufficiency retry) rather than calling the
        # retriever directly, so the harness measures the shipped behaviour.
        t0 = time.time()
        with captured(log_path, args.verbose):
            out = pipeline.retrieve_candidates(query, table_name=table,
                                               sub_queries=decompose.get(row["id"]))
        total_ms = (time.time() - t0) * 1000.0
        docs = out["first_stage"]
        first_texts = [d.get("text", "") for d in docs]

        entry = {
            "id": row["id"],
            "query": query,
            "dimensions": row.get("dimensions", {}),
            "match": match,
            "expected": expected,
            "candidates": len(docs),
            "first_stage_ms": round(total_ms, 1),
            "recall": {f"@{k}": query_hit(first_texts[:k], expected, match) for k in RECALL_KS},
            "ndcg10_first_stage": round(
                ndcg_at_k([relevance(t, expected) for t in first_texts], NDCG_K), 4),
        }
        if out.get("retry"):
            entry["retry"] = out["retry"]
        if decompose.get(row["id"]):
            entry["sub_queries"] = decompose[row["id"]]

        if rerank_enabled and docs:
            reranked = out["documents"]
            entry["rerank_ms"] = 0.0  # folded into first_stage_ms on this path
            if reranked is None or reranked is docs:
                entry["rerank_error"] = "reranker failed to load; see the run log"
            else:
                rr_texts = [d.get("text", "") for d in reranked]
                entry["ndcg10_reranked"] = round(
                    ndcg_at_k([relevance(t, expected) for t in rr_texts], NDCG_K), 4)
                entry["recall_reranked"] = {
                    f"@{k}": query_hit(rr_texts[:k], expected, match) for k in RECALL_KS
                }
        results.append(entry)
        rr_val = entry.get("ndcg10_reranked")
        rr_cell = "n/a  " if rr_val is None else f"{rr_val:.3f}"
        print(f"  [{entry['id']}] r@5={entry['recall']['@5']} r@10={entry['recall']['@10']} "
              f"r@20={entry['recall']['@20']} "
              f"nDCG@10 first={entry['ndcg10_first_stage']:.3f} rerank={rr_cell} "
              f"{entry['first_stage_ms']:.0f}ms+{entry.get('rerank_ms', 0):.0f}ms")

    return {"corpus": corpus, "index": index_record, "queries": results, "coverage": coverage}


def mean(values: list) -> float:
    return sum(values) / len(values) if values else float("nan")


def summarise(corpus_result: dict) -> dict:
    qs = corpus_result["queries"]
    if not qs:
        return {}
    summary = {
        "n_queries": len(qs),
        "recall@5": round(mean([q["recall"]["@5"] for q in qs]), 4),
        "recall@10": round(mean([q["recall"]["@10"] for q in qs]), 4),
        "recall@20": round(mean([q["recall"]["@20"] for q in qs]), 4),
        "ndcg@10_first_stage": round(mean([q["ndcg10_first_stage"] for q in qs]), 4),
        "first_stage_ms_mean": round(mean([q["first_stage_ms"] for q in qs]), 1),
        "first_stage_ms_p90": round(sorted(q["first_stage_ms"] for q in qs)[int(0.9 * (len(qs) - 1))], 1),
    }
    fired = [q for q in qs if q.get("retry")]
    if fired:
        kept = [q for q in fired if q["retry"]["kept"] == "retry"]
        summary["retry_fired"] = len(fired)
        summary["retry_fire_rate"] = round(len(fired) / len(qs), 4)
        summary["retry_kept"] = len(kept)
    reranked = [q for q in qs if "ndcg10_reranked" in q]
    if reranked:
        summary["ndcg@10_reranked"] = round(mean([q["ndcg10_reranked"] for q in reranked]), 4)
        summary["rerank_ms_mean"] = round(mean([q["rerank_ms"] for q in reranked]), 1)
        summary["rerank_ms_p90"] = round(
            sorted(q["rerank_ms"] for q in reranked)[int(0.9 * (len(reranked) - 1))], 1)
    return summary


def by_dimension(all_results: list) -> dict:
    buckets = {}
    for corpus_result in all_results:
        for q in corpus_result["queries"]:
            for key in ("question_type", "difficulty"):
                value = q["dimensions"].get(key)
                if value is None:
                    continue
                bucket = buckets.setdefault(f"{key}={value}", {"n": 0, "r@10": 0.0, "ndcg": 0.0, "ndcg_n": 0})
                bucket["n"] += 1
                bucket["r@10"] += q["recall"]["@10"]
                if "ndcg10_reranked" in q:
                    bucket["ndcg"] += q["ndcg10_reranked"]
                    bucket["ndcg_n"] += 1
    for bucket in buckets.values():
        bucket["recall@10"] = round(bucket.pop("r@10") / bucket["n"], 4)
        ndcg_n = bucket.pop("ndcg_n")
        total = bucket.pop("ndcg")
        bucket["ndcg@10_reranked"] = round(total / ndcg_n, 4) if ndcg_n else None
    return dict(sorted(buckets.items()))


def print_table(all_results: list) -> None:
    header = (f"{'corpus':<10} {'n':>4} {'chunks':>7} {'R@5':>7} {'R@10':>7} {'R@20':>7} "
              f"{'nDCG@10':>9} {'nDCG@10':>9} {'1st ms':>8} {'rr ms':>8}")
    print("\n" + "=" * len(header))
    print(header)
    print(f"{'':<10} {'':>4} {'':>7} {'':>7} {'':>7} {'':>7} {'(1st)':>9} {'(rerank)':>9} {'':>8} {'':>8}")
    print("-" * len(header))
    for corpus_result in all_results:
        s = summarise(corpus_result)
        if not s:
            continue
        rr = s.get("ndcg@10_reranked")
        rr_cell = "n/a" if rr is None else f"{rr:.3f}"
        rr_ms = s.get("rerank_ms_mean")
        rr_ms_cell = "n/a" if rr_ms is None else f"{rr_ms:.0f}"
        print(f"{corpus_result['corpus']:<10} {s['n_queries']:>4} "
              f"{corpus_result['index']['chunks']:>7} "
              f"{s['recall@5']:>7.3f} {s['recall@10']:>7.3f} {s['recall@20']:>7.3f} "
              f"{s['ndcg@10_first_stage']:>9.3f} {rr_cell:>9} "
              f"{s['first_stage_ms_mean']:>8.0f} {rr_ms_cell:>8}")
    print("=" * len(header))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", default="all", choices=["all", *sorted(CORPORA)],
                        help="corpus to evaluate (default: all)")
    parser.add_argument("--embedder", default=EXTERNAL_MODELS["embedding_model"],
                        help="HF embedding model; defaults to EMBEDDING_MODEL / the repo default")
    parser.add_argument("--reranker", default=None,
                        help="reranker model name; naming one enables the rerank stage "
                             "(off by default, matching the shipped profile)")
    parser.add_argument("--k", type=int, default=20, help="first-stage candidates per query")
    parser.add_argument("--chunk-size", type=int, default=512,
                        help="token budget per chunk (512 = what the HTTP path sends)")
    parser.add_argument("--no-rerank", action="store_true", help="skip the cross-encoder stage")
    parser.add_argument("--retry", choices=["profile", "on", "off"], default="profile",
                        help="evidence-sufficiency retry: follow the shipped profile "
                             "(default), or force it on/off for an A/B")
    parser.add_argument("--aggregate", choices=["max", "mean"], default="mean",
                        help="how to combine per-sub-query rerank scores (--decompose only)")
    parser.add_argument("--decompose", action="store_true",
                        help="decompose each query and score candidates against the "
                             "sub-queries AT RERANK (roadmap 2.2). The first stage always "
                             "uses the full original query. No-op without --reranker.")
    parser.add_argument("--coverage-only", action="store_true",
                        help="build/reuse indexes and report gold reachability, then stop")
    parser.add_argument("--force-reindex", action="store_true", help="rebuild even if cached")
    parser.add_argument("--json-out", default=None, help="results path (default: eval/results/<ts>.json)")
    parser.add_argument("--verbose", action="store_true", help="let the pipeline print to stdout")
    args = parser.parse_args()

    seed_everything()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(INDEX_ROOT, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_out = args.json_out or os.path.join(RESULTS_DIR, f"eval_{stamp}.json")
    log_path = os.path.join(RESULTS_DIR, f"eval_{stamp}.log")

    corpora = sorted(CORPORA) if args.corpus == "all" else [args.corpus]

    rerank_enabled, reranker_name = rerank_settings(args)

    print("localGPT retrieval eval")
    print(f"  embedder   {args.embedder}")
    print(f"  reranker   {reranker_name if rerank_enabled else '(disabled)'}")
    retry_cfg = (get_pipeline_config("default").get("retrieval", {}).get("retry") or {})
    if args.retry == "profile":
        retry_desc = ("on (profile)" if retry_cfg.get("enabled") else "off (profile)")
    else:
        retry_desc = f"{args.retry} (forced)"
    print(f"  retry      {retry_desc}   min_top_score {retry_cfg.get('min_top_score')}")
    print(f"  decompose  {'on — applied at rerank only, aggregate=' + args.aggregate if args.decompose else 'off'}")
    print(f"  k          {args.k}   chunk_size {args.chunk_size}")
    print(f"  enrichment OFF   overviews OFF   latechunk OFF   context-expansion OFF")
    print(f"  log        {log_path}")

    t0 = time.time()
    all_results = [evaluate_corpus(c, args, log_path) for c in corpora]
    wall_seconds = time.time() - t0

    if not args.coverage_only:
        print_table(all_results)

    payload = {
        "run": {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "wall_seconds": round(wall_seconds, 1),
            "seed": SEED,
            "platform": f"{platform.system()} {platform.machine()} python {platform.python_version()}",
            "torch": torch.__version__,
            "device": ("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available() else "cpu"),
            "embedder": args.embedder,
            "reranker": reranker_name if rerank_enabled else None,
            "retry": retry_desc,
            "retry_config": retry_cfg,
            "decompose_at_rerank": bool(args.decompose),
            "rerank_aggregate": args.aggregate if args.decompose else None,
            "k": args.k,
            "chunk_size": args.chunk_size,
            "enrichment": False,
            "overviews": False,
            "latechunk": False,
            "context_expansion": False,
            "argv": sys.argv[1:],
        },
        "summary": {r["corpus"]: summarise(r) for r in all_results if summarise(r)},
        "by_dimension": by_dimension(all_results),
        "coverage_failures": {r["corpus"]: r["coverage"] for r in all_results if r["coverage"]},
        "corpora": all_results,
    }
    with open(json_out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"\nwall clock {wall_seconds:.1f}s")
    print(f"results    {json_out}")
    if payload["by_dimension"] and not args.coverage_only:
        print("\nby dimension (recall@10 / nDCG@10 post-rerank):")
        for key, bucket in payload["by_dimension"].items():
            nd = bucket["ndcg@10_reranked"]
            print(f"  {key:<28} n={bucket['n']:<3} recall@10={bucket['recall@10']:.3f} "
                  f"nDCG@10={'n/a' if nd is None else f'{nd:.3f}'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

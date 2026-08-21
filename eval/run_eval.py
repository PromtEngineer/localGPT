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

Two candidate lists are scored, not one:

  first stage        ``retrieve_candidates()["first_stage"]`` — the retriever's
                     own ordering (``recall@k``, ``ndcg10_first_stage``).
  final              ``retrieve_candidates()["documents"]`` — post-rerank AND
                     post-cross-reference-hop, i.e. the list the answer stage
                     would actually see (``recall_final``, ``ndcg10_final``).

The distinction exists because roadmap item 4.2's cross-reference hop *appends*
to ``documents`` and deliberately never mutates ``first_stage``: scoring only
the first stage reports a flat line for the hop no matter how well it works.
With reranking off and the hop off the two lists are the same object, and the
run asserts exactly that (``final == first_stage`` invariant, printed and
recorded in the results JSON).

Gold relevance is text-based, never chunk-id-based, so the set survives
re-chunking and embedder swaps.

Usage (see eval/README.md):

    .venv/bin/python eval/run_eval.py --corpus all                     # shipped defaults
    EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B .venv/bin/python eval/run_eval.py --corpus all

The rerank stage follows the shipped "default" profile, which has reranking ON
with threshold selection since arm G (2026-08-14): ``top_k: 10`` plus
``min_score: 0.5`` / ``min_keep: 3`` pruning of whatever the Qwen scorer marks
irrelevant to every query. The harness keeps that selection when the stage is
on, so the ``final`` metrics describe the list the answer stage actually sees.
Pass ``--reranker <model>`` to swap the model, ``--no-rerank`` to force the
stage off for the first-stage-only control arm.
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

# Index-time code version, mixed into the index fingerprint. The fingerprint
# otherwise covers only file size/mtime and flags — NOT this repo's code — so a
# change to anything that alters what lands in the index (chunker, converter,
# crossref stamping, normalization…) must bump this, or cached indexes built by
# the old code are silently reused (the manual cache deletion in BASELINE.md §
# "Why the rebuild" is what this automates).
INDEX_CODE_VERSION = "2026-08-16.1"

# Decomposition-prompt version, part of the sub-query cache key. Bump when the
# QueryDecomposer prompt or its parameters change; combined with the resolved
# model name in the cache filename, stale decompositions are then never
# silently reused across models or prompt edits.
SUBQUERY_PROMPT_VERSION = "2026-08-16.1"

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
    # Roadmap Phase 4 (4.1 / 4.2 / 4.3) needs what no other corpus here has: many
    # documents that reference *each other*. Ten synthetic M&A documents whose
    # every "Document: <Title>", "Exhibit X" and "Schedule N" pointer resolves
    # inside the corpus. Its gold set carries the extra `requires_crossref`
    # dimension — the 4.2 metric.
    "acq": {
        "label": "acquisition deal room (10 interlinked synthetic M&A PDFs)",
        "glob": os.path.join(EVAL_DIR, "corpora", "acquisition", "*.pdf"),
        "goldset_of": ["acquisition"],
    },
    # The cross-reference corpus with the documentation corpus as distractors.
    # Deliberately NOT folded into `mixed`: `mixed` is the tracked Phase 0/1/2
    # baseline and must keep meaning the same thing across the whole roadmap.
    "acq+docs": {
        "label": "acquisition deal room + Documentation/*.md as distractors",
        "glob": [os.path.join(EVAL_DIR, "corpora", "acquisition", "*.pdf"),
                 os.path.join(REPO_ROOT, "Documentation", "*.md")],
        "goldset_of": ["acquisition", "docs"],
    },
    # Real third-party documents whose naming and referencing conventions this
    # project did not invent — 23 IETF RFCs (the QUIC / HTTP-3 family), fetched
    # verbatim from rfc-editor.org by corpora/rfc/download.py (plain text, 1.44
    # MiB). The honest test of the index-time crossref extractor: it resolves 0
    # of 1403 references here (corpora/rfc/MANIFEST.md,
    # decisions/rfc-shakedown-2026-08-13.md). Deliberately NOT in `mixed`, for
    # the same reason as `acq`.
    "rfc": {
        "label": "23 interlinked IETF RFCs, plain text (QUIC / HTTP-3 family)",
        "glob": os.path.join(EVAL_DIR, "corpora", "rfc", "*.txt"),
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


def corpus_slug(corpus: str) -> str:
    """Filesystem- and LanceDB-safe form of a corpus key (``acq+docs`` has a ``+``)."""
    return corpus.replace("+", "_plus_")


def corpus_files(corpus: str) -> list:
    spec = CORPORA[corpus]
    files = list(spec.get("files", []))
    patterns = spec.get("glob", [])
    if isinstance(patterns, str):
        patterns = [patterns]
    for pattern in patterns:
        files.extend(f for f in glob.glob(pattern)
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
                 aggregate: str = "mean", overviews: bool = False) -> dict:
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
    # Document overviews are off by default here for the same reason enrichment
    # is: one LLM call per document. Roadmap item 4.3's prefilter cannot be
    # measured without them, though — it scores the query against the
    # `.vectors.npz` sidecar that only an overview-enabled build writes. So
    # `--overviews on` switches them back on and, critically, redirects the
    # output *inside this corpus's eval index directory* rather than the repo's
    # shared `index_store/overviews/`: the sidecar is then owned by the index,
    # deleted with it, and can never be read by the wrong corpus.
    #
    # Overviews do not touch the `text` or `vector` columns — `build_and_store`
    # only appends a JSONL line — so an overview-enabled index is chunk-for-chunk
    # identical to one built without them.
    if overviews:
        cfg["overview"] = {"enabled": True, "embed": True}
        cfg["overview_path"] = os.path.join(db_path, "overviews.jsonl")
    else:
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
    # Keep the shipped profile's reranker block and override only the model:
    # since arm G the profile selects (min_score / min_keep) and truncates
    # (top_k), it does not just reorder, and the pipeline only applies that
    # threshold selection when min_score is present. Replacing the block with a
    # bare one would make the `final` metrics describe a reorder-only stack
    # that no longer ships — `final` is supposed to be the list the answer
    # stage actually sees.
    cfg["reranker"]["enabled"] = rerank_enabled
    cfg["reranker"]["model_name"] = reranker
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


def apply_phase4_settings(cfg: dict, crossref_mode: str, overview_mode: str) -> dict:
    """Force the two Phase-4 query-time flags, or leave the profile's values.

    Same shape as ``apply_retry_setting``: ``profile`` means "whatever
    ``main.py`` says" (both are OFF there today), anything else is a forced arm
    for an A/B. Written under ``retrieval.<key>`` because that is the container
    ``RetrievalPipeline._merged_block`` reads the profile from; the
    ``retrievers.<key>`` spelling is the API's runtime-override lane and is left
    alone so a forced arm here cannot be silently overridden.
    """
    if crossref_mode != "profile":
        block = dict(cfg["retrieval"].get("crossref_hop") or {})
        block["enabled"] = (crossref_mode == "on")
        cfg["retrieval"]["crossref_hop"] = block

    if overview_mode != "profile":
        block = dict(cfg["retrieval"].get("overview_prefilter") or {})
        block["enabled"] = (overview_mode != "off")
        if overview_mode in ("boost", "restrict"):
            block["mode"] = overview_mode
        cfg["retrieval"]["overview_prefilter"] = block
    return cfg


def index_fingerprint(corpus: str, embedder: str, chunk_size: int,
                      overviews: bool = False) -> dict:
    files = corpus_files(corpus)
    fingerprint = {
        "corpus": corpus,
        "embedder": embedder,
        "chunk_size": chunk_size,
        "enrichment": False,
        "latechunk": False,
        # Vectors are L2-normalized at write time since the Phase 1 adoption
        # (eval/DECISIONS.md). Carrying it here invalidates every index built
        # before that, exactly once, instead of silently reusing them.
        "normalized": True,
        # Code, not data: size/mtime cannot see a chunker or crossref-stamping
        # fix, so the fingerprint carries the harness's index-code version.
        "index_code_version": INDEX_CODE_VERSION,
        "files": [
            {"path": os.path.relpath(f, REPO_ROOT), "size": os.path.getsize(f),
             "mtime": round(os.path.getmtime(f), 3)}
            for f in files
        ],
    }
    # Overviews leave the chunk index bit-identical, but they are what produces
    # the `.vectors.npz` sidecar item 4.3's prefilter reads — so a cached index
    # built without them cannot serve an `--overviews on` run. Added as a key
    # only when true, so every index cached before this flag existed keeps its
    # fingerprint and is not needlessly rebuilt.
    if overviews:
        fingerprint["overviews"] = True
    return fingerprint


def ensure_index(corpus: str, embedder: str, chunk_size: int, cfg: dict, db_path: str,
                 table: str, force: bool, log_path: str, verbose: bool,
                 overviews: bool = False) -> dict:
    marker = os.path.join(db_path, f"{table}.built.json")
    fingerprint = index_fingerprint(corpus, embedder, chunk_size, overviews)

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

    `QueryDecomposer` is an LLM call, so the result is cached and reused: the
    decomposition-on and decomposition-off arms must differ only in whether the
    sub-queries are *used*, never in what they are. The cache filename names
    the corpus, the resolved decomposer model and SUBQUERY_PROMPT_VERSION, so
    switching ENRICHMENT_MODEL or editing the prompt can never silently reuse
    another run's decompositions. The first stage never sees the sub-queries —
    item 2.2's whole point is that they apply at reranking — which is also why
    a rerank-less run skips the work entirely: they would have no consumer.
    """
    if not args.decompose:
        return {}
    if not rerank_settings(args)[0]:
        print("  decompose              skipped — rerank stage is off, sub-queries "
              "would have no consumer")
        return {}
    os.makedirs(SUBQUERY_CACHE, exist_ok=True)

    from rag_system.retrieval.query_transformer import QueryDecomposer  # noqa: E402
    llm_client, llm_config = _build_llm_client()
    model = llm_config.get("enrichment_model") or llm_config["generation_model"]
    decomposer = QueryDecomposer(llm_client, model)

    path = os.path.join(
        SUBQUERY_CACHE,
        f"{corpus_slug(args.corpus_being_run)}.{slug(model)}.{SUBQUERY_PROMPT_VERSION}.json")
    cache = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            cache = json.load(fh)

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

    The shipped "default" profile has reranking ON since arm G (2026-08-14),
    with threshold selection (``min_score`` / ``min_keep`` / ``top_k``), and a
    bare eval run must measure what ships — including that selection, which
    ``build_config`` preserves. Naming a model with ``--reranker`` keeps the
    stage on but swaps the model, which is what the A/B commands in
    eval/decisions/*.md do; ``--no-rerank`` always wins.
    """
    profile_default = bool(
        get_pipeline_config("default").get("reranker", {}).get("enabled", False)
    )
    enabled = (not args.no_rerank) and (args.reranker is not None or profile_default)
    return enabled, (args.reranker or EXTERNAL_MODELS["reranker_model"])


def evaluate_corpus(corpus: str, args, log_path: str) -> dict:
    embedder = args.embedder
    rerank_enabled, reranker_name = rerank_settings(args)
    # An overview-enabled build lives in its own directory rather than replacing
    # the tracked one. The chunk index is identical either way (overviews only
    # append a JSONL line), so sharing a directory would work — but it would make
    # every alternation between `--overviews off` and `--overviews on` a full
    # re-index, and would silently blow away the index another run is reading.
    index_dir = corpus_slug(corpus) + ("_ov" if args.overviews == "on" else "")
    db_path = os.path.join(INDEX_ROOT, slug(embedder), index_dir)
    table = f"eval_{corpus_slug(corpus)}"

    args.corpus_being_run = corpus
    print(f"\n=== corpus: {corpus} — {CORPORA[corpus]['label']}")
    overviews = (args.overviews == "on")
    cfg = build_config(corpus, embedder, reranker_name, db_path, table,
                       args.k, args.chunk_size, rerank_enabled, args.aggregate,
                       overviews=overviews)
    cfg = apply_retry_setting(cfg, args.retry)
    cfg = apply_phase4_settings(cfg, args.crossref_hop, args.overview_prefilter)
    index_record = ensure_index(corpus, embedder, args.chunk_size, cfg, db_path, table,
                                args.force_reindex, log_path, args.verbose,
                                overviews=overviews)

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

        # The list the answer stage would actually see: post-rerank AND
        # post-crossref-hop. Scoring only `first_stage` is what made roadmap 4.2
        # unmeasurable — the hop appends to `documents` and never touches
        # `first_stage` by design (retrieval_pipeline.py `_crossref_hop`).
        final_docs = out.get("documents") or []
        final_texts = [d.get("text", "") for d in final_docs]
        # The hop only ever *appends*, so removing the tagged rows reconstructs
        # the post-rerank / pre-hop list exactly. That keeps `ndcg10_reranked`
        # meaning what it has always meant across every earlier decision file.
        pre_hop_docs = [d for d in final_docs if not d.get("via_crossref")]
        hopped_docs = [d for d in final_docs if d.get("via_crossref")]

        entry = {
            "id": row["id"],
            "query": query,
            "dimensions": row.get("dimensions", {}),
            "match": match,
            "expected": expected,
            "candidates": len(docs),
            "final_candidates": len(final_docs),
            "first_stage_ms": round(total_ms, 1),
            "recall": {f"@{k}": query_hit(first_texts[:k], expected, match) for k in RECALL_KS},
            "ndcg10_first_stage": round(
                ndcg_at_k([relevance(t, expected) for t in first_texts], NDCG_K), 4),
            "recall_final": {f"@{k}": query_hit(final_texts[:k], expected, match)
                             for k in RECALL_KS},
            "ndcg10_final": round(
                ndcg_at_k([relevance(t, expected) for t in final_texts], NDCG_K), 4),
        }
        # The flags-off invariant, checked per query rather than argued: with no
        # rerank and no hop the final list must BE the first-stage list.
        entry["final_equals_first_stage"] = (
            [d.get("chunk_id") for d in final_docs] == [d.get("chunk_id") for d in docs])

        # Hop instrumentation: rank movement alone cannot say whether the hop
        # pulled the *right* document, so record both what it pulled and
        # whether any of it was on target.
        entry["crossref_chunks_in_final"] = len(hopped_docs)
        if hopped_docs:
            expected_sources = row.get("expected_sources") or []
            hop_docs = sorted({d.get("document_id") for d in hopped_docs if d.get("document_id")})
            entry["crossref_documents"] = hop_docs
            # precision, document level: did a hop land in a gold source document?
            entry["crossref_hit_expected_source"] = bool(
                expected_sources and any(d in expected_sources for d in hop_docs))
            # precision, text level: does a hopped chunk actually carry gold text?
            entry["crossref_chunk_relevant"] = bool(
                any(relevance(d.get("text", ""), expected) for d in hopped_docs))
            # rank of the first relevant hopped chunk inside the final list
            rels = [i for i, t in enumerate(final_texts) if relevance(t, expected)]
            entry["first_relevant_rank_final"] = (rels[0] + 1) if rels else None
        if out.get("crossref_hop"):
            entry["crossref_hop"] = out["crossref_hop"]
        if out.get("retry"):
            entry["retry"] = out["retry"]
        if decompose.get(row["id"]):
            entry["sub_queries"] = decompose[row["id"]]

        if rerank_enabled and docs:
            entry["rerank_ms"] = 0.0  # folded into first_stage_ms on this path
            # Identity, not equality: the rerank stage signals failure by
            # handing its input straight back. (The hop rebuilds `documents` as
            # a new list, so `is` on the list itself is no longer sufficient —
            # the element identities are.)
            rerank_noop = (len(pre_hop_docs) == len(docs)
                           and all(a is b for a, b in zip(pre_hop_docs, docs)))
            if not pre_hop_docs or rerank_noop:
                entry["rerank_error"] = "reranker failed to load; see the run log"
            else:
                rr_texts = [d.get("text", "") for d in pre_hop_docs]
                entry["ndcg10_reranked"] = round(
                    ndcg_at_k([relevance(t, expected) for t in rr_texts], NDCG_K), 4)
                entry["recall_reranked"] = {
                    f"@{k}": query_hit(rr_texts[:k], expected, match) for k in RECALL_KS
                }
        results.append(entry)
        rr_val = entry.get("ndcg10_reranked")
        rr_cell = "n/a  " if rr_val is None else f"{rr_val:.3f}"
        hop_cell = (f" hop=+{entry['crossref_chunks_in_final']}"
                    f"{'✓' if entry.get('crossref_chunk_relevant') else ''}"
                    if entry["crossref_chunks_in_final"] else "")
        print(f"  [{entry['id']}] r@5={entry['recall']['@5']} r@10={entry['recall']['@10']} "
              f"r@20={entry['recall']['@20']} "
              f"nDCG@10 first={entry['ndcg10_first_stage']:.3f} rerank={rr_cell} "
              f"final={entry['ndcg10_final']:.3f}{hop_cell} "
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
        # Final = the list the answer stage sees: post-rerank, post-crossref-hop.
        "recall@5_final": round(mean([q["recall_final"]["@5"] for q in qs]), 4),
        "recall@10_final": round(mean([q["recall_final"]["@10"] for q in qs]), 4),
        "recall@20_final": round(mean([q["recall_final"]["@20"] for q in qs]), 4),
        "ndcg@10_final": round(mean([q["ndcg10_final"] for q in qs]), 4),
        "final_equals_first_stage": all(q.get("final_equals_first_stage") for q in qs),
        "first_stage_ms_mean": round(mean([q["first_stage_ms"] for q in qs]), 1),
        "first_stage_ms_p90": round(sorted(q["first_stage_ms"] for q in qs)[int(0.9 * (len(qs) - 1))], 1),
    }
    fired = [q for q in qs if q.get("retry")]
    if fired:
        kept = [q for q in fired if q["retry"]["kept"] == "retry"]
        summary["retry_fired"] = len(fired)
        summary["retry_fire_rate"] = round(len(fired) / len(qs), 4)
        summary["retry_kept"] = len(kept)
    hopped = [q for q in qs if q.get("crossref_chunks_in_final")]
    if hopped:
        summary["crossref_hop"] = _hop_summary(qs, hopped)
    reranked = [q for q in qs if "ndcg10_reranked" in q]
    if reranked:
        summary["ndcg@10_reranked"] = round(mean([q["ndcg10_reranked"] for q in reranked]), 4)
        summary["rerank_ms_mean"] = round(mean([q["rerank_ms"] for q in reranked]), 1)
        summary["rerank_ms_p90"] = round(
            sorted(q["rerank_ms"] for q in reranked)[int(0.9 * (len(reranked) - 1))], 1)
    # Roadmap 4.2's metric: the rows whose answer text lives in a document other
    # than the one the query's premise points at. Reported next to the whole-corpus
    # numbers rather than only in `by_dimension`, because the point of the slice is
    # the *gap* between it and the rest of the same corpus.
    crossref = [q for q in qs if q["dimensions"].get("requires_crossref") is True]
    if crossref:
        rest = [q for q in qs if q["dimensions"].get("requires_crossref") is False]
        summary["crossref"] = _slice_summary(crossref)
        if rest:
            summary["crossref_control"] = _slice_summary(rest)
    return summary


def _hop_summary(qs: list, hopped: list) -> dict:
    """Did the hop fire, and did what it pulled belong there?

    ``fire_rate`` is over *all* queries in the slice; the precision numbers are
    over the queries that actually hopped, because a query that never hopped is
    not evidence about hop precision either way.
    """
    return {
        "queries_with_hop": len(hopped),
        "fire_rate": round(len(hopped) / len(qs), 4) if qs else 0.0,
        "chunks_added_total": sum(q["crossref_chunks_in_final"] for q in hopped),
        "chunks_added_mean_when_fired": round(
            mean([q["crossref_chunks_in_final"] for q in hopped]), 2),
        # document-level precision: the hop landed in a gold source document
        "hit_expected_source": sum(1 for q in hopped
                                   if q.get("crossref_hit_expected_source")),
        # text-level precision: a hopped chunk actually carries the gold text
        "hopped_chunk_relevant": sum(1 for q in hopped
                                     if q.get("crossref_chunk_relevant")),
    }


def _slice_summary(qs: list) -> dict:
    out = {
        "n_queries": len(qs),
        "recall@5": round(mean([q["recall"]["@5"] for q in qs]), 4),
        "recall@10": round(mean([q["recall"]["@10"] for q in qs]), 4),
        "recall@20": round(mean([q["recall"]["@20"] for q in qs]), 4),
        "ndcg@10_first_stage": round(mean([q["ndcg10_first_stage"] for q in qs]), 4),
        "recall@5_final": round(mean([q["recall_final"]["@5"] for q in qs]), 4),
        "recall@10_final": round(mean([q["recall_final"]["@10"] for q in qs]), 4),
        "recall@20_final": round(mean([q["recall_final"]["@20"] for q in qs]), 4),
        "ndcg@10_final": round(mean([q["ndcg10_final"] for q in qs]), 4),
    }
    reranked = [q for q in qs if "ndcg10_reranked" in q]
    if reranked:
        out["ndcg@10_reranked"] = round(mean([q["ndcg10_reranked"] for q in reranked]), 4)
    hopped = [q for q in qs if q.get("crossref_chunks_in_final")]
    if hopped:
        out["crossref_hop"] = _hop_summary(qs, hopped)
    return out


# `requires_crossref` only exists on the `acq` gold set; rows without it are
# skipped, so adding the corpus cannot move any pre-existing slice.
DIMENSION_KEYS = ("question_type", "difficulty", "requires_crossref")


def by_dimension(all_results: list) -> dict:
    buckets = {}
    for corpus_result in all_results:
        for q in corpus_result["queries"]:
            for key in DIMENSION_KEYS:
                value = q["dimensions"].get(key)
                if value is None:
                    continue
                if isinstance(value, bool):
                    value = "true" if value else "false"
                bucket = buckets.setdefault(f"{key}={value}",
                                            {"n": 0, "r@10": 0.0, "ndcg1": 0.0,
                                             "ndcg": 0.0, "ndcg_n": 0,
                                             "r@10f": 0.0, "ndcgf": 0.0, "hops": 0})
                bucket["n"] += 1
                bucket["r@10"] += q["recall"]["@10"]
                bucket["ndcg1"] += q["ndcg10_first_stage"]
                bucket["r@10f"] += q["recall_final"]["@10"]
                bucket["ndcgf"] += q["ndcg10_final"]
                if q.get("crossref_chunks_in_final"):
                    bucket["hops"] += 1
                if "ndcg10_reranked" in q:
                    bucket["ndcg"] += q["ndcg10_reranked"]
                    bucket["ndcg_n"] += 1
    for bucket in buckets.values():
        bucket["recall@10"] = round(bucket.pop("r@10") / bucket["n"], 4)
        bucket["ndcg@10_first_stage"] = round(bucket.pop("ndcg1") / bucket["n"], 4)
        bucket["recall@10_final"] = round(bucket.pop("r@10f") / bucket["n"], 4)
        bucket["ndcg@10_final"] = round(bucket.pop("ndcgf") / bucket["n"], 4)
        bucket["queries_with_crossref_hop"] = bucket.pop("hops")
        ndcg_n = bucket.pop("ndcg_n")
        total = bucket.pop("ndcg")
        bucket["ndcg@10_reranked"] = round(total / ndcg_n, 4) if ndcg_n else None
    return dict(sorted(buckets.items()))


def print_table(all_results: list) -> None:
    header = (f"{'corpus':<12} {'n':>4} {'chunks':>7} {'R@5':>7} {'R@10':>7} {'R@20':>7} "
              f"{'nDCG@10':>9} {'nDCG@10':>9} | {'R@5':>7} {'R@10':>7} {'R@20':>7} "
              f"{'nDCG@10':>9} {'hop q':>6} {'1st ms':>8}")
    print("\n" + "=" * len(header))
    print(header)
    print(f"{'':<12} {'':>4} {'':>7} {'':>7} {'':>7} {'':>7} {'(1st)':>9} {'(rerank)':>9} | "
          f"{'(fin)':>7} {'(fin)':>7} {'(fin)':>7} {'(final)':>9} {'':>6} {'':>8}")
    print("-" * len(header))
    for corpus_result in all_results:
        s = summarise(corpus_result)
        if not s:
            continue
        rr = s.get("ndcg@10_reranked")
        rr_cell = "n/a" if rr is None else f"{rr:.3f}"
        hop = s.get("crossref_hop") or {}
        hop_cell = str(hop.get("queries_with_hop", 0))
        print(f"{corpus_result['corpus']:<12} {s['n_queries']:>4} "
              f"{corpus_result['index']['chunks']:>7} "
              f"{s['recall@5']:>7.3f} {s['recall@10']:>7.3f} {s['recall@20']:>7.3f} "
              f"{s['ndcg@10_first_stage']:>9.3f} {rr_cell:>9} | "
              f"{s['recall@5_final']:>7.3f} {s['recall@10_final']:>7.3f} "
              f"{s['recall@20_final']:>7.3f} {s['ndcg@10_final']:>9.3f} {hop_cell:>6} "
              f"{s['first_stage_ms_mean']:>8.0f}")
        for label, key in (("  ├ crossref", "crossref"), ("  └ control ", "crossref_control")):
            sl = s.get(key)
            if not sl:
                continue
            sl_rr = sl.get("ndcg@10_reranked")
            sl_hop = sl.get("crossref_hop") or {}
            print(f"{label:<12} {sl['n_queries']:>4} {'':>7} "
                  f"{sl['recall@5']:>7.3f} {sl['recall@10']:>7.3f} {sl['recall@20']:>7.3f} "
                  f"{sl['ndcg@10_first_stage']:>9.3f} "
                  f"{('n/a' if sl_rr is None else f'{sl_rr:.3f}'):>9} | "
                  f"{sl['recall@5_final']:>7.3f} {sl['recall@10_final']:>7.3f} "
                  f"{sl['recall@20_final']:>7.3f} {sl['ndcg@10_final']:>9.3f} "
                  f"{str(sl_hop.get('queries_with_hop', 0)):>6} {'':>8}")
    print("=" * len(header))
    print("left of the bar: first stage (and rerank).  right of the bar: the FINAL "
          "candidate list\n(post-rerank, post-crossref-hop) — what the answer stage "
          "would actually see.")


def check_final_invariant(all_results: list, rerank_enabled: bool, hop_desc: str) -> dict:
    """With reranking and the hop both off, the final list must BE the first stage.

    This is the guard that makes the new ``*_final`` metrics trustworthy: if
    they ever diverge from the first-stage metrics on a run where nothing is
    allowed to reorder or append, the metric is measuring its own bug rather
    than the pipeline.
    """
    applicable = (not rerank_enabled) and hop_desc.startswith("off")
    offenders = []
    for corpus_result in all_results:
        for q in corpus_result["queries"]:
            same_list = q.get("final_equals_first_stage")
            same_metrics = (q["ndcg10_final"] == q["ndcg10_first_stage"]
                            and q["recall_final"] == q["recall"])
            if not (same_list and same_metrics):
                offenders.append(f"{corpus_result['corpus']}/{q['id']}")
    report = {"applicable": applicable, "offenders": offenders,
              "checked": sum(len(r["queries"]) for r in all_results)}
    if not applicable:
        print(f"\ninvariant  final == first_stage: not applicable "
              f"(rerank={'on' if rerank_enabled else 'off'}, crossref hop={hop_desc}); "
              f"{len(offenders)}/{report['checked']} queries have a final list that "
              f"differs from the first stage.")
        return report
    if offenders:
        print(f"\ninvariant  ❌ FAIL — final != first_stage on "
              f"{len(offenders)}/{report['checked']} queries with rerank OFF and hop OFF: "
              f"{', '.join(offenders[:10])}")
    else:
        print(f"\ninvariant  ✅ final == first_stage on all {report['checked']} queries "
              f"(rerank OFF, crossref hop OFF) — chunk-id order and both metrics")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", default="all", choices=["all", *sorted(CORPORA)],
                        help="corpus to evaluate (default: all)")
    parser.add_argument("--embedder", default=EXTERNAL_MODELS["embedding_model"],
                        help="HF embedding model; defaults to EMBEDDING_MODEL / the repo default")
    parser.add_argument("--reranker", default=None,
                        help="reranker model name; defaults to the shipped profile's "
                             "reranker model (the stage itself follows the profile — "
                             "ON since arm G — unless --no-rerank is given)")
    parser.add_argument("--k", type=int, default=20, help="first-stage candidates per query")
    parser.add_argument("--chunk-size", type=int, default=512,
                        help="token budget per chunk (512 = what the HTTP path sends)")
    parser.add_argument("--no-rerank", action="store_true", help="skip the cross-encoder stage")
    parser.add_argument("--retry", choices=["profile", "on", "off"], default="profile",
                        help="evidence-sufficiency retry: follow the shipped profile "
                             "(default), or force it on/off for an A/B")
    parser.add_argument("--crossref-hop", choices=["profile", "on", "off"], default="profile",
                        help="cross-reference hop (roadmap 4.2): follow the shipped "
                             "profile (default — currently OFF), or force it on/off. "
                             "Only the FINAL metrics can move; the hop never touches "
                             "the first stage.")
    parser.add_argument("--overview-prefilter", choices=["profile", "off", "boost", "restrict"],
                        default="profile",
                        help="overview prefilter (roadmap 4.3): follow the shipped "
                             "profile (default — currently OFF), force it off, or "
                             "enable it in boost / restrict mode")
    parser.add_argument("--overviews", choices=["off", "on"], default="off",
                        help="build per-document overviews at index time (one LLM "
                             "call per document) and the .vectors.npz sidecar the "
                             "overview prefilter reads. OFF by default, like every "
                             "other LLM stage here. Required for any "
                             "--overview-prefilter boost/restrict arm to do anything; "
                             "changes the index fingerprint, so it forces one rebuild. "
                             "Chunk text and vectors are unaffected.")
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
    profile_retrieval = get_pipeline_config("default").get("retrieval", {})
    hop_cfg = profile_retrieval.get("crossref_hop") or {}
    pre_cfg = profile_retrieval.get("overview_prefilter") or {}
    if args.crossref_hop == "profile":
        hop_desc = ("on (profile)" if hop_cfg.get("enabled") else "off (profile)")
    else:
        hop_desc = f"{args.crossref_hop} (forced)"
    if args.overview_prefilter == "profile":
        pre_desc = (f"{pre_cfg.get('mode', 'boost')} (profile)"
                    if pre_cfg.get("enabled") else "off (profile)")
    else:
        pre_desc = f"{args.overview_prefilter} (forced)"
    print(f"  crossref   {hop_desc}   max_hops {hop_cfg.get('max_hops')} "
          f"chunks_per_hop {hop_cfg.get('chunks_per_hop')}")
    print(f"  prefilter  {pre_desc}   top_documents {pre_cfg.get('top_documents')}")
    if args.decompose and not rerank_enabled:
        decompose_desc = "skipped — rerank stage is off"
    elif args.decompose:
        decompose_desc = "on — applied at rerank only, aggregate=" + args.aggregate
    else:
        decompose_desc = "off"
    print(f"  decompose  {decompose_desc}")
    print(f"  k          {args.k}   chunk_size {args.chunk_size}")
    print(f"  enrichment OFF   overviews {args.overviews.upper()}   latechunk OFF   "
          f"context-expansion OFF")
    print(f"  log        {log_path}")

    t0 = time.time()
    all_results = [evaluate_corpus(c, args, log_path) for c in corpora]
    wall_seconds = time.time() - t0

    invariant = None
    if not args.coverage_only:
        print_table(all_results)
        invariant = check_final_invariant(all_results, rerank_enabled, hop_desc)

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
            "crossref_hop": hop_desc,
            "crossref_hop_config": hop_cfg,
            "overview_prefilter": pre_desc,
            "overview_prefilter_config": pre_cfg,
            "decompose_at_rerank": bool(args.decompose) and rerank_enabled,
            "rerank_aggregate": (args.aggregate
                                 if args.decompose and rerank_enabled else None),
            "k": args.k,
            "chunk_size": args.chunk_size,
            "enrichment": False,
            "overviews": (args.overviews == "on"),
            "latechunk": False,
            "context_expansion": False,
            "argv": sys.argv[1:],
        },
        "summary": {r["corpus"]: summarise(r) for r in all_results if summarise(r)},
        "by_dimension": by_dimension(all_results),
        # Same slices, not pooled across corpora — a corpus that appears twice in
        # one run (e.g. `acq` and `acq+docs`) would otherwise double-count its rows.
        "by_dimension_per_corpus": {r["corpus"]: by_dimension([r])
                                    for r in all_results if r["queries"]},
        "final_vs_first_stage_invariant": invariant,
        "coverage_failures": {r["corpus"]: r["coverage"] for r in all_results if r["coverage"]},
        "corpora": all_results,
    }
    with open(json_out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"\nwall clock {wall_seconds:.1f}s")
    print(f"results    {json_out}")
    if payload["by_dimension"] and not args.coverage_only:
        print("\nby dimension, pooled over the corpora in this run "
              "(recall@10 / nDCG@10 first stage / nDCG@10 post-rerank / final):")
        for key, bucket in payload["by_dimension"].items():
            nd = bucket["ndcg@10_reranked"]
            print(f"  {key:<28} n={bucket['n']:<3} recall@10={bucket['recall@10']:.3f} "
                  f"nDCG@10(1st)={bucket['ndcg@10_first_stage']:.3f} "
                  f"nDCG@10(rr)={'n/a' if nd is None else f'{nd:.3f}'} "
                  f"recall@10(fin)={bucket['recall@10_final']:.3f} "
                  f"nDCG@10(fin)={bucket['ndcg@10_final']:.3f}")
        for corpus, buckets in payload["by_dimension_per_corpus"].items():
            crossref = {k: v for k, v in buckets.items() if k.startswith("requires_crossref=")}
            if not crossref:
                continue
            print(f"\n  {corpus} — roadmap 4.2 slice:")
            for key, bucket in sorted(crossref.items()):
                print(f"    {key:<26} n={bucket['n']:<3} recall@10={bucket['recall@10']:.3f} "
                      f"nDCG@10(1st)={bucket['ndcg@10_first_stage']:.3f} "
                      f"| recall@10(fin)={bucket['recall@10_final']:.3f} "
                      f"nDCG@10(fin)={bucket['ndcg@10_final']:.3f} "
                      f"hops={bucket['queries_with_crossref_hop']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

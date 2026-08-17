"""Multi-turn end-to-end gate for the conversational decomposer (arms m0–m1d).

The runnable half of the "Rules going forward" in
eval/decisions/multiturn-decomposer-2026-08-16.md: any change to the
multi-turn decomposer prompt gates on eval/goldset/multiturn.jsonl here, plus
the single-turn byte-identity check in eval/multiturn/decomp_stability.py.

Per the decision doc's contract, each conversation is executed sequentially
through ``Agent.run`` with a real ``session_id`` — turn 2 sees whatever the
system actually answered (in-process chat history plus the session-scoped
semantic cache) — and ONLY the final turn's answer is graded: case-insensitive
substring check of the row's ``expected`` strings, "any" (at least one
present) or "all" (every one) per the row's ``match`` field. One Agent serves
the whole run (``table_name`` is passed per ``Agent.run`` call); each
conversation gets a fresh ``session_id`` so history never leaks across rows.

The index per corpus is the SAME cached LanceDB index the single-turn harness
builds (``run_eval.ensure_index`` and its fingerprint), and the config is the
shipped "default" profile via ``run_eval.build_config`` — reranker ON with
arm-G threshold selection — with one deliberate exception: ``build_config``
disables ``query_decomposition`` because ``run_eval`` never calls
``Agent.run()``. This gate does, so the profile's decomposition block (arm H:
enabled, pooled_first_stage) is restored. Everything else ``build_config``
turns off (enrichment, latechunk, context expansion, verification) stays off:
those are index-shape decisions the cached indexes were built to match.

Usage (see eval/README.md):

    .venv/bin/python eval/multiturn/run_e2e_multiturn.py --json-out eval/results/mt_answers.jsonl
    .venv/bin/python eval/multiturn/run_e2e_multiturn.py --ids mt_01,mt_07 --verbose

Needs Ollama running (triage, decomposition and synthesis are LLM calls);
exits 2 naming the required models when Ollama cannot be reached or does not
have them. Exit status is 0 iff every selected row passes, 1 otherwise.
"""

import argparse
import json
import os
import sys
import types
import uuid
from datetime import datetime, timezone

MT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.abspath(os.path.join(MT_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(EVAL_DIR, ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, EVAL_DIR)

import httpx  # noqa: E402
import requests  # noqa: E402

import run_eval  # noqa: E402
from rag_system.agent.loop import Agent  # noqa: E402
from rag_system.factory import _build_llm_client, get_pipeline_config  # noqa: E402
from rag_system.main import EXTERNAL_MODELS  # noqa: E402

GOLDSET_NAME = "multiturn"
K = 20           # run_eval's default; also the shipped profile's retrieval_k
CHUNK_SIZE = 512  # what the HTTP path sends (run_eval default)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def grade_final_answer(answer: str, expected: list, match: str) -> bool:
    """Case-insensitive substring check of the final answer, "any"/"all".

    Exactly ``run_eval.query_hit`` over a one-text list, so the gate grades
    with the same normalisation the retrieval harness scores with.
    """
    return bool(run_eval.query_hit([answer or ""], expected, match))


def require_ollama(llm_config: dict, models: list) -> None:
    """Probe Ollama before doing any work; exit 2 when it is not usable.

    ``OllamaClient`` swallows connection errors into ``{}`` on both its sync
    and async paths, so a missing server or an unpulled model would otherwise
    surface as silently degraded answers (the decomposer falls back to the
    raw query), not as an error.
    """
    host = llm_config["host"]
    needed = sorted(set(models))
    try:
        resp = requests.get(f"{host}/api/tags", timeout=5)
        resp.raise_for_status()
        available = {m.get("name", "") for m in resp.json().get("models", [])}
    except Exception:
        print(f"ERROR: cannot reach Ollama at {host} — this gate needs Ollama "
              f"running with: {', '.join(needed)}.", file=sys.stderr)
        sys.exit(2)
    missing = [m for m in needed
               if m not in available and f"{m}:latest" not in available]
    if missing:
        print(f"ERROR: Ollama at {host} has no model(s): {', '.join(missing)} — "
              f"this gate needs: {', '.join(needed)} (`ollama pull` the missing "
              f"ones first).", file=sys.stderr)
        sys.exit(2)


def corpus_index(corpus: str, embedder: str, force: bool, log_path: str,
                 verbose: bool) -> tuple:
    """Build or reuse the same cached index ``run_eval`` would; return (cfg, table).

    The config is ``build_config``'s shipped-default profile with the arm-H
    decomposition block restored (see the module docstring), so the gate
    measures what ships. ``ensure_index``'s fingerprint keys the cache, so a
    cached index built by ``run_eval.py --corpus <corpus>`` is reused as-is.
    """
    db_path = os.path.join(run_eval.INDEX_ROOT, run_eval.slug(embedder),
                           run_eval.corpus_slug(corpus))
    table = f"eval_{run_eval.corpus_slug(corpus)}"
    # rerank_settings on the shipped profile: reranker ON, profile model.
    rerank_enabled, reranker_name = run_eval.rerank_settings(
        types.SimpleNamespace(no_rerank=False, reranker=None))
    cfg = run_eval.build_config(corpus, embedder, reranker_name, db_path, table,
                                K, CHUNK_SIZE, rerank_enabled)
    cfg["query_decomposition"] = get_pipeline_config("default")["query_decomposition"]
    run_eval.ensure_index(corpus, embedder, CHUNK_SIZE, cfg, db_path, table,
                          force, log_path, verbose)
    return cfg, db_path, table


def point_agent_at(agent: Agent, db_path: str, table: str) -> None:
    """Re-point the agent's pipeline at another corpus's LanceDB path + table.

    The pipeline caches the LanceDB manager (and the dense retriever bound to
    it) on first use, so switching corpora means dropping both; they rebuild
    lazily on the next query. Same reset pattern as
    ``RetrievalPipeline.update_embedding_model``. ``storage`` is mutated in
    place because ``pipeline.storage_config`` is a reference to the same dict.
    """
    pipeline = agent.retrieval_pipeline
    storage = pipeline.config["storage"]
    storage.pop("db_path", None)
    storage["lancedb_uri"] = db_path
    storage["text_table_name"] = table
    pipeline.db_manager = None
    pipeline.dense_retriever = None
    pipeline._dense_retriever_error = None


def run_row(agent: Agent, row: dict, table: str, log_path: str,
            verbose: bool) -> dict:
    """Execute the row's turns in one session; grade the final turn's answer."""
    session_id = f"eval-mt-{row['id']}-{uuid.uuid4().hex[:8]}"
    result = None
    for turn in row["turns"]:
        with run_eval.captured(log_path, verbose):
            result = agent.run(turn, table_name=table, session_id=session_id)
    answer = (result or {}).get("answer", "")
    passed = grade_final_answer(answer, row["expected"], row.get("match", "any"))
    return {"id": row["id"], "corpus": row["corpus"], "class": row["class"],
            "pass": passed, "final_answer": answer, "expected": row["expected"]}


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json-out", default=None,
                        help="write one JSONL result row per conversation to this path")
    parser.add_argument("--ids", default=None,
                        help="comma-separated subset of conversation ids, e.g. mt_01,mt_07")
    parser.add_argument("--verbose", action="store_true",
                        help="let the pipeline print to stdout and show final answers")
    parser.add_argument("--force-reindex", action="store_true",
                        help="rebuild the cached corpus indexes even if fingerprints match")
    args = parser.parse_args()

    rows = run_eval._read_gold_file(GOLDSET_NAME)
    if args.ids:
        wanted = [i.strip() for i in args.ids.split(",") if i.strip()]
        by_id = {row["id"]: row for row in rows}
        unknown = [i for i in wanted if i not in by_id]
        if unknown:
            parser.error(f"unknown id(s): {', '.join(unknown)} "
                         f"(have: {', '.join(sorted(by_id))})")
        rows = [by_id[i] for i in wanted]
    rows = sorted(rows, key=lambda r: r["id"])

    run_eval.seed_everything()
    os.makedirs(run_eval.RESULTS_DIR, exist_ok=True)
    os.makedirs(run_eval.INDEX_ROOT, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = os.path.join(run_eval.RESULTS_DIR, f"multiturn_{stamp}.log")

    embedder = EXTERNAL_MODELS["embedding_model"]
    llm_client, llm_config = _build_llm_client()
    require_ollama(llm_config, [llm_config["generation_model"],
                                llm_config.get("enrichment_model")
                                or llm_config["generation_model"]])

    print("localGPT multi-turn E2E gate")
    print(f"  goldset    {GOLDSET_NAME}.jsonl — {len(rows)} conversation(s)")
    print(f"  embedder   {embedder}")
    print(f"  log        {log_path}")

    by_corpus = {}
    for row in rows:
        by_corpus.setdefault(row["corpus"], []).append(row)

    agent = None
    results = []
    for corpus in sorted(by_corpus):
        print(f"\n=== corpus: {corpus} — {run_eval.CORPORA[corpus]['label']}")
        cfg, db_path, table = corpus_index(corpus, embedder, args.force_reindex,
                                           log_path, args.verbose)
        if agent is None:
            # One Agent for the whole run; table_name goes per Agent.run call.
            agent = Agent(pipeline_configs=cfg, llm_client=llm_client,
                          ollama_config=llm_config)
        point_agent_at(agent, db_path, table)
        for row in by_corpus[corpus]:
            try:
                record = run_row(agent, row, table, log_path, args.verbose)
            except (requests.exceptions.RequestException, httpx.HTTPError) as e:
                print(f"ERROR: Ollama became unreachable mid-run ({e}) — needs "
                      f"Ollama running with {llm_config['generation_model']} / "
                      f"{llm_config.get('enrichment_model')}.", file=sys.stderr)
                return 2
            results.append(record)
            print(f"  [{record['id']}] {'PASS' if record['pass'] else 'FAIL'}")
            if args.verbose:
                print(f"       answer: {record['final_answer']}")

    passed = sum(1 for r in results if r["pass"])
    print(f"\n{passed}/{len(results)} passed")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            for record in sorted(results, key=lambda r: r["id"]):
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"results    {args.json_out}")

    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())

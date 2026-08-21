"""Decomposition stability dump — the single-turn byte-identity gate.

The cheap first check from the "Rules going forward" in
eval/decisions/multiturn-decomposer-2026-08-16.md: the single-turn decomposer
prompt is a frozen measured artifact (arm L measured even cosmetic edits
shifting temp-0 decompositions), so ANY byte change to it must show up here
before it costs a full 5-bench gate.

Dumps the temp-0 decompositions of all 120 single-turn gold queries
(goldset/{atlas7,hr,docs,acquisition,rfc}.jsonl, 24 rows each) through
``QueryDecomposer.decompose(query, [])`` — empty history, i.e. the frozen
single-turn prompt path; temperature 0 is pinned in the decomposer itself.
The dump is perfectly deterministic run-to-run (120/120 byte-identical on a
repeat run, per the decision doc), so a diff against the baseline is a real
effect of a prompt or model change, never noise.

Each dump row: {id, corpus, query, resolved_query, sub_queries}.
``resolved_query`` is the single sub-query when there is exactly one — the
prompt's output rule 2 guarantees that entry IS the resolved query — else
null.

Usage (see eval/README.md):

    # (re)generate the baseline (the original scratchpad one is lost —
    # the first --write-baseline run recreates it)
    .venv/bin/python eval/multiturn/decomp_stability.py --write-baseline

    # the gate: byte-compare a fresh dump against the baseline
    .venv/bin/python eval/multiturn/decomp_stability.py --check

Both flags take an optional path (default eval/multiturn/decomp_dump_pre.jsonl).
Needs Ollama running with the enrichment model; exits 2 when it cannot be
reached. Exit status of --check is 0 iff the fresh dump is byte-identical to
the baseline, 1 otherwise (differing ids are listed).
"""

import argparse
import json
import os
import sys

MT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.abspath(os.path.join(MT_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(EVAL_DIR, ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, EVAL_DIR)

import requests  # noqa: E402

import run_eval  # noqa: E402
from rag_system.factory import _build_llm_client  # noqa: E402
from rag_system.retrieval.query_transformer import QueryDecomposer  # noqa: E402

# The five single-turn gold sets of record, 24 rows each = 120 queries.
SINGLE_TURN_GOLDSETS = ("atlas7", "hr", "docs", "acquisition", "rfc")
DEFAULT_DUMP_PATH = os.path.join(MT_DIR, "decomp_dump_pre.jsonl")


# --------------------------------------------------------------------------
# dump
# --------------------------------------------------------------------------

def load_single_turn_queries() -> list:
    """All 120 single-turn gold rows, sorted by id for a canonical dump order."""
    rows = [row for name in SINGLE_TURN_GOLDSETS
            for row in run_eval._read_gold_file(name)]
    return sorted(rows, key=lambda r: r["id"])


def build_decomposer() -> tuple:
    """The decomposer exactly as run_eval constructs it (decomposer = utility model)."""
    llm_client, llm_config = _build_llm_client()
    model = llm_config.get("enrichment_model") or llm_config["generation_model"]
    return QueryDecomposer(llm_client, model), llm_config, model


def dump_records(decomposer: QueryDecomposer, rows: list) -> list:
    """Temp-0 single-turn decomposition of every gold row, as dump records."""
    records = []
    for row in rows:
        # Empty chat history -> the frozen single-turn prompt path.
        sub_queries = decomposer.decompose(row["query"], [], max_sub_queries=10)
        records.append({
            "id": row["id"],
            "corpus": row.get("corpus"),
            "query": row["query"],
            # Output rule 2: no decomposition => the one sub-query IS the
            # resolved query. With >1 sub-queries the resolved form is the
            # decomposer's internal intermediate, so record null rather than
            # guess.
            "resolved_query": sub_queries[0] if len(sub_queries) == 1 else None,
            "sub_queries": sub_queries,
        })
    return records


def serialize(records: list) -> bytes:
    """Canonical dump bytes: sorted keys, one JSON object per line."""
    return "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
                   for r in records).encode("utf-8")


def parse_dump(data: bytes) -> dict:
    """id -> record, for the human-readable diff report after a byte mismatch."""
    out = {}
    for line in data.decode("utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            out[record["id"]] = record
    return out


def require_ollama(llm_config: dict, models: list) -> None:
    """Probe Ollama before doing any work; exit 2 when it is not usable.

    ``OllamaClient`` swallows connection errors into ``{}``, and the
    decomposer's fail-open then silently returns the raw query — a dead server
    would otherwise WRITE a degraded baseline with no error at all.
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


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write-baseline", nargs="?", const=DEFAULT_DUMP_PATH,
                      default=None, metavar="PATH",
                      help="dump decompositions to PATH "
                           "(default: eval/multiturn/decomp_dump_pre.jsonl)")
    mode.add_argument("--check", nargs="?", const=DEFAULT_DUMP_PATH,
                      default=None, metavar="PATH",
                      help="byte-compare a fresh dump against the baseline at PATH "
                           "(default: eval/multiturn/decomp_dump_pre.jsonl)")
    args = parser.parse_args()

    rows = load_single_turn_queries()
    decomposer, llm_config, model = build_decomposer()
    require_ollama(llm_config, [model])

    os.makedirs(run_eval.RESULTS_DIR, exist_ok=True)
    log_path = os.path.join(run_eval.RESULTS_DIR, "decomp_stability.log")
    print(f"decomposing {len(rows)} single-turn gold queries with {model} "
          f"(temp 0, frozen single-turn prompt)…")
    with run_eval.captured(log_path, verbose=False):
        records = dump_records(decomposer, rows)
    dump = serialize(records)

    if args.write_baseline is not None:
        path = args.write_baseline
        with open(path, "wb") as fh:
            fh.write(dump)
        print(f"wrote {len(records)} decompositions to {path}")
        return 0

    path = args.check
    try:
        with open(path, "rb") as fh:
            baseline = fh.read()
    except FileNotFoundError:
        print(f"ERROR: baseline {path} not found — generate it first with "
              f"--write-baseline (the original scratchpad baseline is lost).",
              file=sys.stderr)
        return 1

    if dump == baseline:
        print(f"identical: {len(records)}/{len(records)} decompositions "
              f"byte-identical to {path}")
        return 0

    # Byte mismatch: parse only now, to name the differing rows.
    base_by_id = parse_dump(baseline)
    fresh_by_id = parse_dump(dump)
    diff_ids = sorted({i for i in fresh_by_id if base_by_id.get(i) != fresh_by_id[i]}
                      | {i for i in base_by_id if i not in fresh_by_id})
    print(f"DIFF: {len(diff_ids)}/{len(records)} decompositions differ from {path}:")
    for i in diff_ids:
        print(f"  {i}")
    return 1


if __name__ == "__main__":
    sys.exit(main())

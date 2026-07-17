from __future__ import annotations

import json
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

from ..config import Config
from ..retrieve.hybrid import Retriever

console = Console()

PAGE_TOL = 1  # a hop counts as hit if a retrieved chunk is on the gold page ±1


def load_benchmark(dataset: str, cfg: Config) -> dict:
    p = cfg.path("benchmarks", create=False) / f"{dataset}.json"
    if not p.exists():
        raise FileNotFoundError(f"benchmark missing: {p}")
    return json.loads(p.read_text())


def hop_hit(hits: list[dict], hop: dict, tol: int = PAGE_TOL) -> bool:
    for h in hits:
        if h["doc_id"] != hop["doc_id"]:
            continue
        if any(abs(h["page"] - p) <= tol for p in hop["pages"]):
            return True
    return False


def eval_retrieval(dataset: str, cfg: Config, k: int = 10) -> dict:
    bench = load_benchmark(dataset, cfg)
    retriever = Retriever(cfg)
    variants = {
        "fts": dict(channels=("fts",), use_rerank=False),
        "dense": dict(channels=("dense",), use_rerank=False),
        "hybrid": dict(channels=("dense", "fts"), use_rerank=False),
        "hybrid+rerank": dict(channels=("dense", "fts"), use_rerank=True),
    }
    from ..index.visual import VisualIndex

    if VisualIndex(cfg).exists(dataset):
        variants["visual"] = dict(channels=("visual",), use_rerank=False)
        variants["tri-hybrid"] = dict(channels=("dense", "fts", "visual"), use_rerank=False)
        variants["tri-hybrid+rerank"] = dict(channels=("dense", "fts", "visual"), use_rerank=True)
    report: dict = {"dataset": dataset, "k": k, "n_questions": len(bench["questions"]), "variants": {}}

    for name, kw in variants.items():
        hop_total = hop_ok = full_ok = 0
        modality_stats: dict[str, list[int]] = {}
        per_q: list[dict] = []
        for q in bench["questions"]:
            hits = retriever.search(q["question"], dataset, k_final=k, **kw)
            q_hops_ok = 0
            for hop in q["hops"]:
                ok = hop_hit(hits, hop)
                hop_total += 1
                hop_ok += ok
                q_hops_ok += ok
                m = hop.get("modality", "text")
                modality_stats.setdefault(m, [0, 0])
                modality_stats[m][0] += ok
                modality_stats[m][1] += 1
            full = q_hops_ok == len(q["hops"])
            full_ok += full
            per_q.append({"id": q["id"], "hops_hit": q_hops_ok, "hops": len(q["hops"]), "full": full})
        report["variants"][name] = {
            "hop_recall": round(hop_ok / max(hop_total, 1), 3),
            "full_question_recall": round(full_ok / len(bench["questions"]), 3),
            "by_modality": {
                m: round(v[0] / max(v[1], 1), 3) for m, v in sorted(modality_stats.items())
            },
            "per_question": per_q,
        }
        if kw.get("use_rerank") and retriever._rerank_broken:
            report["variants"][name]["rerank_active"] = False  # degraded to plain fusion

    ts = time.strftime("%Y%m%d_%H%M%S")
    out = cfg.path("runs") / f"retrieval_{dataset}_{ts}.json"
    out.write_text(json.dumps(report, indent=1))

    t = Table(title=f"Retrieval eval — {dataset} (k={k}, page ±{PAGE_TOL})")
    t.add_column("variant")
    t.add_column("hop recall")
    t.add_column("full-question recall")
    t.add_column("by modality")
    for name, v in report["variants"].items():
        t.add_row(
            name,
            f"{v['hop_recall']:.1%}",
            f"{v['full_question_recall']:.1%}",
            "  ".join(f"{m}:{r:.0%}" for m, r in v["by_modality"].items()),
        )
    console.print(t)
    console.print(f"[dim]written: {out}[/]")
    return report

from __future__ import annotations

import json
import warnings

import pandas as pd

from ..config import Config
from ..index.embedder import Embedder
from ..index.store import Store
from .reranker import Reranker

_page_cache: dict[tuple[str, str], dict[int, str]] = {}


def _page_text(cfg: Config, dataset: str, doc_id: str, page: int, max_chars: int = 1200) -> str:
    key = (dataset, doc_id)
    if key not in _page_cache:
        pages: dict[int, str] = {}
        f = cfg.path("processed", create=False) / dataset / doc_id / "pages.jsonl"
        if f.exists():
            with open(f) as fh:
                for line in fh:
                    rec = json.loads(line)
                    pages[rec["page"]] = rec["text"]
        _page_cache[key] = pages
    return _page_cache[key].get(page, "")[:max_chars]


def rrf_fuse(rank_lists: list[list[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rank_lists:
        for rank, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return scores


class Retriever:
    def __init__(self, cfg: Config, store: Store | None = None, embedder: Embedder | None = None):
        self.cfg = cfg
        self.store = store or Store(cfg)
        self.embedder = embedder or Embedder(cfg)
        self.reranker: Reranker | None = Reranker(cfg) if cfg.retrieval.rerank else None
        self._rerank_broken = False
        self._visual = None  # lazy VisualIndex

    def _visual_index(self):
        if self._visual is None:
            from ..index.visual import VisualIndex

            self._visual = VisualIndex(self.cfg)
        return self._visual

    def _rows_to_hits(self, df: pd.DataFrame, source: str) -> list[dict]:
        hits = []
        for _, r in df.iterrows():
            hits.append(
                {
                    "id": r["id"],
                    "doc_id": r["doc_id"],
                    "page": int(r["page"]),
                    "section": r.get("section", ""),
                    "text": r["text"],
                    "raw_text": r.get("raw_text", r["text"]),
                    "source": source,
                }
            )
        return hits

    def search(
        self,
        query: str,
        dataset: str,
        k_final: int | None = None,
        channels: tuple[str, ...] = ("dense", "fts"),
        use_rerank: bool | None = None,
        doc_id: str | None = None,
    ) -> list[dict]:
        cfg = self.cfg.retrieval
        k_final = k_final or cfg.final_k
        n = cfg.candidates_per_channel

        by_id: dict[str, dict] = {}
        rank_lists: list[list[str]] = []
        if "dense" in channels:
            qv = self.embedder.embed_query(query)
            df = self.store.dense(dataset, qv, n, doc_id=doc_id)
            hits = self._rows_to_hits(df, "dense")
            rank_lists.append([h["id"] for h in hits])
            for h in hits:
                by_id.setdefault(h["id"], h)
        if "fts" in channels:
            df = self.store.fts(dataset, query, n, doc_id=doc_id)
            if len(df):
                hits = self._rows_to_hits(df, "fts")
                rank_lists.append([h["id"] for h in hits])
                for h in hits:
                    by_id.setdefault(h["id"], h)
        if "visual" in channels:
            vi = self._visual_index()
            if vi.exists(dataset):
                vhits = []
                for v in vi.search(query, dataset, k=n):
                    if doc_id and v["doc_id"] != doc_id:
                        continue
                    txt = _page_text(self.cfg, dataset, v["doc_id"], v["page"])
                    vhits.append(
                        {
                            "id": f"vis::{v['doc_id']}::p{v['page']}",
                            "doc_id": v["doc_id"],
                            "page": v["page"],
                            "section": "(page-image match)",
                            "text": txt,
                            "raw_text": txt,
                            "source": "visual",
                        }
                    )
                rank_lists.append([h["id"] for h in vhits])
                for h in vhits:
                    by_id.setdefault(h["id"], h)

        fused = rrf_fuse(rank_lists, k=cfg.rrf_k)
        ordered = sorted(fused, key=fused.get, reverse=True)

        use_rerank = cfg.rerank if use_rerank is None else use_rerank
        if use_rerank and self.reranker and not self._rerank_broken and len(ordered) > 1:
            cand = ordered[: cfg.rerank_candidates]
            try:
                scores = self.reranker.score(query, [by_id[c]["raw_text"] for c in cand])
                cand = [c for _, c in sorted(zip(scores, cand), key=lambda t: -t[0])]
                ordered = cand + [c for c in ordered if c not in set(cand)]
            except Exception as e:  # never let an optional stage kill retrieval
                self._rerank_broken = True
                warnings.warn(f"reranker disabled after error: {e}")

        out = []
        for cid in ordered[:k_final]:
            h = dict(by_id[cid])
            h["rrf_score"] = fused[cid]
            h.setdefault("dataset", dataset)
            out.append(h)
        return out

    def search_multi(
        self,
        query: str,
        datasets: list[str],
        k_final: int | None = None,
        channels: tuple[str, ...] = ("dense", "fts"),
        use_rerank: bool | None = None,
    ) -> list[dict]:
        """Retrieve across several indices at once: pool each source's candidates, RRF-fuse
        the lot, then rerank the combined pool so results compete across sources."""
        datasets = [d for d in dict.fromkeys(datasets)]  # dedupe, keep order
        if len(datasets) == 1:
            return self.search(query, datasets[0], k_final, channels, use_rerank)

        cfg = self.cfg.retrieval
        k_final = k_final or cfg.final_k
        n = cfg.candidates_per_channel
        by_id: dict[str, dict] = {}
        rank_lists: list[list[str]] = []
        qv = self.embedder.embed_query(query) if "dense" in channels else None

        for ds in datasets:
            if "dense" in channels:
                hits = self._rows_to_hits(self.store.dense(ds, qv, n), "dense")
                rank_lists.append([h["id"] for h in hits])
                for h in hits:
                    h["dataset"] = ds
                    by_id.setdefault(h["id"], h)
            if "fts" in channels:
                df = self.store.fts(ds, query, n)
                if len(df):
                    hits = self._rows_to_hits(df, "fts")
                    rank_lists.append([h["id"] for h in hits])
                    for h in hits:
                        h["dataset"] = ds
                        by_id.setdefault(h["id"], h)
            if "visual" in channels:
                vi = self._visual_index()
                if vi.exists(ds):
                    vhits = []
                    for v in vi.search(query, ds, k=n):
                        txt = _page_text(self.cfg, ds, v["doc_id"], v["page"])
                        vhits.append({
                            "id": f"vis::{ds}::{v['doc_id']}::p{v['page']}", "doc_id": v["doc_id"],
                            "page": v["page"], "section": "(page-image match)", "text": txt,
                            "raw_text": txt, "source": "visual", "dataset": ds,
                        })
                    rank_lists.append([h["id"] for h in vhits])
                    for h in vhits:
                        by_id.setdefault(h["id"], h)

        fused = rrf_fuse(rank_lists, k=cfg.rrf_k)
        ordered = sorted(fused, key=fused.get, reverse=True)
        use_rerank = cfg.rerank if use_rerank is None else use_rerank
        if use_rerank and self.reranker and not self._rerank_broken and len(ordered) > 1:
            cand = ordered[: cfg.rerank_candidates]
            try:
                scores = self.reranker.score(query, [by_id[c]["raw_text"] for c in cand])
                cand = [c for _, c in sorted(zip(scores, cand), key=lambda t: -t[0])]
                ordered = cand + [c for c in ordered if c not in set(cand)]
            except Exception as e:
                self._rerank_broken = True
                warnings.warn(f"reranker disabled after error: {e}")

        out = []
        for cid in ordered[:k_final]:
            h = dict(by_id[cid])
            h["rrf_score"] = fused[cid]
            out.append(h)
        return out

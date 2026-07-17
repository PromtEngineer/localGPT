from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel


def repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


class PathsCfg(BaseModel):
    raw: str = "data/raw"
    processed: str = "data/processed"
    index: str = "data/index"
    benchmarks: str = "data/benchmarks"
    runs: str = "runs"


class ServingCfg(BaseModel):
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"
    timeout_s: int = 300


class ModelsCfg(BaseModel):
    orchestrator: str
    utility: str
    embedder: str
    reranker: str
    visual_retriever: str | None = None
    vision: str | None = None  # opt-in view_page reader; None -> orchestrator does vision
    judge: str | None = None  # eval-only: pin the judge independent of utility; None -> utility


class EmbeddingCfg(BaseModel):
    dim: int = 1024
    batch_size: int = 16
    device: str = "auto"
    query_instruction: str = ""


class IngestCfg(BaseModel):
    page_image_dpi: int = 150
    chunk_max_tokens: int = 800
    chunk_min_tokens: int = 120
    contextual_headers: bool = True


class RetrievalCfg(BaseModel):
    candidates_per_channel: int = 30
    rrf_k: int = 60
    final_k: int = 8
    rerank: bool = True
    rerank_candidates: int = 25


class AgentCfg(BaseModel):
    max_tool_calls: int = 12
    max_correction_loops: int = 2
    distillate_max_tokens: int = 2000
    no_new_info_stop: int = 2
    view_page_dpi: int = 220
    view_page_zoom_dpi: int = 500
    view_page_max_px: int = 2600
    # opt-in chart-reading path (see RESULTS.md chart-reading experiment). All off = validated default.
    view_page_auto_zoom: bool = False   # locate the figure, then re-read that region zoomed
    view_page_thinking: bool = False    # let a DENSE VLM interpolate axis ticks; MoE reader starves on it
    view_page_max_tokens: int = 6000    # thinking needs room or it returns empty
    numbers_via_sql: bool = False       # verifier: numeric claims must be grounded in tool output


class Config(BaseModel):
    paths: PathsCfg = PathsCfg()
    serving: ServingCfg = ServingCfg()
    models: ModelsCfg
    embedding: EmbeddingCfg = EmbeddingCfg()
    ingest: IngestCfg = IngestCfg()
    retrieval: RetrievalCfg = RetrievalCfg()
    agent: AgentCfg = AgentCfg()

    @property
    def root(self) -> Path:
        return repo_root()

    def path(self, name: str, create: bool = True) -> Path:
        p = self.root / getattr(self.paths, name)
        if create:
            p.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache(maxsize=1)
def load_config() -> Config:
    cfg_path = os.environ.get("MARAG_CONFIG") or str(repo_root() / "configs" / "default.yaml")
    with open(cfg_path) as f:
        data = yaml.safe_load(f)
    return Config(**data)

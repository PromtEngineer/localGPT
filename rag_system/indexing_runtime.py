"""Transport-neutral index build configuration and process isolation."""

from __future__ import annotations

import copy
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Any, Dict, List


def build_config(base_config: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
    config = copy.deepcopy(base_config)
    table_name = options.get("table_name")
    if table_name:
        config["storage"]["text_table_name"] = table_name
        config.setdefault("retrievers", {}).setdefault("dense", {})[
            "lancedb_table_name"
        ] = table_name

    config.setdefault("retrievers", {}).setdefault("latechunk", {})["enabled"] = bool(
        options.get("enable_latechunk", False)
    )
    if options.get("enable_docling_chunk"):
        config["chunker_mode"] = "docling"

    config.setdefault("contextual_enricher", {})
    config["contextual_enricher"]["enabled"] = bool(
        options.get("enable_enrich", False)
    )
    config["contextual_enricher"]["window_size"] = int(options.get("window_size", 2))

    config.setdefault("indexing", {})
    config["indexing"]["embedding_batch_size"] = int(
        options.get("batch_size_embed", 50)
    )
    config["indexing"]["enrichment_batch_size"] = int(
        options.get("batch_size_enrich", 25)
    )
    config["indexing"].setdefault(
        "conversion_timeout_seconds",
        int(os.getenv("CONVERSION_TIMEOUT_SECONDS", "900")),
    )
    config["indexing"].setdefault("overview_timeout_seconds", 45)
    config["indexing"].setdefault("enrichment_timeout_seconds", 60)

    if options.get("metadata_schema"):
        config["metadata_schema"] = options["metadata_schema"]
    if options.get("file_metadata"):
        config["file_metadata"] = options["file_metadata"]

    config.setdefault("chunking", {})
    config["chunking"]["chunk_size"] = int(options.get("chunk_size", 512))
    config["chunking"]["chunk_overlap"] = int(options.get("chunk_overlap", 64))

    if options.get("embedding_model"):
        config["embedding_model_name"] = options["embedding_model"]
    if options.get("enrich_model"):
        config["enrich_model"] = options["enrich_model"]
    if options.get("enrich_provider") not in (None, "ollama"):
        config["enrich_provider"] = options["enrich_provider"]
        if options.get("enrich_api_key"):
            config["enrich_api_key"] = options["enrich_api_key"]
    if options.get("overview_model_name"):
        config["overview_model_name"] = options["overview_model_name"]
    if options.get("index_id"):
        config["overview_path"] = (
            f"index_store/overviews/{options['index_id']}.jsonl"
        )
    return config


def execute_index_build(
    config: Dict[str, Any],
    ollama_config: Dict[str, Any],
    file_paths: List[str],
    *,
    index_id: str,
    force_reindex: bool,
    job_id: str | None,
    backend_base_url: str,
):
    """Run a build in an isolated child process, or in-process for tests."""
    from rag_system.indexing_worker import run_indexing_job

    args = (
        config,
        ollama_config,
        file_paths,
        index_id,
        force_reindex,
        job_id,
        backend_base_url,
    )
    if os.getenv("RAG_INDEX_IN_PROCESS") == "1":
        return run_indexing_job(*args)

    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as executor:
        future = executor.submit(run_indexing_job, *args)
        try:
            return future.result()
        except BrokenProcessPool as e:
            raise RuntimeError(
                "Indexing process crashed (likely out of memory). Try a smaller "
                "embedding batch size, disable enrichment, or index fewer files at once."
            ) from e

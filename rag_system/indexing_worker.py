"""Entry point for running an index build in an isolated child process.

The API server spawns this in a fresh process so that:
- torch/Docling/embedding memory is returned to the OS when the build ends
  (a long-lived Python process never gives its peak heap back), and
- a crash or OOM during indexing kills the build, not the chat server.

This module must stay import-light: it is imported by the spawned child, so
anything at module level runs before every build. Heavy imports live inside
run_indexing_job().
"""

from typing import Any, Dict, List, Optional


def run_indexing_job(
    config: Dict[str, Any],
    ollama_config: Dict[str, Any],
    file_paths: List[str],
    index_id: str,
    force_reindex: bool,
    job_id: Optional[str],
    backend_base_url: str,
) -> Dict[str, Any]:
    """Build an index and return the pipeline's result dict.

    Progress reporting and cancellation polling go over HTTP to the backend,
    exactly as the in-process path did — so this works unchanged whether it
    runs in a child process or in the caller's process.
    """
    import requests

    from rag_system.pipelines.indexing_pipeline import IndexingPipeline
    from rag_system.utils.ollama_client import OllamaClient

    def report_progress(stage, progress, message, **extra):
        if not job_id:
            return
        try:
            payload = {"stage": stage, "progress": progress, "message": message}
            payload.update({k: v for k, v in extra.items() if v is not None})
            requests.post(
                f"{backend_base_url}/index-jobs/{job_id}/progress",
                json=payload,
                timeout=2,
            )
        except Exception:
            pass

    def is_cancelled():
        if not job_id:
            return False
        try:
            resp = requests.get(f"{backend_base_url}/index-jobs/{job_id}", timeout=2)
            if resp.status_code != 200:
                return False
            job = resp.json()
            return bool(job.get("cancel_requested")) or job.get("status") == "cancelled"
        except Exception:
            return False

    llm_client = OllamaClient(host=ollama_config.get("host", "http://localhost:11434"))
    pipeline = IndexingPipeline(config, llm_client, ollama_config)
    return pipeline.run(
        file_paths,
        index_id=index_id,
        force_reindex=force_reindex,
        progress_callback=report_progress,
        cancel_callback=is_cancelled,
        job_id=job_id,
    )

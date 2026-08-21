"""Ephemeral "ask a folder" mode (roadmap item 4.6).

    python -m rag_system.main ask <folder> "<question>" ["<question>" ...]

Index a folder into a throwaway LanceDB table, answer, delete the table and
everything else the run created. Nothing is added to the user's real index and
nothing survives the process.

Why an ephemeral *index* rather than an ephemeral *agent*
---------------------------------------------------------
The design this borrows from (agentic-file-search) answers folder questions by
letting an agent grep and read files. Our own evidence says not to: filesystem
agents win on small corpora but lose to ranked retrieval as the corpus grows,
at roughly 39x the tokens (BM25-wins-at-scale), and retriever quality dominates
agency (BrowseComp-Plus). So the ephemeral thing here is the *index*, and the
answering path is the one that ships — same chunker, same embedder, same
retriever, same synthesis prompt. There is no second pipeline to keep in sync.

What is switched off, and why
-----------------------------
The roadmap specifies the ``fast`` profile with no enrichment. On top of that
profile this mode also disables **document overviews**: they cost one LLM call
per document at index time and their only consumers are the agent's triage
router and the (default-off) overview prefilter. Neither runs here — the answer
path skips triage entirely, because someone who typed ``ask <folder> "<q>"`` has
already decided the question is about the folder.

``--agent`` opts into the full agent loop (decomposition, verification) with
``force_rag`` set. The roadmap says "same pipeline, no agent loop", so that is
the default and the loop is the opt-in.

Cleanup
-------
Everything this mode writes lives under one ``tempfile.mkdtemp`` directory: the
LanceDB directory and the overview sidecar path both point inside it, so a
single ``rmtree`` in a ``finally`` is the whole teardown. The temp directory is
printed at the start and its removal is reported at the end, so a leak is
visible rather than assumed. ``--keep`` skips the removal for debugging and says
so loudly.

Indexing and synthesis can take minutes, so the run is very likely to be
interrupted at some point. ``SIGINT`` already unwinds through the ``finally``;
``SIGTERM`` does not, because Python's default handler exits without unwinding —
so it is temporarily rebound to raise ``SystemExit``. Both signals therefore
delete the temp index. ``SIGKILL`` cannot be caught and will leak the directory;
that is a property of ``kill -9``, not something this module can fix, and the
leaked directory is a ``localgpt-ask-*`` under ``$TMPDIR``.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import sys
import tempfile
import time
import uuid
from typing import Any, Dict, List, Sequence

# How much of a cited chunk to echo under the answer.
_CITATION_PREVIEW_CHARS = 220

# Citations printed per answer. The point is traceability, not a second copy of
# the corpus on the terminal.
_MAX_CITATIONS = 5


def _ephemeral_config(mode: str, temp_dir: str, table_name: str) -> Dict[str, Any]:
    """The chosen profile, redirected wholly inside *temp_dir*."""
    from rag_system.factory import get_pipeline_config

    config = get_pipeline_config(mode)
    config["storage"]["lancedb_uri"] = os.path.join(temp_dir, "lancedb")
    config["storage"]["db_path"] = os.path.join(temp_dir, "lancedb")
    config["storage"]["text_table_name"] = table_name

    # No enrichment (roadmap 4.6) — an LLM call per chunk is exactly what a
    # throwaway index should not pay for.
    config["contextual_enricher"] = {"enabled": False, "window_size": 1}
    # No overviews: an LLM call per document whose only consumers do not run here.
    config["overview"] = {"enabled": False}
    # Belt and braces — if a future profile turns overviews back on, the JSONL
    # and its vector sidecar still land in the temp directory rather than in the
    # user's index_store/.
    config["overview_path"] = os.path.join(temp_dir, "overviews", f"{table_name}.jsonl")
    # A second embedding pass over every document, for a corpus we are about to
    # delete.
    config.setdefault("retrieval", {})["latechunk"] = {"enabled": False}
    return config


_UNSET = object()


def _sigterm_raises() -> Any:
    """Make SIGTERM raise ``SystemExit`` so ``finally`` blocks actually run.

    Python's default SIGTERM disposition exits the process *without* unwinding,
    so a ``kill`` (or a harness timeout) during a long index build would leave
    the temp directory behind. Returns the previous handler for
    ``_restore_sigterm``, or ``_UNSET`` when the swap was not possible (not the
    main thread, or a platform without SIGTERM).
    """
    def _raise(_signum, _frame):
        raise SystemExit(143)

    try:
        previous = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, _raise)
        return previous
    except (ValueError, OSError, AttributeError):
        return _UNSET


def _restore_sigterm(previous: Any) -> None:
    if previous is _UNSET:
        return
    with contextlib.suppress(Exception):
        signal.signal(signal.SIGTERM, previous)


def _print_citations(source_documents: Sequence[Dict[str, Any]], out) -> None:
    if not source_documents:
        out("  (no sources)")
        return
    for i, doc in enumerate(source_documents[:_MAX_CITATIONS], start=1):
        text = (doc.get("text") or "").strip().replace("\n", " ")
        if len(text) > _CITATION_PREVIEW_CHARS:
            text = text[:_CITATION_PREVIEW_CHARS] + "…"
        score = doc.get("rerank_score", doc.get("score"))
        score_str = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
        out(f"  [{i}] {doc.get('document_id')}#{doc.get('chunk_index')} "
            f"(score {score_str})")
        out(f"      {text}")
    extra = len(source_documents) - _MAX_CITATIONS
    if extra > 0:
        out(f"  … and {extra} more source chunk(s)")


def _answer(agent, question: str, table_name: str, use_agent: bool,
            filters: Any, out) -> Dict[str, Any]:
    """One question, through either the retrieval pipeline or the whole agent."""
    if use_agent:
        # force_rag: the user pointed at a folder, so triage has nothing to decide.
        return agent.run(question, table_name=table_name, force_rag=True,
                         filters=filters)
    return agent.retrieval_pipeline.run(question, table_name, filters=filters)


def ask_folder(path: str, questions: Sequence[str], *, mode: str = "fast",
               interactive: bool = False, use_agent: bool = False,
               filters: Any = None, keep: bool = False, out=print) -> int:
    """Index *path*, answer *questions*, delete the index. Returns an exit code."""
    from rag_system.agent.loop import Agent
    from rag_system.factory import _build_llm_client
    from rag_system.main import SUPPORTED_DOCUMENT_EXTENSIONS, _collect_file_paths
    from rag_system.pipelines.indexing_pipeline import IndexingPipeline

    try:
        file_paths = _collect_file_paths(path)
    except FileNotFoundError as e:
        out(f"❌ {e}")
        return 1
    if not file_paths:
        out(f"❌ No indexable documents in {path} "
            f"(supported: {', '.join(SUPPORTED_DOCUMENT_EXTENSIONS)}).")
        return 1

    if not questions and not interactive:
        out("❌ Nothing to ask. Pass one or more questions, or use --interactive.")
        return 1

    # A fresh table name per run, so two concurrent `ask` runs cannot collide
    # even if they somehow shared a directory.
    table_name = f"ask_{uuid.uuid4().hex[:12]}"
    temp_dir = tempfile.mkdtemp(prefix="localgpt-ask-")
    started = time.time()

    out(f"📂 ask: {len(file_paths)} file(s) from {os.path.abspath(path)}")
    out(f"🗑️  ephemeral index: table '{table_name}' in {temp_dir} (profile '{mode}')")

    exit_code = 0
    previous_sigterm = _sigterm_raises()
    try:
        config = _ephemeral_config(mode, temp_dir, table_name)
        llm_client, llm_config = _build_llm_client()

        index_started = time.time()
        IndexingPipeline(config, llm_client, llm_config).run(file_paths)
        out(f"⏱️  indexed in {time.time() - index_started:.1f}s")

        # The standard agent, on the ephemeral config. `use_agent` picks which of
        # its two entry points answers: the retrieval pipeline (the roadmap's
        # "same pipeline, no agent loop") or the full loop.
        agent = Agent(pipeline_configs=config, llm_client=llm_client,
                      ollama_config=llm_config)

        pending: List[str] = list(questions)
        asked = 0
        while True:
            if not pending:
                if not interactive:
                    break
                try:
                    follow_up = input("\n❓ Ask a follow-up (blank line to finish): ").strip()
                except EOFError:
                    out("")
                    break
                if not follow_up:
                    break
                pending.append(follow_up)

            question = pending.pop(0)
            asked += 1
            out(f"\n{'=' * 72}\n❓ {question}\n{'=' * 72}")
            answer_started = time.time()
            result = _answer(agent, question, table_name, use_agent, filters, out)
            out(f"\n💬 {result.get('answer', '').strip()}")
            out(f"\n📎 Sources ({time.time() - answer_started:.1f}s):")
            _print_citations(result.get("source_documents") or [], out)

        if asked == 0:
            out("ℹ️  No questions asked.")

    except Exception as e:
        out(f"❌ ask failed: {type(e).__name__}: {e}")
        exit_code = 1
    finally:
        _restore_sigterm(previous_sigterm)
        # The teardown is the feature. Everything this run wrote is under
        # temp_dir, so one rmtree is the whole of it — but say what happened
        # either way, because "assume it cleaned up" is how leaks survive.
        if keep:
            out(f"\n⚠️  --keep: the ephemeral index was NOT deleted. Remove it yourself: {temp_dir}")
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)
            if os.path.exists(temp_dir):
                out(f"\n❌ Could not remove the ephemeral index at {temp_dir}")
                exit_code = 1
            else:
                out(f"\n🧹 Removed the ephemeral index ({temp_dir}).")
        out(f"⏱️  total {time.time() - started:.1f}s")

    return exit_code


if __name__ == "__main__":  # pragma: no cover - the real entry point is main.py
    sys.exit(ask_folder(sys.argv[1], sys.argv[2:]))

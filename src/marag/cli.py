from __future__ import annotations

import json

import typer
from rich.console import Console

app = typer.Typer(help="marag — local multimodal agentic RAG", no_args_is_help=True)
console = Console()


@app.command()
def ingest(dataset: str, limit: int = 0, force: bool = False):
    """Parse + chunk a raw dataset into data/processed/."""
    from .config import load_config
    from .ingest.pipeline import ingest_dataset

    r = ingest_dataset(dataset, load_config(), limit=limit or None, force=force)
    console.print(f"[bold]{r['ok']} ok, {r['failed']} failed[/]")


@app.command()
def index(dataset: str):
    """Embed chunks and build the LanceDB dense+FTS index."""
    from .config import load_config
    from .index.embedder import Embedder
    from .index.store import Store

    cfg = load_config()
    root = cfg.path("processed", create=False) / dataset
    chunks: list[dict] = []
    for doc_dir in sorted(root.iterdir()):
        f = doc_dir / "chunks.jsonl"
        if f.exists():
            chunks.extend(json.loads(line) for line in f.read_text().splitlines())
    if not chunks:
        raise typer.Exit(f"no chunks for {dataset} — run ingest first")  # type: ignore[arg-type]
    console.print(f"embedding {len(chunks)} chunks…")
    emb = Embedder(cfg)
    vecs = emb.embed_docs([c["text"] for c in chunks])
    Store(cfg).build(dataset, chunks, vecs)
    console.print(f"[green]indexed {len(chunks)} chunks → chunks_{dataset}[/]")


@app.command("rebuild-tables")
def rebuild_tables_cmd(dataset: str, force: bool = False):
    """Re-extract tables with Docling TableFormer and rebuild the DuckDB catalog."""
    from .config import load_config
    from .ingest.docling_tables import rebuild_tables
    from .ingest.pipeline import _build_duckdb

    cfg = load_config()
    s = rebuild_tables(dataset, cfg, force=force)
    _build_duckdb(dataset, cfg)
    console.print(f"[bold]{s['docs']} docs, {s['tables']} tables, {len(s['failed'])} failed[/]")


@app.command("index-visual")
def index_visual(dataset: str, batch_size: int = 4):
    """Build the late-interaction page-image index (ColModernVBERT class models)."""
    from .config import load_config
    from .index.visual import VisualIndex

    r = VisualIndex(load_config()).build(dataset, batch_size=batch_size)
    console.print(f"[green]visual index: {r['pages']} pages via {r['model']} → {r['path']}[/]")


@app.command("index-text-mv")
def index_text_mv(dataset: str, batch_size: int = 8):
    """Build the late-interaction TEXT-chunk index (pylate GTE-ModernColBERT class models)."""
    from .config import load_config
    from .index.text_multivector import TextMultiVectorIndex

    r = TextMultiVectorIndex(load_config()).build(dataset, batch_size=batch_size)
    console.print(f"[green]text-mv index: {r['chunks']} chunks via {r['model']} → {r['path']}[/]")


@app.command()
def search(query: str, dataset: str, k: int = 8, channels: str = "dense,fts", rerank: bool = True):
    """Hybrid search against an indexed dataset."""
    from .config import load_config
    from .retrieve.hybrid import Retriever

    hits = Retriever(load_config()).search(
        query, dataset, k_final=k, channels=tuple(channels.split(",")), use_rerank=rerank
    )
    for h in hits:
        console.print(f"[bold][{h['doc_id']} p{h['page']}][/] ({h['source']}) {h['raw_text'][:180]}…")


@app.command()
def ask(question: str, dataset: str):
    """Single-shot RAG answer."""
    from .agents.single_shot import answer_single_shot
    from .config import load_config
    from .retrieve.hybrid import Retriever

    cfg = load_config()
    r = answer_single_shot(question, dataset, cfg, Retriever(cfg))
    console.print(r["answer"])


@app.command()
def agent(question: str, dataset: str):
    """Iterative agentic answer (search → grep → read → sql)."""
    from .agents.search_agent import answer_agentic
    from .config import load_config
    from .retrieve.hybrid import Retriever

    cfg = load_config()
    r = answer_agentic(question, dataset, cfg, Retriever(cfg))
    console.print(r["answer"])
    console.print(f"[dim]{r['tool_calls']} tool calls, {len(r['evidence_pages'])} evidence pages[/]")


@app.command()
def answer(question: str, dataset: str):
    """Routed answer: router picks single-shot vs iterative per query."""
    from .agents.router import answer_auto
    from .config import load_config
    from .retrieve.hybrid import Retriever

    cfg = load_config()
    r = answer_auto(question, dataset, cfg, Retriever(cfg))
    console.print(f"[dim]route: {r['route']}[/]")
    console.print(r["answer"])


@app.command("eval-retrieval")
def eval_retrieval_cmd(dataset: str, k: int = 10):
    """Score retrieval hit rates against the QA benchmark."""
    from .config import load_config
    from .eval.retrieval_eval import eval_retrieval

    eval_retrieval(dataset, load_config(), k=k)


@app.command("eval-answers")
def eval_answers_cmd(dataset: str, mode: str = "single_shot", limit: int = 0):
    """Generate + judge answers against the QA benchmark (mode: single_shot|agentic)."""
    from .config import load_config
    from .eval.answer_eval import eval_answers

    eval_answers(dataset, load_config(), mode=mode, limit=limit or None)


@app.command()
def summarize(dataset: str, doc_id: str):
    """Whole-document summary (cached map-reduce)."""
    from .agents.tools import ToolBox
    from .config import load_config
    from .retrieve.hybrid import Retriever

    cfg = load_config()
    console.print(ToolBox(cfg, dataset, Retriever(cfg)).summarize_doc(doc_id))


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000):
    """Run the local workbench API + UI (http://host:port)."""
    import uvicorn

    from .server.app import create_app

    console.print(f"[bold]marag[/] workbench → http://{host}:{port}")
    uvicorn.run(create_app(), host=host, port=port, log_level="warning")


@app.command()
def status():
    """Show corpus/index/benchmark/model status."""
    from .config import load_config
    from .llm import served_models

    cfg = load_config()
    for ds_dir in sorted((cfg.path("raw", create=False)).glob("*/")):
        ds = ds_dir.name
        n_raw = len(list(ds_dir.glob("*.pdf")))
        proc = cfg.path("processed", create=False) / ds
        n_proc = len(list(proc.glob("*/meta.json"))) if proc.exists() else 0
        bench = cfg.path("benchmarks", create=False) / f"{ds}.json"
        console.print(f"[bold]{ds}[/]: raw={n_raw} processed={n_proc} benchmark={'✓' if bench.exists() else '—'}")
    console.print(f"served models: {', '.join(served_models(cfg)) or 'none (is the server up?)'}")


if __name__ == "__main__":
    app()

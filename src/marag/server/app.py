from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..config import Config, load_config
from . import catalog
from .db import Store

STATIC = Path(__file__).parent / "static"

MAX_UPLOAD_BYTES = 200 * 1024 * 1024

# one MPS device: embedder/reranker/visual inference must never run concurrently
_INFER_LOCK = threading.Lock()


def safe_upload_name(filename: str | None) -> str:
    """Upload filename policy: basename only, visible name, ingest-supported extension."""
    from ..ingest.formats import AUDIO_FORMATS, DATA_FORMATS, DOC_FORMATS, IMAGE_FORMATS

    name = Path(filename or "").name
    if not name or name.startswith("."):
        raise ValueError("empty or hidden filename")
    ext = Path(name).suffix.lower()
    if ext not in {".pdf"} | DOC_FORMATS | DATA_FORMATS | IMAGE_FORMATS | AUDIO_FORMATS:
        raise ValueError(f"unsupported file type: {ext or '(none)'}")
    return name


def _path_part(v: str) -> str:
    """Path params that reach the filesystem must be single, traversal-free components."""
    if not v or "/" in v or "\\" in v or ".." in v:
        raise HTTPException(400, "invalid path component")
    return v


def _retriever(cfg: Config):
    # one shared retriever (loads the embedder once); import here to keep startup light
    from ..retrieve.hybrid import Retriever

    if not hasattr(_retriever, "_r"):
        _retriever._r = Retriever(cfg)
    return _retriever._r


class SessionIn(BaseModel):
    title: str = "New session"
    scope: list[str] = []
    mode: str = "auto"


class SessionPatch(BaseModel):
    title: str | None = None
    scope: list[str] | None = None
    mode: str | None = None


class AskIn(BaseModel):
    session_id: str | None = None
    question: str
    scope: list[str]
    mode: str = "auto"  # auto | single | agentic


def create_app(cfg: Config | None = None) -> FastAPI:
    cfg = cfg or load_config()
    store = Store(cfg.path("index") / "sessions.db")
    ingest_jobs: dict[str, dict] = {}
    app = FastAPI(title="marag")

    # ---------- status ----------
    @app.get("/api/status")
    def status():
        from ..llm import served_context, served_models

        srcs = catalog.list_sources(cfg)
        served = set(served_models(cfg))
        roles = [
            ("agent", cfg.models.orchestrator),
            ("vision", cfg.models.vision or cfg.models.orchestrator),
            ("utility", cfg.models.utility),
        ]
        return {
            "models": [{"role": r, "name": m, "up": m in served} for r, m in roles],
            # lazy /api/ps probe: a silently-shrunk served context window shows up here
            "loaded": served_context(cfg),
            "totals": {
                "sources": len(srcs),
                "docs": sum(s["docs"] for s in srcs),
                "pages": sum(s["pages"] for s in srcs),
                "chunks": sum(s["chunks"] for s in srcs),
                "tables": sum(s["tables"] for s in srcs),
            },
            "endpoint": cfg.serving.base_url,
        }

    # ---------- sources & docs ----------
    @app.get("/api/sources")
    def sources():
        return catalog.list_sources(cfg)

    @app.get("/api/sources/{dataset}/docs")
    def docs(dataset: str):
        return catalog.list_docs(cfg, _path_part(dataset))

    # ---------- sessions ----------
    @app.get("/api/sessions")
    def sessions():
        return store.list_sessions()

    @app.post("/api/sessions")
    def new_session(body: SessionIn):
        return store.create_session(body.title, body.scope, body.mode)

    @app.get("/api/sessions/{sid}")
    def get_session(sid: str):
        s = store.get_session(sid)
        if not s:
            raise HTTPException(404, "no such session")
        return s

    @app.patch("/api/sessions/{sid}")
    def patch_session(sid: str, body: SessionPatch):
        store.update_session(sid, title=body.title, scope=body.scope, mode=body.mode)
        return store.get_session(sid)

    @app.delete("/api/sessions/{sid}")
    def del_session(sid: str):
        store.delete_session(sid)
        return {"ok": True}

    # ---------- evidence ----------
    @app.get("/api/page/{dataset}/{doc_id}/{page}.png")
    def page_png(dataset: str, doc_id: str, page: int, region: str = "full"):
        _path_part(dataset)
        _path_part(doc_id)
        if region and region != "full":
            png = catalog.render_region_png(cfg, dataset, doc_id, page, region)
            if png is None:
                raise HTTPException(404, "cannot render region")
            from fastapi import Response

            return Response(png, media_type="image/png")
        p = catalog.page_image_path(cfg, dataset, doc_id, page)
        if not p:
            raise HTTPException(404, "no page image")
        return FileResponse(p, media_type="image/png")

    @app.get("/api/evidence/{dataset}/{doc_id}/{page}")
    def evidence(dataset: str, doc_id: str, page: int):
        _path_part(dataset)
        _path_part(doc_id)
        if dataset == "_":  # resolve which source owns this doc
            dataset = catalog.dataset_of(cfg, doc_id) or dataset
        return catalog.page_evidence(cfg, dataset, doc_id, page)

    @app.get("/api/resolve/{doc_id}")
    def resolve(doc_id: str):
        ds = catalog.dataset_of(cfg, _path_part(doc_id))
        if not ds:
            raise HTTPException(404, "unknown doc")
        return {"doc_id": doc_id, "dataset": ds}

    # ---------- upload + ingest ----------
    @app.post("/api/upload")
    async def upload(source: str, file: UploadFile):
        try:
            fname = safe_upload_name(file.filename)
        except ValueError as e:
            raise HTTPException(400, str(e))
        raw = cfg.path("raw") / _path_part(source)
        raw.mkdir(parents=True, exist_ok=True)
        dest = raw / fname
        size = 0
        try:
            with open(dest, "wb") as out:
                while chunk := await file.read(1 << 20):
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise HTTPException(413, f"file exceeds {MAX_UPLOAD_BYTES >> 20}MB limit")
                    out.write(chunk)
        except HTTPException:
            dest.unlink(missing_ok=True)
            raise
        # register in the source manifest so ingest can find it
        manifest = raw / "manifest.json"
        entries = json.loads(manifest.read_text()) if manifest.exists() else []
        doc_id = f"{source[:3]}{len(entries) + 1:03d}"
        if not any(e["filename"] == fname for e in entries):
            entries.append({"id": doc_id, "filename": fname, "title": Path(fname).stem})
            manifest.write_text(json.dumps(entries, indent=1))
        job = f"job_{int(time.time() * 1000) % 10_000_000}"
        ingest_jobs[job] = {"file": fname, "source": source, "stage": "queued", "pct": 0}
        threading.Thread(target=_run_ingest, args=(job,), daemon=True).start()
        return {"job": job, "doc_id": doc_id, "bytes": size}

    def _run_ingest(job: str):
        from ..ingest.pipeline import ingest_dataset

        j = ingest_jobs[job]
        try:
            for stage, pct in [("parse", 30), ("chunks", 60)]:
                j.update(stage=stage, pct=pct)
            ingest_dataset(j["source"], cfg)  # idempotent: only new docs parse
            j.update(stage="dense", pct=85)
            with _INFER_LOCK:
                _index_dataset(cfg, j["source"])
            j.update(stage="done", pct=100)
        except Exception as e:
            j.update(stage="error", pct=100, error=str(e)[:300])

    @app.get("/api/ingest/{job}")
    def ingest_status(job: str):
        return ingest_jobs.get(job, {"stage": "unknown"})

    # ---------- ask (SSE stream) ----------
    @app.post("/api/ask")
    def ask(body: AskIn):
        if not body.scope:
            raise HTTPException(400, "pick at least one source")
        scope = body.scope  # every in-scope index is searched (search_multi fuses across them)
        sid = body.session_id or store.create_session(body.question[:60], body.scope, body.mode)["id"]
        store.add_message(sid, "user", body.question, {"scope": body.scope})

        def stream():
            q: queue.Queue = queue.Queue()
            result: dict = {}

            def worker():
                from ..agents.router import route
                from ..agents.search_agent import answer_agentic
                from ..agents.single_shot import answer_single_shot

                r = _retriever(cfg)
                mode = body.mode
                if mode == "auto":
                    mode = "single" if route(body.question, cfg) == "single_shot" else "agentic"
                    q.put(("route", {"mode": mode}))
                try:
                    with _INFER_LOCK:  # concurrent sessions must not race on the MPS device
                        if mode == "single":
                            res = answer_single_shot(body.question, scope, cfg, r)
                        else:
                            res = answer_agentic(
                                body.question, scope, cfg, r,
                                on_event=lambda kind, p: q.put((kind, p)),
                            )
                    result.update(res)
                    q.put(("answer", {"answer": res["answer"],
                                      "tool_calls": res.get("tool_calls", 0),
                                      "mode": mode}))
                except Exception as e:
                    q.put(("error", {"message": str(e)[:300]}))
                finally:
                    q.put((None, None))

            threading.Thread(target=worker, daemon=True).start()
            yield _sse("session", {"id": sid})
            while True:
                kind, payload = q.get()
                if kind is None:
                    break
                yield _sse(kind, payload)
            if "answer" in result:
                store.add_message(sid, "assistant", result["answer"],
                                  {"tool_calls": result.get("tool_calls", 0),
                                   "transcript": result.get("transcript", [])})
            yield _sse("done", {})

        return StreamingResponse(stream(), media_type="text/event-stream")

    # ---------- static UI ----------
    if STATIC.exists():
        app.mount("/ui", StaticFiles(directory=STATIC), name="ui")

    @app.get("/")
    def index():
        f = STATIC / "app.html"
        return HTMLResponse(f.read_text()) if f.exists() else HTMLResponse("<h1>marag api</h1>")

    return app


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _index_dataset(cfg: Config, dataset: str) -> None:
    from ..index.embedder import Embedder
    from ..index.store import Store as VStore

    root = cfg.path("processed", create=False) / dataset
    chunks: list[dict] = []
    for d in sorted(root.iterdir()):
        f = d / "chunks.jsonl"
        if f.exists():
            chunks.extend(json.loads(x) for x in f.read_text().splitlines())
    if chunks:
        vecs = Embedder(cfg).embed_docs([c["text"] for c in chunks])
        VStore(cfg).build(dataset, chunks, vecs)

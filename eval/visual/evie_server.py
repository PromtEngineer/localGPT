"""EVIE visual-retrieval sidecar (experiment 2026-08-21).

Embeds document PAGE IMAGES with tencent/EVIE-Preview-4.5B (ColPali-style
late interaction) and serves text-query -> top-page MaxSim search.
Runs in the durable ~/.venvs/localgpt-st6 venv (ST 6.0.0 / torch 2.13 —
incompatible with the repo's pinned MPS stack; also /tmp venvs get reaped).

  POST /index  {"name": str, "pages": [{"page_id": str, "path": str}]}
  POST /search {"name": str, "query": str, "k": int}
               -> {"pages": [{"page_id", "path", "score"}]}
  GET  /       -> status
"""
import hashlib
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from PIL import Image
from sentence_transformers import MultiVectorEncoder

MODEL_NAME = os.environ.get("EVIE_MODEL", "tencent/EVIE-Preview-4.5B")
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evie_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
print(f"loading {MODEL_NAME}…", flush=True)
t0 = time.time()
MODEL = MultiVectorEncoder(MODEL_NAME, device="mps", trust_remote_code=True)
print(f"loaded in {time.time()-t0:.1f}s", flush=True)

INDEXES = {}
LOCK = threading.Lock()


def _cache_path(name, pages):
    h = hashlib.sha256()
    h.update(MODEL_NAME.encode())
    for p in pages:
        h.update(p["page_id"].encode())
        with open(p["path"], "rb") as f:
            h.update(hashlib.sha256(f.read()).digest())
    return os.path.join(CACHE_DIR, f"{name}.{h.hexdigest()[:16]}.pt")


def index_pages(name, pages):
    t0 = time.time()
    path = _cache_path(name, pages)
    if os.path.exists(path):
        embs = torch.load(path, weights_only=True)
        src = "disk-cache"
    else:
        imgs = [Image.open(p["path"]).convert("RGB") for p in pages]
        embs = MODEL.encode_document(imgs, batch_size=1, show_progress_bar=False)
        embs = [e.to("cpu") for e in embs]
        torch.save(embs, path)
        src = "encoded+cached"
    with LOCK:
        INDEXES[name] = {"pages": pages, "embs": embs}
    return {"name": name, "count": len(pages),
            "seconds": round(time.time() - t0, 1), "source": src}


def search(name, query, k):
    t0 = time.time()
    with LOCK:
        idx = INDEXES.get(name)
    if idx is None:
        raise KeyError(f"index '{name}' not loaded (have: {sorted(INDEXES)})")
    qe = [MODEL.encode_query([query])[0].to("cpu")]
    scores = MODEL.similarity(qe, idx["embs"])[0].tolist()
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return {"pages": [{**idx["pages"][i], "score": round(scores[i], 3)} for i in order],
            "seconds": round(time.time() - t0, 3)}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        with LOCK:
            st = {n: len(v["pages"]) for n, v in INDEXES.items()}
        self._send(200, {"status": "ok", "model": MODEL_NAME, "indexes": st})

    def do_POST(self):
        try:
            data = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if self.path == "/index":
                self._send(200, index_pages(data["name"], data["pages"]))
            elif self.path == "/search":
                self._send(200, search(data["name"], data["query"], int(data.get("k", 3))))
            else:
                self._send(404, {"error": f"unknown path {self.path}"})
        except Exception as e:
            self._send(500, {"error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    print("evie sidecar on :8767", flush=True)
    ThreadingHTTPServer(("127.0.0.1", 8767), H).serve_forever()

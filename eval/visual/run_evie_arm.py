"""Arm B: EVIE page-image retrieval (top-3) + minicpm-v:8b answering."""
import base64
import json
import os
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "/Users/prompt/videos/localgpt_08082026/localGPT"
EVIE = "http://127.0.0.1:8767"
OLLAMA = "http://127.0.0.1:11434"
VLM = os.environ.get("VLM_MODEL", "minicpm-v:8b")

JOBS = [
    ("acq_scanned",
     [json.loads(l) for l in open(f"{REPO}/eval/goldset/acquisition.jsonl") if l.strip()]),
    ("vizdocs",
     [json.loads(l) for l in open(f"{HERE}/vizdocs_gold.jsonl") if l.strip()]),
]


def post(url, payload, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def vlm_answer(query, image_paths):
    imgs = [base64.b64encode(open(p, "rb").read()).decode() for p in image_paths]
    prompt = ("You are answering a question using ONLY the attached document page "
              "images. Read them carefully, including charts, tables, checkboxes "
              "and form fields. If the answer is visible, state it precisely with "
              "numbers and names exactly as shown. Question: " + query)
    body = {"model": VLM, "stream": False,
            "messages": [{"role": "user", "content": prompt, "images": imgs}],
            "options": {"temperature": 0, "num_ctx": 8192}}
    return post(f"{OLLAMA}/api/chat", body)["message"]["content"]


OUT = f"{HERE}/answers_evie.jsonl"
done = {json.loads(l)["id"] for l in open(OUT)} if os.path.exists(OUT) else set()
with open(OUT, "a") as out:
    for corpus, gold in JOBS:
        for row in gold:
            if row["id"] in done:
                continue
            t0 = time.time()
            rec = {"id": row["id"], "corpus": corpus, "query": row["query"],
                   "gold_answer": row["answer"], "expected": row["expected"]}
            try:
                sr = post(f"{EVIE}/search", {"name": corpus, "query": row["query"], "k": 3})
                rec["retrieved_pages"] = [p["page_id"] for p in sr["pages"]]
                rec["scores"] = [p["score"] for p in sr["pages"]]
                rec["retrieval_seconds"] = sr["seconds"]
                rec["answer"] = vlm_answer(row["query"], [p["path"] for p in sr["pages"]])
                rec["error"] = None
            except Exception as e:
                rec["answer"] = ""
                rec["error"] = f"{type(e).__name__}: {e}"
            rec["seconds"] = round(time.time() - t0, 1)
            out.write(json.dumps(rec) + "\n")
            out.flush()
            print(f"[{rec['id']}] {rec['seconds']:.0f}s pages={rec.get('retrieved_pages')}", flush=True)
print("EVIE ARM DONE")

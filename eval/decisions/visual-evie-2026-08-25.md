# Visual retrieval benchmark: EVIE page-image RAG vs the OCR pipeline

**Date:** 2026-08-25 · **Judge:** Sonnet (blind, single-voter bulk) · **Model:** tencent/EVIE-Preview-4.5B (ColPali-style late-interaction over page images, #1 on ViDoRe v3)

## Benchmark (new, committed under eval/visual/)

Two sub-corpora, 40 questions:

1. **Scanned-acq (24 q):** the 10 acquisition PDFs rasterized at 150 DPI and rebuilt as
   image-only PDFs (text layer verified empty). Questions = the existing verified
   acquisition gold set. Tests: "same content, text layer gone."
2. **Vizdocs (16 q):** 3 generated documents whose facts live in visuals — labeled charts,
   an unlabeled churn line, an unannotated incident heatmap, an SLA table, a filled form
   with checkboxes. `make_vizdocs.py` IS the ground truth (facts.json written from the
   plot variables). 9 "labeled" facts (recoverable by perfect OCR) + 7 "structural" facts
   (no value labels — require comparing visual elements).

Arms: **baseline** = full current pipeline on Docling+OCR text indexes of the image-only
PDFs; **EVIE** = sidecar (durable `~/.venvs/localgpt-st6`, local weights at
`~/models/evie-preview-4.5b`, disk-cached page embeddings) top-3 page retrieval +
minicpm-v:8b answering over the page images.

## Results

| slice | baseline (OCR pipeline) | EVIE + VLM |
|---|---|---|
| scanned acq (24) | **18** | 9 |
| vizdocs (16) | 7 | **13** |
| — labeled facts (9) | 5 | **9** |
| — structural facts (7) | 2 | **4** |
| total (40) | 25 | 22 |

EVIE retrieval itself: **hit@1 16/16** on vizdocs, correct pages on scanned-acq smoke
checks; page encode ~2.5s/page one-time; query ~0.5s; median E2E 9.3s/question.

## Findings

1. **The use case is real and precisely bounded: visual-native content.** Where facts
   live in charts, checkboxes, heatmaps and layout, page-image RAG nearly doubles the
   score (13 vs 7) — 9/9 on labeled visual facts vs the OCR pipeline's 5/9, and it reads
   structural facts (highlighted bar, checked box, hotspot cell) that no amount of OCR
   can recover (baseline 2/7).
2. **Scanned *prose* is NOT the use case.** macOS OCR is strong: the scanned acq corpus
   indexed to the same 82 chunks as the text-native version and scored 18/24 — within
   the text-native range. The EVIE arm's 9/24 there is an answering-leg failure, not
   retrieval: minicpm-v:8b reading 3 dense text pages loses badly to qwen3.5:9b over
   reranked, decomposed OCR text.
3. **Confounds, stated:** the arms differ in answering model (no local text model can
   read images; no local VLM matches qwen3.5:9b on text) — so the scanned-acq deficit
   bounds the *current best local VLM stack*, not EVIE. Vizdocs retrieval is easy
   (3 pages); the 16/16 hit@1 is necessary, not impressive. n=40, single-voter judging.

## Decision

Not integrated into the product yet. The measured shape supports a **visual index type**:
route documents (or pages) with chart/form/low-text-density content to EVIE page
retrieval + VLM answering; keep everything text-extractable on the shipped pipeline —
including scans of prose. Integration work if pursued: page-image storage per index,
a per-document visual/text router at indexing time, and a VLM answering path; the
sidecar, disk cache, and benchmark are committed and rerunnable.

## Ops notes

- ST 6.0.0 loads EVIE with `trust_remote_code=True` + the `[image]` extra (torchvision);
  loads in ~3s from local weights on MPS.
- HF unauthenticated downloads of the 9GB weights stall via huggingface_hub; plain
  `curl -L -C -` against `/resolve/main/` worked. Weights live outside /tmp.
- macOS's /tmp reaper deleted files *inside* a scratchpad venv (pyvenv.cfg, package
  __init__.py files) — sidecar venvs must live in `~/.venvs/`, never /tmp.

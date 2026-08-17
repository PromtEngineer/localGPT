# STATE OF THE ART: LOCAL/SELF-HOSTED AGENTIC RETRIEVAL STACK
## Component-by-component map, as of 8 August 2026

**Method note:** Primary sources only — arXiv, official GitHub repos, official HuggingFace model cards, official leaderboards (OmniDocBench, BRIGHT, MTEB/RTEB, LLM-AggreFact), and named vendor engineering blogs (Anthropic, Jina AI, Chroma, Weaviate, Qdrant, Naver Labs, Vectara, Zep, Letta, IBM, Microsoft Bing, Ai2, LlamaIndex). The session-wide WebSearch quota (200) was exhausted partway through; the remainder was gathered by direct fetch of primary URLs and the arXiv API. This biases coverage *toward* arXiv and *away* from vendor blogs — a gap I flag rather than paper over. Confidence labels: **established** / **emerging** / **contested**.

---

# 1. DOCUMENT PARSING / INGESTION

## 1.1 The headline: sub-1B specialist VLMs have won the parsing benchmark, decisively

**Claim: PaddleOCR-VL-1.6 (0.9B) leads OmniDocBench v1.6_full at 96.34 overall, ahead of MinerU2.5-Pro (1.2B, 95.75) and GLM-OCR (0.9B, 95.22). Every top slot is an open-weight specialist VLM; the best API model (Gemini 3 Pro) sits at 92.91 and GPT-5.2 at 86.59.**
→ https://github.com/opendatalab/OmniDocBench (leaderboard fetched 2026-08-08)

| Model | Type | Size | Overall↑ | TextEdit↓ | FormulaCDM↑ | TableTEDS↑ |
|---|---|---|---|---|---|---|
| PaddleOCR-VL-1.6 | specialist VLM | 0.9B | **96.34** | 0.0326 | 97.53 | 94.76 |
| MinerU2.5-Pro | specialist VLM | 1.2B | 95.75 | 0.036 | 97.45 | 93.42 |
| GLM-OCR | specialist VLM | 0.9B | 95.22 | 0.044 | 97.18 | 92.83 |
| PaddleOCR-VL-1.5 | specialist VLM | 0.9B | 94.93 | 0.038 | 96.89 | 91.67 |
| Youtu-Parsing | specialist VLM | — | 93.74 | 0.044 | 93.63 | 92.02 |
| Gemini 3 Pro | API | — | 92.91 | 0.064 | 95.99 | 89.15 |
| dots.ocr | specialist VLM | 3B | 90.77 | — | — | — |
| GPT-5.2 | API | — | 86.59 | 0.114 | 88.21 | 82.95 |

**Confidence: established.** This is the single cleanest result in the entire survey. A 0.9B open-weight model that fits in ~2GB of VRAM beats every frontier API model at document parsing.

**Claim: GLM-OCR (Zhipu/Z.ai, 0.9B) was released 2026-03-11 under MIT (layout component Apache 2.0), scoring 94.62 on OmniDocBench v1.5 — the highest of any model, open or closed, at the time. Architecture: 0.4B CogViT encoder + connector + 0.5B GLM decoder. Reported throughput 1.86 PDF pages/sec via Multi-Token Prediction.**
→ https://arxiv.org/pdf/2603.10910 (tech report, CC-BY 4.0) · https://www.llamaindex.ai/blog/omnidocbench-is-saturated-what-s-next-for-ocr-benchmarks (2026-02-24)
**Confidence: established** for the score and license; **emerging** for the throughput figure (vendor self-report).

**Claim: PaddleOCR-VL-0.9B pairs a NaViT-style dynamic-resolution encoder with ERNIE-4.5-0.3B and supports 109 languages.**
→ https://arxiv.org/abs/2510.14528 (v1 2025-10-16, v4 2025-11-25) · v1.6 tech report https://arxiv.org/pdf/2606.03264
**Confidence: established.**

**Claim: MinerU2.5 (1.2B) uses a two-stage coarse-to-fine strategy — layout analysis on a downsampled image, then native-resolution crops — and scores 75.2 on olmOCR-Bench (leading arXiv Math 76.6, Old Scans Math 54.6, Long Tiny Text 83.5).**
→ https://arxiv.org/abs/2509.22186 (2025-09-26)
**Confidence: established.** The decoupled layout→crop pattern is now the dominant architecture; PaddleOCR-VL and GLM-OCR both use variants of it.

## 1.2 The benchmark itself is saturating

**Claim: OmniDocBench is saturated — top models cluster at 94%+ on 1,355–1,651 pages across 9–10 document types, leaving "edge case fixing." LlamaIndex argues the field needs a benchmark rewarding semantic correctness over exact match, because agents care about functional accuracy, not formatting.**
→ https://www.llamaindex.ai/blog/omnidocbench-is-saturated-what-s-next-for-ocr-benchmarks (2026-02-24)
**Confidence: emerging** (vendor opinion piece) but corroborated by the leaderboard's own compression at the top.

**Claim: OmniDocBench maintainers responded with methodology changes rather than a new benchmark. Timeline: 2025-09-25 v1.0→v1.5 (hybrid matching); 2026-04-10 v1.5→v1.6 introducing Multi-Granularity Adaptive Matching (MGAM) to eliminate matching bias, +296 new pages; 2026-04-30 v1.7 with skills-based evaluation; 2026-07-27 community EvalScope integration for OpenAI-compatible endpoints.**
→ https://github.com/opendatalab/OmniDocBench
**Confidence: established.**

## 1.3 The olmOCR lineage has stalled

**Claim: olmOCR 2 (olmOCR-2-7B-1025, Ai2, 2025-10-22) introduced RLVR with binary unit tests as rewards, scoring 82.4 on olmOCR-Bench (+~4 over the prior release), beating Marker (76.1) and MinerU (75.8). olmOCR-Bench runs 8,413 unit tests over 1,403 PDF pages. Permissive open licenses for model, data, and code.**
→ https://arxiv.org/abs/2510.19817 · https://allenai.org/blog/olmocr-2
**Confidence: established.**

**Claim: No new olmOCR model shipped between Oct 2025 and Aug 2026. GitHub releases through v0.4.27 (2026-03-12) are pipeline/infrastructure work (PII tagging, timeouts, GPU deps) — no model version bump and no benchmark movement.**
→ https://github.com/allenai/olmocr/releases (fetched 2026-08-08)
**Confidence: established.** This is a real finding: **the Western open-OCR lineage went quiet for ~10 months while Chinese labs (Baidu/PaddleOCR, Zhipu/GLM, OpenDataLab/MinerU) shipped three generations.** olmOCR's lasting contribution is now olmOCR-Bench (independently maintained, not tied to the models it evaluates) rather than the models.

## 1.4 Docling vs unstructured — the framing changed

**Claim: Docling is governed under the Linux Foundation AI & Data Foundation, MIT-licensed, and in 2026 became an orchestration layer that plugs in whichever VLM is currently best, rather than competing on parsing quality itself.** 2026 changelog additions: DeepSeek-OCR (v2.68), **GLM-OCR with vLLM backend (v2.84)**, LightOnOCR-2-1B and Falcon-OCR (v2.85), Nanonets-OCR2 (v2.87), Nemotron-OCR (v2.105); plus threaded docling-parse v6 backend (v2.96), `docling-slim` modularization (v2.92), pluggable VLM runtime with presets (v2.73), VideoPipeline + ASR (v2.108–v2.116), chunking options in the service datamodel (v2.117).
→ https://github.com/docling-project/docling · https://raw.githubusercontent.com/docling-project/docling/main/CHANGELOG.md (through v2.118.1, 2026-08-07) · original tech report https://arxiv.org/abs/2501.17887 (2025-01-27, AAAI 2025 workshop)
**Confidence: established.**

**Claim: Docling supports both local VLM inference (Transformers + MLX) and remote OpenAI-compatible endpoints (vLLM, Ollama). Measured on a MacBook M3 Max: SmolDocling/MLX 6.15 s/page, Qwen2.5-VL-3B/MLX 23.50 s, Granite Vision/Transformers 104.75 s, Pixtral-12B/Transformers 1,828.21 s.**
→ https://docling-project.github.io/docling/usage/vision_models/
**Confidence: established.** The ~300× spread between MLX-small and Transformers-large is the operative fact for laptop-class local deployment.

**Claim: Granite-Docling-258M (IBM, 2025-09-17, Apache 2.0) emits DocTags — a unified markup for layout + text + semantics — and substantially beats SmolDocling-256M: Table TEDS-with-content 0.96 vs 0.76, code recognition F1 0.988 vs 0.915, full-page OCR F1 0.84 vs 0.80, equations 0.968 vs 0.947. IBM explicitly states it "is designed to complement the Docling library, not replace it" and is "not intended for general image understanding."**
→ https://huggingface.co/ibm-granite/granite-docling-258M
**Confidence: established.**

**Claim: unstructured's differentiator is typed semantic elements (Title, NarrativeText, Table, ListItem, Header) rather than flat Markdown — useful when downstream chunking logic keys off element type.**
→ Synthesized from parser-comparison coverage; **I could not verify this against an official unstructured source within budget.**
**Confidence: emerging / partially unverified.** Treat the typed-element claim as accurate (it is unstructured's documented core abstraction) but the competitive positioning as unsourced.

## 1.5 What's winning for local pipelines in 2026, and why

**The 2026 local pipeline is: Docling as orchestration + a 0.9–1.2B specialist VLM as the parsing engine.** Reasons, each grounded above:
1. **Quality is settled** — a 0.9B specialist beats GPT-5.2 by ~10 points on OmniDocBench (96.34 vs 86.59), so there is no quality argument for an API.
2. **The models fit** — 0.9B–1.2B at 4-bit runs on a laptop; MLX paths give 6–24 s/page on Apple silicon.
3. **Licensing is genuinely permissive** — GLM-OCR MIT, Granite-Docling Apache 2.0, MinerU2.5 open-weight, Docling MIT.
4. **Docling absorbed the churn** — it added five new OCR backends in 2026, so the orchestration layer is stable while the engine underneath is swappable. You are not betting on one model.
5. **Traditional pipeline parsers survive as the fast path, not the quality path** — Docling's own docs still route clean digital PDFs through docling-parse (now threaded, v6) and reserve the VLM for scanned/complex pages.

**Caveat:** the benchmark is saturated and does not reward semantic correctness. If your documents are financial presentations or legal filings, the leaderboard gap between the top five models is inside the noise floor of your actual task.

---

# 2. CHUNKING

## 2.1 The evidence base — and it points at boring defaults

**Claim: In the Chroma technical report, plain RecursiveCharacterTextSplitter at 400 tokens with no overlap scored 88.1–89.5% recall; ClusterSemanticChunker reached 0.913 and LLMSemanticChunker 0.919. Total spread across all methods was ~9% recall. Recommendation: recursive at 200–400 tokens no overlap for simplicity, ClusterSemanticChunker if complexity is acceptable.**
→ https://www.trychroma.com/research/evaluating-chunking (2024-07-03) · https://github.com/brandonstarxel/chunking_evaluation
**Confidence: established**, but **a 2024 result** — it predates long-context embedders and late chunking maturity.

**Claim (the important 2026 update): an eight-method / nine-dataset evaluation found advanced chunking introduces "substantially higher computational overhead" without meaningful effectiveness gains. Recursive Semantic 89.36 and Fixed-Size 87.71 Accuracy@5 lead; DenseX trails at 69.10. Fixed-size ran in <1 second where DenseX averaged 15+ hours; several methods hit 48-hour timeouts or OOM. LumberChunker scored highest on LLM-judged answer generation (4.35) but "results were unreliable due to frequent failures."**
→ https://arxiv.org/html/2606.00881v1 (2026-05-30). Methods: Fixed-Size, Recursive Semantic, Sequential HAC, TextTiling, Max-Min, GraphSeg, LumberChunker, DenseX. Datasets: GutenQA, LiteraryQA, NovelQA, Qasper, TriviaQA, SQuAD, PoQuAD, Natural Questions.
**Confidence: emerging** (single preprint) but it is the largest controlled chunking comparison to date and it **replicates Chroma's 2024 conclusion two years later with a bigger grid.**

## 2.2 Late chunking — real, but narrower than the hype

**Claim: Late chunking embeds the full document with a long-context embedder, then pools token embeddings into per-chunk vectors after the transformer. No additional training required; applies to any long-context embedder.**
→ https://arxiv.org/abs/2409.04701 (v1 2024-09-07, v3 2025-07-07, Jina AI)
**Confidence: established** as a method.

**Claim (contested): head-to-head evaluations find late chunking is *efficient* but *loses on relevance* to contextual retrieval. One study: late chunking "offers higher efficiency but tends to sacrifice relevance and completeness," while contextual retrieval "preserves semantic coherence more effectively but requires greater computational resources." A May 2026 comparison on NFCorpus with jina-v3 found ContextualRankFusion better overall than late chunking, with efficacy varying by dataset and model.**
→ https://arxiv.org/abs/2504.19754 (2025-04-28, ECIR 2025 workshop) · corroborating 2026 comparison surfaced but not independently fetched
**Confidence: contested.** Late chunking's sign flips by corpus. It is a *cheap* fix for cross-reference breakage, not a general quality upgrade.

## 2.3 Contextual retrieval — still the strongest measured chunking intervention

**Claim: Anthropic's contextual retrieval (prepend an LLM-generated ~100-token context blurb to each chunk at index time) reduces top-20 retrieval failure rate by 35% (contextual embeddings alone, 5.7%→3.7%), 49% (+contextual BM25, →2.9%), and 67% (+reranking, →1.9%). Cost ≈ $1.02 per million document tokens with prompt caching, assuming 800-token chunks.**
→ https://www.anthropic.com/news/contextual-retrieval (2024-09-19)
**Confidence: established.** Nearly two years old and still the reference number. Note the stacking: **most of the win comes from BM25 and reranking, not from the contextual blurb alone.**

## 2.4 "No chunking, long-context embedders"

**Claim: long context generally beats RAG on Wikipedia-style QA; summarization-based retrieval is comparable; chunk-based retrieval lags. But open-source LLMs "exhibit limited capacity for processing long contexts and therefore benefit substantially from retrieval," while closed models with stronger long-context ability do better on full context. Practical guidance from the same line of work: retrieval units should be longer and the number of chunks low — top-5 to top-10 typically suffices.**
→ https://arxiv.org/abs/2501.01880 (2025-01-03)
**Confidence: established for the asymmetry.** **This is the load-bearing fact for a self-hosted stack: the "just use long context" argument is an argument about frontier models, and it does not transfer to a local 7B–32B.**

**Claim: even for strong models, retrieval helps on reasoning benchmarks — a minimal RAG pipeline over CompactDS gave +10% MMLU, +33% MMLU Pro, +14% GPQA, +19% MATH relative, holding across 8B–70B.**
→ https://arxiv.org/abs/2507.01297 (2025-07-02)
**Confidence: established.**

**Cross-reference (see §10):** compression benefit shrinks as the reader gets stronger — significant in 9/10 settings — and a compressor tuned for a weak reader hides the gain from upgrading it (https://arxiv.org/abs/2606.21807, 2026-06-20). The same logic applies to chunking: **re-run your chunking ablation when you upgrade your local model.**

## 2.5 Defaults people ship

| Strategy | Ship it? | Evidence |
|---|---|---|
| Recursive/fixed 400–600 tokens, minimal overlap | **YES — the default** | 88–89% recall (Chroma 2024); 87.7–89.4 Acc@5 and best cost profile (2026-05-30) |
| Structure-aware (respect headings/tables from the parser) | **YES where the parser gives you structure** | Granite-Docling DocTags / unstructured typed elements make this nearly free |
| Contextual retrieval (index-time enrichment) | **YES if you can afford the offline pass** | 35–67% failure-rate reduction (Anthropic 2024-09-19) |
| Late chunking | **Conditional** — use when cross-references break | Efficient, but loses relevance vs contextual retrieval (2025-04, 2026-05) |
| Semantic / cluster chunking | **Only after measuring a retrieval gap** | +2–3 pts recall at meaningfully higher cost |
| LLM-driven chunking (LumberChunker, DenseX) | **NO** | DenseX 69.1 Acc@5, 15+ hrs; LumberChunker "frequent failures" |
| No chunking / long context only | **NO for local models** | Open-source LLMs "benefit substantially from retrieval" (2025-01) |

**The two-year-stable conclusion: chunking is not where your quality is. Every controlled study since 2024 puts the total spread across all chunking methods at ~9 points of recall, while a reranker alone is worth ~17 points of MRR@3 (see §5). Spend there instead.**

---

# 3. EMBEDDINGS (LOCAL)

## 3.1 Correction of a widespread error

Multiple secondary sources date Qwen3-Embedding to "early 2026." **It is June 2025.** Verified: https://huggingface.co/Qwen/Qwen3-Embedding-8B and https://arxiv.org/abs/2506.05176 (submitted 2025-06-05, v3 2025-06-11), Apache 2.0.

## 3.2 The scale ladder as of Aug 2026 (MMTEB / MTEB v2 multilingual mean)

| Model | Params | Dims | Ctx | MMTEB v2 | License | Date | Source |
|---|---|---|---|---|---|---|---|
| **harrier-oss-v1-27b** (Microsoft Bing) | 27B | 5,376 | 32K | **74.3** | **MIT** | 2026-03/04 | [HF](https://huggingface.co/microsoft/harrier-oss-v1-27b) |
| **KaLM-Embedding-Gemma3-12B-2511** (Tencent) | 11.76B | 3,840 | 32K | **72.32** | Tencent-KaLM-Community | 2025-11 | [HF](https://huggingface.co/tencent/KaLM-Embedding-Gemma3-12B-2511) |
| **Qwen3-Embedding-8B** | 8B | 4,096 | 32K | 70.58 | **Apache 2.0** | 2025-06-05 | [HF](https://huggingface.co/Qwen/Qwen3-Embedding-8B) |
| Qwen3-Embedding-4B | 4B | 2,560 | 32K | 69.45 | Apache 2.0 | 2025-06-05 | same |
| **harrier-oss-v1-0.6b** | 0.6B | 1,024 | 32K | **69.0** | **MIT** | 2026-03/04 | [HF](https://huggingface.co/microsoft/harrier-oss-v1-0.6b) |
| **jina-embeddings-v5-text-small** | 677M | 1,024 | 32K | 67.0–67.7 | CC BY-NC 4.0 | 2026-02-18 | [HF](https://huggingface.co/jinaai/jina-embeddings-v5-text-small) |
| **harrier-oss-v1-270m** | 270M | 640 | 32K | **66.5** | **MIT** | 2026-03/04 | [HF](https://huggingface.co/microsoft/harrier-oss-v1-270m) |
| jina-embeddings-v5-text-nano | 239M | 768 | 8K | 65.5 | CC BY-NC 4.0 | 2026-02-18 | [Jina blog](https://jina.ai/news/jina-embeddings-v5-text-distilling-4b-quality-into-sub-1b-multilingual-embeddings/) |
| Qwen3-Embedding-0.6B | 0.6B | 1,024 | 32K | 64.33 | Apache 2.0 | 2025-06-05 | HF |
| **EmbeddingGemma-300m** | 300M | 768 | **2K only** | 61.15 | Gemma terms | 2025-09 | [model card](https://ai.google.dev/gemma/docs/embeddinggemma/model_card) |

English MTEB v2 where reported: Qwen3-Embedding-8B **75.22**, 4B 74.60; jina-v5-text-small **71.7**, nano 71.0; EmbeddingGemma 69.67.
**Confidence: established** (all from official model cards / vendor blogs). Note vendor self-report bias is inherent; the MMTEB leaderboard itself renders client-side and could not be fetched directly this session.

## 3.3 Practical picks by scale — and license is the deciding factor

- **~270–300M (edge/CPU):** **harrier-oss-v1-270m (MIT, 66.5, 32K ctx)** now dominates EmbeddingGemma-300m (61.15, **2K ctx**) on both quality and context length. EmbeddingGemma's remaining advantage is QAT checkpoints (int4/int8, Q8_0, Q4_0) with minimal degradation and a mature on-device story.
- **~0.6B (the sweet spot):** **harrier-oss-v1-0.6b (MIT, 69.0)** > jina-v5-text-small (67.0, **non-commercial**) > Qwen3-Embedding-0.6B (64.33, Apache 2.0). If you need a permissive license and 32K context, harrier-0.6b is the 2026 default and it is *4.7 points* above Qwen3-0.6B.
- **4B–8B:** **Qwen3-Embedding-4B/8B (Apache 2.0)** remains the safe permissive choice, and it is the retriever that moved BrowseComp-Plus by +14.2 points when swapped in for BM25 (§8).
- **12B–27B:** harrier-oss-v1-27b (MIT, 74.3) if you have the VRAM; KaLM-Gemma3-12B (72.32) is close but carries a bespoke Tencent community license.

**Licensing summary — this is the most practically consequential fact in this section:** MIT/Apache options are **harrier-oss-v1 (all three, MIT)** and **Qwen3-Embedding (all three, Apache 2.0)**. **jina-embeddings-v5 is CC BY-NC 4.0** and **KaLM is a bespoke community license** — neither is usable in a commercial self-hosted product without contacting the vendor.

## 3.4 Matryoshka truncation — measured degradation is small and roughly linear

**Claim: EmbeddingGemma MRL degradation, 768d → 128d (6× storage reduction): multilingual MTEB v2 61.15 → 58.23 (−2.92); English v2 69.67 → 66.66 (−3.01). Intermediate: 512d costs ~0.44 pts, 256d ~1.4 pts.**
→ https://ai.google.dev/gemma/docs/embeddinggemma/model_card
**Confidence: established.** This is the best publicly documented MRL degradation curve.

**Claim: MRL is now universal at the top of the leaderboard.** Qwen3-Embedding supports 32–4096; jina-v5-text-small supports 32/64/128/256/512/768/1024; KaLM-12B supports 3840/2048/1024/512/256/128/64; Qwen3-VL-Embedding supports MRL.
→ respective model cards
**Confidence: established.**

**Claim: Jina's GOR regularization makes binary quantization "nearly lossless," dropping the effective memory footprint of a production embedding service "by an order of magnitude."**
→ https://jina.ai/news/jina-embeddings-v5-text-distilling-4b-quality-into-sub-1b-multilingual-embeddings/ (2026-02-19)
**Confidence: emerging** (vendor claim, no independent replication). If it holds, binary + MRL together is a ~50× storage reduction, which matters more for local deployment than 2 points of MMTEB.

**Practical rule:** truncate to **256d as the default** (~1.4 pt cost, 3× storage saving); go to 128d only if storage-bound; do not go below 128d without measuring.

## 3.5 Instruction-tuned embedders are now the norm, not a variant

**Claim: every top-tier 2026 embedder is query-instruction-aware, and all use the same `Instruct: {task}\nQuery: {text}` format with no instruction on the document side.** Qwen3-Embedding reports **1–5% improvement** from task-specific instructions. Harrier's card states flatly: "Each query must come with a one-sentence instruction that describes the task."
→ https://huggingface.co/Qwen/Qwen3-Embedding-8B · https://huggingface.co/microsoft/harrier-oss-v1-0.6b
**Confidence: established.** Practical consequence: **if you are running one of these models without an instruction prefix, you are leaving 1–5% on the table and your indexed documents are fine — only the query path changes.**

**Claim: Jina v5 takes a different route — four task-specific LoRA adapters (retrieval, text-matching, classification, clustering) on one base, distilled from Qwen3-Embedding-4B, with adapter-merged checkpoints published for vLLM and TEI.**
→ https://huggingface.co/jinaai/jina-embeddings-v5-text-small
**Confidence: established.** Note the distillation source: **a 677M model distilled from Qwen3-Embedding-4B matches jina-v4 (3.8B) on retrieval at 5.6× smaller.** Distillation-from-a-bigger-embedder is the 2026 recipe for the sub-1B tier — harrier's 270m/0.6b also use knowledge distillation from larger models.

## 3.6 Benchmark integrity: MTEB overfitting is now officially acknowledged

**Claim: MTEB's own maintainers launched RTEB (Retrieval Embedding Benchmark) on 2025-10-01 specifically because "when models are repeatedly evaluated against the same public datasets, a gap emerges between their reported scores and their actual performance on new, unseen data." RTEB pairs open datasets with private held-back datasets across 20 languages and enterprise domains (law, healthcare, finance, code), scored on nDCG@10. The open↔private score gap directly measures overfitting.**
→ https://huggingface.co/blog/rteb (2025-10-01)
**Confidence: established.**

**Claim: MMTEB (ICLR 2025) found "the best-performing publicly available model is multilingual-e5-large-instruct with only 560 million parameters" — i.e. parameter count did not predict rank at the time. 500+ tasks, 250+ languages.**
→ https://arxiv.org/abs/2502.13595 (2025-02-19, final 2025-11-13)
**Confidence: established**, and now partly superseded — the 2026 leaderboard is size-ordered again at the top (27B > 12B > 8B).

**Practical advice: treat MMTEB deltas under ~2 points as noise, and validate your top 2–3 candidates on your own held-out set. The +6.2pp accuracy swing from a bare embedding-model swap measured in the memory literature (§11) is larger than most architectural choices you will make.**

---

# 4. SPARSE LEG + FUSION

## 4.1 BM25 / native FTS locally

**Claim: bm25s (pure Python + NumPy) achieves 1,196 QPS on nfcorpus vs rank-bm25 at 224.66 and Elasticsearch at 45.84 — ~26× faster than Elasticsearch. v0.2.0 added a numba backend for ~2× further speedup on larger datasets.**
→ https://github.com/xhluca/bm25s
**Confidence: established** (self-reported, but the methodology is public and reproducible). **For a local stack this settles it: you do not need Elasticsearch for the sparse leg.** SQLite FTS5, DuckDB FTS, Postgres tsvector, Tantivy, and bm25s are all adequate; the choice is an operational one, not a quality one.

**Claim: "BM25 remains the first-stage retrieval mechanism in many high-throughput production systems even when dense retrieval is layered on top" — SPLADE outperforms BM25 on BEIR but requires GPU inference, where BM25 is a CPU-only inverted index.**
**Confidence: established** as the standard tradeoff, though I could not anchor this specific phrasing to a primary source within budget — treat the *reasoning* as sound and the *quote* as unsourced.

## 4.2 Learned sparse — quietly excellent, and 2026 was a good year

**Claim: SPLADE-v3 is "statistically significantly more effective than both BM25 and SPLADE++, while comparing well to cross-encoder re-rankers." It maps text to a 30,522-dim sparse vector (BERT vocab) with learned term weights and implicit expansion.**
→ https://arxiv.org/abs/2403.06789 · https://github.com/naver/splade
**Confidence: established.**

**Claim (the 2026 headline): SPLARE (Naver Labs Europe) is a new learned sparse retriever producing "generalizable sparse latent representations," available at 7B and a "significantly lighter" 2B. It achieves top results on MMTEB multilingual and English retrieval and beats state-of-the-art dense baselines including Qwen3-8B-Embed. Deployed at WSDM Cup 2026 with Qwen3-Reranker-4B and simple score fusion on top.**
→ https://arxiv.org/abs/2602.20986 (2026-02-24)
**Confidence: emerging** (competition report; the underlying SPLARE paper's full results were not retrievable this session). **But "learned sparse beats Qwen3-8B-Embed on multilingual retrieval" is a significant claim that reopens the sparse-vs-dense question.**

**Claim: SPLADE-Code (2026-03-23) is the first large-scale learned-sparse family for code retrieval, 600M–8B. Sub-1B variants hit 75.4 on MTEB Code (SOTA among sub-1B retrievers); the 8B reaches 79.0. Critically: "sub-millisecond retrieval on a 1M-passage collection with little effectiveness loss."**
→ https://arxiv.org/abs/2603.22008
**Confidence: emerging.** Directly relevant if your corpus is code.

**Claim: a 2026 training fix — rescaling the MLM-head projection by a constant at initialization — resolves why stronger pretrained encoders (ModernBERT, Ettin) previously *failed* in SPLADE training due to inflated MLM-head L2 norms. It converts unstable runs into competitive sparse retrievers, in several cases matching or exceeding classic BERT-SPLADE.**
→ https://arxiv.org/abs/2606.18811 (2026-06-17)
**Confidence: emerging.** This unblocks modern-encoder SPLADE, which had been stuck on BERT-base for years.

**Practical read: learned sparse is no longer a research curiosity, but it costs you GPU inference at index time and query time. For most self-hosted stacks BM25 remains the right sparse leg; SPLADE/SPLARE is worth it if you are multilingual, code-heavy, or latency-critical (sub-ms at 1M passages).**

## 4.3 Fusion — RRF vs weighted vs learned

**Claim: Weaviate's default fusion has been `relativeScoreFusion` (min-max normalize each list, add weighted normalized scores) since v1.24, not RRF. `rankedFusion` (RRF) "keeps only the position of a result in each list and discards the scores" and remains available via `fusionType`. Weaviate publishes no measured comparison between the two.**
→ https://weaviate.io/blog/hybrid-search-explained
**Confidence: established.** Note the common claim that "Weaviate defaults to RRF" is **wrong** as of v1.24+.

**Claim: Qdrant supports RRF (default k=2; weighted variant since v1.17.0) and DBSF (Distribution-Based Score Fusion, normalizing by mean/σ over a 3-sigma range). Its published decision table: tunable eval set → weighted RRF with train/val split; trust raw scores but no eval set → DBSF; no eval set or score priors → RRF ("the safe default"). Explicit caveats: score normalization relies on small top-k samples so single outliers skew it; "a fixed alpha over raw scores tends to be dominated by whichever retriever has larger raw magnitudes"; "neither method dominates universally"; retune when retrievers, embeddings or corpus change.**
→ https://qdrant.tech/documentation/concepts/hybrid-queries/
**Confidence: established.** This is the most honest vendor guidance I found — it explicitly declines to claim a winner.

**Claim: the theoretical case for RRF is that BM25 scores are unbounded positives while cosine is bounded [−1,1], so naive addition is meaningless; RRF sidesteps normalization entirely by using ranks with k=60 (or k=2 in Qdrant).**
**Confidence: established** as reasoning; the specific WANDS numbers circulating (RRF nDCG 0.7068 vs BM25 0.6983 vs KNN 0.6953) come from a secondary source I could not verify — **do not cite those figures.**

**Claim: hybrid+RRF measurably beats both legs. On T2-RAGBench (23,088 financial questions, 7,318 docs): Hybrid RRF Recall@5 0.695 / MRR@3 0.433, vs BM25 0.644/0.411 and dense 0.587/0.351. Adding a cross-encoder takes it to 0.816/0.605.**
→ https://arxiv.org/html/2604.01733v1 (2026-04-02)
**Confidence: established** for this domain. Note **BM25 alone beat dense alone by 5.7 points of Recall@5** here — a useful corrective for anyone considering dropping the sparse leg.

**Claim (learned fusion): projection fusion outperforms RRF when combined with diversity reranking on TREC-COVID, comparing BM25/SPLADE sparse against BGE/DPR dense.**
→ https://arxiv.org/pdf/2604.13728 (2026-04)
**Confidence: emerging / weakly supported** — single-author preprint, single dataset, and the PDF's numeric tables were not extractable. **There is no strong 2026 evidence that learned fusion beats RRF in the general case.**

## 4.4 Bottom line for §4

**Ship: BM25 (bm25s or your DB's native FTS) + a dense leg + RRF. Do not spend time tuning fusion before you have a reranker.** RRF is the safe default precisely because it is scale-free; weighted/DBSF only pays if you have an eval set to tune against, which is Qdrant's own published position. Learned sparse (SPLADE-v3 / SPLARE / SPLADE-Code) is a genuine 2026 upgrade path but costs GPU at index and query time.

---

# 5. RERANKERS (LOCAL)

## 5.1 The single highest-ROI component in the stack

Across every 2026 head-to-head that includes both, reranking beats every query-side technique by roughly an order of magnitude in ROI:
- T2-RAGBench: hybrid 0.433 MRR@3 → **+cross-encoder 0.605** (**+17.2 pp**), Recall@5 0.695 → 0.816. → https://arxiv.org/html/2604.01733v1 (2026-04-02)
- Local 7B ablation: removing the reranker costs **−1.7 EM (p<0.001) at negligible latency**; the authors retain it "unconditionally." → https://arxiv.org/abs/2606.21553 (2026-06-19)
- Anthropic contextual retrieval: reranking took failure-rate reduction from 49% → **67%**. → https://www.anthropic.com/news/contextual-retrieval (2024-09-19)

**Confidence: established.** This is the most replicated finding in the whole survey.

## 5.2 Cross-encoders — the bge lineage and Qwen3-Reranker

**Claim: Qwen3-Reranker (0.6B/4B/8B, Apache 2.0, 2025-06-05) substantially beats bge-reranker-v2-m3 on the Qwen team's own evaluation:**

| Model | Params | MTEB-R | CMTEB-R | MMTEB-R | MLDR | MTEB-Code | FollowIR |
|---|---|---|---|---|---|---|---|
| Qwen3-Reranker-0.6B | 0.6B | 65.80 | 71.31 | 66.36 | 67.28 | 73.42 | 5.41 |
| **Qwen3-Reranker-4B** | 4B | **69.76** | 75.94 | 72.74 | 69.97 | **81.20** | **14.84** |
| Qwen3-Reranker-8B | 8B | 69.02 | **77.45** | **72.94** | **70.19** | 81.22 | 8.05 |
| bge-reranker-v2-m3 | 0.6B | 57.03 | 72.16 | 58.36 | 59.51 | 41.38 | −0.01 |

→ https://qwenlm.github.io/blog/qwen3-embedding/ (2025-06-05)
**Confidence: established** (vendor self-report, but the FollowIR and MTEB-Code gaps are too large to be tuning artifacts). Two things stand out: **bge-reranker-v2-m3 is effectively useless on code (41.38) and on instruction-following (−0.01 FollowIR)**, and **4B beats 8B on several tasks** — do not assume the 8B is the right pick.

**Claim: bge-reranker-v2-m3 (0.6B, Apache 2.0, XLM-RoBERTa on bge-m3) has a 512-token max length. BAAI's own guidance: use it "for efficiency"; use bge-reranker-v2-minicpm-layerwise or -gemma "for maximum performance."**
→ https://huggingface.co/BAAI/bge-reranker-v2-m3
**Confidence: established.** **No bge-reranker-v3 exists as of Aug 2026** — the newest in the family is bge-reranker-v2.5-gemma2-lightweight (July 2024). **The bge reranker lineage has been static for two years.**

**Claim: mxbai-rerank-large-v2 (2B, Apache 2.0, 2025-06-04) scores BEIR 57.49 at 0.89s latency on A100; base-v2 (1.5B) 55.57 at 0.67s. Trained with RL + contrastive + preference learning.**
→ https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v2
**Confidence: established.** Note the multilingual score is weak (29.79).

## 5.3 Late-interaction and "not-late" interaction rerankers

**Claim: jina-reranker-v3 (0.6B on Qwen3-0.6B, 2025-09-29) introduced "last but not late interaction" — causal self-attention between query and all documents inside one shared 131K context window, up to 64 documents at once. BEIR 61.94 nDCG@10, SOTA, beating mxbai-rerank-large-v2 (61.44) with 2.5× fewer parameters, Qwen3-Reranker-0.6B (56.28) and bge-reranker-v2-m3 (56.51). Also HotpotQA 78.56, FEVER 93.95, CoIR 63.28. But MIRACL multilingual 66.50 — *below* bge-reranker-v2-m3 (69.32) and Qwen3-Reranker-4B (67.52).**
→ https://arxiv.org/html/2509.25085v1 · https://huggingface.co/jinaai/jina-reranker-v3
**Confidence: established.**
**⚠️ Critical licensing flag: jina-reranker-v3 is CC BY-NC 4.0 — non-commercial.** The best-scoring small reranker on BEIR is not usable in a commercial self-hosted product without a Jina license.

**Claim: KaLM-Reranker-V1 (Tencent, 2026-06-22, rev 2026-07-07) proposes "fast but not late interaction" (FBNL): an encoder pre-encodes passages with Matryoshka embedding pooling; a decoder processes instructions + query; cross-attention computes relevance. Sizes Nano 0.27B / Small 1B / Large 4B activated params. Claims SOTA on BEIR comparable to Qwen3-Reranker, strong MIRACL despite limited multilingual training, and "even the 0.27B Nano model remaining competitive with 7–12B embedding models" on LMEB.**
→ https://arxiv.org/abs/2606.22807
**Confidence: emerging** — no quantitative latency numbers in the abstract, PDF tables unextractable. Worth tracking as the likely permissive answer to jina-v3's NC license.

**Claim: ColBERT-style late interaction persists as research (ColBERT-Att, 2026-03) but has produced no new production-grade open reranker in the last 12 months. PLAID remains the efficiency engine (up to 7× GPU / 45× CPU speedup over vanilla ColBERTv2).**
→ https://arxiv.org/html/2603.25248v1 · https://arxiv.org/abs/2205.09707
**Confidence: established.** **Late interaction lost the reranking race to "last-but-not-late" and cross-encoders in 2025–26.** Its remaining stronghold is multimodal/document-image retrieval (ColPali lineage), not text reranking.

## 5.4 LLM-as-reranker — the numbers say no, with one exception

**Claim (the cost/latency reality): pointwise, BGE-reranker-v2 scores 0.74 NDCG@10 at 12ms and $2/1k queries; Gemini Flash scores 0.68 at 185ms and $27/1k. Listwise, Gemini Flash reaches 0.78 at 420ms and $18/1k — i.e. LLM listwise buys +0.04 NDCG@10 at ~35× the latency and ~9× the cost. Recommendation: specialized reranker narrows to top-20, LLM listwise only on the top-10.**
→ https://zeroentropy.dev/articles/llm-as-reranker-guide/ (2025-07-20)
**Confidence: established** for the shape; the absolute numbers are one vendor's benchmark. The 35× latency multiplier is corroborated qualitatively across the literature.

**Claim (2026 efficiency work is closing the gap): CompRank (Mistral-7B, 2026-06-10) decouples document representations via block-structured attention, compresses to 10.2% of document tokens, and uses attention-derived decoding-free scoring. BEIR nDCG@10 39.2 average (vs ICR-Mistral-7B 38.4, RankGPT-Mistral-7B 35.1, BM25 35.4), with 4.9×–9.5× speedup over generation-based listwise reranking and 0.050 s/query query-block-forward at 500 documents.**
→ https://arxiv.org/html/2606.11700v1
**Confidence: emerging.**

**Claim: whole-pool setwise reranking with long-context LLMs (DualEnd, 2026-06-01) ranks 100 candidates in 50 serial calls vs 99 for one-at-a-time whole-pool methods, evaluated across nine open-weight LLMs. Framing: "long context is not merely more prompt space, but an opportunity to make LLM re-rankers both effective and efficient."**
→ https://arxiv.org/abs/2606.01782
**Confidence: emerging.**

## 5.5 Recommended reranker for a self-hosted stack

| Constraint | Pick | Why |
|---|---|---|
| **Permissive license + best quality** | **Qwen3-Reranker-4B** (Apache 2.0) | MTEB-R 69.76, MTEB-Code 81.20, FollowIR 14.84 — the only strong option that is both permissive and instruction-following |
| **Permissive + tight latency** | **Qwen3-Reranker-0.6B** (Apache 2.0) | 65.80 MTEB-R, +8.8 over bge-v2-m3, still sub-1B |
| **Best BEIR score, non-commercial OK** | jina-reranker-v3 (CC BY-NC) | BEIR 61.94, 131K ctx, 64 docs/pass |
| **Multilingual first** | bge-reranker-v2-m3 or Qwen3-Reranker-4B | bge still leads MIRACL (69.32); jina-v3 is weaker there (66.50) |
| **Apache + mid-size** | mxbai-rerank-large-v2 (2B) | BEIR 57.49, 0.89s A100; weak multilingual |
| **LLM reranker** | **Only as a top-10 second pass** | +0.04 NDCG for 35× latency / 9× cost |

**Architectural note:** Qwen3's causal LM yes/no-logit design is fundamentally slower than a SequenceClassification single forward pass (bge/mxbai). Budget accordingly — the quality gap is real but so is the throughput gap.

**⚠️ Cross-reference to §10:** rather than adding a compressor after your reranker, consider replacing the reranker with **Provence / OpenProvence**, which folds sentence-level context pruning *into* the reranking pass at ~zero marginal cost.

---

# 6. QUERY PLANNING

## 6.1 HyDE in 2026: demoted, not abandoned

**Claim: HyDE consistently *underperforms* vanilla dense retrieval on entity-centric/numeric corpora. On T2-RAGBench (23,088 financial questions): HyDE Recall@5 0.544 / MRR@3 0.318 / nDCG@10 0.433 vs dense baseline 0.587 / 0.351 / 0.466 — i.e. −7.3% / −9.4% / −7.1%. Stated cause: "LLM-generated pseudo-documents introduce noise through fabricated financial figures." CRAG/adaptive retrieval also underperformed plain hybrid fusion (0.658 vs 0.695 Recall@5).**
→ https://arxiv.org/html/2604.01733v1 (2026-04-02)
**Confidence: established for numeric/entity domains.**

**Claim (the most actionable HyDE result of the year): in a production RAG system, LLM augmentation was needed for only 27.8% of real queries, while synthetic eval sets implied >90% — "the Coverage Illusion." Four ML approaches to *pre-retrieval* routing all failed: "the need for LLM augmentation cannot be determined from the query alone." The fix is a post-retrieval cascade — escalate to HyDE only when initial retrieval returns nothing. Result vs always-HyDE: +0.140 Composite Overall, −31.8% latency, 72.2% of queries served with no LLM augmentation.**
→ https://arxiv.org/abs/2605.27220 (Hussain & Nielbo, 2026-05-26)
**Confidence: emerging** (single production system, preprint) but structurally important.

**Claim (counter-evidence): HyDE still wins in conversational multi-domain QA — "robust yet straightforward methods, such as reranking, hybrid BM25, and HyDE, consistently outperform vanilla RAG," across 8 conversational QA datasets. And in Turkish, HyDE maximizes accuracy (85%) but the Pareto-optimal config reaches 84.60% at far lower cost.**
→ https://arxiv.org/abs/2602.09552 (2026-02-10) · https://arxiv.org/abs/2602.03652 (2026-02-03)
**Confidence: contested — HyDE's sign genuinely flips with corpus type.**

**Claim: on small local models HyDE costs +25–40% response time with high hallucination rates on personal queries.**
→ https://arxiv.org/abs/2506.21568 (2025-06-12) — ⚠️ *2025 result on Gemma 1B/4B, possibly stale.*

**Claim: the 2026 architectural response is to move hypothetical generation to index time. HyPE precomputes hypothetical *questions* per chunk and matches question↔question, reporting up to +42 pp context precision and +45 pp claim recall across six datasets, with zero query-time latency.**
→ https://arxiv.org/abs/2607.29402 (arXiv 2026-07-31; ⚠️ *the underlying paper is IEEE Access 2025 — treat as a 2025 result*)
**Confidence: emerging.**

**Claim (docs lag): LlamaIndex still documents HyDE and multi-step decomposition with no caveats. LangChain's 2026 retrieval page leads with "2-Step RAG vs Agentic RAG vs Hybrid RAG" architecture selection and does not surface HyDE, multi-query, or self-query at all.**
→ https://developers.llamaindex.ai/python/framework/optimizing/advanced_retrieval/query_transformations/ · https://docs.langchain.com/oss/python/langchain/retrieval (both fetched 2026-08-08)
**Confidence: established (direct observation).** LangChain quietly demoted query transformation in its 2026 information architecture; LlamaIndex did not.

**Verdict: HyDE is alive but is no longer a default. Never always-on. Either gate it post-retrieval (cheap, no training) or move the hypothetical to index time.**

## 6.2 Query decomposition — real, modest, and stage-dependent

**Claim (the best local-hardware ablation available): on Qwen2.5-7B-Instruct via Ollama on one RTX A6000, Qdrant + FastEmbed + BM25, HotpotQA distractor dev, n=5,000:**

| Variant | EM | F1 | Latency (ms) |
|---|---|---|---|
| Baseline (single-pass dense) | 43.1 | 54.0 | **546** |
| Agentic full (3 steps) | 53.2 | 61.6 | 5,642 |
| no-decomposition | 51.9 | 60.1 | **2,546** |
| no-reranker | 51.5 | 59.4 | 5,648 |
| steps-1 | 46.1 | 53.9 | 5,426 |
| steps-2 | 52.9 | 61.3 | 5,628 |
| **hybrid-only (no adaptive routing)** | **55.0** | **63.5** | 5,688 |

Key results: decomposition is worth **−1.4 EM when removed (p=0.004) but removing it halves latency**; **"two retrieval iterations capture 95% of the gain from five" (p<0.001)** with the 1→2 jump worth +7.1 EM; reranking worth −1.7 EM at negligible latency. Verbatim recommendation: *"prefer fixed hybrid retrieval over rule-based routing, and cap retrieval depth at two or three steps."* Decomposition is *"the first component to drop under a tight latency budget."*
→ https://arxiv.org/abs/2606.21553 (2026-06-19) — ⚠️ *single-author preprint, unrefereed, but methodologically clean and the closest published match to a self-hosted stack.*
**Confidence: emerging.**

**Claim (the most useful design rule of 2026): decomposition at initial retrieval "frequently harms performance due to semantic dilution," but "substantially improves reranking" via fine-grained constraint verification. Fix: keep the full query for first-stage retrieval, use sub-queries only at rerank.**
→ https://arxiv.org/abs/2606.08577 (2026-06-07), benchmarks MultiConIR + SSRB, code at github.com/EIT-NLP/Query-Decompose
**Confidence: emerging.**

**Claim: bandit-style adaptive decomposition (retrieve one doc at a time, update belief per sub-query) gives +35% document-level precision and +15% α-nDCG.**
→ https://arxiv.org/abs/2510.18633 (2025-10-21), **published EACL 2026** — https://aclanthology.org/2026.eacl-long.322.pdf
**Confidence: established (peer-reviewed).** Note this is adaptive *budget allocation*, not naive fan-out.

**Claim (conflicting): in a structured DevOps domain, agentic decomposition gave +0.04 overall / +0.17 MRR, but on MuSiQue multi-hop **ranking precision declined**. Conclusion: "agentic enhancements are not universally beneficial and must be applied selectively."**
→ https://arxiv.org/abs/2606.05658 (2026-06-04)
**Confidence: emerging; conflicts in sign with the HotpotQA ablation — genuinely contested.**

## 6.3 Multi-query expansion — the weakest of the three

**Claim: on T2-RAGBench, multi-query scored Recall@5 0.640 / MRR@3 0.397 — *worse than plain BM25* (0.644 / 0.411).**
→ https://arxiv.org/html/2604.01733v1 (2026-04-02) — **established for numeric/table domains.**

**Claim (the strongest negative result): prompt-only LLM query rewriting (Ministral-8B rewriter, MPNet + BGE-base retrievers) produced FiQA **−9.0% nDCG@10 (p<0.001)**, TREC-COVID +5.1% (p=0.024, marginal after correction), SciFact no effect. Mechanism: vocabulary-overlap decline on FiQA; gains correlate with term standardization toward corpus vocabulary. Crucially, **an attempt to gate the rewriter achieved only AUC 0.593, with an oracle gating ceiling of ~+3 pp.**
→ https://arxiv.org/html/2603.13301 (Varun Kotte, Adobe, 2026-03-02)
**Confidence: established for the negative result.** This is the strongest evidence that *"just gate your rewriter"* is currently hard to execute.

**Claim (the efficient alternative): STORM trains 0.6B–8B models with BM25-score rewards to generate lexical expansions via reward-guided beam search. At 8B it "rivals far larger proprietary rewriters," matches/surpasses competitive LLM rewriters on TREC DL + BEIR, **retains BM25-level speed**, and zero-shots to 18 languages (MIRACL). Explicitly positioned as "infrastructure-light" — no corpus re-encoding.**
→ https://arxiv.org/abs/2606.10621 (2026-06-09)
**Confidence: emerging.** Directly contradicts Jina's 2025 claim that query rewriting "rules out small LMs."

## 6.4 Self-query / structured metadata filters — the quiet winner

**Claim: LLM→structured-filter translation is near-perfect on easy/medium queries and collapses on comparative/aggregate ones. F1 at optimal threshold on a Chroma-backed nutrition corpus:**

| Difficulty | Gemini-2.0-Flash | Claude-Sonnet-4 | GPT-4o | Mistral Medium 3 |
|---|---|---|---|---|
| Easy | 0.999 | 0.999 | 0.999 | 0.999 |
| Medium | 1.000 | 1.000 | 0.990 | 0.998 |
| **Hard** | **0.445** | **0.450** | **0.396** | **0.425** |

Verbatim: *"Even without fine-tuning, an open-source LLM such as Mistral can serve as a highly reliable metadata filter generator."* Failure mode is structural — queries whose constraints exceed the metadata schema's representational scope — not model capacity.
→ https://arxiv.org/html/2603.09704v2 (2026-03-11)
**Confidence: established for easy/medium; emerging for the hard-tier collapse.**

**This is the one query-planning technique where a small local model is already sufficient.** Unlike HyDE (needs generation quality) or rewriting (needs corpus-vocabulary knowledge), filter extraction is a constrained-decoding classification problem.

**Claim: vector-DB vendors ship this as a product default. Weaviate's Query Agent auto-extracts typed filters (documented example: `IntegerPropertyFilter(property_name='price', operator=LESS_THAN, value=200.0)`), routes across collections, and returns provenance. Qdrant's 2026 engineering investment went into filtered vector search performance (ACORN analysis, 2026-07-27) — which only matters if filter extraction is a mainline path.**
→ https://weaviate.io/blog/query-agent (2025-03-05) · https://qdrant.tech/blog/
**Confidence: established.**

## 6.5 What to ship

| Technique | Default? | Expected gain | Cost |
|---|---|---|---|
| Hybrid BM25+dense, RRF | **YES** | +0.051 R@5 over BM25, +0.108 over dense | 0 extra LLM calls |
| Cross-encoder rerank | **YES, unconditionally** | **+17.2 pp MRR@3** | negligible |
| Self-query metadata filters | **YES where a schema exists** | ~0.999 F1 easy/medium | 1 constrained decode |
| Index-time contextual enrichment | **YES** | +2.2 pp R@5 (T2-RAGBench); 35–67% failure reduction (Anthropic) | offline only |
| Query decomposition | **Conditional** — multi-hop, applied at *rerank* | +1.4 EM local 7B; +35% doc precision w/ bandit | **~2× latency** |
| Multi-query expansion | **NO** | ≈0 to negative | +1 LLM call + N× retrieval |
| HyDE | **NO — gate post-retrieval** | −7.3% (numeric) to positive (narrative) | +25–60% latency on small models |

**Two rules that generalize:** (1) **spend on the reranker before the query** — every 2026 head-to-head puts reranking's ROI an order of magnitude above any query transformation; (2) **escalate, don't pre-decide** — pre-retrieval routing is structurally unreliable; cheapest-first cascades win.

---

# 7. ROUTING / TRIAGE

## 7.1 Small-model routers: lexical beats semantic, and no LLM is needed

**Claim: for RAG-*strategy* routing, TF-IDF + SVM (RBF) beats sentence embeddings and every neural option tested. On RAGRouter-Bench (7,727 queries, 4 corpora, 5 paradigms), 15 classifier × feature combinations, 5-fold CV:**

| Classifier | TF-IDF Acc / F1 | MiniLM Acc / F1 | Structural Acc / F1 |
|---|---|---|---|
| **SVM (RBF)** | **93.2 / 0.928** | 90.3 / 0.897 | 79.1 / 0.774 |
| MLP | 92.7 / 0.923 | 90.1 / 0.896 | 80.3 / 0.788 |
| Logistic Reg. | 92.1 / 0.918 | 86.8 / 0.864 | 78.1 / 0.763 |
| Majority class | 52.9 / 0.231 | — | — |

Token savings vs always-IterativeRAG: TF-IDF+SVM **28.1% at 0.928 F1**; perfect-label ceiling 35.2%. **Critical warning: the majority-class baseline (always route to the cheapest paradigm) achieves 60% savings at 0.231 F1 — "optimising savings in isolation simply collapses routing to the cheapest paradigm."** Domain macro-F1 spread: legal 0.967, literature 0.951, Wikipedia 0.926, medical 0.803.
→ https://arxiv.org/abs/2604.03455 (2026-04-03) — ⚠️ *preprint, in-distribution only; authors flag OOD as untested.*
**Confidence: emerging, but cheap and directly actionable.** Stated reason lexical wins: dense embeddings conflate surface-similar but type-different queries.

**Underlying benchmark: RAGRouter-Bench establishes relative token costs — LLM-Only 1.0× / NaiveRAG 1.4× / GraphRAG 2.1× / HybridRAG 2.8× / IterativeRAG 3.5× — and finds "no one-size-fits-all paradigm exists," with optimal selection driven by *query–corpus interaction* (corpus connectivity, density, intrinsic dimension, hubness), not query text alone.**
→ https://arxiv.org/abs/2602.00296 (2026-01-30, rev 2026-04-04)
**Confidence: established as the reference benchmark; new and not yet replicated.**

**Claim: generative SLMs used zero-shot as front-door routers do not meet production bars. On a 60-case benchmark: Qwen2.5-3B 0.783–0.793 accuracy @ ~1,000 ms median; Phi-3.5-mini (3.8B) 0.717 @ 5,772 ms; **Qwen2.5-1.5B collapses to 0.400**; DeepSeek-V3 (671B) 0.830. Verbatim: "No model meets the standalone viability criterion (≥0.85 accuracy, ≤2,000 ms P95)."**
→ https://arxiv.org/html/2604.02367 (2026-03-26) — ⚠️ *tiny benchmark (neff=60).*
**Confidence: emerging/weak, but directionally clear: 3B is the practical floor for *generative* routing.**

**⚠️ Direct answer on "sub-1B routers like Qwen3-0.6B": no primary 2026 source benchmarks a sub-1B *generative* model as a retrieval router.** Qwen3-0.6B's official positioning is embedding/reranking (https://huggingface.co/Qwen/Qwen3-Reranker-0.6B). What the evidence supports: **use a discriminative sub-1B router** — TF-IDF+SVM (no neural net at all, 0.928 F1) or MiniLM-L6-v2 (22M params, CPU-only, 0.897 F1).

## 7.2 Embedding-similarity routers

- **Aurelio Labs `semantic-router`** — https://github.com/aurelio-labs/semantic-router, actively maintained (2,376 commits, fetched 2026-08-08). Define `Route` objects by example utterances, encode, cosine-match, route or return `None`. **No published latency benchmarks on the repo.** *established that it exists; unsupported quantitatively.*
- **Counter-evidence:** MiniLM sentence embeddings **lose to TF-IDF by 3.1 macro-F1** on RAGRouter-Bench. *emerging.*
- RouterBench's 2024 finding that "KNN and MLP routers on sentence embeddings are competitive" (https://arxiv.org/abs/2403.12031, 2024-03) is **still the most-cited data point and has not been refreshed.**

## 7.3 LLM-as-router
Lineage: RouteLLM (ICLR 2025), FrugalGPT (TMLR 2024), Hybrid LLM (ICLR 2024 — cuts large-model calls 40% with no quality loss using a DeBERTa router). **The consistent four-year pattern: a small discriminative classifier is the right tool; an LLM router is rarely justified. Confidence: established.**

## 7.4 When industry skips routing entirely — increasingly, yes

Three independent 2026 results converge:

1. **Rule-based retriever routing loses to fixed hybrid.** Router rules were "named entities/dates/numbers → BM25; short or conceptual → dense; else hybrid." Failure: named entities appear in nearly every HotpotQA sub-question, so **72% of retrieval calls went to BM25**, and **hybrid-only beat the adaptive pipeline by +1.8 EM / +1.9 F1 (p<0.001)**. → https://arxiv.org/abs/2606.21553 (2026-06-19)
2. **Pre-retrieval routing for augmentation is structurally impossible.** Four ML approaches all failed; the cascade replacement gave **+0.140 quality, −31.8% latency, zero training.** → https://arxiv.org/abs/2605.27220 (2026-05-26)
3. **Simple baselines match learned routers.** *"Retrieval harm is non-negligible"*; *"router rankings vary across datasets and budgets"*; **"simple uncertainty or retrieval-score baselines often rival learned utility routers"**; *"nominal thresholds frequently miss target usage rates."* → https://arxiv.org/abs/2607.24010 (2026-07-27)

**Where routing still pays:** *paradigm/depth* selection when paradigms have genuinely different cost profiles (1.0×–3.5×), where 28.1% token savings at 0.928 F1 is real money.

**Recommendation: skip routing for "which retriever" — always hybrid. Consider routing for "which pipeline depth," implemented as a post-retrieval cascade rather than a pre-retrieval classifier. Confidence: emerging→established (three independent 2026 sources agreeing).**

## 7.5 Adaptive retrieval gating — alive, but its job description changed

**Claim (the key 2026 result): adaptive retrieval underwent a role shift — it is a *noise filter* for weak models and an *efficiency optimizer* for strong ones. Across 8 backbones × 3 datasets (ASQA/QAMPARI/ELI5 via ALCE):**

| Backbone | Vanilla-0 | Vanilla-10 | Rerank-10 | AdaRankLLM | Oracle |
|---|---|---|---|---|---|
| Alpaca-7B | 12.55 | 18.01 | 17.83 | **20.15** | 33.35 |
| Mistral-Instruct | 18.33 | 26.02 | 25.66 | **26.71** | 42.23 |
| Qwen2.5-7B-Inst | 19.52 | 30.00 | **30.05** | 29.42 | 41.58 |
| Qwen3-8B-NoThinking | 20.37 | **32.72** | 32.47 | 32.14 | 43.81 |
| Qwen3-8B-Thinking | 21.85 | 34.18 | **34.43** | 32.75 | 45.17 |
| GPT-4o | 34.50 | 35.26 | 33.62 | **35.60** | 49.09 |

Verbatim: for weak models it "acts as a critical noise filter" (Alpaca/Mistral peak at k=1–3 and degrade with larger sets); for strong models it "serves as an efficiency optimizer," with Qwen3-8B-Thinking's Vanilla-10 often best because explicit reasoning acts as *"a potent internal verification mechanism."* Also **"The Fallacy of Static Retrieval"** — optimal k is *"highly volatile"* across task, dataset and backbone. **The oracle gap persists everywhere** (34.43 vs 45.17): *"the fundamental challenges of RAG remain far from being fully resolved."*
→ https://arxiv.org/abs/2604.15621 (2026-04-17), code at github.com/USTC-StarTeam/adaptive-listwise-ranking-rag
**Confidence: established.** **Direct implication: on a reasoning-capable local model (Qwen3-8B-Thinking class), adaptive filtering buys you *tokens*, not *accuracy*.**

**Claim: training-free gating now matches trained gates. TARG generates a short no-context draft, computes mean token entropy / top-1-vs-top-2 logit margin / small-N variance, and retrieves only above threshold. Model-agnostic, no training, no auxiliary heads. Result: matches or improves EM/F1 vs Always-RAG while **reducing retrieval 70–90%**, at latency near Never-RAG.**
→ https://arxiv.org/abs/2511.09803 (2025-11-12, v2 through 2026-04-14)
**Confidence: emerging. Directly implementable on a local model since you own the logits.**

**Claim: confidence-based gating recovers 95% of the maximum accuracy gain using 58% of retrieval operations on TriviaQA. Qwen3-4B calibration improves AUROC 0.806→0.879, calibration error 0.163→0.034. **Core theoretical result: SFT produces well-calibrated confidence (MLE); RL (PPO/GRPO/DPO) induces overconfidence via reward exploitation.** Fix: post-RL SFT with self-distillation.**
→ https://arxiv.org/abs/2603.06604 (2026-02-18)
**Confidence: emerging.** ⚠️ **Serious, under-appreciated warning: if you deploy an RL-trained search agent (Search-R1 lineage) *and* a logit-based retrieval gate, the RL training actively breaks the gate.**

**Claim: adaptive RAG routers are brittle to surface query perturbations — "surface query changes dramatically alter retrieval decisions."** → https://arxiv.org/abs/2604.10745 (2026-04). *emerging.*

**Lineage status (all confirmed alive and cited in 2026 work):** Self-RAG (ICLR 2024) → FLARE (EMNLP 2023) → SKR (EMNLP 2023) → Adaptive-RAG (NAACL 2024) → CRAG (2024) → Probing-RAG (NAACL Findings 2025) → CTRLA/TARG (2025) → AdaRankLLM (2026).

## 7.6 Routing benchmarks

| Benchmark | URL | Date | Scope |
|---|---|---|---|
| RouterBench | https://arxiv.org/abs/2403.12031 | 2024-03 | 405k inference outcomes, LLM routing. Still the reference. |
| RouterArena | https://arxiv.org/abs/2510.00202 | 2025-09-30 | Open router leaderboard. **No 2026 update found.** |
| **RAGRouter-Bench** ⭐ | https://arxiv.org/abs/2602.00296 | 2026-01-30 | **First benchmark for RAG-strategy routing** |
| KARLBench | https://arxiv.org/abs/2603.05218 | 2026-03-05 | Enterprise search agents, 6 regimes |
| OrchestraBench | https://arxiv.org/abs/2608.05263 | 2026-08-05 | Multi-agent orchestration w/ failure injection |

**Scope note: the RouterBench/RouterArena lineage is about *model* routing, not *retrieval* routing. RAGRouter-Bench (Jan 2026) is the first benchmark answering the retrieval question.**

---

# 8. AGENTIC LOOP PATTERNS

## 8.1 The four patterns, measured

**Pattern A — retrieve-then-generate.** Local 7B HotpotQA: **EM 43.1 / F1 54.0 / 546 ms.** LangChain frames this as "high control / low flexibility / fast."
**Pattern B — ReAct-style search loop.** Same hardware: **EM 53.2 / F1 61.6 / 5,642 ms** at 3 steps; hybrid-only variant **EM 55.0 / F1 63.5**. → **+11.9 EM for ~10× latency.** Depth ≥3 buys nothing.
→ https://arxiv.org/abs/2606.21553 (2026-06-19) · https://docs.langchain.com/oss/python/langchain/retrieval

**Pattern C — plan-execute (plan before search).** Decompose into ordered sub-questions *before any retrieval*, so each search step is "anchored to a pre-designed sub-question instead of drifting under the influence of partially relevant documents." Validated at **3B–14B across three model families**, using a **self-bootstrapping paradigm** (small seed models generate filtered trajectories) rather than frontier distillation — notably practical for self-hosting.
→ https://arxiv.org/abs/2605.28354 (2026-05-27). *emerging.*

**Pattern C′ — parallel evidence acquisition ("search more, think less").** Replaces sequential reasoning with parallel evidence acquisition under a constrained context budget. **BrowseComp 48.6 / GAIA 75.7 / Xbench 82.0 / DeepResearch Bench 45.9**, with **−70.7% average reasoning steps on BrowseComp while improving accuracy.**
→ https://arxiv.org/abs/2602.22675 (2026-02-26). *emerging; base model size not stated.*

**Empirically-grounded ordering for a local 7B–32B: A ≪ B < C/C′ ≤ D, with the A→B jump by far the largest and cheapest.**

## 8.2 RL-trained search agents

**Lineage (all primary):** DeepRetrieval (https://arxiv.org/abs/2503.00223, 2025-02-28, **3B, 65.07% recall on publication search, beats GPT-4o**) → R1-Searcher (https://arxiv.org/abs/2503.05592, 2025-03-07) → Search-R1 (https://github.com/PeterGriffinJin/Search-R1; Llama-3.2-3B / Qwen2.5-7B bases; last notable milestone Oct 2025 — **now a reference implementation, not SOTA**) → ZeroSearch (https://arxiv.org/abs/2505.04588, 2025-05-07, **7B simulation module ≈ real search engine**) → R1-Searcher++ (https://arxiv.org/abs/2505.17005, 2025-05-22).

**2026 successors (selected):**

| Paper | arXiv | Date | Contribution |
|---|---|---|---|
| CuSearch | 2605.11611 | 2026-05-14 | Curriculum rollout sampling; **+11.8 EM over standard GRPO** |
| R²-Searcher | 2606.28566 | 2026-06-26 | Reasoning-reflection RL, tree exploration, 7 multi-hop benchmarks |
| GraphPO | 2606.18954 | 2026-06-17 | Rollouts as DAGs, merges equivalent paths, variance reduction |
| OASES | 2604.03675 | 2026-04-04 | Co-trains search policy + state evaluator |
| Hide to Guide (SMEPO) | 2605.25198 | 2026-05-24 | Semantic masking prevents reward hacking; **+3.2 pts over GRPO** |
| **DEEPRUBRIC** | 2606.17029 | 2026-06-15 | Evidence-tree rubrics; **8B ≈ prior SOTA at ~13× fewer RL GPU-hours** |
| Speculate While You Reason | 2607.25816 | 2026-07-28 | Qwen3-4B/4.5B; **next-tool-call Hit@1 44–49% → 61–66%** |
| **MAPD** | 2607.24280 | 2026-07-27 | Distills a **style-normalized JSON protocol** (task type + plan + grounding facts), not logits. **Qwen3-1.7B 39.4%, Qwen3-4B 44.4%** across 7 QA benchmarks; "consistently outperforms competitive distillation and RL" |

**Meta-observations:** (1) the field moved from **outcome rewards** (2025) to **credit assignment** (2026) — step-level, graph-based, tree-based, rubric supervision; (2) **training cost collapsed** (~13× fewer GPU-hours for equal quality); (3) **local simulation is the cost story** — LiteResearcher logged **73.2M tool calls (45.8M searches + 27.4M browses)** during RL, which would cost **$59K–$243K** online but **zero marginal cost** locally, with **10–46× latency speedup** and no environmental noise (https://arxiv.org/html/2604.17931v5, 2026-07-28). **Direct validation of the self-hosted thesis.**
**Confidence: established for the lineage map; emerging for individual numbers (mostly unrefereed preprints).**

## 8.3 The single most important benchmark result for self-hosters

**Claim: on BrowseComp-Plus (fixed, human-verified corpus with controlled retrieval, so the retriever and the agent are disentangled): Search-R1 + BM25 scores 3.86%; GPT-5 + BM25 scores 55.9%; GPT-5 + Qwen3-Embedding-8B scores 70.1% — with *fewer* search calls.**
→ https://aclanthology.org/2026.acl-long.1023/ (**ACL 2026 Main**) · https://github.com/texttron/BrowseComp-Plus · original https://arxiv.org/abs/2508.06600
**Confidence: established (peer-reviewed).**
**→ Swapping the retriever moved the same agent +14.2 points. This is the strongest published evidence that for a self-hosted agent, retriever quality dominates agent scaffolding.** Fix your embedder before you build a loop.

## 8.4 Chroma Context-1 — the best-documented open artifact for this use case

**Claim: Context-1 is a purpose-trained, open-weight (Apache 2.0) retrieval subagent. 20B derived from gpt-oss-20b, MXFP4-quantized. SFT on Kimi K2.5 trajectories → RL via CISPO (a GRPO variant), ~230 steps, 8,000+ synthetic tasks over web/finance/legal/email, curriculum from recall-weighted (16:1) to precision-weighted (4:1). A `prune_chunks` tool lets it discard irrelevant retrieved docs mid-search within a fixed 32.8k token budget. It returns ranked supporting documents to a downstream answering model, "cleanly separating search from generation."**

| Benchmark | Context-1 (1×) | Context-1 (4×) | Frontier baselines |
|---|---|---|---|
| Web (difficulty 2+) | 0.88 | 0.97 | 0.95–0.99 |
| Finance | 0.64 | 0.82 | 0.65–0.90 |
| Legal | 0.89 | 0.95 | 0.90–0.98 |
| Email (held out) | 0.92 | 0.98 | 0.93–0.98 |
| **BrowseComp-Plus** | **0.87** | **0.96** | 0.82–0.94 |

**10× faster inference at frontier-comparable quality; 400–500 tok/s on B200.** Behavior deltas vs base: 2.56 tool calls/turn (vs 1.52), trajectory 6.7→5.2 turns, prune accuracy 0.941 (vs 0.824), parallel tool calling. Explicitly contrasts itself with Anthropic's parallel-subagent design: **a single specialized retrieval subagent paired with a frontier reasoner**, avoiding running multiple frontier models. Differs from MemGPT (external paging) and ReSum (lossy summarization) by doing **selective document-level retention without compression**, preserving evidence fidelity.
→ https://www.trychroma.com/research/context-1 (2026-03-26)
**Confidence: emerging** (vendor technical report, self-reported) **but it has open weights, open data-generation code, and an Apache-2.0 license — the highest-quality artifact in this survey for self-hosted purposes.**

## 8.5 Open-weight deep-research agents at 4B–32B

| System | Base | GAIA | BrowseComp | FRAMES | xbench-DS |
|---|---|---|---|---|---|
| LiteResearcher-4B ¹ | Qwen3-4B-Thinking-2507 | **71.3** | 27.5 | **83.1** | **78.0** |
| Tongyi DeepResearch-30B-A3B ² | 30.5B MoE / 3.3B active | 70.9 | — | — | — |
| Claude-4.5-Sonnet (ref) | closed | 71.2 | — | 80.7 (Claude-4) | — |
| SMTL ³ | unstated | 75.7 | 48.6 | — | 82.0 |

¹ https://arxiv.org/html/2604.17931v5 (2026-07-28), 64K context + memory compression. ² https://github.com/Alibaba-NLP/DeepResearch (released 2025-09-17, **Apache-2.0**, 128K context; **no 2026 successor announced as of 2026-08-08**). ³ https://arxiv.org/abs/2602.22675 (2026-02-26).

**Headline: a 4B open model now matches Claude-4.5-Sonnet on GAIA-Text (71.3 vs 71.2) and beats Claude-4-Sonnet on FRAMES (83.1 vs 80.7). Confidence: emerging — single-source, unrefereed, self-reported. Treat as an upper bound.**

**⚠️ Flagged unreliable — do not cite:** https://arxiv.org/abs/2607.27562 reports HLE 87.3%, BrowseComp-ZH 85.3%, WebWalkerQA 91.2% — implausible relative to every other 2026 result and uncorroborated.

## 8.6 Subagent fan-out — now genuinely contested

**The canonical pro-fan-out source:** Anthropic's orchestrator-worker design, where **multi-agent (Opus 4 lead + Sonnet 4 subagents) outperformed single-agent Opus 4 by 90.2%** on a breadth-first research eval; **token usage alone explains 80% of performance variance**; agents ≈ 4× chat tokens, **multi-agent ≈ 15× chat tokens**; parallel tool calling + simultaneous subagent spawning cut research time **up to 90%**. Lesson: *"Agent-tool interfaces are as critical as human-computer interfaces."*
→ https://www.anthropic.com/engineering/multi-agent-research-system (2025-06-13)
**Confidence: established as an engineering account.** ⚠️ 14 months old, frontier models, and **the 15× token multiplier is disqualifying for most self-hosted budgets.**

**The 2026 counter-evidence:** on repository-level code QA, **plain semantic search scored 65.2% while deep agentic search scored 46.2% at >2× cost.** **41.8% of deep-agentic failures occurred at the planner→subagent hand-off**, and these were *"usually silent, ending in a fluent and confident answer that was wrong."* For read-only, indexable questions, *"retrieval was the stronger and cheaper option"* — despite deep agentic search being *"now the preferred design in many code agents."*
→ https://arxiv.org/abs/2608.01507 (2026-08-02)
**Confidence: emerging (single domain), but it is the first quantified measurement of the subagent hand-off as a failure surface and it directly contradicts the default 2025 recipe.**

**Corroborating — more search ≠ better answers:** across six agents on BrowseComp-Plus, **"search volume correlates weakly with answer quality"**; accuracy correlates better with **cumulative retrieval recall**; useful information usually appears **early** but agents keep searching; redundant queries characterize *underperforming* agents while exploratory *reformulations* remain valuable. **Recommendation: stopping criteria based on cumulative evidence sufficiency, not query count.**
→ https://arxiv.org/abs/2608.01913 (2026-08-03). *emerging, strong methodology.*

**Production escalation architecture (named industrial deployment, Ontario Power Generation, IEEE SEGE 2026):** four deployed stages — naive RAG → hybrid + rerank → agentic function-calling retrieval → deep multi-agent with code synthesis and explicit planning. Principle **PEA-CAE**: *"begin with low-cost, high-precision retrieval and escalate to full-document reads only when the expected evidence gain justifies latency and cost."* Also: *"context engineering is a more tractable and economically viable path than domain-specific fine-tuning."* ⚠️ No published metrics.
→ https://arxiv.org/abs/2607.24791 (2026-06-28)

## 8.7 Context engineering

**Anthropic's four pillars:** (1) **compaction** — summarize near limits, preserving "architectural decisions, unresolved bugs, and implementation details"; (2) **structured note-taking** — external memory files; (3) **sub-agent architectures** — a subagent burns "tens of thousands of tokens or more, but returns only a condensed, distilled summary (often 1,000–2,000 tokens)"; (4) **just-in-time retrieval** via lightweight identifiers instead of pre-loading. Underpinning: **"context rot"** — accuracy degrades with token count.
→ https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents (2025-09-29) · empirical basis https://www.trychroma.com/research (Context Rot, 2025-07). *established.*

**2026 quantified follow-ups:** selective retention + automated summarization → **91.6% task completion at −63% tokens** (https://arxiv.org/abs/2606.10209, 2026-06-08); SmoothAgent lookahead → **up to 11.9× TTFT reduction** (https://arxiv.org/abs/2607.00151, 2026-06-30); SIEVE structure-aware Boolean retrieval → **higher accuracy at 20.7–50.6% fewer tokens** (https://arxiv.org/abs/2608.02751, 2026-08-03). *all emerging.*

**⚠️ Robustness warning that should gate any open-web deep-research deployment:** a **single misleading document raises false-conclusion adoption from 0% to 54.7%** (https://arxiv.org/abs/2607.20891, 2026-07-23); a single misleading document causes **66–88 pp accuracy drops** across agents (https://arxiv.org/abs/2607.17291, 2026-07-19); research-trajectory hijacking achieves 26.4% PRISM with five injected documents (https://arxiv.org/abs/2607.04718, 2026-07-06). **Three independent 2026 sources agree on the magnitude.** Lower risk over a trusted local corpus; dominant failure mode for anything touching the open web.

## 8.8 Recommended architecture for a self-hosted 7B–32B, by measured ROI

**Tier 1 — do first (large gain, near-zero cost):**
1. **Hybrid BM25 + dense with RRF, always. No retriever routing.** (+0.108 R@5 over dense; +1.8 EM over rule-based routing)
2. **Cross-encoder reranker, unconditionally.** (+17.2 pp MRR@3; +1.7 EM at negligible latency)
3. **Invest in the retriever before the agent.** (BM25 → Qwen3-Embedding-8B = **+14.2 points on BrowseComp-Plus**)
4. **Index-time contextual enrichment.** (offline cost only)

**Tier 2 — the loop:**
5. **Iterative retrieve-reason loop capped at 2–3 steps.** Step 1→2 is +7.1 EM; 3→5 is noise.
6. **Terminate on cumulative evidence sufficiency, not query count.**
7. **Plan-first over pure ReAct** if you can afford one planning call — validated at 3B–14B with self-bootstrapped training requiring no frontier teacher.

**Tier 3 — query planning, selectively:** self-query filters ON; decomposition for multi-hop applied **at the rerank stage**; HyDE OFF by default, wired as a post-retrieval fallback; multi-query OFF (use a STORM-style 0.6B–8B BM25-reward expander instead).

**Tier 4 — adaptive gating (cost-motivated only):** TARG-style prefix-logit gating gives **70–90% retrieval reduction at matched EM/F1** with no training — essentially free when you own the logits. But on a Qwen3-8B-Thinking-class model it buys **tokens, not accuracy**. ⚠️ **Do not combine an RL-trained search agent with a logit-confidence gate without post-RL SFT recalibration.**

**Tier 5 — subagents:** prefer **one specialized retrieval subagent + one reasoner** over parallel fan-out (Chroma Context-1's explicit design choice, Apache-2.0, 20B, 10× speedup, 0.87–0.96 BrowseComp-Plus). If you do fan out, **instrument the planner→subagent hand-off** — 41.8% of failures, silently. **Self-editing context (prune mid-search under a fixed ~32.8k budget) is the 2026 replacement for lossy summarization.**

**What NOT to do:** always-on HyDE; always-on multi-query; rule-based retriever routing on entity heuristics; loops deeper than 3; pre-retrieval classifiers for "do I need augmentation"; optimizing a router on cost savings alone (60% savings at 0.231 F1 is the degenerate solution).

---

# 9. VERIFICATION / GROUNDING + CITATIONS

## 9.1 The 2024 baseline is now the floor

**Claim: MiniCheck's original result — LLM-AggreFact balanced accuracy without per-dataset threshold tuning: AlignScore 70.4, MiniCheck-RoBERTa 72.7, MiniCheck-DeBERTa 72.6, MiniCheck-Flan-T5 74.7, GPT-4 75.3. Inference cost on the 13K test set: AlignScore $0.20, MiniCheck-FT5 $0.24, GPT-4 $107 (~400× cheaper).**
→ https://arxiv.org/html/2404.10774v1 (EMNLP 2024). **established, but a 2024 result — the floor, not the SOTA.**

**Claim: current LLM-AggreFact leaderboard top-5 by average BAcc: Bespoke-MiniCheck-7B 77.4, Claude-3.5 Sonnet 77.2, IBM Granite Guardian 3.3 (8B) 76.5, Mistral-Large-2 (123B) 76.5, gpt-4o 75.9. 39 models × 11 datasets.**
→ https://llm-aggrefact.github.io/ (fetched Aug 2026)
**Confidence: established for the numbers; contested as a *current* signal — the page shows no 2026 refresh and the newest datable entry is Aug 2025.**

**Claim: the MiniCheck project is dormant.** Newest repo news is Sep 2024. Models: MiniCheck-RoBERTa-Large 355M, MiniCheck-DeBERTa-v3-Large 434M, MiniCheck-Flan-T5-Large 770M, Bespoke-MiniCheck-7B. Repo Apache-2.0, but **Bespoke-MiniCheck-7B requires contacting company@bespokelabs.ai for commercial licensing.** Throughput: 29,000 instances in ~30 min on one A6000 48GB with prefix caching. Operates at (document, single sentence) granularity — you must sentence-split claims yourself.
→ https://github.com/Liyan06/MiniCheck. *established.*

## 9.2 The 2026 successor: a 1B model beat the 7B SOTA

**Claim: ThinknCheck — a 1B-parameter Gemma3-1B, 4-bit quantized verifier that emits a short structured rationale then a binary verdict — reaches 78.1 BAcc on LLM-AggreFact, beating Bespoke-MiniCheck-7B (77.4) with 7× fewer parameters. On SciFact it hits 64.7 BAcc, +14.7 absolute over MiniCheck-7B. Ablating the reasoning step collapses it to 57.5 BAcc — the rationale is doing the work, not the fine-tune. Trained on LLMAggreFact-Think (24.1k reasoning-augmented examples). License CC0.**
→ https://arxiv.org/abs/2604.01652 (Rao, Han, Callison-Burch, UPenn; 2026-04-02)
**Confidence: emerging** (single paper) **but this is the single most important §9 change of the year: a 4-bit 1B verifier now beats the 7B 2024 SOTA, which makes per-answer grounding verification cheap enough to always run.**

**Claim: Paladin-mini (3.8B) reports avg BACC 79.31% vs 77.86% prior SOTA. ⚠️ Licensed CC BY-NC-ND 4.0 — non-commercial, no derivatives.**
→ https://arxiv.org/abs/2506.20384 (2025-06-25). *emerging.*

**Claim: Granite Guardian 3.3 8B (IBM, 2025-08-01, Apache 2.0) is the best *permissively licensed* off-the-shelf option on the public leaderboard: LLM-AggreFact avg BAcc 0.761 (AggreFact-CNN 0.669, REVEAL 0.894), and it also covers context relevance, answer relevance, jailbreak, and function-calling hallucination in one model. A 38M variant exists for low latency.**
→ https://huggingface.co/ibm-granite/granite-guardian-3.3-8b. *established.*

## 9.3 Span-level detection — where 2026 moved, and a nasty surprise

**Claim: LettuceDetect (ModernBERT token classification) — RAGTruth example-level F1 79.22%, +14.8% over Luna, ~30× smaller than the best models, 30–60 examples/sec on a single GPU.**
→ https://arxiv.org/abs/2502.17125 (2025-02-24). *established; a 2025 result, now beaten, but still the best latency/quality ratio in the encoder class.*

**Claim (the surprise): a fine-tuned Qwen3.5-2B span detector reaches 0.689 span-F1 on a unified benchmark spanning code, tool output, structured docs and NL RAG — versus **LettuceDetect-large at 0.17** and the **strongest zero-shot LLM judges at ≤0.22**. It stays competitive on classic NL benchmarks: 81.8 RAGTruth example-F1, 0.724 English PsiloQA IoU, 0.60 span-F1 on code-agent output. CC BY 4.0.**
→ https://arxiv.org/abs/2607.00895 (2026-07-01)
**Confidence: emerging.** **The headline finding matters enormously for agentic retrieval: encoder-era span detectors and zero-shot LLM judges both collapse (0.17–0.22 span-F1) once the "context" is code or tool output rather than prose. If your agent verifies tool outputs, off-the-shelf 2025 detectors will not work.**

**Claim: GASP — grounding-sensitivity-by-perturbation, three instruction-tuned scorers at 0.5B–1.7B, RAGTruth ~0.73 response-level AUC / ~0.67 span-level AUC, competitive with entailment verifiers without a dedicated verifier model.**
→ https://arxiv.org/abs/2607.04223 (2026-07-05). *emerging.*

**Claim: full-document (32K-token) verification substantially improves detection of unsupported responses versus chunk-truncated validation, because supporting evidence is frequently outside the truncated passage.**
→ https://arxiv.org/abs/2603.23508 (2026-03-04). *emerging.* **Practical implication: verifying against the retrieved chunk alone systematically under-detects.**

## 9.4 LLM-as-judge reliability — partially rehabilitated, with a metric bug

**Claim: a plain LLM-as-a-Judge baseline performs competitively against specialized SOTA hallucination detectors on TRIVIA+ (long-context RAG); current detectors have "ample room" for improvement on RAG-style benchmarks; label noise materially degrades measured detection performance.**
→ https://arxiv.org/abs/2605.11330 (2026-05-11, **ACL 2026 main**). *emerging, and directly contested against the "small NLI model beats the judge" framing.*

**Claim (standing caution): five SOTA factuality metrics across 11 datasets are inconsistent with each other and often misestimate system-level performance; systematically weak on highly paraphrased outputs and on outputs referencing distant source sections. Recommendation: manually validate any factuality metric in your own domain.**
→ https://arxiv.org/abs/2501.14883 (Godbole & Jia, 2025-01-24). **established; a 2025 result but nothing in 2026 overturns it.**

**Claim (the metric bug): faithfulness metrics measure only *precision* and ignore *recall/coverage*, so a system can score near-perfectly by saying almost nothing. On a 7,253-instance multilingual benchmark, the most *precise* frontier model covered under half the relevant facts and **ranks last by F1**; fine-tuned 1B–7B models reached ~0.98 F1, beating all zero-shot frontier systems.**
→ https://arxiv.org/abs/2606.09376 (2026-06-08, rev 2026-06-16)
**Confidence: emerging.** **If you only gate on "is every claim supported," you will train your system to abstain. Pair precision with a coverage metric.**

**Claim: Cleanlab published a direct head-to-head of LLM-as-a-Judge vs Prometheus, Lynx, HHEM and TLM across six RAG applications, all reference-free.**
→ https://arxiv.org/abs/2503.21157 (2025-03-27). **established that the comparison exists; ⚠️ I could not extract the numeric table — abstract has no numbers, HTML 404s, PDF unparseable. Treat the ranking as unverified.**

## 9.5 Claim decomposition

- **Standard pipeline:** decompose into atomic claims → retrieve evidence → verify each. Known failure: strictly atomic facts strip the context needed to verify them; "molecular facts" / decontextualization is the standard fix. → https://arxiv.org/pdf/2406.20079 (2024-06). *established.*
- **Faithfulness scores are sensitive to the decomposition method itself** — DecMetrics proposes structured scoring of decomposition quality as a prerequisite. → https://arxiv.org/pdf/2509.04483 (2025-09). *emerging.*
- **TriQua (2026-08-05)** resolves the granularity/context trade-off: simple claims → triples, complex claims → **hyperrelational facts with qualifiers**. TriQuaScore correlates strongly with human factuality annotations, outperforms existing decomposition frameworks, and localizes errors per-triple/per-qualifier. → https://arxiv.org/abs/2608.05228. *emerging.*

## 9.6 Citations

**Claim (the key architectural finding): post-hoc citation (P-Cite) achieves high coverage with competitive correctness and moderate latency; generation-time citation (G-Cite) prioritizes precision at the cost of coverage and speed. Across four attribution datasets, **retrieval quality is the main driver of attribution quality in both paradigms.** Recommendation: P-Cite-first for high-stakes domains; G-Cite only for precision-critical strict claim verification.**
→ https://arxiv.org/abs/2509.21557 (2025-09-25, rev 2025-12-18, NeurIPS 2025 LLM-Eval Workshop). *emerging.*

**Claim: deep-research agents cite badly. Link validity >94%, topical relevance >80%, but **factual support only 39–77%**. Factual accuracy drops **~42% on average as tool calls scale from 2 to 150**. **Fewer than half of open-source models successfully produce cited reports one-shot.**
→ https://arxiv.org/abs/2605.06635 (2026-05-07)
**Confidence: emerging.** **For a local model, citation generation is a separate capability you must verify, not something you get by prompting.**

## 9.7 Vectara HHEM lineage

- HHEM-2.1-**open** is the open-weight version; the leaderboard now runs on commercial **HHEM-2.3** and **FaithJudge**. → https://arxiv.org/abs/2505.04847 (EMNLP 2025 Industry). *established.*
- **The open version is materially weaker, especially at longer premises:** TofuEval-MediaSum BAcc **78.48 (2.3) vs 71.57 (2.1-open)**; RAGTruth-Summary recall **min 0.758 (2.3) vs max 0.377 (2.1-open)**. → https://www.vectara.com/blog/hallucination-detection-commercial-vs-open-source-a-deep-dive. *established (vendor self-report; direction credible, magnitude vendor-favorable).* **For a self-hosted stack this is a reason to prefer ThinknCheck / Granite Guardian / LettuceDetect over HHEM-2.1-open.**
- Next-gen leaderboard (2025-11-19) expanded to **7,700+ articles, up to 32,000 tokens**, 10 domains. Under the harder benchmark, hallucination rates rose sharply: Gemini-2.5-flash-lite leads at **3.3%**, Claude Sonnet 4.5 **>10%**. Feb 5 2026 update covers 80+ models. *established.*

---

# 10. CONTEXT COMPRESSION

## 10.1 Provence — the one variant where the economics work

**Claim: Provence (Naver Labs Europe, ICLR 2025) formulates context pruning as **sequence labeling unified with reranking**, dynamically detects how much to prune per context, works out-of-the-box across domains, and achieves "negligible to no drop in performance… at almost no cost in a standard RAG pipeline" — because pruning is folded into the reranker you already run. DeBERTa-v3 trained on Llama-3-8B synthetic sentence-relevance labels, plus self-distillation to preserve reranking.**
→ https://arxiv.org/abs/2501.16214 (2025-01-27) · https://huggingface.co/blog/nadiinchi/provence · `naver/provence-reranker-debertav3-v1`
**Confidence: established.** **The "zero marginal cost" property is the whole argument** — it sidesteps the latency objection that kills LLMLingua.

**Claim: XProvence (ECIR 2026) extends this to 16 trained languages generalizing to 100+, "minimal-to-no performance degradation," outperforming strong baselines on four multilingual QA benchmarks. Weights: `naver/xprovence-reranker-bgem3-v2`.**
→ https://arxiv.org/abs/2601.18886 (2026-01-26). *emerging.*

**Claim: OpenProvence is a fully MIT-licensed ModernBERT reimplementation at 30M / 130M / 149M / 310M, English + Japanese. Large @ threshold 0.10 on MLDR-English: **93.10% Has-Answer with 94.38% positive-passage compression and 99.90% negative-passage compression**, matching the `naver/provence` baseline. Weights, training code, inference code and dataset tooling all MIT.**
→ https://github.com/hotchpotch/open_provence
**Confidence: emerging** (single-author, self-reported) **but this is the one to actually ship in a self-hosted stack** — 30M–310M, MIT, and it replaces a pipeline stage rather than adding one.

## 10.2 LLMLingua lineage — effectively frozen

**Claim: LLMLingua-2 offers 3×–6× speedup over LLMLingua via GPT-4-distilled token classification; the family claims up to 20× compression with minimal loss; LongLLMLingua reports up to +21.4% at 1/4 the tokens. Integrated into LangChain, LlamaIndex, Prompt Flow. MIT, 6.5k stars. **But the repo's newest news item is Dec 2024** — no new compressor in ~20 months.**
→ https://github.com/microsoft/LLMLingua · https://arxiv.org/abs/2310.06839
**Confidence: established as a claim; treat LLMLingua-2 (2024) as the terminal version.**

**Claim: LLMLingua-2 does not transfer to diffusion LLMs; high semantic preservation does not guarantee stable downstream behavior, and mathematical reasoning degrades substantially.** → https://arxiv.org/pdf/2605.17932 (2026-05-18). *emerging.*

## 10.3 The decisive 2026 evidence AGAINST compression

**Claim (the latency argument — best single answer to "is compression worth it in 2026"): across thousands of runs on 30,000 queries, multiple open-source LLMs, and three GPU classes, LLMLingua yields **at most ~18% end-to-end speedup, and only when prompt length, compression ratio, and hardware all align.** Outside that operating window, **compression overhead dominates and cancels the decoding gains entirely.** Output quality was statistically unchanged across summarization, code generation and QA. The authors released an open-source profiler predicting the latency break-even point per model/hardware pair. One genuine win: compression cut memory enough to migrate workloads from datacenter GPUs to commodity hardware with minimal latency penalty.**
→ https://arxiv.org/abs/2604.02985 (Kummer et al., **ECIR 2026 full paper**, 2026-04-03)
**Confidence: established** (large-scale, systems-level, peer-reviewed).

**Claim (the evaluation-validity argument): fixed compression can raise average accuracy while hiding reader upgrades and reversing model rankings. Across 20 readers × 10 domain-method configurations on 4 QA + 1 summarization benchmark: **compression benefit shrinks as the reader gets stronger (significant in 9/10 settings, p<0.05)**. Generic summarization **reversed 31% of pairwise model rankings on LongMemEval-S**. A HotpotQA compressor **obscured 80% of the gain from upgrading Qwen-7B → GPT-4.1-mini**. Mechanism: compression helps weak readers by removing noise they cannot filter, and hurts strong readers by removing detail they could have used. Toolkit `ragscale`, built on 177,000 compression transitions.**
→ https://arxiv.org/abs/2606.21807 (2026-06-20)
**Confidence: emerging but methodologically strong.** **This is precisely what "cheap long-context local models in 2026" changed: a compressor tuned when your reader was weak is now actively costing you accuracy AND masking the benefit of every model upgrade.**

**Claim (the structural-failure argument): hard compressors score units independently, so they split dependent evidence pairs — "referential dangling." At compression ratio 0.30, one system leaves the answer path incomplete in **34–54%** of bridge examples; across six hard compressors on HotpotQA, **dangling rates reach 60%**. Reinserting the missing supporting paragraphs recovers **+29–34 percentage points**; a trained restoration classifier recovers +4.7 points at the same ratio.**
→ https://arxiv.org/abs/2608.04569 (2026-08-05). *emerging.* **Multi-hop RAG is where hard compression breaks.**

**Claim (the baselines-were-wrong argument): in soft/embedding context compression, **mean pooling and a simple bidirectional compression-token variant outperform the widely used causal compression approaches.** Released BenchPress, a standardized suite spanning model scales, datasets and ratios from <1K to 8K tokens.**
→ https://arxiv.org/abs/2510.20797 (2025-10-23, **v2 2026-05-10**) · https://github.com/lil-lab/benchpress
**Confidence: emerging.** Much of the learned-soft-compression literature was beating weak baselines.

**Claim (decision-quality): LLM compression alters downstream *decisions* through decontextualization; fluent, factually plausible compressions still change decision-relevant judgments in financial analysis.** → https://arxiv.org/pdf/2606.29251 (2026-06-28). *emerging.* **Faithfulness-preserving ≠ decision-preserving.**

**Claim (agent control): under compression, degradation surfaces as **tool-execution failures, not lower token counts**. At 35% retained context: section-based 47.0% success, obligation-aware 39.0%, generic rewriting **19.9%**.** → https://arxiv.org/pdf/2608.01056 (2026-08-02). *emerging.*

## 10.4 The 2026 evidence FOR compression

- **CORE-RAG** — performance-driven compression trained with task success as feedback. At a **3% compression ratio it improves average Exact Match by +3.3 points over using full documents.** ICML 2026. → https://arxiv.org/abs/2508.19282 (v4 2026-05-28). *emerging.* Note: learned-for-the-task, not a drop-in.
- On repository-level code tasks at **4× compression**, continuous-latent-vector methods **surpass full context by +28.3% BLEU** — "compression filters noise rather than just truncating." → https://arxiv.org/pdf/2604.13725 (2026-04-15). *emerging.*
- **Telegraph English** — structured rewriting at ~**50% token reduction preserves 99.1% key-fact accuracy** with GPT-4.1, and **beats LLMLingua-2 by up to 11pp on fine-detail tasks in smaller models.** → https://arxiv.org/pdf/2605.04426 (2026-05-06). *emerging.*
- **Evolved linguistic rules** match advanced prompt-compression strategies **with no LM forward pass at deployment** — zero compression latency, removing the overhead that kills LLMLingua economics. → https://arxiv.org/pdf/2607.25335 (2026-07-29). *emerging.*
- **Tool-schema compression**, not document compression, is where agentic RAG wins: **44–50% schema token savings and +20.5pp average exact-match recovery** in constrained contexts. → https://arxiv.org/abs/2605.26165 (2026-05). *emerging, under-appreciated — in an agentic loop, tool schemas often outweigh retrieved text.*
- **RECOMP** remains the canonical baseline Provence/DSLR compare against; **not independently re-verified this session** — treat as superseded in the pruning-reranker line.

## 10.5 Verdict

- **Prune, don't compress.** Sentence-level extractive pruning folded into the reranker (**Provence / OpenProvence / XProvence**) is the only variant where the latency argument is not fatal.
- **Do not ship a token-level LLM compressor without profiling.** ECIR 2026 measured ≤18% best case, net-negative outside a narrow window; upstream quiet since Dec 2024.
- **Do not use a fixed compression ratio.** It hides reader upgrades and reverses rankings.
- **Never hard-compress multi-hop contexts** without a dangling-restoration step.
- **Re-run your compression ablation every time you upgrade the local reader.**

---

# 11. MEMORY

## 11.1 The benchmark situation is bad, and now independently documented

**Claim: LOCOMO — the field's standard benchmark — is broadly discredited. Objections: conversations average only 16k–26k tokens (inside modern context windows, so they do not stress memory at all); a plain full-context baseline beats the memory systems in Mem0's own reported results; plus data defects (missing ground-truth answers, multimodal errors, incorrect speaker attribution, ambiguous questions).**
→ https://blog.getzep.com/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/ (2025-05-06)
**Confidence: contested (vendor-on-vendor), but the token-count and full-context-baseline objections are structural and verifiable.**

**Claim: a plain filesystem agent with no memory tool at all scored 74.0% on LoCoMo with GPT-4o-mini, beating Mem0's reported 68.5% best variant.**
→ https://www.letta.com/blog/benchmarking-ai-agent-memory (2025-08-12)
**Confidence: contested (vendor), but it is the cleanest single refutation: the null baseline wins.**

**Claim (the first independent controlled ablation, and the most important memory finding of the year): MemDelta ran a controlled protocol against RAG and full-context baselines on LongMemEval-S (500 questions, 50+ sessions):**
- **Mem0 beats MiniLM-RAG by +11pp but LOSES to cloud-RAG by 1.2pp.**
- On 2 of 6 question types, **Mem0 matches cloud RAG (72.7% vs 73.9%, p = 1.0) at 50× the cost.**
- **Agent self-memory underperforms basic retrieval: 42% vs 47%.**
- **Swapping only the embedding model changes accuracy by +6.2pp at n=500 (p=0.004)** — enough to reverse which system "wins."
- Reader model flips rankings: **Gemini gains +14pp from full context while Sonnet gains +31pp from RAG.**
→ https://arxiv.org/abs/2606.29914 (2026-06-29). *emerging.*

**Claim: metric choice alone swings LOCOMO — a controlled ablation found a 27.5-point discrepancy between strict token-F1 and LLM-as-judge on the same outputs.** → https://arxiv.org/abs/2606.22030 (v2 2026-06-20). *emerging.* **Any LOCOMO number quoted without naming the scorer is uninterpretable.**

**Claim: LOCOMO is saturated. In Jun–Aug 2026 alone, 25+ distinct memory systems report LOCOMO results, several claiming 93%+ (Maximem Synap 93.2%, MemStack 93.59%, ABot-AgentOS 87.5%).** → arXiv API sweep, fetched 2026-08-08. *established as an observation.*

**2026 successor benchmarks:**

| Benchmark | What it adds | Source | Date |
|---|---|---|---|
| **MemoryArena** | Interdependent multi-session *agentic* tasks. Finds agents **near-saturated on LoCoMo perform poorly** here | https://arxiv.org/abs/2602.16313 | 2026-02-18 |
| **Letta Context-Bench** | Filesystem + Skills suites, live leaderboard. Top: GPT-5.2-Codex-xhigh 93% filesystem; Claude Sonnet 4.5 72% skill-use | https://leaderboard.letta.com/ | refreshed 2026-03-13 |
| **Supersede** | Isolates **fact-supersession** failure (stale memory not overwritten) + RL environment | https://arxiv.org/abs/2606.27472 | 2026-06-25 |
| LoCoMo-Contam | Memory poisoning variant | https://arxiv.org/abs/2607.22962 | 2026-07-25 |
| LoCoMo Temporal Plus | Conflict-heavy "ghost memory" variant | https://arxiv.org/abs/2607.01935 | 2026-07-02 |

**MemoryArena and Context-Bench are the two worth actually running.**

**LongMemEval** (https://arxiv.org/abs/2410.10813, 2024-10-14): 500 questions over scalable chat histories; **commercial assistants and long-context LLMs show a ~30% accuracy drop** across sustained interactions. *established.*

## 11.2 The open implementations

**Mem0** — https://github.com/mem0ai/mem0 · https://arxiv.org/abs/2504.19413
62.8k stars, **Apache 2.0**, Python + TS, three deployment modes (library / self-hosted Docker / cloud), provider-pluggable. Paper claims (2025-04-28): **+26% relative LLM-as-judge over OpenAI memory, 91% lower p95 latency, >90% token cost savings**. Repo claims (Apr 2026): 92.5 LoCoMo, 94.4 LongMemEval. **established as claims; contested as results** — see MemDelta/Letta/Zep above.

**Zep / Graphiti** — https://arxiv.org/abs/2501.13956 · https://github.com/getzep/graphiti
Paper (2025-01-20): DMR 94.8% vs MemGPT 93.4%; **LongMemEval up to +18.5% accuracy and 90% latency reduction**. *established as a claim; vendor-authored.* Graphiti: **29.7k stars, Apache 2.0**, temporal/bitemporal knowledge graph with full provenance. Backends: Neo4j 5.26+, **FalkorDB 1.1.2+ including an embedded "Lite" variant**, Amazon Neptune + OpenSearch, Kuzu 0.11.2 (**deprecated — upstream unmaintained**). **Runs fully local**: FalkorDB Lite (embedded, Python 3.12+) + Ollama / vLLM / llama.cpp / LM Studio via OpenAI-compatible endpoints. **This is the strongest fully-local option.**

**Letta / MemGPT lineage** — https://github.com/letta-ai/letta
24.2k stars, Apache 2.0. **Architecture shift in the last 12 months:** the `letta` repo is now the **legacy V1 API server**; active development moved to `letta-ai/letta-code`, self-hosting via the App Server, with the SDK reorganized around **skills and subagents**. Letta's position: memory tooling matters less than agentic context management — hence Context-Bench rather than LoCoMo. *established / emerging.*

**LangMem** — https://github.com/langchain-ai/langmem. 1.6k stars, **MIT**. Hot-path memory tools + background consolidation; usable with any store plus native LangGraph integration. *established*, but note the thin ecosystem (1.6k vs Mem0's 62.8k).

**GPTCache** — https://github.com/zilliztech/GPTCache. 8.1k stars, **but the latest release is v0.1.44 (2024-08-01)**, and maintainers state they "no longer add support for new API or models."
**Confidence: established — GPTCache is dormant. Do not adopt it for a 2026 build.**

## 11.3 Semantic caching — 2026 research

| Finding | Number | Source | Date |
|---|---|---|---|
| Semantic caches are **attackable** — embedding-similarity collision hijacks cached answers | **86% hijacking hit rate** | https://arxiv.org/abs/2601.23088 | 2026-01-30 |
| Retrieval-quality metrics mislead for caches (calibration mismatch); proposes P-CHR AUC | — | https://arxiv.org/abs/2606.19719 | 2026-06-19 |
| Freshness-aware caching for open-web RAG | **97% search-API savings at 0.1% stale-error rate** | https://arxiv.org/abs/2607.04281 | 2026-07-05 |
| Temporal semantic cache | **30.6× speedup on hits**, fails on time/parameter-dependent outputs | https://arxiv.org/abs/2605.20630 | 2026-05-20 |
| Production NL-to-code multi-agent cache | **67% hit rate, 40–60% token reduction** | https://arxiv.org/abs/2601.11687 | 2026-01-16 |
| LRU/LFU **provably underperform** on semantic workloads; SOLAR | **5–75% relative improvement** | https://arxiv.org/abs/2607.00394 | 2026-07-01 |
| Optimal offline semantic-cache policy is **NP-hard** | — | https://arxiv.org/abs/2603.03301 | 2026-02-07 |

*all emerging.* **Consistent theme: hit rate is the wrong headline metric; calibration, freshness, and collision-resistance decide whether a semantic cache is safe.**

## 11.4 What to ship

- **Graphiti + FalkorDB Lite + Ollama** if you need temporal/entity memory with provenance, Apache 2.0, and no network egress. The only mature genuinely-embeddable option.
- **Plain RAG over a session transcript store as the baseline you must beat.** Cloud-RAG beat Mem0 on LongMemEval-S; a filesystem agent beat Mem0 on LoCoMo. **Build the null baseline first.**
- **Mem0** for the biggest ecosystem, if you accept self-reported benchmarks. Apache 2.0, self-hostable.
- **LangMem** only if already on LangGraph.
- **Skip GPTCache.** Build a semantic cache on your existing vector store and instrument calibration + staleness, not hit rate.
- **Fix your embedding model before comparing memory systems** — a bare embedding swap moves accuracy 6.2pp, larger than most claimed memory-system deltas.

---

# 12. EVALUATION

## 12.1 RAGAS in 2026 — maintained, and sharply criticized

**Claim: RAGAS is actively developed — 15.2k stars, 1,147 commits, currently v0.4.x. v0.4.0 was a substantial breaking migration to a collections-based metrics system with `instructor.from_provider` universal provider support; v0.4.3 added a DSPy MIPROv2 prompt optimizer. Still ships test-data generation and Aspect Critique / `DiscreteMetric`.**
→ https://github.com/explodinggradients/ragas (fetched Aug 2026). *established that it is maintained; **release-date precision is low** — GitHub shows month/day only, most consistent with Dec 2025 / Jan 2026.*

**Claim (the sharpest criticism of the year): under corpus evolution, the best RAGAS metric reaches only **F1 0.570** against ground truth, versus 0.927–1.000 for metamorphic testing — i.e. RAGAS fails to detect faults introduced when the knowledge base changes.**
→ https://arxiv.org/abs/2607.26843 (2026-07-29). *emerging.*

Supporting criticism: "RAGAs-based faithfulness shows limited reliability" on long structured academic documents (https://arxiv.org/abs/2607.01852, 2026-07-02); an alternative comparative scorer (DICE) reports 85.7% agreement with human experts, "substantially outperforming RAGAS" (https://arxiv.org/abs/2512.22629, 2025-12-27, single-paper self-favorable). A direct applied comparison of Ragas / DeepEval / RAGChecker / Opik against human annotators exists (https://arxiv.org/abs/2607.07302, 2026-07-08) but the correlation numbers are not in the abstract.

**Verdict: still recommended, but only as a *relative regression signal* inside a fixed corpus and fixed judge. Do not treat RAGAS faithfulness as an absolute score, and do not rely on it to catch faults when the corpus changes.**

## 12.2 Open judge models — nothing new shipped in 12 months

| Model | Size | License | Measured | Date |
|---|---|---|---|---|
| **Atla Selene 1 Mini** | 8B (Llama-3.1-8B) | **Apache 2.0** | #1 8B generative model on RewardBench; beats GPT-4o on RewardBench, EvalBiasBench, AutoJ; 11 benchmarks / 3 task types; 128K ctx | 2025-01-27 |
| Atla Selene 1 | 70B | — | larger sibling | 2025-01 |
| **Patronus Lynx** | 8B / 70B | **CC-BY-NC-4.0 — non-commercial** | HaluBench **8B 82.9%, 70B 87.4%** vs GPT-4-Turbo 85.0%; GGUF quants for llama.cpp/Ollama/LM Studio | 2024-07 |
| GLIDER (Patronus) | 3B | open | **91.3% human agreement** on arbitrary criteria, 685 domains | 2024-12-18 |
| Prometheus lineage | — | open | Prometheus-Vision: highest human correlation among open VLM evaluators | 2024-01 |
| Judge's Verdict (meta) | — | — | Benchmarked **43 open models; 27 achieve "Tier 1"** human-like judgment (Cohen's κ) | 2025-10-10 |
| Counsel (meta-eval) | — | — | Strongest **open-weight** judge reaches **~88% location agreement** with humans | 2026-06-19 |

Sources: https://huggingface.co/AtlaAI/Selene-1-Mini-Llama-3.1-8B · https://arxiv.org/abs/2501.17195 · https://huggingface.co/PatronusAI/Llama-3-Patronus-Lynx-8B-Instruct · https://arxiv.org/abs/2407.08488 · https://arxiv.org/abs/2412.14140 · https://arxiv.org/abs/2510.09738 · https://arxiv.org/abs/2606.21627

**Flags:** **Lynx is CC-BY-NC — do not ship it commercially.** **Selene 1 Mini (Apache 2.0) is the correct default open judge for a commercial self-hosted stack.** **Prometheus, JudgeLM, and Flow Judge are all 2024-vintage with no 2026 successor found** (⚠️ Flow Judge and JudgeLM status unverified). The 2026 direction is not "a bigger judge" but **judge alignment tooling** (§12.4).

⚠️ LLM-as-judge has a **backdoor/poisoning attack surface** in RAG evaluation → https://arxiv.org/abs/2503.00596 (BadJudge, 2025-03-01). *emerging.*

## 12.3 Frameworks for a small local team

| Tool | License | Scale | State (Aug 2026) |
|---|---|---|---|
| **promptfoo** | MIT | 24.1k ★ | **Evals run 100% locally — prompts never leave the machine.** Ollama + all major providers, CI/CD + PR review, red-teaming. **Acquired by OpenAI, announced 2026-03-09; OSS continues under Ian Webster & Michael D'Angelo.** Node ≥22.22 |
| **Inspect AI** | MIT | 2.5k ★ | UK AI Security Institute. 7,013 commits, **200+ prebuilt evals**, model-graded scoring, tool use, multi-turn. Most auditable; smallest RAG-specific surface |
| **DeepEval** | — | — | v4.x (v4.1.5, Jul 29). **2026 direction is agentic**: task-completion, tool-correctness, `AgentLoopDetectionMetric`, coding-agent eval harness, TUI trace inspection, Arena-GEval |
| **TruLens** | MIT (Snowflake) | 3.5k ★ | **v2.12.0, 2026-08-06** (2.11.0 Aug 4, 2.10.0 Jul 28, 2.9.0 Jul 23 — rapid cadence). OTel-native. **2026 additions are all judge-quality tooling**: `AlignmentReport`, `CrossModelAlignment`, `Jury`, `CriteriaABTest`, `ScoreDistributionAnalyzer`, `GoldenSetGenerator`, `FewShotOptimizer`, plus `citation_accuracy` / `citation_attribution` |
| **Arize Phoenix** | **Elastic License 2.0 — NOT OSI-permissive** | 11k+ ★ | OTel tracing, `arize-phoenix-evals` with **published benchmarks of the evaluators themselves**, datasets/experiments, prompt playground. Self-hostable Docker/K8s/Helm |
| **Braintrust autoevals** | MIT | 994 ★ | Focused scorer library: context precision/relevancy/recall/entity-recall, faithfulness, answer relevancy/similarity/correctness |
| **RAGAS** | — | 15.2k ★ | v0.4.x; see §12.1 |

**Recommendation for a small self-hosted team: promptfoo for local CI gating (genuinely offline, MIT, largest community); RAGAS or Braintrust autoevals for the RAG-specific scorer set; TruLens if you need to prove your judge agrees with your humans; Inspect AI if you need auditability. ⚠️ Phoenix's ELv2 license is a real constraint — check it before embedding in a product.**

## 12.4 The 2026 meta-shift: judge alignment over judge choice

**Claim: the measurable 2026 change is that frameworks stopped shipping new metrics and started shipping **instrumentation to validate the judge** — judge↔human agreement reports, cross-model judge comparison, judge ensembles/juries, golden-set curation from production traces, few-shot judge optimization, and score-distribution calibration diagnostics — all landed in TruLens 2.9–2.12 between 2026-07-23 and 2026-08-06.**
→ https://github.com/truera/trulens/releases · https://pypi.org/project/trulens/
**Confidence: emerging, but the clearest signal of where practice moved** — and consistent with the §9 finding that factuality metrics disagree with each other and misestimate system-level performance.

## 12.5 Retrieval metrics practice — unchanged

**nDCG@10** as the headline (BEIR/BRIGHT/MTEB convention), **recall@k** for the retriever stage, **MRR@10** for MS MARCO-style single-answer settings. Provence, as a representative example, reports R@5 on NQ/HotpotQA, MRR@10 on MS MARCO, nDCG@10 on TREC DL'19, and mean nDCG@10 across 13 BEIR datasets.
→ https://arxiv.org/abs/2501.16214. *established.*

**For a RAG pipeline specifically: recall@k at the retriever and nDCG@10 after reranking are the two that matter, because generation quality is bounded by whether the evidence is in the window at all — reinforced by the finding that retrieval is the main driver of attribution quality in both citation paradigms** (https://arxiv.org/abs/2509.21557). *established.*

## 12.6 Leaderboards, Aug 2026

| Benchmark | Status | Source |
|---|---|---|
| **BEIR** | Still the zero-shot retrieval standard (13–18 datasets, nDCG@10). *established* | — |
| **MTEB** | Still the umbrella; **overfitting concerns now official** | https://huggingface.co/blog/rteb |
| **RTEB** ⭐ NEW | Maintainers' answer to overfitting. **Hybrid open + private datasets**; the open↔private gap directly measures overfitting. 20 languages, enterprise domains. Beta 2025-10-01. *established* | https://huggingface.co/blog/rteb |
| **BRIGHT** | Reasoning-intensive retrieval, 1,385 queries. Current leaders: **Mira-Reasoning-Retrieval 66.9 (2026-04-22), INF-X-Retriever 63.4, RakanEmbed4B 52.4, NeMo Retriever Agentic Retrieval 50.9 (open-weight)**. Baselines: **BM25 14.5** (30.4 with GPT-4 reasoning + rerank), GritLM-7B 21.0, instructor-xl 18.9, text-embedding-3-large 17.9, bge-large-en-v1.5 13.7. Most top submissions insert an **LLM reasoning step before retrieval**. *established* | https://brightbenchmark.github.io/ |
| BRIGHT — ⚠️ caveat | Reproducibility audit found **undocumented implementation details (query-side BM25)**, corpus quality issues, and that **BM25Q gains are largely BRIGHT-specific rather than generalizable**. *contested* | https://arxiv.org/abs/2509.02558 (2025-09-02) |
| BRIGHT — cost caveat | Across 12 BRIGHT tasks, LLM-based retrievers show **weak confidence signals and inconsistent reasoning-augmentation benefit across model families**. *emerging* | https://arxiv.org/abs/2604.03676 (2026-04-04) |
| **BRIGHT-Pro** ⭐ NEW | Expert-annotated expansion with multi-aspect gold evidence, evaluating retrievers under **both static and agentic protocols**; plus RTriever-Synth corpus and RTriever-4B. Core claim: retrievers must provide **"complementary evidence across iterative search and synthesis"**, not topical similarity; **"aspect-aware and agentic evaluation expose behaviors hidden by standard metrics."** *emerging* | https://arxiv.org/abs/2605.04018 (2026-05-06) |
| **MM-BRIGHT** ⭐ NEW | First multimodal reasoning-intensive retrieval benchmark, 29 technical domains. *emerging* | https://arxiv.org/abs/2601.09562 (2026-01-14) |
| **HAKARI-Bench** ⭐ NEW | Nano-dataset reconstruction across **43 languages**, 5 retrieval families; **correlates highly with official MTEB and BEIR rankings at a fraction of the compute.** Genuinely useful for a small team. *emerging* | https://arxiv.org/abs/2606.22778 (2026-06-22) |
| **RAGTruth** | Still the span/response hallucination standard; best open results now **~81.8 example-F1** (Qwen3.5-2B, Jul 2026). *established* | https://arxiv.org/abs/2607.00895 |
| **LLM-AggreFact** | Still the grounded-factuality standard. **Leaderboard appears frozen** — newest datable entry Aug 2025; 2026 models (ThinknCheck 78.1) not listed. *contested as a live signal* | https://llm-aggrefact.github.io/ |
| **TRIVIA+** ⭐ NEW | RAG hallucination-detection with **the longest contexts in the literature** + four label sets simulating realistic label noise. ACL 2026. *emerging* | https://arxiv.org/abs/2605.11330 |
| **BenchPress** ⭐ NEW | Standardized context-compression eval suite. *emerging* | https://arxiv.org/abs/2510.20797 |
| **ragscale** ⭐ NEW | Compression-audit toolkit, 177k transitions; audits a compression paper with 3 readers in ~1 day. *emerging* | https://arxiv.org/abs/2606.21807 |
| **RAISE** ⭐ NEW | RAG hyperparameter/architecture-search benchmark; finding: **optimization is highly task-dependent — methods strong on one dataset don't generalize.** *emerging* | https://arxiv.org/abs/2605.30029 |
| **RAGRouter-Bench** ⭐ NEW | First RAG-strategy routing benchmark (see §7) | https://arxiv.org/abs/2602.00296 |
| **BrowseComp-Plus** ⭐ | Retriever-disentangled agentic search eval, ACL 2026 (see §8) | https://aclanthology.org/2026.acl-long.1023/ |
| **OmniDocBench v1.6/1.7** | Document parsing (see §1) | https://github.com/opendatalab/OmniDocBench |

## 12.7 Synthetic eval-set generation

RAGAS ships test-data generation; TruLens added `GoldenSetGenerator` to curate eval datasets from production records (v2.9.0). A multi-agent framework generates diverse + privacy-masked synthetic RAG eval sets (https://arxiv.org/abs/2508.18929, 2025-08-26) but **does NOT validate against human-labeled evaluation sets.**

**⚠️ Gap flag: no 2026 primary source measures how well synthetic RAG eval sets agree with human-labeled ones. Treat synthetic eval sets as regression tripwires, not as ground truth for absolute quality claims.** Cross-reference §6.1: synthetic evals overstated the need for query augmentation by **60+ percentage points** in the one production system that measured it.

---

# BIGGEST SHIFTS IN THE LAST 12 MONTHS

**One paragraph:** The dominant story of Aug 2025 → Aug 2026 is that **the field turned self-critical and the small models won**. In parsing, sub-1B specialist VLMs (PaddleOCR-VL-1.6 at 96.34, GLM-OCR at 95.22, both 0.9B) decisively beat every frontier API on OmniDocBench and effectively saturated it, while the Western olmOCR lineage shipped no new model for ten months and Docling repositioned itself from parser to orchestration layer that swaps VLM backends. In embeddings, **Microsoft open-sourced Harrier under MIT and took the multilingual MTEB v2 crown at 74.3**, its 0.6B variant (69.0) now beating Qwen3-Embedding-0.6B by 4.7 points — while MTEB's own maintainers launched **RTEB with private held-back test sets** because they concluded public leaderboards are overfit; the practical selection criterion shifted from score to **license**, since the two best sub-1B models (jina-v5, KaLM) are non-commercial. In verification, **a 4-bit 1B reasoning verifier (ThinknCheck, Apr 2026) beat the 7B 2024 SOTA on LLM-AggreFact**, making per-answer grounding checks cheap enough to always run — even as span detectors were shown to **collapse from 0.69 to 0.17 span-F1 when the context is code or tool output** rather than prose. Compression took three independent hits in one year — an ECIR 2026 systems study measuring **≤18% best-case speedup and net-negative outside a narrow window**, a demonstration that **fixed compression reverses 31% of model rankings and hides 80% of a reader upgrade**, and the discovery that hard compressors leave multi-hop answer paths incomplete in **34–60% of bridge cases** — leaving reranker-integrated sentence pruning (Provence / the MIT OpenProvence reimplementation) as the only variant whose economics survive. Memory's standard benchmark **LOCOMO went from canonical to discredited and saturated**, with the first independent controlled ablation (MemDelta, Jun 2026) finding Mem0 *loses* to plain cloud-RAG on LongMemEval-S and that **swapping only the embedding model moves accuracy more (+6.2pp) than most claimed memory-system gains**. On query planning, **HyDE was demoted from default to conditional** — measurably −7.3% Recall@5 on numeric corpora, and needed for only 27.8% of real production queries versus the >90% synthetic evals implied — with the consensus shifting to *escalate, don't pre-decide*: cheapest-first post-retrieval cascades rather than pre-retrieval classifiers, since **four ML approaches to pre-retrieval routing all failed**. Routing followed the same arc: **fixed hybrid BM25+dense with RRF beat rule-based retriever routing by +1.8 EM** on a local 7B, and where routing does pay (pipeline depth), a **TF-IDF + SVM classifier beat sentence embeddings** at 0.928 macro-F1. Agentic loops consolidated and productized — **open-weight 4B deep-research agents now match Claude-4.5-Sonnet on GAIA-Text (71.3 vs 71.2)**, RL training cost collapsed ~13×, and Chroma shipped **Context-1, a 20B Apache-2.0 self-editing retrieval subagent** hitting 0.87–0.96 on BrowseComp-Plus at 10× frontier inference speed — but the year also produced the **first serious negative results on subagent fan-out** (41.8% of deep-agentic failures occur silently at the planner→subagent hand-off; plain semantic search beat deep agentic search 65.2% vs 46.2% at half the cost on repo-level code QA) and confirmed that **search volume correlates weakly with answer quality**, pushing termination criteria toward cumulative evidence sufficiency. Finally, evaluation tooling stopped shipping new metrics and started shipping **judge-validation instrumentation** (TruLens 2.9–2.12: alignment reports, juries, cross-model comparison, golden-set generation), RAGAS took its sharpest criticism yet (**F1 0.570 at detecting corpus-evolution faults**), promptfoo was acquired by OpenAI, and **no new open judge model shipped at all** — leaving Selene 1 Mini (Jan 2025, Apache 2.0) still the best permissive default. **The through-line for a self-hosted stack: the single highest-ROI investment is still the retriever and the reranker — swapping BM25 for Qwen3-Embedding-8B moved the same agent +14.2 points on BrowseComp-Plus, and a cross-encoder is worth +17.2 pp MRR@3 — while nearly every clever technique layered on top (HyDE, multi-query, compression, memory systems, subagent fan-out) was measured this year and found to be conditional at best, and a silent regression at worst.**

---

## Source-quality caveats

**Peer-reviewed anchors:** BrowseComp-Plus (ACL 2026 Main), Query Decomposition Exploration-Exploitation (EACL 2026), LLMLingua latency profiling (ECIR 2026), XProvence (ECIR 2026), Provence (ICLR 2025), MMTEB (ICLR 2025), TRIVIA+ (ACL 2026), OmniDocBench (CVPR 2025), MiniCheck (EMNLP 2024), HHEM (EMNLP 2025 Industry), PEA-CAE (IEEE SEGE 2026).

**Unrefereed preprints carrying disproportionate weight — treat as emerging:** the local-7B ablation (2606.21553, single author), RAGRouter-Bench baselines (2604.03455, in-distribution only), front-door SLM routing (2604.02367, neff=60), Coverage Illusion (2605.27220, single production system), MemDelta (2606.29914), ThinknCheck (2604.01652), plus most 2026 RL-search-agent papers.

**Vendor self-reports (direction credible, magnitude favorable by construction):** all embedding/reranker model-card benchmarks, Chroma Context-1, Mem0, Zep, Letta, Vectara, Anthropic's multi-agent 90.2% figure, Jina's binary-quantization claim.

**Flagged unreliable — do not cite:** https://arxiv.org/abs/2607.27562 (HLE 87.3% / BrowseComp-ZH 85.3%) — implausible and uncorroborated.

**Withdrawn:** https://arxiv.org/abs/2601.04618 (REPAIR) — withdrawn 2026-04-14 for missing institutional permissions.

**Known gaps:** the MTEB/MMTEB and BrowseComp-Plus live leaderboards render client-side and could not be fetched (numbers above come from model cards and papers); the unstructured competitive positioning is unsourced; Flow Judge and JudgeLM current status unverified; RECOMP not re-verified; Cleanlab's detector-comparison numeric table unextractable; no 2026 benchmark exists for a sub-1B *generative* retrieval router; and the session WebSearch quota (200) was exhausted early, so vendor engineering blogs are under-sampled relative to arXiv.
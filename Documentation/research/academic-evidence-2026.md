All four research streams are complete. Here is the deliverable.

---

# ACADEMIC STATE OF THE ART IN AGENTIC RETRIEVAL — 2025–2026

**Method / verification status.** ~200 web searches (session budget exhausted) plus ~180 direct fetches of arXiv abs/HTML pages, ACL Anthology, ICLR/OpenReview, the official BRIGHT leaderboard, the Vectara leaderboard repo, and the arXiv metadata API. Every arXiv ID below was either fetched directly or returned by arXiv's own API. Items I could not verify are flagged ⚠️ inline and listed again at the end. Citation counts come from the Semantic Scholar API (retrieved 8 Aug 2026); OpenAlex was tried first and badly undercounts preprints, so I discarded it.

---

## 1. AGENTIC / ITERATIVE RETRIEVAL TRAINING

### 1.1 The 2025 founding wave — what each actually claimed

| System | Paper | Date | Venue | Core claim |
|---|---|---|---|---|
| **DeepRetrieval** | Pengcheng Jiang, Jiacheng Lin, Lang Cao, Runchu Tian, SeongKu Kang, Zifeng Wang, Jimeng Sun, Jiawei Han. arXiv:2503.00223 | 28 Feb 2025 (v3 12 Apr) | arXiv only, cs.IR | GRPO-trained **one-shot query generation**, reward = retrieval metric, no supervised reference queries |
| **R1-Searcher** | Huatong Song, Jinhao Jiang, Yingqian Min, Jie Chen, Zhipeng Chen, Wayne Xin Zhao, Lei Fang, Ji-Rong Wen (RUC). arXiv:2503.05592 | 7 Mar 2025 | arXiv | **Two-stage outcome-based RL**, no process rewards, no distillation cold start |
| **Search-R1** | Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, Jiawei Han (UIUC + UMass + Google Cloud AI). arXiv:2503.09516, v5 5 Aug 2025 | 12 Mar 2025 | arXiv (**1,317 citations**, 5.3k GitHub stars) | Multi-turn interleaved search+reason, **retrieved-token masking**, outcome-only reward |
| **ReSearch** | Mingyang Chen et al. arXiv:2503.19470 | 25 Mar 2025 | arXiv | End-to-end RL, search treated as part of the reasoning chain, no supervised tool-use trajectories |
| **WebDancer** | Alibaba Tongyi Lab. arXiv (WebAgent family) | May 2025 | **NeurIPS 2025** | 4-stage: data construction → trajectory sampling → SFT → RL |
| **WebSailor** | Kuan Li, Zhongwang Zhang, Huifeng Yin, … Yong Jiang, Ming Yan, Pengjun Xie, Fei Huang, Jingren Zhou (Alibaba Tongyi). arXiv:2507.02592 | 3 Jul 2025 | arXiv | High-uncertainty synthetic tasks + RFT cold start + **DUPO** RL |
| **ZeroSearch** | Hao Sun, Zile Qiao, … Fei Huang, Jingren Zhou (Alibaba). arXiv:2505.04588, v3 19 May 2026 | 7 May 2025 | arXiv | Replace the live search engine with a **simulated LLM retriever** during RL |

**Search-R1's actual numbers (verified from v5 HTML, not the abstract).** Setup: 2018 Wikipedia dump, **E5 retriever, top-3 passages**, PPO (more stable) vs GRPO (faster convergence but reward collapse). Average EM across NQ, TriviaQA, PopQA, HotpotQA, 2Wiki, Musique, Bamboogle:

| Method | Qwen2.5-7B avg EM | Qwen2.5-3B avg EM |
|---|---|---|
| Direct | 0.181 | 0.134 |
| CoT | 0.106 | 0.015 |
| IRCoT | 0.239 | 0.181 |
| RAG | 0.304 | 0.270 |
| R1 (reason, no search) | 0.276 | 0.229 |
| **Search-R1-base** | **0.431** | **0.303** |

The widely-quoted "**41% improvement**" is over the RAG baseline. Per-dataset 7B: NQ 0.480, TriviaQA 0.638, PopQA 0.457, HotpotQA 0.433, 2Wiki 0.382, Musique **0.196**, Bamboogle 0.432.

⚠️ **Caveats that matter.** (a) The baseline is a naive top-3 single-shot RAG over a 2018 Wikipedia dump — a weak reference point by 2026 standards. (b) Musique at 0.196 shows the hard multi-hop case is barely moved. (c) The whole evaluation runs against a **fixed local E5 index**, so nothing is learned about live-web behavior. This last point is the wedge for §1.3.

**DeepRetrieval's numbers:** publication search recall **65.07%** vs 24.68% prior SOTA; trial search recall **63.18%** vs 32.11%; beats GPT-4o and Claude-3.5-Sonnet on 11 of 13 datasets with a **3B** model. ⚠️ These are literature-search domains (PubMed/ClinicalTrials.gov), not general QA; the "hacking real search engines" framing is doing a lot of work — the gains are largely query-formulation gains against a fixed API.

### 1.2 The deep-research agent line and where the open frontier sits in mid-2026

**Tongyi DeepResearch Technical Report** (56 authors, Alibaba Tongyi). arXiv:2510.24701, 28 Oct 2025, v3 18 May 2026. 30.5B total / **3.3B activated** MoE, 128K context, agentic mid-training + agentic post-training. Verified Table 1:

| Model | HLE | BrowseComp | BrowseComp-ZH | GAIA | xbench-DS | WebWalker | FRAMES |
|---|---|---|---|---|---|---|---|
| **Tongyi DeepResearch 30B-A3B** | **32.9** | 43.4 | 46.7 | **70.9** | **75.0** | **72.2** | **90.6** |
| OpenAI DeepResearch | 26.6 | **51.5** | 42.9 | 67.4 | — | — | — |
| OpenAI o3 (ReAct) | 24.9 | 49.7 | **58.1** | — | 67.0 | 71.7 | 84.0 |
| DeepSeek-V3.1 (ReAct) | 29.8 | 30.0 | 49.2 | 63.1 | 71.0 | 61.2 | 83.7 |
| Claude-4-Sonnet (ReAct) | 20.3 | 12.2 | 29.1 | 68.3 | 65.0 | 61.7 | 80.7 |
| GLM-4.5 (ReAct) | 21.2 | 26.4 | 37.5 | 66.0 | 70.0 | 65.6 | 78.9 |
| Kimi Researcher | 26.9 | — | — | — | 69.0 | — | 78.8 |

Note the split verdict: a trained 30B-A3B open model **beats OpenAI Deep Research on HLE (32.9 vs 26.6) and GAIA (70.9 vs 67.4) but loses on BrowseComp-EN (43.4 vs 51.5)**. Also note Claude-4-Sonnet's 12.2 on BrowseComp under a plain ReAct harness — prompted frontier models without a deep-research scaffold are not competitive on this task class.

**What superseded them in 2026:**

- **LiteResearcher** (Bince Qu, Wanli Li, Bo Pan, Jianyu Zhang, Zheng Liu, Pan Zhang, Wei Chen, Bo Zhang). arXiv:2604.17931, 20 Apr 2026, v5 26 Jul 2026. Builds a **"lite virtual world"** mirroring real search dynamics so RL doesn't depend on a live search API. LiteResearcher-**4B**: GAIA-Text 71.3, xbench-DS 78.0, FRAMES 83.1, WebWalker 72.7, Seal-0 41.8, HLE 22.0, BrowseComp 27.5, BrowseComp-ZH 32.5. **Parity with Claude-4.5-Sonnet on GAIA (71.3 vs 71.2) and beats Tongyi-30B on GAIA and xbench, at 4B.** Ablation (their Table 9): **SFT alone 55.58 GAIA → RL 71.3 (+15.7); xbench 64.25 → 78.0 (+13.8).** Stated limitations: 128K context exhausts on deep BrowseComp chains; gains depend on a 32M-page enriched corpus and generalization beyond it is unexplored.

- **OpenSeeker-v2** (Yuwen Du, Rui Ye, Shuo Tang, Keduan Huang, Xinyu Zhu, Yuzhu Cai, Siheng Chen). arXiv:2605.04036, 5 May 2026. **SFT only, no RL**, 10.6k trajectories: **BrowseComp 46.0, BrowseComp-ZH 58.1, HLE 34.6, xbench 78.0** at 30B/ReAct. Self-described as the first SOTA search agent at its scale from a purely academic team using only SFT.

- **WebExplorer** (Junteng Liu et al.). arXiv:2509.06501, 8 Sep 2025. **WebExplorer-8B**, 128K, up to 100 tool turns, averages **16 search turns after RL**; higher BrowseComp-en/zh than **WebSailor-72B** and best among ≤100B on WebWalkerQA and FRAMES.

**→ OPEN DEBATE #1 — is RL actually necessary?** LiteResearcher measures RL contributing **+15.7 GAIA points over its own SFT checkpoint**. OpenSeeker-v2, one month later, reaches **higher BrowseComp (46.0 vs 27.5) and HLE (34.6 vs 22.0) with pure SFT on 10.6k curated high-difficulty trajectories**. Different model scales (4B vs 30B) so it is not a clean head-to-head, but the field has no controlled experiment isolating RL from trajectory-data quality at fixed scale. This is unresolved and under-discussed.

### 1.3 The critical literature — this is the part that changes conclusions

**⭐ BrowseComp-Plus: A Fair and Disentangled Evaluation Benchmark for Deep Search Agents.** Zijian Chen, Xueguang Ma, Shengyao Zhuang, Ping Nie, Kai Zou, Andrew Liu, Joshua Green, Kshama Patel, Ruoxi Meng, Mingyi Su, Sahel Sharifymoghaddam, Yanxi Li, Haoran Hong, Xinyu Shi, Xuye Liu, Nandan Thakur, Crystina Zhang, Luyu Gao, Wenhu Chen, **Jimmy Lin** (Waterloo + CSIRO + CMU + Queensland). arXiv:2508.06600, 8 Aug 2025. **ACL 2026 Main** (aclanthology.org/2026.acl-long.1023). 155 citations.

Fixed curated corpus, human-verified supporting documents, mined hard negatives — so retriever and agent can be varied independently. Verified Table 1 accuracy:

| Agent | + BM25 | + Qwen3-Embed-8B |
|---|---|---|
| **GPT-5** | 55.90% | **70.12%** |
| o3 | 49.28% | 63.49% |
| gpt-oss-120B-high | 28.67% | 42.89% |
| Gemini 2.5 Pro | 19.04% | 28.67% |
| Claude Opus 4 | 15.54% | 36.14% |
| Claude Sonnet 4 | 14.34% | 36.75% |
| gpt-4.1 | 14.58% | 35.42% |
| **Search-R1-32B** | **3.86%** | **10.36%** |
| Qwen3-32B | 3.49% | 10.36% |

Four findings, each load-bearing:

1. **Retriever quality outweighs agent choice.** Swapping BM25 → Qwen3-Embedding-8B buys GPT-5 **+14.2 points** and roughly doubles weaker agents. Better retrievers also **reduce** search-call count.
2. **RL-trained open agents do not transfer.** Search-R1-32B scores **exactly what its untrained base Qwen3-32B scores (10.36% with the good retriever)**. The RL training bought nothing outside its training distribution. This is the single most damaging result for the Search-R1 lineage's generalization claims.
3. **The bottleneck is interleaved tool-use reasoning, not knowledge.** Authors: open models "do not substantially lag behind proprietary models in their ability to answer questions when provided with sufficient evidence." Proprietary models average **20+ search calls/query**; open models fewer than 2 despite explicit tool prompting.
4. **Reasoning-specialized retrievers underperform scaled general ones inside agentic loops.** Qwen3-Embedding-8B: **14.5% Recall@5, 20.3 nDCG@10**; ReasonIR-8B: **12.2% / 16.8**. Note the absolute ceiling — the *best* retriever gets 20.3 nDCG@10. Enormous headroom.

**⭐ Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?** Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Shiji Song, Gao Huang (Tsinghua). arXiv:2504.13837, 18 Apr 2025, v5 24 Nov 2025. **NeurIPS 2025 Oral**; ICML 2025 AI4MATH workshop best paper. RLVR-trained models beat base models at pass@1 but **base models overtake at large k** — the reasoning was already in the base model; RLVR narrows the sampling distribution. Six RLVR algorithms are "far from optimal in leveraging the potential of the base model." Not search-specific, but it is the theoretical frame for interpreting every "+41% over RAG" claim in this section.

**⭐ Demystifying deep search / WebDetective.** Maojia Song, Renhang Liu, Xinyu Wang, Yong Jiang, Pengjun Xie, Fei Huang, Jingren Zhou, Dorien Herremans, Soujanya Poria. arXiv:2510.05137, 1 Oct 2025 (v3 10 Dec 2025). Two indictments of current practice: **most benchmarks leak the reasoning path in the question text**, and single-pass-rate scoring collapses distinct failure modes. Hint-free multi-hop questions in a traceable Wikipedia sandbox, **25 SOTA models**: systematic failure at knowledge utilization *despite sufficient evidence*, and near-absent appropriate refusal when evidence is missing. "Today's systems excel at executing given reasoning paths but fail when required to discover them." Their EvidenceLoop workflow (verification loops + systematic evidence tracking) is the proposed fix.

**⭐ How to Train Your Deep Research Agent? Prompt, Reward, and Policy Optimization in Search-R1.** Yinuo Xu, Shuo Lu, Jianjie Cheng, Meng Wang, Qianlong Xie, Xingxing Wang, Ran He, Jian Liang. arXiv:2602.19526, 23 Feb 2026. Decoupled ablation of the three axes. Findings: the **"Fast Thinking" prompt template is more stable and better-performing than the Slow Thinking template used in prior work**; F1 rewards cause training collapse via answer avoidance (fixed by action-level penalties); **REINFORCE beats PPO with fewer search actions, and GRPO is the least stable**. Search-R1++ moves Qwen2.5-7B 0.403 → 0.442 and 3B 0.289 → 0.331. Read this alongside Search-R1's own PPO-over-GRPO finding — the field's optimizer consensus is unsettled.

### 1.4 The 2026 research frontier has moved to efficiency and context, not capability

Outcome-only RL **provably induces over-search**: the ratio of no-search trajectories drops toward zero while redundant-search ratio rises. Four verified responses:

- **HiPRAG** (Peilin Wu, Mian Zhang, Kun Wan, Wentian Zhao, Kaiyu He, Xinya Du, Zhiyu Chen). arXiv:2510.07794, v2 11 Apr 2026. **Accepted ICLR 2026.** Hierarchical process rewards over decomposed reasoning steps: average accuracy **65.4% (3B) / 67.2% (7B)** across 7 QA benchmarks with **over-search rate driven to 2.3%**.
- **SAAS** (Yunbo Tang, Chengyi Yang, Shiyu Liu, Zhishang Xiang, Zerui Chen, Qinggang Zhang, Jinsong Su). arXiv:2605.29796, 28 May 2026. Contrasts search-disabled vs search-enabled rollouts to model the knowledge boundary; boundary-aware trajectory penalties; stage-wise curriculum to avoid reward hacking. ⚠️ Abstract carries no numbers.
- **AutoSearch** (Jingbo Sun et al., incl. Dongbin Zhao). arXiv:2604.17337, 19 Apr 2026. Self-generated intermediate answers identify a **minimal sufficient search depth**; reward attainment, penalize over-search. ⚠️ No numbers in abstract.
- **FoldAct** (Jiaqi Shao, Yufeng Miao, Wei Zhang, Bing Luo). arXiv:2512.22733, 28 Dec 2025. Context folding for long-horizon RL: identifies gradient dilution on summary tokens, self-conditioning instability, and per-turn context recomputation. **5.19× training speedup.**
- **Erase to Improve (ERL)** (Ziliang Wang et al.). arXiv:2510.00861, v2 20 Apr 2026. Identify → erase → regenerate faulty reasoning steps. **3B: +8.48% EM / +11.56% F1; 7B: +5.38% EM / +7.22% F1** on HotpotQA/MuSiQue/2Wiki/Bamboogle.
- **Agentic-R** (Wenhan Liu, Xinyu Ma, Yutao Zhu, Yuchen Li, Daiting Shi, Dawei Yin, Zhicheng Dou). arXiv:2601.11888, 17 Jan 2026. **Bidirectional iterative co-optimization of the search agent and the retriever**, with passage utility measured by both local relevance and global answer correctness. Directionally the most interesting 2026 idea — it treats the BrowseComp-Plus finding (retriever dominates) as a training target. ⚠️ No numbers in abstract.

**Surveys.** "A Comprehensive Survey on Reinforcement Learning-based Agentic Search" — Minhua Lin, Zongyu Wu, Zhichao Xu, Hui Liu, Xianfeng Tang, Qi He, Charu Aggarwal, Hui Liu, Xiang Zhang, Suhang Wang (Penn State + Amazon + IBM). arXiv:2510.16724, 19 Oct 2025, 38pp. Also "The Landscape of Agentic Reinforcement Learning for LLMs" (arXiv:2509.02547) and "RL Foundations for Deep Research Systems: A Survey" (arXiv:2509.06733).

**Honest bottom line for §1.** RL-trained search behavior demonstrably beats prompted loops **within a fixed base model on the training distribution** (Search-R1 0.431 vs 0.304 RAG; LiteResearcher +15.7 GAIA over its own SFT). It demonstrably **does not transfer** to a new corpus/retriever (Search-R1-32B = its base model on BrowseComp-Plus). The 2026 SOTA on hard deep-research benchmarks belongs to large-scale trajectory curation (SFT or RL) plus a strong retriever — and **retriever choice moves accuracy more than agent choice does**.

---

## 2. REASONING-AWARE RETRIEVAL

### 2.1 BRIGHT and what happened to it

**BRIGHT: A Realistic and Challenging Benchmark for Reasoning-Intensive Retrieval.** Princeton et al. arXiv:2407.12883, 16 Jul 2024. **ICLR 2025** (OpenReview ykuc5q381b). 177 citations. 1,384–1,398 real-world queries, 12 domains (economics, psychology, math, coding, robotics, StackExchange splits, theorem retrieval), corpora 7.9K–414K docs.

At release: **max nDCG@10 = 24.3**. SFR-Embedding-Mistral, then MTEB #1 at 59.0, scored **18.3**. LLM reasoning-step query augmentation helped but stayed under 30. Authors report BRIGHT is robust to data leakage — fine-tuning on the retrieval documents barely moves scores.

**The official leaderboard as of 8 Aug 2026** (brightbenchmark.github.io), short-document track, avg nDCG@10 over 12 datasets:

| Rank | System | Score | Date | Reranker |
|---|---|---|---|---|
| 1 | Mira-Reasoning-Retrieval (Forward AI Labs) | **66.9** | 22 Apr 2026 | Yes |
| 2 | INF-X-Retriever | 63.4 | 20 Dec 2025 | Yes |
| 3 | RakanEmbed4B | 52.4 | 20 Mar 2026 | Yes |
| 4 | NeMo Retriever Agentic Retrieval (NVIDIA) | 50.9 | 13 Mar 2026 | Yes |
| 5 | DIVER-v3-GroupRank | 46.8 | 13 Nov 2025 | Yes |
| 6 | BGE-Reasoner-0928 (BAAI) | 46.4 | 13 Oct 2025 | Yes |
| 7 | Lattice Hierarchical Retrieval (Google) | 42.1 | 17 Oct 2025 | Yes |
| — | BM25 + GPT-4 reasoning + reranking | 30.4 | — | Yes |
| — | **BM25 alone** | **14.5** | — | No |

Long-document track (avg Recall@1, 8 datasets, docs up to ~40k): BM25 = 11.4.

**⚠️ Four caveats that reframe the 66.9.**
1. **Every single top entry uses a reranker.** BRIGHT in 2026 measures *pipelines with heavy test-time compute*, not retrievers.
2. **The gain is mostly LLM query expansion, not retrieval.** DIVER (Duolin Sun et al., Ant Group, arXiv:2508.07995, v5 2 Apr 2026) reports **46.8 overall but 31.9 on original queries** — a 14.9-point gap attributable to iterative LLM query rewriting. BM25 alone 14.5 → BM25 + GPT-4 reasoning + rerank 30.4 makes the same point: **LLM expansion roughly doubles a 1994 lexical baseline**.
3. **Verification varies by entry.** Ranks 5–7 link arXiv papers or GitHub; ranks 1–3 link personal/company webpages. Treat the 66.9 as an unrefereed submission.
4. Compare to BrowseComp-Plus, where the best retriever inside an agentic loop reaches **20.3 nDCG@10**. Standalone BRIGHT scores and agentic-loop utility are not the same quantity.

**⭐ The reproducibility audit: Lighting the Way for BRIGHT.** Sahel Sharifymoghaddam, Yijun Ge, **Jimmy Lin**. arXiv:2509.02558, 2 Sep 2025, **v2 1 Jun 2026, SIGIR 2026 Reproducibility Track**. Findings: (a) BRIGHT's published baseline silently uses **query-side BM25 ("BM25Q")**, an undocumented detail that consistently outperforms standard BM25 on long queries — meaning much of the literature has been comparing against a mis-specified lexical baseline; (b) **BM25Q's advantage is largely BRIGHT-specific** and does not carry to five other benchmarks, while fusion with standard BM25 does; (c) an audit of the BRIGHT corpus **uncovers data-quality issues that affect evaluation**.

**Successors.**
- **BRIGHT-Pro** — "Rethinking Reasoning-Intensive Retrieval: Evaluating and Advancing Retrievers in Agentic Search Systems." Yilun Zhao, Jinbiao Wei, Tingyu Song, Siyue Zhang, Chen Zhao, Arman Cohan. arXiv:2605.04018, 5 May 2026, **ACL 2026**. Direct critique: "benchmarks such as BRIGHT provide narrow gold sets and evaluate retrievers **in isolation**, while synthetic training corpora optimize single-passage relevance rather than **evidence portfolio construction**." Expert-annotated multi-aspect gold evidence; evaluates under **both static and agentic protocols**; introduces RTriever-Synth (aspect-decomposed, positive-conditioned hard negatives) and RTriever-4B (LoRA on Qwen3-Embedding-4B). Key claim: **agentic evaluation exposes retriever behaviors hidden by standard metrics.**
- **MM-BRIGHT** — Abdelrahman Abdallah et al. arXiv:2601.09562, 14 Jan 2026. 2,803 queries, 29 technical domains, four task types. BM25 text-only **8.5** nDCG@10; best text-only (DiVeR) **32.2**; Nomic-Vision multimodal-to-text **27.6**.

### 2.2 Reasoning-augmented retrievers and rerankers

**ReasonIR-8B** — Meta FAIR (facebookresearch/ReasonIR). arXiv:2504.20595, 29 Apr 2025. 76 citations. Synthetic pipeline generating reasoning-requiring queries plus **plausibly-related-but-unhelpful hard negatives**. **29.9 nDCG@10 on BRIGHT without reranker, 36.9 with.** Downstream: **+6.4% MMLU, +22.6% GPQA** over closed-book, beating other retrievers and search engines.

**RaDeR** — Debrup Das, Sam O'Nuallain, Razieh Rahimi. arXiv:2505.18405, 23 May 2025. Trained from **retrieval-augmented math reasoning trajectories** with self-reflective relevance evaluation; generalizes to BRIGHT and RAR-b. Two notable claims: it is the **first dense retriever to outperform BM25 when queries are chain-of-thought reasoning steps** (an admission of how bad the prior state was), and it matches/beats ReasonIR using **2.5% of ReasonIR's training data**.

**Reason-ModernColBERT** (LightOn, 149M params). **⚠️ Blog + HuggingFace model card only — no arXiv paper.** The marketing says it "outperforms all models up to 7B on BRIGHT" and beats ReasonIR-8B "by more than 2.5 nDCG on average." Its own model-card table says otherwise:

| Split group | Reason-ModernColBERT (149M) | ReasonIR-8B | BM25 |
|---|---|---|---|
| Mean StackExchange | **27.43** | 24.76 | 17.21 |
| Mean Coding | 19.79 | **22.75** | 16.15 |
| Mean Theorem | 15.38 | **24.60** | 7.17 |
| **Full BRIGHT mean** | 22.62 | **24.38** | 14.53 |

**The +2.5 claim holds only on StackExchange. On the full BRIGHT mean it loses, 22.62 vs 24.38, and loses badly on Theorem.** The "45× smaller" framing is real; the "beats it" framing is split-selective. License is cc-by-nc-4.0 (training-data restriction), so it is not commercially usable either.

**ReasonRank** — Wenhan Liu, Xinyu Ma, Weiwei Sun, Yutao Zhu, Yuchen Li, Dawei Yin, Zhicheng Dou (RUC + Baidu). arXiv:2508.07050, 9 Aug 2025, **ACL 2026 Main**. DeepSeek-R1-synthesized reasoning-intensive ranking labels; cold-start SFT then RL with a **multi-view ranking reward** for the multi-turn nature of listwise ranking. Outperforms baselines with **lower latency than pointwise rerankers**. Was #2 on BRIGHT (40.8 reranking RaDeR, Aug 2025).

**BGE-Reasoner** (BAAI + USTC, VectorSpaceLab/agentic-search). Multiple rewritten queries + ensembled reranking across model sizes. **46.4 nDCG@10 (BGE-Reasoner-0928, Oct 2025)**, held BRIGHT #1 briefly. ⚠️ GitHub/model-card, no verified paper.

### 2.3 Test-time compute for retrieval — the most interesting thread

**Rank1** — Orion Weller, Kathryn Ricci, Eugene Yang, Andrew Yates, Dawn Lawrie, Benjamin Van Durme (JHU HLTCOE). arXiv:2502.18418, 25 Feb 2025, **CoLM 2025**, 72 citations. First reranker trained to exploit test-time compute; **600,000+ R1 reasoning traces over MS MARCO** open-sourced. SOTA on reasoning and instruction-following ranking, "works remarkably well out of distribution."

**⭐ LATTICE: LLM-guided Hierarchical Search for End-to-end Reasoning Intensive Retrieval.** Nilesh Gupta, Wei-Cheng Chang, Ngot Bui, Cho-Jui Hsieh, Inderjit S. Dhillon (Google). arXiv:2510.13217, 15 Oct 2025, v2 25 May 2026. **Base LATTICE with a single off-the-shelf LLM reaches 46.7 nDCG@10 on BRIGHT — matching the best fine-tuned ensemble baseline overall — and LATTICE++ (fused with cheap retrieval) reaches 49.1.** Budget behavior: "reranking offers a better tradeoff at low token budgets, but LATTICE converges to a higher asymptote after a moderate budget."

This is the strongest single result in reasoning-aware retrieval, and it is under-cited: **an untrained LLM doing hierarchical search matches purpose-trained reasoning-retrieval ensembles.** It reframes the whole area as a test-time-compute allocation problem rather than a representation-learning problem.

**Reranker-Guided Search (RGS)** — Haike Xu, Tong Chen. arXiv:2509.07163, 8 Sep 2025. Greedy search on proximity graphs to select *which* documents to send to the reranker, rather than reranking a fixed top-k. **+3.5 BRIGHT, +2.9 FollowIR, +5.1 M-BEIR**, all within a 100-document reranker budget.

**State Machine Reasoning (SMR)** — Dohyeon Lee, Yeonseok Jeong, Seung-won Hwang. arXiv:2505.23059, 29 May 2025. Discrete Refine/Rerank/Stop actions with early stopping instead of free-form CoT. On BEIR and BRIGHT: **+3.4% nDCG@10 while cutting token usage 74.4%.** Generalizes across LLMs and retrievers without task-specific tuning. The cleanest "overthinking is real in IR" result.

**Verbal-R3** — Sangkwon Park, Donghun Kang, Jisoo Mok, Sungroh Yoon (SNU). arXiv:2605.01399, 2 May 2026, **ACL 2026 Main**. A "Verbal Reranker" emitting both relevance scores and analytic narratives connecting query to context, plus **relevance-guided test-time scaling** for trajectory expansion.

**⭐ Beyond Semantic Similarity: Rethinking Retrieval for Agentic Search via Direct Corpus Interaction.** Zhuofeng Li, Haoxiang Zhang, Cong Wei, Pan Lu, Ping Nie, Yi Lu, Yuyang Bai, Shangbin Feng, Hangxiao Zhu, Ming Zhong, Yuyu Zhang, Jianwen Xie, **Yejin Choi, James Zou, Jiawei Han, Wenhu Chen, Jimmy Lin**, Dongfu Jiang, Yu Zhang. arXiv:2605.05242, 3 May 2026. Agents search raw corpora with **grep and shell commands** — no embedding model, no vector index. Claims DCI "substantially outperforms sparse, dense, and reranking baselines" on BRIGHT and BEIR, and performs strongly on BrowseComp-Plus and multi-hop QA **without any semantic retriever**. Thesis: "retrieval quality depends not only on reasoning ability but also on the **resolution of the interface** through which models access corpora." ⚠️ **I could not extract the numeric tables** — the arXiv HTML 404s and the PDF's tables are in compressed streams. Existence, authorship, date, and qualitative claims verified; **the numbers are not**. Given the author list this deserves a follow-up read.

**Orion** — Supriti Vijay, Aman Priyanshu, Anu Vellore, Baturay Saglam, Amin Karbasi. arXiv:2511.07581, 10 Nov 2025. 350M–1.2B models doing iterative retrieval via synthetic trajectories + SFT + RL + inference-time beam search: **SciFact 77.6 (vs 72.6), BRIGHT 25.2 (vs 22.1), NFCorpus 63.2 (vs 57.8)**, beating retrievers 200–400× larger on 5 of 6 benchmarks with 3% of the training data.

**⚠️ Withdrawn paper:** "Adaptive Retrieval for Reasoning-Intensive Retrieval" (REPAIR), arXiv:2601.04618, submitted 8 Jan 2026, **withdrawn by the authors 14 Apr 2026**. It claimed +5.6pp. Do not cite it.

**→ OPEN DEBATE #2 — is reasoning-aware retrieval a training problem or a test-time-compute problem?** ReasonIR/RaDeR/BGE-Reasoner say train the retriever. LATTICE (Google) matches the best trained ensembles with an **off-the-shelf** LLM. BrowseComp-Plus says the reasoning-specialized retriever (ReasonIR-8B) **loses to a general scaled embedder (Qwen3-8B)** inside an agentic loop. SMR says a large fraction of the reasoning tokens are pure waste (−74.4% tokens for +3.4% nDCG). The training camp has never been evaluated against the test-time-compute camp under a matched compute budget.

---

## 3. RETRIEVAL ARCHITECTURE COMPONENTS

### 3.1 Late-interaction revival

**The modern models.** *GTE-ModernColBERT-v1* (LightOn, on `gte-modernbert-base`, 128-dim/token): **BEIR avg nDCG@10 = 54.67** vs answerai-colbert-small 53.79 reported / 53.35 on LightOn's rerun. LongEmbed(32k) mean 88.39. ⚠️ Vendor model-card numbers; the **+0.88 to +1.32 margin is under 1.5 points**, and the card's rerun of the competitor scores below the competitor's published figure — the classic pattern warranting skepticism.

**The best-documented efficiency table in the lineage** — *mxbai-edge-colbert-v0* (Rikiya Takehi, Benjamin Clavié, Sean Lee, Aamir Shakir; Mixedbread/Answer.AI). arXiv:2510.14880, 16 Oct 2025:

| model | params | dim | BEIR avg | LongEmbed 32k | CPU time | Mem/10k docs |
|---|---|---|---|---|---|---|
| mxbai-edge-colbert-17m | 17M | 48 | 0.490 | 0.847 | 487s | **275 MB** |
| mxbai-edge-colbert-32m | 32M | 64 | 0.521 | 0.849 | 589s | 366 MB |
| ColBERTv2 | 130M | 128 | 0.488 | 0.428 | **1540s** | **732 MB** |
| answerai-colbert-small-v1 | 33M | 96 | 0.534 | — | 621s | 549 MB |
| GTE-ModernColBERT-v1 | 130M+ | 128 | 0.547 | 0.898 | — | — |

**The efficiency lineage, verified:**
- **XTR** — Jinhyuk Lee, Zhuyun Dai, Sai Meher Karthik Duddu, Tao Lei, Iftekhar Naim, Ming-Wei Chang, Vincent Y. Zhao (Google DeepMind). arXiv:2304.01982, **NeurIPS 2023**. +2.8 BEIR nDCG@10, scoring **2–3 orders of magnitude cheaper** than ColBERT.
- **MUVERA** — Laxman Dhulipala, Majid Hadian, Rajesh Jayaram, Jason Lee, Vahab Mirrokni (Google Research). arXiv:2405.19504, 29 May 2024, **v2 8 Jun 2026 corrected the Theorem 2.1 dimension bound**. Reduces multi-vector to single-vector MIPS via asymmetric Fixed Dimensional Encodings. **2–5× fewer candidates at equal recall; 10% higher recall with 90% lower latency** vs PLAID across BEIR. Now theoretically bracketed by the same group: arXiv:2607.20393 proves near-matching lower bounds (MUVERA is near-optimal for FDE-style reductions); arXiv:2606.23475 proves multi-vector is formally more expressive than single-vector.
- **CRISP** — Veneroso, Jayaram, Rao, Hernández Ábrego, Hadian, Cer (Google Research). arXiv:2505.11471, 16 May 2025. Clustering trained *into* the model: **~3× vector reduction while beating the unpruned model**; 11× at 3.6% loss.
- **WARP** — Jan Luca Scheerer, Matei Zaharia, Christopher Potts, Gustavo Alonso, **Omar Khattab**. arXiv:2501.17788, **SIGIR 2025**. **41× latency reduction vs XTR reference; 3× over ColBERTv2/PLAID.**
- **ColBERT-serve** (arXiv:2504.14903): memory-mapped scoring, **90% RAM reduction**. **Constant-space multi-vector** — MacAvaney, Mallia, Tonellotto, **ECIR 2025** (arXiv:2504.01818): fixed vector count decoupled from doc length. ⚠️ exact deltas in PDF, unverified.
- 2026 kernels/indexes: **ColBERTSaR** (Eugene Yang, Andrew Yates, Dawn Lawrie, Mayfield, Samuel, Jha, JHU HLTCOE; arXiv:2606.05568) **50–70% smaller than a 1-bit PLAID index**; **No More K-means** (arXiv:2605.30120, **ICML 2026**) **15× faster indexing than ColBERTv2**; **TileMaxSim** (arXiv:2606.26439) 100K-candidate scoring **268ms → 1.2ms** on H100; **FLASH-MAXSIM** (IBM, arXiv:2605.29517) **9× less inference / ~100× less training memory** at ColPali scale; **FastLane** (Ramnath Kumar, Prateek Jain, Cho-Jui Hsieh, Google; arXiv:2601.06389) **up to 30× lower compute**; **LEMUR** (arXiv:2601.21853) order-of-magnitude faster MV search; **PLAID-PRF** (Xiao Wang, MacAvaney, Macdonald; arXiv:2607.18626) **+4.3% nDCG@10** from pseudo-relevance feedback over PLAID centroids.

**Honest 2026 storage/latency multiplier.** Naive ColBERTv2 at 128-dim fp16/token is **~732 MB per 10k docs** vs ~15 MB for a 768-dim single-vector index — that is the folk "50–100×". Stack the 2025–26 techniques (dim 128→48, small backbone, CRISP training-time clustering, PQ) and you land at **~3–10×**. Anyone quoting 100× in 2026 is citing ColBERTv1. Latency: PLAID's fastest measured point is **73–80.5 ms/query** on MS MARCO; WARP claims 3× on top; **~10–30 ms/query** is the current well-engineered figure.

**⭐ The strongest paper arguing late interaction is NOT worth it.** "A Reproducibility Study of PLAID" — **Sean MacAvaney & Nicola Tonellotto, SIGIR 2024 Reproducibility Track**, arXiv:2404.14989.

| PLAID setting | nprobe | t_cs | ndocs | latency | DL19 nDCG@10 |
|---|---|---|---|---|---|
| (a) | 1 | 0.50 | 256 | 80.5 ms | 0.739 |
| (b) | 2 | 0.45 | 1024 | 103.4 ms | 0.745 |
| (c) | 4 | 0.40 | 4096 | 163.9 ms | 0.745 |

**Re-ranking a BM25 candidate list with ColBERTv2 runs at as low as 9 ms/query at n=200, vs 73 ms/query for the fastest PLAID pipeline — an ~8× latency advantage — reaching RR@10 = 0.373 on MS MARCO Dev**, which the authors note beats early BERT cross-encoders. PLAID only wins at high latency where lexical recall binds. Their mechanistic finding: **most PLAID token clusters are predominantly aligned with a single token** — the centroid machinery approximates lexical matching. Their charge is methodological: prior late-interaction efficiency work **omitted the obvious baseline**. **This has not been rebutted with an updated head-to-head.**

Corroborating skepticism: *Are LLM-Based Retrievers Worth Their Cost?* (Abdallah, Holdcroft, Ali, Jatowt, **SIGIR 2026**, arXiv:2604.03676) — 14 retrievers × 12 BRIGHT tasks; large LLM bi-encoders incur substantial latency for modest gains, reasoning augmentation shows diminishing returns, and **confidence calibration is weak across all families**, so raw scores are unreliable for routing. *KaLM-Reranker-V1: **Fast but Not Late Interaction*** (arXiv:2606.22807). *MICE* (arXiv:2602.16299): **4× lower latency than standard cross-encoders while matching ColBERT-class quality**. *MINER* (arXiv:2605.06460): narrows the MV-to-dense gap to **0.2 nDCG@5**. *Your Embedding Model is SMARTer Than You Think* (arXiv:2605.24938): late interaction over **frozen hidden states** of existing single-vector models, no dedicated MV index needed.

And the proponents concede it: the **LIR @ ECIR 2026** workshop proposal (Benjamin Clavié, Xianming Li, Antoine Chaffin, **Omar Khattab**, Tom Aarsen, Manuel Faysse, Jing Li; arXiv:2511.00444) says these models pose "significant challenges of efficiency, usability, and integration" and "prohibitive storage and computational overhead," and explicitly solicits **negative or puzzling results**.

**⭐ The strongest pro-multi-vector argument is theoretical.** "On the Theoretical Limitations of Embedding-Based Retrieval" — **Orion Weller, Michael Boratko, Iftekhar Naim, Jinhyuk Lee (Google DeepMind + JHU)**, arXiv:2508.21038, v2 12 Mar 2026, **ICLR 2026**. The number of top-k document subsets returnable by *any* query is bounded by embedding dimension. LIMIT (50k docs, 1000 queries, k=2):

| model | R@2 | R@10 | R@100 |
|---|---|---|---|
| **BM25** | **97.8** | **100.0** | **100.0** |
| GTE-ModernColBERT (MV) | 23.1 | 34.6 | **54.8** |
| Promptriever Llama3 8B | 3.0 | 6.8 | 18.9 |
| GritLM 7B | 2.4 | 4.1 | 12.9 |
| Gemini Embedding | 1.6 | 3.5 | 10.0 |
| E5-Mistral 7B | 1.3 | 2.2 | 8.3 |
| Qwen3 Embedding | 0.8 | 1.8 | 4.8 |

⚠️ **LIMIT is adversarially constructed to hit the bound.** It proves an existence claim, not a claim about natural query distributions. And it is contested: **Bangachev, Bresler, Kogan, Polyanskiy (MIT)**, arXiv:2605.23556, 22 May 2026, don't dispute the bound but prove near-optimal margins are achievable at **d = O(k log(n/k))** in the sparse regime (necessary and sufficient), margin Θ(k^(-1/2)), holding to trillions of points — i.e. d≈1000 is provably near-sufficient. Separately, *Spectral Retrieval* (arXiv:2605.24764) lifts LIMIT-small R@10 from **0.33 to 0.90 without retraining**, suggesting part of the failure is a scoring-function artifact.

**→ OPEN DEBATE #3.** Weller et al. (DeepMind, ICLR'26): single-vector is dimensionally capped, MV is the escape. Bangachev et al. (MIT): the cap is far looser than worst-case. MacAvaney & Tonellotto (SIGIR'24): even granting MV's quality edge, **BM25 + ColBERT reranking dominates the Pareto frontier at deployable latencies**. No 2026 paper reconciles the three.

### 3.2 Hybrid sparse+dense fusion

**The canonical fusion-function result is still pre-2025 and still unrebutted.** "An Analysis of Fusion Functions for Hybrid Retrieval" — Sebastian Bruch, Siyu Gai, Amir Ingber, **ACM TOIS Aug 2023**, arXiv:2210.11934. RRF is **sensitive to its parameters**; convex combination is **agnostic to score-normalization choice** (min-max, z-score, any linear transform are rank-equivalent); **CC beats RRF in- and out-of-domain** and is sample-efficient. ⚠️ **Practice/theory gap:** essentially every 2025–26 applied paper uses RRF anyway, because it needs no tuning data. Nobody has re-tested.

**2025–26 evidence:**

| Paper | Date | Finding |
|---|---|---|
| From Retrieval to Generation (Abdallah, Mozafari, Piryani, Ali, Jatowt), arXiv:2502.20245 | Feb 2025 | **BEIR nDCG@10: BM25 43.42 → hybrid 52.59 (+9.17)** |
| From BM25 to Corrective RAG: Text-and-Table (Akarsu, Karaman, Mierbach), arXiv:2604.01733 | Apr 2026 | **23,088 financial QA queries** — largest here. Two-stage hybrid + neural rerank Recall@5 = **0.816**. **BM25 outperforms dense on financial documents.** Recommends hybrid RRF + cross-encoder as the minimum viable baseline |
| Dissecting Agentic RAG, arXiv:2606.21553 | Jun 2026 | **Fixed hybrid RRF beats rule-based adaptive routing (+1.8 EM, +1.9 F1)**; EM 53.2% HotpotQA. Negative result for adaptive routing |
| KohakuRAG (Yeh, Ku, Huang, Tu), arXiv:2603.07612 | Mar 2026 | ⚠️ **Contrarian: hierarchical dense alone matches hybrid; BM25 adds only +3.1pp** |
| Training-Free Lexical-Dense Fusion for Conversational Memory, arXiv:2606.04194 | Jun 2026 | Late-interaction dense + BM25: **+8.8 to +17.2 Hit@1**, no training |
| DAT: Dynamic Alpha Tuning (Hsu, Tzeng), arXiv:2503.23013 | Mar 2025 | LLM-judged per-query dense/BM25 weighting beats fixed-weight hybrid |
| Hybrid Retrieval for Hallucination Mitigation (ISTI-CNR), arXiv:2504.05324 | 2025 | On HaluBench, hybrid gives highest accuracy on fails and lowest hallucination + rejection rates |

**Where BM25 still wins outright:** LIMIT (97.8 R@2 vs 0.8 for the best dense); financial text-and-table at 23k queries; low-latency first-stage (9 ms/q + rerank beating the entire PLAID frontier below ~70 ms/q). **Where it clearly loses:** aggregate suites — HAKARI-Bench (551 tasks, 43 languages) puts BM25 at **50.24** macro nDCG@10×100 vs best sub-1B dense **64.93**; and agentic loops — BrowseComp-Plus, GPT-5 at **55.9% with BM25 vs 70.12% with Qwen3-Embedding-8B**, with the authors noting BM25's documents are "less useful in the iterative deep research process."

**→ OPEN DEBATE #4.** BM25's standing is strongly task-conditional; blanket claims fail in both directions. And **+9.17 (arXiv:2502.20245) vs +3.1pp (KohakuRAG) for the hybrid gain is a direct, unexplained, unreplicated conflict** — plausibly a corpus-structure effect where hierarchical indexing already captures what BM25 contributes.

**Reproducibility infrastructure:** Pyserini (Lin et al., SIGIR 2021) remains the reference. ⚠️ "Gosling Grows Up: Retrieval with Learned Dense and Sparse Representations Using Anserini" (Lin group, SIGIR 2025) surfaced in search only — arXiv ID unverified, not guessed. *GPUSparse* (arXiv:2606.26441): **235× speedup over Pyserini CPU at 8.8M docs** ⚠️ single-author, unreviewed.

### 3.3 Learned sparse

**SPLADE-v3** — Carlos Lassance, Hervé Déjean, Thibault Formal, Stéphane Clinchant (Naver Labs Europe). arXiv:2403.06789, 11 Mar 2024. **>40 MRR@10 on MS MARCO dev; +2% out-of-domain on BEIR** over SPLADE++. Meta-analysis over **40+ query sets**: statistically significantly better than BM25 and SPLADE++, **competitive with cross-encoder rerankers**.

**⭐ The 2026 headline: LACONIC** — Zhichao Xu, Shengyao Zhuang, Crystina Zhang, Xueguang Ma, Yijun Tian, Maitrey Mehta, **Jimmy Lin**, Vivek Srikumar (Utah + CSIRO + Waterloo). arXiv:2601.01684, 4 Jan 2026. Two-phase curriculum on Llama-3 1B/3B/8B: weakly-supervised pre-finetuning for bidirectional contextualization, then hard-negative finetuning. **8B: 60.2 nDCG on MTEB Retrieval, ranked 15th as of 1 Jan 2026, with 71% less index memory than an equivalent dense model**, running on standard CPU hardware. ⚠️ Self-reported rank; "15th" also means 14 dense models beat it.

**SPLARE** — Thibault Formal, Antoine Louis, Hervé Déjean, Stéphane Clinchant (Naver Labs). arXiv:2603.13277, **ICLR 2026**. Replaces the vocabulary projection with **sparse-autoencoder features**; SPLARE-7B posts top results on MMTEB multilingual + English retrieval.

Other verified: **CSPLADE** (Zhichao Xu, Aosong Feng, Yijun Tian, Haibo Ding, Lin Lee Cheong, Amazon; arXiv:2504.10816, **IJCNLP-AACL 2025 Main**) — 8B-scale LSR, fixes early-stage contrastive instability and unidirectional attention. **Li-LSR** (arXiv:2505.01452) — **inference-free query encoding** via table lookup, **+1–1.8 nDCG over Splade-v3-Doc**. **Sparton** (arXiv:2603.25011) — fused Triton kernel, **+33% batch size, 14% faster training**. **V-SPLADE** (Naver, arXiv:2605.30917) — inference-free multimodal LSR for production visual document search. **MILCO** (arXiv:2510.00671) — multilingual LSR via a shared English lexical space. **UEmbed** (Alibaba/Tongyi, Pengjun Xie et al., arXiv:2608.02583) — decoder-only model emitting sparse and dense simultaneously, 2B–9B. ⚠️ Skeptical note: *Understanding Wacky Weights* (Polyakov, Scells, Eickhoff, arXiv:2605.19628) finds larger vocabularies correlate with **semantically unrelated** expansion terms — SPLADE is less interpretable than assumed.

**Efficiency asymmetry worth knowing** (from HAKARI-Bench on SPLADE-v3): **document-side pruning costs only +0.01–0.04 points** at d=256→512, while **query-side reduction costs +2.5–3.6 points** at q=8→32.

**→ EMERGING SYNTHESIS.** ColBERTSaR (JHU, arXiv:2606.05568) demonstrates **ColBERT with product quantization is equivalent to learned-sparse retrieval**. Combined with MacAvaney & Tonellotto's finding that PLAID clusters align ~1:1 with tokens, there is a coherent thread arguing **late interaction and learned sparse are converging on the same mechanism** — which would dissolve Debate #3 into an implementation question. Not yet consensus.

### 3.4 Embedding model scaling and MTEB standings

⚠️ **I could not scrape the live MTEB leaderboard.** The HF Space is a client-rendered Gradio app; every fetch returned the loading shell. **I will not state a mid-2026 #1 as fact.**

**Verified anchors:** *Qwen3-Embedding-8B* — **70.58 MTEB Multilingual, No.1 as of 5 Jun 2025** (arXiv:2506.05176, Alibaba Tongyi; 0.6B/4B/8B, 119 languages). *Gemini Embedding* — **68.32 Task Mean on MTEB(Multilingual), highest at time of writing** (Jinhyuk Lee, Feiyang Chen, Sahil Dua, Daniel Cer + 43 co-authors, Google DeepMind; arXiv:2503.07891, 10 Mar 2025; size undisclosed). Both self-reported and now stale.

**Verified mid-2026 cross-vendor table** (jina-embeddings-v5-text, Jina AI, arXiv:2602.15547, 17 Feb 2026):

| model | params | MMTEB avg | MTEB-Eng | Retrieval nDCG@10 |
|---|---|---|---|---|
| Qwen3-4B (teacher) | 4B | **69.5** | **74.6** | **69.60** |
| jina-v5-text-small | 677M | 67.0 | 71.7 | 64.88 |
| jina-v5-text-nano | 239M | 65.5 | 71.0 | 63.26 |
| Qwen3-0.6B (instruct) | 596M | 64.3 | 70.5 | 64.65 |
| multilingual-e5-large-instruct | 560M | 63.2 | 65.5 | 57.12 |
| Qwen3-0.6B (generic) | 596M | 61.1 | 67.0 | — |
| embeddinggemma-300m | 308M | 61.1 | 69.7 | 62.49 |
| voyage-4-nano | 340M | 58.9 | 63.3 | 63.58 |
| jina-v3 | 572M | 58.4 | 65.7 | 55.76 |
| snowflake-arctic-embed-l-v2 | 568M | 57.0 | 63.6 | 58.36 |

⚠️ Vendor-authored; comparison set deliberately sub-1B except the teacher.

**Independent 2026 comparison: HAKARI-Bench** (Yuichi Tateno, arXiv:2606.22778, 22 Jun 2026) — 35 benchmarks / 551 tasks / 43 languages / 55 models, nano-sets validated at **Spearman 0.983 vs MTEB v2, 0.975 vs MMTEB v2, 0.973 vs full BEIR**. Top by macro nDCG@10×100: **jina-v5-text-small 64.93**, jina-v5-text-nano 63.80, **microsoft/harrier-oss-v1-0.6b 63.68**, **perplexity-ai/pplx-embed-v1-0.6b 63.64**, embeddinggemma-300m 62.58 — *BM25 baseline 50.24*. Rerankers beat all: Qwen3-Reranker-0.6B **68.03**. Quantization deltas (mean over 33 models): binary **−6.50**, int8 **−1.95**, binary+rescore **−0.93**, **int8+rescore −0.09 (effectively lossless)**. ⚠️ Single-author, ≤~1B scope, nano-sets not full benchmarks — but the only independent unified-conditions 2026 comparison found.

**Net honest answer:** among ≤1B open-weight models, **jina-embeddings-v5-text-small** sits at or near the top on two independent 2026 sources. Overall, **Qwen3-Embedding-8B / Qwen3-4B** remain the reference ceiling with **Gemini Embedding** the strongest closed model on last-verified data.

**⭐ Scaling laws.** "Scaling Laws for Embedding Dimension in Information Retrieval" — Julian Killingback, Mahta Rafiee, Madine Manas, **Hamed Zamani** (UMass CIIR). arXiv:2602.05062, 4 Feb 2026. Retrieval performance vs embedding dimension **fits a power law**, with predictive models on dimension alone and jointly with model size. **Aligned tasks: monotone improvement with diminishing returns. Misaligned tasks: unpredictable — larger embedding dimension can actively degrade results.** That second half is the non-obvious finding and directly cautions against "bigger dim is safer." Adjacent: *Retrieval Capabilities of LLMs Scale with Pretraining FLOPs* (arXiv:2508.17400); *BitNet Text Embeddings* (Zhen Li, Xin Huang, Liang Wang, Nan Yang, **Furu Wei**, MSR; arXiv:2606.25674) — extreme quantization "largely comparable" to full-precision teachers on MMTEB(eng,v2). Low end: *Bekko Embedding* (arXiv:2607.25180) — **8M active params → 56.2 MMTEB Multilingual v2; 25M → 57.5** (vs BM25's 50.24).

**⭐ The leaderboard-integrity failure, primary-source verified.** GitHub `embeddings-benchmark/mteb` **issue #3934**, "Decision: Temporary removal of the private RTEB column", **14 Jan 2026**. RTEB was built with private test sets specifically to defeat overfitting — but it was **co-developed with Voyage AI (since acquired by MongoDB), who therefore had direct access to the private evaluation data while competing on the leaderboard.** The MTEB team's words: *"the uneven playing field fundamentally undermines trust in MTEB leaderboards, which is unacceptable for a community benchmark."* No misuse alleged; the structural conflict alone forced the call. The private column was removed; it returns once the private pool is diversified with contributions from orgs that don't ship competing models.

**Peer-reviewed critiques, six independent groups reaching the same conclusion:** *On the Robustness of Multilingual Text Embedding Rankings* (Gjorgjevikj, Koroušić Seljak, Eftimov, arXiv:2605.31142) — **MTEB rankings are sensitive to dataset composition and aggregation method; conclusions lack robustness**. *MTEB-BR* (arXiv:2607.04581) — multilingual leaderboard correlates only **ρ=0.75** with Brazilian Portuguese performance. *MTEB-PT* (arXiv:2607.04071) — same for Portuguese. *LMEB* (Xinping Zhao et al., arXiv:2603.12572) — long-horizon memory results **orthogonal to MTEB; larger models don't reliably win**. *STEB* (Rivera Soto, Wegmann, Aggazzotti, arXiv:2606.31741) — semantic embeddings **consistently fail on stylistic tasks**. *PosIR* (arXiv:2601.08363, 310 datasets, 10 MMTEB SOTA models) — **position bias is pervasive and invisible to MTEB**. *CS-MTEB* (arXiv:2604.17632) — **up to 27% degradation** on code-switched queries. *SABER-Math* (arXiv:2606.29894) — "general-purpose IR benchmarks such as MTEB do not reliably predict mathematical performance." Also *HTEB* (arXiv:2605.28190) and *PTEB* (arXiv:2510.06730) proposing stochastic re-paraphrasing at eval time.

**The best-supported claim in this whole section: a high MTEB/MMTEB average does not transfer to your specific language, domain, or task.**

⚠️ **Instruction-following embeddings — partial coverage.** Search budget ran out before a dedicated pass. What is verified: instruction conditioning is worth **+3.2 MMTEB / +3.5 MTEB-Eng at identical parameter count** (Qwen3-0.6B instruct 64.3/70.5 vs generic 61.1/67.0); **Promptriever Llama3 8B was the best single-vector model on LIMIT (R@100 = 18.9 vs Qwen3's 4.8)**, i.e. instruction-following retrievers are meaningfully more robust to the dimensional bound; and **MMTEB** (Kenneth Enevoldsen + 64 co-authors, **ICLR 2025**, arXiv:2502.13595, 500+ tasks / 250+ languages) added instruction-following as a task category and found **smaller multilingual models often outperform large LLMs**.

---

## 4. CONTEXT HANDLING

### 4.1 Late chunking

**The primary paper is arXiv-only, from the vendor that sells the embeddings it was tested on.** "Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models" — Michael Günther, Isabelle Mohr, Daniel James Williams, Bo Wang, Han Xiao (Jina AI). arXiv:2409.04701, v1 7 Sep 2024, v3 7 Jul 2025. Comments field says "11 pages, 3rd draft." **No venue.** It has an OpenReview page (74QmBTV0Zf) with no acceptance record.

Verified numbers (Table 2, nDCG@10, fixed 256-token boundaries, naive → late):

| Dataset | jina-v2-small | jina-v3 | nomic-v1 |
|---|---|---|---|
| SciFact | 64.2 → 66.1 | 71.8 → 73.2 | 70.7 → **70.6 (↓)** |
| NFCorpus | 23.5 → 30.0 | 35.6 → 36.7 | 35.3 → **35.3 (=)** |
| FiQA | 33.3 → 33.8 | 46.3 → 47.6 | 37.0 → 38.3 |
| TRECCOVID | 63.4 → 64.7 | 73.0 → 77.2 | 72.9 → 75.0 |
| **Average** | — | — | **52.2 → 54.0** |

**The headline effect is +1.5 to +1.9 nDCG@10 absolute (2.7–3.6% relative)**, and it is not uniform. The authors' own conceded failure case: late chunking **hurts on synthetic needle tasks** (Needle-8192, Passkey-8192), because contextualizing a planted needle with unrelated filler dilutes it. Their comparison against contextual retrieval (Table 4) is **a single anecdotal example with one query** — cosine 0.6343 naive → 0.8516 late → 0.8590 contextual. Not an experiment. Their argument against contextual retrieval is cost, not accuracy.

**⭐ The independent replication is mixed-to-negative.** "Reconstructing Context: Evaluating Advanced Chunking Strategies for RAG" — Carlo Merola, Jaspinder Singh. arXiv:2504.19754, **2nd Workshop on Knowledge-Enhanced IR, ECIR 2025**. NDCG@5, early → late: NFCorpus Stella-V5 0.443→0.445; jina-v3 0.374→0.380; jina-v2 0.261→0.280; **BGE-M3 0.246 → 0.070**. MSMarco, Stella-V5: **0.630 → 0.503**.

**This is the single most important refutation datapoint: late chunking is model-dependent, can catastrophically fail (BGE-M3, ~72% relative collapse), and loses badly on MSMarco with a strong non-Jina encoder.** The gains appear to hold mainly on small domain-specific corpora with Jina's own models. ⚠️ Workshop paper, small scale; the BGE-M3 collapse is plausibly a pooling-compatibility artifact. But nobody has published a rebuttal, which is itself informative. Same paper's head-to-head: contextual retrieval NDCG@5 **0.317** vs late chunking **0.309** on an NFCorpus subset — a ~2.6% relative edge at much higher cost.

**⭐ The best systematic 2026 evaluation.** "Beyond Chunk-Then-Embed: A Comprehensive Taxonomy and Evaluation of Document Chunking Strategies for IR" — Yongjie Zhou, Shuai Wang, Bevan Koopman, **Guido Zuccon** (Queensland/CSIRO). arXiv:2602.16974, 19 Feb 2026. Abstract, verbatim: *"Contextualized chunking improves in-corpus effectiveness but degrades in-document retrieval."*

- **In-corpus (BEIR, nDCG@10):** jina-v3 — **Paragraph 0.4948** > Fixed 0.4849 > Semantic 0.4726 ≈ Sentence 0.4723 > LumberChunker 0.4690 > Proposition 0.3888. **Structure-based beats LLM-guided by 5–27%.**
- **In-document (GutenQA, 3k QA over 100 books, DCG@10):** LumberChunker wins decisively — jina-v3 0.5640 vs Paragraph 0.4574.
- **Effect of late chunking:** in-corpus, Proposition **+22.87% to +26.94%**; LumberChunker +2.42% to +4.80%; structure-based **0% to +7.20%**. In-document, Paragraph **−10.76% to −62.47%**; Sentence −5.39% to −62.57%.
- Throughput: LumberChunker **1.11 docs/s vs paragraph-based 1,854 docs/s — ~1,600× slower.**

Late chunking's benefit is largest exactly where naive chunks are most context-starved, near-zero where chunks are already coherent, and **actively harmful for needle-style in-document retrieval** — independently corroborating the authors' own caveat.

**The chunking-strategy benchmark literature.** *Is Semantic Chunking Worth the Computational Cost?* (Renyi Qu, Ruixuan Tu, Forrest Sheng Bao; arXiv:2410.13070, **Findings of NAACL 2025**) — cost not justified; fixed ~200-word chunks match or beat it. *Chunking Methods on RAG: Effectiveness vs Computational Cost* (Wrocław UST, arXiv:2606.00881) — fixed-size and recursive-semantic most stable; **LumberChunker highest answer quality but completed on only ~30% of datasets** (timeouts); runtimes <1s vs **8.37 h**; DenseX 15+ h; only 5 of 8 methods completed everywhere. *A Systematic Investigation of Document Chunking Strategies* (Shaukat, Adnan, Kuhn, U. Canberra; arXiv:2603.06976) — largest sweep: 36 approaches × 6 domains × 5 embedders = 1,080 configs; **Paragraph Group Chunking best (mean nDCG@5 ≈ 0.459, P@1 24%), naive fixed-*character* worst (nDCG@5 < 0.244, P@1 ≈ 2–3%)**. *Evaluating Chunking Strategies for RAG on Academic Texts* (arXiv:2607.01852) — cluster-based semantic chunking **did not outperform** fixed-size or recursive; also warns **RAGAs faithfulness "shows limited reliability in this setup."**

**→ OPEN DEBATE #5, and it dissolves on inspection.** Four papers say simple wins; Shaukat et al. reports a ~2× nDCG spread implying chunking is a vital lever. The reconciliation: **Shaukat's worst baseline is fixed-*character* splitting (which shreds words), while the "simple wins" camp baselines against fixed-token or recursive splitting.** Consensus as of mid-2026: **paragraph- or sentence-respecting structural chunking is a strong, near-optimal, essentially free baseline; embedding-similarity semantic chunking and LLM-guided chunking do not reliably beat it in-corpus at 100–1,600× the cost.**

**Descendants.** *Context is Gold to find the Gold Passage* (Conti, Faysse, Viaud, Bosselut, Hudelot, Colombo; arXiv:2505.24782) — introduces **ConTEB** benchmark and **InSeNT** in-sequence-negative contrastive post-training combined with late-chunking pooling; the most credible academic endorsement of the pooling operator, **but it requires training — zero-shot late chunking is the weaker claim.** Also *ColChunk* visual late chunking (arXiv:2604.10167), *Graph-Aware Late Chunking for Biomedical* (arXiv:2603.22633), *pplx-embed* (Perplexity, arXiv:2602.11151 — uses late chunking in production; ⚠️ Bo Wang is a Jina co-author, not independent).

### 4.2 Contextual retrieval

**The source is a blog post. There is no paper.** Anthropic engineering blog, 19 Sep 2024. Verified claims: top-20 retrieval **failure rate 5.7% → 3.7% (−35%)** with contextual embeddings; **→ 2.9% (−49%)** adding contextual BM25; **→ 1.9% (−67%)** adding reranking. Cost **$1.02 per million document tokens**. Anthropic's own caveat: performance varies by embedding/source combination, run your own evals.

**No arXiv version, no peer review, no dataset release, and no independent reproduction of 35/49/67 exists as of August 2026.** Anyone quoting "67%" is quoting a vendor blog with an unreleased internal eval.

What academic work reports: the only head-to-head (Merola & Singh, ECIR 2025 workshop) puts contextual retrieval **~2.6% relative ahead of late chunking on an NFCorpus subset**, at substantially higher cost. Zhou et al. (2026) find **structure-based segmentation beats LLM-guided methods by 5–27% nDCG@10 in-corpus** at 1,600× the throughput; contextualization helps most for propositions (+23–27%) and barely at all for paragraphs (0 to +7.2%). ⚠️ A single-author, non-peer-reviewed Spanish-language preprint (*More Context Is Not Better: The Vector Dilution Paradox*, arXiv:2601.08851) reports an inverted-U: **moderate injection +18% recall; past a Contextualization Injection Ratio > 0.4, precision drops 22%** on targeted queries. Treat the numbers as unverified, but the shape is consistent with Zhou et al.

Cost reality the blog omits: contextual retrieval **doubles index-build storage and forces full re-contextualization whenever chunk boundaries change**.

**→ OPEN DEBATE #6.** Position A (Anthropic, unreplicated): 49–67% failure reduction, cheap. Position B (Zhou et al. 2026): LLM-side chunk processing loses to paragraph splitting in-corpus by 5–27%. Position C (Merola & Singh): best of the advanced methods, by ~1–3% on a small subset. Position D (unreviewed): over-injection costs 22% precision. **Practitioner verdict: directionally sound, most valuable for short/proposition-sized chunks in anaphora-heavy corpora (filings, codebases). The 35/49/67 figures should be cited as vendor-internal, not established.**

### 4.3 Context compression

**The lineage, all verified:**

| Method | Authors / lab | arXiv | Venue | Headline |
|---|---|---|---|---|
| LLMLingua | Jiang, Wu, Lin, Yang, Qiu (MSR) | 2310.05736 | **EMNLP 2023** | up to 20× compression, little loss |
| LongLLMLingua | Jiang, Wu, Luo, Li, Lin, Yang, Qiu (MSR) | 2310.06839 | **ACL 2024** | NaturalQuestions **+21.4% with ~4× fewer tokens**; **94.0% cost reduction** on LooGLE; **1.4–2.6× latency** speedup |
| LLMLingua-2 | Pan, Wu, Jiang, Xia, Luo, Zhang, Lin, Rühle, Yang, C.-Y. Lin, Zhao, Qiu, Zhang (MSR + Tsinghua) | 2403.12968 | **Findings ACL 2024** | **3–6× faster than prior compressors; 1.6–2.9× end-to-end latency reduction** at 2–5× |
| RECOMP | Fangyuan Xu, Weijia Shi, Eunsol Choi | 2310.04408 | ICLR 2024 | compression to **6%** with minimal loss; can emit empty string |
| xRAG | Cheng, Wang, Zhang, Ge, Chen, Wei, Zhang, Zhao (MSRA + PKU) | 2405.13792 | **NeurIPS 2024** | context → **one token**; **>10% avg improvement** on six tasks; **3.53× FLOPs reduction** |
| CompAct | Yoon, Lee, Hwang, Jeong, Kang | 2407.09014 | **EMNLP 2024** | **47× compression**, strongest on multi-hop |
| PISCO | Naver Labs Europe | 2501.16075 | **Findings ACL 2025** | **16× compression, 0–3% loss**; +8% over prior compressors; 48 h on one A100 |

**⭐ Provence is the compression result that survives scrutiny.** Nadezhda Chirkova, Thibault Formal, Vassilina Nikoulina, Stéphane Clinchant (NAVER LABS Europe). arXiv:2501.16214, 27 Jan 2025, **ICLR 2025** (confirmed: poster 29557, OpenReview TDy5Ih78b4). Sentence-level pruning as binary sequence labeling on a DeBERTa cross-encoder, **fused with the reranker** so pruning is free in a pipeline that already reranks. One hyperparameter (threshold ∈ {0.1, 0.5}) that transfers across domains.

LLM-Eval scores: **NQ 72.4 at 62.2% compression vs 71.8 full context** (pruning *improves* — denoising); **HotpotQA 56.7 @ 66.4% vs 57.0 full** (−0.3); **PopQA 59.3 @ 68.6% vs 57.8 full** (+1.5). Baselines on NQ: LLMLingua-2 **59.5** @74%; LongLLMLingua **61.3** @69%; RECOMP-extractive **70.6** @44%; DSLR **71.7** @45%. **Cross-domain across 7 datasets — NQ, HotpotQA, TyDi QA, PopQA, BioASQ (biomedical), SyllabusQA (education), RGB (news) — "negligible to no drop."** Overhead essentially zero when unified with the reranker; generation speedups 1.2–1.4× at batch 1, 1.9–2.0× at batch 256.

⚠️ Caveat: 50–80% compression is far less aggressive than CompAct's 47× or xRAG's one token. The OOD robustness comes partly from not compressing very hard.

**⭐ The systematic evaluation bug in the entire compression literature.** "Fixed RAG Compression Collapses Measured Reader Scaling" — Sugam Panthi, Rabab Abdelfattah. arXiv:2606.21807, 20 Jun 2026. A *fixed* compressor helps weak readers (removes noise they can't filter) and hurts strong readers (removes detail they could have used). Across **20 readers × 10 domain-method settings, compression gains decreased with reader baseline in 9 of 10 settings (p < 0.05)**. Generic summarization **flipped 31% of pairwise model rankings on LongMemEval-S**. A fixed HotpotQA compressor **obscured 80% of the Qwen-7B → GPT-4.1-mini improvement**. Pattern holds across compressor types and an external audit of **nine published papers**. Released `ragscale` (177k row-level transitions).

**Every "X× compression with negligible loss" number above is reader-dependent, and the strong-reader case is systematically under-reported.**

Also: *No Mean Feat: Simple, Strong Baselines for Context Compression* — Yair Feldman, **Yoav Artzi** (Cornell Tech). arXiv:2510.20797, rev 10 May 2026. Introduces **BenchPress** and shows **mean pooling and a bidirectional compression-token variant strongly outperform the widely-used causal compression-token approach** — the design underlying much of the gist-token line — across scales, datasets, and ratios. *RAISE* (arXiv:2605.30029): 13 RAG algorithms × 7 datasets, "optimization performance is highly task-dependent" with **poor cross-dataset generalization**. *Control Under Compression* (arXiv:2608.01056): reliability "diverges sharply" in the 50–35% retained-context band and compressor rankings are **not universal**.

**2026 successors:** *CORE-RAG* (Cui, Weng, Tang, Liu, Li, He, Chen, Zhang, He, Ma; arXiv:2508.19282, v4 28 May 2026, **ICML 2026**) — performance-driven compression, **at a 3% compression ratio, +3.3 EM over feeding full documents**. *ARC-Encoder* (Kyutai, arXiv:2510.20535) — 4–8× compression adapting to multiple decoders. *Sentinel* (arXiv:2505.23277) — a **0.5B proxy achieves 5× compression competitive with 7B-scale methods**. Plus ECoRAG, ACC-RAG, AttnComp, EXIT, BRIEF (+3.0 EM / +4.16 F1 on HotpotQA), and *A Unified Model and Document Representation for On-Device RAG* (Killingback, Meshi, Li, **Zamani**, Karimzadehgan; arXiv:2604.14403 — matches traditional RAG with **1/10 the context**).

**→ OPEN DEBATE #7.** The field publishes 16–47× compression with 0–3% loss and even "compression improves accuracy." Panthi & Abdelfattah show this is largely an artifact of weak readers. Feldman & Artzi show the dominant architectural choice is beaten by mean pooling. **Reading: extractive, query-conditioned, moderate compression (Provence-style, 50–70%) is genuinely robust; soft/gist/extreme compression (xRAG one-token, CompAct 47×) has not been shown to survive either a strong reader or a domain shift.**

### 4.4 Long context vs RAG — what the evidence actually says

**The 2024 axis of debate.** *Retrieval Augmented Generation or Long-Context LLMs?* — Zhuowan Li, Cheng Li, Mingyang Zhang, Qiaozhu Mei, Michael Bendersky (Google DeepMind + Michigan). arXiv:2407.16833, **EMNLP 2024 industry track**. When sufficiently resourced, **LC consistently beats RAG on average** — but **LC and RAG predictions are identical for >60% of queries**, and SELF-ROUTE achieves LC-comparable quality at **65% cost reduction (Gemini-1.5-Pro), 39% (GPT-4o)**.

*In Defense of RAG in the Era of Long-Context LLMs* — Tan Yu, Anbang Xu, Rama Akkiraju (NVIDIA). arXiv:2409.01666. **OP-RAG** keeps retrieved chunks in original document order rather than relevance order; reports an inverted-U in chunk count. ∞Bench EN.QA F1: Llama3.1-70B full context **34.26 @117K tokens**; GPT-4o 32.36 @117K; Gemini-1.5-Pro 43.08 @196K; SELF-ROUTE GPT-4o 34.95 @85K. **OP-RAG on Llama3.1-70B: 44.43 @16K, 45.45 @24K, 47.25 @48K.** EN.MC accuracy: full-context Llama3.1-70B 71.62 @117K vs **OP-RAG 88.65 @24K**. So **47.25 F1 with 48K tokens vs 34.26 F1 with 117K**. ⚠️ Single benchmark (book-length novel QA — the format most favorable to retrieval), single model family, arXiv-only.

**Databricks.** *Long Context RAG Performance of LLMs* — Quinn Leng, Jacob Portes, Sam Havens, **Matei Zaharia**, Michael Carbin (Databricks Mosaic). arXiv:2411.03538, **NeurIPS 2024 Workshop on Adaptive Foundation Models** (workshop, not main track). 20 models, 2K→128K, text-embedding-3-large, 512-token chunks, FAISS. Accuracy 64k → 125k: **holds up** — o1-preview 0.831→0.763, GPT-4o 0.769→0.767, Claude 3.5 Sonnet 0.741→0.706; **degrades** — Llama 3.1 405B 0.587→0.426, GPT-4 Turbo 0.623→0.560. The valuable part is that **failure modes are qualitatively distinct**: Claude 3 Sonnet refused on copyright grounds increasingly with length; Gemini 1.5 Pro tripped safety filters; DBRX summarized instead of answering above 16k; Mixtral emitted repeated nonsense; Llama 3.1 405B gave consistent wrong answers. ⚠️ Late-2024 model generation; no equally systematic 2026 redo exists.

**Effective context length ≪ advertised.**
- **RULER** — Hsieh, Sun, Kriman, Acharya, Rekesh, Jia, Zhang, Ginsburg (NVIDIA). arXiv:2404.06654, **COLM 2024**. All models claim ≥32K; **only half maintain satisfactory performance at 32K**, despite near-perfect vanilla NIAH.
- **⭐ NoLiMa** — Modarressi, Deilamsalehy, Dernoncourt, Bui, Rossi, Yoon, Schütze (LMU + Adobe Research). arXiv:2502.05167, **ICML 2025**. Needles share **no lexical overlap** with the question — only associative links. **13 models all claiming ≥128K: at 32K tokens, 11 of 13 fall below 50% of their short-context baseline. GPT-4o: 99.3% → 69.7%.** Reasoning-enhanced models and CoT don't rescue it. **The cleanest demonstration that NIAH scores are an artifact of literal matching.**
- **HELMET** — Yen, Gao, Hou, Ding, Fleischer, Izsak, Wasserblat, Chen (Princeton + Intel Labs). arXiv:2410.02694, **ICLR 2025**. Seven application-centric categories to 128k. **"Synthetic tasks like NIAH do not reliably predict downstream performance"**; categories show low correlation with each other; the open-vs-closed gap **widens with length**.
- **LongBench v2** — arXiv:2412.15204, **ACL 2025**. 503 MCQs, 8k–2M words. **Human experts under a 15-min limit: 53.7%. Best direct-answering model: 50.1%. o1-preview with extended reasoning: 57.7%.**
- **⭐ Context Length Alone Hurts LLM Performance Despite Perfect Retrieval** — Du, Tian, Ronanki, Rongali, Bodapati, Galstyan, Wells, Schwartz, Huerta, Peng. arXiv:2510.05381, **Findings of EMNLP 2025**. **Even with perfect retrieval, accuracy degrades 13.9%–85% as input length grows — and the degradation persists when irrelevant content is replaced with whitespace or masked entirely.** Length itself, independent of distraction and retrieval quality, is a failure axis. Mitigation: prompt the model to recite retrieved evidence before answering (+up to 4% for GPT-4o on RULER).
- ⚠️ **"Context Rot"** (Kelly Hong, Anton Troynikov, Jeff Huber, Chroma, Jul 2025) is a **blog/tech report, not peer-reviewed**. Cite Du et al. (Findings EMNLP 2025) instead for the same claim.

**"Lost in the middle" has been substantially revised.** *Positional Biases Shift as Inputs Approach Context Window Limits* — Veseli, Chibane, Toneva, Koller (Saarland/MPI-SWS). arXiv:2508.07479, **COLM 2025**. **The U-curve is strongest only when the input occupies up to ~50% of the context window.** Beyond that, primacy weakens, recency holds, and it becomes a **distance-based bias**. Measuring in *relative* rather than absolute length is the methodological point most prior work got wrong. Also: *Lost in the Middle: An Emergent Property from Information Retrieval Demands* (arXiv:2510.10276); *On the Emergence of Position Bias in Transformers* (arXiv:2502.01951); and arXiv:2511.05850 reporting **Gemini 2.5 Flash shows no lost-in-the-middle effect for simple factoid QA**. Counterweight: *Stable-RAG* (arXiv:2601.02993) shows retrieval-**permutation**-induced hallucinations remain measurable.

**⭐ The most replicated practical finding of 2025–2026 is the dullest: preserve document order.** *Stronger Baselines for RAG with Long-Context LMs* — Alex Laitenberger, **Christopher D. Manning**, Nelson F. Liu (Stanford). arXiv:2506.03989, **EMNLP 2025**. **DOS RAG** ("Document's Original Structure") = retrieve-then-read preserving original passage order. It **matches or outperforms ReadAgent and RAPTOR** across long-context QA benchmarks and systematically varied token budgets; recommended as the mandatory baseline for future RAG papers. Same insight as OP-RAG, independently arrived at by a different group, and peer-reviewed.

**The cost evidence is unambiguous.** *The Token Tax of Epistemic Accuracy* — Hamilton, Singh, Wise, Yousif, Carvalho, Shan, Mayyas, Cavuoto, Megahed. arXiv:2606.20898, 18 Jun 2026. 972 answers, expert-validated manufacturing-safety benchmark: **long-context 73.1% correctness vs 65.4% for semantic RAG — at 26× the per-query token cost.** ⚠️ Small LMs, narrow domain; the gap would likely shrink with frontier models and a better-engineered RAG arm. Also: clinical EHR reasoning (arXiv:2508.14817) — **RAG with <8K tokens matches long-context**; on-device unified compression matches traditional RAG at **1/10 the context** (arXiv:2604.14403).

**And a strong result for minimal RAG:** *Frustratingly Simple Retrieval Improves Challenging, Reasoning-Intensive Benchmarks* — Lyu, Duan, Shao, Koh, Min (UW/Berkeley/AI2). arXiv:2507.01297. A *minimal* RAG pipeline over CompactDS gives **+10% MMLU, +33% MMLU Pro, +14% GPQA, +19% MATH** across 8B–70B, matching or beating Google Search and agentic RAG.

**→ OPEN DEBATE #8 — LC vs RAG.** Google DeepMind (EMNLP 2024) and Li et al. (arXiv:2501.01880) say LC wins on average. NVIDIA/OP-RAG and Lyu et al. say a properly ordered RAG beats LC at a fraction of the tokens. LaRA (arXiv:2502.09977, 2,326 test cases) says the question is ill-posed — it depends on parameter size, long-text ability, context length, task type, and chunk characteristics. Hamilton et al. quantify it as +7.7 points for 26× cost. **The reconciliation the 2025–26 evidence supports: most published "LC beats RAG" results used a weak RAG arm (relevance-ordered, badly chunked, small top-k). Fix ordering (DOS/OP-RAG) and recall and RAG closes most of the gap at 3–25× lower cost. Meanwhile LC has an independent problem — Du et al. show accuracy falls 13.9–85% with length even under perfect retrieval with distractors literally blanked out — so "just use the long window" is not stable as context grows.** Note also Li et al.'s under-quoted finding that **summarization-based retrieval performs comparably to LC while chunk-based retrieval lags** — the RAG side of most comparisons is handicapped by chunking, not by retrieval.

---

## 5. VERIFICATION, GROUNDING, ATTRIBUTION

### 5.1 Claim-level attribution

**The benchmarks.** *ALCE* (Tianyu Gao, Howard Yen, Jiatong Yu, Danqi Chen; **EMNLP 2023**, arXiv:2305.14627) — citation recall + precision via a TRUE-style NLI model; **~50% of the best models' generations on ELI5 are not fully supported by their own cited passages**. *AttrScore* (Xiang Yue, Boshi Wang, Ziru Chen, Kai Zhang, Yu Su, Huan Sun; **Findings EMNLP 2023**, arXiv:2305.06311) — defines the attributable/extrapolatory/contradictory taxonomy. ⚠️ **Per-model F1 numbers not extractable from the abstract or Anthology page — do not quote a specific AttrScore F1.** *HAGRID* (Ehsan Kamalloo, Aref Jafari, Xinyu Zhang, Nandan Thakur, **Jimmy Lin**; arXiv:2307.16883). *AttributionBench* (arXiv:2402.15089, Findings ACL 2024, OSU NLP) — ⚠️ from a search summary: **even a fine-tuned GPT-3.5 reaches only ~80% macro-F1** on binary supportedness. *LFRQA / RAG-QA Arena* (arXiv:2407.13998, EMNLP 2024, AWS AI Labs) — 26K queries, 7 domains; ⚠️ frequently misdescribed as an attribution benchmark, it is an answer-quality arena.

**⭐ The single most important 2026 result: attribution evaluators do NOT transfer.** "Do LLM Attribution Metrics Transfer? Auditing RAG Evaluation Across Datasets and Constructs" — Tianyu Ding, Aditya Nannapaneni, Juan Pablo De la Cruz Weinstein. arXiv:2606.23915, 22 Jun 2026. 8 scorers × 3 constructs; 1,610 AttributionBench + 2,150 HAGRID examples.

- **No scorer stayed inside the 95% CI across all datasets within any construct.**
- **Metric rankings invert across datasets: Kendall tau = −0.64, p = 0.031.** An actual reversal, not noise.
- **One NLI scorer: AUROC 0.90 on short-claim AttributedQA → 0.53 (chance) on long-form LFQA.** The standard NLI citation checker used by ALCE-style pipelines is **at chance on long-form answers**.
- **BERTScore was the best scorer on LFQA at 0.91 AUROC** — the "dumb" metric beat entailment models exactly where entailment should shine.
- Selecting a scorer by average cross-dataset performance: **mean regret 0.172 AUROC** under leave-one-dataset-out.
- LLM judges avoid total collapses but cost **~100×** and are non-deterministic.

⚠️ Single preprint, not peer-reviewed; 8 scorers is decent but not exhaustive. Still the strongest available evidence that **any single reported attribution-evaluator score is a property of the dataset, not the evaluator.**

**⭐ Citation quality in deployed deep-research agents.** "Cited but Not Verified: Parsing and Evaluating Source Attribution in LLM Deep Research Agents" — Hailey Onweller, Elias Lumer, Austin Huber, Pia Ramchandani, Vamse Kumar Subbiah, Corey Feld. arXiv:2605.06635, 7 May 2026.
- Frontier models: **>94% link accessibility, >80% topical relevance, but only 39–77% of citations are factually accurate against the source content.**
- **Fewer than half of open-source models can produce a cited report at all** one-shot.
- **Fact-check accuracy drops ~42% on average across two frontier models as tool calls scale from 2 to 150.** Longer research trajectories, monotonically worse grounding.

⚠️ Preprint; the fact-check judge is itself an LLM (so §5.1's transfer problem applies to this paper's own instrument); "two frontier models" is thin. But *links resolve, topics match, claims don't* is the most actionable attribution finding of 2026 — and the tool-call scaling result is directly relevant to agentic retrieval design.

**Does citation generation degrade answer quality? Yes, and granularity is the knob.** "Are Finer Citations Always Better? Rethinking Granularity for Attributed Generation" — Hexuan Wang, Jingyu Zhang, **Benjamin Van Durme, Daniel Khashabi** (JHU). arXiv:2604.01432, Apr 2026. **Forcing sentence-level citations degrades attribution quality by 16–276% relative to the optimal granularity**, across model sizes; quality **peaks at intermediate (paragraph-level) granularity**; **the penalty for fine-grained constraints grows with model scale**. At the right granularity you get both attribution and correctness. Proposed mechanism: attention dilution during synthesis plus atomic units fracturing the semantic context needed to synthesize.

### 5.2 Hallucination detection — the numbers

**Benchmarks.** *RAGTruth* (Cheng Niu, Yuanhao Wu, Juno Zhu, Siliang Xu, Kashun Shum, Randy Zhong, Juntong Song, Tong Zhang; **ACL 2024**, arXiv:2401.00396) — **~18,000 naturally generated RAG responses with word-level manual hallucination annotations** from 6 LLMs across QA / data-to-text / summarization. Still the de facto standard; ⚠️ generators are 2023-vintage, a real staleness concern for 2026. *HaluBench* (via Lynx) — 15,000 samples across HaluEval, DROP, CovidQA, PubMedQA, FinanceBench, RAGTruth; human validation agreement **0.90–0.96**; ⚠️ constructed partly by **semantic perturbation**, a distribution unlike naturally-occurring hallucination. *FaithBench* (**NAACL 2025 short**, Vectara-affiliated) — deliberately built from summaries **where SOTA detectors including GPT-4o-as-judge disagree**; **most SOTA detectors score near 50% (chance)**; ⚠️ adversarial by construction, so ~50% is partly definitional, and the team also ships a competing detector.

**⭐ TRIVIA+ / the 2026 benchmark-hygiene paper.** "Rethinking Evaluation for LLM Hallucination Detection: A Desiderata, A New RAG-based Benchmark, New Insights" — Wenbo Chen, Veena Padmanabhan, Tootiya Giyahchi, Elaine Wong, Leman Akoglu. arXiv:2605.11330, 11 May 2026. RAG-based with **"the longest context in the literature"** plus **four synthetic label-noise sets**. Three findings: current detectors have ample room on RAG tasks; **plain LLM-as-a-Judge is competitive with specialized detectors**; **label noise degrades detection and shifts detector rankings unpredictably**. ⚠️ **I could not extract the AUROC table** (PDF compressed streams) — qualitative findings only.

**Detector numbers I can stand behind.** *LettuceDetect* — Ádám Kovács, Gábor Recski (KR Labs / TU Wien). arXiv:2502.17125, 24 Feb 2025. ModernBERT-based, 8k context, token-level classification. RAGTruth example-level F1:

| System | F1 |
|---|---|
| GPT-4 (prompt-based) | 63.4% |
| Luna (Galileo encoder) | 65.4% |
| **LettuceDetect-large** | **79.22%** |
| Fine-tuned Llama-2-13B | 78.7% |
| **Fine-tuned Llama-3-8B (prior SOTA)** | **83.9%** |

**LettuceDetect is NOT SOTA on RAGTruth — fine-tuned Llama-3-8B at 83.9% is.** LettuceDetect is the best *cost-adjusted* detector: ~30× smaller, 30–60 examples/sec on one GPU, +14.8% relative over Luna. The paper itself says "competitive with." The widely-repeated "LettuceDetect is SOTA" framing is wrong.

*Lynx* — Selvan Sunitha Ravi, Bartosz Mielczarek, Anand Kannappan, **Douwe Kiela**, Rebecca Qian (Patronus AI). arXiv:2407.08488. HaluBench accuracy:

| Model | Overall | HaluEval | DROP | CovidQA | PubMedQA | FinanceBench | RAGTruth |
|---|---|---|---|---|---|---|---|
| **Lynx 70B** | **87.4%** | 88.4 | 86.4 | 97.5 | 90.4 | 81.4 | 80.2 |
| GPT-4o | 86.5% | 87.9 | 84.3 | 95.0 | 82.1 | **85.3** | **84.3** |
| Lynx 8B | 82.9% | 85.7 | 77.8 | 96.3 | 85.2 | 72.5 | 80.0 |
| Claude-3-Sonnet | 78.8% | 84.5 | 84.3 | 95.0 | 82.9 | 69.7 | 79.1 |
| RAGAS Faithfulness | 66.9% | — | — | — | — | — | — |

⚠️ **The marketing omits that the 87.4 vs 86.5 win is 0.9 points, and GPT-4o beats Lynx-70B on FinanceBench (85.3 vs 81.4) and RAGTruth (84.3 vs 80.2).** Lynx's margin comes almost entirely from PubMedQA and CovidQA — a benchmark-composition win. Vendor-authored, vendor-constructed benchmark, perturbation-synthesized hallucinations. "Lynx 2.0" is blog-announced with no paper found.

**Vectara HHEM leaderboard — vendor-run, flagged.** Fetched github.com/vectara/hallucination-leaderboard. **Last updated 11 May 2026; current scoring model HHEM-2.3** (not 2.1). Methodology: **>7,700 source articles, 50 to 24,000 words**, across news/tech/science/medicine/legal/sports/business/education; instruction "Summarize using only the information in the given passage. Do not infer"; **temperature 0**; reports hallucination rate, factual consistency, and **answer rate**. Top as of 11 May 2026: antgroup/finix_s1_32b **1.8%** (99.5% answer rate); openai/gpt-5.4-nano **3.1%** (100%); google/gemini-2.5-flash-lite **3.3%**; microsoft/Phi-4 **3.7%** but only **80.7% answer rate**; meta-llama/Llama-3.3-70B **4.1%**.

⚠️ Caveats that matter: it is **graded by the vendor's own product**, so it is simultaneously a benchmark and a product demo; it measures **summarization faithfulness only**; the **answer-rate column is the tell** (3.7% at 80.7% answer rate is not comparable to 3.1% at 100%); a 32B model from a payments company topping a board of frontier models is the pattern you'd expect from optimization-toward-the-metric. Vectara's own repo text: determining hallucinations is "impossible without a reference source," and the problem "is far from solved." ⚠️ Figures circulating in SEO summaries (GPT-5 Pro ~1.0%, Claude Opus 4.7 ~1.2%) **do not appear on the leaderboard I fetched — do not use them.**

**Semantic entropy and its 2026 pressure.** *SelfCheckGPT* (Manakul, Liusie, Gales; EMNLP 2023) is now used almost exclusively as a baseline; the standard criticism is that surface-token divergence conflates paraphrase with contradiction. *Semantic entropy* (Farquhar, Kossen, Kuhn, Gal, **Nature 2024**) clusters samples by bidirectional entailment into meaning classes. *Semantic Entropy Probes* (Kossen et al., arXiv:2406.15927) approximate it in a **single forward pass**.

2026 pressure (⚠️ all from an arXiv listing fetch, IDs as returned, abstracts not individually opened): arXiv:2607.16868 — **+7.1% AUROC over semantic entropy** via logical graphs, arguing entailment clustering is the wrong equivalence relation; arXiv:2606.10198 — **+5–20 points** via geometry; arXiv:2605.04295 — **0.88 vs 0.65 AUROC** on TriviaQA with adaptive conformal SE; **arXiv:2605.05166 "The First Token Knows"** — **first-token confidence matches or modestly exceeds semantic self-consistency with zero sampling overhead** (the sharpest efficiency critique); arXiv:2607.07670 — 5-sample SE reaches only **0.71–0.83 AUROC at 5× inference cost**, losing to activation probes; arXiv:2603.22812 — **~50% fewer samples** via variance-based early termination; arXiv:2606.24115 — clustering underperforms plain token statistics in a VLM medical domain.

**Honest synthesis:** nobody has published a clean refutation, and the *idea* (uncertainty over meanings, not tokens) is not seriously contested. What is contested is (a) whether the cost is justified vs single-pass probes, (b) whether bidirectional-entailment clustering is the right equivalence relation, (c) whether it transfers across domains. Also: **it is gray-box** — it needs token probabilities, unusable behind most commercial APIs.

**Probing / internal states.** *LLMs Know More Than They Show* — Hadas Orgad, Michael Toker, Zorik Gekhman, Roi Reichart, Idan Szpektor, Hadas Kotek, **Yonatan Belinkov**. **ICLR 2025**, arXiv:2410.02707. Four findings, and the third is the one people skip: (1) internal representations encode more truthfulness signal than thought; (2) the signal is **concentrated in specific tokens**; (3) **error detectors fail to generalize across datasets — truthfulness encoding is not universal but multifaceted**; (4) models can internally encode the right answer and still emit the wrong one. Counter-current: arXiv:2510.09033 argues the probe signal is largely **recall/familiarity, not truthfulness**. ⚠️ Active dispute; both cannot be right in their strong forms.

*ReDeEP* (Zhongxiang Sun, Xiaoxue Zang, Kai Zheng, Yang Song, Jun Xu, Xiao Zhang, Weijie Yu, Han Li; arXiv:2410.11414) — mechanistic claim that RAG hallucination occurs when **Knowledge FFNs over-weight parametric knowledge in the residual stream while Copying Heads fail to integrate retrieved context**. ⚠️ **The abstract carries no numbers**; a third-party 2026 paper (arXiv:2605.07209) reports **ReDeEP token-level AUC 0.73 on RAGTruth**, beating it by 7.4–10.3 points — treat as unverified secondary reporting.

**Which methods generalize? None of them, cleanly.** Three independent results converge: attribution scorers invert rankings across datasets (τ = −0.64) and one drops to chance; internal-state error detectors fail to generalize (Orgad et al., ICLR 2025); detector rankings shift unpredictably under label noise and plain LLM-as-a-Judge is competitive (TRIVIA+). **The practical implication is uniform: validate a detector on your own target distribution. Picking by published averages carries ~0.17 AUROC mean regret. And the honest baselines to beat are LLM-as-a-Judge and BERTScore, not SelfCheckGPT.**

### 5.3 Do verification passes measurably help?

**The negative camp.** *Large Language Models Cannot Self-Correct Reasoning Yet* — Jie Huang, Xinyun Chen, Swaroop Mishra, Huaixiu Steven Zheng, Adams Wei Yu, Xinying Song, **Denny Zhou** (Google DeepMind/Research). **ICLR 2024**, arXiv:2310.01798. **"LLMs struggle to self-correct their responses without external feedback, and at times, their performance even degrades after self-correction."** The methodological contribution is showing prior reported gains came from **oracle labels leaking in** (the loop was told when to stop) or from baselines not given equal compute.

*On the Self-Verification Limitations of LLMs on Reasoning and Planning Tasks* — Kaya Stechly, Karthik Valmeekam, **Subbarao Kambhampati**. **ICLR 2025**. ⚠️ Search returned arXiv:2402.08115; I could not verify that ID. Finding: LLM self-critique does not deliver iterative improvement; sound *external* verifiers do.

*SELF-[IN]CORRECT* (arXiv:2404.04298) gives the mechanism: models are no better at **discriminating** among their own candidates than at generating a good one. If generation and discrimination are equally strong, a self-critique loop has no information advantage.

**⭐ *Feedback Friction: LLMs Struggle to Fully Incorporate External Feedback*** — Dongwei Jiang, Alvin Zhang, Andrew Wang, Nicholas Andrews, **Daniel Khashabi**. arXiv:2506.11930, v2 21 Sep 2025. Uncomfortable for the "just add external feedback" fix: **even under near-perfect, complete external feedback, models plateau below the achievable ceiling** across math, knowledge, scientific, and multi-domain reasoning including Claude 3.7 with extended thinking. Progressive temperature sampling and explicit rejection of prior wrong answers **still fail**. **Model confidence, measured via semantic entropy, predicts feedback resistance** — high-confidence wrong answers are the ones that won't budge. ⚠️ No numeric ceilings in the abstract.

*Decomposing LLM Self-Correction: The Accuracy-Correction Paradox and Error Depth Hypothesis* (arXiv:2601.00828, Jan 2026) — **weaker models show 1.6× higher intrinsic correction rates than stronger models**. Strong models' remaining errors are *deep* errors self-correction structurally cannot reach, so raw "correction rate" is inversely correlated with base capability and is an actively misleading metric.

**The positive camp.** *CorrectBench* (Guiyao Tie et al., arXiv:2510.16062) — self-correction does improve accuracy, especially on complex reasoning. But three quiet negatives inside: **reasoning models (DeepSeek-R1) show limited additional gain at high time cost** (RL-trained reasoning already internalizes the loop); combining strategies helps at an efficiency cost; and **plain CoT is competitive on accuracy *and* efficiency.**

**⭐ The strongest RAG-specific positive, and it credits the verifier not the reflection.** *Self-Correcting RAG (NLI-guided MCTS)* — Shijia Xu, Zhou Wu, Xiaolong Jia, Yu Wang, Kai Liu, April Xiaowen Dong. arXiv:2604.10734, 12 Apr 2026:

| Metric | Standard RAG | Self-Correcting RAG |
|---|---|---|
| Attribution Precision | 0.52 | **0.85** |
| Contradiction Rate | 0.15 | **0.04** |
| Supportability | 0.65 | **0.88** |
| EM (avg, 6 datasets) | 25.8% | **37.1%** |
| F1 (avg, 6 datasets) | 36.1% | **45.8%** |

**The ablation is load-bearing: MMKP context selection alone gives EM 34.5% but attribution precision only 0.58. NLI-guided MCTS alone gives attribution precision 0.82 at EM 31.2%.** Better context alone barely moves faithfulness; **the verifier does essentially all the faithfulness work.** Datasets: NQ, PopQA, MuSiQue, 2Wiki, HotpotQA (1,000 queries each) + MultiHop-RAG (2,556). ⚠️ Preprint; "Attribution Precision" is itself an automatic scorer, which §5.1 says may not transfer.

*CRAG* (Yan et al., **ICLR 2024**, arXiv:2401.15884) — lightweight T5 retrieval evaluator triggering correct/ambiguous/incorrect actions with web-search fallback. **2026 reproduction** (arXiv:2603.16169) finds two things: (a) the original relied on the Google Search API and closed weights; swapping in Wikipedia API + Phi-3-mini **reproduces comparable performance**; (b) **SHAP analysis shows CRAG's T5 evaluator primarily keys on named-entity alignment, not semantic similarity** — the "retrieval quality evaluator" is substantially a lexical entity-overlap detector. A meaningful deflation even though the numbers replicate. *Self-RAG* (Asai et al., ICLR 2024) trains the critique in via reflection tokens; ⚠️ **I found no rigorous 2025–26 independent reproduction.**

**→ THE HONEST NET VERDICT.** The apparent conflict dissolves along one axis:
1. **Intrinsic self-correction — no external signal, no oracle stopping — does not reliably help and sometimes hurts.** Huang et al. (ICLR 2024) canonical; Stechly/Kambhampati extends to planning; SELF-[IN]CORRECT gives the mechanism; the 2026 error-depth paper shows the apparent gains shrink as base models strengthen. **Nothing in 2025–26 overturns this.**
2. **Verification against an external, sound signal does help substantially in RAG.** The NLI-MCTS ablation is the cleanest demonstration: verifier alone moves attribution precision 0.52 → 0.82; better retrieval alone moves it 0.52 → 0.58. **The gain is in the verifier, not in the reflection.**
3. **But external feedback is not a solved fix.** Feedback Friction: models resist even near-perfect feedback, and semantic-entropy confidence predicts which errors are unfixable.
4. **If you are running a modern reasoning model, a bolted-on critique pass is likely a latency tax** (CorrectBench).
5. **Verification-loop gains are hard to measure honestly**, because the faithfulness metrics used to score them are the same ones §5.1 showed don't transfer.

**Practical distillation: add a verification pass only when it consumes something the generator did not already see** — retrieved evidence checked by an independent NLI model, a search result, an executor, a sound checker. A pass that re-reads the model's own output and asks "are you sure?" is a latency tax.

### 5.4 Groundedness benchmarks in 2026

**FACTS Benchmark Suite (Google DeepMind + Kaggle), released 9 Dec 2025.** Supersedes the standalone FACTS Grounding leaderboard by absorbing it. Four pillars: Parametric (2,104), Search (1,884, standardized web-search tool, often multi-hop), Multimodal (1,522), Grounding v2. **3,513 examples publicly released** (⚠️ the four listed sizes exceed this — presumably the public split only; I could not resolve the inconsistency). **Kaggle owns the private held-out sets, runs the evaluations, and hosts the leaderboard** — the right structural answer to contamination and the main reason to cite FACTS in 2026.

**Results: 15 leading models tested. Gemini 3 Pro leads at 68.8% FACTS Score. Every model — Claude, GPT, Llama included — scores below 70%.**

⚠️ **Google builds the benchmark and Google's model tops it.** Kaggle's custody of the private sets mitigates contamination but not design bias. The earlier FACTS Grounding used an **ensemble of three frontier judges** (Gemini 1.5 Pro, GPT-4o, Claude 3.5 Sonnet) to dilute single-judge bias; the blog does not state whether the Suite keeps that ensemble. ⚠️ Older grounding-only figures circulating (Gemini 2.0 Flash Exp 83.6%, Claude 3.5 Sonnet 79.4%, GPT-4o 78.8%) come from a vendor blog, not DeepMind, and I could not confirm them against Kaggle (page rendered empty). Note the ~80% grounding-only and ~69% composite are **different scales measuring different things** — not a trend.

⚠️ 2026 entrants surfaced but not fetched: **LayerRAG-Bench** (arXiv:2607.27353) — argues groundedness-only evaluation produces **false positives** when answers look grounded but fail at the evidence, tool-contract, authorization, or session-state layer (the right critique for agentic RAG); **From Binary Groundedness to Support Relations** (Sarkar, Poelitz, Kewenig, **Microsoft Research**, arXiv:2604.08082) — argues the binary supported/unsupported framing is wrong.

**Where frontier models actually sit, in one paragraph.** On the most rigorous contamination-controlled composite (FACTS Suite, private held-out sets): **no model exceeds 70%; the leader is 68.8%.** On grounding-in-provided-documents, the best models sit around 80–84% (older leaderboard). On summarization faithfulness with an explicit "do not infer" instruction at temp 0, the best hallucinate on **~2–4%** of summaries (vendor-scored). On RAG QA faithfulness, error rates are several times higher. **On citation-level factual support in deep research agents, only 39–77% of citations actually support their claim, degrading ~42% as tool calls scale from 2 to 150.** The spread between those numbers *is* the story: the more constrained the grounding task, the better models look, and **the number that matters most for production agentic RAG is the worst one**.

---

## 6. GRAPHRAG — DID THE EVIDENCE HOLD UP?

### 6.1 The origin paper never got a venue

"From Local to Global: A Graph RAG Approach to Query-Focused Summarization" — Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, Jonathan Larson (**Microsoft Research**). arXiv:2404.16130, v1 24 Apr 2024, v2 19 Feb 2025. **1,875 citations.**

Two corpora — podcast transcripts (~1M tokens, 8,564 entities, 20,691 edges) and news (~1.7M tokens, 15,754 entities, 19,520 edges). **72–83% comprehensiveness win rates and 62–82% diversity win rates vs naive vector RAG**, judged pairwise by an LLM.

⚠️ **I could not verify any peer-reviewed venue.** No journal-ref or venue comment on arXiv through v2 (Feb 2025). As best I can establish, **the single most influential GraphRAG paper remained an arXiv preprint / MSR tech report and was never published at ACL/EMNLP/NeurIPS/ICLR.** The field built on an unrefereed baseline. And the numbers are **LLM-as-judge pairwise preferences on comprehensiveness and diversity** — not accuracy, not F1, not human evaluation — over **LLM-generated evaluation questions**, with no cost figures.

### 6.2 The successor line

| System | Paper | Venue | Headline |
|---|---|---|---|
| **RAPTOR** | Sarthi, Abdullah, Tuli, Khanna, Goldie, **Manning** (Stanford), arXiv:2401.18059 | **ICLR 2024** | **+20% on QuALITY** with GPT-4; +2% over DPR, +5.1% over BM25 |
| **LightRAG** | Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, Chao Huang (HKU), arXiv:2410.05779 | **EMNLP 2025** | Dual-level graph retrieval; lower cost + faster incremental update than GraphRAG |
| **HippoRAG 2** | Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi, Sizhe Zhou, Yu Su (Ohio State), arXiv:2502.14802 | **ICML 2025** | **+7% associative memory** over SOTA embeddings. **The abstract itself concedes prior graph methods' "performance on more basic factual memory tasks drops considerably below standard RAG"** |
| **GraphReader** | arXiv:2406.14550 | **Findings EMNLP 2024** | 4k-context GraphReader beats GPT-4-128k on LV-Eval across 16k–256k |
| **Think-on-Graph 2.0** | arXiv:2407.10805 | **ICLR 2025** | +5.51% over CoK on HotpotQA |
| **KAG** | Ant Group / OpenSPG, arXiv:2409.13731 | **ACM Web Conf 2025** | Schema-constrained KG to fix OpenIE noise in HippoRAG/GraphRAG |
| **PathRAG** | arXiv:2502.14902 | ⚠️ venue unverified | Flow-based path pruning. Key framing: **the problem with GraphRAG is redundancy, not insufficiency** |
| **Youtu-GraphRAG** | Junnan Dong, Siyu An, … Xiao Huang, Yunsheng Wu, Di Yin, Xing Sun (Tencent Youtu + Monash + HK PolyU), arXiv:2508.19855 | ⚠️ GitHub says ICLR 2026; arXiv comments do not | **+16.62% accuracy, −90.71% token cost** in graph construction |
| **KET-RAG** | arXiv:2502.09304 | **KDD 2025** | **20% indexing cost reduction** |
| **LazyGraphRAG** | Microsoft Research, 25 Nov 2024 | ⚠️ **blog, not a paper** | Indexing cost **identical to vector RAG, 0.1% of full GraphRAG**; comparable global-query quality at **>700× lower query cost** |

That last row is the tell. **You don't build LazyGraphRAG unless the original was unaffordable.**

### 6.3 The skeptical line — this is the substantive part

**⭐ RAG vs. GraphRAG: A Systematic Evaluation and Key Insights** — Haoyu Han, Li Ma, Yu Wang, Harry Shomer, Yongjia Lei, Zhisheng Qi, Kai Guo, Zhigang Hua, Bo Long, Hui Liu, **Charu C. Aggarwal**, Jiliang Tang (**Michigan State + Meta + IBM** — *not* NVIDIA; I found no NVIDIA study of this name). arXiv:2502.11371, v1 17 Feb 2025, **v3 4 Mar 2026.**

- **Natural Questions (single-hop):** RAG **64.78 F1** > Community-GraphRAG(Local) 63.01 > HippoRAG2 61.03. **Plain RAG wins.**
- **HotpotQA:** HippoRAG2 63.01 > Community-GraphRAG(Local) 61.66 > RAG 60.04.
- **MultiHop-RAG (accuracy):** HippoRAG2 **70.27** > Community-GraphRAG(Local) 69.01 > RAG 67.02. **The multi-hop win is ~+3 points, not a category change.**
- **Complementarity: 13.6% of MultiHop-RAG queries are GraphRAG-only wins, 11.6% are RAG-only wins** — nearly symmetric.
- **Summarization (SQuALITY, ROUGE-2 F1): RAG 10.08, Community-GraphRAG(Local) 10.10, Community-GraphRAG(Global) 6.99.** The global/community mode — Microsoft GraphRAG's entire selling point — **loses badly on a real summarization metric.**
- **Cost (MultiHop-RAG):** construction time RAG **135s** vs KG-GraphRAG **7,702s** vs Community-GraphRAG **5,560s** — a **41–57× indexing penalty**. Retrieval latency 1,724s / 14,434s / 1,249s. Storage 127 / 117 / 165 MB.
- Hybrid integration: **+6.4%** on MultiHop-RAG with Llama-3.1-70B.
- **Methodological warning: "position bias is clearly present in LLM-as-a-Judge evaluations"** — directly undercutting Edge et al.'s win-rate methodology.

**⭐ When to use Graphs in RAG / GraphRAG-Bench** — Zhishang Xiang, Chuanjie Wu, Qinggang Zhang, Shengyuan Chen, Zijin Hong, Xiao Huang, Jinsong Su. arXiv:2506.05690, v1 6 Jun 2025, **v3 22 Feb 2026**. ⚠️ GitHub labels it ICLR 2026; unverified from arXiv. The abstract opens with the negative framing: *"recent studies report that GraphRAG frequently underperforms vanilla RAG on many real-world tasks."*

Across 7 frameworks (MS-GraphRAG, HippoRAG, HippoRAG2, LightRAG, Fast-GraphRAG, RAPTOR, Lazy-GraphRAG):
- **Level 1 fact retrieval, Novel:** Basic RAG w/ rerank **60.92%** vs best GraphRAG (HippoRAG2) 60.14% — *graph loses.*
- **Level 2 complex reasoning, Novel:** Basic RAG 42.93% → HippoRAG2 **53.38% (+24.4% relative)** — the clearest genuine graph win.
- **Level 3 contextual summarization, Novel:** Basic RAG 51.30% → MS-GraphRAG **64.40%**.
- **Evidence recall inverts:** on fact retrieval, Basic RAG recall **83.21%** vs HippoRAG2 70.29%. Graph retrieval is *worse at finding the right evidence* for simple questions.
- **The cost table is the headline — average prompt tokens per query:**

| Method | Novel | Medical |
|---|---|---|
| Vanilla RAG | **879** | **954** |
| HippoRAG2 | 1,008 | 1,020 |
| RAPTOR | 3,441 | 3,510 |
| HippoRAG | 7,208 | 7,342 |
| LightRAG | 100,832 | 100,310 |
| **MS-GraphRAG (global)** | **331,375** | 332,881 |

**Microsoft GraphRAG global search burns ~377× the prompt tokens of vanilla RAG per query** — inference cost, on top of indexing cost. Conclusion: graph "introduces redundant information, which in turn degrades context relevance."

Also: **GraphRAG-Bench dataset paper** (Yilin Xiao, Junnan Dong, Chuang Zhou, Su Dong, Qian-wen Zhang, Di Yin, Xing Sun, Xiao Huang; arXiv:2506.02404) — college-level questions across 16 disciplines / 20 textbooks, 9 methods, scoring reasoning coherence not just final answers.

**⭐ Do We Still Need GraphRAG? Benchmarking RAG and GraphRAG for Agentic Search Systems** — Dongzhe Fan, Zheyi Xue, Siyuan Liu, Qiaoyu Tan. arXiv:2604.09666, **1 Apr 2026.** The 2026 reframing: does agentic multi-round retrieval make explicit graphs redundant? Standardized LLM backbone, retrieval budget, inference protocol, full test sets.
- **Single-shot:** general QA — graph gives only **+0.47 avg**. Multi-hop — graph gives **+27.23 avg** (Contain-EM).
- **With training-free agentic search:** dense RAG + GraphSearch **narrows the multi-hop gap by 32.3%**.
- **With GRPO agentic search:** dense RAG becomes best on Natural Questions; graph backends still lead multi-hop.
- Verdict: agentic search "substantially improves dense RAG and narrows the gap... Nevertheless, GraphRAG remains advantageous for complex multi-hop reasoning... **when its offline cost is amortized.**"

**⭐ The harshest result: BM25 Wins at Scale: A Scaling Study of RAG Paradigms** — Pengyu Wang, Benfeng Xu, Shaohan Wang, Mingxuan Du, Xin Zeng, Huarui Wu, Lei Zhang, Licheng Zhang. arXiv:2607.26497, **29 Jul 2026** (v3 31 Jul). 28 strictly nested corpus tiers spanning ~450× expansion, fixed questions and reference docs.
- A file-system agent leads at small scale but **costs 39× more query tokens** at the largest tier.
- **BM25 overtakes it around 10M corpus tokens and leads at every larger shared tier, with a margin approaching 20 points at full scale.**
- **Graph-based RAG "encounters construction walls before deployment scale, and its scalable variants remain below BM25 at shared tiers."** At real corpus sizes GraphRAG cannot even be built, and the cheap variants lose to lexical search from 1994.
- Conclusion: "lexical retrieval is the strongest scalable default, while agentic reasoning works best **after** ranked discovery rather than in place of it."

⚠️ I searched for a paper titled "GraphRAG under fire" and **found no such academic paper.** Not asserting it exists.

### 6.4 Honest 2026 verdict

**The evidence did not stall — it substantially deflated and re-scoped.**

1. **Single-hop / factual lookup: graph loses.** 64.78 vs 63.01 F1 (NQ); 60.92% vs 60.14% (GraphRAG-Bench). Consistent across independent groups. The HippoRAG 2 authors concede it in their own abstract.
2. **Multi-hop: graph genuinely wins, but the magnitude is wildly contested — +3 points (controlled same-backbone, MultiHop-RAG) to +24% relative (GraphRAG-Bench) to +27 EM (RAGSearch single-shot).** The spread *is* the finding: it depends almost entirely on whether the vector baseline is well-tuned and whether the questions are natively bridge-entity questions.
3. **Global/sensemaking summarization: the original claim is the least replicated.** Edge et al. reported 72–83% comprehensiveness win rates on LLM-judge preference; the controlled ROUGE-2 replication gives Community-GraphRAG Global **6.99 vs RAG 10.08** — global search *loses*. GraphRAG-Bench does find graph wins on Level-3 summarization. Verdict: **graph helps on breadth/coverage summarization, but the specific "global search over community summaries" mechanism is not well supported, and the original evidence rested on an LLM-judge protocol later shown to have position bias.**
4. **Cost is the decisive variable, not accuracy.** Indexing 41–57× (135s → 5,560–7,702s); inference **331k prompt tokens/query vs 879** (~377×). Microsoft's own LazyGraphRAG (0.1% indexing, >700× lower query cost), Youtu's −90.71% construction tokens, and KET-RAG's −20% are the same admission from three different groups.
5. **The 2026 threat isn't better vector RAG — it's agentic search and BM25.** Agentic retrieval closes 32.3% of the multi-hop gap with no graph; at deployment scale, graph can't be constructed at all.
6. **The rule the literature converges on:** build a graph only when (a) queries are genuinely multi-hop over bridge entities, (b) the corpus is small/static enough to amortize indexing, and (c) you have already lost to a hybrid BM25+dense+reranker baseline. The 13.6%/11.6% complementarity says the right answer is **hybrid routing**, not graph-everything.

---

## 7. MULTI-TURN / MEMORY-AUGMENTED RETRIEVAL AGENTS

### 7.1 The load-bearing context paper

**LLMs Get Lost In Multi-Turn Conversation** — Philippe Laban, Hiroaki Hayashi, Yingbo Zhou, Jennifer Neville (**Microsoft Research + Salesforce Research**). arXiv:2505.06120, 9 May 2025. **378 citations.**

- **Average −39% performance across six generation tasks**, single-turn → multi-turn, for *every* top open- and closed-weight LLM tested. **200,000+ simulated conversations.**
- **The decomposition is the important part: the drop is a minor loss in aptitude and a large increase in unreliability.** Models aren't dumber multi-turn — they're erratic.
- Mechanism: LLMs "make assumptions in early turns and prematurely attempt to generate final solutions, on which they overly rely."

**Why this matters for memory systems:** multi-turn failure is substantially a *generation-side* pathology, not purely a retrieval/recall pathology. A memory layer that only fixes recall addresses part of the problem. Papers claiming large multi-turn wins from memory alone should be read against this.

### 7.2 Benchmarks

**LongMemEval** — Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, Dong Yu (UCLA / Tencent AI Lab / UCSD). arXiv:2410.10813, **ICLR 2025** (confirmed in arXiv comments + proceedings). 500 curated questions, freely scalable histories; five abilities including **knowledge updates and abstention**. **30% accuracy drop** for commercial assistants and long-context LLMs on sustained interaction.

**LongMemEval-V2** — Di Wu, Zixiang Ji, Asmi Kawatkar, Bryan Kwan, Jia-Chen Gu, Nanyun Peng, Kai-Wei Chang. arXiv:2605.12493, **12 May 2026**. 451 questions, up to **500 trajectories / 115M tokens**. **AgentRunbook-C 72.5%; AgentRunbook-R (RAG-based memory) 48.5%; off-the-shelf coding agent baseline 69.3%.** Note: **the RAG-memory approach loses to no memory system at all by 21 points.**

**LoCoMo** — Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, Yuwei Fang. **ACL 2024**, arXiv:2402.17753. ~600 turns / 16K tokens avg over up to 32 sessions. **This is the benchmark the entire industry memory-scoring war is fought on, and it is small and partly defective — see §7.4.**

**MemoryAgentBench** — Yuanzhe Hu, Yu Wang, Julian McAuley (UCSD). arXiv:2507.05257, latest 28 Jun 2026. Four competencies: accurate retrieval, test-time learning, long-range understanding, **conflict resolution / selective forgetting**. "Current methods fall short of mastering all four." Memory agents on GPT-4o reach only **~60% on single-hop conflict resolution.**

**MTRAG** — Yannis Katsis, Sara Rosenthal, Kshitij Fadnis, Chulaka Gunasekara, Young-Suk Lee, Lucian Popa, Vraj Shah, Huaiyu Zhu, Danish Contractor, Marina Danilevsky (**IBM Research**). arXiv:2501.03468. ⚠️ IBM's publications page lists it as **TACL**; arXiv v1 has no venue comment. **110 fully human-generated conversations, avg 7.7 turns, 4 domains, 842 tasks.** SOTA RAG systems struggle specifically on **later turns, unanswerable questions, and non-standalone questions**. Now the basis of **SemEval-2026 Task 8 (MTRAGEval)**.

**HELMET** (**ICLR 2025**) covers multi-turn categories; key relevant finding: **synthetic tasks like NIAH do not reliably predict downstream performance**, and category scores correlate poorly with each other.

**⭐ Memora: From Recall to Forgetting** — Md Nayem Uddin, Kumar Shubham, Eduardo Blanco, Chitta Baral, Gengyu Wang. arXiv:2604.20006, **21 Apr 2026**. Weeks-to-months conversations; introduces **FAMA (Forgetting-Aware Memory Accuracy)**, penalizing reliance on obsolete or invalidated memory. **Evaluating 4 LLMs and 6 memory agents: "frequent reuse of invalid memories and failures to reconcile evolving memories. Memory agents offer marginal improvements."** The strongest 2026 negative result in this space.

Also *PerLTQA* (arXiv:2402.16288, 8,593 questions / 30 characters, semantic + episodic memory).

### 7.3 Systems

| System | Paper | Venue | Numbers |
|---|---|---|---|
| **MemGPT / Letta** | Packer et al., arXiv:2310.08560 | preprint | OS-style paged memory; established DMR as its eval |
| **Zep / Graphiti** | Preston Rasmussen, Pavlo Paliychuk, Travis Beauvais, Jack Ryan, Daniel Chalef. arXiv:2501.13956, 20 Jan 2025 | preprint, no venue comment | **DMR 94.8% vs MemGPT 93.4%**; LongMemEval **up to +18.5%**, **−90% latency** |
| **Mem0** | Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, Deshraj Yadav. arXiv:2504.19413, 28 Apr 2025 | ⚠️ arXiv shows no venue; secondary sources say ECAI 2025 (unverified) | **+26% relative LLM-as-Judge over OpenAI memory**; graph variant +~2%; **−91% p95 latency**, **>90% token cost saving** vs full-context |
| **MemoRAG** | arXiv:2409.05591 | — | Light long-range model builds global memory + draft clues; expensive model answers |
| **Cognitive Workspace** | arXiv:2508.13171 | preprint | **58.6% avg memory reuse vs 0% for traditional RAG**; 17–18% net efficiency gain despite 3.3× operations; p<0.001, Cohen's d>23. ⚠️ **Effect sizes that large are a red flag for a self-defined metric, and "0% for RAG" is true by construction** |
| **Memory-R1** | arXiv:2508.19828 | preprint | RL (PPO/GRPO) over ADD/UPDATE/DELETE/NOOP memory ops. On LLaMA-3.1-8B: **+48% F1, +69% BLEU-1, +37% LLM-as-Judge**, trained on only **152 QA pairs**. Beats Mem0, LangMem, A-MEM |

### 7.4 ⚠️ CONTESTED — the Mem0 vs Zep benchmark dispute

A real, documented, unresolved public dispute. **Provenance warning: the underlying claims are in arXiv papers; the rebuttals live in blog posts and a GitHub issue, not in peer review.**

1. **Zep (arXiv:2501.13956, Jan 2025)** claims DMR 94.8% vs MemGPT 93.4% and LongMemEval +18.5% / −90% latency; separately publicized **~84% on LoCoMo**.
2. **Mem0 (arXiv:2504.19413, Apr 2025)** benchmarks 10 approaches on LoCoMo: **Mem0 at 67.13% LLM-as-Judge, +26% relative over OpenAI memory, and Zep at 65.99%.**
3. **Mem0 → Zep, GitHub issue getzep/zep-papers#5, filed 8 May 2025 by Deshraj Yadav (Mem0 CTO):** alleges Zep's 84% is invalid because Zep **included questions from LoCoMo's excluded adversarial 5th category in the numerator while excluding them from the denominator**; also alleges Zep modified the system prompt with timestamp-favoring instructions not given to baselines, changed the retrieval template vs its own prior DMR work, and reported a **single run** rather than Mem0's 10-run-with-variance standard. **Mem0's re-run of Zep: 58.44% ± 0.20 — a 25.56-point reduction.** ⚠️ As of the fetch, the issue was open with no Zep response in-thread.
4. **Zep → Mem0, blog "Lies, Damn Lies, and Statistics":** alleges Mem0 misconfigured Zep three ways — assigned the **user role to both participants** in a graph designed for single user-assistant interaction; passed **timestamps appended to message text instead of Zep's dedicated `created_at` field**, breaking temporal reasoning; ran **searches sequentially rather than in parallel**, inflating latency. Zep's corrected numbers: **75.14% ± 0.17** (vs Mem0's reported 65.99%), p95 search latency **0.632s** vs Mem0's reported 0.778s.
5. **Zep also attacks the benchmark:** LoCoMo conversations are only **16k–26k tokens**, contain **no knowledge-update tests**, and have data-quality defects — missing ground truths, multimodal errors, incorrect speaker attribution, ambiguous questions.

**Net state of the Zep-on-LoCoMo number: 84% (Zep original) → 65.99% (Mem0's measurement) → 75.14% (Zep's correction) → 58.44% (Mem0's re-run of Zep's own pipeline). A ~26-point spread on the same system and same benchmark, with no neutral adjudication.**

**Honest reading: both parties are vendors self-reporting on a 2024 academic benchmark that both agree is partly broken. No corrected figure has been independently replicated in a refereed venue. Treat all LoCoMo numbers from memory vendors as unreliable.** The academic side has moved on — MemoryAgentBench, LongMemEval-V2, and Memora were all built partly because LoCoMo is inadequate.

**⭐ Independent counter-evidence.** "Beyond the Context Window: A Cost-Performance Analysis of Fact-Based Memory vs. Long-Context LLMs for Persistent Agents" — Natchanon Pollertlam, Witchayut Kornsuwannawit. arXiv:2603.04814, **5 Mar 2026**. Mem0-based fact memory vs long-context GPT-5-mini on LongMemEval, LoCoMo, PersonaMemv2.
- **Long-context GPT-5-mini achieves *higher* factual recall on both LongMemEval and LoCoMo.** Memory is only competitive on PersonaMemv2.
- The real finding is the cost model: long-context cost grows per-turn even under prompt caching; memory read cost is roughly fixed after a one-time write. **At 100k context, memory becomes cheaper after ~10 turns**, break-even falling as context grows.
- **Interpretation: as of 2026, memory systems are an economic argument, not an accuracy argument** — directly contradicting Mem0's paper framing.

⚠️ **Two false attributions circulating in blog aggregators, corrected:** (a) "Letta scored 49.0% on LongMemEval in independent evaluation (arXiv 2603.04814)" — **that paper does not evaluate Letta or Zep at all**, only Mem0 vs long-context GPT-5-mini. Do not use it. (b) "Mem0's new algorithm hits 93.4% on LongMemEval vs Zep 63.8%" traces to **Mem0's own 2026 marketing blog**, not a paper.

### 7.5 Query rewriting vs end-to-end retrieval in multi-turn

The field splits into **conversational query rewriting (CQR)** — produce a standalone de-contextualized query — and **conversational dense retrieval (CDR)** — encode the whole session end-to-end.

**TREC iKAT** 2024 and 2025. iKAT 2025 has a **SIGIR 2026 resource paper** (ACM DOI 10.1145/3805712.3808591) and a NIST track overview; the 2025 edition moved to a **live API where systems must rewrite, retrieve, and ground in real time per turn**. ⚠️ **Methodological caveat from the track itself: manual runs use the human rewrite as input, which advantages them and leaves assessment less complete for automatic reformulations — so CQR-vs-ceiling comparisons are biased.** Participant papers: RALI@TREC iKAT 2024 (arXiv:2412.07998), CFDA & CLIP @ TREC iKAT 2025 (arXiv:2509.15588), Adaptive Personalized Conversational IR (arXiv:2508.08634).

**⭐ The clearest 2026 evidence comes from SemEval-2026 Task 8 (MTRAGEval), built on IBM's MTRAG.**
- **Sifei @ SemEval-2026 Task 8** (arXiv:2606.28352): training-free hybrid dense+sparse retrieval with **controlled query rewriting** + cross-encoder reranking → **0.5453 nDCG@5, 3rd of 38 teams**, vs strongest baseline 0.4795.
- **The most useful negative finding: controlled conversational rewriting combined with last-turn concatenation gives consistent gains across domains, while retrieval-oriented rewrites — keyword lists, hypothetical-document expansion — consistently HURT**, by distorting intent and over-amplifying rare terms.
- Others: uva-irlab-conv (arXiv:2606.11945 — learned sparse + listwise reranking, five complementary LLM reformulations fused via variance-aware nested RRF); H-RAG (arXiv:2605.00631); AILS-NTUA (arXiv:2603.10524). Earlier: arXiv:2406.18960.

**Verdict: as of 2026, CQR has not been displaced by end-to-end conversational dense retrieval — every competitive MTRAGEval system rewrites. But the *type* of rewrite matters and the naive "make it retrieval-friendly" instinct is empirically wrong: decontextualize and preserve intent, keep the raw last turn, don't keyword-ify.**

### 7.6 Honest 2026 verdict

1. **The problem is real and large:** −39% single→multi-turn (200k+ sims), −30% on LongMemEval, MTRAG failure concentrated on later/non-standalone/unanswerable turns.
2. **The cause is not purely retrieval.** Laban et al.'s decomposition — unreliability, not aptitude — means a memory layer can only fix part of it.
3. **Vendor memory-system accuracy claims are not trustworthy right now.** A 26-point swing on one system on one benchmark with no neutral referee, and both sides agree the benchmark is inadequate.
4. **Independent 2026 work is unkind to memory systems.** Memora: 6 agents, "marginal improvements," frequent reuse of invalidated memory. LongMemEval-V2: RAG-based memory 48.5% vs a plain coding agent at 69.3%. arXiv:2603.04814: long-context GPT-5-mini beats Mem0 on factual recall on both LongMemEval and LoCoMo.
5. **The surviving case for memory is economic, not qualitative:** fixed per-turn read cost vs context cost growing even under caching; break-even ~10 turns at 100k context.
6. **The unsolved competency is not recall — it's update and forgetting.** ~60% on single-hop conflict resolution with GPT-4o; Memora's whole FAMA metric exists because systems keep citing superseded facts; LongMemEval flagged knowledge-updates and abstention in Oct 2024 and they remain the failure modes in 2026.
7. **Most promising direction with real numbers: learned memory management.** Memory-R1 (+48% F1 / +69% BLEU-1 / +37% judge on LLaMA-3.1-8B with 152 training pairs) suggests the ADD/UPDATE/DELETE policy is learnable and is where the headroom is — consistent with (6).

---

## 8. CROSS-CUTTING OBSERVATIONS AND THE FULL LIST OF OPEN DEBATES

### 8.1 The recurring arc

Every one of the seven areas shows the same shape. **A 2024 preprint from a named lab makes a large claim on an LLM-as-judge or vendor-internal evaluation** (Edge et al.'s 72–83% GraphRAG win rates; Anthropic's 67% failure reduction; Jina's late chunking; Zep/Mem0's LoCoMo scores; Search-R1's +41%). **A successor wave builds on it.** Then **2025–2026 controlled, same-backbone, full-test-set replications with standardized budgets shrink the effect to a few points, relocate it to a narrow query class, and reveal the cost was the real story.**

**In every area, the strongest 2026 papers are benchmark, reproducibility, and cost papers — not method papers.** BrowseComp-Plus (ACL 2026), Lighting the Way for BRIGHT (SIGIR 2026 Repro), the PLAID reproduction (SIGIR 2024 Repro), Fixed RAG Compression Collapses Measured Reader Scaling, Do LLM Attribution Metrics Transfer, RAG vs GraphRAG, BM25 Wins at Scale, Memora, and the MTEB RTEB governance decision. If you read only ten things from this report, read those.

### 8.2 A second recurring pattern: the "obvious baseline was omitted"

This charge appears independently in at least six places and is the most reliable predictor of a deflated result:
- PLAID papers omitted BM25 + ColBERT reranking (MacAvaney & Tonellotto).
- BRIGHT baselines silently used an undocumented BM25 variant (Sharifymoghaddam, Ge, Lin).
- Long-context-beats-RAG papers used relevance-ordered, badly-chunked RAG arms (DOS RAG / OP-RAG).
- Self-correction papers leaked oracle stopping signals and under-resourced their baselines (Huang et al., ICLR 2024).
- Compression papers evaluated on weak readers (Panthi & Abdelfattah).
- Chunking papers baselined against fixed-*character* splitting rather than sentence-aware splitting.

### 8.3 The complete open-debate list

1. **Is RL necessary for agentic search, or is it trajectory-data quality?** LiteResearcher measures +15.7 GAIA from RL over its own SFT; OpenSeeker-v2 beats it on BrowseComp/HLE with **pure SFT** on 10.6k curated trajectories. No controlled experiment at matched scale exists.
2. **Do RL-trained search agents generalize?** BrowseComp-Plus: Search-R1-32B scores **exactly its untrained base model's score** on an out-of-distribution corpus. Unrebutted.
3. **Reasoning-aware retrieval: training problem or test-time-compute problem?** LATTICE (Google) matches the best fine-tuned ensembles with an off-the-shelf LLM. Never evaluated against the training camp at matched compute.
4. **Is the single-vector dimensional bound practically binding?** Weller et al. (DeepMind, ICLR'26) yes; Bangachev et al. (MIT) the bound is far looser; Spectral Retrieval lifts LIMIT-small R@10 0.33→0.90 without retraining.
5. **Does late interaction earn its cost?** SIGIR'24 reproduction says BM25+ColBERT rerank at 9 ms/q beats the fastest PLAID at 73 ms/q. Not directly rebutted with an updated head-to-head.
6. **Are late interaction and learned sparse the same thing?** ColBERTSaR proves quantized ColBERT ≡ learned sparse; PLAID clusters align 1:1 with tokens. Emerging synthesis, not consensus.
7. **RRF vs convex combination.** Bruch et al. (TOIS 2023) showed CC wins; every 2025–26 applied paper uses RRF anyway; nobody has re-tested.
8. **Does BM25 still beat dense?** Yes on LIMIT and financial tables and at low latency; no by ~15 points on aggregate suites and by ~14 points in agentic loops. Task-conditional.
9. **How much does hybrid add?** +9.17 nDCG (arXiv:2502.20245) vs +3.1pp (KohakuRAG). Unexplained, unreplicated.
10. **How much does chunking strategy matter?** Resolves once you notice the disagreeing papers use different worst-case baselines (fixed-character vs fixed-token).
11. **Does contextual retrieval work as advertised?** No independent reproduction of 35/49/67 exists.
12. **Are compression numbers real?** Reader-strength confound flips 31% of model rankings and hides 80% of a reader upgrade.
13. **LC vs RAG.** Reconciles as "most LC-wins papers used a weak RAG arm," but LC has its own length-alone degradation problem.
14. **Is "lost in the middle" still true?** Largely trained away for simple retrieval in frontier models; order-sensitivity for composition over many chunks persists.
15. **Does self-verification help?** Resolved along the intrinsic/external axis, but two live sub-disputes: whether external feedback is sufficient (Feedback Friction says no), and whether reported self-correction gains survive controlling for base capability (error-depth says no).
16. **Do internal states encode truthfulness or just recall?** Orgad et al. (ICLR 2025) vs arXiv:2510.09033. Both cannot be right in their strong forms.
17. **Is semantic entropy worth its cost?** No refutation, but three 2026 papers claim single-pass alternatives match or beat it.
18. **Are specialized hallucination detectors better than LLM-as-a-Judge?** TRIVIA+ says LLM-as-Judge is competitive. Every vendor detector paper says otherwise. Note who benefits from each answer.
19. **Do finer citations help or hurt?** Sentence-level requirements degrade attribution 16–276%, and the penalty grows with model scale — contradicting prevailing benchmark design.
20. **Did GraphRAG pay off?** Multi-hop win contested between +3 points and +27 EM; global-summarization claim not replicated; cost is 41–57× indexing and ~377× inference; and at scale it cannot be built.
21. **Do memory systems beat long context?** Independent 2026 work says long context wins on recall; memory wins only on cost past ~10 turns.

---

## 9. WHAT I COULD NOT VERIFY — DO NOT CITE THESE AS FACTS

**Numbers/claims:**
- Current (Aug 2026) MTEB/MMTEB #1. The leaderboard is a client-rendered Gradio Space; every fetch returned the loading shell.
- Numeric tables in arXiv:2605.05242 (grep-based Direct Corpus Interaction vs BM25/dense/rerankers). Existence, authorship (incl. Yejin Choi, Jiawei Han, Jimmy Lin), date, and qualitative claims verified; **the numbers are not.** Worth a follow-up given the author list.
- AttrScore per-model F1 and its human-agreement figure.
- ReDeEP's own reported AUROC (abstract omits it; 0.73 is third-party).
- TRIVIA+ detector AUROC table (PDF tables unreadable).
- Exact figures inside arXiv:2504.01818, arXiv:2604.03676, arXiv:2403.06789.
- 2026 numbers for NV-Embed, Stella, BGE-M3 successors, Voyage flagship tiers.
- Current Kaggle FACTS Grounding standings (page rendered empty).
- HHEM figures for GPT-5 Pro / Claude Opus 4.7 that appear in SEO summaries but **not** on the actual Vectara leaderboard.
- Lynx 2.0 numbers (blog-only). Galileo Hallucination Index methodology.
- "BEIR is no longer zero-shot because researchers train on it" and "MTEB has 400+ models with marginal differences" — search-snippet only.
- SAAS, AutoSearch, and Agentic-R report no numbers in their abstracts.

**Venues:**
- **A peer-reviewed venue for Edge et al. (GraphRAG, arXiv:2404.16130) — likely never formally published.**
- ICLR 2026 for GraphRAG-Bench (2506.05690) and Youtu-GraphRAG (2508.19855) — asserted in GitHub titles only.
- ECAI 2025 for Mem0 (2504.19413) — secondary sources only.
- TACL for MTRAG — IBM's page says so; arXiv has no comment.
- PathRAG (2502.14902).
- arXiv ID for Stechly/Valmeekam/Kambhampati ICLR 2025 (search said 2402.08115; unverified).
- "Gosling Grows Up" (SIGIR 2025) — arXiv ID not verified and deliberately not guessed.

**Corrections to things circulating as fact:**
- There is **no NVIDIA "RAG vs GraphRAG" study**; the systematic evaluation is arXiv:2502.11371 from **Michigan State + Meta + IBM**.
- There is **no academic paper titled "GraphRAG under fire."**
- **"Letta 49.0% on LongMemEval per arXiv:2603.04814" is false** — that paper evaluates Mem0 vs long-context only.
- **arXiv:2601.04618 (REPAIR) was withdrawn by its authors on 14 Apr 2026.** Do not cite its +5.6pp.
- **Reason-ModernColBERT has no paper**, and its "beats ReasonIR-8B by 2.5 nDCG" claim holds only on the StackExchange splits; it loses on the full BRIGHT mean.
- **LettuceDetect is not SOTA on RAGTruth**; fine-tuned Llama-3-8B at 83.9% is.

**Citation counts** (Semantic Scholar API, 8 Aug 2026; the API was rate-limited for most of this session and OpenAlex badly undercounts preprints, so this is a partial set): GraphRAG 1,875 · Search-R1 1,317 · LLMs Get Lost 378 · BRIGHT 177 (ICLR) · BrowseComp-Plus 155 · ReasonIR 76 · Rank1 72. GitHub traction as a secondary proxy: Alibaba-NLP/DeepResearch 19.8k stars · Search-R1 5.3k · FlashRAG 3.5k · PyLate 877 · BrowseComp-Plus 327 · ReasonIR 230.
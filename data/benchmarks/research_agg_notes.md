# Verification: rp_agg_q01..q05 (corpus-aggregate questions, research_papers)

Method: every pattern was run over ALL 25 papers' `pages.jsonl` with a robust searcher
(`cgrep2.py` in this scratchpad) that searches BOTH the raw page text AND a dehyphenated
view (`re.sub(r'-\s+', '', text)`) of every page, so line-break hyphenations like
"Trivi- aQA" or "Self- RAG" are caught. "—" = zero matches in the whole paper.

Known pitfalls found and handled during verification:
- Case-insensitive `BEIR` matches "Ri**beir**o" (author name) in rp020 — BEIR was DROPPED as an axis.
- `\bBEIR\b` misses "BEIR6" (footnote-marker concatenation) in rp016 — another reason BEIR was dropped.
- rp016 writes "Trivi- aQA" across a line break; rp009 writes "Natural Qustion(NQ)" (typo) —
  both caught via the dehyphenated view / `\bNQ\b` alternation.
- rp012 writes "NaturalQuestions-Open" (one word) — pattern uses `Natural\s*Questions?`.

## rp_agg_q01 — papers mentioning MTEB

Pattern: `MTEB|Massive\s+Text\s+Embedding` (case-insensitive, raw+dehyph).
Both alias forms verified; "C-MTEB" / "MTEB Multilingual" contain the substring and occur
only inside already-matched papers.

| doc | paper | MTEB mention pages |
|---|---|---|
| rp001 | ColPali | pp. 14 |
| rp002 | VisRAG | pp. 8,13 |
| rp003 | DSE | — |
| rp004 | Self-RAG | — |
| rp005 | CRAG | — |
| rp006 | Ragas | — |
| rp007 | RAPTOR | — |
| rp008 | GraphRAG | — |
| rp009 | RAG Survey | pp. 9 |
| rp010 | Speculative RAG | — |
| rp011 | RankRAG | — |
| rp012 | Lost in the Middle | — |
| rp013 | FLARE | — |
| rp014 | Qwen3 Embedding | pp. 1,2,6,7,8,9,10,13,14 |
| rp015 | NV-Embed | pp. 1,2,3,4,6,7,8,9,10,13,15,16,17,18,21,22 |
| rp016 | BGE-M3 | — |
| rp017 | Gecko | pp. 1,2,3,7,8,9,12,15,16,17,18 |
| rp018 | Nomic Embed | pp. 1,2,3,7,9,10,11,14 |
| rp019 | Reflexion | — |
| rp020 | Agents Survey | — |
| rp021 | ToolLLM | — |
| rp022 | Qwen2-VL | — |
| rp023 | Qwen2.5-VL | — |
| rp024 | InternVL | — |
| rp025 | DeepSeek-R1 | — |

Gold: 7 papers = rp001, rp002, rp009, rp014, rp015, rp017, rp018.
Context reading: rp001 p14 = bibliography entry only (cites the MTEB paper); rp002 p8 =
passing comparison ("bge-large outperforms BM25 on benchmarks like MTEB"), p13 = reference;
rp009 p9 = survey discussion of the MTEB leaderboard (+C-MTEB); rp014/rp015/rp017/rp018 =
evaluate on MTEB (headline results). Question therefore uses "mention ... including
references". Confirmed non-mentions: rp016 (BGE-M3) never says MTEB (evaluates on
MIRACL/MKQA/MLDR instead) — a deliberate distractor.

## rp_agg_q02 — count of papers mentioning HotpotQA

Pattern: `Hotpot\W?QA` (case-insensitive, raw+dehyph). Variants HotPotQA/Hotpotqa covered.

| doc | paper | HotpotQA mention pages |
|---|---|---|
| rp001 | ColPali | — |
| rp002 | VisRAG | — |
| rp003 | DSE | — |
| rp004 | Self-RAG | — |
| rp005 | CRAG | — |
| rp006 | Ragas | — |
| rp007 | RAPTOR | — |
| rp008 | GraphRAG | pp. 3,16,18 |
| rp009 | RAG Survey | pp. 7,13,19 |
| rp010 | Speculative RAG | pp. 15,18,19 |
| rp011 | RankRAG | pp. 6,7,8,9,15,16,19,20,22,23 |
| rp012 | Lost in the Middle | — |
| rp013 | FLARE | — |
| rp014 | Qwen3 Embedding | pp. 13 |
| rp015 | NV-Embed | pp. 6,15,20,21,22 |
| rp016 | BGE-M3 | pp. 3,12,15 |
| rp017 | Gecko | pp. 6,14,17,18 |
| rp018 | Nomic Embed | pp. 7,16 |
| rp019 | Reflexion | pp. 2,5,6,7,11,12,14,17,18,19 |
| rp020 | Agents Survey | — |
| rp021 | ToolLLM | — |
| rp022 | Qwen2-VL | — |
| rp023 | Qwen2.5-VL | — |
| rp024 | InternVL | — |
| rp025 | DeepSeek-R1 | — |

Gold: count = 10 (rp008, rp009, rp010, rp011, rp014, rp015, rp016, rp017, rp018, rp019).
Context reading: rp008 p3 names HotPotQA as an existing multi-hop QA dataset (contrast to
its summarization focus) + refs; rp010/rp011/rp019 evaluate on it; rp014/rp015/rp016/rp017/
rp018 list it as training data; rp009 survey. All are mentions, matching the question wording.

## rp_agg_q03 — papers mentioning Self-RAG (bridge: rp004 is in the collection)

Pattern: `Self\W?RAG` (case-insensitive, raw+dehyph; catches SELF-RAG, Self-rag, "Self- RAG").

| doc | paper | Self-RAG mention pages |
|---|---|---|
| rp001 | ColPali | — |
| rp002 | VisRAG | pp. 11 |
| rp003 | DSE | — |
| rp004 | Self-RAG | pp. 1,2,3,4,5,6,8,9,10,16,17,18,19,20,21 |
| rp005 | CRAG | pp. 2,3,4,7,8,9,10,15,16 |
| rp006 | Ragas | — |
| rp007 | RAPTOR | — |
| rp008 | GraphRAG | — |
| rp009 | RAG Survey | pp. 5,6,11,12,17 |
| rp010 | Speculative RAG | pp. 6,7,8,9,11,16,18 |
| rp011 | RankRAG | pp. 6,7,10 |
| rp012 | Lost in the Middle | — |
| rp013 | FLARE | — |
| rp014 | Qwen3 Embedding | — |
| rp015 | NV-Embed | — |
| rp016 | BGE-M3 | — |
| rp017 | Gecko | — |
| rp018 | Nomic Embed | — |
| rp019 | Reflexion | — |
| rp020 | Agents Survey | — |
| rp021 | ToolLLM | — |
| rp022 | Qwen2-VL | — |
| rp023 | Qwen2.5-VL | — |
| rp024 | InternVL | — |
| rp025 | DeepSeek-R1 | — |

Gold: excluding rp004 itself -> 5 papers = rp002, rp005, rp009, rp010, rp011.
Context reading: rp002 p11 = bibliography entry only; rp005 (CRAG) implements CRAG on top
of Self-RAG and compares extensively (pp.2-10,15-16); rp009 discusses Self-RAG as adaptive
retrieval (pp.5-6,11-12,17); rp010 compares Self-RAG/Self-CRAG baselines (pp.6-9,11);
rp011 lists Self-RAG 7B as a baseline in its results table (pp.6-7,10).

## rp_agg_q04 — superlative: MS MARCO vs Natural Questions vs HotpotQA vs TriviaQA

Patterns (all raw+dehyph):
- MS MARCO: `MS\W?MARCO|MSMARCO` ci — covers MS MARCO / MS-MARCO / MSMARCO / MSMarco.
- Natural Questions: `Natural\s*Questions?\b|\bNQ\b` case-SENSITIVE — covers
  "Natural Questions", "Natural Question", "NaturalQuestions-Open", bare "NQ";
  case-sensitivity + word boundary prevents false positives; every NQ hit context read
  and confirmed genuine (incl. rp011 figure axis label "(a) NQ", rp009 typo "Natural Qustion(NQ)").
- HotpotQA: as q02. TriviaQA: `Trivia\W?QA` ci (catches rp016 "Trivi- aQA" via dehyph;
  rp013 hit is a bibliography entry — still a mention).

| doc | paper | MS MARCO | Natural Questions/NQ | HotpotQA | TriviaQA |
|---|---|---|---|---|---|
| rp001 | ColPali | pp. 11 | — | — | — |
| rp002 | VisRAG | — | — | — | — |
| rp003 | DSE | pp. 3,10 | pp. 1,2,3,5,6,7,8,9,11,13,14 | — | pp. 7,11,13 |
| rp004 | Self-RAG | pp. 7,20 | pp. 17,18 | — | pp. 7,12,20 |
| rp005 | CRAG | — | — | — | — |
| rp006 | Ragas | — | — | — | — |
| rp007 | RAPTOR | — | — | — | — |
| rp008 | GraphRAG | — | — | pp. 3,16,18 | — |
| rp009 | RAG Survey | pp. 6,13,19 | pp. 13 | pp. 7,13,19 | pp. 13,19 |
| rp010 | Speculative RAG | pp. 16 | — | pp. 15,18,19 | pp. 1,6,7,8,9,10,12,16,17,18,22 |
| rp011 | RankRAG | pp. 4,5,8,9,10,17 | pp. 3,6,7,8,9,10,16,19,20,21,22 | pp. 6,7,8,9,15,16,19,20,22,23 | pp. 3,6,7,8,9,12,16,19,20,21,22 |
| rp012 | Lost in the Middle | pp. 3,9 | pp. 3,9,12,14 | — | — |
| rp013 | FLARE | — | — | — | pp. 11 |
| rp014 | Qwen3 Embedding | pp. 13 | pp. 13 | pp. 13 | — |
| rp015 | NV-Embed | pp. 3,6,11,14,18,20,21,22 | pp. 6,20,21 | pp. 6,15,20,21,22 | — |
| rp016 | BGE-M3 | pp. 3,6,10,11,15 | pp. 3,15 | pp. 3,12,15 | pp. 3,10 |
| rp017 | Gecko | pp. 8,16,17,18 | pp. 6,16,17,18 | pp. 6,14,17,18 | — |
| rp018 | Nomic Embed | pp. 3,7,8,12 | pp. 7,8 | pp. 7,16 | — |
| rp019 | Reflexion | — | — | pp. 2,5,6,7,11,12,14,17,18,19 | — |
| rp020 | Agents Survey | — | — | — | — |
| rp021 | ToolLLM | — | — | — | — |
| rp022 | Qwen2-VL | — | — | — | — |
| rp023 | Qwen2.5-VL | — | — | — | — |
| rp024 | InternVL | — | — | — | — |
| rp025 | DeepSeek-R1 | — | — | — | — |

Counts: MS MARCO **12** (rp001, rp003, rp004, rp009, rp010, rp011, rp012, rp014, rp015,
rp016, rp017, rp018) > Natural Questions 10 > HotpotQA 10 > TriviaQA 7.
Gold: MS MARCO, 12 papers. Spot-checked borderline contexts: rp010 p16 & rp012 p3/p9 =
"Contriever fine-tuned on MS-MARCO" (genuine mentions); rp001 p11 = bibliography entry.

## rp_agg_q05 — rubric: retriever/embedder trained on LLM/VLM-generated synthetic data

Patterns: `synthetic|synthesi[zs]e` ci (raw+dehyph) to find candidates, plus
`pseudo.?quer|quer(y|ies)\s+generat|generat\w*\s+quer|LLM.generated` ci to catch papers
that might qualify without the word "synthetic". Every candidate context was read.

| doc | paper | synthetic/synthesize pages |
|---|---|---|
| rp001 | ColPali | pp. 6,10,11,16,17 |
| rp002 | VisRAG | pp. 1,2,5,6,7,9,11,17,19,20,21,24 |
| rp003 | DSE | — |
| rp004 | Self-RAG | pp. 22 |
| rp005 | CRAG | — |
| rp006 | Ragas | — |
| rp007 | RAPTOR | — |
| rp008 | GraphRAG | — |
| rp009 | RAG Survey | pp. 3,6,12,20 |
| rp010 | Speculative RAG | pp. 4 |
| rp011 | RankRAG | pp. 4,5,18 |
| rp012 | Lost in the Middle | pp. 2,6,7 |
| rp013 | FLARE | — |
| rp014 | Qwen3 Embedding | pp. 2,5,6,8,9,12,13,14 |
| rp015 | NV-Embed | pp. 1,2,3,4,5,7,8,9,10,18 |
| rp016 | BGE-M3 | pp. 3,4,5,14 |
| rp017 | Gecko | pp. 1,3,4,5,9,10 |
| rp018 | Nomic Embed | — |
| rp019 | Reflexion | — |
| rp020 | Agents Survey | pp. 38,59,81 |
| rp021 | ToolLLM | — |
| rp022 | Qwen2-VL | pp. 5,46 |
| rp023 | Qwen2.5-VL | pp. 6,7,9,10,16 |
| rp024 | InternVL | pp. 5,15,23 |
| rp025 | DeepSeek-R1 | pp. 13,20,72 |

Qualifying (synthetic data used to TRAIN the paper's retriever/embedding model):
- rp001 ColPali p6: training set = 63% academic + 37% "synthetic dataset ... augmented with VLM-generated (Claude-3 Sonnet) pseudo-questions".
- rp002 VisRAG p1/p5/p17/p20: "collect both open-source and synthetic data to train the retriever"; "utilize GPT-4o to generate queries" on web-crawled PDFs.
- rp014 Qwen3 Embedding p5-p6/p12: "Large-Scale Synthetic Data-Driven Weak Supervision Training"; pairs synthesized with Qwen3-32B.
- rp015 NV-Embed p1-p2: "For training data, we utilize the hard-negative mining, synthetic data generation and existing publicly available datasets".
- rp016 BGE-M3 p3/p14: "we generate synthetic data to mitigate the shortage of long document retrieval" (MultiLongDoc); appendix A.2 gives the GPT-3.5 generation prompt.
- rp017 Gecko p1/p3-p4: "two-step distillation process begins with generating diverse, synthetic paired data using an LLM" (FRet).

Non-qualifying hits (read and excluded, listed as do-not-accept in the gold):
- rp004 p22: "Synthesize a poem" = an example prompt; Self-RAG's retriever is off-the-shelf Contriever.
- rp010 p4: strong LM synthesizes rationales to instruction-tune the RAG *drafter* (a generator).
- rp011: "Synthetic Conversation" = LLM instruction-tuning blend, not retriever pairs.
- rp012: "synthetic key-value retrieval task" = an evaluation task.
- rp009 p20 (bibliography, speech), rp020 (simulated societies / refs), rp022-rp025 (general VLM/LLM pre-training data).
- rp003 DSE and rp018 Nomic Embed: zero qualifying hits on either pattern family (deliberate distractors among retrieval/embedding papers).
- rp013 FLARE generates queries at INFERENCE time (active retrieval), not training data.

Gold rubric: name >=3 of {rp001, rp002, rp014, rp015, rp016, rp017}.

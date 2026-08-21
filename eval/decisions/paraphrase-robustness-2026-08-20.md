# Paraphrase-robustness study: does multi-vector help once queries stop matching document wording?

**Date:** 2026-08-20 · **Judge:** Sonnet subagents throughout (user-directed; no 4b numbers in this record)

## Question

The multi-vector experiments (multivector-retrieval-2026-08-19.md) lost or tied on the
standard gold sets, and the leading explanation was question style: our gold queries
reuse document vocabulary, which favors FTS+dense. User-directed test: rewrite every
gold question as a same-meaning paraphrase a person who never read the documents would
ask, then re-run the retrieval configurations.

## The paraphrase set — eval/goldset/paraphrases.jsonl

120 rows ({id, original, para}), written by 5 Sonnet agents under rules: meaning exactly
preserved (the existing gold answer must remain the unique correct answer), maximum
vocabulary drift, identifiers replaced by unambiguous descriptions where possible.
Mean content-word Jaccard overlap with originals: **0.21**. A separate Sonnet verifier
checked all 120 pairs for answer-preservation: **0 flags**. The gold sets themselves are
untouched; the runner substitutes queries by id (PARA_QUERIES hook in the scratch runner).

## Results (Sonnet bulk = 5 blind agents/arm; every flip vs control panel-verified by 3 voters)

| retrieval config | original questions | paraphrased questions |
|---|---|---|
| dense + FTS (shipped hybrid) | **100/120** | **95/120** (−5: paraphrasing genuinely hurts; rfc/docs hardest hit) |
| MV replaces dense (LFM2.5-ColBERT) | 88 (panel net −1) | 90 — panel net **−5 real** vs ctlp (1 gain / 6 losses, 21/21 votes unanimous; rfc 13/24) |
| FTS + dense + MV (3-leg RRF) | 100 (panel net −1) | 97 — panel net **+2 real** vs ctlp (3 gains: docs_d13/d17/d18, 1 loss: rfc_q24; 3 of 8 panel cells split-vote) |

## Findings

1. **Paraphrasing costs the shipped pipeline 5 rows** (100→95). The reworded set does
   what it was built to do: shifts load from lexical matching to semantics.
2. **MV as a dense replacement fails even here** — net −5 real, the worst cell measured.
   The 0.6B dense embedder absorbs vocabulary drift better than the 350M ColBERT
   (rfc_q01/q02/q08 lost on rewritten wording that dense handled). "It was only the
   question style" is REFUTED for replacement mode.
3. **MV as a third leg is the first genuinely positive cell** (+2 real): with FTS
   weakened by paraphrasing, the extra semantic leg recovers docs rows the rewording
   broke (d13/d17 were paraphrase-losses; 3-leg wins them back). On original questions
   the same config was net −1. Caveats: +2 on n=120 is at the noise floor, and unlike
   the mvp verdict the deciding panels carry 3 split votes.

## Decision

Defaults unchanged: dense+FTS hybrid stays. The 3-leg's +2 appears only on
paraphrase-style queries, costs −1 on document-phrased ones, and carries the sidecar +
token-level-storage overhead. **Recommendation:** treat 3-leg (`MV_RRF_LEG=1`) as the
configuration of choice only for a deployment whose real users ask in their own words
rather than the documents' — and validate on that workload first.

## Fusion recommendation (recorded for follow-up)

Rank-based RRF is where MV's candidates go to die: a leg that disagrees re-shuffles
everything, so gains pay a tax elsewhere (rfc_q24 won in replacement mode, lost in both
3-leg runs). If MV integration is ever pursued seriously, change the fusion, not the leg:
1. **Candidate-pool union**: feed the union of each leg's top-k to the Qwen reranker and
   let it arbitrate — no rank fighting, bounded by reranker latency (~linear in pool size).
2. **Weighted RRF** (down-weight the MV leg) as a cheaper middle ground.
3. Score-normalized fusion only if (1) is too slow — score scales across legs are not
   comparable without calibration.
Option 1 is the recommended next experiment because the reranker is this pipeline's
strongest component and the observed failure mode is exactly "right chunk found by one
leg, pushed out of the reranker window by fusion".

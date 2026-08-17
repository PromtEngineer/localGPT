"""Full-document escalation wiring (roadmap item 4.1).

**Off by default.** ``retrieval.document_escalation.enabled`` must be set to
``True`` for anything in this module to run; until it has been measured on the
gold set it is a flag, not a behaviour.

What it does
------------
When candidate selection finishes — *after* the evidence-sufficiency retry
(§5) has had its one attempt — and the evidence signal is still below
threshold, the top-ranked chunk's whole document is reassembled in chunk order
and appended to the synthesis context as one clearly-delimited block. One
document, one time per user query, capped at a token budget. The chunk
citations are untouched: escalation adds reading material, it does not add or
reorder sources.

Why it is a subclass
--------------------
The decision point sits *between* candidate selection and synthesis, and both
of those live inside ``RetrievalPipeline.run()``. Rather than restructure
``rag_system/pipelines/retrieval_pipeline.py`` (owned by another workstream this
wave), this subclass hooks the two methods ``run()`` already calls in sequence:

* ``retrieve_candidates()`` — read the final, post-retry evidence signal and
  remember the top-ranked chunk;
* ``_synthesize_final_answer()`` — append the document to the facts string
  before the base implementation builds its prompt.

That keeps escalation to a **single** generation pass — the alternative,
re-synthesising after ``run()`` returns, would pay for two answers and stream
tokens twice. The handoff between the two hooks is thread-local, so the agent's
parallel sub-query fan-out cannot cross-wire one sub-query's document into
another's synthesis.

If a future refactor renames either hook, escalation stops firing and logs a
warning; it never breaks retrieval.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

from rag_system.pipelines.retrieval_pipeline import RetrievalPipeline
from rag_system.retrieval.document_fetch import fetch_document, format_escalation_block

# Defaults live here, not in rag_system/main.py, because the profiles are not
# editable this wave. Every read goes through `config.get(...)` with these
# fallbacks, so adding the block to a profile later changes behaviour without
# touching this file. The keys the gate should add are listed in
# eval/decisions/phase4-escalation-tokens.md.
DEFAULT_DOCUMENT_ESCALATION: Dict[str, Any] = {
    "enabled": False,
    "max_documents": 1,
    "token_budget": 6000,
}

# Fallback trigger threshold, used only when neither the escalation block nor
# the retry block names one. Same value the shipped retry calibrated to.
_FALLBACK_MIN_EVIDENCE = 0.12


class _EscalationBudget:
    """How many documents this user query is still allowed to escalate."""

    def __init__(self, max_documents: int) -> None:
        self._lock = threading.Lock()
        self._remaining = max(0, int(max_documents))
        self.events: List[Dict[str, Any]] = []

    def take(self) -> bool:
        with self._lock:
            if self._remaining <= 0:
                return False
            self._remaining -= 1
            return True

    def note(self, payload: Dict[str, Any]) -> None:
        with self._lock:
            self.events.append(payload)


class EscalatingRetrievalPipeline(RetrievalPipeline):
    """``RetrievalPipeline`` plus roadmap 4.1. Inert unless the flag is on."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._escalation_local = threading.local()
        self._escalation_budget: Optional[_EscalationBudget] = None
        if not hasattr(RetrievalPipeline, "retrieve_candidates"):
            print(
                "⚠️  RetrievalPipeline has no retrieve_candidates(); full-document "
                "escalation (roadmap 4.1) is inert on this build."
            )

    # -------------------------------------------------------------- config

    def escalation_config(self) -> Dict[str, Any]:
        """The ``document_escalation`` block, merged across both containers.

        Mirrors how ``_retry_config`` resolves ``retrieval.retry`` versus a
        runtime ``retrievers.retry`` override.
        """
        merged = dict(DEFAULT_DOCUMENT_ESCALATION)
        for container_key in ("retrieval", "retrievers"):
            block = (self.config.get(container_key) or {}).get("document_escalation")
            if isinstance(block, dict):
                merged.update(block)
        return merged

    def _escalation_threshold(self, cfg: Dict[str, Any]) -> float:
        """Below this evidence score, escalate.

        Defaults to the retry's own threshold: escalation is what happens when
        the retry has already run and the evidence is *still* weak, so the two
        should be judged against the same bar unless told otherwise.
        """
        explicit = cfg.get("min_evidence")
        if explicit is not None:
            try:
                return float(explicit)
            except (TypeError, ValueError):
                pass
        try:
            retry_cfg = self._retry_config()
        except Exception:
            retry_cfg = {}
        for key in ("min_top_score", "min_rerank_score"):
            value = retry_cfg.get(key)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
        return _FALLBACK_MIN_EVIDENCE

    # ------------------------------------------------------- request scope

    def begin_escalation_request(self) -> Optional[_EscalationBudget]:
        """Open a per-user-query escalation budget. Returns ``None`` when off."""
        cfg = self.escalation_config()
        if not cfg.get("enabled"):
            self._escalation_budget = None
            return None
        self._escalation_budget = _EscalationBudget(cfg.get("max_documents", 1))
        return self._escalation_budget

    def end_escalation_request(self) -> Optional[_EscalationBudget]:
        budget, self._escalation_budget = self._escalation_budget, None
        self._escalation_local.pending = None
        return budget

    # -------------------------------------------------------------- hooks

    def retrieve_candidates(self, query, table_name=None, sub_queries=None,
                            event_callback=None, *, filters=None):
        # `filters` (roadmap 4.4) restores signature parity with the base
        # class: keyword-only and passed straight through — the base compiles
        # it and opens its filter_scope around the retrieval. None keeps the
        # thread-local scope opened by run(), so the default path is unchanged.
        result = super().retrieve_candidates(query, table_name, sub_queries,
                                             event_callback, filters=filters)
        self._escalation_local.pending = None
        if self._escalation_budget is None:
            return result
        try:
            self._escalation_local.pending = self._plan_escalation(result, table_name)
        except Exception as e:  # never let observability break retrieval
            print(f"⚠️  Escalation planning failed: {e}")
            self._escalation_local.pending = None
        return result

    def _synthesize_final_answer(self, query: str, facts: str, *, event_callback=None) -> str:
        plan = getattr(self._escalation_local, "pending", None)
        self._escalation_local.pending = None
        if plan is not None:
            block = self._materialise(plan, event_callback)
            if block:
                facts = f"{facts}\n\n{block}"
        return super()._synthesize_final_answer(query, facts, event_callback=event_callback)

    # ------------------------------------------------------------ internals

    def _plan_escalation(self, candidates: Dict[str, Any],
                         table_name: Optional[str]) -> Optional[Dict[str, Any]]:
        """Decide whether the *final* evidence still warrants a deep read."""
        documents = candidates.get("documents") or []
        first_stage = candidates.get("first_stage") or []
        if not documents and not first_stage:
            return None

        # Same preference order the retry uses: a calibrated reranker
        # probability when there is one, else the dense contrast score.
        score = self._rerank_evidence_score(documents)
        signal = "rerank"
        if score is None:
            score = self._dense_evidence_score(first_stage)
            signal = "dense_contrast"
        if score is None:
            # fts_only, or a legacy unnormalized table: no signal, no escalation
            # — exactly the rule the retry follows.
            return None

        cfg = self.escalation_config()
        threshold = self._escalation_threshold(cfg)
        if score >= threshold:
            return None

        top = (documents or first_stage)[0]
        document_id = top.get("document_id") or (top.get("metadata") or {}).get("document_id")
        if not document_id:
            return None

        return {
            "document_id": document_id,
            "table_name": table_name or self.storage_config.get("text_table_name"),
            "token_budget": int(cfg.get("token_budget", 6000) or 0),
            "signal": signal,
            "score": round(float(score), 4),
            "threshold": float(threshold),
        }

    def _materialise(self, plan: Dict[str, Any], event_callback) -> Optional[str]:
        budget = self._escalation_budget
        if budget is None or not budget.take():
            return None
        try:
            document = fetch_document(
                self._get_db_manager(),
                plan["table_name"],
                plan["document_id"],
                token_budget=plan["token_budget"],
            )
        except Exception as e:
            print(f"⚠️  Full-document escalation failed: {e}")
            return None
        if document is None:
            return None

        payload = document.as_event_payload()
        payload.update({
            "signal": plan["signal"],
            "score": plan["score"],
            "threshold": plan["threshold"],
            "token_budget": plan["token_budget"],
        })
        budget.note(payload)
        print(
            f"\n📄 Full-document escalation: {payload['document_name']} "
            f"({payload['chunks_used']}/{payload['chunks_total']} chunks, "
            f"~{payload['approx_tokens']} tokens, {plan['signal']}="
            f"{plan['score']} < {plan['threshold']})"
        )
        if event_callback:
            try:
                event_callback("document_escalation", payload)
            except Exception:
                pass
        return format_escalation_block(document)

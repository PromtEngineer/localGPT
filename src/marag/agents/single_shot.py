from __future__ import annotations

from ..config import Config
from ..llm import LLM
from ..retrieve.hybrid import Retriever

SYSTEM = """You are a precise document-QA assistant. Answer ONLY from the provided context.
Rules:
- Cite evidence as [doc_id pN] immediately after each claim.
- If the context is insufficient to answer fully, say exactly what is missing — do not guess.
- Numbers must be copied exactly from the context, with units/metric names.
- Be concise: answer first, no preamble."""


def answer_single_shot(
    question: str, dataset: str | list[str], cfg: Config, retriever: Retriever, llm: LLM | None = None
) -> dict:
    datasets = [dataset] if isinstance(dataset, str) else list(dict.fromkeys(dataset))
    hits = (retriever.search_multi(question, datasets) if len(datasets) > 1
            else retriever.search(question, datasets[0]))
    ctx = "\n\n".join(f"[{h['doc_id']} p{h['page']}] {h['raw_text'][:1500]}" for h in hits)
    llm = llm or LLM("orchestrator", cfg)
    answer = llm.text(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"CONTEXT:\n{ctx}\n\nQUESTION: {question}"},
        ],
        max_tokens=4096,  # reasoning models think first; a small budget starves the answer
        temperature=0.0,
    )
    return {
        "mode": "single_shot",
        "answer": answer,
        "contexts": [{"doc_id": h["doc_id"], "page": h["page"]} for h in hits],
        "tool_calls": 1,
    }

from __future__ import annotations

import json

from ..config import Config
from ..llm import LLM, strip_thinking
from ..retrieve.hybrid import Retriever
from .tools import TOOL_SPECS, ToolBox

SYSTEM = """You are a research agent answering questions over a private document corpus using tools.

DISCIPLINE (follow in order):
1. Decompose the question: what distinct pieces of evidence are needed?
2. hybrid_search for each piece (reformulate rather than repeat a failed query).
3. Narrow with grep (exact names/numbers) and read_doc to confirm EXACT wording before citing.
4. For ANY arithmetic over table data (differences, growth rates, totals): list_tables then sql. Never compute numbers in your head. When two similarly-named metrics exist (e.g. a segment vs a market category), confirm the exact row label via sql or read_doc before answering.
5. If the question or the evidence involves a CHART, FIGURE, DIAGRAM, or DRAWING, you MUST confirm the values with view_page before answering — even when numbers appear in the extracted text. Chart bar/point labels in text lose their visual association (a prior-year bar's label looks identical to the current one); only vision can attribute a value to the right bar, series, or period. Name the series/line you need by its style and label. If a detail is too small to read, you may re-call view_page with a `region` to zoom, but do NOT re-read the same chart repeatedly — if two readings disagree, report the value as approximate and say which series you read.
6. If evidence is not surfacing: step back — reformulate, try grep with different terms, check list_docs for the right document, then search again.

RULES:
- Cite every claim as [doc_id pN]. Only cite pages you actually retrieved or read.
- Report numbers EXACTLY as printed in the source (units included) — never round or
  approximate ("almost 95%" is wrong when the page says 94.9%).
- When you have verified all evidence pieces, STOP calling tools and give the final answer.
- If evidence genuinely isn't in the corpus, say exactly what is missing.
- Answer format: the direct answer in the FIRST sentence (never your plan or reasoning),
  then one short evidence summary line per hop."""


def _serialize_tool_calls(tool_calls) -> list[dict]:
    return [
        {
            "id": tc.id,
            "type": "function",
            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
        }
        for tc in tool_calls
    ]


def answer_agentic(question: str, dataset: str, cfg: Config, retriever: Retriever) -> dict:
    tb = ToolBox(cfg, dataset, retriever)
    llm = LLM("orchestrator", cfg)
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]
    transcript: list[dict] = []
    used = 0
    no_new_rounds = 0
    nudged = False

    while used < cfg.agent.max_tool_calls:
        resp = llm.chat(messages, tools=TOOL_SPECS, max_tokens=3072, temperature=0.0)
        msg = resp.choices[0].message

        if not msg.tool_calls:
            answer = strip_thinking(msg.content or "")
            if not answer.strip():
                # never accept an empty final (thinking overflow / vision quirk):
                # force a no-thinking retry, which always yields content
                answer = llm.text(
                    messages + [{"role": "user", "content": "Give your FINAL answer now with citations."}],
                    max_tokens=2048,
                    temperature=0.0,
                    reasoning="none",
                )
            return _final(answer, tb, transcript, used)

        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": _serialize_tool_calls(msg.tool_calls),
            }
        )
        round_new_evidence = 0
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = tb.dispatch(tc.function.name, args)
            used += 1
            round_new_evidence += tb.new_evidence_last_call
            transcript.append({"tool": tc.function.name, "args": args, "result_chars": len(result)})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        # marginal-utility stop: budgets beat "do you have enough?" prompts
        no_new_rounds = no_new_rounds + 1 if round_new_evidence == 0 else 0
        if no_new_rounds >= cfg.agent.no_new_info_stop and not nudged:
            nudged = True
            messages.append(
                {
                    "role": "user",
                    "content": "No new evidence is surfacing. Give your FINAL answer now with "
                    "citations, or state precisely what is missing.",
                }
            )

    messages.append(
        {"role": "user", "content": "Tool budget exhausted. Give your FINAL answer now with citations."}
    )
    final = llm.text(messages, max_tokens=4096, temperature=0.0)
    if not final.strip():
        final = llm.text(messages, max_tokens=2048, temperature=0.0, reasoning="none")
    return _final(final, tb, transcript, used)


def _final(answer: str, tb: ToolBox, transcript: list[dict], used: int) -> dict:
    return {
        "mode": "agentic",
        "answer": answer,
        "tool_calls": used,
        "evidence_pages": sorted(tb.evidence_seen),
        "transcript": transcript,
    }

"""Binary groundedness judge (Phase 0.3).

Binary pass/fail, not a Likert score, per the practitioner canon the roadmap
cites. Given (question, answer, evidence chunks) the judge returns::

    {"grounded": true | false, "reason": "<short justification>"}

It runs on the utility model (``qwen3.5:4b`` by default) through the repo's own
``OllamaClient``, which sends ``think: false`` alongside ``format="json"`` —
without that, a thinking model puts its JSON in the ``thinking`` field and
returns an empty ``response``.

A model name starting with ``claude-`` routes to the Anthropic API instead
(eval-only — the product stays fully local; requires ``pip install anthropic``
and credentials in the environment). Motivation: the Phase-4 A/Bs showed the
4b judge returning verdicts its own reasons contradict on exactly the rows
that decide feature adoption (``eval/decisions/phase4-escalation-rerun.md``
§6). Select it per run with ``JUDGE_MODEL=claude-sonnet-5``.

Regardless of backend, the verifier's ``[Confidence: N%] [Warning: ...]``
suffix is stripped from the ANSWER before judging — one judge reason was
observed citing the confidence figure as grounds for rejection.

Validate before trusting it::

    .venv/bin/python eval/judge.py --validate

That scores the judge against ``eval/judge_validation.jsonl`` (20 hand-built
cases, 10 grounded / 10 subtly ungrounded) and prints the confusion matrix, TPR
and TNR. The roadmap's gate is >=90% overall agreement.

Judge a single case ad hoc::

    .venv/bin/python eval/judge.py --question "..." --answer "..." --evidence "..."
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(EVAL_DIR, "..")))

from rag_system.utils.ollama_client import OllamaClient  # noqa: E402

VALIDATION_PATH = os.path.join(EVAL_DIR, "judge_validation.jsonl")
RESULTS_DIR = os.path.join(EVAL_DIR, "results")

# Prompt versions are kept, not overwritten, so the history stays auditable.
# All three were run against judge_validation.jsonl on 2026-08-08 (numbers in
# eval/BASELINE.md): v1 20/20, v2 15/20 (TPR 0.50 — the extra strictness makes
# it reject correct answers), v3 20/20. v1 is the default: it ties v3 on this
# set and is the shorter prompt. v2 is kept as the recorded counter-example.
PROMPTS = {
    "v1": """You are a strict fact-checker.

EVIDENCE:
{evidence}

QUESTION: {question}

ANSWER: {answer}

Decide whether the ANSWER is fully supported by the EVIDENCE.

Rules:
- Every factual claim in the ANSWER must appear in the EVIDENCE.
- If any number, name, part identifier, duration or condition differs from the
  EVIDENCE, the answer is NOT grounded.
- If the ANSWER adds a claim the EVIDENCE does not state, it is NOT grounded,
  even if that claim sounds plausible or is true in the real world.
- Do not use any outside knowledge. The EVIDENCE is the only truth.

Respond with JSON only: {{"grounded": true or false, "reason": "<one sentence>"}}
""",

    "v2": """You are a strict fact-checker. You compare an ANSWER against the EVIDENCE
it is supposed to be based on, and you have no other source of truth.

EVIDENCE:
{evidence}

QUESTION: {question}

ANSWER: {answer}

Work through the ANSWER one claim at a time. A claim is any number, quantity,
duration, percentage, part identifier, model name, person, department, place,
condition, threshold or instruction.

For each claim ask: does the EVIDENCE state exactly this?

Mark grounded = false if ANY of the following is true:
- a number, code or identifier in the ANSWER differs from the EVIDENCE, even by
  one digit or one transposed pair of digits;
- the ANSWER attaches a value to the wrong subject (for example it swaps two
  quantities, or credits one component with another component's figure);
- the ANSWER states something the EVIDENCE does not state at all, however
  reasonable it sounds;
- the ANSWER attributes a procedure or condition to the wrong item.

Mark grounded = true only when every claim in the ANSWER is directly supported
by the EVIDENCE. Extra caution, brevity or hedging in the ANSWER is fine and
does not by itself make it ungrounded.

Respond with JSON only: {{"grounded": true or false, "reason": "<one sentence naming the specific claim you checked>"}}
""",

    "v3": """You are a strict fact-checker. The EVIDENCE below is the only truth that
exists. Ignore everything you know about the world.

EVIDENCE:
{evidence}

QUESTION: {question}

ANSWER: {answer}

Task: decide whether every claim in the ANSWER is supported by the EVIDENCE.

Procedure:
1. List, silently, every number, code, identifier, name, duration, percentage,
   threshold and instruction that appears in the ANSWER.
2. For each one, locate it in the EVIDENCE, character by character for numbers
   and part codes.
3. If one of them is absent, altered (9.2 vs 9.5, TS-71 vs TS-17, 60 vs 90), or
   attached to a different subject than in the EVIDENCE, the ANSWER is NOT
   grounded.
4. If the ANSWER contains an instruction, entitlement or consequence that the
   EVIDENCE never states, the ANSWER is NOT grounded, no matter how plausible.
5. Otherwise the ANSWER is grounded. Paraphrasing, reordering, summarising and
   omitting details are all fine — only added or altered content is a failure.

Respond with JSON only:
{{"grounded": true or false, "reason": "<one sentence naming the exact claim that decided it>"}}
""",
}

DEFAULT_VERSION = "v1"
DEFAULT_MODEL = os.getenv("JUDGE_MODEL") or os.getenv("ENRICHMENT_MODEL") or "qwen3.5:4b"


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()


# The agent's verifier appends this to every answer it checks. It is metadata
# about the answer, not part of it, and it measurably perturbs the judge.
_VERIFIER_SUFFIX = re.compile(
    r"\s*\[Confidence:\s*\d+%\]\s*(\[Warning:[^\]]*\])?\s*$")


def strip_verifier_suffix(text: str) -> str:
    return _VERIFIER_SUFFIX.sub("", text or "").strip()


class GroundednessJudge:
    def __init__(self, model: str = DEFAULT_MODEL, host: str | None = None,
                 version: str = DEFAULT_VERSION):
        if version not in PROMPTS:
            raise ValueError(f"unknown prompt version {version!r}; have {sorted(PROMPTS)}")
        self.model = model
        self.version = version
        self._use_anthropic = model.startswith("claude-")
        if self._use_anthropic:
            import anthropic  # deferred so local-only runs don't need the package
            self._anthropic = anthropic.Anthropic()
        else:
            self.client = OllamaClient(host=host or os.getenv("OLLAMA_HOST", "http://localhost:11434"))

    def _complete_anthropic(self, prompt: str) -> str:
        """One judgment via the Anthropic API, JSON shape enforced server-side."""
        response = self._anthropic.messages.create(
            model=self.model,
            max_tokens=4096,  # hard cap on thinking + response text together
            output_config={"format": {"type": "json_schema", "schema": {
                "type": "object",
                "properties": {
                    "grounded": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["grounded", "reason"],
                "additionalProperties": False,
            }}},
            messages=[{"role": "user", "content": prompt}],
        )
        if response.stop_reason == "refusal":
            return ""
        return next((b.text for b in response.content if b.type == "text"), "")

    def judge(self, question: str, answer: str, evidence) -> dict:
        if isinstance(evidence, (list, tuple)):
            evidence_text = "\n\n---\n\n".join(str(e) for e in evidence)
        else:
            evidence_text = str(evidence)

        prompt = PROMPTS[self.version].format(
            evidence=strip_verifier_suffix(evidence_text), question=question,
            answer=strip_verifier_suffix(answer))
        if self._use_anthropic:
            raw = self._complete_anthropic(prompt)
        else:
            # Temperature 0: sampling noise flipped 11/24 verdicts on
            # byte-identical answers in the 2026-08-15 ftslc screen
            # (eval/decisions/ftslc-index-fix-2026-08-15.md); a judge must be
            # deterministic to be comparable across runs.
            raw = strip_think((self.client.generate_completion(
                model=self.model, prompt=prompt, format="json",
                options={"temperature": 0}) or {}).get("response", ""))

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {"grounded": None, "reason": "judge returned unparseable JSON",
                    "raw": raw, "error": "parse_error"}

        grounded = parsed.get("grounded")
        if isinstance(grounded, str):
            grounded = grounded.strip().lower() in ("true", "yes", "grounded")
        if not isinstance(grounded, bool):
            return {"grounded": None, "reason": "judge omitted a boolean 'grounded'",
                    "raw": raw, "error": "missing_field"}
        return {"grounded": grounded, "reason": str(parsed.get("reason", "")).strip()}


def load_validation() -> list:
    rows = []
    with open(VALIDATION_PATH, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return sorted(rows, key=lambda r: r["id"])


def validate(model: str, version: str, host: str | None) -> dict:
    judge = GroundednessJudge(model=model, host=host, version=version)
    cases = load_validation()

    tp = tn = fp = fn = errors = 0
    rows = []
    for case in cases:
        verdict = judge.judge(case["question"], case["answer"], case["evidence"])
        predicted, label = verdict["grounded"], case["label_grounded"]
        if predicted is None:
            errors += 1
            outcome = "ERROR"
        elif label and predicted:
            tp += 1
            outcome = "TP"
        elif label and not predicted:
            fn += 1
            outcome = "FN"
        elif not label and not predicted:
            tn += 1
            outcome = "TN"
        else:
            fp += 1
            outcome = "FP"
        rows.append({**case, "predicted": predicted, "outcome": outcome,
                     "judge_reason": verdict.get("reason"), "raw": verdict.get("raw")})
        flag = " " if outcome in ("TP", "TN") else "<"
        print(f"  {flag} {case['id']:<28} label={'grounded ' if label else 'UNgrounded'} "
              f"pred={str(predicted):<5} {outcome:<5} {verdict.get('reason', '')[:70]}")

    positives, negatives = tp + fn, tn + fp
    summary = {
        "prompt_version": version,
        "model": model,
        "n": len(cases),
        "confusion": {"TP": tp, "FN": fn, "TN": tn, "FP": fp, "unparseable": errors},
        "tpr": round(tp / positives, 4) if positives else None,
        "tnr": round(tn / negatives, 4) if negatives else None,
        "agreement": round((tp + tn) / len(cases), 4) if cases else None,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    print(f"\n  prompt {version} on {model}")
    print(f"  confusion  TP={tp}  FN={fn}  TN={tn}  FP={fp}  unparseable={errors}")
    print(f"  TPR (grounded correctly accepted)   {summary['tpr']}")
    print(f"  TNR (ungrounded correctly rejected) {summary['tnr']}")
    print(f"  overall agreement                   {summary['agreement']}  (gate: >= 0.90)")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(RESULTS_DIR, f"judge_{version}_{stamp}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "cases": rows}, fh, indent=2)
    print(f"  written    {out_path}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--validate", action="store_true", help="score against judge_validation.jsonl")
    parser.add_argument("--prompt-version", default=DEFAULT_VERSION, choices=sorted(PROMPTS))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--host", default=None)
    parser.add_argument("--question")
    parser.add_argument("--answer")
    parser.add_argument("--evidence", action="append", default=None)
    args = parser.parse_args()

    if args.validate:
        summary = validate(args.model, args.prompt_version, args.host)
        return 0 if (summary["agreement"] or 0) >= 0.90 else 1

    if not (args.question and args.answer and args.evidence):
        parser.error("pass --validate, or all of --question / --answer / --evidence")

    judge = GroundednessJudge(model=args.model, host=args.host, version=args.prompt_version)
    print(json.dumps(judge.judge(args.question, args.answer, args.evidence), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

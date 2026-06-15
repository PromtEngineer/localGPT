"""Provider-neutral data-egress policy for contextual enrichment.

LocalGPT is local-first but supports optional cloud enrichment (Anthropic /
OpenAI / Groq) during indexing, which sends document text off the machine. This
module is the governance layer in front of that egress:

    text  ->  scan_text() deterministic secret/PII detectors
          ->  evaluate() policy decision (allow / redact / block)
          ->  PolicyGuardedEnricher enforces it at the single cloud factory

Design rules:
- **Fail-closed.** Secrets block cloud egress by default; an invalid/unknown
  policy value resolves to "block", never "allow".
- **No secret ever leaves or is recorded.** Detectors return categories, types,
  and spans — never the matched value. Audit records carry counts only.
- **Deterministic.** Pure regex + Luhn; no LLM, no network, no randomness, so
  the same text always yields the same decision (and it's unit-testable).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

# --- categories & actions -------------------------------------------------

SECRET = "secret"  # credentials/keys — leaking these is the worst case
PII = "pii"  # personal data — often the legitimate content being indexed

ALLOW = "allow"
REDACT = "redact"
BLOCK = "block"

_ACTION_RANK = {ALLOW: 0, REDACT: 1, BLOCK: 2}
_VALID_ACTIONS = set(_ACTION_RANK)

# Default: never ship credentials to the cloud, but don't block ordinary
# documents that happen to contain names/emails — that would make cloud
# enrichment useless. Tighten per-index via config if needed.
DEFAULT_POLICY: Dict[str, str] = {SECRET: BLOCK, PII: ALLOW}


# --- detectors ------------------------------------------------------------


@dataclass(frozen=True)
class _Detector:
    name: str
    category: str
    pattern: "re.Pattern[str]"
    validator: Optional[Callable[[str], bool]] = None


def _luhn_ok(number: str) -> bool:
    """Luhn checksum — gates credit-card matches so long IDs don't false-fire."""
    digits = [int(c) for c in number if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total, parity = 0, len(digits) % 2
    for i, d in enumerate(digits):
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


# Order matters only for readability; scanning runs them all.
_DETECTORS: List[_Detector] = [
    # --- secrets (block by default) ---
    _Detector(
        "private_key_block",
        SECRET,
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----"),
    ),
    _Detector("aws_access_key_id", SECRET, re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    _Detector("anthropic_key", SECRET, re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}")),
    _Detector("openai_key", SECRET, re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{20,}")),
    _Detector("groq_key", SECRET, re.compile(r"\bgsk_[A-Za-z0-9]{20,}")),
    _Detector(
        "github_token",
        SECRET,
        re.compile(
            r"\b(?:ghp|gho|ghs|ghu)_[A-Za-z0-9]{36}\b|\bgithub_pat_[A-Za-z0-9_]{40,}"
        ),
    ),
    _Detector("slack_token", SECRET, re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}")),
    _Detector("google_api_key", SECRET, re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    _Detector(
        "jwt",
        SECRET,
        re.compile(r"\beyJ[A-Za-z0-9_\-]+\.eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+"),
    ),
    _Detector(
        "authorization_bearer",
        SECRET,
        re.compile(r"(?i)authorization\s*:\s*bearer\s+[A-Za-z0-9._\-]+"),
    ),
    _Detector(
        "assigned_secret",
        SECRET,
        re.compile(
            r"(?i)(?:api[_-]?key|secret|password|passwd|access[_-]?token)"
            r"\s*[:=]\s*['\"]?[A-Za-z0-9/+_\-]{12,}"
        ),
    ),
    # --- PII (allowed by default; configurable to redact/block) ---
    _Detector(
        "email",
        PII,
        re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    ),
    _Detector("us_ssn", PII, re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    _Detector(
        "credit_card", PII, re.compile(r"\b(?:\d[ \-]?){13,19}\b"), validator=_luhn_ok
    ),
]


@dataclass(frozen=True)
class Finding:
    detector: str
    category: str
    start: int
    end: int


def scan_text(text: str) -> List[Finding]:
    """All policy-relevant matches in `text`, as (type, category, span) — never
    the matched value. Overlapping matches from different detectors are kept;
    redaction handles overlaps left-to-right."""
    if not text:
        return []
    findings: List[Finding] = []
    for det in _DETECTORS:
        for m in det.pattern.finditer(text):
            if det.validator is not None and not det.validator(m.group(0)):
                continue
            findings.append(Finding(det.name, det.category, m.start(), m.end()))
    return findings


def redact_text(text: str, findings: List[Finding]) -> str:
    """Replace each finding's span with a typed placeholder, so content can be
    sent on with the sensitive substring masked. Non-overlapping by construction
    (later spans are skipped if they fall inside an already-redacted region)."""
    if not findings:
        return text
    spans = sorted(findings, key=lambda f: (f.start, -f.end))
    out: List[str] = []
    cursor = 0
    for f in spans:
        if f.start < cursor:  # overlaps a region already redacted
            continue
        out.append(text[cursor : f.start])
        out.append(f"[REDACTED:{f.detector}]")
        cursor = f.end
    out.append(text[cursor:])
    return "".join(out)


# --- policy ---------------------------------------------------------------


def normalize_policy(raw: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """Coerce a config-supplied policy into a known-good action per category.

    Fail-closed: an unrecognized action resolves to BLOCK rather than silently
    allowing egress. Categories absent from `raw` fall back to DEFAULT_POLICY.
    """
    policy = dict(DEFAULT_POLICY)
    if isinstance(raw, dict):
        for category in DEFAULT_POLICY:
            if category in raw:
                value = str(raw[category]).strip().lower()
                policy[category] = value if value in _VALID_ACTIONS else BLOCK
    return policy


def _decide(findings: List[Finding], policy: Dict[str, str]) -> str:
    """Strictest action triggered by any finding (block > redact > allow).

    Fail-closed: a finding whose category isn't in the policy resolves to BLOCK.
    """
    action = ALLOW
    for f in findings:
        category_action = policy.get(f.category, BLOCK)
        if _ACTION_RANK[category_action] > _ACTION_RANK[action]:
            action = category_action
    return action


@dataclass(frozen=True)
class PolicyDecision:
    action: str
    summary: Dict[str, int] = field(default_factory=dict)  # detector -> count
    redacted_text: Optional[str] = None

    @property
    def blocked(self) -> bool:
        return self.action == BLOCK


def evaluate(text: str, policy: Optional[Dict[str, Any]] = None) -> PolicyDecision:
    """Scan `text` and resolve the egress decision under `policy`."""
    resolved = normalize_policy(policy)
    findings = scan_text(text)
    if not findings:
        return PolicyDecision(ALLOW)
    summary: Dict[str, int] = {}
    for f in findings:
        summary[f.detector] = summary.get(f.detector, 0) + 1
    action = _decide(findings, resolved)
    redacted = redact_text(text, findings) if action == REDACT else None
    return PolicyDecision(action, summary, redacted)


# --- enforcement ----------------------------------------------------------


class PolicyGuardedEnricher:
    """Wraps a cloud enrichment client and enforces the egress policy before any
    request leaves the machine.

    - BLOCK  -> never call the cloud client; fall back to the local client if one
      was provided, else return {} (enrichment skipped, like a cloud error).
    - REDACT -> forward the prompt with sensitive substrings masked.
    - ALLOW  -> forward unchanged.

    Same duck-type as OllamaClient: generate_completion(model, prompt, **kw).
    """

    def __init__(
        self,
        inner: Any,
        *,
        local_fallback: Any = None,
        policy: Optional[Dict[str, Any]] = None,
        audit: Optional[Callable[[Dict[str, Any]], None]] = None,
        provider: str = "",
    ):
        self._inner = inner
        self._local = local_fallback
        self._policy = normalize_policy(policy)
        self._audit = audit
        self._provider = provider

    def _record(self, decision: PolicyDecision) -> None:
        if self._audit is None or decision.action == ALLOW:
            return
        # Counts and types only — never the matched secret value.
        try:
            self._audit(
                {
                    "provider": self._provider,
                    "action": decision.action,
                    "findings": decision.summary,
                }
            )
        except Exception:
            pass  # auditing must never break indexing

    def generate_completion(
        self, model: str, prompt: str, **kwargs: Any
    ) -> Dict[str, Any]:
        decision = evaluate(prompt, self._policy)
        self._record(decision)
        if decision.action == BLOCK:
            if self._local is not None:
                return self._local.generate_completion(model, prompt, **kwargs)
            return {}
        if decision.action == REDACT and decision.redacted_text is not None:
            return self._inner.generate_completion(
                model, decision.redacted_text, **kwargs
            )
        return self._inner.generate_completion(model, prompt, **kwargs)

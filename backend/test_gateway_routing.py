"""Unit tests for the gateway's deterministic routing gate (no HTTP, no LLM).

Covers `should_use_rag` / `is_smalltalk_or_meta` in `backend/server.py`, the
retrieval-first cascade that replaced the per-message enrichment-model router:

    force_rag → RAG · no indexes → direct · smalltalk/meta → direct · else RAG

Run it:

    .venv/bin/python backend/test_gateway_routing.py
"""

import os
import sys

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BACKEND_DIR)

from server import is_smalltalk_or_meta, should_use_rag  # noqa: E402

IDX = ["2fb7a91a-e7ae-46f5-93cd-c8ac52b3ea79"]
NO_IDX: list = []

# (message, expected_use_rag, why)
CASES = [
    # --- planted-fact questions from eval/smoke_e2e.py must reach retrieval ---
    ("What pressure does the brew boiler operate at during extraction?", True, "planted fact"),
    ("Which sensor part should be replaced when error code E11 appears?", True, "planted fact"),
    ("How long is the Atlas-7 parts warranty?", True, "planted fact"),
    ("Where is the serial number engraved?", True, "planted fact"),
    ("Summarize the service manual.", True, "document operation"),
    ("Who manufactures the Atlas-7?", True, "entity question"),
    ("9.2 bar?", True, "terse but factual"),
    ("descaling", True, "bare keyword, no core smalltalk phrase"),

    # --- the old defect: any message containing "test" went direct ---
    ("Which test procedure applies after replacing the pump?", True, "old defect: 'test'"),
    ("What is the test point voltage on the control board?", True, "old defect: 'test'"),
    ("Explain the E11 diagnostic test in detail.", True, "old defect: 'test'"),
    ("test", True, "bare 'test' is not an allowlisted smalltalk phrase"),
    ("Is this document checked and verified?", True, "old defect: 'check'"),
    ("How do I test the group head gasket?", True, "old defect: 'test'"),

    # --- smalltalk shortcuts ---
    ("hello", False, "greeting"),
    ("Hello!", False, "greeting, punctuated"),
    ("hi there", False, "greeting + filler"),
    ("hey", False, "greeting"),
    ("Good morning", False, "greeting"),
    ("how are you?", False, "greeting"),
    ("thanks!", False, "thanks"),
    ("Thank you so much", False, "thanks"),
    ("thx", False, "thanks"),
    ("bye", False, "farewell"),
    ("goodbye, see you later", False, "farewell"),
    ("ok", False, "acknowledgement"),
    ("got it, thanks", False, "acknowledgement + thanks"),
    ("nevermind", False, "acknowledgement"),
    ("", False, "empty message never needs retrieval"),
    ("   ", False, "whitespace-only"),

    # --- assistant-meta shortcuts ---
    ("who are you?", False, "meta"),
    ("Who are you", False, "meta"),
    ("what model are you", False, "meta"),
    ("what model are you?", False, "meta"),
    ("which model do you use?", False, "meta"),
    ("what are you?", False, "meta"),
    ("what is your name?", False, "meta"),
    ("are you an AI?", False, "meta"),
    ("who built you?", False, "meta"),
    ("what can you do?", False, "meta"),
    ("tell me about yourself", False, "meta"),

    # --- smalltalk words inside a real question must NOT shortcut ---
    ("Hello, what is the brew boiler pressure?", True, "greeting prefix + real question"),
    ("Thanks - now summarize section 4 for me", True, "thanks prefix + real question"),
    ("Who are the authors of the service manual?", True, "'who are' but not about the assistant"),
    ("What model number is the pressure sensor?", True, "'what model' but not about the assistant"),
    ("Is the machine ok to run at 1.45 bar?", True, "'ok' inside a real question"),
    ("no problem code is listed for E12?", True, "'no problem' inside a real question"),
    ("What is a good night mode setting?", True, "'good night' inside a real question"),
]


def check(label, actual, expected, why, kind="route"):
    ok = actual == expected
    fmt = (lambda v: ("RAG" if v else "DIRECT")) if kind == "route" else (lambda v: str(v))
    print(f"  {'PASS' if ok else 'FAIL'}  {label:<62} -> {fmt(actual):<6}"
          f" (expected {fmt(expected)}; {why})")
    return ok


def main():
    failures = 0
    total = 0

    print("\n[1] Session WITH linked indexes")
    for message, expected, why in CASES:
        total += 1
        label = repr(message) if len(message) < 60 else repr(message[:57] + "...")
        if not check(label, should_use_rag(message, IDX), expected, why):
            failures += 1

    print("\n[2] Session with NO linked indexes -> always direct")
    for message, _expected, _why in CASES:
        total += 1
        label = repr(message) if len(message) < 60 else repr(message[:57] + "...")
        if not check(label, should_use_rag(message, NO_IDX), False, "no indexes linked"):
            failures += 1
    for empty in (None, [], ()):
        total += 1
        if not check(f"idx_ids={empty!r}", should_use_rag("What is the brew pressure?", empty),
                     False, "no indexes linked"):
            failures += 1

    print("\n[3] force_rag is honored")
    force_cases = [
        ("hello", IDX),
        ("thanks!", IDX),
        ("who are you?", IDX),
        ("What pressure does the brew boiler operate at?", IDX),
        ("hello", NO_IDX),          # force_rag wins even with no indexes
        ("", NO_IDX),
    ]
    for message, idx in force_cases:
        total += 1
        if not check(f"force_rag {message!r} idx={bool(idx)}",
                     should_use_rag(message, idx, force_rag=True), True, "force_rag=True"):
            failures += 1

    print("\n[4] No LLM call is made by the gate")
    # server.should_use_rag must not reference an Ollama client at all: the gate
    # is a module-level function with no client argument and no network use.
    import inspect
    import server
    total += 1
    src = inspect.getsource(server.should_use_rag) + inspect.getsource(server.is_smalltalk_or_meta)
    clean = not any(tok in src for tok in ("ollama", "requests", "ENRICHMENT_MODEL", "http"))
    if not check("gate source is free of LLM/network calls", clean, True,
                 "deterministic gate", kind="bool"):
        failures += 1
    total += 1
    removed = not any(hasattr(server.ChatHandler, name) for name in
                      ("_should_use_rag", "_simple_pattern_routing",
                       "_route_using_overviews", "_load_document_overviews"))
    if not check("old LLM router + pattern fallback deleted", removed, True,
                 "no _simple_pattern_routing / _route_using_overviews", kind="bool"):
        failures += 1

    print("\n[5] is_smalltalk_or_meta is index-independent")
    for message, expected_rag, why in CASES:
        total += 1
        # With indexes linked, should_use_rag is exactly `not is_smalltalk_or_meta`.
        if not check(f"consistency {message[:40]!r}",
                     should_use_rag(message, IDX), not is_smalltalk_or_meta(message), why):
            failures += 1

    print(f"\n{total - failures}/{total} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

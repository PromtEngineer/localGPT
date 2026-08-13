"""Re-download the `rfc` corpus: 23 interlinked IETF RFCs (the QUIC / HTTP-3 family).

This corpus is **not** synthetic and **not** authored by this project. It exists to
answer one question the other corpora cannot: does the pipeline — and in
particular the index-time cross-reference extractor in
``rag_system/indexing/crossref.py`` — behave on documents whose naming and
referencing conventions we did not invent?

Every file is fetched verbatim from the RFC Editor's canonical plain-text
endpoint ``https://www.rfc-editor.org/rfc/rfcNNNN.txt``. Nothing is edited after
download; only the *filename* is ours, and that choice is deliberate — see
MANIFEST.md § "Naming".

    .venv/bin/python eval/corpora/rfc/download.py            # fetch anything missing
    .venv/bin/python eval/corpora/rfc/download.py --force    # re-fetch everything
    .venv/bin/python eval/corpora/rfc/download.py --check    # verify sizes only

Selection rule (enforced by ``--check``): every document in the set must
reference, or be referenced by, at least two others in the set. The check is
mechanical — it counts ``RFC NNNN`` / ``[RFCNNNN]`` mentions across the corpus.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import urllib.request
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_URL = "https://www.rfc-editor.org/rfc/rfc{num}.txt"

# (rfc number, filename title). The filename is
#   "RFC <num> - <Title>.txt"
# which is the convention a human filing these on disk would plausibly use, and
# it is what the cross-reference resolver is being tested against.
DOCUMENTS = [
    # --- normative boilerplate that literally every document below cites ---
    (2119, "Key Words for Use in RFCs to Indicate Requirement Levels"),
    (8174, "Ambiguity of Uppercase vs Lowercase in RFC 2119 Key Words"),
    (8126, "Guidelines for Writing an IANA Considerations Section in RFCs"),
    # --- TLS-side dependencies of QUIC and HTTP ---
    (6066, "TLS Extensions Extension Definitions"),
    (7301, "TLS Application-Layer Protocol Negotiation Extension"),
    # --- the QUIC core ---
    (8999, "Version-Independent Properties of QUIC"),
    (9000, "QUIC A UDP-Based Multiplexed and Secure Transport"),
    (9001, "Using TLS to Secure QUIC"),
    (9002, "QUIC Loss Detection and Congestion Control"),
    (9221, "An Unreliable Datagram Extension to QUIC"),
    (9369, "QUIC Version 2"),
    (9308, "Applicability of the QUIC Transport Protocol"),
    (9312, "Manageability of the QUIC Transport Protocol"),
    # --- the HTTP-over-QUIC layer ---
    (9114, "HTTP3"),
    (9204, "QPACK Field Compression for HTTP3"),
    (9218, "Extensible Prioritization Scheme for HTTP"),
    (9220, "Bootstrapping WebSockets with HTTP3"),
    (9297, "HTTP Datagrams and the Capsule Protocol"),
    (9298, "Proxying UDP in HTTP"),
    (9412, "The ORIGIN Extension in HTTP3"),
    # --- the HTTP/2 counterparts the HTTP/3 documents are defined against ---
    (8336, "The ORIGIN HTTP2 Frame"),
    (8441, "Bootstrapping WebSockets with HTTP2"),
    # --- a QUIC application protocol other than HTTP ---
    (9250, "DNS over Dedicated QUIC Connections"),
]

_MENTION_RE = re.compile(r"\bRFC\s?(\d{3,5})\b")


def filename(num: int, title: str) -> str:
    return f"RFC {num} - {title}.txt"


def path_for(num: int, title: str) -> str:
    return os.path.join(HERE, filename(num, title))


def fetch(num: int, title: str, force: bool) -> tuple:
    target = path_for(num, title)
    if os.path.exists(target) and not force:
        return ("cached", os.path.getsize(target))
    url = BASE_URL.format(num=num)
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = response.read()
    with open(target, "wb") as fh:
        fh.write(payload)
    return ("downloaded", len(payload))


def link_graph() -> dict:
    """{rfc number: set(other corpus rfc numbers it mentions in its text)}."""
    numbers = {num for num, _ in DOCUMENTS}
    graph = {}
    for num, title in DOCUMENTS:
        target = path_for(num, title)
        if not os.path.exists(target):
            continue
        with open(target, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        mentioned = {int(m) for m in _MENTION_RE.findall(text)}
        graph[num] = (mentioned & numbers) - {num}
    return graph


def check() -> int:
    graph = link_graph()
    missing = [n for n, _ in DOCUMENTS if n not in graph]
    if missing:
        print(f"MISSING files for: {missing}")
        return 1

    inbound = defaultdict(set)
    for source, targets in graph.items():
        for target in targets:
            inbound[target].add(source)

    total = 0
    failures = []
    print(f"{'RFC':>6}  {'bytes':>8}  {'->':>3} {'<-':>3}  degree")
    for num, title in DOCUMENTS:
        size = os.path.getsize(path_for(num, title))
        total += size
        out_degree = len(graph[num])
        in_degree = len(inbound[num])
        degree = len(graph[num] | inbound[num])
        print(f"{num:>6}  {size:>8}  {out_degree:>3} {in_degree:>3}  {degree}")
        if degree < 2:
            failures.append(num)
    print(f"\n{len(DOCUMENTS)} documents, {total} bytes ({total / 1024 / 1024:.2f} MiB)")
    edges = sum(len(v) for v in graph.values())
    print(f"{edges} directed intra-corpus RFC-number references")
    if failures:
        print(f"\nFAIL: {failures} are connected to fewer than 2 other documents")
        return 1
    print("every document is connected to at least 2 others in the set.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="re-download files that exist")
    parser.add_argument("--check", action="store_true", help="skip downloading; verify only")
    args = parser.parse_args()

    if not args.check:
        for num, title in DOCUMENTS:
            status, size = fetch(num, title, args.force)
            print(f"  {status:<11} {filename(num, title)}  ({size} bytes)")
    return check()


if __name__ == "__main__":
    sys.exit(main())

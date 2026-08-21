# `rfc` — 23 interlinked IETF RFCs (the QUIC / HTTP-3 family)

**Real, third-party, plain-text documents the pipeline has never seen.** Every
other corpus in `eval/corpora/` is either synthetic (`atlas7`, `hr`,
`acquisition`) or this project's own writing (`docs`). This one is neither: the
files are byte-for-byte what the RFC Editor serves, written by people who never
heard of localGPT, and their naming and cross-referencing conventions are
therefore *not ones we invented*. That is the whole point of the corpus — it is
the first honest test of the index-time cross-reference extractor
(`rag_system/indexing/crossref.py`) and of the chunker on documents whose
structure we did not choose.

* **23 documents, 1,511,267 bytes (1.44 MiB).**
* Source: `https://www.rfc-editor.org/rfc/rfcNNNN.txt` — the canonical
  plain-text rendering. Nothing is edited after download.
* Reproduce with `.venv/bin/python eval/corpora/rfc/download.py`
  (`--check` re-verifies sizes and the link graph without downloading).
* Answer-bearing anchors: `rfc.facts.json` (26 facts). Gold set:
  `eval/goldset/rfc.jsonl` (24 rows). Row-level gate:
  `.venv/bin/python eval/verify_rfc_goldset.py`.

## Selection rule

The cluster is the QUIC / HTTP-3 protocol family plus the documents it is
defined against. **Every file references, or is referenced by, at least two
others in the set** — checked mechanically by `download.py --check`, which
counts `RFC NNNN` / `[RFCNNNN]` mentions across the corpus. There are **110
directed intra-corpus references**; the lowest-degree document touches 4 others.

Three sub-clusters, which is what makes the cross-reference gold rows possible:

1. **QUIC core** (8999, 9000, 9001, 9002, 9221, 9369, 9308, 9312) — the
   transport, its TLS binding, its loss recovery, its version invariants. These
   defer to each other constantly: 9000 hands the Retry Integrity Tag to 9001
   and the probe timeout to 9002; 9002 hands the anti-amplification limit back
   to 9000; 9369 redefines 9001's key-derivation constants.
2. **HTTP over QUIC** (9114, 9204, 9218, 9220, 9297, 9298, 9412) plus their
   **HTTP/2 counterparts** (8336, 8441). Each HTTP/3 document is deliberately
   thin and defers its semantics to the HTTP/2 document it mirrors: 9220 reuses
   8441's `:protocol` pseudo-header, 9412 reuses 8336's ORIGIN payload.
3. **Shared normative dependencies** (2119, 8174, 8126, 6066, 7301) — BCP 14,
   the IANA registration policies, and the two TLS extensions the QUIC
   documents build on.

## Files

`->` is the number of other corpus documents this file cites; `<-` is the number
that cite it.

| File | Bytes | `->` | `<-` | Cites (RFC numbers in this corpus) |
|---|---:|---:|---:|---|
| `RFC 2119 - Key Words for Use in RFCs to Indicate Requirement Levels.txt` | 4,723 | 0 | 20 | — |
| `RFC 8174 - Ambiguity of Uppercase vs Lowercase in RFC 2119 Key Words.txt` | 6,071 | 1 | 16 | 2119 |
| `RFC 8126 - Guidelines for Writing an IANA Considerations Section in RFCs.txt` | 109,907 | 1 | 5 | 2119 |
| `RFC 6066 - TLS Extensions Extension Definitions.txt` | 55,079 | 1 | 3 | 2119 |
| `RFC 7301 - TLS Application-Layer Protocol Negotiation Extension.txt` | 17,439 | 1 | 7 | 2119 |
| `RFC 8999 - Version-Independent Properties of QUIC.txt` | 17,393 | 4 | 4 | 2119, 8174, 9000, 9001 |
| `RFC 9000 - QUIC A UDP-Based Multiplexed and Secure Transport.txt` | 403,442 | 7 | 14 | 2119, 7301, 8126, 8174, 8999, 9001, 9002 |
| `RFC 9001 - Using TLS to Secure QUIC.txt` | 126,175 | 5 | 8 | 2119, 7301, 8174, 9000, 9002 |
| `RFC 9002 - QUIC Loss Detection and Congestion Control.txt` | 89,071 | 4 | 7 | 2119, 8174, 9000, 9001 |
| `RFC 9221 - An Unreliable Datagram Extension to QUIC.txt` | 18,624 | 5 | 3 | 2119, 8174, 9000, 9001, 9002 |
| `RFC 9369 - QUIC Version 2.txt` | 26,887 | 9 | 0 | 2119, 7301, 8174, 8999, 9000, 9001, 9002, 9114, 9250 |
| `RFC 9308 - Applicability of the QUIC Transport Protocol.txt` | 60,645 | 8 | 1 | 7301, 8999, 9000, 9001, 9114, 9218, 9221, 9312 |
| `RFC 9312 - Manageability of the QUIC Transport Protocol.txt` | 80,543 | 9 | 1 | 6066, 7301, 8999, 9000, 9001, 9002, 9114, 9250, 9308 |
| `RFC 9114 - HTTP3.txt` | 155,206 | 7 | 9 | 2119, 6066, 7301, 8126, 8174, 9000, 9204 |
| `RFC 9204 - QPACK Field Compression for HTTP3.txt` | 99,258 | 4 | 1 | 2119, 8174, 9000, 9114 |
| `RFC 9218 - Extensible Prioritization Scheme for HTTP.txt` | 53,974 | 6 | 2 | 2119, 8126, 8174, 9000, 9002, 9114 |
| `RFC 9220 - Bootstrapping WebSockets with HTTP3.txt` | 6,619 | 4 | 2 | 2119, 8174, 8441, 9114 |
| `RFC 9297 - HTTP Datagrams and the Capsule Protocol.txt` | 31,835 | 9 | 1 | 2119, 8126, 8174, 8441, 9000, 9114, 9218, 9220, 9221 |
| `RFC 9298 - Proxying UDP in HTTP.txt` | 37,023 | 8 | 0 | 2119, 8174, 8441, 9000, 9114, 9220, 9221, 9297 |
| `RFC 9412 - The ORIGIN Extension in HTTP3.txt` | 6,879 | 5 | 0 | 2119, 8174, 8336, 9000, 9114 |
| `RFC 8336 - The ORIGIN HTTP2 Frame.txt` | 22,168 | 3 | 1 | 2119, 6066, 8174 |
| `RFC 8441 - Bootstrapping WebSockets with HTTP2.txt` | 16,639 | 2 | 3 | 2119, 8174 |
| `RFC 9250 - DNS over Dedicated QUIC Connections.txt` | 65,667 | 7 | 2 | 2119, 7301, 8126, 8174, 9000, 9001, 9002 |

Every file's URL is `https://www.rfc-editor.org/rfc/rfc<NNNN>.txt` for the RFC
number in its filename; `download.py` is the authoritative list.

### Why each document is in the cluster

| RFC | Why it is here |
|---|---|
| 2119, 8174 | BCP 14. Cited by 20 and 16 of the other 22 files respectively — the corpus's shared boilerplate, and a deliberate hard negative: the phrase "MUST NOT" appears in all 23 documents. |
| 8126 | Defines the registration policies (`Specification Required`, `Expert Review`, `Standards Action`) that the QUIC and HTTP/3 IANA sections *name* without restating. Pure cross-reference material. |
| 6066, 7301 | The two TLS extensions the family builds on: SNI/`server_name` and ALPN. 9001, 9114 and 9250 name their constructs; only these two define them. |
| 8999 | The QUIC invariants. Its 0-255-byte connection-ID range deliberately contradicts 9000's version-1 20-byte cap, which makes version-qualified questions discriminative. |
| 9000 | The hub: cited by 14 of the other 22 files. Also the largest document at 403 kB. |
| 9001 | QUIC's TLS binding. Holds the key-derivation constants 9369 replaces and the Retry Integrity Tag 9000 defers to. |
| 9002 | QUIC loss detection. Two-way deferral with 9000 (PTO ↔ anti-amplification). |
| 9221 | The DATAGRAM extension; the transport layer under 9297's HTTP Datagrams. |
| 9369 | QUIC v2. Defined almost entirely as a diff against 9000/9001/8999 — the densest out-degree in the corpus (9). |
| 9308, 9312 | Applicability and manageability. Cite nearly everything and define nothing, so they are the corpus's "pointer-heavy" documents. |
| 9114 | HTTP/3. Second hub, cited by 9 others. |
| 9204 | QPACK. Its two settings ride in 9114's SETTINGS frame. |
| 9218 | Extensible priorities, in both the HTTP/2 and HTTP/3 spellings. |
| 9220, 8441 | WebSockets over HTTP/3 and over HTTP/2. 9220 is 6.6 kB and reuses 8441's `:protocol` pseudo-header wholesale — the cleanest thin-document/definition pair in the corpus. |
| 9412, 8336 | The ORIGIN extension for HTTP/3 and the ORIGIN frame for HTTP/2. Same pattern as the pair above. |
| 9297, 9298 | HTTP Datagrams / Capsule Protocol and Proxying UDP in HTTP. 9298 defers its data-stream format to 9297. |
| 9250 | DNS over QUIC — a QUIC application protocol that is *not* HTTP, so the corpus is not one protocol stack repeated. |

## Budget, and what was excluded

The budget was ~1.5 MB of text so indexing stays tractable. Six documents that
belong to this family on the merits were left out because of it, and their
absence is a real limitation of the corpus, not a neutral choice:

| Excluded | Bytes | Consequence |
|---|---:|---|
| RFC 9110 (HTTP Semantics) | 502,941 | The most-cited document in the HTTP/3 sub-cluster. 9114, 9111, 9112, 9204 and 9218 all defer core semantics to it; those deferrals are now dangling. |
| RFC 8446 (TLS 1.3) | 337,736 | RFC 9001's principal normative dependency. Cross-reference rows about QUIC packet protection therefore anchor on 7301/6066 instead of on the TLS 1.3 handshake itself. |
| RFC 9113 (HTTP/2) | 191,811 | 8336, 8441 and 9218's HTTP/2 halves refer to it. |
| RFC 6455 (WebSocket) | 162,067 | Referenced by 8441 and 9220. |
| RFC 7541 (HPACK) | 117,827 | QPACK's predecessor, cited by 9204. |
| RFC 9111 / 9112 (HTTP caching, HTTP/1.1) | 84,477 / 109,913 | Same family, dropped for budget. |

RFC 5234 (ABNF) was in an earlier draft and removed: `download.py --check`
showed it at **degree 0** — no other document in the selected set cites it,
because the family's ABNF references all go through RFC 9110/9112, which are
excluded. RFC 8126 replaced it.

## Naming

Files are named `RFC <number> - <Title>.txt`. This is deliberate and it is part
of what is being measured. `rag_system/indexing/crossref.py` resolves a document
mention by searching the corpus text for the document's *whole normalised
filename*, so this scheme resolves a reference only if some document literally
contains the string "RFC 9000 QUIC A UDP Based Multiplexed and Secure
Transport". None does. Measured on the built index: **731 references extracted,
0 resolved.** The alternative schemes were measured too, on the same chunks —
`RFC 9000.txt` resolves 91 mentions across 18 documents, and
`<Title> - RFC 9000.txt` (the word order the RFC references section actually
uses) resolves 53 across 12. (Those figures are the post-fix ones, over the full
683-chunk index; on the pre-fix half-indexed corpus they were 731 / 0, 91 and
35 respectively — the resolution rate was 0 either way.)

The corpus keeps the plausible-human naming rather than the naming that scores
best, because the finding is the point: see the shakedown report for the full
breakdown. Post-fix, on the fully indexed corpus, the number is **0 of 1403**.

## What this corpus found on its first run

Indexing these 23 files with product defaults originally produced a LanceDB
table holding only **52% of the corpus's whitespace-normalised characters**:
every document over ~10,000 markdown tokens retained 45-57%, every document
under it retained ~100%. The cause was a segment-dropping bug in
`MarkdownRecursiveChunker._split_text`, which only fires on documents large
enough to need splitting — which is why no earlier corpus here exposed it (only
`Documentation/design_rationale.md` crosses the threshold, and it was at 49%
retained). **10 of the 24 gold rows failed gate 2** as a direct result.

That bug was fixed in `rag_system/ingestion/chunking.py` on 2026-08-13. Post-fix,
measured on the same corpus: character retention **1.02** (the >1 is the
chunker's one-sentence overlap), 683 chunks instead of 387, and **gate 2 passes
24/24**. Both states are recorded in the shakedown report.

The other finding is **not** fixed and is a property of the extractor rather
than of this corpus: index-time cross-reference extraction resolves **0 of 1403**
references here (see *Naming* above). Treat that as the corpus's standing
purpose — it is the only corpus in `eval/` whose reference conventions the
project did not author.

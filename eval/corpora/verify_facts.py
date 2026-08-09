"""Assert every planted-fact 'expected' string really occurs in its source document.

This is the first of the two gold-set verification gates. It checks the *source
text*; ``eval/run_eval.py --coverage-only`` checks the second gate — that the
string survives conversion and chunking into at least one indexed chunk.

    .venv/bin/python eval/corpora/verify_facts.py
"""

import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def normalise(text: str) -> str:
    return " ".join(text.split())


def pdf_text(path: str) -> str:
    import pymupdf

    doc = pymupdf.open(path)
    try:
        return " ".join(page.get_text() for page in doc)
    finally:
        doc.close()


def source_text_for(sidecar: dict) -> dict:
    """Return {source_label: normalised_text} for one sidecar file."""
    corpus = sidecar["corpus"]
    if corpus == "docs":
        texts = {}
        for path in sorted(glob.glob(os.path.join(REPO_ROOT, sidecar["source_glob"]))):
            with open(path, "r", encoding="utf-8") as fh:
                texts[os.path.basename(path)] = normalise(fh.read())
        return texts
    path = os.path.join(HERE, sidecar["document"])
    return {sidecar["document"]: normalise(pdf_text(path))}


def main() -> int:
    failures = 0
    total = 0
    for sidecar_path in sorted(glob.glob(os.path.join(HERE, "*.facts.json"))):
        with open(sidecar_path, "r", encoding="utf-8") as fh:
            sidecar = json.load(fh)
        texts = source_text_for(sidecar)
        joined = " || ".join(texts.values())
        print(f"\n{os.path.basename(sidecar_path)} — corpus '{sidecar['corpus']}', "
              f"{len(sidecar['facts'])} facts over {len(texts)} source file(s)")
        for fact in sidecar["facts"]:
            total += 1
            expected = normalise(fact["expected"])
            named_source = fact.get("source")
            haystack = texts.get(named_source, joined) if named_source else joined
            if expected not in haystack:
                failures += 1
                print(f"  MISSING {fact['id']}: {expected!r}"
                      f"{' in ' + named_source if named_source else ''}")
    print(f"\n{total - failures}/{total} planted facts verified present in their source document.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

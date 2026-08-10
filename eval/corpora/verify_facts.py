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
    """Return {source_label: normalised_text} for one sidecar file.

    Three shapes, keyed off which field the sidecar carries rather than off the
    corpus name, so a new corpus needs no edit here:

    * ``source_glob``  — markdown files under the repo root (the ``docs`` corpus)
    * ``documents_dir`` — every PDF in a directory next to this script (``acq``)
    * ``document``     — one PDF next to this script (``atlas7``, ``hr``)
    """
    if "source_glob" in sidecar:
        texts = {}
        for path in sorted(glob.glob(os.path.join(REPO_ROOT, sidecar["source_glob"]))):
            with open(path, "r", encoding="utf-8") as fh:
                texts[os.path.basename(path)] = normalise(fh.read())
        return texts
    if "documents_dir" in sidecar:
        directory = os.path.join(HERE, sidecar["documents_dir"])
        return {os.path.basename(p): normalise(pdf_text(p))
                for p in sorted(glob.glob(os.path.join(directory, "*.pdf")))}
    path = os.path.join(HERE, sidecar["document"])
    return {sidecar["document"]: normalise(pdf_text(path))}


def check_cross_references(sidecar: dict, texts: dict) -> int:
    """Every declared cross-reference cue must really occur in its 'from' document.

    The ``acq`` corpus exists so roadmap items 4.2/4.3 have something to hop
    across; a cue that is not literally in the source would make the corpus lie
    about its own link graph. ``to: null`` marks a deliberately dangling
    reference (the referenced document is not in the corpus) — the cue is still
    checked, the target is not.
    """
    refs = sidecar.get("cross_references") or []
    if not refs:
        return 0
    failures = 0
    dangling = 0
    for ref in refs:
        cue = normalise(ref["cue"])
        source = ref["from"]
        if source not in texts:
            failures += 1
            print(f"  MISSING xref source document {source!r}")
            continue
        if cue not in texts[source]:
            failures += 1
            print(f"  MISSING xref cue in {source}: {cue!r}")
        target = ref.get("to")
        if target is None:
            dangling += 1
        elif target not in texts:
            failures += 1
            print(f"  MISSING xref target document {target!r} (from {source})")
    print(f"  {len(refs) - failures}/{len(refs)} cross-reference cues verified "
          f"({dangling} deliberately dangling)")
    return failures


def main() -> int:
    failures = 0
    total = 0
    xref_failures = 0
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
        xref_failures += check_cross_references(sidecar, texts)
    print(f"\n{total - failures}/{total} planted facts verified present in their source document.")
    if xref_failures:
        print(f"{xref_failures} cross-reference cue(s) NOT found in their source document.")
    return 1 if (failures or xref_failures) else 0


if __name__ == "__main__":
    sys.exit(main())

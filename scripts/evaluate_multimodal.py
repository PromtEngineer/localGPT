#!/usr/bin/env python3
"""Generate fixtures and evaluate parser plus multimodal retrieval matrices."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_system.evaluation.multimodal_harness import (  # noqa: E402
    DEFAULT_MANIFEST,
    generate_fixture_corpus,
    load_manifest,
    model_matrix,
    report_failed,
    run_parser_matrix,
    run_retrieval_matrix,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("parser", "retrieval", "all"), default="all")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--fixture-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--token")
    parser.add_argument("--generation-model", default="qwen3:0.6b")
    parser.add_argument("--embedding-model", action="append", dest="embedding_models")
    parser.add_argument("--vision-model", action="append", dest="vision_models")
    parser.add_argument("--parser-backend", action="append", dest="parser_backends")
    parser.add_argument("--parser", action="append", dest="parsers")
    parser.add_argument("--include-parser-text", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--require-parsers", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    temporary: tempfile.TemporaryDirectory[str] | None = None
    if args.fixture_dir:
        fixture_dir = args.fixture_dir.resolve()
    else:
        temporary = tempfile.TemporaryDirectory(prefix="localgpt-multimodal-")
        fixture_dir = Path(temporary.name)

    try:
        fixtures = generate_fixture_corpus(fixture_dir)
        report: dict[str, object] = {
            "manifest_version": manifest["version"],
            "fixture_dir": str(fixture_dir),
            "fixtures": {name: str(path) for name, path in fixtures.items()},
        }
        parser_backends = args.parser_backends or ["docling"]
        if args.mode in {"parser", "all"}:
            report["parsers"] = run_parser_matrix(
                fixtures,
                manifest,
                args.parsers or ["localgpt", "liteparse", "docling"],
                fixture_dir / "parser-output",
                include_text=args.include_parser_text,
            )
        if args.mode in {"retrieval", "all"}:
            matrix = model_matrix(
                args.embedding_models or ["qwen3-embedding:0.6b"],
                args.vision_models or ["qwen2.5vl:3b"],
                parser_backends,
            )
            report["model_matrix"] = matrix
            report["retrieval"] = run_retrieval_matrix(
                base_url=args.base_url,
                ollama_url=args.ollama_url,
                token=args.token,
                fixtures=fixtures,
                manifest=manifest,
                generation_model=args.generation_model,
                matrix=matrix,
            )

        rendered = json.dumps(report, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        if args.strict and report_failed(report, require_parsers=args.require_parsers):
            raise SystemExit(1)
    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    main()

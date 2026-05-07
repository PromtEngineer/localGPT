from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

from rag_system.pipelines.indexing_pipeline import convert_and_chunk_document


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python -m rag_system.tools.convert_chunk_worker <input.json>", file=sys.stderr)
        return 2

    input_path = Path(sys.argv[1])
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        output_path = Path(payload["output_path"])
        result = convert_and_chunk_document(
            payload["file_path"],
            payload["document_id"],
            payload["config"],
        )
        output_path.write_text(json.dumps(result, default=str), encoding="utf-8")
        return 0
    except Exception as e:
        try:
            output_path = Path(payload["output_path"])  # type: ignore[name-defined]
            output_path.write_text(
                json.dumps({"error": str(e), "traceback": traceback.format_exc()}, default=str),
                encoding="utf-8",
            )
        except Exception:
            print(traceback.format_exc(), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

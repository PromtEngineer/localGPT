# Multimodal parser research and retrieval design

Research date: 2026-07-15. Versions and capabilities below are intentionally
date-stamped because both projects are changing quickly.

## Recommendation

Use a tiered parser architecture instead of choosing one parser globally:

1. Run LiteParse as the fast local PDF first pass. Preserve its page text,
   bounding boxes, screenshots, embedded images, and per-page complexity
   verdicts.
2. Route pages marked complex, and documents whose type LiteParse cannot handle
   natively, to Docling's standard pipeline.
3. Escalate diagrams, charts, dense tables, formulas, handwriting, and other
   visually meaningful pages to a Docling VLM or picture-enrichment path.
4. Normalize every backend into one internal page/region contract. Retrieval
   code must not depend on LiteParse or Docling object types.

This gives the common case a small, fast, local path while retaining a deeper
path for the pages where visual structure determines the answer. LiteParse's
own documentation says its heuristic Markdown reconstruction is weaker on
dense tables, multi-column documents, charts, handwriting, and scanned PDFs;
those are routing signals, not edge cases to ignore.

## Current LiteParse capabilities

The current Python, Node, Docker, and CLI release is 2.6.0. The project is
Apache-2.0 licensed.

- Rust/PDFium spatial PDF text extraction with text bounding boxes.
- Local, bundled Tesseract OCR; optional EasyOCR, PaddleOCR, or custom HTTP OCR.
- A cheap `is-complex` pass with per-page reasons including `scanned`,
  `no-text`, `sparse-text`, `embedded-images`, `garbled`, and `vector-text`.
- Page screenshot rendering at configurable DPI.
- JSON, text, and heuristic Markdown output, including embedded-image extraction.
- Python, Node/TypeScript, Rust, CLI, and browser/WASM surfaces.
- Office conversion through LibreOffice and image conversion through
  ImageMagick before the PDF parsing path.
- In 2.6.0, complexity signals can optionally be included in parse JSON.

Validation note: the installed Python 2.6.0 wheel exposed `--complexity` on
`lit parse`, but its top-level CLI did not expose the separately documented
`lit is-complex` subcommand. The harness therefore reads inline complexity
signals. Treat the standalone command as a packaging/version compatibility
check rather than assuming it is always present.

Useful LocalGPT roles:

- fast PDF text-layer extraction;
- complexity/cost routing before expensive OCR or VLM work;
- page screenshots for page-image embeddings and VLM context;
- bounding boxes for page-aware citations and visual grounding;
- an air-gapped baseline with few model dependencies.

It should not be treated as the sole multimodal parser. OCR recovers characters,
but it does not reliably encode arrow direction, chart semantics, table
relationships, or figure meaning.

Sources: [LiteParse repository and CLI reference](https://github.com/run-llama/liteparse),
[LiteParse releases](https://github.com/run-llama/liteparse/releases).

## Current Docling capabilities

The current release is 2.113.0. The codebase is MIT licensed, while individual
model licenses still need separate review.

- A unified `DoclingDocument` with structure, pages, bounding boxes, and
  provenance.
- Inputs including PDF, DOCX, XLSX, PPTX, ODF, EPUB, Markdown, AsciiDoc, LaTeX,
  HTML, CSV, images, audio/video transcription, WebVTT, and multiple XML/document
  schemas. Outputs include lossless JSON, Markdown, HTML, text, DocTags, chunks,
  and archives with page images.
- A standard PDF pipeline combining parsing, OCR, layout analysis, and table
  structure extraction.
- A full-page VLM pipeline with local and API engines, including Ollama-compatible
  presets. Current documented presets include Granite-Docling, SmolDocling,
  Qwen2.5-VL, Granite Vision, Pixtral, and others.
- Page and picture image generation, picture classification, and local or remote
  picture descriptions.
- Chart-data extraction, code/formula enrichment, and table-aware output. Release
  2.113.0 also adds native PowerPoint chart parsing as classified pictures with
  data.
- Native hierarchical, hybrid, and line-based token chunkers; the hybrid chunker
  can repeat table headers in split chunks.
- CLI, Python API, asynchronous service API, visual-grounding examples, and RAG
  integrations.

Useful LocalGPT roles:

- the canonical structure-preserving conversion path for Office and complex PDF;
- table, formula, chart, picture, and layout enrichment;
- page/picture images and provenance for multimodal indexing;
- VLM escalation through the same local Ollama deployment where compatible;
- structure-aware chunks that do not discard table headers or hierarchy.

Costs and caveats:

- its standard and VLM pipelines have much larger dependencies and startup/model
  costs than LiteParse;
- VLM extras and some accelerators are platform-specific;
- enabling remote picture-description endpoints can violate the local-only
  expectation unless explicitly configured and surfaced to the user;
- model output still needs deterministic provenance and evaluation rather than
  being trusted as ground truth.

Sources: [Docling overview](https://docling-project.github.io/docling/),
[supported formats](https://docling-project.github.io/docling/usage/supported_formats/),
[CLI and VLM options](https://docling-project.github.io/docling/reference/cli/),
[chunking](https://docling-project.github.io/docling/concepts/chunking/),
[model catalog](https://docling-project.github.io/docling/usage/model_catalog/),
[2.113.0 release](https://github.com/docling-project/docling/releases/tag/v2.113.0).

## Internal contract

The ingestion boundary should emit a parser-independent representation similar
to:

```text
ParsedDocument
  document_id
  parser_backend + parser_version
  pages[]
    page_number, width, height, screenshot_uri
    text
    regions[]
      region_id
      kind: text | table | picture | chart | formula | code
      bbox
      text
      caption_or_description
      image_uri
      confidence
      provenance
```

Every derived text description must keep a link back to its source page/region
image. A citation is only “visual” when it includes document, page, region or
bounding box, and the image artifact used for the answer.

## Retrieval design

Index two related evidence channels:

- Text channel: native text, OCR, table serialization, headings, and VLM/image
  descriptions embedded with the selected text embedding model.
- Visual channel: page crops, pictures, charts, and diagrams embedded with the
  selected visual embedding model.

Fuse independently ranked text and visual results, initially with weighted
reciprocal-rank fusion. Keep modality scores separate in debug output. For final
answering, pass the top relevant page/region images plus their associated text
to a vision-capable generation model. Do not send every page image; use retrieval
and a configurable image/token budget.

For multi-document questions, decompose only when necessary, retain evidence
from every subquery, then require the synthesizer to cite all documents needed
for the conclusion.

## Harness and acceptance criteria

`scripts/evaluate_multimodal.py` generates five deterministic documents:

- one born-digital PDF with a real text layer;
- one scanned, image-only PDF for OCR;
- one image-only topology diagram whose answer depends on arrow direction;
- one image-only bar chart whose answer depends on comparing values;
- one Markdown registry used with the diagram for cross-document synthesis.

The parser matrix records extraction text, page count, images, bounding boxes,
screenshots, complexity signals, errors, versions, and latency. The retrieval
matrix is the full Cartesian product of requested parser, text-embedding, and
vision-model configurations. Before indexing, it sends the scanned page as an
actual image to each configured Ollama vision model and requires that model to
read `EMBER-73`; it also sends a real probe to every text embedding model.

A strict retrieval case passes only when:

1. the answer contains the expected facts;
2. every required source document is cited;
3. visual cases include page-level visual provenance rather than an OCR-only
   citation; and
4. the entire configuration completes successfully.

This deliberately prevents OCR of labels from being reported as multimodal
understanding.

Example parser run:

```bash
pip install -r requirements-evaluation.txt
python scripts/evaluate_multimodal.py \
  --mode parser \
  --parser liteparse \
  --parser docling \
  --output data/evals/multimodal-parsers.json
```

Example end-to-end matrix with the backend and RAG services running:

```bash
python scripts/evaluate_multimodal.py \
  --mode all \
  --embedding-model qwen3-embedding:0.6b \
  --embedding-model nomic-embed-text \
  --vision-model qwen2.5vl:3b \
  --parser-backend liteparse \
  --parser-backend docling \
  --strict \
  --require-parsers \
  --output data/evals/multimodal-retrieval.json
```

Until the active LocalGPT indexing and retrieval pipelines implement the
internal contract and visual channel above, the strict visual cases are expected
to fail. That failure is the harness correctly identifying a product gap.

## Observed LocalGPT run (2026-07-15)

The generated corpus was run through the actual backend, RAG service, Ollama,
Docling conversion, LanceDB indexing, session/index linking, durable run API,
answer synthesis, citation collection, and cleanup.

- LiteParse 2.6.0: passed native text, scanned OCR, chart labels/values,
  screenshots, complexity routing, and spatial output. It missed three of four
  topology labels, so the strict diagram extraction check failed.
- Docling 2.113.0 standard pipeline: passed native text and scanned OCR and
  produced page images/provenance. It represented the topology and chart as
  pictures rather than extracting the facts needed by the questions, confirming
  that standard conversion alone is not VLM understanding.
- `gemma3:12b`: received the actual scanned page image through Ollama and read
  `EMBER-73` correctly.
- `qwen3-embedding:0.6b` and `bge-m3:latest`: each returned a real 1024-dimensional
  embedding and each completed a separate five-document/five-question run.
- Both end-to-end runs processed all 5 documents and indexed 5 chunks. Both
  passed born-digital retrieval. Both answered scanned OCR correctly but failed
  the deliberate visual-provenance rule. Both failed diagram reasoning, chart
  reasoning, and the diagram-plus-registry synthesis case.

The run also exposed and fixed two non-multimodal service defects: the backend
and RAG service were resolving session index scope from different SQLite files,
and macOS ingestion selected OCRMac when only Docling's option class—not the
OCRMac runtime—was installed. The backend now sends its authoritative table
scope explicitly, and conversion falls back to Docling's available OCR engine.

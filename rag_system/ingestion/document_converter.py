"""Document conversion with dependency-light native parsers and optional Docling.

Textual formats are deliberately handled without importing ML/OCR dependencies.
Docling is loaded lazily for layout-oriented formats such as PDF and Office files.
"""

from __future__ import annotations

import csv
import html.parser
import io
import json
import sys
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import Any, Callable


ConvertedPage = tuple[str, dict[str, Any]]


class _TextExtractor(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        value = data.strip()
        if value:
            self.parts.append(value)


class DocumentConverter:
    """Convert supported documents to Markdown plus provenance metadata."""

    SUPPORTED_FORMATS = {
        ".pdf",
        ".docx",
        ".pptx",
        ".xlsx",
        ".html",
        ".htm",
        ".md",
        ".txt",
        ".csv",
        ".json",
        ".eml",
    }

    def __init__(self) -> None:
        self.converter_no_ocr = None
        self.converter_ocr = None
        self.converter_general = None
        self._input_format: Any = None
        self._native: dict[str, Callable[[Path], list[ConvertedPage]]] = {
            ".txt": self._convert_text,
            ".md": self._convert_text,
            ".html": self._convert_html,
            ".htm": self._convert_html,
            ".csv": self._convert_csv,
            ".json": self._convert_json,
            ".eml": self._convert_email,
        }
        self._initialize_docling()

    def _initialize_docling(self) -> None:
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import (
                DocumentConverter as DoclingConverter,
                PdfFormatOption,
            )

            self._input_format = InputFormat
            no_ocr = PdfPipelineOptions()
            no_ocr.do_ocr = False
            self.converter_no_ocr = DoclingConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=no_ocr)}
            )

            ocr = PdfPipelineOptions()
            ocr.do_ocr = True
            # Full-page OCR is required for scans and image-only diagrams. Use
            # OCRMac only when its runtime is actually installed; importing the
            # Docling option class alone does not prove the engine is available.
            ocr.ocr_options.force_full_page_ocr = True
            if sys.platform == "darwin":
                try:
                    import ocrmac  # noqa: F401

                    from docling.datamodel.pipeline_options import OcrMacOptions

                    ocr.ocr_options = OcrMacOptions(force_full_page_ocr=True)
                except ImportError:
                    pass
            self.converter_ocr = DoclingConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=ocr)}
            )
            self.converter_general = DoclingConverter()
        except (ImportError, RuntimeError, ValueError) as exc:
            # Native formats remain fully available in minimal installations.
            self._docling_error = str(exc)

    def convert_to_markdown(self, file_path: str) -> list[ConvertedPage]:
        path = Path(file_path)
        extension = path.suffix.lower()
        if extension not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported file format: {extension}")
        native = self._native.get(extension)
        if native is not None:
            return native(path)
        if self.converter_general is None or self._input_format is None:
            raise RuntimeError(
                f"{extension} conversion requires the optional docling dependencies"
            )
        if extension == ".pdf":
            return self._convert_pdf(path)
        return self._perform_docling(path, self.converter_general)

    @staticmethod
    def _metadata(path: Path, **extra: Any) -> dict[str, Any]:
        return {"source": str(path), "filename": path.name, **extra}

    def _convert_text(self, path: Path) -> list[ConvertedPage]:
        content = path.read_text(encoding="utf-8", errors="replace")
        return [(content, self._metadata(path))]

    def _convert_html(self, path: Path) -> list[ConvertedPage]:
        parser = _TextExtractor()
        parser.feed(path.read_text(encoding="utf-8", errors="replace"))
        return [("\n\n".join(parser.parts), self._metadata(path))]

    def _convert_csv(self, path: Path) -> list[ConvertedPage]:
        content = path.read_text(encoding="utf-8-sig", errors="replace")
        rows = list(csv.reader(io.StringIO(content)))
        if not rows:
            return [("", self._metadata(path, rows=0))]
        width = max(len(row) for row in rows)
        normalized = [row + [""] * (width - len(row)) for row in rows]
        header = normalized[0]
        separator = ["---"] * width
        markdown = "\n".join(
            "| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |"
            for row in [header, separator, *normalized[1:]]
        )
        return [(markdown, self._metadata(path, rows=max(0, len(rows) - 1)))]

    def _convert_json(self, path: Path) -> list[ConvertedPage]:
        value = json.loads(path.read_text(encoding="utf-8"))
        return [(f"```json\n{json.dumps(value, indent=2, ensure_ascii=False)}\n```", self._metadata(path))]

    def _convert_email(self, path: Path) -> list[ConvertedPage]:
        message = BytesParser(policy=policy.default).parsebytes(path.read_bytes())
        body = message.get_body(preferencelist=("plain", "html")) if message.is_multipart() else message
        content = body.get_content() if body is not None else ""
        if body is not None and body.get_content_type() == "text/html":
            parser = _TextExtractor()
            parser.feed(str(content))
            content = "\n\n".join(parser.parts)
        markdown = (
            f"# {message.get('subject', '(no subject)')}\n\n"
            f"From: {message.get('from', '')}\n\n"
            f"To: {message.get('to', '')}\n\n{content}"
        )
        return [(markdown, self._metadata(path, message_id=message.get("message-id")))]

    def _convert_pdf(self, path: Path) -> list[ConvertedPage]:
        has_text = False
        try:
            import fitz

            with fitz.open(path) as document:
                has_text = any(page.get_text("text").strip() for page in document)
        except ImportError:
            pass
        converter = self.converter_no_ocr if has_text else self.converter_ocr
        return self._perform_docling(path, converter or self.converter_general)

    def _perform_docling(self, path: Path, converter: Any) -> list[ConvertedPage]:
        result = converter.convert(str(path))
        markdown = result.document.export_to_markdown()
        # The optional third item is retained for the structure-aware chunker.
        return [(markdown, self._metadata(path), result.document)]  # type: ignore[list-item]

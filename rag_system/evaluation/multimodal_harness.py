"""Deterministic fixtures and runners for multimodal retrieval evaluation.

The harness deliberately separates parser evaluation from full retrieval. A
parser can successfully extract text or screenshots while the product still
fails to index visual evidence, retrieve it, send it to a VLM, or cite it.
"""

from __future__ import annotations

import base64
import importlib.metadata
import itertools
import json
import math
import mimetypes
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol

import fitz
import requests
from PIL import Image, ImageDraw, ImageFont


DEFAULT_MANIFEST = Path(__file__).resolve().parents[2] / "evals" / "multimodal_retrieval.json"


def _font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _image_bytes(image: Image.Image, image_format: str = "PNG") -> bytes:
    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


def _write_image_pdf(path: Path, image: Image.Image) -> None:
    document = fitz.open()
    page = document.new_page(width=612, height=792)
    page.insert_image(page.rect, stream=_image_bytes(image))
    document.save(path)
    document.close()


def _write_text_pdf(path: Path) -> None:
    document = fitz.open()
    page = document.new_page(width=612, height=792)
    page.insert_text((54, 72), "Borealis Digital Operations Note", fontsize=18)
    page.insert_text(
        (54, 118),
        "The current born-digital validation phrase is AURORA-17.",
        fontsize=12,
    )
    page.insert_text(
        (54, 146),
        "This sentence is a real PDF text layer and does not require OCR.",
        fontsize=11,
    )
    document.save(path)
    document.close()


def _write_scanned_pdf(path: Path) -> None:
    image = Image.new("RGB", (1224, 1584), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((80, 90, 1140, 420), outline="#25324a", width=8)
    draw.text((125, 145), "SCANNED INCIDENT NOTICE", fill="#172033", font=_font(54))
    draw.text((125, 250), "Incident key: EMBER-73", fill="#991b1b", font=_font(48))
    draw.text(
        (125, 335),
        "This page contains pixels only; there is no PDF text layer.",
        fill="#172033",
        font=_font(30),
    )
    _write_image_pdf(path, image)


def _arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int]) -> None:
    draw.line((start, end), fill="#334155", width=10)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    length = 28
    spread = 0.55
    left = (
        end[0] - length * math.cos(angle - spread),
        end[1] - length * math.sin(angle - spread),
    )
    right = (
        end[0] - length * math.cos(angle + spread),
        end[1] - length * math.sin(angle + spread),
    )
    draw.polygon((end, left, right), fill="#334155")


def _write_topology_pdf(path: Path) -> None:
    image = Image.new("RGB", (1400, 900), "#f8fafc")
    draw = ImageDraw.Draw(image)
    title_font = _font(52)
    node_font = _font(54)
    draw.text((70, 55), "Processing topology", fill="#0f172a", font=title_font)
    boxes = {
        "Intake": (90, 220, 410, 360),
        "Archive": (90, 540, 410, 680),
        "Synthesis": (600, 370, 1010, 520),
        "Review": (1130, 370, 1350, 520),
    }
    for label, bounds in boxes.items():
        draw.rounded_rectangle(bounds, radius=24, fill="#dbeafe", outline="#1d4ed8", width=8)
        text_box = draw.textbbox((0, 0), label, font=node_font)
        width = text_box[2] - text_box[0]
        height = text_box[3] - text_box[1]
        x = (bounds[0] + bounds[2] - width) // 2
        y = (bounds[1] + bounds[3] - height) // 2
        draw.text((x, y), label, fill="#172554", font=node_font)
    _arrow(draw, (410, 290), (600, 410))
    _arrow(draw, (410, 610), (600, 480))
    _arrow(draw, (1010, 445), (1130, 445))
    draw.text(
        (70, 790),
        "Follow arrow direction to determine component relationships.",
        fill="#475569",
        font=_font(28),
    )
    _write_image_pdf(path, image)


def _write_chart_pdf(path: Path) -> None:
    image = Image.new("RGB", (1400, 900), "white")
    draw = ImageDraw.Draw(image)
    draw.text((70, 45), "Regional throughput", fill="#111827", font=_font(52))
    values = [
        ("North", 31, "#60a5fa"),
        ("South", 74, "#14b8a6"),
        ("East", 52, "#f59e0b"),
        ("West", 19, "#a78bfa"),
    ]
    baseline = 760
    scale = 7
    for index, (label, value, color) in enumerate(values):
        x1 = 120 + index * 300
        x2 = x1 + 170
        top = baseline - value * scale
        draw.rectangle((x1, top, x2, baseline), fill=color, outline="#1f2937", width=5)
        draw.text((x1 + 45, top - 52), str(value), fill="#111827", font=_font(34))
        draw.text((x1 + 15, baseline + 25), label, fill="#111827", font=_font(34))
    draw.line((80, baseline, 1320, baseline), fill="#111827", width=6)
    _write_image_pdf(path, image)


def generate_fixture_corpus(output_dir: Path) -> dict[str, Path]:
    """Create a small corpus whose hard questions require visual evidence."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fixtures = {
        "born_digital": output_dir / "01_born_digital.pdf",
        "scanned_notice": output_dir / "02_scanned_notice.pdf",
        "visual_topology": output_dir / "03_visual_topology.pdf",
        "visual_chart": output_dir / "04_visual_chart.pdf",
        "component_registry": output_dir / "05_component_registry.md",
    }
    _write_text_pdf(fixtures["born_digital"])
    _write_scanned_pdf(fixtures["scanned_notice"])
    _write_topology_pdf(fixtures["visual_topology"])
    _write_chart_pdf(fixtures["visual_chart"])
    fixtures["component_registry"].write_text(
        """# Component registry

The operational codename assigned to the Synthesis component is **KESTREL-42**.
The Intake component uses INLET-10, Archive uses VAULT-22, and Review uses REVIEW-08.
This registry does not describe topology; consult the topology diagram for arrows.
""",
        encoding="utf-8",
    )
    return fixtures


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class ParserResult:
    backend: str
    document: str
    status: str
    latency_ms: float
    version: str | None = None
    text: str = ""
    page_count: int | None = None
    extracted_image_count: int = 0
    bounding_box_count: int = 0
    screenshots: list[str] = field(default_factory=list)
    complexity: Any = None
    error: str | None = None

    def public_dict(self, *, include_text: bool = False) -> dict[str, Any]:
        value = asdict(self)
        if not include_text:
            value.pop("text", None)
        else:
            value["text"] = self.text[:10_000]
        return value


class ParserAdapter(Protocol):
    name: str

    def available(self) -> tuple[bool, str | None]: ...

    def parse(self, document: Path, work_dir: Path) -> ParserResult: ...


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


class LocalGPTParserAdapter:
    name = "localgpt"

    def __init__(self) -> None:
        self._converter: Any = None

    def available(self) -> tuple[bool, str | None]:
        return True, None

    def parse(self, document: Path, work_dir: Path) -> ParserResult:
        started = time.perf_counter()
        try:
            from rag_system.ingestion.document_converter import DocumentConverter

            if self._converter is None:
                self._converter = DocumentConverter()
            converted = self._converter.convert_to_markdown(str(document))
            text = "\n\n".join(str(item[0]) for item in converted)
            doc_objects = [item[2] for item in converted if len(item) > 2]
            page_count = None
            image_count = 0
            bbox_count = 0
            if doc_objects:
                doc_object = doc_objects[0]
                pages = getattr(doc_object, "pages", None)
                page_count = len(pages) if pages is not None else None
                image_count = len(getattr(doc_object, "pictures", []) or [])
                iterator = getattr(doc_object, "iterate_items", None)
                if callable(iterator):
                    for item, _level in iterator():
                        bbox_count += len(getattr(item, "prov", []) or [])
            return ParserResult(
                backend=self.name,
                document=document.name,
                status="passed",
                latency_ms=(time.perf_counter() - started) * 1000,
                version=_version("docling"),
                text=text,
                page_count=page_count,
                extracted_image_count=image_count,
                bounding_box_count=bbox_count,
            )
        except Exception as exc:
            return ParserResult(
                backend=self.name,
                document=document.name,
                status="failed",
                latency_ms=(time.perf_counter() - started) * 1000,
                version=_version("docling"),
                error=f"{type(exc).__name__}: {exc}",
            )


def _collect_json_text(value: Any) -> list[str]:
    output: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() in {"text", "markdown", "content", "value"} and isinstance(child, str):
                output.append(child)
            else:
                output.extend(_collect_json_text(child))
    elif isinstance(value, list):
        for child in value:
            output.extend(_collect_json_text(child))
    return output


def _count_spatial_items(value: Any) -> int:
    if isinstance(value, dict):
        count = int(
            all(key in value for key in ("x", "y", "width", "height"))
            or "bbox" in value
        )
        return count + sum(_count_spatial_items(child) for child in value.values())
    if isinstance(value, list):
        return sum(_count_spatial_items(child) for child in value)
    return 0


class LiteParseAdapter:
    name = "liteparse"

    @staticmethod
    def _executable() -> str | None:
        executable = shutil.which("lit")
        if executable:
            return executable
        sibling = Path(sys.executable).with_name("lit")
        return str(sibling) if sibling.is_file() else None

    def available(self) -> tuple[bool, str | None]:
        executable = self._executable()
        return (executable is not None, None if executable else "lit CLI is not installed")

    def parse(self, document: Path, work_dir: Path) -> ParserResult:
        started = time.perf_counter()
        available, reason = self.available()
        if not available:
            return ParserResult(
                backend=self.name,
                document=document.name,
                status="skipped",
                latency_ms=0,
                version=_version("liteparse"),
                error=reason,
            )
        document_dir = work_dir / self.name / document.stem
        image_dir = document_dir / "images"
        screenshot_dir = document_dir / "screenshots"
        document_dir.mkdir(parents=True, exist_ok=True)
        image_dir.mkdir(parents=True, exist_ok=True)
        output_path = document_dir / "parsed.json"
        try:
            executable = self._executable()
            if executable is None:
                raise RuntimeError("lit CLI disappeared after availability check")
            completed = subprocess.run(
                [
                    executable,
                    "parse",
                    str(document),
                    "--format",
                    "json",
                    "--image-mode",
                    "embed",
                    "--image-output-dir",
                    str(image_dir),
                    "--complexity",
                    "--output",
                    str(output_path),
                    "--quiet",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=300,
            )
            raw = output_path.read_text(encoding="utf-8") if output_path.exists() else completed.stdout
            parsed = json.loads(raw)
            text = "\n".join(_collect_json_text(parsed)) or raw
            complexity: Any = [
                page.get("complexity")
                for page in parsed.get("pages", [])
                if page.get("complexity") is not None
            ]
            subprocess.run(
                [
                    executable,
                    "screenshot",
                    str(document),
                    "--output-dir",
                    str(screenshot_dir),
                    "--quiet",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=300,
            )
            screenshots = sorted(str(path) for path in screenshot_dir.glob("*.png"))
            return ParserResult(
                backend=self.name,
                document=document.name,
                status="passed",
                latency_ms=(time.perf_counter() - started) * 1000,
                version=_version("liteparse"),
                text=text,
                page_count=len(screenshots) or None,
                extracted_image_count=len(list(image_dir.glob("*"))),
                bounding_box_count=_count_spatial_items(parsed),
                screenshots=screenshots,
                complexity=complexity,
            )
        except Exception as exc:
            return ParserResult(
                backend=self.name,
                document=document.name,
                status="failed",
                latency_ms=(time.perf_counter() - started) * 1000,
                version=_version("liteparse"),
                error=f"{type(exc).__name__}: {exc}",
            )


class DoclingAdapter:
    name = "docling"

    def __init__(self) -> None:
        self._converter: Any = None

    def available(self) -> tuple[bool, str | None]:
        try:
            import docling  # noqa: F401

            return True, None
        except ImportError:
            return False, "docling is not installed"

    def parse(self, document: Path, work_dir: Path) -> ParserResult:
        started = time.perf_counter()
        available, reason = self.available()
        if not available:
            return ParserResult(
                backend=self.name,
                document=document.name,
                status="skipped",
                latency_ms=0,
                version=_version("docling"),
                error=reason,
            )
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import (
                DocumentConverter,
                PdfFormatOption,
            )

            pipeline_options = PdfPipelineOptions()
            pipeline_options.generate_page_images = True
            pipeline_options.images_scale = 1.5
            pipeline_options.ocr_options.force_full_page_ocr = True
            if self._converter is None:
                self._converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(
                            pipeline_options=pipeline_options
                        )
                    }
                )
            converted = self._converter.convert(str(document)).document
            bbox_count = 0
            for item, _level in converted.iterate_items():
                bbox_count += len(getattr(item, "prov", []) or [])
            screenshot_dir = work_dir / self.name / document.stem / "screenshots"
            screenshot_dir.mkdir(parents=True, exist_ok=True)
            screenshots: list[str] = []
            for page_number, page in (getattr(converted, "pages", {}) or {}).items():
                image = getattr(page, "image", None)
                pil_image = getattr(image, "pil_image", None)
                if pil_image is not None:
                    screenshot = screenshot_dir / f"page_{page_number}.png"
                    pil_image.save(screenshot)
                    screenshots.append(str(screenshot))
            return ParserResult(
                backend=self.name,
                document=document.name,
                status="passed",
                latency_ms=(time.perf_counter() - started) * 1000,
                version=_version("docling"),
                text=converted.export_to_markdown(),
                page_count=len(getattr(converted, "pages", {}) or {}),
                extracted_image_count=len(getattr(converted, "pictures", []) or []),
                bounding_box_count=bbox_count,
                screenshots=screenshots,
            )
        except Exception as exc:
            return ParserResult(
                backend=self.name,
                document=document.name,
                status="failed",
                latency_ms=(time.perf_counter() - started) * 1000,
                version=_version("docling"),
                error=f"{type(exc).__name__}: {exc}",
            )


PARSER_ADAPTERS: dict[str, type[Any]] = {
    "localgpt": LocalGPTParserAdapter,
    "liteparse": LiteParseAdapter,
    "docling": DoclingAdapter,
}


def evaluate_parser_result(result: ParserResult, check: dict[str, Any]) -> dict[str, Any]:
    expected_tokens = [str(token) for token in check.get("expected_tokens", [])]
    missing_tokens = [token for token in expected_tokens if token.lower() not in result.text.lower()]
    requires_screenshot = bool(check.get("requires_screenshot"))
    requires_bbox = bool(check.get("requires_bounding_boxes"))
    passed = (
        result.status == "passed"
        and not missing_tokens
        and (not requires_screenshot or bool(result.screenshots))
        and (not requires_bbox or result.bounding_box_count > 0)
    )
    return {
        "backend": result.backend,
        "document": result.document,
        "name": check["name"],
        "status": result.status,
        "passed": passed,
        "missing_tokens": missing_tokens,
        "requires_screenshot": requires_screenshot,
        "requires_bounding_boxes": requires_bbox,
        "error": result.error,
    }


def run_parser_matrix(
    fixtures: dict[str, Path],
    manifest: dict[str, Any],
    parser_names: Iterable[str],
    work_dir: Path,
    *,
    include_text: bool = False,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    for parser_name in parser_names:
        adapter_type = PARSER_ADAPTERS.get(parser_name)
        if adapter_type is None:
            raise ValueError(f"Unknown parser backend: {parser_name}")
        adapter = adapter_type()
        for check in manifest["parser_checks"]:
            document = fixtures[check["fixture"]]
            result = adapter.parse(document, work_dir)
            results.append(result.public_dict(include_text=include_text))
            checks.append(evaluate_parser_result(result, check))
    return {"results": results, "checks": checks}


def model_matrix(
    embedding_models: Iterable[str],
    vision_models: Iterable[str],
    parser_backends: Iterable[str],
) -> list[dict[str, str]]:
    return [
        {"embedding_model": embedding, "vision_model": vision, "parser_backend": parser}
        for embedding, vision, parser in itertools.product(
            embedding_models, vision_models, parser_backends
        )
    ]


def _expect(response: requests.Response) -> dict[str, Any]:
    if not response.ok:
        raise RuntimeError(
            f"{response.request.method} {response.url}: {response.status_code} {response.text}"
        )
    return response.json()


def _wait_for_run(
    base_url: str, headers: dict[str, str], run_id: str, timeout: float = 900
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        run = _expect(
            requests.get(f"{base_url}/v1/runs/{run_id}", headers=headers, timeout=30)
        )
        if run["status"] in {"completed", "failed", "cancelled"}:
            return run
        time.sleep(0.2)
    raise TimeoutError(f"Run did not finish: {run_id}")


def _citation_document(citation: dict[str, Any]) -> str:
    return str(citation.get("document_id") or "")


def _has_visual_provenance(citation: dict[str, Any]) -> bool:
    modality = str(citation.get("modality") or "").lower()
    return modality in {"image", "page_image", "visual"} and citation.get("page") is not None


def probe_vision_model(
    *, ollama_url: str, model: str, scanned_document: Path
) -> dict[str, Any]:
    with fitz.open(scanned_document) as document:
        png = document[0].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).tobytes("png")
    started = time.perf_counter()
    response = requests.post(
        f"{ollama_url.rstrip('/')}/api/generate",
        json={
            "model": model,
            "prompt": (
                "Read the incident key in this image. Reply with only the key, "
                "including its hyphen."
            ),
            "images": [base64.b64encode(png).decode("ascii")],
            "stream": False,
            "options": {"temperature": 0},
        },
        timeout=600,
    )
    payload = _expect(response)
    answer = str(payload.get("response") or "").strip()
    if "EMBER-73" not in answer.upper():
        raise AssertionError(
            f"Vision model {model!r} did not read the image marker: {answer!r}"
        )
    return {
        "model": model,
        "answer": answer,
        "latency_ms": (time.perf_counter() - started) * 1000,
        "passed": True,
    }


def run_retrieval_matrix(
    *,
    base_url: str,
    ollama_url: str,
    token: str | None,
    fixtures: dict[str, Path],
    manifest: dict[str, Any],
    generation_model: str,
    matrix: list[dict[str, str]],
) -> list[dict[str, Any]]:
    base = base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    outputs: list[dict[str, Any]] = []
    _expect(requests.get(f"{base}/health", headers=headers, timeout=30))
    vision_probes: dict[str, dict[str, Any]] = {}

    for config in matrix:
        session_id: str | None = None
        index_id: str | None = None
        matrix_result: dict[str, Any] = {"config": config, "cases": []}
        try:
            vision_model = config["vision_model"]
            if vision_model not in vision_probes:
                vision_probes[vision_model] = probe_vision_model(
                    ollama_url=ollama_url,
                    model=vision_model,
                    scanned_document=fixtures["scanned_notice"],
                )
            matrix_result["vision_probe"] = vision_probes[vision_model]
            embedding = _expect(
                requests.post(
                    f"{base}/v1/embeddings",
                    headers=headers,
                    json={"model": config["embedding_model"], "input": "multimodal harness probe"},
                    timeout=600,
                )
            )
            vector = embedding["data"][0]["embedding"]
            matrix_result["embedding_dimensions"] = len(vector)
            if len(vector) < 32:
                raise AssertionError("Embedding vector is implausibly small")

            session_id = _expect(
                requests.post(
                    f"{base}/sessions",
                    headers=headers,
                    json={"title": "Multimodal harness", "model": generation_model},
                    timeout=30,
                )
            )["session_id"]
            index_id = _expect(
                requests.post(
                    f"{base}/indexes",
                    headers=headers,
                    json={
                        "name": f"multimodal-{int(time.time())}",
                        "description": "Ephemeral multimodal retrieval evaluation",
                        "options": {
                            "embedding_model": config["embedding_model"],
                            "enable_enrich": False,
                            "enable_docling_chunk": config["parser_backend"] == "docling",
                            "enable_latechunk": False,
                            "retrieval_mode": "hybrid",
                            "chunk_size": 256,
                            "chunk_overlap": 32,
                            "enable_multimodal": True,
                            "parser_backend": config["parser_backend"],
                            "vision_model": config["vision_model"],
                            "page_render_dpi": 160,
                        },
                    },
                    timeout=30,
                )
            )["index_id"]

            selected_names = list(manifest["index_documents"])
            with _FileStack() as stack:
                files = []
                for fixture_name in selected_names:
                    path = fixtures[fixture_name]
                    handle = stack.open(path)
                    files.append(
                        (
                            "files",
                            (
                                path.name,
                                handle,
                                mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                            ),
                        )
                    )
                _expect(
                    requests.post(
                        f"{base}/indexes/{index_id}/upload",
                        headers=headers,
                        files=files,
                        timeout=180,
                    )
                )
            built = _expect(
                requests.post(
                    f"{base}/indexes/{index_id}/build",
                    headers={**headers, "Content-Type": "application/json"},
                    json={},
                    timeout=1800,
                )
            )
            matrix_result["chunks_indexed"] = built.get("chunks_indexed")
            matrix_result["files_processed"] = built.get("files_processed")
            expected_files = len(selected_names)
            if built.get("files_processed") != expected_files:
                raise AssertionError(
                    "Index build did not process the complete corpus: "
                    f"expected {expected_files}, got {built.get('files_processed')}"
                )
            _expect(
                requests.post(
                    f"{base}/sessions/{session_id}/indexes/{index_id}",
                    headers=headers,
                    timeout=30,
                )
            )

            for case in manifest["retrieval_cases"]:
                submitted = _expect(
                    requests.post(
                        f"{base}/v1/runs",
                        headers=headers,
                        json={
                            "session_id": session_id,
                            "message": case["question"],
                            "model": generation_model,
                            "force_rag": True,
                            "retrieval_k": case.get("retrieval_k", 8),
                            "search_type": case.get("search_type", "hybrid"),
                            "query_decompose": case.get("query_decompose"),
                            "context_window_size": 0,
                        },
                        timeout=30,
                    )
                )
                run = _wait_for_run(base, headers, submitted["id"])
                result = run.get("result") or {}
                answer = str(result.get("content") or "")
                citations = list(result.get("citations") or [])
                missing_terms = [
                    term
                    for term in case["expected_terms"]
                    if str(term).lower() not in answer.lower()
                ]
                citation_documents = [_citation_document(item) for item in citations]
                missing_sources = [
                    fixtures[source].name
                    for source in case["expected_sources"]
                    if not any(
                        document.endswith(fixtures[source].name)
                        for document in citation_documents
                    )
                ]
                visual_provenance = any(_has_visual_provenance(item) for item in citations)
                requires_visual = bool(case.get("requires_visual_evidence"))
                case_result = {
                    "name": case["name"],
                    "status": run["status"],
                    "answer": answer,
                    "citation_documents": citation_documents,
                    "missing_terms": missing_terms,
                    "missing_sources": missing_sources,
                    "requires_visual_evidence": requires_visual,
                    "visual_provenance": visual_provenance,
                }
                case_result["passed"] = (
                    run["status"] == "completed"
                    and not missing_terms
                    and not missing_sources
                    and (not requires_visual or visual_provenance)
                )
                matrix_result["cases"].append(case_result)
            matrix_result["passed"] = all(case["passed"] for case in matrix_result["cases"])
        except Exception as exc:
            matrix_result["passed"] = False
            matrix_result["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if index_id:
                requests.delete(f"{base}/indexes/{index_id}", headers=headers, timeout=180)
            if session_id:
                requests.delete(f"{base}/sessions/{session_id}", headers=headers, timeout=30)
        outputs.append(matrix_result)
    return outputs


class _FileStack:
    def __init__(self) -> None:
        self._handles: list[Any] = []

    def __enter__(self) -> _FileStack:
        return self

    def open(self, path: Path):
        handle = path.open("rb")
        self._handles.append(handle)
        return handle

    def __exit__(self, *_args: Any) -> None:
        for handle in reversed(self._handles):
            handle.close()


def report_failed(report: dict[str, Any], *, require_parsers: bool = False) -> bool:
    parser_report = report.get("parsers") or {}
    parser_checks = parser_report.get("checks") or []
    if require_parsers and any(not check.get("passed") for check in parser_checks):
        return True
    retrieval = report.get("retrieval") or []
    return bool(retrieval) and any(not item.get("passed") for item in retrieval)

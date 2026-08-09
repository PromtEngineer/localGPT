from typing import List, Tuple, Dict, Any
import os
import platform

# torch.compile's inductor backend has no MPS support; docling's layout model
# calls it and crashes on Apple Silicon unless dynamo is disabled up front.
if platform.system() == "Darwin":
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from docling.document_converter import DocumentConverter as DoclingConverter, PdfFormatOption
from docling.datamodel import pipeline_options as docling_options
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
import fitz  # PyMuPDF for quick text inspection
import importlib.util
import shutil

# docling options class -> the module or binary its backend needs at runtime.
OCR_BACKENDS = [
    ("OcrMacOptions", "module", ("ocrmac",)),
    ("EasyOcrOptions", "module", ("easyocr",)),
    # rapidocr renamed its package: >=3.x installs as `rapidocr`, older
    # releases as `rapidocr_onnxruntime` — accept either.
    ("RapidOcrOptions", "module", ("rapidocr", "rapidocr_onnxruntime")),
    ("TesseractOcrOptions", "module", ("tesserocr",)),
    ("TesseractCliOcrOptions", "binary", ("tesseract",)),
]


def build_ocr_options():
    """Pick an OCR engine docling can actually run on this host.

    OcrMac is only tried on macOS; the remaining engines are tried in order and
    only if their backend is installed. Returns None when nothing is available,
    in which case docling's own default OCR settings are used.
    """
    for name, kind, dependencies in OCR_BACKENDS:
        if name == "OcrMacOptions" and platform.system() != "Darwin":
            continue
        options_cls = getattr(docling_options, name, None)
        if options_cls is None:
            continue
        if kind == "module" and not any(importlib.util.find_spec(d) for d in dependencies):
            continue
        if kind == "binary" and not any(shutil.which(d) for d in dependencies):
            continue
        try:
            kwargs = {"force_full_page_ocr": True}
            # docling's RapidOCR default is lang=['chinese']; pin an explicit
            # recognition language (OCR_LANG env, comma-separated, to override).
            if name == "RapidOcrOptions":
                kwargs["lang"] = [
                    l.strip() for l in os.getenv("OCR_LANG", "english").split(",") if l.strip()
                ]
            options = options_cls(**kwargs)
        except Exception as e:
            print(f"OCR engine {name} is not usable here: {e}")
            continue
        print(f"OCR engine: {name}")
        return options

    print("No OCR engine available; using docling's default OCR settings.")
    return None


class DocumentConverter:
    """
    A class to convert various document formats to structured Markdown using the docling library.
    Supports PDF, DOCX, HTML, and other formats.
    """
    
    # Mapping of file extensions to InputFormat
    SUPPORTED_FORMATS = {
        '.pdf': InputFormat.PDF,
        '.docx': InputFormat.DOCX,
        '.html': InputFormat.HTML,
        '.htm': InputFormat.HTML,
        '.md': InputFormat.MD,
        '.txt': 'TXT',  # Special handling for plain text files
    }
    
    def __init__(self):
        """Initializes one docling converter per path (no-OCR, OCR, general).

        Each converter is built independently so that a failure in one (typically
        the OCR engine) does not disable the others.
        """
        self.converter_no_ocr = self._build_pdf_converter(do_ocr=False)
        self.converter_ocr = self._build_pdf_converter(do_ocr=True)
        try:
            self.converter_general = DoclingConverter()
        except Exception as e:
            print(f"Error initializing general docling converter: {e}")
            self.converter_general = None

        available = [
            name for name, conv in (
                ("no-OCR", self.converter_no_ocr),
                ("OCR", self.converter_ocr),
                ("general", self.converter_general),
            ) if conv is not None
        ]
        print(f"docling DocumentConverter(s) initialized ({', '.join(available) or 'none'}).")

    @staticmethod
    def _build_pdf_converter(*, do_ocr: bool):
        try:
            pipeline = PdfPipelineOptions()
            pipeline.do_ocr = do_ocr
            if do_ocr:
                ocr_options = build_ocr_options()
                if ocr_options is not None:
                    pipeline.ocr_options = ocr_options
            return DoclingConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline)}
            )
        except Exception as e:
            print(f"Error initializing docling PDF converter (ocr={do_ocr}): {e}")
            return None

    def convert_to_markdown(self, file_path: str) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Converts a document to a single Markdown string, preserving layout and tables.
        Supports PDF, DOCX, HTML, and other formats.
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in self.SUPPORTED_FORMATS:
            print(f"Unsupported file format: {file_ext}")
            return []

        input_format = self.SUPPORTED_FORMATS[file_ext]

        if input_format == InputFormat.PDF:
            return self._convert_pdf_to_markdown(file_path)
        elif input_format == 'TXT':
            return self._convert_txt_to_markdown(file_path)
        else:
            return self._convert_general_to_markdown(file_path, input_format)

    def _convert_pdf_to_markdown(self, pdf_path: str) -> List[Tuple[str, Dict[str, Any]]]:
        """Convert PDF with OCR detection logic."""
        # Quick heuristic: if the PDF already contains a text layer, skip OCR for speed
        def _pdf_has_text(path: str) -> bool:
            try:
                doc = fitz.open(path)
                for page in doc:
                    if page.get_text("text").strip():
                        return True
            except Exception:
                pass
            return False

        use_ocr = not _pdf_has_text(pdf_path)
        if use_ocr and self.converter_ocr is None:
            print(f"{pdf_path} has no text layer but no OCR converter is available; trying without OCR.")
            use_ocr = False
        converter = self.converter_ocr if use_ocr else self.converter_no_ocr
        ocr_msg = "(OCR enabled)" if use_ocr else "(no OCR)"

        if converter is None:
            print(f"No docling PDF converter available. Skipping {pdf_path}.")
            return []

        print(f"Converting {pdf_path} to Markdown using docling {ocr_msg}...")
        return self._perform_conversion(pdf_path, converter, ocr_msg)

    def _convert_txt_to_markdown(self, file_path: str) -> List[Tuple[str, Dict[str, Any]]]:
        """Convert plain text files to markdown by reading content directly."""
        print(f"Converting {file_path} (TXT) to Markdown...")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            markdown_content = f"```\n{content}\n```"
            metadata = {"source": file_path}
            
            print(f"Successfully converted {file_path} (TXT) to Markdown.")
            return [(markdown_content, metadata)]
        except Exception as e:
            print(f"Error processing TXT file {file_path}: {e}")
            return []
    
    def _convert_general_to_markdown(self, file_path: str, input_format: InputFormat) -> List[Tuple[str, Dict[str, Any]]]:
        """Convert non-PDF formats using general converter."""
        if self.converter_general is None:
            print(f"General docling converter not available. Skipping {file_path}.")
            return []
        print(f"Converting {file_path} ({input_format.name}) to Markdown using docling...")
        return self._perform_conversion(file_path, self.converter_general, f"({input_format.name})")
    
    def _perform_conversion(self, file_path: str, converter, format_msg: str) -> List[Tuple[str, Dict[str, Any]]]:
        """Perform the actual conversion using the specified converter."""
        pages_data = []
        try:
            result = converter.convert(file_path)
            markdown_content = result.document.export_to_markdown()
            
            metadata = {"source": file_path}
            # Return the *DoclingDocument* object as third tuple element so downstream
            # chunkers that understand the element tree can use it.  Legacy callers that
            # expect only (markdown, metadata) can simply ignore the extra value.
            pages_data.append((markdown_content, metadata, result.document))
            print(f"Successfully converted {file_path} with docling {format_msg}.")
            return pages_data
        except Exception as e:
            print(f"Error processing {file_path} with docling: {e}")
            return []

from typing import List, Tuple, Dict, Any
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, OcrMacOptions
from docling.datamodel.base_models import InputFormat
import fitz  # PyMuPDF for quick text inspection

class PDFConverter:
    """
    A class to convert PDF files to structured Markdown using the docling library.
    """
    def __init__(self):
        """Initializes the docling document converter with forced OCR enabled for macOS."""
        try:
            # --- Converter WITHOUT OCR (fast path) ---
            pipeline_no_ocr = PdfPipelineOptions()
            pipeline_no_ocr.do_ocr = False
            format_no_ocr = {
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_no_ocr)
            }
            self.converter_no_ocr = DocumentConverter(format_options=format_no_ocr)

            # --- Converter WITH OCR (fallback) ---
            pipeline_ocr = PdfPipelineOptions()
            pipeline_ocr.do_ocr = True
            ocr_options = OcrMacOptions(force_full_page_ocr=True)
            pipeline_ocr.ocr_options = ocr_options
            format_ocr = {
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_ocr)
            }
            self.converter_ocr = DocumentConverter(format_options=format_ocr)

            print("docling DocumentConverter(s) initialized (OCR + no-OCR).")
        except Exception as e:
            print(f"Error initializing docling DocumentConverter(s): {e}")
            self.converter_no_ocr = None
            self.converter_ocr = None

    def convert_to_markdown(self, pdf_path: str) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Converts a PDF to a single Markdown string, preserving layout and tables.
        """
        if not (self.converter_no_ocr and self.converter_ocr):
            print("docling converters not available. Skipping conversion.")
            return []
            
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
        converter = self.converter_ocr if use_ocr else self.converter_no_ocr
        ocr_msg = "(OCR enabled)" if use_ocr else "(no OCR)"

        print(f"Converting {pdf_path} to Markdown using docling {ocr_msg}...")
        pages_data = []
        try:
            result = converter.convert(pdf_path)
            markdown_content = result.document.export_to_markdown()
            
            metadata = {"source": pdf_path}
            # Return the *DoclingDocument* object as third tuple element so downstream
            # chunkers that understand the element tree can use it.  Legacy callers that
            # expect only (markdown, metadata) can simply ignore the extra value.
            pages_data.append((markdown_content, metadata, result.document))
            print(f"Successfully converted {pdf_path} with docling {ocr_msg}.")
            return pages_data
        except Exception as e:
            print(f"Error processing PDF {pdf_path} with docling: {e}")
            return []

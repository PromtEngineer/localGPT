from typing import List, Tuple, Dict, Any
from docling.document_converter import DocumentConverter as DoclingConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, OcrMacOptions
from docling.datamodel.base_models import InputFormat
import fitz  # PyMuPDF for quick text inspection
import os

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
        """Initializes the docling document converter with forced OCR enabled for macOS."""
        try:
            # --- Converter WITHOUT OCR (fast path) ---
            pipeline_no_ocr = PdfPipelineOptions()
            pipeline_no_ocr.do_ocr = False
            format_no_ocr = {
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_no_ocr)
            }
            self.converter_no_ocr = DoclingConverter(format_options=format_no_ocr)

            # --- Converter WITH OCR (fallback) ---
            pipeline_ocr = PdfPipelineOptions()
            pipeline_ocr.do_ocr = True
            ocr_options = OcrMacOptions(force_full_page_ocr=True)
            pipeline_ocr.ocr_options = ocr_options
            format_ocr = {
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_ocr)
            }
            self.converter_ocr = DoclingConverter(format_options=format_ocr)
            
            self.converter_general = DoclingConverter()

            print("docling DocumentConverter(s) initialized (OCR + no-OCR + general).")
        except Exception as e:
            print(f"Error initializing docling DocumentConverter(s): {e}")
            self.converter_no_ocr = None
            self.converter_ocr = None
            self.converter_general = None

    def convert_to_markdown(self, file_path: str) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Converts a document to a single Markdown string, preserving layout and tables.
        Supports PDF, DOCX, HTML, and other formats.
        """
        if not (self.converter_no_ocr and self.converter_ocr and self.converter_general):
            print("docling converters not available. Skipping conversion.")
            return []
        
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
        converter = self.converter_ocr if use_ocr else self.converter_no_ocr
        ocr_msg = "(OCR enabled)" if use_ocr else "(no OCR)"

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

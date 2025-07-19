from __future__ import annotations

"""Docling-aware chunker (simplified).

For now we proxy the old MarkdownRecursiveChunker but add:
• sentence-aware packing to max_tokens with overlap
• breadcrumb metadata stubs so downstream code already handles them

In a follow-up we can replace the internals with true Docling element-tree
walking once the PDFConverter returns structured nodes.
"""
from typing import List, Dict, Any, Tuple
import math
import re
from itertools import islice
from rag_system.ingestion.chunking import MarkdownRecursiveChunker
from transformers import AutoTokenizer

class DoclingChunker:
    def __init__(self, *, max_tokens: int = 512, overlap: int = 1, tokenizer_model: str = "Qwen/Qwen3-Embedding-0.6B"):
        self.max_tokens = max_tokens
        self.overlap = overlap  # sentences of overlap
        repo_id = tokenizer_model
        if "/" not in tokenizer_model and not tokenizer_model.startswith("Qwen/"):
            repo_id = {
                "qwen3-embedding-0.6b": "Qwen/Qwen3-Embedding-0.6B",
            }.get(tokenizer_model.lower(), tokenizer_model)
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
        except Exception as e:
            print(f"Warning: Failed to load tokenizer {repo_id}: {e}")
            print("Falling back to character-based approximation (4 chars ≈ 1 token)")
            self.tokenizer = None
        # Fallback simple sentence splitter (period, question, exclamation, newline)
        self._sent_re = re.compile(r"(?<=[\.\!\?])\s+|\n+")
        self.legacy = MarkdownRecursiveChunker(max_chunk_size=10_000, min_chunk_size=100)

    # ------------------------------------------------------------------
    def _token_len(self, text: str) -> int:
        if self.tokenizer is not None:
            return len(self.tokenizer.tokenize(text))
        else:
            # Fallback: approximate 4 characters per token
            return max(1, len(text) // 4)

    def split_markdown(self, markdown: str, *, document_id: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split one Markdown doc into chunks with max_tokens limit."""
        base_chunks = self.legacy.chunk(markdown, document_id, metadata)
        new_chunks: List[Dict[str, Any]] = []
        global_idx = 0
        for ch in base_chunks:
            sentences = [s.strip() for s in self._sent_re.split(ch["text"]) if s.strip()]
            if not sentences:
                continue
            window: List[str] = []
            while sentences:
                # Add until over limit
                while sentences and self._token_len(" ".join(window + [sentences[0]])) <= self.max_tokens:
                    window.append(sentences.pop(0))
                if not window:  # single sentence > limit → hard cut
                    window.append(sentences.pop(0))
                chunk_text = " ".join(window)
                new_chunk = {
                    "chunk_id": f"{document_id}_{global_idx}",
                    "text": chunk_text,
                    "metadata": {
                        **metadata,
                        "document_id": document_id,
                        "chunk_index": global_idx,
                        "heading_path": metadata.get("heading_path", []),
                        "heading_level": len(metadata.get("heading_path", [])),
                        "block_type": metadata.get("block_type", "paragraph"),
                    },
                }
                new_chunks.append(new_chunk)
                global_idx += 1
                # Overlap: prepend last `overlap` sentences of the current window to the remaining queue
                if self.overlap and sentences:
                    back = window[-self.overlap:] if self.overlap <= len(window) else window[:]
                    sentences = back + sentences
                window = []
        return new_chunks

    # ------------------------------------------------------------------
    # Element-tree based chunking (true Docling path)
    # ------------------------------------------------------------------
    def chunk_document(self, doc, *, document_id: str, metadata: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
        """Walk a DoclingDocument and emit chunks.

        Tables / Code / Figures are emitted as atomic chunks.
        Paragraph-like nodes are sentence-packed to <= max_tokens.
        """
        metadata = metadata or {}

        def _token_len(txt: str) -> int:
            if self.tokenizer is not None:
                return len(self.tokenizer.tokenize(txt))
            else:
                # Fallback: approximate 4 characters per token
                return max(1, len(txt) // 4)

        chunks: List[Dict[str, Any]] = []
        global_idx = 0

        # Helper to create a chunk and append to list
        def _add_chunk(text: str, block_type: str, heading_path: List[str], page_no: int | None = None):
            nonlocal global_idx
            if not text.strip():
                return
            chunk_meta = {
                **metadata,
                "document_id": document_id,
                "chunk_index": global_idx,
                "heading_path": heading_path,
                "heading_level": len(heading_path),
                "block_type": block_type,
            }
            if page_no is not None:
                chunk_meta["page"] = page_no
            chunks.append({
                "chunk_id": f"{document_id}_{global_idx}",
                "text": text,
                "metadata": chunk_meta,
            })
            global_idx += 1

        # The Docling API exposes .body which is a tree of nodes; we fall back to .texts/.tables lists if available
        try:
            # We walk doc.texts (reading order). We'll buffer consecutive paragraph items
            current_heading_path: List[str] = []
            buffer: List[str] = []
            buffer_tokens = 0
            buffer_page = None

            def flush_buffer():
                nonlocal buffer, buffer_tokens, buffer_page
                if buffer:
                    _add_chunk(" ".join(buffer), "paragraph", heading_path=current_heading_path[:], page_no=buffer_page)
                buffer, buffer_tokens, buffer_page = [], 0, None

            # Create quick lookup for table items by id to preserve later insertion order if needed
            tables_by_anchor = {
                getattr(t, "anchor_text_id", None): t
                for t in getattr(doc, "tables", [])
                if getattr(t, "anchor_text_id", None) is not None
            }

            for txt_item in getattr(doc, "texts", []):
                # If this text item is a placeholder for a table anchor, emit table first
                anchor_id = getattr(txt_item, "id", None)
                if anchor_id in tables_by_anchor:
                    flush_buffer()
                    tbl = tables_by_anchor[anchor_id]
                    try:
                        tbl_md = tbl.export_to_markdown(doc)  # pass doc for deprecation compliance
                    except Exception:
                        tbl_md = tbl.export_to_markdown() if hasattr(tbl, "export_to_markdown") else str(tbl)
                    _add_chunk(tbl_md, "table", heading_path=current_heading_path[:], page_no=getattr(tbl, "page_no", None))

                role = getattr(txt_item, "role", None)
                if role == "heading":
                    flush_buffer()
                    level = getattr(txt_item, "level", 1)
                    current_heading_path = current_heading_path[: max(0, level - 1)]
                    current_heading_path.append(txt_item.text.strip())
                    continue  # skip heading as content

                text_piece = txt_item.text if hasattr(txt_item, "text") else str(txt_item)
                piece_tokens = _token_len(text_piece)
                if piece_tokens > self.max_tokens:  # very long paragraph
                    flush_buffer()
                    _add_chunk(text_piece, "paragraph", heading_path=current_heading_path[:], page_no=getattr(txt_item, "page_no", None))
                    continue

                if buffer_tokens + piece_tokens > self.max_tokens:
                    flush_buffer()

                buffer.append(text_piece)
                buffer_tokens += piece_tokens
                if buffer_page is None:
                    buffer_page = getattr(txt_item, "page_no", None)

            flush_buffer()

            # Emit any remaining tables that were not anchored
            for tbl in getattr(doc, "tables", []):
                if tbl in tables_by_anchor.values():
                    continue  # already emitted
                try:
                    tbl_md = tbl.export_to_markdown(doc)
                except Exception:
                    tbl_md = tbl.export_to_markdown() if hasattr(tbl, "export_to_markdown") else str(tbl)
                _add_chunk(tbl_md, "table", heading_path=current_heading_path[:], page_no=getattr(tbl, "page_no", None))
        except Exception as e:
            print(f"⚠️  Docling tree walk failed: {e}. Falling back to markdown splitter.")
            return self.split_markdown(doc.export_to_markdown(), document_id=document_id, metadata=metadata)

        # --------------------------------------------------------------
        # Second-pass consolidation: merge small consecutive paragraph
        # chunks that share heading & page into up-to-max_tokens blobs.
        # --------------------------------------------------------------
        consolidated: List[Dict[str, Any]] = []
        buf_txt: List[str] = []
        buf_meta: Dict[str, Any] | None = None

        def flush_paragraph_buffer():
            nonlocal buf_txt, buf_meta
            if not buf_txt:
                return
            merged_text = " ".join(buf_txt)
            # Re-use meta from first piece but update chunk_id later
            new_chunk = {
                "chunk_id": buf_meta["chunk_id"],
                "text": merged_text,
                "metadata": buf_meta["metadata"],
            }
            consolidated.append(new_chunk)
            buf_txt = []
            buf_meta = None

        for ch in chunks:
            if ch["metadata"].get("block_type") != "paragraph":
                flush_paragraph_buffer()
                consolidated.append(ch)
                continue

            if not buf_txt:
                buf_txt.append(ch["text"])
                buf_meta = ch
                continue

            same_page = ch["metadata"].get("page") == buf_meta["metadata"].get("page")
            same_heading = ch["metadata"].get("heading_path") == buf_meta["metadata"].get("heading_path")

            prospective_len = self._token_len(" ".join(buf_txt + [ch["text"]]))
            if same_page and same_heading and prospective_len <= self.max_tokens:
                buf_txt.append(ch["text"])
            else:
                flush_paragraph_buffer()
                buf_txt.append(ch["text"])
                buf_meta = ch

        flush_paragraph_buffer()

        return consolidated

    # Public API expected by IndexingPipeline --------------------------------
    def chunk(self, text: str, document_id: str, document_metadata: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
        return self.split_markdown(text, document_id=document_id, metadata=document_metadata or {})    
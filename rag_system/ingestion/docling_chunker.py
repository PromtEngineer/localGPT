from __future__ import annotations

"""Docling-aware chunker.

Two entry points:
• chunk_document(doc) walks a DoclingDocument element tree, emitting tables and
  code as atomic chunks and token-packing paragraphs up to max_tokens.
• chunk()/split_markdown() fall back to MarkdownRecursiveChunker plus
  sentence-aware packing when only Markdown is available.

Both attach heading-path / block-type metadata to every chunk.
"""
from typing import List, Dict, Any
import re
from rag_system.ingestion.chunking import MarkdownRecursiveChunker
from transformers import AutoTokenizer

class DoclingChunker:
    def __init__(self, *, max_tokens: int = 512, overlap: int = 1, tokenizer_model: str | None = None):
        self.max_tokens = max_tokens
        self.overlap = overlap  # sentences of overlap

        if not tokenizer_model:
            from rag_system.main import EXTERNAL_MODELS
            tokenizer_model = EXTERNAL_MODELS["embedding_model"]

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, trust_remote_code=True)
        except Exception as e:
            print(f"Warning: Failed to load tokenizer {tokenizer_model}: {e}")
            print("Falling back to character-based approximation (4 chars ≈ 1 token)")
            self.tokenizer = None
        # Fallback simple sentence splitter (period, question, exclamation, newline)
        self._sent_re = re.compile(r"(?<=[\.\!\?])\s+|\n+")
        # Markdown fallback chunker, built lazily on first use so the common
        # Docling path doesn't pay for a second tokenizer load.
        self._legacy: MarkdownRecursiveChunker | None = None
        self._legacy_tokenizer_model = tokenizer_model

    @property
    def legacy(self) -> MarkdownRecursiveChunker:
        if self._legacy is None:
            self._legacy = MarkdownRecursiveChunker(
                max_chunk_size=10_000, min_chunk_size=100,
                tokenizer_model=self._legacy_tokenizer_model,
            )
        return self._legacy

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
            # Index-based window over the sentence list. Each iteration emits
            # sentences[i:j] and the next window starts strictly after i, so
            # the loop always makes progress and cannot re-emit the same
            # window forever (the old queue-prepend version could).
            i = 0
            while i < len(sentences):
                # Grow the window [i..j) until the next sentence would exceed the limit
                j = i
                while j < len(sentences) and self._token_len(" ".join(sentences[i:j + 1])) <= self.max_tokens:
                    j += 1
                if j == i:  # single sentence > limit → hard cut
                    j = i + 1
                chunk_text = " ".join(sentences[i:j])
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
                if j >= len(sentences):
                    break
                # Overlap: restart up to `overlap` sentences back, but never at
                # or before i — the window must strictly advance.
                i = max(j - self.overlap, i + 1) if self.overlap else j
        return new_chunks

    # ------------------------------------------------------------------
    # Element-tree based chunking (true Docling path)
    # ------------------------------------------------------------------
    def chunk_document(self, doc, *, document_id: str, metadata: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
        """Walk a DoclingDocument and emit chunks.

        Tables are emitted as atomic chunks, inline in reading order.
        Section headers update the heading path and are not emitted.
        Text-less items (pictures, groups) are skipped; everything with
        text is token-packed up to max_tokens.
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

        # Walk the document with docling's iterate_items(), which yields
        # (item, level) in true reading order — tables included inline.
        # Attributes are read through getattr so unknown item types degrade
        # gracefully; anything unexpected falls back to the markdown splitter.
        try:
            current_heading_path: List[str] = []
            buffer: List[str] = []
            buffer_tokens = 0
            buffer_page = None

            def flush_buffer():
                nonlocal buffer, buffer_tokens, buffer_page
                if buffer:
                    _add_chunk(" ".join(buffer), "paragraph", heading_path=current_heading_path[:], page_no=buffer_page)
                buffer, buffer_tokens, buffer_page = [], 0, None

            def _page_of(item) -> int | None:
                prov = getattr(item, "prov", None) or []
                return getattr(prov[0], "page_no", None) if prov else None

            def _emit_table(tbl) -> None:
                try:
                    tbl_md = tbl.export_to_markdown(doc)  # pass doc for deprecation compliance
                except Exception:
                    tbl_md = tbl.export_to_markdown() if hasattr(tbl, "export_to_markdown") else str(tbl)
                _add_chunk(tbl_md, "table", heading_path=current_heading_path[:], page_no=_page_of(tbl))

            for item, _level in doc.iterate_items():
                label = getattr(item, "label", None)
                label_value = getattr(label, "value", label)

                # Tables are atomic chunks, emitted where they appear in the flow
                if label_value == "table":
                    flush_buffer()
                    _emit_table(item)
                    continue

                # Section headings update the heading path; they are not content
                if label_value == "section_header":
                    flush_buffer()
                    level = getattr(item, "level", 1) or 1
                    current_heading_path = current_heading_path[: max(0, level - 1)]
                    current_heading_path.append((getattr(item, "text", "") or "").strip())
                    continue  # skip heading as content

                text_piece = getattr(item, "text", None)
                if not text_piece:
                    continue  # pictures, groups and other text-less items
                piece_tokens = _token_len(text_piece)
                if piece_tokens > self.max_tokens:  # very long paragraph
                    flush_buffer()
                    _add_chunk(text_piece, "paragraph", heading_path=current_heading_path[:], page_no=_page_of(item))
                    continue

                if buffer_tokens + piece_tokens > self.max_tokens:
                    flush_buffer()

                buffer.append(text_piece)
                buffer_tokens += piece_tokens
                if buffer_page is None:
                    buffer_page = _page_of(item)

            flush_buffer()
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
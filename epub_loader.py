"""Loads an epub file into a list of documents."""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

from epub2txt import epub2txt
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from loguru import logger


@dataclass
class EpubLoader(BaseLoader):
    """Load an epub file into a list of documents.

    Args:
        file_path: file path or url to epub
    Returns:
        self.load() -> list of Documents
    """
    file_path: Union[str, Path]

    def load(self) -> List[Document]:
        """Load data into document objects."""
        try:
            texts = epub2txt(self.file_path, outputlist=True)
            ch_titles = epub2txt.content_titles

        except Exception as exc:
            logger.error(exc)
            raise

        docs = []
        for title, text in zip(ch_titles, texts):
            metadata = {"source": self.file_path, "ch.": title}
            docs.append(Document(page_content=text, metadata=metadata))

        return docs

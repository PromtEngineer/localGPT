# localGPT/registry.py
from collections import defaultdict
from typing import Optional, Text

import magic
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

from localGPT import MIME_TYPES


class LoaderRegistry:
    def __init__(self):
        self.loader_map = defaultdict(list)

        # MIME_TYPES: tuple[tuple[str, BaseLoader]]
        for mime_type, loader in MIME_TYPES:
            self.register_loader(mime_type, loader)

    @staticmethod
    def get_mime_type(file_path: str) -> Text:
        mime = magic.Magic(mime=True)
        return mime.from_file(file_path)

    @staticmethod
    def has_extension(document: Document, source: str) -> bool:
        return document.metadata["source"].endswith(source)

    def register_loader(self, mime_type: str, loader_class: BaseLoader) -> None:
        self.loader_map[mime_type].append(loader_class)

    def get_loader(self, mime_type: str) -> Optional[BaseLoader]:
        loader_classes = self.loader_map.get(mime_type)
        if loader_classes:
            return loader_classes[0]  # Return the first matching loader class
        else:
            return None

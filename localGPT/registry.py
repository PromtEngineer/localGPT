# localGPT/registry.py
import os
from collections import defaultdict
from typing import Optional, Text, Type

import magic
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

from localGPT import LANGUAGE_TYPES, MIME_TYPES


class LoaderRegistry:
    """
    A registry for loaders based on MIME types.
    """

    def __init__(self):
        """
        Initializes the LoaderRegistry.
        """
        self.loader_map = defaultdict()

        # Register loaders for MIME_TYPES
        # MIME_TYPES: Tuple[Tuple[str, Type[BaseLoader]], ...]
        for mime_type, loader in MIME_TYPES:
            self.register_loader(mime_type, loader)

    @staticmethod
    def get_mime_type(file_path: str) -> Text:
        """
        Returns the MIME type of a file based on its path.

        Args:
            file_path (str): The path of the file.

        Returns:
            Text: The MIME type of the file.
        """
        mime = magic.Magic(mime=True)
        return mime.from_file(file_path)

    def register_loader(
        self,
        mime_type: str,
        loader_class: Type[BaseLoader],
    ) -> None:
        """
        Registers a loader for a specific MIME type.

        Args:
            mime_type (str): The MIME type to register the loader for.
            loader_class (Type[BaseLoader]): The loader class to register.
        """
        if mime_type in self.loader_map.keys():
            raise ValueError(f"A loader for MIME type {mime_type} is already registered.")
        self.loader_map[mime_type] = loader_class

    def get_loader(self, mime_type: str) -> Optional[Type[BaseLoader]]:
        """
        Returns the loader class for a specific MIME type.

        Args:
            mime_type (str): The MIME type to retrieve the loader for.

        Returns:
            Optional[Type[BaseLoader]]: The loader class if found, None otherwise.
        """
        loader_class = self.loader_map.get(mime_type)
        if loader_class:
            return loader_class  # Return the first matching loader class
        else:
            return None


class TextSplitterRegistry:
    """
    A registry for languages based on file extensions.
    """

    def __init__(self):
        """
        Initializes the TextSplitterRegistry.
        """
        self.language_map = defaultdict()

        # Register languages for file extensions
        for file_extension, language in LANGUAGE_TYPES:
            self.register_language(file_extension, language)

    @staticmethod
    def get_extension(document: Document) -> str:
        """
        Checks if the document has a specific source extension.

        Args:
            document (Document): The document to check.
            source (str): The source extension to compare.

        Returns:
            bool: True if the document has the specified source extension,
            False otherwise.
        """
        return os.path.splitext(document.metadata["source"])[1][1:]  # Get file extension without dot

    def register_language(
        self,
        file_extension: str,
        language: str,
    ) -> None:
        """
        Registers a language for a specific file extension.

        Args:
            file_extension (str): The file extension to register the language for.
            language (str): The language to register.
        """
        self.language_map[file_extension] = language

    def get_language(
        self,
        file_extension: str,
    ) -> Optional[str]:
        """
        Returns the language for a specific file extension.

        Args:
            file_extension (str): The file extension to retrieve the language for.

        Returns:
            Optional[str]: The language if found, None otherwise.
        """
        return self.language_map.get(file_extension)

from collections import defaultdict

import magic
from langchain.document_loaders import (
    CSVLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredExcelLoader,
)


class LoaderRegistry:
    def __init__(self):
        self.loader_map = defaultdict(list)

    @staticmethod
    def get_mime_type(file_path):
        mime = magic.Magic(mime=True)
        return mime.from_file(file_path)

    def register_loader(self, mime_type, loader_class):
        self.loader_map[mime_type].append(loader_class)

    def get_loader(self, mime_type):
        loader_classes = self.loader_map.get(mime_type)
        if loader_classes:
            return loader_classes[0]  # Return the first matching loader class
        else:
            return None


# Create an instance of the LoaderRegistry
loader_registry = LoaderRegistry()

# Register the Loader classes with the LoaderRegistry using MIME types
loader_registry.register_loader("text/plain", TextLoader)
loader_registry.register_loader("application/pdf", PDFMinerLoader)
loader_registry.register_loader("text/csv", CSVLoader)
loader_registry.register_loader("application/vnd.ms-excel", UnstructuredExcelLoader)

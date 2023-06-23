from collections import defaultdict

import magic
from localGPT import MIME_TYPES


class LoaderRegistry:
    def __init__(self):
        self.loader_map = defaultdict(list)

        for mime_type, loader in MIME_TYPES:
            self.register_loader(mime_type, loader)

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

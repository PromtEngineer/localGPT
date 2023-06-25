# localGPT/database.py
from typing import Optional, List
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.vectorstores.base import VectorStoreRetriever

from localGPT import (
    CHROMA_SETTINGS,
    DEFAULT_DEVICE_TYPE,
    DEFAULT_EMBEDDING_TYPE,
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_TYPES,
    PERSIST_DIRECTORY,
    SOURCE_DIRECTORY,
)


class ChromaDBLoader:
    def __init__(
        self,
        source_directory: Optional[str],
        persist_directory: Optional[str],
        embedding_model: Optional[str],
        embedding_type: Optional[str],
        device_type: Optional[str],
    ):
        self.source_directory = source_directory or SOURCE_DIRECTORY
        self.persist_directory = persist_directory or PERSIST_DIRECTORY
        self.embedding_model = embedding_model or DEFAULT_EMBEDDING_MODEL
        self.embedding_type = embedding_type or DEFAULT_EMBEDDING_TYPE
        self.device_type = device_type or DEFAULT_DEVICE_TYPE

    def load_embedding_function(self) -> Optional[Embeddings]:
        if self.embedding_type in EMBEDDING_TYPES.keys():
            embedding_class = EMBEDDING_TYPES[self.embedding_type]
            return embedding_class(
                model_name=self.embedding_model,
                model_kwargs={"device": self.device_type},
            )
        else:
            raise AttributeError(
                f"Invalid embeddings type provided: {self.embedding_type}"
            )

    def load_retriever(self) -> VectorStoreRetriever:
        database = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.load_embedding_function(),
            client_settings=CHROMA_SETTINGS,
        )
        return database.as_retriever()

    def persist(self, documents: List[Document]) -> None:
        # Persist the embeddings to Chroma database
        database = Chroma.from_documents(
            documents,
            self.load_embedding_function(),
            persist_directory=self.persist_directory,
            client_settings=CHROMA_SETTINGS,
        )
        database.persist()

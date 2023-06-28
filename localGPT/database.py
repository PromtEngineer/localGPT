"""
localGPT/database.py

This module provides functionality for loading and persisting
documents to the Chroma database.

Classes:
- ChromaDBLoader: A class for loading and persisting documents to the Chroma database.

The ChromaDBLoader class handles loading documents, generating
embeddings using different embedding models, and persisting the
documents along with their embeddings to the Chroma database.

Usage:
    # Example usage of ChromaDBLoader
    loader = ChromaDBLoader(
        source_directory="path/to/source",
        persist_directory="path/to/database",
        embedding_model="huggingface/model",
        embedding_type="HuggingFaceInstructEmbeddings",
        device_type="cuda",
    )

    # Load the vector store retriever
    retriever = loader.load_retriever()

    # Persist a list of documents to the Chroma database
    documents = [document1, document2, document3]
    loader.persist(documents)
"""

from typing import List, Optional

from langchain.base_language import BaseLanguageModel
from langchain.chains import RetrievalQA
from langchain.chains.retrieval_qa.base import BaseRetrievalQA
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import Chroma
from langchain.vectorstores.base import VectorStoreRetriever

from localGPT import (
    CHROMA_SETTINGS,
    DEFAULT_DEVICE_TYPE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_TYPE,
    EMBEDDING_TYPES,
    PERSIST_DIRECTORY,
    SOURCE_DIRECTORY,
)


class ChromaDBLoader:
    """
    ChromaDBLoader class handles loading and persisting documents to Chroma database.

    Args:
        source_directory (str, optional): Directory path for source documents.
            Defaults to SOURCE_DIRECTORY.

        persist_directory (str, optional): Directory path for persisting the database.
            Defaults to PERSIST_DIRECTORY.

        embedding_model (str, optional): Name of the embedding model.
            Defaults to DEFAULT_EMBEDDING_MODEL.

        embedding_type (str, optional): Type of the embedding.
            Defaults to DEFAULT_EMBEDDING_TYPE.

        device_type (str, optional): Device type for embeddings.
            Defaults to DEFAULT_DEVICE_TYPE.
    """

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
        """
        Load the embedding function based on the specified embedding type.

        Returns:
            Optional[Embeddings]: Embeddings object for the specified embedding type.

        Raises:
            AttributeError: If an unsupported embedding type is provided.
        """
        if self.embedding_type in EMBEDDING_TYPES.keys():
            embedding_class = EMBEDDING_TYPES[self.embedding_type]
            return embedding_class(
                model_name=self.embedding_model,
                model_kwargs={"device": self.device_type},
            )
        else:
            raise AttributeError(f"Unsupported embeddings type provided: {self.embedding_type}")

    def load_retriever(self) -> VectorStoreRetriever:
        """
        Load the vector store retriever from the Chroma database.

        Returns:
            VectorStoreRetriever: VectorStoreRetriever object.
        """
        database = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.load_embedding_function(),
            client_settings=CHROMA_SETTINGS,
        )
        return database.as_retriever()

    def load_retrieval_qa(self, llm: BaseLanguageModel) -> BaseRetrievalQA:
        """
        Loads a retrieval-based question answering model.

        Args:
            llm (BaseLanguageModel): The language model for answering questions.

        Returns:
            BaseRetrievalQA: The retrieval-based question answering model.
        """
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.load_retriever(),
            return_source_documents=True,
        )

    def persist(self, documents: List[Document]) -> None:
        """
        Persist the documents and their embeddings to the Chroma database.

        Args:
            documents (List[Document]): List of Document objects to be persisted.
        """
        # Persist the embeddings to Chroma database
        database = Chroma.from_documents(
            documents=documents,
            embedding=self.load_embedding_function(),
            persist_directory=self.persist_directory,
            client_settings=CHROMA_SETTINGS,
        )
        database.persist()

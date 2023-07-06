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
        path_documents="path/to/source",
        path_database="path/to/database",
        repo_id="huggingface/model",
        embeddings_class="HuggingFaceInstructEmbeddings",
        device_type="cuda",
    )

    # Load the vector store retriever
    retriever = loader.load_retriever()

    # Persist a list of documents to the Chroma database
    documents = [document1, document2, document3]
    loader.persist(documents)
"""

from chromadb.config import Settings
from langchain.base_language import BaseLanguageModel
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chains.retrieval_qa.base import BaseRetrievalQA
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain.vectorstores.base import VectorStoreRetriever

from localGPT import (
    HF_EMBEDDINGS_REPO_ID,
    LC_EMBEDDINGS_CLASS,
    MAP_LC_EMBEDDINGS_CLASSES,
    PATH_DATABASE,
    PATH_DOCUMENTS,
    TORCH_DEVICE_TYPE,
)


class ChromaDBLoader:
    """
    ChromaDBLoader class handles loading and persisting documents to Chroma database.

    Args:
        path_documents (str, optional): Directory path for source documents.
            Defaults to SOURCE_DIRECTORY.

        path_database (str, optional): Directory path for persisting the database.
            Defaults to PERSIST_DIRECTORY.

        repo_id (str, optional): Name of the embedding model.
            Defaults to EMBEDDING_MODEL.

        embeddings_class (str, optional): Type of the embedding.
            Defaults to EMBEDDING_TYPE.

        device_type (str, optional): Device type for embeddings.
            Defaults to DEVICE_TYPE.

    Docs:
        https://python.langchain.com/docs/modules/chains/popular/vector_db_qa.html
    """

    def __init__(
        self,
        path_documents: str | None,
        path_database: str | None,
        repo_id: str | None,
        embeddings_class: str | None,
        device_type: str | None,
        settings: Settings | None,
    ):
        self.path_documents = path_documents or PATH_DOCUMENTS
        self.path_database = path_database or PATH_DATABASE
        self.repo_id = repo_id or HF_EMBEDDINGS_REPO_ID
        self.embeddings_class = embeddings_class or LC_EMBEDDINGS_CLASS
        self.device_type = device_type or TORCH_DEVICE_TYPE

        # The settings for the Chroma database
        # - chroma_db_impl: Chroma database implementation (duckdb+parquet)
        # - path_database: Directory for persisting the database
        # - anonymized_telemetry: Whether anonymized telemetry is enabled (False)
        self.settings: Settings = settings or Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=self.path_database,
            anonymized_telemetry=False,
        )

    def load_embedding_function(self) -> Embeddings | None:
        """
        Load the embedding function based on the specified embedding type.

        Returns:
            Optional[Embeddings]: Embeddings object for the specified embedding type.

        Raises:
            AttributeError: If an unsupported embedding type is provided.
        """
        if self.embeddings_class in MAP_LC_EMBEDDINGS_CLASSES.keys():
            embedding_class = MAP_LC_EMBEDDINGS_CLASSES[self.embeddings_class]
            return embedding_class(
                model_name=self.repo_id,
                model_kwargs={"device": self.device_type},
            )
        else:
            raise AttributeError(f"Unsupported embeddings type provided: {self.embeddings_class}")

    def load_retriever(self) -> VectorStoreRetriever:
        """
        Load the vector store retriever from the Chroma database.

        Returns:
            VectorStoreRetriever: VectorStoreRetriever object.
        """
        database = Chroma(
            persist_directory=self.path_database,
            embedding_function=self.load_embedding_function(),
            client_settings=self.settings,
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

    def load_conversational_qa(self, llm: BaseLanguageModel) -> BaseRetrievalQA:
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
        )
        return ConversationalRetrievalChain.from_llm(
            llm,
            self.load_retriever(),
            memory=memory,
            return_source_documents=True,
        )

    def persist(self, documents: list[Document]) -> None:
        """
        Persist the documents and their embeddings to the Chroma database.

        Args:
            documents (List[Document]): List of Document objects to be persisted.
        """
        # Persist the embeddings to Chroma database
        database = Chroma.from_documents(
            documents=documents,
            embedding=self.load_embedding_function(),
            persist_directory=self.path_database,
            client_settings=self.settings,
        )
        database.persist()

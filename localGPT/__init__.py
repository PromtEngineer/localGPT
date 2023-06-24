"""
localGPT/__init__.py

ROOT_DIRECTORY: A string variable that stores the absolute path of the current working directory.

SOURCE_DIRECTORY: A string variable that concatenates the ROOT_DIRECTORY with "/SOURCE_DOCUMENTS". This defines the folder for storing the source documents.

PERSIST_DIRECTORY: A string variable that concatenates the ROOT_DIRECTORY with "/DB". This defines the folder for storing the database.

INGEST_THREADS: An integer variable that stores the number of CPU threads for ingestion. It uses os.cpu_count() to determine the number of CPU cores, and if the count is not available, it defaults to 8.

CHROMA_SETTINGS: A Settings object from the chromadb.config module. It is initialized with three arguments: chroma_db_impl, persist_directory, and anonymized_telemetry. These arguments define the Chroma database implementation (duckdb+parquet), the directory for persisting the database, and whether anonymized telemetry is enabled (False).

MIME_TYPES: A tuple of tuples that associates MIME types with loader classes. Each inner tuple consists of a MIME type string and a loader class. The loader classes are imported from various modules: TextLoader from langchain.document_loaders.base, PDFMinerLoader from langchain.document_loaders, CSVLoader from langchain.document_loaders, and UnstructuredExcelLoader from langchain.document_loaders.
"""
import os
from typing import Tuple, Type

from chromadb.config import Settings
from langchain.document_loaders import (
    CSVLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredExcelLoader,
)
from langchain.document_loaders.base import BaseLoader
from langchain.embeddings import (
    CohereEmbeddings,
    HuggingFaceEmbeddings,
    HuggingFaceInstructEmbeddings,
    OpenAIEmbeddings,
    SentenceTransformerEmbeddings,
)
from langchain.text_splitter import Language

# NOTE: Handling the default paths this way is not a good idea because
# this will look for the parent path relative to the package it self.
# It's probably better to include these paths within the package it self
# or allow the user to configure or set a path instead.
#
# Get the path for this source file
SOURCE_PATH = os.path.dirname(os.path.realpath(__file__))
# Get the absolute path of the package root directory
ROOT_DIRECTORY: str = os.path.abspath(os.path.join(SOURCE_PATH, ".."))
# Set the default path for storing the source documents
SOURCE_DIRECTORY: str = os.path.join(ROOT_DIRECTORY, "SOURCE_DOCUMENTS")
# Set the default path for storing the database
PERSIST_DIRECTORY: str = os.path.join(ROOT_DIRECTORY, "database")

# The number of CPU threads for ingestion
# If os.cpu_count() is not available, it defaults to 8
INGEST_THREADS: int = os.cpu_count() or 8

# The settings for the Chroma database
# - chroma_db_impl: Chroma database implementation (duckdb+parquet)
# - persist_directory: Directory for persisting the database
# - anonymized_telemetry: Whether anonymized telemetry is enabled (False)
CHROMA_SETTINGS: Settings = Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=PERSIST_DIRECTORY,
    anonymized_telemetry=False,
)

# A tuple of tuples associating MIME types with loader classes
# Each inner tuple consists of a MIME type string and a loader class
# NOTE: <type>[<type>[], ...] syntax states that you expect a <type>
# that can contain any number of inner <type>s.
MIME_TYPES: Tuple[Tuple[str, Type[BaseLoader]], ...] = (
    ("text/plain", TextLoader),
    ("application/pdf", PDFMinerLoader),
    ("text/csv", CSVLoader),
    ("application/vnd.ms-excel", UnstructuredExcelLoader),
)

# `str` is the file extension and Language is the Enum mapped to it
LANGUAGE_TYPES: Tuple[Tuple[str, str], ...] = (
    ("cpp", Language.CPP),  # C++ source files
    ("go", Language.GO),  # Go source files
    ("java", Language.JAVA),  # Java source files
    ("js", Language.JS),  # JavaScript source files
    ("php", Language.PHP),  # PHP source files
    ("proto", Language.PROTO),  # Protocol Buffers files
    ("py", Language.PYTHON),  # Python source files
    ("rst", Language.RST),  # reStructuredText files
    ("rb", Language.RUBY),  # Ruby source files
    ("rs", Language.RUST),  # Rust source files
    ("scala", Language.SCALA),  # Scala source files
    ("swift", Language.SWIFT),  # Swift source files
    ("md", Language.MARKDOWN),  # Markdown files
    ("tex", Language.LATEX),  # LaTeX files
    ("html", Language.HTML),  # HTML files
    ("sol", Language.SOL),  # Solidity files
)


# NOTE: SentenceTransformerEmbeddings is a pointer to HuggingFaceEmbeddings
# e.g. They reference the same class definition
# It's included as an option and isn't clarified anywhere else and should
# allow users the ability to use it if explicitly defined for any reason.
EMBEDDING_TYPES = {
    "HuggingFaceInstructEmbeddings": HuggingFaceInstructEmbeddings,
    "HuggingFaceEmbeddings": HuggingFaceEmbeddings,
    "SentenceTransformerEmbeddings": SentenceTransformerEmbeddings,
    "OpenAIEmbeddings": OpenAIEmbeddings,
    "CohereEmbeddings": CohereEmbeddings,
    # Add more embedding types here as needed
}

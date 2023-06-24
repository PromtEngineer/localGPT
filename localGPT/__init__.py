"""
localGPT/__init__.py

This module contains the initialization code and configuration settings 
for the localGPT package.

Constants:
- ROOT_DIRECTORY: The absolute path of the current working directory.
- SOURCE_DIRECTORY: The folder path for storing the source documents.
- PERSIST_DIRECTORY: The folder path for storing the database.
- INGEST_THREADS: The number of CPU threads for ingestion.
- CHROMA_SETTINGS: The settings object for the Chroma database.
- MIME_TYPES: A mapping of MIME types to loader classes.
- LANGUAGE_TYPES: A mapping of file extensions to the Language enumeration.
- EMBEDDING_TYPES: A mapping of embedding type names to embedding classes.

Classes:
- Language: An enumeration representing programming language types.

Note: The default paths for SOURCE_DIRECTORY and PERSIST_DIRECTORY are set 
based on the package structure and can be customized if needed.
"""

import logging
import os
from typing import Tuple, Type

from chromadb.config import Settings
from langchain.document_loaders import (
    CSVLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredExcelLoader,
)
from langchain.embeddings.base import Embeddings
from langchain.document_loaders.base import BaseLoader
from langchain.embeddings import (
    CohereEmbeddings,
    HuggingFaceEmbeddings,
    HuggingFaceInstructEmbeddings,
    OpenAIEmbeddings,
    SentenceTransformerEmbeddings,
)
from langchain.text_splitter import Language

# Set logging configuration
# NOTE: Can be overridden on a script-by-script basis
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s",
    level=logging.INFO,
)

# Get the path for this source file
SOURCE_PATH: str = os.path.dirname(os.path.realpath(__file__))
# Get the absolute path of the package root directory
ROOT_DIRECTORY: str = os.path.abspath(os.path.join(SOURCE_PATH, ".."))
# Set the default path for storing the source documents
SOURCE_DIRECTORY: str = os.path.join(ROOT_DIRECTORY, "SOURCE_DOCUMENTS")
# Set the default path for storing the database
PERSIST_DIRECTORY: str = os.path.join(ROOT_DIRECTORY, "DB")

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

# A mapping of MIME types to loader classes
MIME_TYPES: Tuple[Tuple[str, Type[BaseLoader]], ...] = (
    ("text/plain", TextLoader),
    ("application/pdf", PDFMinerLoader),
    ("text/csv", CSVLoader),
    ("application/vnd.ms-excel", UnstructuredExcelLoader),
)

# A mapping of file extensions to the Language enumeration
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

# A mapping of embedding type names to embedding classes
EMBEDDING_TYPES: dict[str, Type[Embeddings]] = {
    "HuggingFaceInstructEmbeddings": HuggingFaceInstructEmbeddings,
    "HuggingFaceEmbeddings": HuggingFaceEmbeddings,
    "SentenceTransformerEmbeddings": SentenceTransformerEmbeddings,
    "OpenAIEmbeddings": OpenAIEmbeddings,
    "CohereEmbeddings": CohereEmbeddings,
    # Add more embedding types here as needed
}

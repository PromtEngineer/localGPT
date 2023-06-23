"""
localGPT:

ROOT_DIRECTORY: A string variable that stores the absolute path of the current working directory.

SOURCE_DIRECTORY: A string variable that concatenates the ROOT_DIRECTORY with "/SOURCE_DOCUMENTS". This defines the folder for storing the source documents.

PERSIST_DIRECTORY: A string variable that concatenates the ROOT_DIRECTORY with "/DB". This defines the folder for storing the database.

INGEST_THREADS: An integer variable that stores the number of CPU threads for ingestion. It uses os.cpu_count() to determine the number of CPU cores, and if the count is not available, it defaults to 8.

CHROMA_SETTINGS: A Settings object from the chromadb.config module. It is initialized with three arguments: chroma_db_impl, persist_directory, and anonymized_telemetry. These arguments define the Chroma database implementation (duckdb+parquet), the directory for persisting the database, and whether anonymized telemetry is enabled (False).

MIME_TYPES: A tuple of tuples that associates MIME types with loader classes. Each inner tuple consists of a MIME type string and a loader class. The loader classes are imported from various modules: TextLoader from langchain.document_loaders.base, PDFMinerLoader from langchain.document_loaders, CSVLoader from langchain.document_loaders, and UnstructuredExcelLoader from langchain.document_loaders.
"""

import os
from chromadb.config import Settings
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders import (
    CSVLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredExcelLoader,
)

# The absolute path of the current working directory
ROOT_DIRECTORY: str = os.path.dirname(os.environ["PWD"])

# The folder for storing the source documents
SOURCE_DIRECTORY: str = f"{ROOT_DIRECTORY}/source_directory"

# The folder for storing the database
PERSIST_DIRECTORY: str = f"{ROOT_DIRECTORY}/database"

# The number of CPU threads for ingestion
# If os.cpu_count() is not available, it defaults to 8
INGEST_THREADS: int = os.cpu_count() or 8

# The settings for the Chroma database
# - chroma_db_impl: Chroma database implementation (duckdb+parquet)
# - persist_directory: Directory for persisting the database
# - anonymized_telemetry: Whether anonymized telemetry is enabled (False)
CHROMA_SETTINGS: Settings = Settings(
    chroma_db_impl="duckdb+parquet", persist_directory=PERSIST_DIRECTORY, anonymized_telemetry=False
)

# A tuple of tuples associating MIME types with loader classes
# Each inner tuple consists of a MIME type string and a loader class
MIME_TYPES: tuple[tuple[str, BaseLoader]] = (
    ("text/plain", TextLoader),
    ("application/pdf", PDFMinerLoader),
    ("text/csv", CSVLoader),
    ("application/vnd.ms-excel", UnstructuredExcelLoader),
)

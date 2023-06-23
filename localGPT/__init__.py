import os

# from dotenv import load_dotenv
from chromadb.config import Settings
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders import (
    CSVLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredExcelLoader,
)

ROOT_DIRECTORY: str = os.path.dirname(os.path.realpath(__file__))

# Define the folder for storing database
SOURCE_DIRECTORY: str = f"{ROOT_DIRECTORY}/SOURCE_DOCUMENTS"

PERSIST_DIRECTORY: str = f"{ROOT_DIRECTORY}/DB"

# Can be changed to a specific number
INGEST_THREADS: int = os.cpu_count() or 8

# Define the Chroma settings
CHROMA_SETTINGS: Settings = Settings(
    chroma_db_impl="duckdb+parquet", persist_directory=PERSIST_DIRECTORY, anonymized_telemetry=False
)

# Register MIME types with Loaders here
MIME_TYPES: tuple[tuple[str, BaseLoader]] = (
    ("text/plain", TextLoader),
    (
        "application/pdf",
        PDFMinerLoader,
    ),
    (
        "text/csv",
        CSVLoader,
    ),
    (
        "application/vnd.ms-excel",
        UnstructuredExcelLoader,
    ),
)

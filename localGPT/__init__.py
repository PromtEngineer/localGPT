"""
"No plan survives contact with the enemy."
    - Helmuth von Moltke

localGPT/__init__.py

This module contains the initialization code and configuration
settings for the localGPT package.

Constants:
- ROOT_DIRECTORY: The absolute path of the current working directory.
- SOURCE_DIRECTORY: The folder path for storing the source documents.
- PERSIST_DIRECTORY: The folder path for storing the database.
- INGEST_THREADS: The number of CPU threads for ingestion.
- CHROMA_SETTINGS: The settings object for the Chroma database.
- MIME_TYPES: A mapping of MIME types to loader classes.
- LANGUAGE_TYPES: A mapping of file extensions to the Language enumeration.
- EMBEDDING_TYPES: A mapping of embedding type names to embedding classes.
- DEFAULT_DEVICE_TYPE: The default device type for embeddings.
- DEFAULT_EMBEDDING_MODEL: The default embedding model.
- DEFAULT_EMBEDDING_TYPE: The default embedding type.
- DEFAULT_MODEL_REPOSITORY: The default model git repository.
- DEFAULT_MODEL_SAFETENSORS: The default model weights base name.

Classes:
- Language: An enumeration representing programming language types.

Note: The default paths for SOURCE_DIRECTORY and PERSIST_DIRECTORY are
set based on the package structure and can be customized if needed.
"""

import logging
import os

from chromadb.config import Settings
from langchain.document_loaders import CSVLoader, PDFMinerLoader, TextLoader, UnstructuredExcelLoader
from langchain.document_loaders.base import BaseLoader
from langchain.embeddings import (
    CohereEmbeddings,
    HuggingFaceEmbeddings,
    HuggingFaceInstructEmbeddings,
    OpenAIEmbeddings,
    SentenceTransformerEmbeddings,
)
from langchain.embeddings.base import Embeddings
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

# NOTE: IMPORTANT: README!
# If the safetensors are required for certain models and
# can cause failures if not provided, it's important to
# handle this in the code.
# One approach could be to still allow the user to specify
# the safetensors, but provide clear documentation and error
# messages to guide them.
# For example, when a user selects a model, we could display
# a message indicating whether safetensors are required for
# that model and how to find the correct ones.
# e.g.
#   ~/.cache/huggingface/hub
#   ~/.cache/torch/sentence_transformers
# If a user tries to load a model without providing the
# necessary safetensors, we could catch the resulting error
# and display a helpful message explaining what went wrong
# and how to fix it.

# NOTE: IMPORTANT: MODEL_REPOSITORY
# Models are downloaded at runtime.
# The label convention is <username>/<repository>
# where <username>/<repository> represents the url endpoint

# NOTE: IMPORTANT: MODEL_SAFETENSOR
# The label convention is <model-identifier>.safetensors
# where *.safetensors are generated locally at runtime.
# Safetensors must match the given model or will generate
# an exception and fail.

# The default device type to compute with
DEFAULT_DEVICE_TYPE: str = "cuda"
# The default embedding model
DEFAULT_EMBEDDING_MODEL: str = "hkunlp/instructor-large"
# The default embedding type
DEFAULT_EMBEDDING_TYPE: str = "HuggingFaceInstructEmbeddings"
# The default model git repository
DEFAULT_MODEL_REPOSITORY: str = "TheBloke/vicuna-7B-1.1-HF"
# The default class to use
DEFAULT_MODEL_TYPE: str = "huggingface"
# The default model weights base name
DEFAULT_MODEL_SAFETENSORS: str | None = "model.safetensors"

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
MIME_TYPES: tuple[tuple[str, type[BaseLoader]], ...] = (
    ("text/plain", TextLoader),
    ("application/pdf", PDFMinerLoader),
    ("text/csv", CSVLoader),
    ("application/vnd.ms-excel", UnstructuredExcelLoader),
)

# A mapping of file extensions to the Language enumeration
LANGUAGE_TYPES: tuple[tuple[str, str], ...] = (
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
EMBEDDING_TYPES: dict[str, type[Embeddings]] = {
    "HuggingFaceInstructEmbeddings": HuggingFaceInstructEmbeddings,
    "HuggingFaceEmbeddings": HuggingFaceEmbeddings,
    "SentenceTransformerEmbeddings": SentenceTransformerEmbeddings,
    "OpenAIEmbeddings": OpenAIEmbeddings,
    "CohereEmbeddings": CohereEmbeddings,
    # Add more embedding types here as needed
}

CHOICE_EMBEDDING_TYPES: list[str] = [
    "HuggingFaceEmbeddings",
    "HuggingFaceInstructEmbeddings",
]

CHOICE_EMBEDDING_MODELS: list[str] = [
    "hkunlp/instructor-base",
    "hkunlp/instructor-large",
    "hkunlp/instructor-xl",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-MiniLM-L12-v2",
]

CHOICE_MODEL_TYPES: list[str] = [
    "huggingface",
    "llama",
    "ggml",
    "gptq",
]

CHOICE_MODEL_REPOSITORIES: list[str] = [
    "TheBloke/vicuna-7B-1.1-HF",
    "TheBloke/vicuna-7B-1.1-GPTQ-4bit-128g",
    "TheBloke/vicuna-7B-1.1-GGML",
    "TheBloke/wizardLM-7B-HF",
    "TheBloke/WizardLM-7B-V1.0-Uncensored-GPTQ",
    "TheBloke/WizardLM-7B-V1.0-Uncensored-GGML",
    "NousResearch/Nous-Hermes-13b",
    "TheBloke/Nous-Hermes-13B-GPTQ",
    "TheBloke/Nous-Hermes-13B-GGML",
]

CHOICE_MODEL_SAFETENSORS: list[str] = [
    "model.safetensors",
    "WizardLM-7B-uncensored-GPTQ-4bit-128g.compat.no-act-order.safetensors",
    "wizardLM-7B-GPTQ-4bit.compat.no-act-order.safetensors",
    "nous-hermes-13b-GPTQ-4bit-128g.no-act.order.safetensors",
    "WizardLM-30B-Uncensored-GPTQ-4bit.act-order.safetensors",
]

CHOICE_DEVICE_TYPES: list[str] = [
    "cpu",
    "cuda",
    "ipu",
    "xpu",
    "mkldnn",
    "opengl",
    "opencl",
    "ideep",
    "hip",
    "ve",
    "fpga",
    "ort",
    "xla",
    "lazy",
    "vulkan",
    "mps",
    "meta",
    "hpu",
    "mtia",
]

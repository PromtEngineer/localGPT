"""
"No plan survives contact with the enemy."
    - Helmuth von Moltke

localGPT/__init__.py

localGPT Package Initialization and Configuration

This module contains the initialization code and configuration settings for the localGPT package.

Constants:
- PATH_HOME: The user's default home directory.
- PATH_ROOT: The absolute path of the current working directory.
- PATH_DOCUMENTS: The default path for storing the source documents.
- PATH_DATABASE: The default path for storing the database.
- CPU_COUNT: The number of CPU threads for ingestion.
- TORCH_DEVICE_TYPE: The default PyTorch device type to compute with.
- TRITON_DEVICE_TYPE: Whether the device is AMD and supports Triton.

Langchain Settings:
- HF_EMBEDDINGS_REPO_ID: The default Hugging Face embeddings repository ID.
- LC_EMBEDDINGS_CLASS: The default langchain embeddings class.
- HF_MODEL_REPO_ID: The default Hugging Face model repository ID.
- LC_MODEL_CLASS: The default langchain model class.
- HF_MODEL_SAFETENSORS: The default Hugging Face model weights base name.
- SHOW_DOCUMENT_SOURCE: Toggle the RetrievalQA document source reference.

Llama.Cpp Settings:
- GGML_N_CTX: The maximum context size for Llama.Cpp.
- GGML_MAX_TOKENS: The default maximum number of tokens for Llama.Cpp.
- GGML_TEMPERATURE: The default temperature for Llama.Cpp.
- GGML_TOP_P: The default top-p value for Llama.Cpp.
- GGML_N_BATCH: The default batch size for Llama.Cpp.
- GGML_N_GPU_LAYERS: The default number of GPU layers for Llama.Cpp.
- GGML_LOW_VRAM: The default setting for low VRAM usage in Llama.Cpp.
- GGML_REPO_ID: The default Llama.Cpp model git repository ID.
- GGML_FILENAME: The default Llama.Cpp model filename.

Mappings:
- MAP_LC_EMBEDDINGS_CLASSES: A mapping of langchain embeddings classes.
- MAP_LC_MIME_TYPES: A mapping of MIME types to langchain loader classes.
- MAP_LC_LANGUAGE_TYPES: A mapping of file extensions to the Language enumeration.

Choices:
- CHOICE_LC_EMBEDDINGS_CLASSES: A list of available langchain embeddings classes.
- CHOICE_HF_EMBEDDINGS_REPO_IDS: A list of available Hugging Face embeddings repository IDs.
- CHOICE_LC_MODEL_CLASSES: A list of available langchain model classes.
- CHOICE_HF_MODEL_REPO_IDS: A list of available Hugging Face model repository IDs.
- CHOICE_HF_MODEL_SAFETENSORS: A list of available Hugging Face model weights base names.

Classes:
- Language: An enumeration representing programming language types.

NOTE: IMPORTANT: Default PATH_*
The default paths for PATH_DOCUMENTS and PATH_DATABASE are set based on the package structure and can be customized if needed.

NOTE: IMPORTANT: README!
If the safetensors are required for certain models and
can cause failures if not provided, it's important to
handle this in the code.
One approach could be to still allow the user to specify
the safetensors, but provide clear documentation and error
messages to guide them.
For example, when a user selects a model, we could display
a message indicating whether safetensors are required for
that model and how to find the correct ones.
e.g.
  ~/.cache/huggingface/hub
  ~/.cache/torch/sentence_transformers
If a user tries to load a model without providing the
necessary safetensors, we could catch the resulting error
and display a helpful message explaining what went wrong
and how to fix it.

NOTE: IMPORTANT: MODEL_REPOSITORY
Models are downloaded at runtime.
The label convention is <username>/<repository>
where <username>/<repository> represents the url endpoint

NOTE: IMPORTANT: MODEL_SAFETENSOR
The label convention is <model-identifier>.safetensors
where *.safetensors are generated locally at runtime.
Safetensors must match the given model or will generate
an exception and fail.
The default llama.cpp Model settings
"""

import logging
import os
from pathlib import Path

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

# The users default home directory
PATH_HOME: str = str(Path.home())
# The root path is the users current working directory
PATH_ROOT: str = os.getcwd()
# Set the default path for storing the source documents
PATH_DOCUMENTS: str = os.path.join(PATH_ROOT, "SOURCE_DOCUMENTS")
# Set the default path for storing the database
PATH_DATABASE: str = os.path.join(PATH_ROOT, "DB")

# The number of CPU threads for ingestion
CPU_COUNT: int = os.cpu_count() or 2

# The default pytorch device type to compute with
TORCH_DEVICE_TYPE: str = "cuda"
# Whether the device is AMD and supports triton
TRITON_DEVICE_TYPE: bool = False

#
# Langchain Settings
#
# The default embeddings model
HF_EMBEDDINGS_REPO_ID: str = "hkunlp/instructor-large"
# The default langchain embeddings class
LC_EMBEDDINGS_CLASS: str = "HuggingFaceInstructEmbeddings"
# The default model git repository
HF_MODEL_REPO_ID: str = "TheBloke/vicuna-7B-1.1-HF"
# The default langchain class to use
LC_MODEL_CLASS: str = "huggingface"
# The default model weights base name
HF_MODEL_SAFETENSORS: str | None = "model.safetensors"
# Toggle the RetrievalQA document source reference
SHOW_DOCUMENT_SOURCE: bool = False

#
# Llama.Cpp Settings
#
GGML_N_CTX: int = 2048
GGML_MAX_TOKENS: int = 512
GGML_TEMPERATURE: float = 0.8
GGML_TOP_P: float = 0.95
GGML_N_THREADS: int | None = CPU_COUNT

# The default llama.cpp GPU settings
GGML_N_BATCH: int = 512
GGML_N_GPU_LAYERS: int = 0
GGML_LOW_VRAM: bool = False

# The default llama.cpp model git repository
GGML_REPO_ID: str = "TheBloke/vicuna-7B-v1.3-GGML"
# The default llama.cpp model filename from the given git repository
GGML_FILENAME: str = "vicuna-7b-v1.3.ggmlv3.q4_1.bin"

# A mapping of langchain embeddings classes
MAP_LC_EMBEDDINGS_CLASSES: dict[str, type[Embeddings]] = {
    "HuggingFaceInstructEmbeddings": HuggingFaceInstructEmbeddings,
    "HuggingFaceEmbeddings": HuggingFaceEmbeddings,
    "SentenceTransformerEmbeddings": SentenceTransformerEmbeddings,
    "OpenAIEmbeddings": OpenAIEmbeddings,
    "CohereEmbeddings": CohereEmbeddings,
    # Add more embedding types here as needed
}

# A mapping of MIME types to langchain loader classes
MAP_LC_MIME_TYPES: tuple[tuple[str, type[BaseLoader]], ...] = (
    ("text/plain", TextLoader),
    ("application/pdf", PDFMinerLoader),
    ("text/csv", CSVLoader),
    ("application/vnd.ms-excel", UnstructuredExcelLoader),
)

# A mapping of file extensions to the Language enumeration
MAP_LC_LANGUAGE_TYPES: tuple[tuple[str, str], ...] = (
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

CHOICE_LC_EMBEDDINGS_CLASSES: list[str] = [
    "HuggingFaceEmbeddings",
    "HuggingFaceInstructEmbeddings",
]

CHOICE_HF_EMBEDDINGS_REPO_IDS: list[str] = [
    "hkunlp/instructor-base",
    "hkunlp/instructor-large",
    "hkunlp/instructor-xl",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-MiniLM-L12-v2",
]

CHOICE_LC_MODEL_CLASSES: list[str] = ["huggingface", "huggingface-llama", "gptq"]

CHOICE_HF_MODEL_REPO_IDS: list[str] = [
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

CHOICE_HF_MODEL_SAFETENSORS: list[str] = [
    "model.safetensors",
    "WizardLM-7B-uncensored-GPTQ-4bit-128g.compat.no-act-order.safetensors",
    "wizardLM-7B-GPTQ-4bit.compat.no-act-order.safetensors",
    "nous-hermes-13b-GPTQ-4bit-128g.no-act.order.safetensors",
    "WizardLM-30B-Uncensored-GPTQ-4bit.act-order.safetensors",
]

# Supported PyTorch Device Types
CHOICE_PT_DEVICE_TYPES: list[str] = [
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

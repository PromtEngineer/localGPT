import logging
import os
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)

import click
from langchain.docstore.document import Document
from langchain.document_loaders import TextLoader, UnstructuredExcelLoader
from langchain.embeddings import (
    HuggingFaceEmbeddings,
    HuggingFaceInstructEmbeddings,
)
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from localGPT import (
    CHROMA_SETTINGS,
    INGEST_THREADS,
    PERSIST_DIRECTORY,
    SOURCE_DIRECTORY,
)
from localGPT.registry import LoaderRegistry


def load_single_document(file_path: str) -> Document:
    # Loads a single document from a file path
    loader_registry = LoaderRegistry()
    mime_type = loader_registry.get_mime_type(file_path)
    loader_class = loader_registry.get_loader(mime_type)
    if loader_class:
        loader = loader_class(file_path)
    else:
        raise ValueError("Document type is undefined")
    return loader.load()[0]


def load_document_batch(filepaths):
    logging.info("Loading document batch")
    # create a thread pool
    with ThreadPoolExecutor(len(filepaths)) as exe:
        # load files
        futures = [exe.submit(load_single_document, name) for name in filepaths]
        # collect data
        data_list = [future.result() for future in futures]
        # return data and file paths
        return (data_list, filepaths)


def load_documents(source_dir: str) -> list[Document]:
    # Loads all documents from the source documents directory
    paths = []
    all_files = os.listdir(source_dir)
    loader_registry = LoaderRegistry()

    for file_path in all_files:
        source_file_path = os.path.join(source_dir, file_path)
        mime_type = loader_registry.get_mime_type(file_path)
        loader_class = loader_registry.get_loader(mime_type)
        if loader_class:
            paths.append(source_file_path)

    # Have at least one worker and at most INGEST_THREADS workers
    n_workers = min(INGEST_THREADS, max(len(paths), 1))
    chunksize = round(len(paths) / n_workers)
    docs = []
    with ProcessPoolExecutor(n_workers) as executor:
        futures = []
        # split the load operations into chunks
        for i in range(0, len(paths), chunksize):
            # select a chunk of filenames
            filepaths = paths[i : (i + chunksize)]
            # submit the task
            future = executor.submit(load_document_batch, filepaths)
            futures.append(future)
        # process all results
        for future in as_completed(futures):
            # open the file and load the data
            contents, _ = future.result()
            docs.extend(contents)

    return docs


def split_documents(documents: list[Document]) -> tuple[list[Document], list[Document]]:
    # Splits documents for correct Text Splitter
    text_docs, python_docs = [], []
    loader_registry = LoaderRegistry()

    for doc in documents:
        mime_type = loader_registry.get_mime_type(doc.metadata["source"])
        loader_class = loader_registry.get_loader(mime_type)
        if loader_class == TextLoader:
            text_docs.append(doc)
        elif loader_class == UnstructuredExcelLoader:
            python_docs.append(doc)

    return text_docs, python_docs


# Default Instructor Model
#   You can also choose a smaller model.
#   Don't forget to change HuggingFaceInstructEmbeddings
#   to HuggingFaceEmbeddings in both ingest.py and run.py
@click.command()
@click.option(
    "--embedding_model",
    default="hkunlp/instructor-large",
    type=click.Choice(
        [
            "hkunlp/instructor-base",
            "hkunlp/instructor-large",
            "hkunlp/instructor-xl",
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-MiniLM-L12-v2",
        ]
    ),
    help="Instructor Model to generate embeddings with (default: hkunlp/instructor-large)",
)
@click.option(
    "--embedding_type",
    default="HuggingFaceInstructEmbeddings",
    type=click.Choice(
        [
            "HuggingFaceEmbeddings",
            "HuggingFaceInstructEmbeddings",
        ]
    ),
    help="Embedding type to use (default: HuggingFaceInstructEmbeddings)",
)
@click.option(
    "--device_type",
    default="cuda",
    type=click.Choice(
        [
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
        ],
    ),
    help="Device to run on (default: cuda)",
)
def main(embedding_model, embedding_type, device_type):
    # Load documents and split them into chunks
    logging.info(f"Loading documents from {SOURCE_DIRECTORY}")
    documents = load_documents(SOURCE_DIRECTORY)
    text_documents, python_documents = split_documents(documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=1000, chunk_overlap=200
    )
    texts = text_splitter.split_documents(text_documents)
    texts.extend(python_splitter.split_documents(python_documents))
    logging.info(f"Loaded {len(documents)} documents from {SOURCE_DIRECTORY}")
    logging.info(f"Split into {len(texts)} chunks of text")

    # Create embeddings
    if embedding_type == "HuggingFaceInstructEmbeddings":
        embeddings = HuggingFaceInstructEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": device_type},
        )
    elif embedding_type == "HuggingFaceEmbeddings":
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model, model_kwargs={"device": device_type},)
    else:
        raise ValueError("Invalid embeddings type provided.")

    # Persist the embeddings to Chroma database
    db = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=PERSIST_DIRECTORY,
        client_settings=CHROMA_SETTINGS,
    )
    db.persist()
    db = None


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()

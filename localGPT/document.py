# localGPT/document.py
import logging
import os
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from typing import List, Tuple

from langchain.docstore.document import Document
from langchain.document_loaders import (
    PDFMinerLoader,
    TextLoader,
    UnstructuredExcelLoader,
)
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter

from localGPT import INGEST_THREADS
from localGPT.registry import LoaderRegistry


def load_single_document(file_path: str) -> Document:
    """
    Loads a single document from the given file path.

    Args:
        file_path (str): The path to the document file.

    Returns:
        Document: The loaded document.

    Raises:
        ValueError: If the document type is undefined.
    """
    loader_registry = LoaderRegistry()
    mime_type = loader_registry.get_mime_type(file_path)
    loader_class = loader_registry.get_loader(mime_type)
    if loader_class:
        loader = loader_class(file_path)
    else:
        raise ValueError(f"Document type is undefined: {mime_type}")
    return loader.load()[0]


def load_document_batch(
    filepaths: List[str],
) -> Tuple[List[Document], List[str]]:
    """
    Loads a batch of documents from the given file paths.

    Args:
        filepaths (List[str]): List of file paths to load the documents from.

    Returns:
        Tuple[List[Document], List[str]]: A tuple containing the loaded documents and the corresponding file paths.

    Raises:
        ValueError: If the document type is undefined.
    """
    logging.info("Loading document batch")
    # create a thread pool
    with ThreadPoolExecutor(len(filepaths)) as exe:
        # load files
        futures = [exe.submit(load_single_document, path) for path in filepaths]
        # collect data
        data_list = [future.result() for future in futures]
        # return data and file paths
        return (data_list, filepaths)


def load_documents(source_dir: str) -> List[Document]:
    """
    Loads all documents from the specified source documents directory.

    Args:
        source_dir (str): The path to the source documents directory.

    Returns:
        List[Document]: A list of loaded documents.

    Raises:
        ValueError: If the document type is undefined.
    """
    paths = []
    all_files = os.listdir(source_dir)
    loader_registry = LoaderRegistry()

    logging.info(f"Loading documents: {source_dir}")
    logging.info(f"Loading document files: {all_files}")

    for file_path in all_files:
        source_file_path = os.path.join(source_dir, file_path)
        mime_type = loader_registry.get_mime_type(source_file_path)
        loader_class = loader_registry.get_loader(mime_type)

        logging.info(f"Detected {mime_type} for {file_path}")

        if loader_class:
            logging.info(f"Loading {source_file_path}")
            paths.append(source_file_path)

    # Have at least one worker and at most INGEST_THREADS workers
    n_workers = min(INGEST_THREADS, max(len(paths), 1))
    chunk_size = round(len(paths) / n_workers)
    docs = []

    with ProcessPoolExecutor(n_workers) as executor:
        futures = []
        # split the load operations into chunks
        for i in range(0, len(paths), chunk_size):
            # select a chunk of filenames
            filepaths = paths[i : (i + chunk_size)]
            # submit the task
            future = executor.submit(load_document_batch, filepaths)
            futures.append(future)
        # process all results
        for future in as_completed(futures):
            # open the file and load the data
            contents, _ = future.result()
            docs.extend(contents)

    return docs


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Splits the given documents based on their type for the correct Text Splitter.

    Args:
        documents (List[Document]): The list of documents to split.

    Returns:
        List[Document]: A list of split documents.

    """
    logging.info(f"Splitting: {[doc.metadata['source'] for doc in documents]}")

    text_docs, python_docs = [], []
    loader_registry = LoaderRegistry()

    for doc in documents:
        logging.info(f"Splitting: {doc.metadata['source']}")
        mime_type = loader_registry.get_mime_type(doc.metadata["source"])
        logging.info(f"Splitting: {mime_type}")
        loader_class = loader_registry.get_loader(mime_type)

        if isinstance(loader_class, TextLoader):
            if doc.metadata["source"].endswith(".py"):
                python_docs.append(doc)
            else:
                text_docs.append(doc)
        else:
            text_docs.append(doc)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=1000, chunk_overlap=200
    )
    text_documents = text_splitter.split_documents(text_docs)
    python_documents = python_splitter.split_documents(python_docs)

    return text_documents + python_documents

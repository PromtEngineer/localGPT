import logging
import os
from pathlib import Path
from typing import List
import shutil
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceBgeEmbeddings
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from config import (
    CHROMA_SETTINGS,
    DOCUMENT_MAP,
    EMBEDDING_MODEL_NAME,
    INGEST_THREADS,
    PERSIST_DIRECTORY,
    RESET_DB,
    SOURCE_DIRECTORY,
    DEVICE_TYPE
)

def file_log(logentry):
   file1 = open("file_ingest.log","a")
   file1.write(logentry + "\n")
   file1.close()
   print(logentry + "\n")

def load_single_document(file_path: str) -> list[Document]:
    # Loads a single document from a file path
    try:
       file_extension = os.path.splitext(file_path)[1]
       loader_class = DOCUMENT_MAP.get(file_extension)
       if loader_class:
           file_log(file_path + ' loaded.')
           loader = loader_class(file_path)
       else:
           file_log(file_path + ' document type is undefined.')
           raise ValueError("Document type is undefined")
       return loader.load()
    except Exception as ex:
       file_log('%s loading error: \n%s' % (file_path, ex))
       return None 


def load_document_batch(filepaths):
    logging.info("Loading document batch")
    # create a thread pool
    with ThreadPoolExecutor(len(filepaths)) as exe:
        # load files
        futures = [exe.submit(load_single_document, name) for name in filepaths]
        # collect data
        if futures is None:
           file_log(name + ' failed to submit')
           return None
        else:
           nested_data_list = [future.result() for future in futures]
           data_list = [item for sublist in nested_data_list for item in sublist]
           # return data and file paths
           return (data_list, filepaths)


def load_documents(source_dir: str) -> list[Document]:
    # Loads all documents from the source documents directory, including nested folders
    paths = []
    for root, _, files in os.walk(source_dir):
        for file_name in files:
            print('Importing: ' + file_name)
            file_extension = os.path.splitext(file_name)[1]
            source_file_path = os.path.join(root, file_name)

            # Rename the file to have a lowercase extension
            if file_extension != file_extension.lower():
                new_file_name = os.path.splitext(file_name)[0] + file_extension.lower()
                new_file_path = os.path.join(root, new_file_name)
                os.rename(source_file_path, new_file_path)
                source_file_path = new_file_path

            if file_extension.lower() in DOCUMENT_MAP.keys():
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
            try:
               future = executor.submit(load_document_batch, filepaths)
            except Exception as ex:
               file_log('executor task failed: %s' % (ex))
               future = None
            if future is not None:
               futures.append(future)
        # process all results
        for future in as_completed(futures):
            # open the file and load the data
            try:
                contents, _ = future.result()
                docs.extend(contents)
            except Exception as ex:
                file_log('Exception: %s' % (ex))
    return docs


def split_documents(documents: list[Document]) -> tuple[list[Document], list[Document]]:
    # Splits documents for correct Text Splitter
    text_docs, python_docs = [], []
    for doc in documents:
        if doc is not None:
           file_extension = os.path.splitext(doc.metadata["source"])[1]
           if file_extension == ".py":
               python_docs.append(doc)
           else:
               text_docs.append(doc)

    return text_docs, python_docs


def create_db(db_name: str = None):

    persist_directory = PERSIST_DIRECTORY
    source_directory = SOURCE_DIRECTORY
    if db_name:
        persist_directory = os.path.join(persist_directory, db_name)
        os.makedirs(persist_directory, exist_ok=True)
        CHROMA_SETTINGS.persist_directory = persist_directory
        source_directory = os.path.join(source_directory, db_name)

    if os.path.exists(persist_directory):
        if RESET_DB:
            logging.info(f"DB Already Exists - Removing it")
            shutil.rmtree(persist_directory)
        else:
            logging.info(f"DB Already Exists - Using the existing DB")
    
    logging.info(f"Loading documents from {source_directory}")
    
    documents = load_documents(source_directory)
    text_documents, python_documents = split_documents(documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=1080, chunk_overlap=200
    )
    texts = text_splitter.split_documents(text_documents)
    texts.extend(python_splitter.split_documents(python_documents))
    logging.info(f"Loaded {len(documents)} documents from {source_directory}")
    logging.info(f"Split into {len(texts)} chunks of text")

    for text in texts:
        text.page_content = f"Document Name: {text.metadata['source'][text.metadata['source'].rindex('/')+1:]}\n{text.page_content}"

    if "instructor" in EMBEDDING_MODEL_NAME:
        embeddings = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": DEVICE_TYPE})
    elif "bge" in EMBEDDING_MODEL_NAME:
        embeddings = HuggingFaceBgeEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": DEVICE_TYPE},
            encode_kwargs={'normalize_embeddings': True}
        )

    db = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=persist_directory,  # Use the customized directory
        client_settings=CHROMA_SETTINGS,
    )
    db = None

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--db_name", help="Optional database name", default=None)
    args = parser.parse_args()
    
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO)
    create_db(args.db_name)

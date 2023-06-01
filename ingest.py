#!/usr/bin/python3

from typing import List

import click
from langchain.docstore.document import Document
from langchain.document_loaders import TextLoader, PDFMinerLoader, CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from constants import *  # CHROMA_SETTINGS, SOURCE_DIRECTORY, PERSIST_DIRECTORY


def load_single_document(file_path: str) -> Document:
    """ Loads a single document from a file path """
    if file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf8")
    elif file_path.endswith(".pdf"):
        loader = PDFMinerLoader(file_path)
    elif file_path.endswith(".csv"):
        loader = CSVLoader(file_path)
    else:
        raise UserWarning(f"Unknown file extension: {file_path.split('.')[-1]}")
    return loader.load()[0]


def load_documents(source_dir: str) -> List[Document]:
    """ Loads all documents from source documents directory """
    all_files = os.listdir(source_dir)
    return [load_single_document(os.path.join(source_dir, file_path)) for file_path
            in all_files if file_path[-4:]
            in ['.txt', '.pdf', '.csv'] ]


@click.command()
@click.option('--device_type', default='gpu', help='device to run on, select gpu or cpu')
def main(device_type, ):
    """ load the instructorEmbeddings """
    if device_type.lower() in ['cpu']:
        device='cpu'
    else:
        device='cuda'

    # Load documents and split in chunks
    print(f"Loading documents from {SOURCE_DIRECTORY}")
    documents = load_documents(SOURCE_DIRECTORY)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    print(f"Loaded {len(documents)} documents from {SOURCE_DIRECTORY}")
    print(f"Split into {len(texts)} chunks of text")

    # Create embeddings
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl", model_kwargs={"device": device})
    
    db = Chroma.from_documents(texts, embeddings, persist_directory=PERSIST_DIRECTORY, client_settings=CHROMA_SETTINGS)
    # del db ?
    db.persist()
    db = None



if __name__ == "__main__":
    main()

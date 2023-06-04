import os
import click
from typing import List
from utils import xlxs_to_csv
from langchain.document_loaders import TextLoader, PDFMinerLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from constants import CHROMA_SETTINGS, SOURCE_DIRECTORY, PERSIST_DIRECTORY
from langchain.embeddings import HuggingFaceInstructEmbeddings


def load_single_document(file_path: str) -> Document:
    # Loads a single document from a file path
    if file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf8")
    elif file_path.endswith(".pdf"):
        loader = PDFMinerLoader(file_path)
    elif file_path.endswith(".csv"):
        loader = CSVLoader(file_path)
    return loader.load()[0]


def load_documents(source_dir: str) -> List[Document]:
    # Loads all documents from source documents directory
    all_files = os.listdir(source_dir)
    docs = []
    for file_path in all_files:
        if file_path[-4:] == 'xlsx':
            for doc in xlxs_to_csv(f"{source_dir}/{file_path}"):
                docs.append(load_single_document(doc))
        elif file_path[-4:] in ['.txt', '.pdf', '.csv']:
            docs.append(load_single_document(f"{source_dir}/{file_path}"))
    return docs
    # return [load_single_document(f"{source_dir}/{file_path}") for file_path in all_files if
    #         file_path[-4:] in ['.txt', '.pdf', '.csv']]


# @click.command()
# @click.option('--device_type', default='gpu', help='device to run on, select gpu or cpu')
# def main(device_type, ):
#     # load the instructorEmbeddings
#     if device_type in ['cpu', 'CPU']:
#         device='cpu'
#     else:
#         device='cuda'


@click.command()
@click.option('--device_type', default='cuda', help='device to run on, select gpu, cpu or mps')
def main(device_type, ):
    # load the instructorEmbeddings
    if device_type in ['cpu', 'CPU']:
        device='cpu'
    elif device_type in ['mps', 'MPS']:
        device='mps'
    else:
        device='cuda'

    #  Load documents and split in chunks
    print(f"Loading documents from {SOURCE_DIRECTORY}")
    documents = load_documents(SOURCE_DIRECTORY)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    print(f"Loaded {len(documents)} documents from {SOURCE_DIRECTORY}")
    print(f"Split into {len(texts)} chunks of text")

    # Create embeddings
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",
                                               model_kwargs={"device": device})

    db = Chroma.from_documents(texts, embeddings, persist_directory=PERSIST_DIRECTORY, client_settings=CHROMA_SETTINGS)
    db.persist()
    db = None


if __name__ == "__main__":
    main()

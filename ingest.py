import os
import glob
from typing import List
from dotenv import load_dotenv
import pickle

from langchain.document_loaders import TextLoader, PDFMinerLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from constants import CHROMA_SETTINGS
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings
load_dotenv()


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
    docs_path = f"/privateGPT/{source_dir}" # replace this with the absolute path of the source_documents folder
    all_files = os.listdir(docs_path)
    return [load_single_document(f"{docs_path}/{file_path}") for file_path in all_files if file_path[-4:] in ['.txt', '.pdf', '.csv'] ]


def main():
    # Load environment variables
    persist_directory = os.environ.get('PERSIST_DIRECTORY')
    source_directory = os.environ.get('SOURCE_DIRECTORY', 'source_documents')
    llama_embeddings_model = os.environ.get('LLAMA_EMBEDDINGS_MODEL')
    model_n_ctx = os.environ.get('MODEL_N_CTX')

    # Load documents and split in chunks
    print(f"Loading documents from {source_directory}")
    documents = load_documents(source_directory)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    print(f"Loaded {len(documents)} documents from {source_directory}")
    print(f"Split into {len(texts)} chunks of text (max. 500 tokens each)")

    # Create embeddings
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",
                                                model_kwargs={"device": "cuda"})
    
    db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
    db.persist()
    db = None


if __name__ == "__main__":
    main()

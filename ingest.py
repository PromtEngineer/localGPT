import logging
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import faiss
import pickle
from transformers import AutoModel, AutoTokenizer
import psycopg2

import click
import torch
from langchain.docstore.document import Document
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma , FAISS
from utils import get_embeddings
from langchain_community.docstore.in_memory import InMemoryDocstore

from langchain_cohere import CohereEmbeddings
# from langchain_core.documents import Document
# from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector

from pgvector.psycopg import register_vector
import psycopg

from constants import (
    CHROMA_SETTINGS,
    DOCUMENT_MAP,
    EMBEDDING_MODEL_NAME,
    INGEST_THREADS,
    PERSIST_DIRECTORY,
    SOURCE_DIRECTORY,
    # INDEX_PATH,
    # METADATA_PATH,

)

import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')


def file_log(logentry):
    file1 = open("file_ingest.log", "a")
    file1.write(logentry + "\n")
    file1.close()
    print(logentry + "\n")


def load_single_document(file_path: str) -> Document:
    # Loads a single document from a file path
    try:
        file_extension = os.path.splitext(file_path)[1]
        loader_class = DOCUMENT_MAP.get(file_extension)
        if loader_class:
            file_log(file_path + " loaded.")
            loader = loader_class(file_path)
        else:
            file_log(file_path + " document type is undefined.")
            raise ValueError("Document type is undefined")
        return loader.load()[0]
    except Exception as ex:
        file_log("%s loading error: \n%s" % (file_path, ex))
        return None


def load_document_batch(filepaths):
    logging.info("Loading document batch")
    # create a thread pool
    with ThreadPoolExecutor(len(filepaths)) as exe:
        # load files
        futures = [exe.submit(load_single_document, name) for name in filepaths]
        # collect data
        if futures is None:
            file_log(name + " failed to submit")
            return None
        else:
            data_list = [future.result() for future in futures]
            # return data and file paths
            return (data_list, filepaths)


def load_documents(source_dir: str) -> list[Document]:
    # Loads all documents from the source documents directory, including nested folders
    paths = []
    for root, _, files in os.walk(source_dir):
        for file_name in files:
            print("Importing: " + file_name)
            file_extension = os.path.splitext(file_name)[1]
            source_file_path = os.path.join(root, file_name)
            if file_extension in DOCUMENT_MAP.keys():
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
                file_log("executor task failed: %s" % (ex))
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
                file_log("Exception: %s" % (ex))

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


# @click.command()
# @click.option(
#     "--device_type",
#     default="cuda" if torch.cuda.is_available() else "cpu",
#     type=click.Choice(
#         [
#             "cpu",
#             "cuda",
#             "ipu",
#             "xpu",
#             "mkldnn",
#             "opengl",
#             "opencl",
#             "ideep",
#             "hip",
#             "ve",
#             "fpga",
#             "ort",
#             "xla",
#             "lazy",
#             "vulkan",
#             "mps",
#             "meta",
#             "hpu",
#             "mtia",
#         ],
#     ),
#     help="Device to run on. (Default is cuda)",
# )
# def save_faiss_index(db, index_path, metadata_path):
#     faiss.write_index(db.index, index_path)
#     metadata = {
#         "index_to_docstore_id": db.index_to_docstore_id,
#         "docstore": db.docstore,
#     }
#     with open(metadata_path, "wb") as f:
#         pickle.dump(metadata, f)

# def load_faiss_index(index_path, metadata_path):
#     index = faiss.read_index(index_path)
#     with open(metadata_path, "rb") as f:
#         metadata = pickle.load(f)
#     docstore = metadata["docstore"]
#     index_to_docstore_id = metadata["index_to_docstore_id"]
#     db = FAISS(index=index,
#                docstore=docstore,
#                index_to_docstore_id=index_to_docstore_id)
#     return db
device_type = 'cpu'


def main(device_type):
    print(f"Running on device: {device_type}")
    # Load documents and split in chunks
    logging.info(f"Loading documents from {SOURCE_DIRECTORY}")
    documents = load_documents(SOURCE_DIRECTORY)
    text_documents, python_documents = split_documents(documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=880, chunk_overlap=200
    )
    texts = text_splitter.split_documents(text_documents)
    texts.extend(python_splitter.split_documents(python_documents))
    logging.info(f"Loaded {len(documents)} documents from {SOURCE_DIRECTORY}")
    logging.info(f"Split into {len(texts)} chunks of text")

    """
    (1) Chooses an appropriate langchain library based on the enbedding model name.  Matching code is contained within fun_localGPT.py.
    
    (2) Provides additional arguments for instructor and BGE models to improve results, pursuant to the instructions contained on
    their respective huggingface repository, project page or github repository.
    
    """

    embeddings = get_embeddings(device_type)

    logging.info(f"Loaded embeddings from {EMBEDDING_MODEL_NAME}")

    # # See docker command above to launch a postgres instance with pgvector enabled.
    connection = "postgresql+psycopg://postgres:123456@localhost:5432/postgres"  # Uses psycopg3!
    # # "dbname=postgres user=postgres password=123456 host=localhost port=5432"
    # connection.execute('CREATE EXTENSION IF NOT EXISTS vector')
    # register_vector(connection)
    
    # connection.execute('DROP TABLE IF EXISTS documents')
    # connection.execute('CREATE TABLE documents (id bigserial PRIMARY KEY, content text, embedding vector(384))')

    collection_name = "PG_VECTOR_SAudi"
    # embeddings = CohereEmbeddings()
# -------------------------------
# Q
#--------------------------------
    # connection = psycopg2.connect("dbname=postgres user=postgres password=123456 host=localhost port=5432")
    # connection = connection.cursor()
    # db = PGVector(
    #     documents= texts,
    #     embeddings=embeddings,
    #     collection_name=collection_name,
    #     connection=connection,
    #     use_jsonb=True,
    # )
    # db.add_documents(texts, ids=[doc.metadata["id"] for doc in texts])
    # "dbname=postgres user=postgres password=123456 host=localhost port=5432"
    #changing to more programatically conn string
    db = PGVector.from_documents(
        documents= texts,
        embedding=embeddings,
        collection_name=collection_name,
        connection=connection,
        use_jsonb=True,
    )
    print(">>>>>>>>/n/n>>>>>>>>>>Connected AND Loaded to the database successfully!")

    # collection_name = "PG_VECTOR_SAudi"
    # db = PGVector.from_documents(
    #     embedding=embeddings,
    #     documents=texts,
    #     connection_string=CONNECTION_STRING,
    #     collection_name=collection_name,
    # )
# -------------------------------
# Q
#--------------------------------
    # db = Chroma.from_documents(
    #     texts,
    #     embeddings,
    #     persist_directory=PERSIST_DIRECTORY,
    #     client_settings=CHROMA_SETTINGS,
    # )
    # if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
    #     db = load_faiss_index(INDEX_PATH, METADATA_PATH)
    #     logging.info("Loaded FAISS index and metadata from disk.")
    # else:
        
    #     d = embeddings.shape[1]
    #     index = faiss.IndexFlatL2(d)
    #     index.add(embeddings)
        
    #     docstore = InMemoryDocstore()
    #     index_to_docstore_id = {i: doc["id"] for i, doc in enumerate(texts)}

    #     db = FAISS(index=index, docstore=docstore, index_to_docstore_id=index_to_docstore_id)
        
    #     save_faiss_index(db, INDEX_PATH, METADATA_PATH)
    #     logging.info("Saved FAISS index and metadata to disk.")

    # Load the model and tokenizer
    # model_name = EMBEDDING_MODEL_NAME
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModel.from_pretrained(model_name)
    # # Tokenize the input texts
    # inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    # # Get the embeddings from the model
    # with torch.no_grad():
    #     outputs = model(**inputs)
    # # Extract the last hidden states (embeddings)
    # embeddings = outputs.last_hidden_state
    # # Pool the embeddings (e.g., mean pooling)
    # pooled_embeddings = embeddings.mean(dim=1)
    # # Convert the embeddings to a NumPy array
    # numpy_embeddings = pooled_embeddings.cpu().numpy()
    
    # # Get the dimension of the vectors
    # vector_dimension = numpy_embeddings.shape[1]

    # Create the FAISS index
    # faiss_index = faiss.IndexFlatL2(vector_dimension)
    # print(faiss_index.is_trained)
    # # Add the embeddings to the index
    # faiss_index.add(numpy_embeddings)
    # # Save the index
    # faiss.write_index(faiss_index, index_file_path)
    # print(f"Index saved to {index_file_path}")
    # print(faiss_index.ntotal)
    
    # Define the directory and file name to save the index
    # persist_dir = PERSIST_DIRECTORY
    # index_file_path = os.path.join(persist_dir, 'faiss_index.index')
    
    # # Load the index to verify
    # faiss_index_loaded = faiss.read_index(index_file_path)
    # print(f"Index loaded from {index_file_path}")

    # Verify the loaded index
    # print(f"Number of vectors in the loaded index: {faiss_index_loaded.ntotal}")
    
    # db = FAISS.from_documents(
    #     texts,
    #     embeddings,
    #     # persist_directory=PERSIST_DIRECTORY,
    #     # client_settings=CHROMA_SETTINGS,
    #     )
    # db.save_local("DB/faiss")

import argparse

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    # parser = argparse.ArgumentParser(description="Ingest script for localGPT")
    # parser.add_argument("--device_type", type=str, required=True, help="Device type (cpu or gpu)")
    # args = parser.parse_args()
    main(device_type='cpu')#args.device_type)

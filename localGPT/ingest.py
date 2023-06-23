# localGPT/ingest.py
import logging

import click
from langchain.embeddings import (
    HuggingFaceEmbeddings,
    HuggingFaceInstructEmbeddings,
)
from langchain.vectorstores import Chroma

from localGPT import CHROMA_SETTINGS, PERSIST_DIRECTORY, SOURCE_DIRECTORY
from localGPT.document import load_documents, split_documents


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
    # Using model and types
    logging.info(f"Using Embedding Model: {embedding_model}")
    logging.info(f"Using Embedding Type: {embedding_type}")
    logging.info(f"Using Device Type: {device_type}")

    # Load documents and split them into chunks
    logging.info(f"Loading documents from {SOURCE_DIRECTORY}")

    documents = load_documents(SOURCE_DIRECTORY)
    texts = split_documents(documents)

    logging.info(f"Loaded {len(documents)} documents from {SOURCE_DIRECTORY}")
    logging.info(f"Split into {len(texts)} chunks of text")

    # Create embeddings
    # NOTE: Models should be abstracted to allow for plug n' play
    logging.info(f"Split into {len(texts)} chunks of text")
    if embedding_type == "HuggingFaceInstructEmbeddings":
        embeddings = HuggingFaceInstructEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": device_type},
        )
    elif embedding_type == "HuggingFaceEmbeddings":
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": device_type},
        )
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
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s",
        level=logging.INFO,
    )
    main()

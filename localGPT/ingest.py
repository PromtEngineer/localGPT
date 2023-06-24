# localGPT/ingest.py
import logging

import click
from langchain.vectorstores import Chroma

from localGPT import (
    CHROMA_SETTINGS,
    EMBEDDING_TYPES,
    PERSIST_DIRECTORY,
    SOURCE_DIRECTORY,
)
from localGPT.document import load_documents, split_documents


@click.command()
@click.option(
    "--source_directory",
    default=SOURCE_DIRECTORY,
    type=click.STRING,
    help=f"The path the documents are read from (default: {SOURCE_DIRECTORY})",
)
@click.option(
    "--persist_directory",
    default=PERSIST_DIRECTORY,
    type=click.STRING,
    help=f"The path the embeddings are written to (default: {PERSIST_DIRECTORY})",
)
# Default Instruct Model
#   You can also choose a smaller model.
#   Don't forget to change HuggingFaceInstructEmbeddings
#   to HuggingFaceEmbeddings in both ingest.py and run.py
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
    help="Instruct model to generate embeddings (default: hkunlp/instructor-large)",
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
def main(
    source_directory,
    persist_directory,
    embedding_model,
    embedding_type,
    device_type,
):
    # Using model and types
    logging.info(f"Using Embedding Model: {embedding_model}")
    logging.info(f"Using Embedding Type: {embedding_type}")
    logging.info(f"Using Device Type: {device_type}")

    # Load documents and split them into chunks
    logging.info(f"Loading documents from {source_directory}")

    documents = load_documents(source_directory)
    texts = split_documents(documents)

    logging.info(f"Loaded {len(documents)} documents from {source_directory}")
    logging.info(f"Split into {len(texts)} chunks of text")

    # Create embeddings
    # NOTE: Models should be abstracted to allow for plug n' play
    logging.info(f"Split into {len(texts)} chunks of text")
    if embedding_type in EMBEDDING_TYPES.keys():
        EmbeddingClass = EMBEDDING_TYPES[embedding_type]
        embeddings = EmbeddingClass(
            model_name=embedding_model,
            model_kwargs={"device": device_type},
        )
    else:
        raise ValueError(f"Invalid embeddings type provided: {embedding_type}")

    # Persist the embeddings to Chroma database
    db = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=persist_directory,
        client_settings=CHROMA_SETTINGS,
    )
    db.persist()
    db = None


if __name__ == "__main__":
    main()

"""
localGPT/ingest.py

This script provides functionality for ingesting documents, 
generating embeddings, and persisting them to the Chroma database.

Usage:
    $ python ingest.py [OPTIONS]

Options:
    --source_directory TEXT       The path where the documents are read from
                                  (default: SOURCE_DIRECTORY)
    --persist_directory TEXT      The path where the embeddings are written to
                                  (default: PERSIST_DIRECTORY)
    --embedding_model TEXT        The embedding model to use for generating embeddings
                                  (default: DEFAULT_EMBEDDING_MODEL)
    --embedding_type TEXT         The type of embeddings to use
                                  (default: DEFAULT_EMBEDDING_TYPE)
    --device_type TEXT            The device type to run on
                                  (default: DEFAULT_DEVICE_TYPE)

The script uses the provided options to load documents from the source 
directory, split them into chunks, generate embeddings using the specified 
embedding model and type, and persist the embeddings to the Chroma 
database located in the persist directory.

The default values for the options are set based on the configuration in 
the localGPT package.

You can specify different values for the options by providing the corresponding 
command-line arguments.

Example usage:
    $ python ingest.py \
            --source_directory path/to/documents \
            --persist_directory path/to/embeddings
"""

import logging

import click

from localGPT import (
    CHOICE_DEVICE_TYPES,
    CHOICE_EMBEDDING_MODELS,
    CHOICE_EMBEDDING_TYPES,
    DEFAULT_DEVICE_TYPE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_TYPE,
    PERSIST_DIRECTORY,
    SOURCE_DIRECTORY,
)
from localGPT.database import ChromaDBLoader
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
@click.option(
    "--embedding_model",
    default=DEFAULT_EMBEDDING_MODEL,
    type=click.Choice(CHOICE_EMBEDDING_MODELS),
    help="Instruct model to generate embeddings (default: hkunlp/instructor-large)",
)
@click.option(
    "--embedding_type",
    default=DEFAULT_EMBEDDING_TYPE,
    type=click.Choice(CHOICE_EMBEDDING_TYPES),
    help="Embedding type to use (default: HuggingFaceInstructEmbeddings)",
)
@click.option(
    "--device_type",
    default=DEFAULT_DEVICE_TYPE,
    type=click.Choice(CHOICE_DEVICE_TYPES),
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

    # Create ChromaDBLoader instance
    db_loader = ChromaDBLoader(
        source_directory=source_directory,
        persist_directory=persist_directory,
        embedding_model=embedding_model,
        embedding_type=embedding_type,
        device_type=device_type,
    )

    # Persist the embeddings to Chroma database
    db_loader.persist(texts)
    logging.info("Embeddings persisted to Chroma database.")


if __name__ == "__main__":
    main()

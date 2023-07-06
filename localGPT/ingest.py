"""
localGPT/ingest.py

This script provides functionality for ingesting documents,
generating embeddings, and persisting them to the Chroma database.

Usage:
    $ python ingest.py [OPTIONS]

Options:
    --path_documents TEXT       The path where the documents are read from
                                  (default: PATH_DOCUMENTS)
    --path_database TEXT      The path where the embeddings are written to
                                  (default: PATH_DATABASE)
    --repo_id TEXT        The embedding model to use for generating embeddings
                                  (default: HF_EMBEDDINGS_REPO_ID)
    --embeddings_class TEXT         The type of embeddings to use
                                  (default: LC_EMBEDDINGS_CLASS)
    --device_type TEXT            The device type to run on
                                  (default: TORCH_DEVICE_TYPE)

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
            --path_documents path/to/documents \
            --path_database path/to/embeddings
"""

import logging

import click

from localGPT import (
    CHOICE_LC_EMBEDDINGS_CLASSES,
    CHOICE_PT_DEVICE_TYPES,
    HF_EMBEDDINGS_REPO_ID,
    LC_EMBEDDINGS_CLASS,
    PATH_DATABASE,
    PATH_DOCUMENTS,
    TORCH_DEVICE_TYPE,
)
from localGPT.database.chroma import ChromaDBLoader
from localGPT.database.document import load_documents, split_documents


@click.command()
@click.option(
    "--path_documents",
    default=PATH_DOCUMENTS,
    type=click.STRING,
    help=f"The path the documents are read from (default: {PATH_DOCUMENTS})",
)
@click.option(
    "--path_database",
    default=PATH_DATABASE,
    type=click.STRING,
    help=f"The path the embeddings are written to (default: {PATH_DATABASE})",
)
@click.option(
    "--repo_id",
    default=HF_EMBEDDINGS_REPO_ID,
    type=click.STRING,
    help="The embeddings model repository id (default: hkunlp/instructor-large)",
)
@click.option(
    "--embeddings_class",
    default=LC_EMBEDDINGS_CLASS,
    type=click.Choice(CHOICE_LC_EMBEDDINGS_CLASSES),
    help="LangChain embeddings class to use for the model (default: HuggingFaceInstructEmbeddings)",
)
@click.option(
    "--device_type",
    default=TORCH_DEVICE_TYPE,
    type=click.Choice(CHOICE_PT_DEVICE_TYPES),
    help="Device to run on (default: cuda)",
)
def main(
    path_documents,
    path_database,
    repo_id,
    embeddings_class,
    device_type,
):
    """
    Ingest documents, generate embeddings, and persist them to the Chroma database.

    The script loads documents from the source directory, splits them into chunks,
    generates embeddings using the specified embedding model and type, and persists
    the embeddings to the Chroma database.

    Args:
        path_documents (str): The path where the documents are read from.
        path_database (str): The path where the embeddings are written to.
        repo_id (str): The embedding model to use for generating embeddings.
        embeddings_class (str): The type of embeddings to use for the model.
        device_type (str): The device type to run the embeddings on.

    Returns:
        None
    """

    # Using model and types
    logging.info(f"Using Embedding Model: {repo_id}")
    logging.info(f"Using Embedding Type: {embeddings_class}")
    logging.info(f"Using Device Type: {device_type}")

    # Load documents and split them into chunks
    logging.info(f"Loading documents from {path_documents}")

    documents = load_documents(path_documents)
    texts = split_documents(documents)

    logging.info(f"Loaded {len(documents)} documents from {path_documents}")
    logging.info(f"Split into {len(texts)} chunks of text")

    # Create ChromaDBLoader instance
    db_loader = ChromaDBLoader(
        path_documents=path_documents,
        path_database=path_database,
        repo_id=repo_id,
        embeddings_class=embeddings_class,
        device_type=device_type,
        settings=None,
    )

    # Persist the embeddings to Chroma database
    db_loader.persist(texts)
    logging.info("Embeddings persisted to Chroma database.")


if __name__ == "__main__":
    main()

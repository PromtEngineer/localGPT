# localGPT/run.py
import logging

import click

# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from localGPT import (
    CHOICE_DEVICE_TYPES,
    CHOICE_EMBEDDING_MODELS,
    CHOICE_EMBEDDING_TYPES,
    CHOICE_MODEL_REPOSITORIES,
    CHOICE_MODEL_SAFETENSORS,
    DEFAULT_DEVICE_TYPE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_TYPE,
    DEFAULT_MODEL_REPOSITORY,
    DEFAULT_MODEL_SAFETENSORS,
    PERSIST_DIRECTORY,
)
from localGPT.model import ModelLoader
from localGPT.database import ChromaDBLoader


@click.command()
@click.option(
    "--model_repository",
    default=DEFAULT_MODEL_REPOSITORY,
    type=click.Choice(CHOICE_MODEL_REPOSITORIES),
    help=f"The model repository (default: {DEFAULT_MODEL_REPOSITORY})",
)
@click.option(
    "--model_safetensors",
    default=DEFAULT_MODEL_SAFETENSORS,
    type=click.Choice(CHOICE_MODEL_SAFETENSORS),
    help=f"The model safetensors (default: {DEFAULT_MODEL_SAFETENSORS})",
)
@click.option(
    "--embedding_model",
    default=DEFAULT_EMBEDDING_MODEL,
    type=click.Choice(CHOICE_EMBEDDING_MODELS),
    help=f"The embedding model repository (default: {DEFAULT_EMBEDDING_MODEL})",
)
@click.option(
    "--embedding_type",
    default=DEFAULT_EMBEDDING_TYPE,
    type=click.Choice(CHOICE_EMBEDDING_TYPES),
    help=f"The embedding model type (default: {DEFAULT_EMBEDDING_TYPE})",
)
@click.option(
    "--device_type",
    default=DEFAULT_DEVICE_TYPE,
    type=click.Choice(CHOICE_DEVICE_TYPES),
    help="The compute device used by the model (default: cuda)",
)
@click.option(
    "--persist_directory",
    default=PERSIST_DIRECTORY,
    type=click.STRING,
    help=f"The embeddings database path (default: {PERSIST_DIRECTORY})",
)
@click.option(
    "--show_sources",
    type=click.BOOL,
    default=False,
    help="Display the documents source text (default: False)",
)
def main(
    model_repository,
    model_safetensors,
    embedding_model,
    embedding_type,
    device_type,
    persist_directory,
    show_sources,
):
    """
    This function implements the information retrieval task.

    1. Loads an embedding model, can be HuggingFaceInstructEmbeddings
       or HuggingFaceEmbeddings
    2. Loads the existing vectorestore that was created by ingest.py
    3. Loads the local LLM using load_model function - You can now set different LLMs.
    4. Setup the Question Answer retreival chain.
    5. Question answers.
    """

    # Create ChromaDBLoader instance
    db_loader = ChromaDBLoader(
        source_directory=None,
        persist_directory=persist_directory,
        embedding_model=embedding_model,
        embedding_type=embedding_type,
        device_type=device_type,
    )

    # load the LLM for generating Natural Language responses
    model_loader = ModelLoader(device_type, model_repository, model_safetensors)
    llm = model_loader.load_model()
    # Setup the Question Answer retrieval chain.
    qa = db_loader.load_retrieval_qa(llm)
    # Interactive questions and answers
    logging.info(f"Show Sources: {show_sources}")

    while True:
        query = input("\nEnter a query: ")
        if query.lower() == "exit":
            break
        # Get the answer from the chain
        res = qa(query)
        answer, docs = res["result"], res["source_documents"]

        # Print the result
        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(answer)

        if show_sources:
            # Print the relevant sources used for the answer
            print(
                "--------------------START-SOURCE-DOCUMENT--------------------"
            )
            for document in docs:
                print("\n> " + document.metadata["source"] + ":")
                print(document.page_content)
            print(
                "----------------------END-SOURCE-DOCUMENT--------------------"
            )


if __name__ == "__main__":
    main()

"""
localGPT/run.py

This script implements the information retrieval task using 
a Question-Answer retrieval chain.

The steps involved are as follows:
1. Load an embedding model, which can be 
   HuggingFaceInstructEmbeddings or HuggingFaceEmbeddings.
2. Load the existing vector store that was created by ingest.py.
3. Load the local LLM using the load_model function.
4. Set up the Question-Answer retrieval chain.
5. Prompt the user for questions and provide answers based on the 
   retrieval chain.

Usage:
    python run.py [OPTIONS]

Options:
    --model_repository TEXT      The model repository.
                                 Default: TheBloke/WizardLM-7B-V1.0-Uncensored-GGML
    --model_safetensors TEXT     The model safetensors.
                                 Default: 
                                 WizardLM-7B-uncensored-GPTQ-4bit-128g.compat.no-act-order.safetensors
    --embedding_model TEXT       The embedding model repository.
                                 Default: hkunlp/instructor-large
    --embedding_type TEXT        The embedding model type.
                                 Default: HuggingFaceInstructEmbeddings
    --device_type TEXT           The compute device used by the model.
                                 Choices: cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, 
                                          ideep, hip, ve, fpga, ort, xla, lazy, vulkan,
                                          mps, meta, hpu, mtia
                                 Default: cuda
    --persist_directory TEXT     The embeddings database path.
                                 Default: DB
    --show_sources / --no_show_sources
                                 Display the documents' source text.
                                 Default: False
"""

import logging
import click

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

# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


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
    Execute the information retrieval task using a Question-Answer retrieval chain.
    """
    # Create ChromaDBLoader instance
    db_loader = ChromaDBLoader(
        source_directory=None,
        persist_directory=persist_directory,
        embedding_model=embedding_model,
        embedding_type=embedding_type,
        device_type=device_type,
    )

    # Load the LLM for generating Natural Language responses
    model_loader = ModelLoader(device_type, model_repository, model_safetensors)
    llm = model_loader.load_model()

    # Setup the Question-Answer retrieval chain
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
            print("----START-SOURCE-DOCUMENT----")
            for document in docs:
                print(f"\n> {document.metadata['source']}:")
                print(document.page_content)
            print("----END-SOURCE-DOCUMENT----")


if __name__ == "__main__":
    main()

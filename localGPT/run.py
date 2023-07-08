"""
localGPT/run.py

This script implements the information retrieval task using
a Question-Answer retrieval chain.

Usage:
    python run.py [OPTIONS]

Options:
    --repo_id TEXT                 The model repository to load the LLM model from.
                                   Default: TheBloke/WizardLM-7B-V1.0-Uncensored-GGML
    --model_class TEXT             The model class to use.
                                   Default: LlamaCpp
    --safetensors TEXT             The model safetensors.
                                   Default: WizardLM-7B-uncensored-GPTQ-4bit-128g.compat.no-act-order.safetensors
    --embeddings_repo_id TEXT      The embedding model repository to use.
                                   Default: hkunlp/instructor-large
    --embeddings_class TEXT        The embedding model class to use.
                                   Default: HuggingFaceInstructEmbeddings
    --torch_device_type TEXT       The compute device used by the LLM model.
                                   Choices: cpu, cuda, ipu, xpu, mkldnn, opengl, opencl,
                                            ideep, hip, ve, fpga, ort, xla, lazy, vulkan,
                                            mps, meta, hpu, mtia
                                   Default: cuda
    --path_database TEXT           The path where the embeddings database is located.
                                   Default: DB
    --show_sources / --no_show_sources
                                   Display the source text of the retrieved documents.
                                   Default: False
    --use_triton / --no_use_triton
                                   Use AMD triton for CUDA backend.
                                   Default: False
"""

import logging

import click

from localGPT import (
    CHOICE_HF_EMBEDDINGS_REPO_IDS,
    CHOICE_HF_MODEL_SAFETENSORS,
    CHOICE_LC_EMBEDDINGS_CLASSES,
    CHOICE_LC_MODEL_CLASSES,
    CHOICE_PT_DEVICE_TYPES,
    HF_EMBEDDINGS_REPO_ID,
    HF_MODEL_REPO_ID,
    HF_MODEL_SAFETENSORS,
    LC_EMBEDDINGS_CLASS,
    LC_MODEL_CLASS,
    PATH_DATABASE,
    TORCH_DEVICE_TYPE,
)
from localGPT.database.chroma import ChromaDBLoader
from localGPT.model.loader import ModelLoader

# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


@click.command()
@click.option(
    "--repo_id",
    default=HF_MODEL_REPO_ID,
    type=click.STRING,
    help=f"The model repository to load the LLM model from. Default is {HF_MODEL_REPO_ID}",
)
@click.option(
    "--model_class",
    default=LC_MODEL_CLASS,
    type=click.Choice(CHOICE_LC_MODEL_CLASSES),
    help=f"The model class to use. Default is {LC_MODEL_CLASS}",
)
@click.option(
    "--safetensors",
    default=HF_MODEL_SAFETENSORS,
    type=click.Choice(CHOICE_HF_MODEL_SAFETENSORS),
    help=f"The model safetensors to use. Default is {HF_MODEL_SAFETENSORS}",
)
@click.option(
    "--embeddings_repo_id",
    default=HF_EMBEDDINGS_REPO_ID,
    type=click.Choice(CHOICE_HF_EMBEDDINGS_REPO_IDS),
    help=f"The embedding model repository to use. Default is {HF_EMBEDDINGS_REPO_ID}",
)
@click.option(
    "--embeddings_class",
    default=LC_EMBEDDINGS_CLASS,
    type=click.Choice(CHOICE_LC_EMBEDDINGS_CLASSES),
    help=f"The embedding model class to use. Default is {LC_EMBEDDINGS_CLASS}",
)
@click.option(
    "--torch_device_type",
    default=TORCH_DEVICE_TYPE,
    type=click.Choice(CHOICE_PT_DEVICE_TYPES),
    help=f"The compute device used by the LLM model. Default is {CHOICE_PT_DEVICE_TYPES}",
)
@click.option(
    "--path_database",
    default=PATH_DATABASE,
    type=click.STRING,
    help=f"The path where the embeddings database is located. Default is {PATH_DATABASE}",
)
@click.option(
    "--show_sources",
    type=click.BOOL,
    default=bool(),
    help="Display the source text of the retrieved documents. Default is False.",
)
@click.option(
    "--use_triton",
    type=click.BOOL,
    default=bool(),
    help="Use AMD triton for CUDA backend. Default is False.",
)
def main(
    model_class,
    repo_id,
    safetensors,
    embeddings_repo_id,
    embeddings_class,
    torch_device_type,
    path_database,
    show_sources,
    use_triton,
):
    """
    Execute the information retrieval task using a Question-Answer retrieval chain.

    Args:
        model_class (str): The model class to use.
        repo_id (str): The model repository to load the LLM model from.
        safetensors (str): The model safetensors to use.
        embeddings_repo_id (str): The embedding model repository to use.
        embeddings_class (str): The embedding model class to use.
        torch_device_type (str): The compute device used by the LLM model.
        path_database (str): The path where the embeddings database is located.
        show_sources (bool): Display the source text of the retrieved documents.
        use_triton (bool): Use AMD triton for CUDA backend.
    """

    # Create ChromaDBLoader instance
    db_loader = ChromaDBLoader(
        path_documents=None,
        path_database=path_database,
        repo_id=embeddings_repo_id,
        embeddings_class=embeddings_class,
        device_type=torch_device_type,
        settings=None,
    )

    # Load the LLM for generating Natural Language responses
    model_loader = ModelLoader(
        repo_id=repo_id,
        model_class=model_class,
        safetensors=safetensors,
        device_type=torch_device_type,
        use_triton=use_triton,
    )

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

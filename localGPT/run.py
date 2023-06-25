# localGPT/run.py
import logging

import click

# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from localGPT import (
    DEFAULT_DEVICE_TYPE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_TYPE,
    DEFAULT_MODEL_REPOSITORY,
    DEFAULT_MODEL_SAFETENSORS,
    PERSIST_DIRECTORY,
)
from localGPT.model import ModelLoader
from localGPT.database import ChromaDBLoader


# NOTE:
# If the safetensors are required for certain models and
# can cause failures if not provided, it's important to
# handle this in the code.
# One approach could be to still allow the user to specify
# the safetensors, but provide clear documentation and error
# messages to guide them.
# For example, when a user selects a model, we could display
# a message indicating whether safetensors are required for
# that model and how to find the correct ones.
# e.g.
#   ~/.cache/huggingface/hub
#   ~/.cache/torch/sentence_transformers
# If a user tries to load a model without providing the
# necessary safetensors, we could catch the resulting error
# and display a helpful message explaining what went wrong
# and how to fix it.
@click.command()
@click.option(
    "--model_repository",
    default=DEFAULT_MODEL_REPOSITORY,
    type=click.Choice(
        [
            "TheBloke/vicuna-7B-1.1-HF",
            "TheBloke/vicuna-7B-1.1-GPTQ-4bit-128g",
            "TheBloke/vicuna-7B-1.1-GGML",
            "TheBloke/wizardLM-7B-HF",
            "TheBloke/WizardLM-7B-V1.0-Uncensored-GPTQ",
            "TheBloke/WizardLM-7B-V1.0-Uncensored-GGML",
            "NousResearch/Nous-Hermes-13b",
            "TheBloke/Nous-Hermes-13B-GPTQ",
            "TheBloke/Nous-Hermes-13B-GGML",
        ]
    ),
    help=f"The model repository (default: {DEFAULT_MODEL_REPOSITORY})",
)
@click.option(
    "--model_safetensors",
    default=DEFAULT_MODEL_SAFETENSORS,
    type=click.Choice(
        [
            "WizardLM-7B-uncensored-GPTQ-4bit-128g.compat.no-act-order.safetensors",
            "wizardLM-7B-GPTQ-4bit.compat.no-act-order.safetensors",
            "nous-hermes-13b-GPTQ-4bit-128g.no-act.order.safetensors",
            "WizardLM-30B-Uncensored-GPTQ-4bit.act-order.safetensors",
        ]
    ),
    help=f"The model safetensors (default: {DEFAULT_MODEL_SAFETENSORS})",
)
@click.option(
    "--embedding_model",
    default=DEFAULT_EMBEDDING_MODEL,
    type=click.Choice(
        [
            "hkunlp/instructor-base",
            "hkunlp/instructor-large",
            "hkunlp/instructor-xl",
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-MiniLM-L12-v2",
        ]
    ),
    help=f"The embedding model repository (default: {DEFAULT_EMBEDDING_MODEL})",
)
@click.option(
    "--embedding_type",
    default=DEFAULT_EMBEDDING_TYPE,
    type=click.Choice(
        [
            "HuggingFaceEmbeddings",
            "HuggingFaceInstructEmbeddings",
        ]
    ),
    help=f"The embedding model type (default: {DEFAULT_EMBEDDING_TYPE})",
)
@click.option(
    "--device_type",
    default=DEFAULT_DEVICE_TYPE,
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

    # load the vectorstore
    retriever = db_loader.load_retriever()

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

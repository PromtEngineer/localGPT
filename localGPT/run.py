# localGPT/run.py
import logging

import click
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings

# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma

from localGPT import (
    CHROMA_SETTINGS,
    DEFAULT_DEVICE_TYPE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_BASE_NAME,
    PERSIST_DIRECTORY,
)
from localGPT.model import ModelLoader


@click.command()
@click.option(
    "--model_id",
    default=DEFAULT_MODEL_ID,
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
    help=f"The models git repository endpoint (default: {DEFAULT_MODEL_ID})",
)
@click.option(
    "--model_basename",
    default=DEFAULT_MODEL_BASE_NAME,
    type=click.Choice(
        [
            "WizardLM-7B-uncensored-GPTQ-4bit-128g.compat.no-act-order.safetensors",
            "wizardLM-7B-GPTQ-4bit.compat.no-act-order.safetensors",
            "nous-hermes-13b-GPTQ-4bit-128g.no-act.order.safetensors",
            "WizardLM-30B-Uncensored-GPTQ-4bit.act-order.safetensors",
        ]
    ),
    help="Embedding type to use (default: HuggingFaceInstructEmbeddings)",
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
    help="Instruct model to generate embeddings (default: hkunlp/instructor-large)",
)
@click.option(
    "--persist_directory",
    default=PERSIST_DIRECTORY,
    type=click.STRING,
    help=f"The path the embeddings are written to (default: {PERSIST_DIRECTORY})",
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
    help="Device to run on. (Default is cuda)",
)
@click.option(
    "--show_sources",
    type=click.BOOL,
    default=False,
    help="Show sources along with answers (Default is False)",
)
def main(model_id, model_basename, embedding_model, device_type, show_sources):
    """
    This function implements the information retrieval task.

    1. Loads an embedding model, can be HuggingFaceInstructEmbeddings
       or HuggingFaceEmbeddings
    2. Loads the existing vectorestore that was created by inget.py
    3. Loads the local LLM using load_model function - You can now set different LLMs.
    4. Setup the Question Answer retreival chain.
    5. Question answers.
    """

    model_loader = ModelLoader(device_type, model_id, model_basename)

    logging.info(f"Running on: {device_type}")
    logging.info(f"Display Source Documents set to: {show_sources}")

    embeddings = HuggingFaceInstructEmbeddings(
        model_name=embedding_model, model_kwargs={"device": device_type}
    )

    # load the vectorstore
    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS,
    )
    retriever = db.as_retriever()

    # load the LLM for generating Natural Language responses
    llm = model_loader.load_model()

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    # Interactive questions and answers
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
                "----------------------------SOURCE-DOCUMENTS----------------------------"
            )
            for document in docs:
                print("\n> " + document.metadata["source"] + ":")
                print(document.page_content)
            print(
                "----------------------------SOURCE-DOCUMENTS----------------------------"
            )


if __name__ == "__main__":
    main()

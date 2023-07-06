"""
localGPT/ggml.py

This script provides functionality for interacting with the LlamaCpp model and performing document retrieval and question answering tasks.

Documentation:
- LlamaCpp: https://python.langchain.com/docs/modules/model_io/models/llms/integrations/llamacpp.html
- Prompt ChromaDBLoader: https://python.langchain.com/docs/modules/chains/popular/vector_db_qa.html
- Chat ChromaDBLoader: https://python.langchain.com/docs/modules/chains/popular/chat_vector_db
"""

import os
import sys

import click
from huggingface_hub import hf_hub_download
from langchain import LLMChain, PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp
from langchain.memory import ConversationBufferMemory

from localGPT import (
    CHOICE_HF_EMBEDDINGS_REPO_IDS,
    CHOICE_LC_EMBEDDINGS_CLASSES,
    CHOICE_PT_DEVICE_TYPES,
    GGML_FILENAME,
    GGML_LOW_VRAM,
    GGML_MAX_TOKENS,
    GGML_N_BATCH,
    GGML_N_CTX,
    GGML_N_GPU_LAYERS,
    GGML_REPO_ID,
    GGML_TEMPERATURE,
    GGML_TOP_P,
    HF_EMBEDDINGS_REPO_ID,
    LC_EMBEDDINGS_CLASS,
    PATH_DATABASE,
    PATH_HOME,
    SHOW_DOCUMENT_SOURCE,
    TORCH_DEVICE_TYPE,
    logging,
)
from localGPT.database.chroma import ChromaDBLoader


@click.command()
@click.option(
    "--repo_id",
    type=click.STRING,
    default=GGML_REPO_ID,
    help=f"The repository to download the model from. Default is {GGML_REPO_ID}",
)
@click.option(
    "--filename",
    type=click.STRING,
    default=GGML_FILENAME,
    help=f"The filename of the model from the given repository. Default is {GGML_FILENAME}.",
)
@click.option(
    "--prompt",
    type=click.STRING,
    default=str(),
    help="Query the model with a string. Default is an empty string.",
)
@click.option(
    "--chat",
    type=click.BOOL,
    default=bool(),
    help="Enable chat-like query loop with the model. Default is False.",
)
@click.option(
    "--n_ctx",
    type=click.INT,
    default=GGML_N_CTX,
    help=f"Maximum context size. Default is {GGML_N_CTX}.",
)
@click.option(
    "--n_batch",
    type=click.INT,
    default=GGML_N_BATCH,
    help=f"Number of batches to use. Default is {GGML_N_BATCH}.",
)
@click.option(
    "--n_gpu_layers",
    type=click.INT,
    default=GGML_N_GPU_LAYERS,
    help="Number of GPU layers to use. Default is 0.",
)
@click.option(
    "--f16_kv",
    type=click.BOOL,
    default=bool(),
    help="Use half-precision for key/value cache. Default is False.",
)
@click.option(
    "--low_vram",
    type=click.BOOL,
    default=GGML_LOW_VRAM,
    help="Set to True if the GPU device has low VRAM. Default is False.",
)
@click.option(
    "--max_tokens",
    type=click.INT,
    default=GGML_MAX_TOKENS,
    help=f"The maximum number of tokens to generate. Default is {GGML_MAX_TOKENS}.",
)
@click.option(
    "--temperature",
    type=click.FLOAT,
    default=GGML_TEMPERATURE,
    help=f"The temperature to use for sampling. Default is {GGML_TEMPERATURE}.",
)
@click.option(
    "--top_p",
    type=click.FLOAT,
    default=GGML_TOP_P,
    help=f"The top-p value to use for sampling. Default is {GGML_TOP_P}.",
)
@click.option(
    "--embeddings_repo_id",
    default=HF_EMBEDDINGS_REPO_ID,
    type=click.Choice(CHOICE_HF_EMBEDDINGS_REPO_IDS),
    help=f"The embedding model repository to use. default: {HF_EMBEDDINGS_REPO_ID})",
)
@click.option(
    "--embeddings_class",
    default=LC_EMBEDDINGS_CLASS,
    type=click.Choice(CHOICE_LC_EMBEDDINGS_CLASSES),
    help=f"The embedding model class to use. (default: {LC_EMBEDDINGS_CLASS})",
)
@click.option(
    "--embeddings_device_type",
    default=TORCH_DEVICE_TYPE,
    type=click.Choice(CHOICE_PT_DEVICE_TYPES),
    help=f"The compute device used by the embedding model. (default: {TORCH_DEVICE_TYPE})",
)
@click.option(
    "--path_database",
    default=PATH_DATABASE,
    type=click.STRING,
    help=f"The path where the embeddings database is located. (default: {PATH_DATABASE})",
)
@click.option(
    "--show_sources",
    type=click.BOOL,
    default=SHOW_DOCUMENT_SOURCE,
    help="Display the source text of the retrieved documents. Default is False.",
)
def main(
    repo_id,
    filename,
    n_ctx,
    n_batch,
    n_gpu_layers,
    f16_kv,
    low_vram,
    max_tokens,
    temperature,
    top_p,
    prompt,
    chat,
    path_database,
    embeddings_device_type,
    embeddings_repo_id,
    embeddings_class,
    show_sources,
):
    """
    Run LlamaCpp model for document retrieval and question answering.

    Args:
        repo_id (str): The repository to download the model from.
        filename (str): The filename of the model from the given repository.
        n_ctx (int): Maximum context size.
        n_batch (int): Number of batches to use.
        n_gpu_layers (int): Number of GPU layers to use.
        low_vram (bool): Set to True if the GPU device has low VRAM.
        max_tokens (int): The maximum number of tokens to generate.
        temperature (float): The temperature to use for sampling.
        top_p (float): The top-p value to use for sampling.
        prompt (str): Query the model with a string.
        chat (bool): Enable chat-like query loop with the model.
        path_database (str): The path where the embeddings database is located.
        embeddings_device_type (str): The compute device used by the embedding model.
        embeddings_repo_id (str): The embedding model repository to use.
        embeddings_class (str): The embedding model class to use.
        show_sources (bool): Display the source text of the retrieved documents.
    """
    if not (bool(prompt) ^ bool(chat)):
        print("Use either --prompt or --chat, but not both. See --help for more information.")
        sys.exit(1)

    try:
        logging.info(f"Using model from {repo_id}")
        cache_dir = os.path.join(PATH_HOME, ".cache", "huggingface", "hub")
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
            resume_download=True,
        )
    except Exception as e:
        logging.error(f"Error downloading model: {e}")
        sys.exit(1)

    logging.info(f"Loading {repo_id} into memory from {model_path}")

    # Callbacks support token-wise streaming
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    # Initialize LlamaCpp
    llm = LlamaCpp(
        model_path=model_path,
        callback_manager=callback_manager,
        n_ctx=n_ctx,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        n_batch=n_batch,
        n_gpt_layers=n_gpu_layers,
        f16_kv=f16_kv,
        low_vram=low_vram,
        stop=[".", "\n", "[DONE]"],
        echo=True,
        verbose=False,
    )

    # Create ChromaDBLoader instance
    db_loader = ChromaDBLoader(
        path_documents=None,
        path_database=path_database,
        repo_id=embeddings_repo_id,
        embeddings_class=embeddings_class,
        device_type=embeddings_device_type,
        settings=None,
    )

    if prompt:
        """
        Query the model with a string and print the response.

        Args:
            prompt (str): The query string.
        """
        logging.info("----START-MODEL-GENERATION----")
        retrieval_qa = db_loader.load_retrieval_qa(llm)
        result = retrieval_qa({"query": prompt})
        print()
        logging.info("----END-MODEL-GENERATION----")
        if show_sources:
            """
            Print the relevant sources used for the answer.
            """
            logging.info("----START-SOURCE-DOCUMENT----")
            for document in result["source_documents"]:
                print(document.metadata["source"])
                print(document.page_content)
            logging.info("----END-SOURCE-DOCUMENT----")

    elif chat:
        """
        Enter a chat loop with the model.
        """
        logging.info("Starting QA loop...")
        chat_history = []
        retrieval_qa = db_loader.load_conversational_qa(llm)
        while True:
            try:
                query = input("query> ")
            except (EOFError, KeyboardInterrupt):
                sys.exit(0)

            print("Thinking...")

            try:
                result = retrieval_qa({"question": query, "chat_history": chat_history})
                print()
            except TypeError as e:
                logging.error(e)
                exit(1)

            chat_history.append((query, result["answer"]))

            if show_sources:
                """
                Print the relevant sources used for the answer.
                """
                logging.info("----START-SOURCE-DOCUMENT----")
                for document in result["source_documents"]:
                    print(document.metadata["source"])
                    print(document.page_content)
                logging.info("----END-SOURCE-DOCUMENT----")


if __name__ == "__main__":
    main()

"""
localGPT/ggml.py

Docs:
    https://python.langchain.com/docs/modules/chains/popular/vector_db_qa.html
    https://python.langchain.com/docs/modules/chains/popular/chat_vector_db
"""
import os
import sys
from typing import List

import click
from huggingface_hub import hf_hub_download
from langchain import LLMChain, PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.llms import LlamaCpp
from llama_cpp import ChatCompletionMessage, Llama

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
    help="Query the model with a string. Default is ''",
)
@click.option(
    "--chat",
    type=click.BOOL,
    default=bool(),
    help="Chat-like query loop with the model. Default is False",
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
    "--low_vram",
    type=click.BOOL,
    default=GGML_LOW_VRAM,
    help="Set to True if GPU device has low VRAM. Default is False.",
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
    help=f"The embedding model repository (default: {HF_EMBEDDINGS_REPO_ID})",
)
@click.option(
    "--embeddings_class",
    default=LC_EMBEDDINGS_CLASS,
    type=click.Choice(CHOICE_LC_EMBEDDINGS_CLASSES),
    help=f"The embedding model type (default: {LC_EMBEDDINGS_CLASS})",
)
@click.option(
    "--embeddings_device_type",
    default=TORCH_DEVICE_TYPE,
    type=click.Choice(CHOICE_PT_DEVICE_TYPES),
    help=f"The compute device used by the model (default: {TORCH_DEVICE_TYPE})",
)
@click.option(
    "--path_database",
    default=PATH_DATABASE,
    type=click.STRING,
    help=f"The embeddings database path (default: {PATH_DATABASE})",
)
@click.option(
    "--show_sources",
    type=click.BOOL,
    default=SHOW_DOCUMENT_SOURCE,
    help="Display the documents source text (default: False)",
)
def main(
    repo_id,
    filename,
    n_ctx,
    n_batch,
    n_gpu_layers,
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
    if not (bool(prompt) ^ bool(chat)):
        print(
            "Use either --prompt or --chat, but not both.",
            "See --help for more information.",
            sep="\n",
        )
        exit(1)

    cache_dir = os.path.join(PATH_HOME, ".cache", "huggingface", "hub")

    logging.info(f"Using {repo_id} to load {filename}")

    try:
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
        )
    except Exception as e:
        logging.error(f"Error downloading model: {e}")
        sys.exit(1)

    logging.info(f"Using {model_path} to load {repo_id} into memory")

    # Callbacks support token-wise streaming
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    # Make sure the model path is correct for your system!
    llm = LlamaCpp(
        model_path=model_path,
        callback_manager=callback_manager,
        n_ctx=n_ctx,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        n_batch=n_batch,
        n_gpt_layers=n_gpu_layers,
        low_vram=low_vram,
        stop=["[DONE]\n", ".\n", "\n"],
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

    retrieval_qa = db_loader.load_retrieval_qa(llm)

    try:
        if prompt:
            logging.info("----START-MODEL-GENERATION----")
            source = retrieval_qa({"query": prompt})
            print()
            logging.info("----END-MODEL-GENERATION----")
            if show_sources:
                # Print the relevant sources used for the answer
                logging.info("----START-SOURCE-DOCUMENT----")
                logging.info(source["result"])
                logging.info("----END-SOURCE-DOCUMENT----")
        elif chat:
            logging.info("Starting QA loop...")
            # Enter a chat loop
            pass
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

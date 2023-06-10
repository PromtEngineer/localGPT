import click
import torch
import logging
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from transformers import LlamaForCausalLM, LlamaTokenizer, pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from constants import CHROMA_SETTINGS, PERSIST_DIRECTORY, ROOT_DIRECTORY
from transformers import GenerationConfig


def load_model(device_type):
    """
    Select a model on huggingface.
    If you are running this for the first time, it will download a model for you.
    subsequent runs will use the model from the disk.
    """
    # The code supports all huggingface models that ends with -HF or which have a .bin file in their HF repo.
    model_id = "TheBloke/vicuna-7B-1.1-HF"
    # model_id = "TheBloke/guanaco-7B-HF"
    # model_id = 'NousResearch/Nous-Hermes-13b'
    logging.info(f'Loading Model: {model_id}, on : {device_type}')
    logging.info(f'This action can take a few minutes!')

    if device_type.lower() == 'cuda':
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        logging.info(f'Tokenizer loaded')

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map='auto',
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        model.tie_weights()
    else:
        tokenizer = LlamaTokenizer.from_pretrained(model_id)
        model = LlamaForCausalLM.from_pretrained(model_id, )

    # load configuration from the model to avoid warnings.
    generation_config = GenerationConfig.from_pretrained(model_id)
    # see here for details: https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig.from_pretrained.returns

    # create pipeline for text generation
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=2048,
        temperature=0,
        top_p=0.95,
        repetition_penalty=1.15,
        generation_config=generation_config
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    logging.info(f'Local LLM Loaded')

    return local_llm


# chose device typ to run on as well as to show source documents.
@click.command()
@click.option(
    "--device_type",
    default="cuda",
    type=click.Choice(
        [
            "cpu", "cuda", "ipu", "xpu", "mkldnn", "opengl", "opencl", "ideep", "hip", "ve", "fpga", "ort",
            "xla", "lazy", "vulkan", "mps", "meta", "hpu", "mtia",
        ],
    ),
    help="Device to run on. (Default is cuda)",
)
@click.option(
    "--show_sources",
    default=True,
    type=click.Choice(
        [
            False,
            True,
        ]
    ),
    help="Show sources along with answers (Default is Fals)",
)
def main(device_type, show_sources):
    '''
    This function implements the information retreival task.


    1. Loads an embedding model, can be HuggingFaceInstructEmbeddings or HuggingFaceEmbeddings
    2. Loads the existing vectorestore that was created by inget.py
    3. Loads the local LLM using load_model function - You can now set different LLMs.
    4. Setup the Question Answer retreival chain.
    5. Question answers.
    '''

    logging.info(f'Running on: {device_type}')
    logging.info(f'Display Source Documents set to: {show_sources}')

    embeddings = HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-large", model_kwargs={"device": device_type}
    )

    # uncomment the following line if you used HuggingFaceEmbeddings in the ingest.py
    # embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

    # load the vectorstore
    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS,
    )
    retriever = db.as_retriever()

    # load the LLM for generating Natural Language responses.
    llm = load_model(device_type)

    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
    )
    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        # Get the answer from the chain
        res = qa(query)
        answer, docs = res["result"], res["source_documents"]

        # Print the result
        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(answer)

        if show_sources:  # this is a flag that you can set to disable showing answers.
            # # Print the relevant sources used for the answer
            print(
                "----------------------------------SOURCE DOCUMENTS---------------------------"
            )
            for document in docs:
                print("\n> " + document.metadata["source"] + ":")
                print(document.page_content)
            print(
                "----------------------------------SOURCE DOCUMENTS---------------------------"
            )


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s',
                        level=logging.INFO)
    main()

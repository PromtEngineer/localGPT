#!/usr/bin/python3
import sys

import click
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from transformers import LlamaTokenizer, LlamaForCausalLM, pipeline
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel

from constants import *  # CHROMA_SETTINGS, PERSIST_DIRECTORY



def load_model():
    """
    Select a model on huggingface.
    If you are running this for the first time, it will download a model for you.
    subsequent runs will use the model from the disk.
    """

    # Use local model
    #model_id = Path(MODEL_DIRECTORY) / LOCAL_MODEL_NAME            # path like
    #model_id = os.path.join("..", "vicuna-7B-1.1-HF" )                          # string

    # download model from HF
    #model_id = "vicuna/ggml-old-vic7b-uncensored-q5_0"  # recommendation for uncensored model
    #model_id = "TheBloke/vicuna-7B-1.1-HF"  # recommendation form prompt-engineer
    model_id = VICUNA_DIRECTORY
    print(f"model_id={model_id}")
    """
    OSError: We couldn't connect to 'https://huggingface.co' to load this file, couldn't find it in the cached files 
    and it looks like TheBloke/vicuna-7B-1.1-HF is not the path to a directory containing a file 
    named pytorch_model-00001-of-00002.bin.
    """

    # Define a tokenizer to split a sentence into tokens
    # NOTE: Does it need a relative path?
    tokenizer = LlamaTokenizer.from_pretrained(model_id,
                                               # local_files_only=True
                                               )
    print(f"tokenizer={tokenizer}")

    # What happens here?
    model = LlamaForCausalLM.from_pretrained(model_id,
                                             #   load_in_8bit=True, # set these options if your GPU supports them!
                                             #   device_map=1#'auto',
                                             #   torch_dtype=torch.float16,
                                                low_cpu_mem_usage=True,
                                             #   local_files_only=True
                                             )

    # ERROR CODE: Process finished with exit code 137 (interrupted by signal 9: SIGKILL)
    #  If you didn't manually stop the script and still got this error code, then the script was killed by your OS. In most of the cases, it is caused by excessive memory usage.

    print(f"model={model}")


    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=2048,
        temperature=0,  # adjust 0-to-100 for precise-to-creative
        top_p=0.95,
        repetition_penalty=1.15
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)

    return local_llm


def load_model_2():
    """
    Select a model on huggingface.
    If you are running this for the first time, it will download a model for you.
    subsequent runs will use the model from the disk.
    """

    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_NAME)
    print(f"tokenizer={tokenizer}")


    model = AutoModel.from_pretrained("bert-base-uncased")
    #model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
    #print(f"model of type {type(model)}: {model}")
    print(f"model: {model}")

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=2048,
        temperature=0,  # adjust 0-to-100 for precise-to-creative
        top_p=0.95,
        repetition_penalty=1.15
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)

    return local_llm



@click.command()
#@click.option('--device_type', default='gpu', help='device to run on, select gpu or cpu')
@click.option('--device_type', default='cpu', help='device to run on, select gpu or cpu')
def main(device_type, ):
    """ load the instructorEmbeddings """
    if device_type.lower() in ['cpu']:
        device = 'cpu'
    else:
        device = 'cuda'

    print(f"Running on: {device.upper()}")

    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl", model_kwargs={"device": device})
    # load the vectorstore
    db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()

    # Prepare the LLM
    # callbacks = [StreamingStdOutCallbackHandler()]

    # load the LLM for generating Natural Language responses. 
    llm = load_model()
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

    # Interactive questions and answers
    while True:
        print("===== write 'exit' to leave the query interface =====")
        query = input("\nEnter a query: ")
        if query == "exit":
            break

        # Get the answer from the chain
        res = qa(query)
        answer, docs = res['result'], res['source_documents']

        # Print the result
        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(answer)

        # # Print the relevant sources used for the answer
        line = "-"  # line separator for answer
        print(line * 3 + "SOURCE DOCUMENTS" + line * 3)
        for document in docs:
            print("\n> " + document.metadata["source"] + ":")
            print(document.page_content)
        print(line * 3 + "SOURCE DOCUMENTS" + line * 3)


if __name__ == "__main__":
    main()






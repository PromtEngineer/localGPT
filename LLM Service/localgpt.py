import subprocess
import logging
import os

from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory

from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceBgeEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler  # for streaming response
from langchain.callbacks.manager import CallbackManager

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

from prompt_template_utils import get_prompt_template

# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from transformers import (
    pipeline
)

from load_models import (
    load_quantized_model_awq,
    load_quantized_model_gguf_ggml,
    load_quantized_model_qptq,
    load_full_model,
)

from config import (
    CHROMA_SETTINGS,
    EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY,
    DEVICE_TYPE,
    MODEL_ID,
    MODEL_BASENAME,
    MAX_NEW_TOKENS,
)

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO)

if "instructor" in EMBEDDING_MODEL_NAME:
    EMBEDDINGS = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": DEVICE_TYPE})
elif "bge" in EMBEDDING_MODEL_NAME:
    EMBEDDINGS = HuggingFaceBgeEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": DEVICE_TYPE},
        encode_kwargs={'normalize_embeddings': True}
    )
# uncomment the following line if you used HuggingFaceEmbeddings in the ingest.py
# EMBEDDINGS = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

class TextQAEngine:
    def __init__(self):
        logging.info("Initializing localgpt")
        self.QA = None
        self.LLM = None
        self.MODEL_LOADED = False

    def ingest(self, db_name=None):
        logging.info("Executing ingest.py")

        # Prepare the command
        command = ["python", "ingest.py"]
        # If db_name is provided and is a string, add it as an argument
        if db_name and isinstance(db_name, str):
            command.extend(["--db_name", db_name])

        # Execute the command
        result = subprocess.run(command, capture_output=True)

        if result.returncode != 0:
            return "Script execution failed: {}".format(result.stderr.decode("utf-8")), 500
        else:
            return "Script executed successfully", 200


    def load_model(self, model_id = MODEL_ID, model_basename= MODEL_BASENAME, LOGGING=logging):
        """
        Select a model for text generation using the HuggingFace library.
        If you are running this for the first time, it will download a model for you.
        subsequent runs will use the model from the disk.
        persist_directory
                Args:
                    device_type (str): Type of device to use, e.g., "cuda" for GPU or "cpu" for CPU.
                    model_id (str): Identifier of the model to load from HuggingFace's model hub.
                    model_basename (str, optional): Basename of the model if using quantized models.
                        Defaults to None.

        Returns:
            HuggingFacePipeline: A pipeline object for text generation using the loaded model.

        Raises:
            ValueError: If an unsupported model or device type is provided.
        """
        logging.info(f"Loading Model: {model_id}, on: {DEVICE_TYPE}")
        logging.info("This action can take a few minutes!")

        if model_basename is not None:
            if ".gguf" in model_basename.lower():
                self.LLM = load_quantized_model_gguf_ggml(model_id, model_basename, DEVICE_TYPE, LOGGING)
                return
            elif ".ggml" in model_basename.lower():
                model, tokenizer = load_quantized_model_gguf_ggml(model_id, model_basename, DEVICE_TYPE, LOGGING)
            elif ".awq" in model_basename.lower():
                model, tokenizer = load_quantized_model_awq(model_id, LOGGING)
            else:
                model, tokenizer = load_quantized_model_qptq(model_id, DEVICE_TYPE, LOGGING)
        else:
            model, tokenizer = load_full_model(model_id, model_basename, DEVICE_TYPE, LOGGING)

        # Create a pipeline for text generation

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            repetition_penalty=1.15
        )

        self.LLM = HuggingFacePipeline(pipeline=pipe)
        logging.info("Local LLM Loaded")


    def load_QA(self, use_history=False, promptTemplate_type="ChatML", db_name = None, conversation_history = []):
        """
        Initializes and returns a retrieval-based Question Answering (QA) pipeline.

        This function sets up a QA system that retrieves relevant information using embeddings
        from the HuggingFace library. It then answers questions based on the retrieved information.

        Parameters:
        - device_type (str): Specifies the type of device where the model will run, e.g., 'cpu', 'cuda', etc.
        - use_history (bool): Flag to determine whether to use chat history or not.

        Returns:
        - RetrievalQA: An initialized retrieval-based QA system.

        Notes:
        - The function uses embeddings from the HuggingFace library, either instruction-based or regular.
        - The Chroma class is used to load a vector store containing pre-computed embeddings.
        - The retriever fetches relevant documents or data based on a query.
        - The prompt and memory, obtained from the `get_prompt_template` function, might be used in the QA system.
        - The model is loaded onto the specified device using its ID and basename.
        - The QA system retrieves relevant documents using the retriever and then answers questions based on those documents.
        """

        # get the prompt template and memory if set by the user.
        prompt, memory = get_prompt_template(promptTemplate_type=promptTemplate_type, conversation_history = conversation_history)

        if db_name == None:
            self.QA = ConversationChain(
                llm=self.LLM, 
                verbose=False,
                prompt = prompt,
                memory=ConversationBufferMemory()
            )
        else:
            persist_directory = os.path.join(PERSIST_DIRECTORY, db_name)
            CHROMA_SETTINGS.persist_directory = persist_directory

            # load the vectorstore
            db = Chroma(
                persist_directory=persist_directory,
                embedding_function=EMBEDDINGS,
                client_settings=CHROMA_SETTINGS,
            )
            retriever = db.as_retriever(search_kwargs={"k": 6})
            # retriever = db.as_retriever()

            if use_history:
                self.QA = RetrievalQA.from_chain_type(
                    llm=self.LLM,
                    chain_type="stuff",  # try other chains types as well. refine, map_reduce, map_rerank
                    retriever=retriever,
                    verbose = True,
                    return_source_documents=True,  # verbose=True,
                    callbacks=callback_manager,
                    chain_type_kwargs={"prompt": prompt, "memory": memory},
                )
            else:
                self.QA = RetrievalQA.from_chain_type(
                    llm=self.LLM,
                    chain_type="stuff",  # try other chains types as well. refine, map_reduce, map_rerank
                    retriever=retriever,
                    return_source_documents=True,  # verbose=True,
                    callbacks=callback_manager,
                    chain_type_kwargs={
                        "prompt": prompt,
                    },
                )

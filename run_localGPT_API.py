import os
import shutil
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
import json

from constants import (
    SOURCE_DIRECTORY
)

# API queue
from threading import Lock
llm_request_lock = Lock()
ingest_request_lock = Lock()
ocr_request_lock = Lock()

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image

model_id = "llava-hf/llava-1.5-7b-hf"
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True,
    load_in_4bit=True,
)
processor = AutoProcessor.from_pretrained(model_id)

from paddleocr import PaddleOCR
# Paddleocr supports Chinese, English, French, German, Korean and Japanese.
# You can set the parameter `lang` as `ch`, `en`, `fr`, `german`, `korean`, `japan`
# to switch the language model in order.
paddle_ocr = PaddleOCR(lang='en') # need to run only once to download and load model into memory

def perform_ocr(img_path, filename):

    # Perform OCR on the image
    result = paddle_ocr.ocr(img_path, cls=False)

    # Initialize an empty string to store extracted text
    # extracted_text = f"Document type: Image\nDocument name: {filename}\nDocument text: "
    extracted_text = "Text in Image: "

    # Iterate through the results and concatenate text
    if result != [None]:
        for res in result:
            for line in res:
                extracted_text += line[-1][0] + "\n"
    else:
        extracted_text += "No text was found\n"

    prompt = "USER: <image>\nDescribe the type of image. Also describe in detail what is in the image.\nASSISTANT:"
    raw_image = Image.open(img_path)
    inputs = processor(prompt, raw_image, return_tensors='pt').to(0, torch.float16)

    output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    extracted_text += "Image Description: "
    extracted_text += processor.decode(output[0], skip_special_tokens=True)[len(prompt)-5:]

    # Save the extracted text to a text file but still with the original file extension
    with open(img_path, "w") as file:
        file.write(extracted_text)
    
    # try:
    #     os.remove(img_path)
    #     print(f"{img_path} has been successfully removed.")
    # except OSError as e:
    #     print(f"Error: {e}")

    return "OCR Completed and saved to ocr_output.txt"

app = Flask(__name__)

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

from constants import (
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
    
logging.info("Initializing localgpt")
QA = None
LLM = None

def ingest(db_name=None):
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

def load_model(model_id = MODEL_ID, model_basename= MODEL_BASENAME, LOGGING=logging):
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
    global LLM

    logging.info(f"Loading Model: {model_id}, on: {DEVICE_TYPE}")
    logging.info("This action can take a few minutes!")

    if model_basename is not None:
        if ".gguf" in model_basename.lower():
            LLM = load_quantized_model_gguf_ggml(model_id, model_basename, DEVICE_TYPE, LOGGING)
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

    LLM = HuggingFacePipeline(pipeline=pipe)
    logging.info("Local LLM Loaded")

def load_QA(use_history=False, promptTemplate_type="ChatML", db_name = None, conversation_history = []):
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

    global QA
    global LLM

    # get the prompt template and memory if set by the user.
    prompt, memory = get_prompt_template(promptTemplate_type=promptTemplate_type, conversation_history = conversation_history)

    if db_name == None:
        QA = ConversationChain(
            llm=LLM, 
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
            QA = RetrievalQA.from_chain_type(
                llm=LLM,
                chain_type="stuff",  # try other chains types as well. refine, map_reduce, map_rerank
                retriever=retriever,
                verbose = True,
                return_source_documents=True,  # verbose=True,
                callbacks=callback_manager,
                chain_type_kwargs={"prompt": prompt, "memory": memory},
            )
        else:
            QA = RetrievalQA.from_chain_type(
                llm=LLM,
                chain_type="stuff",  # try other chains types as well. refine, map_reduce, map_rerank
                retriever=retriever,
                return_source_documents=True,  # verbose=True,
                callbacks=callback_manager,
                chain_type_kwargs={
                    "prompt": prompt,
                },
            )

load_model()

@app.route("/api/delete_source/<db_name>", methods=["POST"])
def delete_source_route(db_name):
    source_directory = SOURCE_DIRECTORY
    source_directory = os.path.join(source_directory, db_name)

    if os.path.exists(source_directory):
        shutil.rmtree(source_directory)

    return jsonify({"message": f"Folder '{source_directory}' successfully deleted and recreated."})

@app.route("/api/save_document/<db_name>", methods=["POST"])  # Use POST method
def save_document_route(db_name):
    global ocr_request_lock

    # Ensure there is a file in the request
    if 'document' not in request.files:
        return jsonify({"error": "No document part"}), 400

    file = request.files['document']
    
    # If the user does not select a file, the browser submits an empty file without a filename
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        filename = secure_filename(file.filename)
        source_directory = os.path.join(SOURCE_DIRECTORY, db_name)
        
        # Create the directory if it does not exist
        if not os.path.exists(source_directory):
            os.makedirs(source_directory)

        file_path = os.path.join(source_directory, filename)
        file.save(file_path)

        if file_path.split('.')[-1] in ['png', 'jpg']:
            with ocr_request_lock:
                perform_ocr(file_path, filename)
        
        return jsonify({"message": "File saved successfully", "file_path": file_path}), 200

    else:
        return jsonify({"error": "Invalid file"}), 400

@app.route("/api/run_ingest/<db_name>", methods=["POST"])
def run_ingest_route(db_name):
    global ingest_request_lock
    try:
        with ingest_request_lock:
            ingest(db_name=db_name)
        return "Script executed successfully", 200
    except Exception as e:
        return f"Error occurred: {str(e)}", 500

@app.route("/api/prompt_route", methods=["GET", "POST"])
def prompt_route():
    global llm_request_lock # Make sure to use the global lock instance

    db_name = request.form.get("db_name")
    user_prompt = request.form.get("user_prompt")
    conversation_history_str = request.form.get("conversation_history")
    
    # Convert the string back to a list of dictionaries
    conversation_history = json.loads(conversation_history_str)
    
    with llm_request_lock: # Acquire the lock before processing the prompt
        if db_name.upper() == "DEFAULT":
            if user_prompt:
                load_QA(use_history=False, promptTemplate_type="DefaultChatML", conversation_history = conversation_history)
                
                res = QA.predict(input=user_prompt).replace('<|im_end|>', '')
                print(res)
                prompt_response_dict = {
                    "Prompt": user_prompt,
                    "Answer": res,
                    "Sources": []
                }
                return jsonify(prompt_response_dict), 200
            else:
                return "No user prompt received", 400
            
        if user_prompt:
            load_QA(use_history=False, promptTemplate_type="ChatML", db_name=db_name, conversation_history = conversation_history)
            res = QA(user_prompt)
            answer, docs = res["result"].replace('<|im_end|>', ''), res["source_documents"]

            print(answer)

            prompt_response_dict = {
                "Prompt": user_prompt,
                "Answer": answer,
            }

            prompt_response_dict["Sources"] = []
            for document in docs:
                prompt_response_dict["Sources"].append(
                    (os.path.basename(str(document.metadata["source"])), str(document.page_content))
                )
            
            return jsonify(prompt_response_dict), 200
        else:
            return "No user prompt received", 400

# change chat completions to openAI

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5110)

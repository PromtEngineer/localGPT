import os
import shutil
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
import json

from config import (
    SOURCE_DIRECTORY
)

import localgpt

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

localgpt_object = localgpt.TextQAEngine()
localgpt_object.load_model()

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
            localgpt_object.ingest(db_name=db_name)
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
                localgpt_object.load_QA(use_history=False, promptTemplate_type="DefaultChatML", conversation_history = conversation_history)
                
                res = localgpt_object.QA.predict(input=user_prompt).replace('<|im_end|>', '')
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
            localgpt_object.load_QA(use_history=False, promptTemplate_type="ChatML", db_name=db_name, conversation_history = conversation_history)
            res = localgpt_object.QA(user_prompt)
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

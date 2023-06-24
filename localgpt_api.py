import os
import shutil
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

import localgpt
localgpt_object = localgpt.DocumentProcessor()

app = Flask(__name__)
localgpt_object.ingest()
localgpt_object.load_model()
localgpt_object.load_QA()

@app.route("/api/delete_source", methods=["GET"])
def delete_source_route():
    folder_name = "SOURCE_DOCUMENTS"

    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)

    os.makedirs(folder_name)

    return jsonify({"message": f"Folder '{folder_name}' successfully deleted and recreated."})

@app.route("/api/save_document", methods=["GET", "POST"])
def save_document_route():
    if "document" not in request.files:
        return "No document part", 400
    file = request.files["document"]
    if file.filename == "":
        return "No selected file", 400
    if file:
        filename = secure_filename(file.filename)
        folder_path = "SOURCE_DOCUMENTS"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_path = os.path.join(folder_path, filename)
        file.save(file_path)
        return "File saved successfully", 200

@app.route("/api/run_ingest", methods=["GET"])
def run_ingest_route():
    try:
        localgpt_object.ingest()
        localgpt_object.load_QA()
        return "Script executed successfully", 200
    except Exception as e:
        return f"Error occurred: {str(e)}", 500

@app.route("/api/prompt_route", methods=["GET", "POST"])
def prompt_route():
    user_prompt = request.form.get("user_prompt")
    if user_prompt:
        res = localgpt_object.QA(user_prompt)
        answer, docs = res["result"], res["source_documents"]

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

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5110)

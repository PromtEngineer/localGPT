from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

import os
import sys
import requests

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

app = Flask(__name__)
app.secret_key = "LeafmanZSecretKey"

### PAGES ###
@app.route('/', methods=['GET', 'POST'])
def home_page():
    if request.method == 'POST':
        if 'user_prompt' in request.form:
            user_prompt = request.form['user_prompt']
            print(f'User Prompt: {user_prompt}')
            
            main_prompt_url = 'http://localhost:5110/api/prompt_route'
            response = requests.post(main_prompt_url, data={'user_prompt': user_prompt})
            print(response.status_code)  # print HTTP response status code for debugging
            if response.status_code == 200:
                # print(response.json())  # Print the JSON data from the response
                return render_template('home.html', show_response_modal=True, response_dict = response.json())
        elif 'documents' in request.files:
            delete_source_url = 'http://localhost:5110/api/delete_source'  # URL of the /api/delete_source endpoint
            if request.form.get('action') == 'reset':
                response = requests.get(delete_source_url)

            save_document_url = 'http://localhost:5110/api/save_document'
            run_ingest_url = 'http://localhost:5110/api/run_ingest'  # URL of the /api/run_ingest endpoint
            files = request.files.getlist('documents')
            for file in files:
                print(file.filename)
                filename = secure_filename(file.filename)
                file_path = os.path.join('temp', filename)  # replace with your preferred path
                file.save(file_path)
                with open(file_path, 'rb') as f:
                    response = requests.post(save_document_url, files={'document': f})
                    print(response.status_code)  # print HTTP response status code for debugging
                os.remove(file_path)  # remove the file after sending the request
                # Make a GET request to the /api/run_ingest endpoint
            response = requests.get(run_ingest_url)
            print(response.status_code)  # print HTTP response status code for debugging
            
    # Display the form for GET request
    return render_template('home.html', show_response_modal=False, response_dict={'Prompt': 'None','Answer': 'None', 'Sources': [('ewf','wef')]})
if __name__ == '__main__':
    app.run(debug=False, port =5111)

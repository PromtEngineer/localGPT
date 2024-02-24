from flask import Flask, request, render_template, redirect, url_for, session, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user

import requests
import json
import os
import uuid
from werkzeug.utils import secure_filename
import time
import html
import shutil

from pygments import highlight
from pygments.lexers import get_lexer_by_name, guess_lexer
from pygments.formatters import HtmlFormatter

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a random secret key

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

DATABASES_FILE = 'databases.json'
API_BASE_URL = 'http://localhost:5110/api'

@app.route('/test')
def test():
     return render_template('test.html')

# User class
class User(UserMixin):
    def __init__(self, id):
        self.id = id

# This callback is used to reload the user object
@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

# Load users from JSON file
def load_users():
    with open('user_data.json') as f:
        users = json.load(f)
    return users

@app.route('/')
def home():
    # Check if the user is authenticated
    if current_user.is_authenticated:
        username = session.get('username')
        
        # Load the list of conversations
        conversation_list = load_conversations()
        databases = load_databases()

        # Sort the conversations by most recent first based on 'Epoch' of the first message
        sorted_conversations = sorted(conversation_list.items(), key=lambda kv: kv[1][0]['Epoch'], reverse=True)

        return render_template('index.html', conversation_list=sorted_conversations, db_names=databases.keys(), username=username)
    else:
        # Redirect to login page if user is not authenticated
        return redirect(url_for('login'))
    
@app.route('/login', methods=['GET', 'POST'])
def login():
    invalid_credentials = False  # Flag to indicate invalid credentials

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        users = load_users()
        for user_id, user_info in users.items():
            if user_info['username'] == username and user_info['password'] == password:
                user = User(user_id)
                login_user(user)
                session['username'] = username
                return redirect(url_for('home'))

        invalid_credentials = True  # Set flag to True if credentials are invalid

    return render_template('login.html', invalid_credentials=invalid_credentials)

# Route for logout
@app.route('/logout')
@login_required
def logout():
    session.pop('username', None)  # Remove the username from the session
    logout_user()
    return redirect(url_for('login'))

@app.route('/conversation/<conversation_id>')
def load_conversation(conversation_id):
    conversations = load_conversations()
    conversation = conversations.get(conversation_id, [])
    return jsonify(conversation)

@app.route('/new_conversation')
def new_conversation():
    conversation_id = str(uuid.uuid4())
    # Set current epoch time for the new conversation
    epoch_time = int(time.time() * 1000)  # in milliseconds
    return jsonify(conversation_id=conversation_id, timestamp=epoch_time)

def highlight_code(code_snippet, language=None):
    # If the language is specified, use it, otherwise try to guess
    if language:
        lexer = get_lexer_by_name(language)
    else:
        try:
            lexer = guess_lexer(code_snippet)
        except ValueError:
            # Default to plain text if language cannot be guessed
            lexer = get_lexer_by_name("text")

    # Use HtmlFormatter without full document structure
    formatter = HtmlFormatter(style='friendly', full=False, cssclass='codehilite')
    highlighted_code = highlight(code_snippet, lexer, formatter)
    return highlighted_code

def format_response(s, conversation_id, epoch_time):
    parts = s.split("```")
    modified_string = ""
    for i in range(len(parts)):
        if i % 2 == 0:
            # Even parts are outside the backticks
            # Convert text between single quotes to <strong>
            single_quote_parts = parts[i].split("`")
            for k in range(len(single_quote_parts)):
                print(f'single_quote_parts: {single_quote_parts[k]} {k}')
                if k % 2 == 0:
                    double_star_parts = single_quote_parts[k].split("**")
                    for j in range(len(double_star_parts)):
                        if j % 2 == 0:
                            modified_string += html.escape(double_star_parts[j], quote=True).replace('\n', '<br>')
                        else:
                            # Text between double stars
                            modified_string += f'<strong>{html.escape(double_star_parts[j], quote=True)}</strong>'
                else:
                    # Text between single quotes
                    modified_string += f'<strong">{html.escape(single_quote_parts[k], quote=True)}</strong>'
        else:
            # Odd parts are between backticks
            # Split the first line to wrap with <strong>
            first_line, code_snippet = parts[i].split("\n", 1)
            
            if first_line == "":
                first_line = "text"
            first_line_html = f'''<div class="px-4 py-2" style="background-color: #7f6b6b; border-radius: 10px 10px 0px 0px; color: #ffffff; margin-top: 20px; font-size: 14px; display: flex; justify-content: space-between;">{first_line} 
            <a class="copy-text-btn" id="copy-text-btn-{conversation_id}-{epoch_time}-{i}" onclick="copyCodeText('{conversation_id}-{epoch_time}-{i}')"><i class="bi-clipboard"></i> copy text</a></div>'''

            modified_code_string = highlight_code(code_snippet, language=first_line)
            modified_string += f'{first_line_html}<div class="px-4 py-2" id="copy-text-{conversation_id}-{epoch_time}-{i}" style="background-color: #ffffff; border-radius: 0px 0px 10px 10px;"><div style="margin-top: 15px;"></div>{modified_code_string}</div>'
        
    return modified_string

@app.route('/send/<conversation_id>', methods=['POST'])
def send(conversation_id):
    db_name = request.form['db_name']
    user_prompt = request.form['message'] # equivalent text to ResponseRaw
    main_prompt_url = f"{API_BASE_URL}/prompt_route"

    # Load existing conversations, we will only send the last two chains of conversation to save bandwidth and tokens.
    conversations = load_conversations()
    conversation_history = []
    if (conversation_id in conversations) and len(conversations[conversation_id]) > 0:
        for conversation in conversations[conversation_id][-2:]:
            conversation_history.append({key: conversation[key] for key in ["UserRaw", "ResponseRaw"] if key in conversation})

    # Serialize the list into a JSON string
    conversation_history_json = json.dumps(conversation_history)

    try:
        response = requests.post(main_prompt_url, data={"db_name": db_name, "user_prompt": user_prompt, "conversation_history": conversation_history_json}, timeout=90)  # Set timeout to 20 seconds
        print(response.status_code)  # print HTTP response status code for debugging
        
        # Get the server's response
        server_response = response.json()

    except requests.exceptions.Timeout:
        # If a timeout occurs, set a default response
        server_response = "Sorry, something went wrong. Please try again."
    except Exception as e:
        # Handle other exceptions
        print(e)
        # server_response = "An error occurred. Please try again later."
        server_response = """An error occurred. Please try again later."""
    
    # Get current epoch time
    epoch_time = int(time.time() * 1000)  # in milliseconds
    
    # Append the exchange to the conversation list
    if conversation_id not in conversations:
        conversations[conversation_id] = []

    server_response["Answer"] = server_response["Answer"].lstrip()
    server_response_raw = server_response["Answer"]
    server_response["Answer"] = format_response(server_response["Answer"], conversation_id, epoch_time)

    server_source_raw = server_response["Sources"]
    for source_idx in range(len(server_response["Sources"])):
        server_response["Sources"][source_idx].append(db_name)
        server_source_raw[source_idx].append(db_name)

        server_response["Sources"][source_idx][1] = server_response["Sources"][source_idx][1][server_response["Sources"][source_idx][1].index('\n')+1:]
        server_response["Sources"][source_idx][1] = server_response["Sources"][source_idx][1].replace('\n', '<br>')

    conversations[conversation_id].append({
        "Epoch": epoch_time,
        "User": html.escape(user_prompt, quote=True),
        "UserRaw": user_prompt,
        "Response": server_response["Answer"],
        "ResponseRaw": server_response_raw,
        "Sources": server_response["Sources"],
        "SourcesRaw": server_source_raw,
    })

    # Save the updated conversations list
    save_conversations(conversations)
    
    # Send back the server's response and epoch time to the client
    return jsonify(server_response=server_response, epoch_time=epoch_time)

def load_conversations():
    if os.path.exists(f"{session.get('username')}_conversations_list.json"):
        with open(f"{session.get('username')}_conversations_list.json", 'r') as json_file:
            return json.load(json_file)
    else:
        return {}

def save_conversations(conversations):
    with open(f"{session.get('username')}_conversations_list.json", 'w') as json_file:
        json.dump(conversations, json_file, indent=4)

@app.route('/delete_conversation/<conversation_id>', methods=['DELETE'])
def delete_conversation(conversation_id):
    conversations = load_conversations()
    if conversation_id in conversations:
        del conversations[conversation_id]
        save_conversations(conversations)
        return jsonify({"status": "success"}), 200
    else:
        return jsonify({"status": "error", "message": "Conversation not found"}), 404

### DATABASE MANAGEMENT ###
@app.route('/check_database_existence', methods=['POST'])
def check_database_existence():
    db_name = request.form.get('databaseName').strip().upper()
    databases = load_databases()

    if db_name.upper() in (name.upper() for name in databases.keys()):
        return jsonify({'exists': True})
    else:
        return jsonify({'exists': False})

@app.route('/create_database', methods=['POST'])
def create_database():
    db_name = request.form.get('databaseName').strip().upper()
    databases = load_databases()
    files = request.files.getlist('databaseFiles')

    file_names = [secure_filename(file.filename) for file in files]
    save_documents_and_run_ingest(db_name, files)
    
    databases[db_name] = file_names
    save_databases(databases)

    return jsonify({'message': 'Database created successfully'})

@app.route('/delete_database/<db_name>', methods=['POST'])
def delete_database(db_name):
    # Delete the database source via API
    response = requests.post(f"{API_BASE_URL}/delete_source/{db_name.upper()}")
    response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code

    database_directory = os.path.join(os.getcwd(), 'static', 'databases', db_name.upper())
    shutil.rmtree(database_directory)

    # Remove the database from the databases list
    databases = load_databases()
    del databases[db_name.upper()]
    save_databases(databases)

    return jsonify({'message': 'Database deleted successfully'})

@app.route('/reset_database/<db_name>', methods=['POST'])
def reset_database(db_name):
    files = request.files.getlist('documents[]')
    # Delete the source using API
    requests.post(f"{API_BASE_URL}/delete_source/{db_name.upper()}").raise_for_status()

    database_directory = os.path.join(os.getcwd(), 'static', 'databases', db_name.upper())
    shutil.rmtree(database_directory)

    file_names = [secure_filename(file.filename) for file in files]
    save_documents_and_run_ingest(db_name.upper(), files)
    databases = load_databases()
    databases[db_name.upper()] = file_names
    save_databases(databases)

    return jsonify({'message': 'Database reset successfully'})

@app.route('/get_database_documents/<db_name>', methods=['POST'])
def get_database_documents(db_name):
    databases = load_databases()
    return jsonify({'documents': databases[db_name.upper()]})

@app.route('/add_documents_to_database/<db_name>', methods=['POST'])
def add_documents_to_database(db_name):
    files = request.files.getlist('documents[]')
    file_names = [secure_filename(file.filename) for file in files]
    save_documents_and_run_ingest(db_name.upper(), files)
    databases = load_databases()
    databases[db_name.upper()] = databases[db_name.upper()] + file_names
    save_databases(databases)
    return jsonify({'message': 'Documents added to database successfully'})

def load_databases():
    # Read existing data from databases.json
    if os.path.exists(DATABASES_FILE):
        with open(DATABASES_FILE, 'r') as file:
            return json.load(file)
    else:
        return {}
    
def save_databases(databases):
    with open(DATABASES_FILE, 'w') as f:
        json.dump(databases, f)

def save_documents_and_run_ingest(db_name, files):
    for file in files:
        filename = secure_filename(file.filename)

        database_directory = os.path.join(os.getcwd(), 'static', 'databases', db_name.upper())
        os.makedirs(database_directory, exist_ok=True)
        file.save(os.path.join(database_directory, filename))
        file = open(os.path.join(database_directory, filename), 'rb')

        with requests.post(f"{API_BASE_URL}/save_document/{db_name.upper()}", files={"document": (filename, file)}) as response:
            response.raise_for_status()  # Raises an HTTPError if the request returned an unsuccessful status code

    response = requests.post(f"{API_BASE_URL}/run_ingest/{db_name.upper()}")
    response.raise_for_status()

    return {'message': 'Files uploaded and ingestion started successfully'}

if __name__ == '__main__':
    # app.run(debug=False, host='0.0.0.0', port=80)
    app.run(debug=False, host='0.0.0.0', port=5000)
    # app.run(debug=True, port=80)
    # app.run(debug=True, port=5000)
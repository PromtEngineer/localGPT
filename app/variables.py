# Description: This file contains all the variables used in the app

# Replace with your GitHub API KEY
GITHUB_API_KEY = "*********************************"

# Define open ai api key and organization
OPENAI_API_KEY = "*********************************"
OPENAI_ORGANIZATION = "*********************************"

# Define the folder for storing database
SOURCE_DIRECTORY = f"tmp_data"
# Define the folder for storing the embeddings
PERSIST_DIRECTORY = f"tmp_persist"

# Choose the model to use
MODEL_ID = "TheBloke/Llama-2-7b-Chat-GGUF"
MODEL_BASENAME = "llama-2-7b-chat.Q4_K_M.gguf"

# If llama in model_id, use promptTemplate_type="llama"
promptTemplate_type=None
if "llama" in MODEL_ID.lower():
    promptTemplate_type="llama"
elif "mistral" in MODEL_ID.lower():
    promptTemplate_type="mistral"

# Define the embedding model to use
EMBEDDING_NAME = "hkunlp/instructor-large"

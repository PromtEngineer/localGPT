import os
# from dotenv import load_dotenv
from chromadb.config import Settings

# load_dotenv()
ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

# Define the LLM Model
LLM_MODEL_NAME = "TheBloke/vicuna-7B-1.1-HF"

# Define the Instruct Embedding Model 
INSTRUCT_EMBEDDING_MODEL_NAME = "hkunlp/instructor-xl"

# Define the folder for storing database
SOURCE_DIRECTORY = f"{ROOT_DIRECTORY}/SOURCE_DOCUMENTS"

PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/DB"

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory=PERSIST_DIRECTORY,
        anonymized_telemetry=False
)

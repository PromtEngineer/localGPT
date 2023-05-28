import os
# from dotenv import load_dotenv
from chromadb.config import Settings

# load_dotenv()
PERSIST_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

# Define the folder for storing database
SOURCE_DIRECTORY = f"{PERSIST_DIRECTORY}/source_documents"

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory=PERSIST_DIRECTORY,
        anonymized_telemetry=False
)
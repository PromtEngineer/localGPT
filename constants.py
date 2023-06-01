import os
# from dotenv import load_dotenv
from chromadb.config import Settings

# load_dotenv()
ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

# Define the folder for storing database
SOURCE_DIRECTORY = os.path.join(ROOT_DIRECTORY, "SOURCE_DOCUMENTS")

PERSIST_DIRECTORY = os.path.join(ROOT_DIRECTORY, "DB")

MODEL_DIRECTORY = os.path.join(ROOT_DIRECTORY, "MODELS")

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory=PERSIST_DIRECTORY,
        anonymized_telemetry=False
)

if __name__ == '__main__':
    print("ROOT_DIRECTORY:", ROOT_DIRECTORY)
    print("PERSIST_DIRECTORY:", PERSIST_DIRECTORY)
    print("CHROMA_SETTINGS:", CHROMA_SETTINGS)
    print("MODEL_DIRECTORY:", MODEL_DIRECTORY)




import os
# from dotenv import load_dotenv
from chromadb.config import Settings

# load_dotenv()
ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

# Define the folder for storing database
SOURCE_DIRECTORY = os.path.join(ROOT_DIRECTORY, "SOURCE_DOCUMENTS")

PERSIST_DIRECTORY = os.path.join(ROOT_DIRECTORY, "DB")

MODEL_DIRECTORY = os.path.join("/home/wager/ai/models")
VICUNA_DIRECTORY = os.path.join(MODEL_DIRECTORY, "vicuna-7B-1.1-HF")    # "TheBloke/vicuna-7B-1.1-HF"
LOCAL_MODEL_NAME = os.path.join("ggml-vic7b-uncensored-q5_0.bin")

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory=PERSIST_DIRECTORY,
        anonymized_telemetry=False
)

if __name__ == '__main__':
    print("ROOT_DIRECTORY:", ROOT_DIRECTORY)
    print("PERSIST_DIRECTORY:", PERSIST_DIRECTORY)
    print("MODEL_DIRECTORY:", MODEL_DIRECTORY)
    print("VICUNA_DIRECTORY:", VICUNA_DIRECTORY)




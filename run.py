import ingest
import run_localGPT
from constants import (SOURCE_DIRECTORY, PERSIST_DIRECTORY,
            MODEL_ID, MODEL_BASENAME, EMBEDDING_MODEL_NAME)
import os
import glob
from langchain.embeddings import HuggingFaceInstructEmbeddings

# Device type
DEVICE_TYPE='cpu'
# DEVICE_TYPE='cuda'

# Use history
USE_HISTORY = False

# Create embeddings
EMBEDDINGS = HuggingFaceInstructEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": DEVICE_TYPE},
    )

# Get folders and features
FOLDERS = [os.path.basename(folder) for folder in glob.glob(f"{SOURCE_DIRECTORY}/*")]
FEATURES = ["fees", "liquidity_model", "license"]
DIRECTORIES = [f"{folder}/{feature}" for folder in FOLDERS for feature in FEATURES]

# k, chunk_size, chunk_overlap
#CONFIGS = [(5, 500, 100), (4, 700, 100), (3, 1000, 200)]
CONFIGS = [(5, 500, 100)]

def run(model_id=MODEL_ID, model_basename=MODEL_BASENAME):

    # Load model
    llm = run_localGPT.load_model(DEVICE_TYPE, model_id=model_id, model_basename=model_basename)

    for k, cs, co in CONFIGS:
        for dir in DIRECTORIES:
            dex_name = dir.split("/")[0]
            feature = dir.split("/")[1]

            print(f"Running for {dex_name} {feature} with k={k}, cs={cs}, co={co}")

            print("Ingesting...")
            source_directory = os.path.join(SOURCE_DIRECTORY, dir)
            #save_path = os.path.join(PERSIST_DIRECTORY, dir)
            # incllude embedding model id in save_path
            save_path = os.path.join(PERSIST_DIRECTORY, dir, os.path.basename(EMBEDDING_MODEL_NAME))
            ingest.main(device_type=DEVICE_TYPE, embedding_model=EMBEDDINGS, chunk_size=cs, chunk_overlap=co,\
                        source_directory=source_directory, save_path=save_path)

            persist_directory = os.path.join(save_path, f'cs_{cs}_co_{co}')

            # Getting the query from queries/feature.txt
            with open(f"queries/{feature}.txt", "r") as f:
                query = f.read()

            print("Running localGPT...")
            answer, docs = run_localGPT.main(DEVICE_TYPE, llm, k, persist_directory, query, USE_HISTORY, verbose=False, show_sources=False)

            # Saving the answer in answers/dex_name/feature/model_id/k_cs_co.txt
            os.makedirs(f"answers/{dex_name}/{feature}/{os.path.basename(MODEL_ID)}", exist_ok=True)
            with open(f"answers/{dex_name}/{feature}/{os.path.basename(MODEL_ID)}/k_{k}_cs_{cs}_co_{co}.txt", "w") as f:
                f.write(answer)


if __name__ == "__main__":
    models = {"TheBloke/Llama-2-7b-Chat-GGUF": "llama-2-7b-chat.Q4_K_M.gguf"}
    print('==================== Running all ====================')
    for model_id, model_basename in models.items():
        print(f'==================== Running {model_id} ====================')
        run(model_id, model_basename)
        print(f'==================== Finished {model_id} ====================')
    print('==================== Finished all ====================')

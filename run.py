import ingest
import run_localGPT
from constants import (SOURCE_DIRECTORY, PERSIST_DIRECTORY,
            MODEL_ID, MODEL_BASENAME, EMBEDDING_MODEL_NAME)
import os
import glob
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

import torch


# Device type
# DEVICE_TYPE='cpu'
DEVICE_TYPE='cuda'


# Create embeddings
EMBEDDINGS = HuggingFaceInstructEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": "cpu"},
    )

# Get folders and features
FOLDERS = [os.path.basename(folder) for folder in glob.glob(f"{SOURCE_DIRECTORY}/*")]
FEATURES = ["liquidity_model", "license"]
DIRECTORIES = [f"{folder}/{feature}" for folder in FOLDERS for feature in FEATURES]

# k, chunk_size, chunk_overlap
CONFIGS = [(5, 500, 100), (5, 500, 200), (4, 700, 100), (4, 700, 200) (3, 1000, 200), (3, 1000, 300)]
#CONFIGS = [(5, 500, 100)]

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

            # Replacing the DEX with the name of the DEX
            query = query.replace("the DEX", dex_name)

            print("Running localGPT...")

            # If llama in model_id, use promptTemplate_type="llama"
            promptTemplate_type=None
            if "llama" in model_id.lower():
                promptTemplate_type="llama"
            answer, docs = run_localGPT.main(DEVICE_TYPE, llm, k, persist_directory, query,\
                            verbose=False, show_sources=False, promptTemplate_type=promptTemplate_type)

            # Saving the answer in answers/dex_name/feature/model_id/k_cs_co.txt
            os.makedirs(f"answers/{dex_name}/{feature}/{os.path.basename(model_id)}", exist_ok=True)
            with open(f"answers/{dex_name}/{feature}/{os.path.basename(model_id)}/k_{k}_cs_{cs}_co_{co}.txt", "w", encoding='utf-8') as f:
                f.write(answer)

    # When done, unload the model
    del llm
    torch.cuda.empty_cache()


if __name__ == "__main__":
    models = {"TheBloke/Llama-2-70B-chat-GPTQ": "model.safetensors",
              "TheBloke/WizardLM-Uncensored-Falcon-40B-GPTQ": "model.safetensors",
              "TheBloke/Wizard-Vicuna-30B-Uncensored-GPTQ" : "model.safetensors",
              "TheBloke/Airoboros-L2-70B-2.1-GPTQ" : "model.safetensors",
              "TheBloke/llama-2-70b-Guanaco-QLoRA-GPTQ" : "model.safetensors",
              }
#MODEL_ID = "TheBloke/guanaco-65B-GPTQ"
# MODEL_BASENAME = "model.safetensors"
# MODEL_ID = "TheBloke/Airoboros-65B-GPT4-2.0-GPTQ"
# MODEL_BASENAME = "model.safetensors"
# MODEL_ID = "TheBloke/gpt4-alpaca-lora_mlp-65B-GPTQ"
# MODEL_BASENAME = "model.safetensors"
# MODEL_ID = "TheBloke/Upstage-Llama1-65B-Instruct-GPTQ"
# MODEL_BASENAME = "model.safetensors"
    print('==================== Running all ====================')
    for model_id, model_basename in models.items():
        print(f'==================== Running {model_id} ====================')
        run(model_id, model_basename)
        print(f'==================== Finished {model_id} ====================')
    print('==================== Finished all ====================')

import os

# from dotenv import load_dotenv
from chromadb.config import Settings

# https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/excel.html?highlight=xlsx#microsoft-excel
from langchain.document_loaders import CSVLoader, PDFMinerLoader, TextLoader, UnstructuredExcelLoader, Docx2txtLoader

# load_dotenv()
ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

# Define the folder for storing database
SOURCE_DIRECTORY = f"{ROOT_DIRECTORY}/SOURCE_DOCUMENTS"

PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/DB"

# Can be changed to a specific number
INGEST_THREADS = os.cpu_count() or 8

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
    chroma_db_impl="duckdb+parquet", persist_directory=PERSIST_DIRECTORY, anonymized_telemetry=False
)

# https://python.langchain.com/en/latest/_modules/langchain/document_loaders/excel.html#UnstructuredExcelLoader
DOCUMENT_MAP = {
    ".txt": TextLoader,
    ".md": TextLoader,
    ".py": TextLoader,
    ".pdf": PDFMinerLoader,
    ".csv": CSVLoader,
    ".xls": UnstructuredExcelLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
}

# Default Instructor Model
EMBEDDING_MODEL_NAME = "hkunlp/instructor-large"
# You can also choose a smaller model, don't forget to change HuggingFaceInstructEmbeddings
# to HuggingFaceEmbeddings in both ingest.py and run_localGPT.py
# EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Select the Model ID and model_basename
# load the LLM for generating Natural Language responses

#### GPU VRAM Memory required for LLM Models by Billion Parameter value (B Model)
####
#### (B Model)   (float32)    (float16)  (GPTQ 8bit)  (GPTQ 4bit)
####    7b         28 GB        14 GB       7 GB         3.5 GB    
####    13b        52 GB        26 GB       13 GB        6.5 GB    
####    32b        130 GB       65 GB       32.5 GB      16.25 GB 
####    65b        260.8 GB     130.4 GB    65.2 GB      32.6 GB  

MODEL_ID = "TheBloke/Llama-2-7B-Chat-GGML"
MODEL_BASENAME = "llama-2-7b-chat.ggmlv3.q4_0.bin"

# for HF models
# MODEL_ID = "TheBloke/vicuna-7B-1.1-HF"
# MODEL_BASENAME = None
# MODEL_ID = "TheBloke/Wizard-Vicuna-7B-Uncensored-HF"
# MODEL_ID = "TheBloke/guanaco-7B-HF"
# MODEL_ID = 'NousResearch/Nous-Hermes-13b' # Requires ~ 23GB VRAM. Using STransformers
# alongside will 100% create OOM on 24GB cards.
# llm = load_model(device_type, model_id=model_id)

 # For GPTQ (quantized) select an llm model based on your GPU and VRAM GB

    ##### 48GB VRAM Graphics Cards (RTX 6000, RTX A6000 and other 48GB VRAM GPUs) #####

    ### 65b GPTQ Models for 48GB GPUs
    # model_id = "TheBloke/guanaco-65B-GPTQ"
    # model_basename = "model.safetensors"
    # model_id = "TheBloke/Airoboros-65B-GPT4-2.0-GPTQ"
    # model_basename = "model.safetensors"
    # model_id = "TheBloke/gpt4-alpaca-lora_mlp-65B-GPTQ"
    # model_basename = "model.safetensors"
    # model_id = "TheBloke/Upstage-Llama1-65B-Instruct-GPTQ" 
    # model_basename = "model.safetensors"    
    
    ##### 24GB VRAM Graphics Cards (RTX 3090 - RTX 4090 (35% Faster) - RTX A5000 - RTX A5500) #####

    ### 13b GPTQ Models for 24GB GPUs (*** With best embedding model: hkunlp/instructor-xl ***)
    # model_id = "TheBloke/Wizard-Vicuna-13B-Uncensored-GPTQ"
    # model_basename = "Wizard-Vicuna-13B-Uncensored-GPTQ-4bit-128g.compat.no-act-order.safetensors"
    # model_id = "TheBloke/vicuna-13B-v1.5-GPTQ"
    # model_basename = "model.safetensors"
    # model_id = "TheBloke/Nous-Hermes-13B-GPTQ"
    # model_basename = "nous-hermes-13b-GPTQ-4bit-128g.no-act.order"
    # model_id = "TheBloke/WizardLM-13B-V1.2-GPTQ" 
    # model_basename = "gptq_model-4bit-128g.safetensors

    ### 30b GPTQ Models for 24GB GPUs (*** Requires using intfloat/e5-base-v2 instead of hkunlp/instructor-large as embedding model ***)
    # model_id = "TheBloke/Wizard-Vicuna-30B-Uncensored-GPTQ"
    # model_basename = "Wizard-Vicuna-30B-Uncensored-GPTQ-4bit--1g.act.order.safetensors" 
    # model_id = "TheBloke/WizardLM-30B-Uncensored-GPTQ"
    # model_basename = "WizardLM-30B-Uncensored-GPTQ-4bit.act-order.safetensors" 
    
    ##### 8-10GB VRAM Graphics Cards (RTX 3080 - RTX 3080 Ti - RTX 3070 Ti - 3060 Ti - RTX 2000 Series, Quadro RTX 4000, 5000, 6000) #####
    
    ### 7b GPTQ Models for 8GB GPUs
    # model_id = "TheBloke/Wizard-Vicuna-7B-Uncensored-GPTQ"
    # model_basename = "Wizard-Vicuna-7B-Uncensored-GPTQ-4bit-128g.no-act.order.safetensors"
    # model_id = "TheBloke/WizardLM-7B-uncensored-GPTQ"
    # model_basename = "WizardLM-7B-uncensored-GPTQ-4bit-128g.compat.no-act-order.safetensors"
    # model_id = "TheBloke/wizardLM-7B-GPTQ"
    # model_basename = "wizardLM-7B-GPTQ-4bit.compat.no-act-order.safetensors"

# for GGML (quantized cpu+gpu+mps) models - check if they support llama.cpp
# MODEL_ID = "TheBloke/wizard-vicuna-13B-GGML"
# MODEL_BASENAME = "wizard-vicuna-13B.ggmlv3.q4_0.bin"
# MODEL_BASENAME = "wizard-vicuna-13B.ggmlv3.q6_K.bin"
# MODEL_BASENAME = "wizard-vicuna-13B.ggmlv3.q2_K.bin"
# MODEL_ID = "TheBloke/orca_mini_3B-GGML"
# MODEL_BASENAME = "orca-mini-3b.ggmlv3.q4_0.bin"

import os
import subprocess
import logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
)
import torch
from auto_gptq import AutoGPTQForCausalLM

# from dotenv import load_dotenv
from chromadb.config import Settings

# https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/excel.html?highlight=xlsx#microsoft-excel
from langchain.document_loaders import CSVLoader, PDFMinerLoader, TextLoader, UnstructuredExcelLoader
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import Chroma
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    pipeline,
)

ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
SOURCE_DIRECTORY = f"{ROOT_DIRECTORY}/SOURCE_DOCUMENTS"
PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/DB"
INGEST_THREADS = os.cpu_count() or 8
CHROMA_SETTINGS = Settings(
    chroma_db_impl="duckdb+parquet", persist_directory=PERSIST_DIRECTORY, anonymized_telemetry=False
)
DOCUMENT_MAP = {
    ".txt": TextLoader,
    ".pdf": PDFMinerLoader,
    ".csv": CSVLoader,
    ".xls": UnstructuredExcelLoader,
    ".xlxs": UnstructuredExcelLoader,
}
EMBEDDING_MODEL_NAME = "hkunlp/instructor-large"
DEVICE_TYPE = "cuda"
SHOW_SOURCES = True
MODEL_ID = "TheBloke/WizardLM-7B-uncensored-GPTQ"
MODEL_BASENAME = "WizardLM-7B-uncensored-GPTQ-4bit-128g.compat.no-act-order.safetensors"

EMBEDDINGS = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": DEVICE_TYPE})

class DocumentProcessor:
    def __init__(self):
        logging.info(f"Initializing localgpt")
        self.QA = None
        self.LLM = None
        self.MODEL_LOADED = False

    def ingest(self):
        logging.info(f"Executing ingest.py")
        result = subprocess.run(["python", "ingest.py"], capture_output=True)
        if result.returncode != 0:
            return "Script execution failed: {}".format(result.stderr.decode("utf-8")), 500


    def load_model(self, model_id = MODEL_ID, model_basename= MODEL_BASENAME):
        """
        Select a model for text generation using the HuggingFace library.
        If you are running this for the first time, it will download a model for you.
        Subsequent runs will use the model from the disk.

        Args:
            model_id (str): Identifier of the model to load from HuggingFace's model hub.
            model_basename (str, optional): Basename of the model if using quantized models.
                Defaults to None.

        Raises:
            ValueError: If an unsupported model or device type is provided.
        """
        if self.MODEL_LOADED:
            logging.info(f"Model already loaded: {model_id}, on: {DEVICE_TYPE}")
            return
        else:
            self.MODEL_LOADED = True
        
        logging.info(f"Loading Model: {model_id}, on: {DEVICE_TYPE}")
        logging.info("This action can take a few minutes!")

        if model_basename is not None:
            # The code supports all huggingface models that ends with GPTQ and have some variation
            # of .no-act.order or .safetensors in their HF repo.
            logging.info("Using AutoGPTQForCausalLM for quantized models")

            if ".safetensors" in model_basename:
                # Remove the ".safetensors" ending if present
                model_basename = model_basename.replace(".safetensors", "")

            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            logging.info("Tokenizer loaded")

            model = AutoGPTQForCausalLM.from_quantized(
                model_id,
                model_basename=model_basename,
                use_safetensors=True,
                trust_remote_code=True,
                device="cuda:0",
                use_triton=False,
                quantize_config=None,
            )
        elif DEVICE_TYPE.lower() == "cuda":  
            # The code supports all huggingface models that ends with -HF or which have a .bin
            # file in their HF repo.
            logging.info("Using AutoModelForCausalLM for full models")
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            logging.info("Tokenizer loaded")

            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                # max_memory={0: "15GB"} # Uncomment this line with you encounter CUDA out of memory errors
            )
            model.tie_weights()
        else:
            logging.info("Using LlamaTokenizer")
            tokenizer = LlamaTokenizer.from_pretrained(model_id)
            model = LlamaForCausalLM.from_pretrained(model_id)

        # Load configuration from the model to avoid warnings
        generation_config = GenerationConfig.from_pretrained(model_id)

        # Create a pipeline for text generation
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=2048,
            temperature=0,
            top_p=0.95,
            repetition_penalty=1.15,
            generation_config=generation_config,
        )

        self.LLM = HuggingFacePipeline(pipeline=pipe)
        logging.info("Local LLM Loaded")

    def load_QA(self):
        # load the vectorstore
        db = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=EMBEDDINGS,
            client_settings=CHROMA_SETTINGS,
        )
        retriever = db.as_retriever()
        self.QA = RetrievalQA.from_chain_type(
            llm=self.LLM, chain_type="stuff", retriever=retriever, return_source_documents=SHOW_SOURCES
        )

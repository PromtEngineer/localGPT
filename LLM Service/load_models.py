import torch
from huggingface_hub import hf_hub_download
from langchain.llms import LlamaCpp
from auto_gptq import exllama_set_max_input_length

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer
)
from config import CONTEXT_WINDOW_SIZE, MAX_NEW_TOKENS, N_GPU_LAYERS, N_BATCH, MODELS_PATH, INGEST_THREADS


def load_quantized_model_gguf_ggml(model_id, model_basename, device_type, logging):
    """
    Load a GGUF/GGML quantized model using LlamaCpp.

    This function attempts to load a GGUF/GGML quantized model using the LlamaCpp library.
    If the model is of type GGML, and newer version of LLAMA-CPP is used which does not support GGML,
    it logs a message indicating that LLAMA-CPP has dropped support for GGML.

    Parameters:
    - model_id (str): The identifier for the model on HuggingFace Hub.
    - model_basename (str): The base name of the model file.
    - device_type (str): The type of device where the model will run, e.g., 'mps', 'cuda', etc.
    - logging (logging.Logger): Logger instance for logging messages.

    Returns:
    - LlamaCpp: An instance of the LlamaCpp model if successful, otherwise None.

    Notes:
    - The function uses the `hf_hub_download` function to download the model from the HuggingFace Hub.
    - The number of GPU layers is set based on the device type.
    """

    try:
        logging.info("Using Llamacpp for GGUF/GGML quantized models")
        model_path = hf_hub_download(
            repo_id=model_id,
            filename=model_basename,
            resume_download=True,
            cache_dir=MODELS_PATH,
        )
        kwargs = {
            "model_path": model_path,
            "n_ctx": CONTEXT_WINDOW_SIZE,
            "max_tokens": MAX_NEW_TOKENS,
            "n_threads": INGEST_THREADS,
            "n_batch": N_BATCH,  # set this based on your GPU & CPU RAM
            "repetition_penalty": 1.2,
            "top_p": .95,
            "top_k": 10
        }
        if device_type.lower() == "mps":
            kwargs["n_gpu_layers"] = 1
        if device_type.lower() == "cuda":
            kwargs["n_gpu_layers"] = N_GPU_LAYERS  # set this based on your GPU

        return LlamaCpp(**kwargs)
    except:
        if "ggml" in model_basename:
            logging.INFO("If you were using GGML model, LLAMA-CPP Dropped Support, Use GGUF Instead")
        return None


def load_quantized_model_qptq(model_id, device_type, logging):
    """
    Load a GPTQ quantized model using AutoGPTQForCausalLM.

    This function loads a quantized model that ends with GPTQ and may have variations
    of .no-act.order or .safetensors in their HuggingFace repo.

    Parameters:
    - model_id (str): The identifier for the model on HuggingFace Hub.
    - device_type (str): The type of device where the model will run.
    - logging (logging.Logger): Logger instance for logging messages.

    Returns:
    - model (AutoGPTQForCausalLM): The loaded quantized model.
    - tokenizer (AutoTokenizer): The tokenizer associated with the model.
    """

    # The code supports all huggingface models that ends with GPTQ and have some variation
    # of .no-act.order or .safetensors in their HF repo.
    logging.info("Using AutoGPTQForCausalLM for quantized models")

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    logging.info("Tokenizer loaded")

    # To use a different branch, change revision
    # For example: revision="gptq-4bit-128g-actorder_True"
    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                device_map="auto",
                                                trust_remote_code=False,
                                                revision="gptq-4bit-32g-actorder_True")
    
    model = exllama_set_max_input_length(model, CONTEXT_WINDOW_SIZE)

    return model, tokenizer

def load_full_model(model_id, device_type, logging):
    """
    Load a full model using either LlamaTokenizer or AutoModelForCausalLM.

    This function loads a full model based on the specified device type.
    If the device type is 'mps' or 'cpu', it uses LlamaTokenizer and LlamaForCausalLM.
    Otherwise, it uses AutoModelForCausalLM.

    Parameters:
    - model_id (str): The identifier for the model on HuggingFace Hub.
    - device_type (str): The type of device where the model will run.
    - logging (logging.Logger): Logger instance for logging messages.

    Returns:
    - model (Union[LlamaForCausalLM, AutoModelForCausalLM]): The loaded model.
    - tokenizer (Union[LlamaTokenizer, AutoTokenizer]): The tokenizer associated with the model.

    Notes:
    - The function uses the `from_pretrained` method to load both the model and the tokenizer.
    - Additional settings are provided for NVIDIA GPUs, such as loading in 4-bit and setting the compute dtype.
    """

    if device_type.lower() in ["mps", "cpu"]:
        logging.info("Using LlamaTokenizer")
        tokenizer = LlamaTokenizer.device_typerom_pretrained(model_id, cache_dir="./models/")
        logging.info("Tokenizer loaded")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            cache_dir=MODELS_PATH,
            trust_remote_code=True, # set these if you are using NVIDIA GPU
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            max_memory={0: "15GB"} # Uncomment this line with you encounter CUDA out of memory errors
        )
        model.tie_weights()
    return model, tokenizer

def load_quantized_model_awq(model_id, logging):
    """
    Load a AWQ quantized model using AutoModelForCausalLM.
    This function loads a quantized model that ends with AWQ.
    Parameters:
    - model_id (str): The identifier for the model on HuggingFace Hub.
    - logging (logging.Logger): Logger instance for logging messages.
    Returns:
    - model (AutoModelForCausalLM): The loaded quantized model.
    - tokenizer (AutoTokenizer): The tokenizer associated with the model.
    """

    # The code supports all huggingface models that ends with AWQ.
    logging.info("Using AutoModelForCausalLM for AWQ quantized models")

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    logging.info("Tokenizer loaded")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        use_safetensors=True,
        trust_remote_code=True,
        device_map="auto",
    )
    return model, tokenizer
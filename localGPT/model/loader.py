"""
localGPT/model.py

This module contains the ModelLoader class for loading models and creating
pipelines for text generation.

Classes:
- ModelLoader: A class for loading models and creating text generation pipelines.

Note: The module relies on imports from localGPT and other external libraries.
"""

import logging
import sys
from typing import Any

import torch
from auto_gptq import AutoGPTQForCausalLM
from langchain.llms import HuggingFacePipeline
from torch.cuda import OutOfMemoryError
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    pipeline,
)

from localGPT import HF_MODEL_REPO_ID, HF_MODEL_SAFETENSORS, LC_MODEL_CLASS, TORCH_DEVICE_TYPE


class ModelLoader:
    """
    A class for loading models and creating text generation pipelines.

    Methods:
    - __init__: Initializes the ModelLoader with optional device type, model ID,
      and model basename.
    - load_quantized_model: Loads a quantized model for text generation.
    - load_full_model: Loads a full model for text generation.
    - load_llama_model: Loads a Llama model for text generation.
    - create_pipeline: Creates a text generation pipeline.
    - load_model: Loads the appropriate model based on the configuration.
    """

    def __init__(
        self,
        repo_id: str | None,
        model_class: str | None,
        safetensors: str | None,
        device_type: str | None,
        use_triton: bool | None,
    ):
        """
        Initializes the ModelLoader with optional device type, model ID, and
        model basename.

        Args:
        - device_type (str, optional): The device type for model loading.
          Defaults to TORCH_DEVICE_TYPE.
        - repo_id (str, optional): The model ID.
          Defaults to HF_MODEL_REPO_ID.
        - safetensors (str, optional): The model basename.
          Defaults to HF_MODEL_SAFETENSORS.
        """
        self.device_type = (device_type or TORCH_DEVICE_TYPE).lower()
        self.model_class = (model_class or LC_MODEL_CLASS).lower()
        self.repo_id = repo_id or HF_MODEL_REPO_ID
        self.safetensors = safetensors or HF_MODEL_SAFETENSORS
        self.use_triton = use_triton or False

    def load_huggingface_model(self):
        """
        Loads a full model for text generation.

        Returns:
        - model: The loaded full model.
        - tokenizer: The tokenizer associated with the model.
        """
        logging.info("Using AutoModelForCausalLM for full models")

        config = AutoConfig.from_pretrained(self.repo_id)
        logging.info(f"Configuration loaded for {self.repo_id}")

        tokenizer = AutoTokenizer.from_pretrained(self.repo_id)
        logging.info(f"Tokenizer loaded for {self.repo_id}")

        kwargs: dict[str, Any] = {
            "low_cpu_mem_usage": True,
            "resume_download": True,
            "trust_remote_code": False,
            # NOTE: Uncomment this line if you encounter CUDA out of memory errors
            # "max_memory": {0: "7GB"},
            # NOTE: According to the Hugging Face documentation, `output_loading_info` is
            # for when you want to return a tuple with the pretrained model and a dictionary
            # containing the loading information.
            # "output_loading_info": True,
        }

        if self.device_type != "cpu":
            kwargs["device_map"] = self.device_type
            # NOTE: This loads at half precision: 32 / 2 = 16
            kwargs["torch_dtype"] = torch.float16

        try:
            model = AutoModelForCausalLM.from_pretrained(self.repo_id, config=config, **kwargs)
        except (OutOfMemoryError,) as e:
            logging.error("Encountered CUDA out of memory error while loading the model.")
            logging.error(str(e))
            sys.exit(1)

        logging.info(f"Model loaded for {self.repo_id}")

        if not isinstance(model, tuple):
            model.tie_weights()
            logging.warning("Model Weights Tied: Effectiveness depends on the specific type of model.")

        return model, tokenizer

    def load_huggingface_llama_model(self):
        """
        Loads a Llama model for text generation.

        Returns:
        - model: The loaded Llama model.
        - tokenizer: The tokenizer associated with the model.
        """
        logging.info("Using LlamaTokenizer")
        # vocab_file (str) — Path to the vocabulary file.
        # NOTE: Path to the tokenizer
        tokenizer = LlamaTokenizer.from_pretrained(self.repo_id)
        logging.info(f"Tokenizer loaded for {self.repo_id}")
        # NOTE: Path to the pytorch bin
        model = LlamaForCausalLM.from_pretrained(self.repo_id)
        logging.info(f"Model loaded for {self.repo_id}")
        return model, tokenizer

    def load_gptq_model(self):
        """
        Loads a quantized model for text generation.

        Returns:
        - model: The loaded quantized model.
        - tokenizer: The tokenizer associated with the model.
        """
        # NOTE: The code supports all huggingface models that ends with GPTQ and
        # have some variation of .no-act.order or .safetensors in their HF repo.
        logging.info("Using AutoGPTQForCausalLM for quantized models")
        logging.warning("GGML models may supersede GPTQ models in future releases")

        if self.safetensors.endswith(".safetensors"):
            split_string = self.safetensors.split(".")
            self.safetensors = ".".join(split_string[:-1])
            logging.info(f"Stripped {self.safetensors}. Moving on.")

        tokenizer = AutoTokenizer.from_pretrained(self.repo_id, use_fast=True)
        logging.info(f"Tokenizer loaded for {self.repo_id}")

        kwargs: dict[str, Any] = {
            "low_cpu_mem_usage": True,
            "resume_download": True,
            "trust_remote_code": False,
            "use_safetensors": True,
            "device_map": "auto",
            "safetensors": self.safetensors,
            # NOTE: Uncomment this line if you encounter CUDA out of memory errors
            # "max_memory": {0: "7GB"},
            # NOTE: According to the Hugging Face documentation, `output_loading_info` is
            # for when you want to return a tuple with the pretrained model and a dictionary
            # containing the loading information.
            # "output_loading_info": True,
        }

        if self.device_type != "cpu":
            kwargs["use_cuda_fp16"] = True
            kwargs["use_triton"] = self.use_triton
            kwargs["device"] = f"{self.device_type}:0"

        model = AutoGPTQForCausalLM.from_quantized(self.repo_id, **kwargs)
        logging.info(f"Model loaded for {self.repo_id}")

        return model, tokenizer

    @staticmethod
    def create_pipeline(model, tokenizer, generation_config):
        """
        Creates a text generation pipeline.

        Args:
        - model: The model for text generation.
        - tokenizer: The tokenizer associated with the model.
        - generation_config: The generation configuration.

        Returns:
        - pipeline: The created text generation pipeline.
        """
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512,
            temperature=0,
            top_p=0.95,
            repetition_penalty=1.15,
            generation_config=generation_config,
        )
        return HuggingFacePipeline(pipeline=pipe)

    def load_model(self):
        """
        Loads the appropriate model based on the configuration.

        Returns:
        - local_llm: The loaded local language model (LLM).
        """
        # NOTE: This should be replaced with mapping for smooth extensibility
        if self.model_class.lower() == "huggingface":
            model, tokenizer = self.load_huggingface_model()
        elif self.model_class.lower() == "huggingface-llama":
            model, tokenizer = self.load_huggingface_llama_model()
        elif self.model_class.lower() == "gptq":
            model, tokenizer = self.load_gptq_model()
        elif self.model_class.lower() == "ggml":
            raise NotImplementedError("GGML support is in research and development")
        else:
            raise AttributeError(
                "Unsupported model type given. "
                "Expected one of: "
                "huggingface, "
                "huggingface-llama, "
                "ggml, "
                "gptq"
            )

        # Load configuration from the model to avoid warnings
        generation_config = GenerationConfig.from_pretrained(self.repo_id)
        # see here for details:
        # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig.from_pretrained.returns

        # Create a pipeline for text generation
        local_llm = self.create_pipeline(model, tokenizer, generation_config)

        logging.info("Local LLM Loaded")

        return local_llm

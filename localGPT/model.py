"""
localGPT/model.py

This module contains the ModelLoader class for loading models and creating 
pipelines for text generation.

Classes:
- ModelLoader: A class for loading models and creating text generation pipelines.

Note: The module relies on imports from localGPT and other external libraries.
"""

import logging
from typing import Optional

import torch
from auto_gptq import AutoGPTQForCausalLM
from langchain.llms import HuggingFacePipeline
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    pipeline,
)

from localGPT import (
    DEFAULT_DEVICE_TYPE,
    DEFAULT_MODEL_REPOSITORY,
    DEFAULT_MODEL_SAFETENSORS,
)


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
        device_type: Optional[str],
        model_repository: Optional[str],
        model_safetensors: Optional[str],
    ):
        """
        Initializes the ModelLoader with optional device type, model ID, and
        model basename.

        Args:
        - device_type (str, optional): The device type for model loading.
          Defaults to DEFAULT_DEVICE_TYPE.
        - model_repository (str, optional): The model ID.
          Defaults to DEFAULT_MODEL_REPOSITORY.
        - model_safetensors (str, optional): The model basename.
          Defaults to DEFAULT_MODEL_SAFETENSORS.
        """
        self.device_type = device_type or DEFAULT_DEVICE_TYPE
        self.model_repository = model_repository or DEFAULT_MODEL_REPOSITORY
        self.model_safetensors = model_safetensors or DEFAULT_MODEL_SAFETENSORS

    def load_quantized_model(self, use_triton: bool = False):
        """
        Loads a quantized model for text generation.

        Args:
        - use_triton (bool, optional): Whether to use Triton for model loading.
          Defaults to False.

        Returns:
        - model: The loaded quantized model.
        - tokenizer: The tokenizer associated with the model.
        """
        # NOTE: The code supports all huggingface models that ends with GPTQ and
        # have some variation of .no-act.order or .safetensors in their HF repo.
        logging.info("Using AutoGPTQForCausalLM for quantized models")

        if self.model_safetensors.endswith(".safetensors"):
            # Remove the ".safetensors" ending if present
            # NOTE: Using replace is not ideal because it can
            # have unintended side effects.
            split_string = self.model_safetensors.split(".")
            self.model_safetensors = ".".join(split_string[:-1])

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_repository, use_fast=True
        )
        logging.info("Tokenizer loaded")

        model = AutoGPTQForCausalLM.from_quantized(
            self.model_repository,
            model_safetensors=self.model_safetensors,
            use_safetensors=True,
            trust_remote_code=True,
            device=f"{self.device_type}:0",
            use_triton=use_triton,
            quantize_config=None,
        )
        return model, tokenizer

    def load_full_model(self):
        """
        Loads a full model for text generation.

        Returns:
        - model: The loaded full model.
        - tokenizer: The tokenizer associated with the model.
        """
        # The code supports all huggingface models that ends with -HF
        # or which have a .bin file in their HF repo.
        logging.info("Using AutoModelForCausalLM for full models")
        tokenizer = AutoTokenizer.from_pretrained(self.model_repository)
        logging.info("Tokenizer loaded")

        model = AutoModelForCausalLM.from_pretrained(
            self.model_repository,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            # NOTE: Uncomment if you encounter out of memory errors
            # max_memory={0: "15GB"}
        )
        model.tie_weights()
        return model, tokenizer

    def load_llama_model(self):
        """
        Loads a Llama model for text generation.

        Returns:
        - model: The loaded Llama model.
        - tokenizer: The tokenizer associated with the model.
        """
        logging.info("Using LlamaTokenizer")
        tokenizer = LlamaTokenizer.from_pretrained(self.model_repository)
        model = LlamaForCausalLM.from_pretrained(self.model_repository)
        return model, tokenizer

    def load_ggml_model(self):
        # TODO
        pass

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
            max_length=2048,
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
        if self.model_safetensors is not None:
            model, tokenizer = self.load_quantized_model()
        elif self.device_type.lower() == DEFAULT_DEVICE_TYPE:
            model, tokenizer = self.load_full_model()
        else:
            model, tokenizer = self.load_llama_model()

        # Load configuration from the model to avoid warnings
        generation_config = GenerationConfig.from_pretrained(
            self.model_repository
        )
        # see here for details:
        # https://huggingface.co/docs/transformers/
        # main_classes/text_generation#transformers.GenerationConfig.from_pretrained.returns

        # Create a pipeline for text generation
        local_llm = self.create_pipeline(model, tokenizer, generation_config)

        logging.info("Local LLM Loaded")

        return local_llm

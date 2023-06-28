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
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    pipeline,
)

from localGPT import DEFAULT_DEVICE_TYPE, DEFAULT_MODEL_REPOSITORY, DEFAULT_MODEL_SAFETENSORS, DEFAULT_MODEL_TYPE


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
        model_type: Optional[str],
        model_repository: Optional[str],
        model_safetensors: Optional[str],
        use_triton: Optional[bool],
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
        self.device_type = (device_type or DEFAULT_DEVICE_TYPE).lower()
        self.model_type = (model_type or DEFAULT_MODEL_TYPE).lower()
        self.model_repository = model_repository or DEFAULT_MODEL_REPOSITORY
        self.model_safetensors = model_safetensors or DEFAULT_MODEL_SAFETENSORS
        self.use_triton = use_triton or False

    def load_huggingface_model(self):
        """
        Loads a full model for text generation.

        Returns:
        - model: The loaded full model.
        - tokenizer: The tokenizer associated with the model.
        """
        # The code supports all huggingface models that ends with -HF
        # or which have a .bin file in their HF repo.
        logging.info("Using AutoModelForCausalLM for full models")

        config = AutoConfig.from_pretrained(self.model_repository)
        logging.info(f"Configuration loaded for {self.model_repository}")

        tokenizer = AutoTokenizer.from_pretrained(self.model_repository)
        logging.info(f"Tokenizer loaded for {self.model_repository}")

        model = AutoModelForCausalLM.from_pretrained(
            config=config,
            resume_download=True,
            trust_remote_code=False,
            output_loading_info=True,
        )
        logging.info(f"Model loaded for {self.model_repository}")

        model.tie_weights()
        logging.warn("Model Weights Tied: " "Effectiveness depends on specific type of model.")

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
        tokenizer = LlamaTokenizer.from_pretrained(self.model_repository)
        logging.info(f"Tokenizer loaded for {self.model_repository}")
        # NOTE: Path to the pytorch bin
        model = LlamaForCausalLM.from_pretrained(self.model_repository)
        logging.info(f"Model loaded for {self.model_repository}")
        return model, tokenizer

    def load_ggml_model(self):
        # TODO: Implement supporting 4, 5, and 8, -bit quant model support
        # NOTE: This method potentially supersedes `load_gptq_model`
        pass

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
        logging.warn("GGML models may supersede GPTQ models in future releases")

        if not torch.cuda.is_available():
            raise NotImplementedError("Only CUDA based devices are officially supported")

        if self.model_safetensors.endswith(".safetensors"):
            split_string = self.model_safetensors.split(".")
            self.model_safetensors = ".".join(split_string[:-1])
            logging.info(f"Stripped {self.model_safetensors}. Moving on.")

        tokenizer = AutoTokenizer.from_pretrained(self.model_repository, use_fast=True)
        logging.info(f"Tokenizer loaded for {self.model_repository}")

        model = AutoGPTQForCausalLM.from_quantized(
            self.model_repository,
            device_map="auto",
            # NOTE: # Uncomment if you encounter Out of Memory Errors
            # max_memory={0: "15GB"},
            device=f"{self.device_type}:0",
            low_cpu_mem_usage=True,
            use_triton=self.use_triton,
            use_safetensors=True,
            use_cuda_fp16=True,
            model_safetensors=self.model_safetensors,
        )
        logging.info(f"Model loaded for {self.model_repository}")

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
        # NOTE: This should be replaced with mapping for smooth extensibility
        if self.model_type.lower() == "huggingface":
            model, tokenizer = self.load_huggingface_model()
        elif self.model_type.lower() == "huggingface-llama":
            model, tokenizer = self.load_huggingface_llama_model()
        elif self.model_type.lower() == "gptq":
            model, tokenizer = self.load_gptq_model()
        elif self.model_type.lower() == "ggml":
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
        generation_config = GenerationConfig.from_pretrained(self.model_repository)
        # see here for details:
        # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig.from_pretrained.returns

        # Create a pipeline for text generation
        local_llm = self.create_pipeline(model, tokenizer, generation_config)

        logging.info("Local LLM Loaded")

        return local_llm

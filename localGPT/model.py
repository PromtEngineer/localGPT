# localGPT/model.py
import logging
from typing import Optional
import torch
from localGPT import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_BASE_NAME,
    DEFAULT_DEVICE_TYPE,
)
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


class ModelLoader:
    def __init__(
        self,
        device_type: Optional[str],
        model_id: Optional[str],
        model_basename: Optional[str],
    ):
        self.device_type = device_type or DEFAULT_DEVICE_TYPE
        self.model_id = model_id or DEFAULT_MODEL_ID
        self.model_basename = model_basename or DEFAULT_MODEL_BASE_NAME

    def load_quantized_model(self, use_triton: bool = False):
        # The code supports all huggingface models that ends with GPTQ and have some
        # variation of .no-act.order or .safetensors in their HF repo.
        logging.info("Using AutoGPTQForCausalLM for quantized models")

        if ".safetensors" in self.model_basename:
            # Remove the ".safetensors" ending if present
            # NOTE: Using replace is not ideal because it can
            # have unintended side effects.
            self.model_basename = self.model_basename.replace(
                ".safetensors", ""
            )

        tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        logging.info("Tokenizer loaded")

        model = AutoGPTQForCausalLM.from_quantized(
            self.model_id,
            model_basename=self.model_basename,
            use_safetensors=True,
            trust_remote_code=True,
            device=f"{self.device_type}:0",
            use_triton=use_triton,
            quantize_config=None,
        )
        return model, tokenizer

    def load_full_model(self):
        # The code supports all huggingface models that ends with -HF or which have
        # a .bin file in their HF repo.
        logging.info("Using AutoModelForCausalLM for full models")
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        logging.info("Tokenizer loaded")

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            # max_memory={0: "15GB"} # Uncomment this line with you encounter CUDA out
            # of memory errors
        )
        model.tie_weights()
        return model, tokenizer

    def load_llama_model(self):
        logging.info("Using LlamaTokenizer")
        tokenizer = LlamaTokenizer.from_pretrained(self.model_id)
        model = LlamaForCausalLM.from_pretrained(self.model_id)
        return model, tokenizer

    @staticmethod
    def create_pipeline(model, tokenizer, generation_config):
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
        if self.model_basename is not None:
            model, tokenizer = self.load_quantized_model()
        elif self.device_type.lower() == DEFAULT_DEVICE_TYPE:
            model, tokenizer = self.load_full_model()
        else:
            model, tokenizer = self.load_llama_model()

        # Load configuration from the model to avoid warnings
        generation_config = GenerationConfig.from_pretrained(self.model_id)
        # see here for details:
        # https://huggingface.co/docs/transformers/
        # main_classes/text_generation#transformers.GenerationConfig.from_pretrained.returns

        # Create a pipeline for text generation
        local_llm = self.create_pipeline(model, tokenizer, generation_config)

        logging.info("Local LLM Loaded")

        return local_llm

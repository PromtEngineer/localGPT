import copy
import os
import torch
from pathlib import Path
from typing import List

import habana_frameworks.torch.hpu as torch_hpu

from habana_frameworks.torch.hpu import wrap_in_hpu_graph
from huggingface_hub import snapshot_download
from optimum.habana.transformers.generation import MODELS_OPTIMIZED_WITH_STATIC_SHAPES
from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
from optimum.habana.utils import set_seed
from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline
from transformers.utils import is_offline_mode


def get_repo_root(model_name_or_path, local_rank=-1, token=None):
    """
    Downloads the specified model checkpoint and returns the repository where it was downloaded.
    """
    if Path(model_name_or_path).is_dir():
        # If it is a local model, no need to download anything
        return model_name_or_path
    else:
        # Checks if online or not
        if is_offline_mode():
            if local_rank == 0:
                print("Offline mode: forcing local_files_only=True")

        # Only download PyTorch weights by default
        allow_patterns = ["*.bin"]

        # Download only on first process
        if local_rank in [-1, 0]:
            cache_dir = snapshot_download(
                model_name_or_path,
                local_files_only=is_offline_mode(),
                cache_dir=os.getenv("TRANSFORMERS_CACHE", None),
                allow_patterns=allow_patterns,
                max_workers=16,
                token=token,
            )
            if local_rank == -1:
                # If there is only one process, then the method is finished
                return cache_dir

        # Make all processes wait so that other processes can get the checkpoint directly from cache
        torch.distributed.barrier()

        return snapshot_download(
            model_name_or_path,
            local_files_only=is_offline_mode(),
            cache_dir=os.getenv("TRANSFORMERS_CACHE", None),
            allow_patterns=allow_patterns,
            token=token,
        )


def get_optimized_model_name(config):
    for model_type in MODELS_OPTIMIZED_WITH_STATIC_SHAPES:
        if model_type == config.model_type:
            return model_type

    return None


def model_is_optimized(config):
    """
    Checks if the given config belongs to a model in optimum/habana/transformers/models, which has a
    new input token_idx.
    """
    return get_optimized_model_name(config) is not None


class GaudiTextGenerationPipeline(TextGenerationPipeline):
    """
    An end-to-end text-generation pipeline that can used to initialize LangChain classes.
    """
    def __init__(self, model_name_or_path=None, revision="main", **kwargs):
        self.task = "text-generation"
        self.device = "hpu"

        # Tweak generation so that it runs faster on Gaudi
        adapt_transformers_to_gaudi()
        set_seed(27)

        # Initialize tokenizer and define datatype
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, revision=revision)
        model_dtype = torch.bfloat16

        # Intialize model
        get_repo_root(model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, revision=revision, torch_dtype=model_dtype)
        model = model.eval().to(self.device)
        is_optimized = model_is_optimized(model.config)
        model = wrap_in_hpu_graph(model)
        self.model = model

        # Used for padding input to fixed length
        self.tokenizer.padding_side = "left"
        self.max_padding_length = kwargs.get("max_padding_length", self.model.config.max_position_embeddings)

        # Define config params for llama and mistral models
        if self.model.config.model_type in ["llama", "mistral"]:
            self.model.generation_config.pad_token_id = 0
            self.model.generation_config.bos_token_id = 1
            self.model.generation_config.eos_token_id = 2
            self.tokenizer.bos_token_id = self.model.generation_config.bos_token_id
            self.tokenizer.eos_token_id = self.model.generation_config.eos_token_id
            self.tokenizer.pad_token_id = self.model.generation_config.pad_token_id
            self.tokenizer.pad_token = self.tokenizer.decode(self.tokenizer.pad_token_id)
            self.tokenizer.eos_token = self.tokenizer.decode(self.tokenizer.eos_token_id)
            self.tokenizer.bos_token = self.tokenizer.decode(self.tokenizer.bos_token_id)

        # Applicable to models that do not have pad tokens
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id

        # Edit generation configuration based on input arguments
        self.generation_config = copy.deepcopy(self.model.generation_config)
        self.generation_config.max_new_tokens = kwargs.get("max_new_tokens", 100)
        self.generation_config.use_cache = kwargs.get("use_kv_cache", True)
        self.generation_config.static_shapes = is_optimized
        self.generation_config.do_sample = kwargs.get("do_sample", False)
        self.generation_config.num_beams = kwargs.get("num_beams", 1)
        self.generation_config.temperature = kwargs.get("temperature", 1.0)
        self.generation_config.top_p = kwargs.get("top_p", 1.0)
        self.generation_config.repetition_penalty = kwargs.get("repetition_penalty", 1.0)
        self.generation_config.num_return_sequences = kwargs.get("num_return_sequences", 1)
        self.generation_config.bad_words_ids = None
        self.generation_config.force_words_ids = None
        self.generation_config.ignore_eos = False

        # Define empty post-process params dict as there is no postprocesing
        self._postprocess_params = {}

        # Warm-up hpu and compile computation graphs
        self.compile_graph()

    def __call__(self, prompt: List[str]):
        """
        __call__ method of pipeline class
        """
        # Tokenize input string
        model_inputs = self.tokenizer.encode_plus(prompt[0], return_tensors="pt", max_length=self.max_padding_length, padding="max_length", truncation=True)

        # Move tensors to hpu
        for t in model_inputs:
            if torch.is_tensor(model_inputs[t]):
                model_inputs[t] = model_inputs[t].to(self.device)

        # Call model's generate method
        output = self.model.generate(**model_inputs, generation_config=self.generation_config, lazy_mode=True, hpu_graphs=True, profiling_steps=0, profiling_warmup_steps=0).cpu()

        # Decode and return result
        output_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        del output, model_inputs
        return [{"generated_text": output_text}]

    def compile_graph(self):
        """
        Function to compile computation graphs and synchronize hpus.
        """
        for _ in range(3):
            self(["Here is my prompt"])
        torch_hpu.synchronize()

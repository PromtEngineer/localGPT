Great! Here's a draft of the usage guide based on the information you've provided:

# Usage Guide for LocalGPT

LocalGPT is a powerful language model that can perform a variety of tasks including document retrieval, question
answering, and more. This guide will walk you through how to use the `ggml.py` and `run.py` scripts in LocalGPT.

## Using `ggml.py`

The `ggml.py` script is used to run the LlamaCpp model for document retrieval and question answering. Here's how to use
it:

1. Open your terminal and navigate to the directory where LocalGPT is located.

2. Run the `ggml.py` script with the desired options. Here's an example command:

```bash
python -m localGPT.ggml --show_sources 1 --low_vram 1 --n_gpu_layers 8192 --f16_kv 1 --max_tokens 512 --repo_id "TheBloke/orca_mini_3B-GGML" --filename "orca-mini-3b.ggmlv3.q4_1.bin" --prompt "What is Bash?"
```

This command will run the LlamaCpp model with the specified options and return an answer to the question "What is
Bash?".

The `ggml.py` script also supports a chat loop, which allows for more interactive and dynamic conversations with the
model. To enable the chat loop, use the `--chat` option:

```bash
python -m localGPT.ggml --chat
```

## Using `run.py`

The `run.py` script is used to execute the information retrieval task using a Question-Answer retrieval chain. Here's
how to use it:

1. Open your terminal and navigate to the directory where LocalGPT is located.

2. Run the `run.py` script with the desired options. Here's an example command:

```bash
python -m localGPT.run --repo_id "TheBloke/vicuna-7B-1.1-HF" --model_class "huggingface" --safetensors "model.safetensors" --embeddings_repo_id "hkunlp/instructor-large" --embeddings_class "HuggingFaceInstructEmbeddings" --torch_device_type "cuda" --path_database "/path/to/database" --show_sources False --use_triton False
```

This command will execute the information retrieval task with the specified options.

Please replace `"/path/to/database"` with the actual path to your embeddings database.

That's it! You're now ready to use LocalGPT. If you have any questions or run into any issues, don't hesitate to ask for
help.

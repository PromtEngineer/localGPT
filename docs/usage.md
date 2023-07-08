# Usage Guide

This guide provides instructions on how to use localGPT via the Command Line Interface (CLI). The Web UI is not covered
in this guide.

## Platform-Specific Instructions

The instructions provided in this guide are based on a Linux environment. For Windows and Mac users, the commands may
vary slightly. For example, the path to the documents and database may need to be specified differently. Always refer to
the official documentation or platform-specific guides for accurate instructions.

## Directory Structure

Understanding the directory structure of localGPT is important for navigating and using the tool effectively. Here is
the directory structure:

```
localGPT
├── api.py
├── database
│   ├── chroma.py
│   ├── document.py
│   ├── __init__.py
│   └── registry.py
├── ggml.py
├── ingest.py
├── __init__.py
├── model
│   ├── ggml.py
│   ├── gptq.py
│   ├── huggingface.py
│   ├── __init__.py
│   ├── llama.py
│   └── loader.py
└── run.py
```

## Running localGPT

To run localGPT, navigate to the root directory of localGPT and run the `run.py` script using the `-m` option:

```bash
python -m localGPT.run
```

This will start localGPT and you can begin using its features. The `-m` option is important as it ensures that the
Python interpreter will run the `run.py` script as a module, which allows the script to load other modules in the same
package, ensuring that the new script is used instead of the original one.

All command-line tools in localGPT support the `--help` option, which provides more detailed information about the
tool's usage and available options. For example, to get help on the `run.py` script, you can use:

```bash
python -m localGPT.run --help
```

This will display a help message with information about how to use the `run.py` script and the options it supports.

## Ingesting Documents

The `ingest.py` script is used to ingest documents, generate embeddings, and persist them to the Chroma database. The
script loads documents from the source directory, splits them into chunks, generates embeddings using the specified
embedding model and type, and persists the embeddings to the Chroma database.

Here is an example of how to use `ingest.py`:

```bash
python -m localGPT.ingest --path_documents /path/to/documents --path_database /path/to/database --repo_id hkunlp/instructor-large --embeddings_class HuggingFaceInstructEmbeddings --device_type cuda
```

This command will load documents from the specified path, generate embeddings using the `hkunlp/instructor-large` model
and `HuggingFaceInstructEmbeddings` type, and persist the embeddings to the specified database path. The embeddings will
be generated on the `cuda` device.

Please note that the specific usage and commands may vary based on your specific setup and the version of localGPT you
are using. Always refer to the official documentation for the most accurate and up-to-date information.

## Using `ggml.py`

The `ggml.py` script is used to run the LlamaCpp model for document retrieval and question answering. This script is
designed for lower-end consumer devices due to its efficient quantization. Here's how to use it:

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

The `run.py` script is used to execute the information retrieval task using a Question-Answer retrieval chain. This
script is designed for models from sources like Hugging Face and GPTQ, which typically require more resources. Here's
how to use it:

1. Open your terminal and navigate to the directory where LocalGPT is located.

2. Run the `run.py` script with the desired options. Here's an example command:

```bash
python -m localGPT.run --repo_id "TheBloke/vicuna-7B-1.1-HF" --model_class "huggingface" --safetensors "model.safetensors" --embeddings_repo_id "hkunlp/instructor-large" --embeddings_class "HuggingFaceInstructEmbeddings" --torch_device_type "cuda" --path_database "/path/to/database" --show_sources False --use_triton False
```

This command will execute the information retrieval task with the specified options.

Please replace `"/path/to/database"` with the actual path to your embeddings database.

That's it!

You're now ready to use LocalGPT. If you have any questions or run into any issues, don't hesitate to ask for help.

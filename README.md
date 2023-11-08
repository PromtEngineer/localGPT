# DEX Explorer: Your Comprehensive DEX Information Tool

**DEX Explorer** is a tool designed to provide crucial insights about Decentralized Exchanges (DEXs). It leverages online data and open-source Language Models (LLMs) to deliver information about a DEX of your choice or from the top DEXs on CoinMarketCap. You can also utilize the OpenAI API by entering your OpenAI IDs.

## Features 🌟
- **Versatile Model Support**: Seamlessly integrate a variety of open-source models, including HF, GPTQ, GGML, and GGUF.
- **Diverse Embeddings**: Choose from a range of open-source embeddings.
- **Reuse Your LLM**: Once downloaded, reuse your LLM without the need for repeated downloads.
- **GPU, CPU & MPS Support**: Supports mulitple platforms out of the box, Chat with your data using `CUDA`, `CPU` or `MPS` and more!

## Technical Details 🛠️

This project was inspired by the original [privateGPT](https://github.com/imartinez/privateGPT) and [localGPT](https://github.com/PromtEngineer/localGPT). This project is based on a fork from localGPT and uses the same codebase, but has been modified for to be used for different purpose.

## Built Using 🧩
- [localGPT](https://github.com/PromtEngineer/localGPT)
- [Gradio](https://github.com/gradio-app/gradio/tree/main)
- [LangChain](https://github.com/hwchase17/langchain)
- [HuggingFace LLMs](https://huggingface.co/models)
- [InstructorEmbeddings](https://instructor-embedding.github.io/)
- [LLAMACPP](https://github.com/abetlen/llama-cpp-python)
- [ChromaDB](https://www.trychroma.com/)
- [Streamlit](https://streamlit.io/)

# Environment Setup 🌍

1. 📥 Clone the repo using git:

```shell
git clone https://github.com/moustapha00/DEXexplorer.git
```

2. 🐍 Install [conda](https://www.anaconda.com/download) for virtual environment management. Create and activate a new virtual environment.

```shell
conda create -n dexplorer python=3.10.0
conda activate dexplorer
```

3. 🛠️ Install the dependencies using pip

To set up your environment to run the code, first install all requirements:

```shell
pip install -r requirements.txt
```

***Installing LLAMA-CPP :***

We use [LlamaCpp-Python](https://github.com/abetlen/llama-cpp-python)
- For GGML quantized models, you will need llama-cpp-python <=0.1.76.
- For GGUF quantized models, you will need llama-cpp-python >=0.1.83.


If you want to use BLAS or Metal with [llama-cpp](https://github.com/abetlen/llama-cpp-python#installation-with-openblas--cublas--clblast--metal) you can set appropriate flags:

For `NVIDIA` GPUs support, use `cuBLAS`

```shell
# Example: cuBLAS
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python==0.1.83 --no-cache-dir
```

For Apple Metal (`M1/M2`) support, use

```shell
# Example: METAL
CMAKE_ARGS="-DLLAMA_METAL=on"  FORCE_CMAKE=1 pip install llama-cpp-python==0.1.83 --no-cache-dir
```
For more details, please refer to [llama-cpp](https://github.com/abetlen/llama-cpp-python#installation-with-openblas--cublas--clblast--metal)

### app
The `app` folder is structured like this:

- `app/`: Main application directory.
    - `__init__.py`: Initializes the Python package.
    - `app.py`: The main entry point of the application.
    - `variables.py`: Contains the variables used in the application.
    - `utils.py`: Contains utility functions used across the application.
    - `ingest.py`: Contains functions for ingesting data.
    - `load_models.py`: Contains functions for loading large language models.
    - `run_localGPT.py`: Implements the main information retrieval task with the loaded llm.
    - `url_map.json`: Contains mapping of file names to their corresponding URLs.
    - `models/`: Directory for storing model files.
    - `dataframes/`: Directory for storing excel tables (table and dex list).
    - `data/`: Directory for storing scraped data for each DEX.
    - `persist/`: Directory for storing embeddings for each DEX.
    - `queries/`: Directory for storing queries. Each feature has its own txt file. You can edit the queries to your liking.

### variables.py

This file contains all the variables used in the application.
The main variables are:
- `DEVICE_EMBEDDING`: The device to run the embedding process on. Default is "cpu". If you have installed use "cuda" instead. Use "mps" for M1/M2 devices.
- `DEVICE_MODEL`: The device to run the model on. Default is "cpu". If you have installed use "cuda" instead. Use "mps" for M1/M2 devices.
- `GITHUB_API_KEY`: Your GitHub API key.
- `OPENAI_API_KEY`: Your OpenAI API key.
- `OPENAI_ORGANIZATION`: Your OpenAI organization ID.
- `SOURCE_DIRECTORY`: The directory where the source data is stored.
- `PERSIST_DIRECTORY`: The directory where the embeddings are stored.
- `MODEL_ID`: The ID of the model to use.
- `MODEL_BASENAME`: The base name of the model to use.
- `promptTemplate_type`: The type of prompt template to use. It's determined by the `MODEL_ID`.
- `EMBEDDING_NAME`: The name of the embedding model to use.
If you are using an `OPENAI_API_KEY` there is no need to define embedding and model related variables.

Note: When you run this for the first time, it will need internet access to download the embedding model (default: `Instructor Embedding`). In the subseqeunt runs it will use the downloaded model.

# Run the Application 🚀

1. In `app` folder open `variables.py` and define your `OPENAI_API_KEY` and `OPENAI_ORGANIZATION` if you have one. If you don't have one, you can skip this step.

2. Define the `MODEL_ID` and `MODEL_BASENAME` for the model you want to use. If you don't know how to do this, please refer to the section [How to select different LLM models?](#how-to-select-different-llm-models).

3. Define the `GITHUB_API_KEY`. You can create a new GitHub API key by following the steps [here](https://docs.github.com/en/github/authenticating-to-github/keeping-your-account-and-data-secure/creating-a-personal-access-token).

4. Open up a terminal and activate your python environment that contains the dependencies installed from requirements.txt.

5. Navigate to the `/app` directory.

6. Run the following command `python app.py`.

7. Wait until everything has loaded in. You should see `Running on local URL:  http://127.0.0.1:7860`

8. Open up a web browser and go the address ` http://127.0.0.1:7860`.


# How to select different LLM models?

To change the models you will need to set both `MODEL_ID` and `MODEL_BASENAME`.

1. Open up `variables.py`.
2. Change the `MODEL_ID` and `MODEL_BASENAME`. If you are using a quantized model (`GGML`, `GPTQ`, `GGUF`), you will need to provide `MODEL_BASENAME`. For unquatized models, set `MODEL_BASENAME` to `NONE`
5. There are a number of example models from HuggingFace that have already been tested to be run with the original trained model (ending with HF or have a .bin in its "Files and versions"), and quantized models (ending with GPTQ or have a .no-act-order or .safetensors in its "Files and versions").
6. For models that end with HF or have a .bin inside its "Files and versions" on its HuggingFace page.

   - Make sure you have a `MODEL_ID` selected. For example -> `MODEL_ID = "TheBloke/guanaco-7B-HF"`
   - Go to the [HuggingFace Repo](https://huggingface.co/TheBloke/guanaco-7B-HF)

7. For models that contain GPTQ in its name and or have a .no-act-order or .safetensors extension inside its "Files and versions on its HuggingFace page.

   - Make sure you have a `MODEL_ID` selected. For example -> model_id = `"TheBloke/wizardLM-7B-GPTQ"`
   - Got to the corresponding [HuggingFace Repo](https://huggingface.co/TheBloke/wizardLM-7B-GPTQ) and select "Files and versions".
   - Pick one of the model names and set it as  `MODEL_BASENAME`. For example -> `MODEL_BASENAME = "wizardLM-7B-GPTQ-4bit.compat.no-act-order.safetensors"`

8. Follow the same steps for `GGUF` and `GGML` models.

# GPU and vRAM Requirements

Below is the vRAM requiment for different models depending on their size (Billions of paramters). The estimates in the table does not include vRAM used by the Embedding models - which use an additional 2GB-7GB of VRAM depending on the model.

| Mode Size (B) | float32   | float16   | GPTQ 8bit      | GPTQ 4bit          |
| ------- | --------- | --------- | -------------- | ------------------ |
| 7B      | 28 GB     | 14 GB     | 7 GB - 9 GB    | 3.5 GB - 5 GB      |
| 13B     | 52 GB     | 26 GB     | 13 GB - 15 GB  | 6.5 GB - 8 GB      |
| 32B     | 130 GB    | 65 GB     | 32.5 GB - 35 GB| 16.25 GB - 19 GB   |
| 65B     | 260.8 GB  | 130.4 GB  | 65.2 GB - 67 GB| 32.6 GB - 35 GB    |


# System Requirements

## Python Version

To use this software, you must have Python 3.10 or later installed. Earlier versions of Python will not compile.

## C++ Compiler

If you encounter an error while building a wheel during the `pip install` process, you may need to install a C++ compiler on your computer.

### For Windows 10/11

To install a C++ compiler on Windows 10/11, follow these steps:

1. Install Visual Studio 2022.
2. Make sure the following components are selected:
   - Universal Windows Platform development
   - C++ CMake tools for Windows
3. Download the MinGW installer from the [MinGW website](https://sourceforge.net/projects/mingw/).
4. Run the installer and select the "gcc" component.

### NVIDIA Driver's Issues:

Follow this [page](https://linuxconfig.org/how-to-install-the-nvidia-drivers-on-ubuntu-22-04) to install NVIDIA Drivers.

# Common Errors

 - [Torch not compatible with CUDA enabled](https://github.com/pytorch/pytorch/issues/30664)

   -  Get CUDA version
      ```shell
      nvcc --version
      ```
      ```shell
      nvidia-smi
      ```
   - Try installing PyTorch depending on your CUDA version
      ```shell
         conda install -c pytorch torchvision cudatoolkit=10.1 pytorch
      ```
   - If it doesn't work, try reinstalling
      ```shell
         pip uninstall torch
         pip cache purge
         pip install torch -f https://download.pytorch.org/whl/torch_stable.html
      ```

- [ERROR: pip's dependency resolver does not currently take into account all the packages that are installed](https://stackoverflow.com/questions/72672196/error-pips-dependency-resolver-does-not-currently-take-into-account-all-the-pa/76604141#76604141)
  ```shell
     pip install h5py
     pip install typing-extensions
     pip install wheel
  ```
- [Failed to import transformers](https://github.com/huggingface/transformers/issues/11262)
  - Try re-install
    ```shell
       conda uninstall tokenizers, transformers
       pip install transformers
    ```

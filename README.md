# localGPT

This project was inspired by the original [privateGPT](https://github.com/imartinez/privateGPT). Most of the description here is inspired by the original privateGPT.

For detailed overview of the project, Watch these videos
- [Detailed code-walkthrough](https://youtu.be/MlyoObdIHyo).
- [Llama-2 with LocalGPT](https://youtu.be/lbFmceo4D5E)
- [Adding Chat History](https://youtu.be/d7otIM_MCZs)

In this model, I have replaced the GPT4ALL model with Vicuna-7B model and we are using the InstructorEmbeddings instead of LlamaEmbeddings as used in the original privateGPT. Both Embeddings as well as LLM will run on GPU instead of CPU. It also has CPU support if you do not have a GPU (see below for instruction).

Ask questions to your documents without an internet connection, using the power of LLMs. 100% private, no data leaves your execution environment at any point. You can ingest documents and ask questions without an internet connection!

Built with [LangChain](https://github.com/hwchase17/langchain) and [Vicuna-7B](https://huggingface.co/TheBloke/vicuna-7B-1.1-HF) (+ alot more!) and [InstructorEmbeddings](https://instructor-embedding.github.io/)

# Environment Setup

Install conda

```shell
conda create -n localGPT
```

Activate

```shell
conda activate localGPT
```

In order to set your environment up to run the code here, first install all requirements:

```shell
pip install -r requirements.txt
```


If you want to use BLAS or Metal with [llama-cpp](<(https://github.com/abetlen/llama-cpp-python#installation-with-openblas--cublas--clblast--metal)>) you can set appropriate flags:

```shell
# Example: cuBLAS
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install -r requirements.txt
```

## Docker

Installing the required packages for GPU inference on Nvidia GPUs, like gcc 11 and CUDA 11, may cause conflicts with other packages in your system.
As an alternative to Conda, you can use Docker with the provided Dockerfile.
It includes CUDA, your system just needs Docker, BuildKit, your Nvidia GPU driver and the Nvidia container toolkit.
Build as `docker build . -t localgpt`, requires BuildKit.
Docker BuildKit does not support GPU during *docker build* time right now, only during *docker run*.
Run as `docker run -it --mount src="$HOME/.cache",target=/root/.cache,type=bind --gpus=all localgpt`.

## Test dataset

This repo uses a [Constitution of USA ](https://constitutioncenter.org/media/files/constitution.pdf) as an example.

## Instructions for ingesting your own dataset

Put any and all of your .txt, .pdf, or .csv files into the SOURCE_DOCUMENTS directory
in the load_documents() function, replace the docs_path with the absolute path of your source_documents directory.

The current default file types are .txt, .pdf, .csv, and .xlsx, if you want to use any other file type, you will need to convert it to one of the default file types.

Run the following command to ingest all the data.

`defaults to cuda`

```shell
python ingest.py
```

Use the device type argument to specify a given device.

```sh
python ingest.py --device_type cpu
```

Use help for a full list of supported devices.

```sh
python ingest.py --help
```

It will create an index containing the local vectorstore. Will take time, depending on the size of your documents.
You can ingest as many documents as you want, and all will be accumulated in the local embeddings database.
If you want to start from an empty database, delete the `index`.

Note: When you run this for the first time, it will download take time as it has to download the embedding model. In the subseqeunt runs, no data will leave your local enviroment and can be run without internet connection.

## Ask questions to your documents, locally!

In order to ask a question, run a command like:

```shell
python run_localGPT.py
```

And wait for the script to require your input.

```shell
> Enter a query:
```

Hit enter. Wait while the LLM model consumes the prompt and prepares the answer. Once done, it will print the answer and the 4 sources it used as context from your documents; you can then ask another question without re-running the script, just wait for the prompt again.

Note: When you run this for the first time, it will need internet connection to download the vicuna-7B model. After that you can turn off your internet connection, and the script inference would still work. No data gets out of your local environment.

Type `exit` to finish the script.

# Run it on CPU

By default, localGPT will use your GPU to run both the `ingest.py` and `run_localGPT.py` scripts. But if you do not have a GPU and want to run this on CPU, now you can do that (Warning: Its going to be slow!). You will need to use `--device_type cpu`flag with both scripts.

For Ingestion run the following:

```shell
python ingest.py --device_type cpu
```

In order to ask a question, run a command like:

```shell
python run_localGPT.py --device_type cpu
```

# Run quantized for M1/M2:

GGML quantized models for Apple Silicon (M1/M2) are supported through the llama-cpp library, [example](https://huggingface.co/TheBloke/Wizard-Vicuna-13B-Uncensored-GGML). GPTQ quantized models that leverage auto-gptq will not work, [see here](https://github.com/PanQiWei/AutoGPTQ/issues/133#issuecomment-1575002893). GGML models will work for CPU or MPS.

## Troubleshooting

**Install MPS:**
1- Follow this [page](https://developer.apple.com/metal/pytorch/) to build up PyTorch with Metal Performance Shaders (MPS) support. PyTorch uses the new MPS backend for GPU training acceleration. It is good practice to verify mps support using a simple Python script as mentioned in the provided link.

2- By following the page, here is an example of what you may initiate in your terminal

```shell
xcode-select --install
conda install pytorch torchvision torchaudio -c pytorch-nightly
pip install chardet
pip install cchardet
pip uninstall charset_normalizer
pip install charset_normalizer
pip install pdfminer.six
pip install xformers
```

**Upgrade packages:**
Your langchain or llama-cpp version could be outdated. Upgrade your packages by running install again.

```shell
pip install -r requirements.txt
```

If you are still getting errors, try installing the latest llama-cpp-python with these flags, and [see thread](https://github.com/abetlen/llama-cpp-python/issues/317#issuecomment-1587962205).

```shell
CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 pip install -U llama-cpp-python --no-cache-dir
```

# Run the UI

1. Open `constants.py` in an editor of your choice and depending on choice add the LLM you want to use. By default, the following model will be used:

   ```shell
   MODEL_ID = "TheBloke/Llama-2-7B-Chat-GGML"
   MODEL_BASENAME = "llama-2-7b-chat.ggmlv3.q4_0.bin"
   ```

3. Open up a terminal and activate your python environment that contains the dependencies installed from requirements.txt.

4. Navigate to the `/LOCALGPT` directory.

5. Run the following command `python run_localGPT_API.py`. The API should being to run.

6. Wait until everything has loaded in. You should see something like `INFO:werkzeug:Press CTRL+C to quit`.

7. Open up a second terminal and activate the same python environment.

8. Navigate to the `/LOCALGPT/localGPTUI` directory.

9. Run the command `python localGPTUI.py`.

10. Open up a web browser and go the address `http://localhost:5111/`.

# How does it work?

Selecting the right local models and the power of `LangChain` you can run the entire pipeline locally, without any data leaving your environment, and with reasonable performance.

- `ingest.py` uses `LangChain` tools to parse the document and create embeddings locally using `InstructorEmbeddings`. It then stores the result in a local vector database using `Chroma` vector store.
- `run_localGPT.py` uses a local LLM to understand questions and create answers. The context for the answers is extracted from the local vector store using a similarity search to locate the right piece of context from the docs.
- You can replace this local LLM with any other LLM from the HuggingFace. Make sure whatever LLM you select is in the HF format.

# How to select different LLM models?

The following will provide instructions on how you can select a different LLM model to create your response:

1. Open up `constants.py` in the editor of your choice.
2. Change the `MODEL_ID` and `MODEL_BASENAME`. If you are using a quantized model (`GGML`, `GPTQ`), you will need to provide `MODEL_BASENAME`. For unquatized models, set `MODEL_BASENAME` to `NONE`
5. There are a number of example models from HuggingFace that have already been tested to be run with the original trained model (ending with HF or have a .bin in its "Files and versions"), and quantized models (ending with GPTQ or have a .no-act-order or .safetensors in its "Files and versions").
6. For models that end with HF or have a .bin inside its "Files and versions" on its HuggingFace page.

   - Make sure you have a model_id selected. For example -> `MODEL_ID = "TheBloke/guanaco-7B-HF"`
   - If you go to its HuggingFace [repo](https://huggingface.co/TheBloke/guanaco-7B-HF) and go to "Files and versions" you will notice model files that end with a .bin extension.
   - Any model files that contain .bin extensions will be run with the following code where the `# load the LLM for generating Natural Language responses` comment is found.
   - `MODEL_ID = "TheBloke/guanaco-7B-HF"`

7. For models that contain GPTQ in its name and or have a .no-act-order or .safetensors extension inside its "Files and versions on its HuggingFace page.

   - Make sure you have a model_id selected. For example -> model_id = `"TheBloke/wizardLM-7B-GPTQ"`
   - You will also need its model basename file selected. For example -> `model_basename = "wizardLM-7B-GPTQ-4bit.compat.no-act-order.safetensors"`
   - If you go to its HuggingFace [repo](https://huggingface.co/TheBloke/wizardLM-7B-GPTQ) and go to "Files and versions" you will notice a model file that ends with a .safetensors extension.
   - Any model files that contain no-act-order or .safetensors extensions will be run with the following code where the `# load the LLM for generating Natural Language responses` comment is found.
   - `MODEL_ID = "TheBloke/WizardLM-7B-uncensored-GPTQ"`

     `MODEL_BASENAME = "WizardLM-7B-uncensored-GPTQ-4bit-128g.compat.no-act-order.safetensors"`


8. Comment out all other instances of `MODEL_ID="other model names"`, `MODEL_BASENAME=other base model names`, and `llm = load_model(args*)`

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

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=PromtEngineer/localGPT&type=Date)](https://star-history.com/#PromtEngineer/localGPT&Date)

# Disclaimer

This is a test project to validate the feasibility of a fully local solution for question answering using LLMs and Vector embeddings. It is not production ready, and it is not meant to be used in production. Vicuna-7B is based on the Llama model so that has the original Llama license.

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

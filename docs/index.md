# localGPT

This project was inspired by the original [privateGPT](https://github.com/imartinez/privateGPT). Most of the description
here is inspired by the original privateGPT.

For detailed overview of the project, Watch this [Youtube Video](https://youtu.be/MlyoObdIHyo).

In this model, I have replaced the GPT4ALL model with Vicuna-7B model and we are using the InstructorEmbeddings instead
of LlamaEmbeddings as used in the original privateGPT. Both Embeddings as well as LLM will run on GPU instead of CPU. It
also has CPU support if you do not have a GPU (see below for instruction).

Ask questions to your documents without an internet connection, using the power of LLMs. 100% private, no data leaves
your execution environment at any point. You can ingest documents and ask questions without an internet connection!

Built with [LangChain](https://github.com/hwchase17/langchain) and
[Vicuna-7B](https://huggingface.co/TheBloke/vicuna-7B-1.1-HF) and
[InstructorEmbeddings](https://instructor-embedding.github.io/)

# Build local documentation

For full documentation visit [mkdocs.org](https://www.mkdocs.org).

## Commands

- `mkdocs new [dir-name]` - Create a new project.
- `mkdocs serve` - Start the live-reloading docs server.
- `mkdocs build` - Build the documentation site.
- `mkdocs -h` - Print help message and exit.

## Project layout

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files.

# Environment Setup

In order to set your environment up to run the code here, first install all requirements:

```shell
pip install -r requirements.txt
```

Then install AutoGPTQ - if you want to run quantized models for GPU

```shell
git clone https://github.com/PanQiWei/AutoGPTQ.git
cd AutoGPTQ
git checkout v0.2.2
pip install .
```

For more support on [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ).

## Test dataset

This repo uses a [Constitution of USA](https://constitutioncenter.org/media/files/constitution.pdf) as an example.

## Instructions for Ingesting Your Own Dataset

The following instructions will guide you through the process of ingesting your own dataset using the `localGPT` tool.

### Dataset Preparation

1. Ensure that your dataset is stored in the `SOURCE_DOCUMENTS` directory.
   - The default location for the `SOURCE_DOCUMENTS` directory is based on the package structure but can be customized
     if needed.
   - Supported document types include `.txt`, `.pdf`, `.py`, and `.csv`.
   - Additional file types can be supported by extending the embedding types, models, or sources.

### Ingestion Command

To ingest your dataset and generate embeddings, use the following command:

```shell
python -m localGPT.ingest --source_directory <source_directory> --persist_directory <persist_directory> --embedding_model <embedding_model> --embedding_type <embedding_type> --device_type <device_type>
```

Replace the following placeholders:

- `<source_directory>`: The path to the directory containing your source documents.
- `<persist_directory>`: The path to the directory where the embeddings will be written.
- `<embedding_model>`: The desired embedding model to use for generating embeddings.
- `<embedding_type>`: The type of embeddings to use.
- `<device_type>`: The device type to run the embeddings on.

For example, to ingest documents from the `data/documents` directory using the `hkunlp/instructor-large` embedding model
and `HuggingFaceInstructEmbeddings` embedding type, and running on the `cuda` device, you would run the following
command:

```shell
python -m localGPT.ingest --source_directory data/documents --persist_directory data/embeddings --embedding_model hkunlp/instructor-large --embedding_type HuggingFaceInstructEmbeddings --device_type cuda
```

### Data Ingestion Process

Running the ingestion command will load the documents from the specified source directory, split them into chunks,
generate embeddings using the selected embedding model and type, and persist the embeddings to the Chroma database.

- The number of documents loaded and the chunks created will be logged during the process.
- The time required for ingestion depends on the size of your documents.
- All ingested documents will be accumulated in the local embeddings database.

### Initial Setup

The first time you run the ingestion process, it may require downloading the required embedding model, which could take
some time. However, subsequent runs will not require an internet connection as the data remains within your local
environment.

Note: If you want to start with an empty database, you can delete the existing `index` file located in the persist
directory.

Please ensure that you replace the placeholders with the actual paths and values specific to your setup.

## Ask Questions to Your Documents Locally

You can use the `localGPT` package's CLI tool to ask questions and retrieve answers from your documents locally. Follow
the instructions below to run the script and interactively ask questions:

1. Open a terminal or command prompt.
2. Navigate to the directory where the `localGPT` package is located.
3. To ingest your dataset, run the following command:

```shell
python -m localGPT.ingest
```

4. Wait for the ingestion process to complete. The script will load the documents, generate embeddings, and persist them
   to the Chroma database.

   - The time required for ingestion depends on the size of your documents.

5. Once the ingestion process is complete, you can run the question-answering script. Run the following command:

```shell
python -m localGPT.run
```

`localGPT.run` supports the following options:

```shell
python -m localGPT.run --model_repository <model_repository> --model_safetensors <model_safetensors> --embedding_model <embedding_model> --embedding_type <embedding_type> --device_type <device_type> --persist_directory <persist_directory> --show_sources
```

Replace the following placeholders:

- `<model_repository>`: The desired model repository.
- `<model_safetensors>`: The desired model safetensors. This is only applicable for GPTQ models, not GGML models.
- `<embedding_model>`: The embedding model repository used during ingestion.
- `<embedding_type>`: The embedding model type used during ingestion.
- `<device_type>`: The device type used during ingestion.
- `<persist_directory>`: The directory where the embeddings are persisted.

For example, to run the question-answering script using the `TheBloke/WizardLM-7B-uncensored-GPTQ` model repository,
`WizardLM-7B-uncensored-GPTQ-4bit-128g.compat.no-act-order.safetensors` model safetensors, `hkunlp/instructor-large`
embedding model, `HuggingFaceInstructEmbeddings` embedding type, `cuda` device type, and displaying the sources, you
would run the following command:

```shell
python -m localGPT.run --model_repository TheBloke/WizardLM-7B-uncensored-GPTQ --model_safetensors WizardLM-7B-uncensored-GPTQ-4bit-128g.compat.no-act-order.safetensors --embedding_model hkunlp/instructor-large --embedding_type HuggingFaceInstructEmbeddings --device_type cuda --persist_directory data/embeddings --show_sources
```

Note that the `--model_safetensors` option is only applicable for GPTQ models and should not be used with GGML models.

6. Wait for the script to require your input.

   - The script will display the prompt: `> Enter a query:`
   - You can enter your question or query at this prompt.
   - Hit Enter to submit the query.

7. The script will process the query using the Question-Answer retrieval chain and provide an answer.

   - The answer will be displayed below the query prompt.
   - The script will also display the sources used as context from your documents (if enabled).

8. You can ask another question without re-running the script.

   - Simply wait for the prompt to appear again after the previous answer.
   - Enter your next question and hit Enter to get the answer

9. To exit the script, enter `exit` when prompted for a query.

### Notes:

- The first time you run the ingestion script, it may require an internet connection to download the necessary model
  files.
  - After the initial download, the script can be run offline without an internet connection.
- By default, the script uses the GPU (`cuda`) for running the retrieval chain.
  - If you want to run the script on CPU, you can specify the `--device_type cpu` flag when running both the ingestion
    and question-answering scripts.
- If you want to display the source text of the documents used for the answer, you can enable the `--show_sources`
  option when running the question-answering script.

### Example Commands:

- To run the ingestion script on CPU:

```shell
python -m localGPT.ingest --device_type cpu
```

- To run the question-answering script on CPU:

```shell
python -m localGPT.run --device_type cpu
```

These instructions provide an overview of how to use the `localGPT` package's CLI tool to ask questions and retrieve
answers from your documents locally. You can follow these steps to interactively ask questions and obtain answers
without having to re-run the script each time.

Please make sure to update the paths and default values in the instructions as needed.

# Run the UI

1. Start by opening up `run_localGPT_API.py` in a code editor of your choice. If you are using gpu skip to step 3.

2. If you are running on cpu change `DEVICE_TYPE = 'cuda'` to `DEVICE_TYPE = 'cpu'`.

   - Comment out the following:

   ```shell
   model_id = "TheBloke/WizardLM-7B-uncensored-GPTQ"
   model_basename = "WizardLM-7B-uncensored-GPTQ-4bit-128g.compat.no-act-order.safetensors"
   LLM = load_model(device_type=DEVICE_TYPE, model_id=model_id, model_basename = model_basename)
   ```

   - Uncomment:

   ```shell
   model_id = "TheBloke/guanaco-7B-HF" # or some other -HF or .bin model
   LLM = load_model(device_type=DEVICE_TYPE, model_id=model_id)
   ```

   - If you are running gpu there should be nothing to change. Save and close `run_localGPT_API.py`.

3. Open up a terminal and activate your python environment that contains the dependencies installed from
   requirements.txt.

4. Navigate to the `/LOCALGPT` directory.

5. Run the following command `python run_localGPT_API.py`. The API should being to run.

6. Wait until everything has loaded in. You should see something like `INFO:werkzeug:Press CTRL+C to quit`.

7. Open up a second terminal and activate the same python environment.

8. Navigate to the `/LOCALGPT/localGPTUI` directory.

9. Run the command `python localGPTUI.py`.

10. Open up a web browser and go the address `http://localhost:5111/`.

# How does it work?

Selecting the right local models and the power of `LangChain` you can run the entire pipeline locally, without any data
leaving your environment, and with reasonable performance.

- `ingest.py` uses `LangChain` tools to parse the document and create embeddings locally using `InstructorEmbeddings`.
  It then stores the result in a local vector database using `Chroma` vector store.
- `run_localGPT.py` uses a local LLM (Vicuna-7B in this case) to understand questions and create answers. The context
  for the answers is extracted from the local vector store using a similarity search to locate the right piece of
  context from the docs.
- You can replace this local LLM with any other LLM from the HuggingFace. Make sure whatever LLM you select is in the HF
  format.

# How to select different LLM models?

The following will provide instructions on how you can select a different LLM model to create your response:

1. Open up `run_localGPT.py`
2. Go to `def main(device_type, show_sources)`
3. Go to the comment where it says `# load the LLM for generating Natural Language responses`
4. Below it, it details a bunch of examples on models from HuggingFace that have already been tested to be run with the
   original trained model (ending with HF or have a .bin in its "Files and versions"), and quantized models (ending with
   GPTQ or have a .no-act-order or .safetensors in its "Files and versions").
5. For models that end with HF or have a .bin inside its "Files and versions" on its HuggingFace page.

   - Make sure you have a model_id selected. For example -> `model_id = "TheBloke/guanaco-7B-HF"`
   - If you go to its HuggingFace [Site] (https://huggingface.co/TheBloke/guanaco-7B-HF) and go to "Files and versions"
     you will notice model files that end with a .bin extension.
   - Any model files that contain .bin extensions will be run with the following code where the
     `# load the LLM for generating Natural Language responses` comment is found.
   - `model_id = "TheBloke/guanaco-7B-HF"`

     `llm = load_model(device_type, model_id=model_id)`

6. For models that contain GPTQ in its name and or have a .no-act-order or .safetensors extension inside its "Files and
   versions on its HuggingFace page.

   - Make sure you have a model_id selected. For example -> model_id = `"TheBloke/wizardLM-7B-GPTQ"`
   - You will also need its model basename file selected. For example ->
     `model_basename = "wizardLM-7B-GPTQ-4bit.compat.no-act-order.safetensors"`
   - If you go to its HuggingFace [Site] (https://huggingface.co/TheBloke/wizardLM-7B-GPTQ) and go to "Files and
     versions" you will notice a model file that ends with a .safetensors extension.
   - Any model files that contain no-act-order or .safetensors extensions will be run with the following code where the
     `# load the LLM for generating Natural Language responses` comment is found.
   - `model_id = "TheBloke/WizardLM-7B-uncensored-GPTQ"`

     `model_basename = "WizardLM-7B-uncensored-GPTQ-4bit-128g.compat.no-act-order.safetensors"`

     `llm = load_model(device_type, model_id=model_id, model_basename = model_basename)`

7. Comment out all other instances of `model_id="other model names"`, `model_basename=other base model names`, and
   `llm = load_model(args*)`

# System Requirements

## Python Version

To use this software, you must have Python 3.10 or later installed. Earlier versions of Python will not compile.

## C++ Compiler

If you encounter an error while building a wheel during the `pip install` process, you may need to install a C++
compiler on your computer.

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

### M1/M2 Macbook users:

1- Follow this [page](https://developer.apple.com/metal/pytorch/) to build up PyTorch with Metal Performance Shaders
(MPS) support. PyTorch uses the new MPS backend for GPU training acceleration. It is good practice to verify mps support
using a simple Python script as mentioned in the provided link.

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

3- Please keep in mind that the quantized models are not yet supported by Apple Silicon (M1/M2) by auto-gptq library
that is being used for loading quantized models,
[see here](https://github.com/PanQiWei/AutoGPTQ/issues/133#issuecomment-1575002893). Therefore, you will not be able to
run quantized models on M1/M2.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=PromtEngineer/localGPT&type=Date)](https://star-history.com/#PromtEngineer/localGPT&Date)

# Disclaimer

This is a test project to validate the feasibility of a fully local solution for question answering using LLMs and
Vector embeddings. It is not production ready, and it is not meant to be used in production. Vicuna-7B is based on the
Llama model so that has the original Llama license.

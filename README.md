# localGPT

This project was inspired by the original [privateGPT](https://github.com/imartinez/privateGPT). Most of the description here is inspired by the original privateGPT.

For detailed overview of the project, Watch this [Youtube Video](https://youtu.be/MlyoObdIHyo).

In this model, I have replaced the GPT4ALL model with a different HF model and we are using the InstructorEmbeddings instead of LlamaEmbeddings as used in the original privateGPT. Both Embeddings as well as LLM will run on GPU instead of CPU. It also has CPU support if you do not have a GPU (see below for instruction).

Ask questions to your documents without an internet connection, using the power of LLMs. 100% private, no data leaves your execution environment at any point. You can ingest documents and ask questions without an internet connection!

Built with [LangChain](https://github.com/hwchase17/langchain) and [Vicuna-7B](https://huggingface.co/TheBloke/vicuna-7B-1.1-HF) and [InstructorEmbeddings](https://instructor-embedding.github.io/)

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

For more support on [AutoGPTQ] (https://github.com/PanQiWei/AutoGPTQ).

## Test dataset

This repo uses a [Constitution of USA ](https://constitutioncenter.org/media/files/constitution.pdf) as an example.

## Edit config.py

Open up config.py and edit the following based on your computer specifications.
If you are using an Nvidia GPU, set `DEVICE_TYPE = "cuda"`.
If you are running from CPU, set `DEVICE_TYPE = "cpu"`. (Warning: Its going to be slow!)

To reset your vector database (document knowledge base) when you ingest documents from `/SOURCE_DOCUMENTS` set `RESET_DB=True`.
To add documents to your vector database and not delete existing ingested knowledge set `RESET_DB=False`.

To change your vector database location set it from `PERSIST_DIRECTORY`.

To change your models from AutoGPTQ to normal HF models. (This is only recommended if you are running from CPU).

   - Comment out the following:

   ```shell
   MODEL_ID = "TheBloke/WizardLM-7B-uncensored-GPTQ"
   MODEL_BASENAME = "WizardLM-7B-uncensored-GPTQ-4bit-128g.compat.no-act-order.safetensors"
   LLM = load_model(device_type=DEVICE_TYPE, model_id=model_id, model_basename = model_basename)
   ```

   - Uncomment:

   ```shell
   MODEL_ID = "TheBloke/guanaco-7B-HF" # or some other -HF or .bin model
   MODEL_BASENAME = None
   LLM = load_model(device_type=DEVICE_TYPE, model_id=model_id)
   ```

## Instructions for ingesting your own dataset

Put any and all of your .txt, .pdf, or .csv files into the SOURCE_DOCUMENTS directory
in the load_documents() function, replace the docs_path with the absolute path of your source_documents directory.

The current default file types are .txt, .pdf, .csv, and .xlsx, if you want to use any other file type, you will need to convert it to one of the default file types.

Run the following command to ingest all the data.

```shell
python ingest.py  # defaults to cuda, change the device type in "config.py".
```

It will create an index containing the local vectorstore. Will take time, depending on the size of your documents.
You can ingest as many documents as you want, and all will be accumulated in the local embeddings database.
If you want to start from an empty database, delete the `index` or set `RESET_DB=True` in `config.py`.

Note: When you run this for the first time, it will download take time as it has to download the embedding model. In the subseqeunt runs, no data will leave your local enviroment and can be run without internet connection.

## Ask questions to your documents, locally!

In order to ask a question, run a command like:

```shell
python localgpt_cli.py
```

And wait for the script to require your input.

```shell
> Enter a query:
```

Hit enter. Wait while the LLM model consumes the prompt and prepares the answer. Once done, it will print the answer and the 4 sources it used as context from your documents; you can then ask another question without re-running the script, just wait for the prompt again.

Note: When you run this for the first time, it will need internet connection to download the vicuna-7B model. After that you can turn off your internet connection, and the script inference would still work. No data gets out of your local environment.

To add more documents to the database, put the additional documents into the `/SOURCE_DOCUMENTS` folder and type `reingest`.

Type `exit` to finish the script.

# Run the UI

1. Open up a terminal and activate your python environment that contains the dependencies installed from requirements.txt.

2. Navigate to the `/LOCALGPT` directory.

3. Run the following command `python localgpt_api.py`.

4. Wait until everything has loaded in. You should see something like `INFO:werkzeug:Press CTRL+C to quit`.

5. Open up a second terminal and activate the same python environment.

8. Navigate to the `/LOCALGPT/UI` directory.

9. Run the command `python app.py`.

10. Open up a web browser and go the address `http://localhost:5111/`.
    
11. To stop everything exit out of your terminals.

# How does it work?

Selecting the right local models and the power of `LangChain` you can run the entire pipeline locally, without any data leaving your environment, and with reasonable performance.

- `ingest.py` uses `LangChain` tools to parse the document and create embeddings locally using `InstructorEmbeddings`. It then stores the result in a local vector database using `Chroma` vector store.
- `localgpt.py` uses a local LLM to understand questions and create answers. The context for the answers is extracted from the local vector store using a similarity search to locate the right piece of context from the docs.
- You can replace this local LLM with any other LLM from the HuggingFace. Make sure whatever LLM you select is in the HF format or AutoGPTQ format.

# How to select different LLM models?

The following will provide instructions on how you can select a different LLM model to create your response:

1. Open up `config.py`
2. Go to the comment where it says `# load the LLM for generating Natural Language responses`
3. Below it, it details a bunch of examples on models from HuggingFace that have already been tested to be run with the original trained model (ending with HF or have a .bin in its "Files and versions"), and quantized models (ending with GPTQ or have a .no-act-order or .safetensors in its "Files and versions").
4. For models that end with HF or have a .bin inside its "Files and versions" on its HuggingFace page.

   - Make sure you have a model_id selected. For example -> `model_id = "TheBloke/guanaco-7B-HF"`
   - If you go to its HuggingFace [Site] (https://huggingface.co/TheBloke/guanaco-7B-HF) and go to "Files and versions" you will notice model files that end with a .bin extension.
   - Any model files that contain .bin extensions will be run with the following code where the `# load the LLM for generating Natural Language responses` comment is found.
   - `MODEL_ID = "TheBloke/guanaco-7B-HF"`
   - `MODEL_BASENAME = None`

5. For models that contain GPTQ in its name and or have a .no-act-order or .safetensors extension inside its "Files and versions on its HuggingFace page.

   - Make sure you have a model_id selected. For example -> model_id = `"TheBloke/wizardLM-7B-GPTQ"`
   - You will also need its model basename file selected. For example -> `model_basename = "wizardLM-7B-GPTQ-4bit.compat.no-act-order.safetensors"`
   - If you go to its HuggingFace [Site] (https://huggingface.co/TheBloke/wizardLM-7B-GPTQ) and go to "Files and versions" you will notice a model file that ends with a .safetensors extension.
   - Any model files that contain no-act-order or .safetensors extensions will be run with the following code where the `# load the LLM for generating Natural Language responses` comment is found.
   - `MODEL_ID = "TheBloke/WizardLM-7B-uncensored-GPTQ"`
   - `MODEL_BASENAME = "WizardLM-7B-uncensored-GPTQ-4bit-128g.compat.no-act-order.safetensors"`

6. Comment out all other instances of `MODEL_ID="other model names"`, `MODEL_BASENAME=other base model names`

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

### M1/M2 Macbook users:

1- Follow this [page](https://developer.apple.com/metal/pytorch/) to build up PyTorch with Metal Performance Shaders (MPS) support. PyTorch uses the new MPS backend for GPU training acceleration. It is good practice to verify mps support using a simple Python script as mentioned in the provided link.

2- By following the page, here is an example of you may initiate in your terminal

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

3- Create a new `verifymps.py` in the same directory (localGPT) where you have all files and environment.

    import torch
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        x = torch.ones(1, device=mps_device)
        print (x)
    else:
        print ("MPS device not found.")

4- Find `instructor.py` and open it in VS Code to edit.

The `instructor.py` is probably embeded similar to this:

    file_path = "/System/Volumes/Data/Users/USERNAME/anaconda3/envs/LocalGPT/lib/python3.10/site-packages/InstructorEmbedding/instructor.py"

You can open the `instructor.py` and then edit it using this code:

#### Open the file in VSCode

    subprocess.run(["open", "-a", "Visual Studio Code", file_path])

Once you open `instructor.py` with VS Code, replace the code snippet that has `device_type` with the following codes:

         if device is None:
            device = self._target_device

        # Replace the line: self.to(device)

        if device in ['cpu', 'CPU']:
            device = torch.device('cpu')

        elif device in ['mps', 'MPS']:
            device = torch.device('mps')

        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.to(device)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=PromtEngineer/localGPT&type=Date)](https://star-history.com/#PromtEngineer/localGPT&Date)

# Disclaimer

This is a test project to validate the feasibility of a fully local solution for question answering using LLMs and Vector embeddings. It is not production ready, and it is not meant to be used in production. Most models are based on the Llama model so that has the original Llama license.

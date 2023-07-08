# Installation Guide

This guide provides instructions on how to install and set up localGPT, including its dependencies LangChain and
llama-cpp-python.

## Setting Up a Virtual Environment

Before installing localGPT and its dependencies, it's recommended to set up a virtual environment. This helps to keep
the dependencies required by different projects separate by creating isolated python environments for them.

### Windows and Mac Users

For Windows and Mac users, Anaconda is recommended. You can download Anaconda from
[here](https://www.anaconda.com/products/distribution).

Once Anaconda is installed, you can create a new environment using the following command:

```bash
conda create -n myenv python=3.8
```

Activate the environment using:

```bash
conda activate myenv
```

### Linux Users

For Linux users, `virtualenv` is recommended. Install `virtualenv` using pip:

```bash
pip install virtualenv
```

Create a new virtual environment:

```bash
virtualenv myenv
```

Activate the environment:

```bash
source myenv/bin/activate
```

## LangChain Installation

LangChain can be installed using pip or from source.

### Official Release

To install the bare minimum requirements of LangChain, run:

```bash
pip install langchain
```

To install modules needed for the common LLM providers, run:

```bash
pip install langchain[llms]
```

To install all modules needed for all integrations, run:

```bash
pip install langchain[all]
```

Note: If you are using zsh, you'll need to quote square brackets when passing them as an argument to a command, for
example:

```bash
pip install 'langchain[all]'
```

### From Source

If you want to install from source, you can do so by cloning the repo and running:

```bash
pip install -e .
```

## llama-cpp-python Installation

llama-cpp-python can be installed from PyPI or with different BLAS backends for faster processing.

### Installation from PyPI (recommended)

Install from PyPI (requires a C compiler):

```bash
pip install llama-cpp-python
```

If you have previously installed llama-cpp-python through pip and want to upgrade your version or rebuild the package
with different compiler options, please add the following flags to ensure that the package is rebuilt correctly:

```bash
pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
```

Note: If you are using Apple Silicon (M1) Mac, make sure you have installed a version of Python that supports arm64
architecture.

### Installation with OpenBLAS / cuBLAS / CLBlast / Metal

llama.cpp supports multiple BLAS backends for faster processing. Use the `FORCE_CMAKE=1` environment variable to force
the use of `cmake` and install the pip package for the desired BLAS backend.

To install with OpenBLAS:

```bash
CMAKE_ARGS="-DLLAMA_OPENBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python
```

To install with cuBLAS:

```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python
```

To install with CLBlast:

```bash
CMAKE_ARGS="-DLLAMA_CLBLAST=on" FORCE_CMAKE=1 pip install llama-cpp-python
```

To install with Metal (MPS):

```bash
CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 pip install llama-cpp-python
```

To install with AMD:

```bash
CMAKE_ARGS="-DGGML_OPENCL_PLATFORM=AMD -DGGML_OPENCL_DEVICE=1" FORCE_CMAKE=1 pip install llama-cpp-python
```

Detailed MacOS Metal GPU install documentation is available at [docs/install/macos.md](https://llama-cpp-python.readthedocs.io/en/latest/install/macos/).

## Additional Dependencies

The latest version of LangChain is required and is v0.0.227. There are known CVE vulnerabilities which involve Arbitrary
Code Execution and this should be noted.

Other dependencies are listed in the requirements.txt, but may vary from system to system. Here is a sample
`requirements.txt`:

```txt
# Natural Language Processing
langchain==0.0.227
chromadb==0.3.22
llama-cpp-python==0.1.66
pdfminer.six==20221105
InstructorEmbedding
sentence-transformers
faiss-cpu
huggingface_hub
transformers
protobuf==3.20.0; sys_platform != 'darwin'
protobuf==3.20.0; sys_platform == 'darwin' and platform_machine != 'arm64'
protobuf==3.20.3; sys_platform == 'darwin' and platform_machine == 'arm64'
auto-gptq
docx2txt

# Utilities
urllib3==1.26.6
accelerate
bitsandbytes
click
flask
requests
python-magic
mkdocs
mkdocstrings
mkdocs-material

# Excel File Manipulation
openpyxl
```

You can install these dependencies using pip:

```bash
pip install -r requirements.txt
```

Please note that the versions and specific dependencies may vary based on your specific setup and the version of
localGPT you are installing. Always refer to the official documentation for the most accurate and up-to-date
information.

# Use the latest version of Dockerfile syntax
# and specify that we're using Dockerfile syntax version 1
# This enables certain features like cache mounts
# Build as `docker build . -t localgpt`, requires BuildKit.
# Run as `docker run -it --mount src="$HOME/.cache",target=/root/.cache,type=bind --gpus=all localgpt`, requires Nvidia container toolkit.
# You can use these instructions to build and run the container

# Use a base image with CUDA runtime and Ubuntu 22.04
FROM nvidia/cuda:11.7.1-runtime-ubuntu22.04

# Update package manager and install necessary packages
RUN apt-get update && \
    apt-get install -y \
    software-properties-common \
    g++-11 \
    make \
    python3 \
    python-is-python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file to the container
# to install Python dependencies
COPY ./requirements.txt .

# Use BuildKit cache mount to reduce redownloading from pip on repeated builds
# Install Python dependencies listed in requirements.txt
# Additionally install llama-cpp-python==0.1.83 package
RUN --mount=type=cache,target=/root/.cache \
    CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 \
    pip install --timeout 100 -r requirements.txt llama-cpp-python==0.1.83

# Copy source documents and utility files into the container
COPY SOURCE_DOCUMENTS ./SOURCE_DOCUMENTS
COPY utils.py .
COPY ingest.py constants.py ./

# Run the ingest script with the specified device type argument
# This step might need internet connection to download required models or embeddings
# Docker BuildKit does not support GPU during *docker build* time right now, only during *docker run*.
# See <https://github.com/moby/buildkit/issues/1436>.
# If this changes in the future you can `docker build --build-arg device_type=cuda  . -t localgpt` (+GPU argument to be determined).
ARG device_type=cpu
RUN --mount=type=cache,target=/root/.cache \
    python ingest.py --device_type $device_type

# Copy the rest of the application code into the container
COPY . .

# Set the environment variable for the device type
ENV device_type=cuda

# Specify the command to run when the container starts
# This command runs the Python script for localGPT with the specified device type
CMD python run_localGPT.py --device_type $device_type

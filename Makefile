# Makefile for Docker Compose operations

# Default values for environment variables
DEVICE_TYPE := cpu
SOURCE_DOCUMENTS := SOURCE_DOCUMENTS  # Replace with your default source documents directory

# Default target executed when no arguments are given to make.
.PHONY: all
all: build

# Target for building the Docker image using Docker Compose
.PHONY: build
build:
	@echo "Building Docker image with DEVICE_TYPE=$(DEVICE_TYPE) and SOURCE_DOCUMENTS=$(SOURCE_DOCUMENTS)..."
	@DEVICE_TYPE=$(DEVICE_TYPE) SOURCE_DOCUMENTS=$(SOURCE_DOCUMENTS) docker-compose build

# Target for running the Docker container using Docker Compose
.PHONY: run
run:
	@echo "Running Docker container with DEVICE_TYPE=$(DEVICE_TYPE) and SOURCE_DOCUMENTS=$(SOURCE_DOCUMENTS)..."
	@DEVICE_TYPE=$(DEVICE_TYPE) SOURCE_DOCUMENTS=$(SOURCE_DOCUMENTS) docker-compose up 

# Target to stop the running container using Docker Compose
.PHONY: stop
stop:
	@echo "Stopping Docker container..."
	@docker-compose down

# Target for setting the device type to GPU (CUDA)
.PHONY: use-gpu
use-gpu:
	$(eval DEVICE_TYPE := cuda)

# Target for setting the device type to CPU
.PHONY: use-cpu
use-cpu:
	$(eval DEVICE_TYPE := cpu)

# Target to set the source documents directory
.PHONY: set-source-docs
set-source-docs:
	$(eval SOURCE_DOCUMENTS := $(dir))

# Help target describing how to use the Makefile
.PHONY: help
help:
	@echo "Makefile for Docker Compose operations"
	@echo "  make build                 - Builds the Docker image with the current DEVICE_TYPE and SOURCE_DOCUMENTS"
	@echo "  make run                   - Runs the Docker container with the current DEVICE_TYPE and SOURCE_DOCUMENTS"
	@echo "  make stop                  - Stops the running Docker container"
	@echo "  make use-gpu               - Sets the DEVICE_TYPE to cuda (GPU)"
	@echo "  make use-cpu               - Sets the DEVICE_TYPE to cpu"
	@echo "  make set-source-docs dir=<dir> - Sets the SOURCE_DOCUMENTS directory"
	@echo "  make help                  - Displays this help"

# Use this command to set the source documents directory
# Example usage: make set-source-docs dir=MyDocuments

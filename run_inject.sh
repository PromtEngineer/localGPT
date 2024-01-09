#!/bin/bash

# Check if enough arguments are passed
if [ $# -lt 3 ]; then
    echo "Usage: $0 <device_type> <exec_env> <action>"
    echo "device_type: cpu | mps"
    echo "exec_env: machine | container"
    echo "action: download | copy"
    exit 1
fi

# Name of the Docker container
CONTAINER_NAME="localgpt_localgpt_1"

# Set the device type (cpu or mps)
DEVICE_TYPE="$1"

# Execution environment: "container" or "machine"
EXEC_ENV="$2"

# Action: "download" or "copy"
ACTION="$3"


# PDF source - URL or local path
PDF_SOURCE="https://github.com/TatevKaren/free-resources-books-papers/blob/main/DataEngineering_HandBook.pdf" # URL of the PDF to download
LOCAL_PDF_PATH="$HOME/localGPT/SOURCE_DOCUMENTS/" # Local path of the PDF (if not downloading)

# Destination path for the PDF
DESTINATION_PATH="$HOME/SOURCE_DOCUMENTS/" # Directory to save the PDF

# Create destination directory if it doesn't exist
mkdir -p "$DESTINATION_PATH"

# Function to download PDF
download_pdf() {
    echo "Downloading PDF from $PDF_SOURCE..."
    curl -o "$DESTINATION_PATH/downloaded_document.pdf" "$PDF_SOURCE"
}

# Function to copy PDF
# Function to copy PDF files
copy_pdfs() {
    echo "Copying PDF files from $LOCAL_PDF_PATH to $DESTINATION_PATH..."
    for pdf_file in "$LOCAL_PDF_PATH"/*.pdf; do
        if [ -f "$pdf_file" ]; then
            pdf_filename=$(basename "$pdf_file")
            cp "$pdf_file" "$DESTINATION_PATH/$pdf_filename"
            echo "Copied: $pdf_filename"
        fi
    done
}

# Run python script with device type
run_ingest() {
    echo "Running ingest.py with device type $DEVICE_TYPE..."
    if [ "$EXEC_ENV" = "container" ]; then
        # Run in Docker container
        docker exec "$CONTAINER_NAME" python ingest.py --device_type "$DEVICE_TYPE"
    else
        # Run on local machine
        python ingest.py --device_type "$DEVICE_TYPE"
    fi
}

# Choose action based on the argument
case $ACTION in
    download)
        download_pdf
        ;;
    copy)
        copy_pdfs
        ;;
    *)
        echo "Invalid action: $ACTION"
        exit 2
        ;;
esac

run_ingest

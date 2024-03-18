import os
import shutil
import click
import subprocess

from constants import (
    DOCUMENT_MAP,
    SOURCE_DIRECTORY
)

def logToFile(logentry):
   file1 = open("crawl.log","a")
   file1.write(logentry + "\n")
   file1.close()
   print(logentry + "\n")

@click.command()
@click.option(
    "--device_type",
    default="cuda",
    type=click.Choice(
        [
            "cpu",
            "cuda",
            "ipu",
            "xpu",
            "mkldnn",
            "opengl",
            "opencl",
            "ideep",
            "hip",
            "ve",
            "fpga",
            "ort",
            "xla",
            "lazy",
            "vulkan",
            "mps",
            "meta",
            "hpu",
            "mtia",
        ],
    ),
    help="Device to run on. (Default is cuda)",
)
@click.option(
    "--landing_directory",
    default="./LANDING_DOCUMENTS"
)
@click.option(
    "--processed_directory",
    default="./PROCESSED_DOCUMENTS"
)
@click.option(
    "--error_directory",
    default="./ERROR_DOCUMENTS"
)
@click.option(
    "--unsupported_directory",
    default="./UNSUPPORTED_DOCUMENTS"
)

def main(device_type, landing_directory, processed_directory, error_directory, unsupported_directory):
    paths = []

    os.makedirs(processed_directory, exist_ok=True)
    os.makedirs(error_directory, exist_ok=True)
    os.makedirs(unsupported_directory, exist_ok=True)

    for root, _, files in os.walk(landing_directory):
        for file_name in files:
            file_extension = os.path.splitext(file_name)[1]
            short_filename = os.path.basename(file_name)

            if not os.path.isdir(root + "/" + file_name):
               if file_extension in DOCUMENT_MAP.keys():
                   shutil.move(root + "/" + file_name, SOURCE_DIRECTORY+ "/" + short_filename)
                   logToFile("START: " + root + "/" + short_filename)
                   process = subprocess.Popen("python ingest.py --device_type=" + device_type, shell=True, stdout=subprocess.PIPE)
                   process.wait()
                   if process.returncode > 0:
                       shutil.move(SOURCE_DIRECTORY + "/" + short_filename, error_directory + "/" + short_filename)
                       logToFile("ERROR: " + root + "/" + short_filename)
                   else:
                       logToFile("VALID: " + root + "/" + short_filename)
                       shutil.move(SOURCE_DIRECTORY + "/" + short_filename, processed_directory+ "/" + short_filename)
               else:
                   shutil.move(root + "/" + file_name, unsupported_directory+ "/" + short_filename)

if __name__ == "__main__":
    main()

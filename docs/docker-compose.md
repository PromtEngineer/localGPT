How to Use:
Set the Environment Variable:

Before running Docker Compose, you need to set the DEVICE_TYPE environment variable on your host machine. You can do this by running export DEVICE_TYPE=cuda for GPU or export DEVICE_TYPE=cpu for CPU.
Build and Run:

Use docker-compose up --build to build and start the container.
Stopping the Container:

Use docker-compose down to stop and remove the container.
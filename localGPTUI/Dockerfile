# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Install any needed packages specified in requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Make port 5111 available to the world outside this container
EXPOSE 5111

# Run localGPTUI.py when the container launches
CMD ["python", "localGPTUI.py", "--host=0.0.0.0", "--port=5111"]

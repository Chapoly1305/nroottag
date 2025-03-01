FROM nvidia/cuda:12.3.0-runtime-ubuntu22.04

# Set noninteractive installation
ENV DEBIAN_FRONTEND=noninteractive

# Install python and pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Create a symbolic link for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install Python packages
RUN pip3 install --no-cache-dir \
    requests \
    urllib3

# Set the working directory in the container
WORKDIR /app

# Copy the Python script into the container
COPY executables/main.py .
COPY executables/Seeker_CUDA_12 .

# Set the entry point to run the Python script
ENTRYPOINT ["python3", "main.py"]


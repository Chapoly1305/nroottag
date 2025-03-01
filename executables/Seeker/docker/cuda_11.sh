#!/bin/bash

# make sure we run from the root of the repository
[ "$(basename "$PWD")" == "cuda" ] && cd ../..
[ "$(basename "$PWD")" == "docker" ] && cd ..

# Find the CUDA version at
# https://hub.docker.com/r/nvidia/cuda/tags

CCAP="${CCAP:-8.0}"
CUDA="${CUDA:-11.8.0}"
#CUDA="${CUDA:-12.3.1}"

# Create executables directory if it doesn't exist
mkdir -p executables

# Use nvidia/cuda image to compile
# We are using old container (18.04) for best compatibility with old CUDA versions
# You can use latest if you want
docker run --rm -v "$(pwd):/app" -w /app nvidia/cuda:${CUDA}-devel-ubuntu18.04 \
    bash -c "make clean && make CUDA=/usr/local/cuda CXXCUDA=/usr/bin/g++ gpu=1 CCAP=${CCAP} all && \
             cp Seeker /app/executables/Seeker_CUDA_11 && chmod 777 /app/executables/Seeker_CUDA_11"

echo "Executable compiled and copied to ./executables/Seeker_CUDA"


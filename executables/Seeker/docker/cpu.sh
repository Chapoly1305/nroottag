#!/bin/bash

# make sure we run from the root of the repository
[ "$(basename "$PWD")" == "cpu" ] && cd ../..
[ "$(basename "$PWD")" == "docker" ] && cd ..

# Create executables directory if it doesn't exist
mkdir -p executables

# Use nvidia/cuda image to compile
docker run --rm -v "$(pwd):/app" -w /app gcc:10.1 \
    bash -c "make clean && make all && \
             cp Seeker /app/executables/Seeker_CPU"

echo "Executable compiled and copied to ./executables/Seeker"
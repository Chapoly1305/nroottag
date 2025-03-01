# Seeker
Brute-force prefix search for SECP224R1.

Seeker is a component of nRootTag, fork from previous VanitySearch projects, including [JeanLucPons](https://github.com/JeanLucPons/VanitySearch) and [bojlahg](https://github.com/bojlahg/VanitySearchPublicKey).
The Seeker alike the previous project, it can search for a private key that its public key matches user's given value (aka. prefix). It supports computing with pure CPU and utilizing CUDA GPU.

## Dependencies
- For CPU only, the dependencies are light. It has almost no dependency on the OS (we did not test on other than Linux).
- The CUDA version has a dependency on CUDA, thus, you are required to install adequate CUDA on your host. If you are using the docker, then you shall ensure your environment is CUDA compatible.

## Example Usage

### Pure CPU


``` bash
make all
```

Compile for CPU only, without GPU support. PS. executable with GPU support can also run with CPU only. The CPU search method is quite slow, especially when the prefix is long (more than 3 bytes), it's unlikely to find a match momently. In the following, we introduce the use of GPU to accelerate the process.


### CUDA GPU
**Compile.** We assume you have installed CUDA dependencies. We recommend building on/for the execution machine for the best performance. The compile command,
``` bash
make gpu=1 CCAP=89 CUDA=/usr/local/cuda-12.5 CXXCUDA=/usr/bin/g++ all
```
- the `gpu=1` enables GPU support.
- the `CCAP` shall match the GPU, you may find them from [NVIDIA](https://developer.nvidia.com/cuda-gpus). For RTX4090 uses 89, A100 uses 80, and H100 uses 90. If uncertain, you _may_  use 80 for all, however, the performance _may_ downgrade.
- the `CUDA` is the home path of the CUDA sdk.
- the `CXXCUDA` is the path to the standard g++ executable.

**Execution.**
``` bash
# Print found keys to stdout
./Seeker -t 0 -gpu -g 1024,128 -p deadbe

# Save found keys to file keys.out
./Seeker -t 0 -gpu -g 1024,128 -o keys.out -p deadbe
```

- We can disable searching with CPU by using `-t 0`, CPU will still be used when a match is found for validation.
- `-gpu` enables the GPU module.
- `-g` sets the gridSize in pairs. For example, the `1024,128` represents XX and YY. If you wish to use multiple GPUs concurrently, you need to set for each GPU. Assume you have two GPU, then `1024,128,2048,256` or `1024,128,2048,256`. gpu0x,gpu0y,gpu1x,gpu1y. We will discuss further in the next section.

### (GPU) Performance Tuning
The parameter `-g` affects the performance significantly. The value shall adjust based on the GPU used accordingly, getting greedy on this value may downgrade the performance. The value may be adjusted if searching multiple prefixes concurrently. Assuming searching a three-byte prefix, our experiments show the optimal selection is `SM*32,512`. The SM (Streaming Multiprocessor) of each GPU can be found at (e.g. RTX4090) specification []() or third-party website [techpowerup](https://www.techpowerup.com/gpu-specs/geforce-rtx-4090.c3889).


## Information For Developers
[Developer.md](Developer.md)



## Docker Deployment Options

### Using Pre-built Image

We provide a [public image](https://hub.docker.com/repository/docker/chiba765/nroottag-seeker/general) on Docker Hub for easy deployment to your cluster.

### Building Local Image

1. Compiled executable `Seeker_CUDA_12` with `./docker/cuda_12.sh`

2. Build the Docker image:

   ```bash
   docker build -f docker/Dockerfile . -t chiba765/nroottag-seeker:latest
   ```

3. Run the Docker image:
   ```bash
   docker run --gpus all -e CNC_SERVER_URL=http://localhost:7898  chiba765/nroottag-seeker:latest
   ```

### Docker Runtime Configuration

Remember, Set the CNC Server URL using (Change to yours):

  ```bash
  -e CNC_SERVER_URL=http://localhost:7898
  ```

### Change of Runner

The Seeker executable is executed by `executables/main.py`. So if you would like to change how it runs, update this script. 



## Common Issue

### GPUEngine: Kernel: no kernel image is available for execution on the device
Please modify the compile script and update the CCAP and CUDA version to match your device.

### nvidia-smi not found
 If the program crashes with "nvidia-smi not found" error, verify that you've properly enabled GPU access. Enable GPU access by adding `--gpus all` to your Docker run command



## License

Seeker inherits licensed from previous projects under GPLv3.

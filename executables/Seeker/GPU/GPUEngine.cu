/*
 * This file is part of the VanitySearch distribution (https://github.com/JeanLucPons/VanitySearch).
 * Copyright (c) 2019 Jean Luc PONS.
 * Copyright (c) 2025 Chapoly1305, William Flores
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#ifdef __JETBRAINS_IDE__
#define __host__
#define __device__
#define __shared__
#define __constant__
#define __global__

// This is slightly mental, but gets it to properly index device function calls like __popc and whatever.
#define __CUDACC__
#include <device_functions.h>

// These headers are all implicitly present when you compile CUDA with clang. Clion doesn't know that, so
// we include them explicitly to make the indexer happy. Doing this when you actually build is, obviously,
// a terrible idea :D
#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_intrinsics.h>
#include <__clang_cuda_math_forward_declares.h>
#include <__clang_cuda_complex_builtins.h>
#include <__clang_cuda_cmath.h>
#endif // __JETBRAINS_IDE__

#ifndef WIN64
#include <unistd.h>
#include <stdio.h>
#endif

#include "GPUEngine.h"
#include <cuda.h>
#include <cuda_runtime.h>

#include <stdint.h>
#include "../hash/sha256.h"
#include "../Timer.h"

#include "GPUGroup.h"
#include "GPUMath.h"
#include "GPUWildcard.h"
#include "GPUCompute.h"

// ---------------------------------------------------------------------------------------

__global__ void
comp_keys(uint32_t mode, prefix_t *prefix, uint32_t *lookup32, uint64_t *keys, uint32_t maxFound, uint32_t *found) {

  int xPtr = (blockIdx.x * blockDim.x) * 8;
  int yPtr = xPtr + 4 * blockDim.x;
  //  toHex("starting key", keys+xPtr);
  ComputeKeys(mode, keys + xPtr, keys + yPtr, prefix, lookup32, maxFound, found);
}

__global__ void
comp_keys_pattern(uint32_t mode, prefix_t *pattern, uint64_t *keys, uint32_t maxFound, uint32_t *found) {

  int xPtr = (blockIdx.x * blockDim.x) * 8;
  int yPtr = xPtr + 4 * blockDim.x;
  ComputeKeys(mode, keys + xPtr, keys + yPtr, NULL, (uint32_t *)pattern, maxFound, found);
}

#define FULLCHECK
#ifdef FULLCHECK

// ---------------------------------------------------------------------------------------

__global__ void chekc_mult(uint64_t *a, uint64_t *b, uint64_t *r) {
  _ModMult(r, a, b);
  r[4] = 0;
}

__global__ void check_mod_inv(uint64_t *a, uint64_t *r) {
  uint64_t p[NBBLOCK];
  p[0] = a[0];
  p[1] = a[1];
  p[2] = a[2];
  p[3] = a[3];
  p[4] = a[4];
  _ModInv(p);
  r[0] = p[0];
  r[1] = p[1];
  r[2] = p[2];
  r[3] = p[3];
  r[4] = p[4];
}

// ---------------------------------------------------------------------------------------

__global__ void get_endianness(uint32_t *endian) {

  uint32_t a = 0x01020304;
  uint8_t fb = *(uint8_t *)(&a);
  *endian = (fb == 0x04);
}

#endif // FULLCHECK

// ---------------------------------------------------------------------------------------

using namespace std;

std::string toHex(unsigned char *data, int length) {

  string ret;
  ret.append("0x");
  char tmp[3];
  for (int i = 0; i < length; i++) {
    sprintf(tmp, "%02x", (int)data[i]);
    ret.append(tmp);
  }
  return ret;
}

int _ConvertSMVer2Cores(int major, int minor) {

  // Defines for GPU Architecture types (using the SM version to determine
  // the # of cores per SM
  typedef struct {
    int SM; // 0xMm (hexidecimal notation), M = SM Major version,
    // and m = SM minor version
    int Cores;
  } sSMtoCores;

  sSMtoCores nGpuArchCoresPerSM[] = {
    {0x20, 32}, // Fermi Generation (SM 2.0) GF100 class
    {0x21, 48}, // Fermi Generation (SM 2.1) GF10x class
    {0x30, 192},
    {0x32, 192},
    {0x35, 192},
    {0x37, 192},
    {0x50, 128},
    {0x52, 128},
    {0x53, 128},
    {0x60, 64},
    {0x61, 128},
    {0x62, 128},
    {0x70, 64},
    {0x72, 64},
    {0x75, 64},
    {0x80, 64},
    {0x86, 128},
    {0x89, 128},
    {0x90, 128},
    {-1, -1}};

  int index = 0;

  while (nGpuArchCoresPerSM[index].SM != -1) {
    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchCoresPerSM[index].Cores;
    }

    index++;
  }

  return 0;
}

GPUEngine::GPUEngine(int nbThreadGroup, int nbThreadPerGroup, int gpuId, uint32_t maxFound, bool rekey) {

  // Initialise CUDA
  this->rekey = rekey;
  this->nbThreadPerGroup = nbThreadPerGroup;
  initialised = false;
  cudaError_t err;

  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

  if (error_id != cudaSuccess) {
    printf("GPUEngine: CudaGetDeviceCount %s %d\n", cudaGetErrorString(error_id), error_id);
    return;
  }

  // This function call returns 0 if there are no CUDA capable devices.
  if (deviceCount == 0) {
    printf("GPUEngine: There are no available device(s) that support CUDA\n");
    return;
  }

  err = cudaSetDevice(gpuId);
  if (err != cudaSuccess) {
    printf("GPUEngine: cudaSetDevice %s\n", cudaGetErrorString(err));
    return;
  }

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, gpuId);

  if (nbThreadGroup == -1)
    nbThreadGroup = deviceProp.multiProcessorCount * 32;

  this->nbThread = nbThreadGroup * nbThreadPerGroup;
  this->maxFound = maxFound;
  this->outputSize = (maxFound * ITEM_SIZE + 4);

  char tmp[512];
  sprintf(
    tmp,
    "GPU #%d %s (%dx%d cores) Grid(%dx%d)",
    gpuId,
    deviceProp.name,
    deviceProp.multiProcessorCount,
    _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
    nbThread / nbThreadPerGroup,
    nbThreadPerGroup);
  deviceName = std::string(tmp);

  // Prefer L1 (We do not use __shared__ at all)
  err = cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
  if (err != cudaSuccess) {
    printf("GPUEngine: cudaDeviceSetCacheConfig %s\n", cudaGetErrorString(err));
    return;
  }

  // Try different stack sizes, starting with a smaller value
  size_t stackSizes[] = {16384, 24576, 32768, 49152}; // 16KB, 24KB, 32KB, 48KB
  bool stackSet = false;
  size_t last_good_stack_size = 0;

  for (size_t stackSize : stackSizes) {
  err = cudaDeviceSetLimit(cudaLimitStackSize, stackSize);
    if (err == cudaSuccess) {
      stackSet = true;
      last_good_stack_size = stackSize;
    } else {
      printf("GPUEngine: Warning - Failed to set stack size to %zu KB\n", stackSize/1024);
      if (last_good_stack_size != 0) {
        err = cudaDeviceSetLimit(cudaLimitStackSize, last_good_stack_size);
        if (err == cudaSuccess) {
          printf("GPUEngine: Reverted to last good stack size %zu KB\n", last_good_stack_size/1024);
          break;
        } else {
          printf("GPUEngine: Error - Failed to revert to last good stack size\n");
          last_good_stack_size = 0;
          stackSet = false;
        }
      }
    }
  }

  if (!stackSet) {
    printf("GPUEngine: Warning - Failed to set custom stack size. Using default.\n");
  }

  /*
  size_t heapSize = ;
  err = cudaDeviceSetLimit(cudaLimitMallocHeapSize, heapSize);
  if (err != cudaSuccess) {
    printf("Error: %s\n", cudaGetErrorString(err));
    exit(0);
  }

  size_t size;
  cudaDeviceGetLimit(&size, cudaLimitStackSize);
  printf("Stack Size %lld\n", size);
  cudaDeviceGetLimit(&size, cudaLimitMallocHeapSize);
  printf("Heap Size %lld\n", size);
  */

  // Allocate memory
  err = cudaMalloc((void **)&inputPrefix, _64K * 2);
  if (err != cudaSuccess) {
    printf("GPUEngine: Allocate prefix memory: %s\n", cudaGetErrorString(err));
    return;
  }
  err = cudaHostAlloc(&inputPrefixPinned, _64K * 2, cudaHostAllocWriteCombined | cudaHostAllocMapped);
  if (err != cudaSuccess) {
    printf("GPUEngine: Allocate prefix pinned memory: %s\n", cudaGetErrorString(err));
    return;
  }
  err = cudaMalloc((void **)&inputKey, nbThread * 32 * 2);
  if (err != cudaSuccess) {
    printf("GPUEngine: Allocate input memory: %s\n", cudaGetErrorString(err));
    return;
  }
  err = cudaHostAlloc(&inputKeyPinned, nbThread * 32 * 2, cudaHostAllocWriteCombined | cudaHostAllocMapped);
  if (err != cudaSuccess) {
    printf("GPUEngine: Allocate input pinned memory: %s\n", cudaGetErrorString(err));
    return;
  }
  err = cudaMalloc((void **)&outputPrefix, outputSize);
  if (err != cudaSuccess) {
    printf("GPUEngine: Allocate output memory: %s\n", cudaGetErrorString(err));
    return;
  }
  err = cudaHostAlloc(&outputPrefixPinned, outputSize, cudaHostAllocMapped);
  if (err != cudaSuccess) {
    printf("GPUEngine: Allocate output pinned memory: %s\n", cudaGetErrorString(err));
    return;
  }

  searchMode = SEARCH_COMPRESSED;
  searchType = P2PKH;
  initialised = true;
  pattern = "";
  hasPattern = false;
  inputPrefixLookUp = NULL;
}

int GPUEngine::GetGroupSize() { return GRP_SIZE; }

void GPUEngine::PrintCudaInfo() {

  cudaError_t err;

  const char *sComputeMode[] = {
    "Multiple host threads", "Only one host thread", "No host thread", "Multiple process threads", "Unknown", NULL};

  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

  if (error_id != cudaSuccess) {
    printf("GPUEngine: CudaGetDeviceCount %s\n", cudaGetErrorString(error_id));
    return;
  }

  // This function call returns 0 if there are no CUDA capable devices.
  if (deviceCount == 0) {
    printf("GPUEngine: There are no available device(s) that support CUDA\n");
    return;
  }

  for (int i = 0; i < deviceCount; i++) {

    err = cudaSetDevice(i);
    if (err != cudaSuccess) {
      printf("GPUEngine: cudaSetDevice(%d) %s\n", i, cudaGetErrorString(err));
      return;
    }

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, i);
    printf(
      "GPU #%d %s (%dx%d cores) (Cap %d.%d) (%.1f MB) (%s)\n",
      i,
      deviceProp.name,
      deviceProp.multiProcessorCount,
      _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
      deviceProp.major,
      deviceProp.minor,
      (double)deviceProp.totalGlobalMem / 1048576.0, // MB
      sComputeMode[deviceProp.computeMode]);
  }
}

GPUEngine::~GPUEngine() {
    // Safely free pinned memory if it is still allocated
    if (inputPrefixPinned) {
        cudaFreeHost(inputPrefixPinned);
        inputPrefixPinned = nullptr;
    }
    if (inputKeyPinned) {
        cudaFreeHost(inputKeyPinned);
        inputKeyPinned = nullptr;
    }
    if (inputPrefixLookUpPinned) {
        cudaFreeHost(inputPrefixLookUpPinned);
        inputPrefixLookUpPinned = nullptr;
    }

    if (inputPrefixLookUp) {
        cudaFree(inputPrefixLookUp);
        inputPrefixLookUp = nullptr;
    }

    if (inputPrefix) {
        cudaFree(inputPrefix);
        inputPrefix = nullptr;
    }
    if (inputKey) {
        cudaFree(inputKey);
        inputKey = nullptr;
    }
    if (outputPrefixPinned) {
        cudaFreeHost(outputPrefixPinned);
        outputPrefixPinned = nullptr;
    }
    if (outputPrefix) {
        cudaFree(outputPrefix);
        outputPrefix = nullptr;
    }
}


int GPUEngine::GetNbThread() { return nbThread; }

void GPUEngine::SetSearchMode(int searchMode) { this->searchMode = searchMode; }

void GPUEngine::SetSearchType(int searchType) { this->searchType = searchType; }

void GPUEngine::SetPrefix(std::vector<prefix_t> prefixes) {

  memset(inputPrefixPinned, 0, _64K * 2);
  for (int i = 0; i < (int)prefixes.size(); i++)
    inputPrefixPinned[prefixes[i]] = 1;

  // Fill device memory
  cudaMemcpy(inputPrefix, inputPrefixPinned, _64K * 2, cudaMemcpyHostToDevice);

  // We do not need the input pinned memory anymore
  cudaFreeHost(inputPrefixPinned);
  inputPrefixPinned = NULL;
  lostWarning = false;

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("GPUEngine: SetPrefix: %s\n", cudaGetErrorString(err));
  }
}

void GPUEngine::SetPattern(const char *pattern) {

  strcpy((char *)inputPrefixPinned, pattern);

  // Fill device memory
  cudaMemcpy(inputPrefix, inputPrefixPinned, _64K * 2, cudaMemcpyHostToDevice);

  // We do not need the input pinned memory anymore
  cudaFreeHost(inputPrefixPinned);
  inputPrefixPinned = NULL;
  lostWarning = false;

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("GPUEngine: SetPattern: %s\n", cudaGetErrorString(err));
  }

  hasPattern = true;
}

void GPUEngine::SetPrefix(std::vector<LPREFIX> prefixes, uint32_t totalPrefix) {

  // Allocate memory for the second level of lookup tables
  cudaError_t err = cudaMalloc((void **)&inputPrefixLookUp, (_64K + totalPrefix) * 4);
  if (err != cudaSuccess) {
    printf("GPUEngine: Allocate prefix lookup memory: %s\n", cudaGetErrorString(err));
    return;
  }
  err =
    cudaHostAlloc(&inputPrefixLookUpPinned, (_64K + totalPrefix) * 4, cudaHostAllocWriteCombined | cudaHostAllocMapped);
  if (err != cudaSuccess) {
    printf("GPUEngine: Allocate prefix lookup pinned memory: %s\n", cudaGetErrorString(err));
    return;
  }

  uint32_t offset = _64K;
  memset(inputPrefixPinned, 0, _64K * 2);
  memset(inputPrefixLookUpPinned, 0, _64K * 4);
  for (int i = 0; i < (int)prefixes.size(); i++) {
    int nbLPrefix = (int)prefixes[i].lPrefixes.size();
    inputPrefixPinned[prefixes[i].sPrefix] = (uint16_t)nbLPrefix;
    inputPrefixLookUpPinned[prefixes[i].sPrefix] = offset;
    for (int j = 0; j < nbLPrefix; j++) {
      inputPrefixLookUpPinned[offset++] = prefixes[i].lPrefixes[j];
    }
  }

  if (offset != (_64K + totalPrefix)) {
    printf("GPUEngine: Wrong totalPrefix %d!=%d!\n", offset - _64K, totalPrefix);
    return;
  }

  // Fill device memory
  cudaMemcpy(inputPrefix, inputPrefixPinned, _64K * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(inputPrefixLookUp, inputPrefixLookUpPinned, (_64K + totalPrefix) * 4, cudaMemcpyHostToDevice);

  // We do not need the input pinned memory anymore
  cudaFreeHost(inputPrefixPinned);
  inputPrefixPinned = NULL;
  cudaFreeHost(inputPrefixLookUpPinned);
  inputPrefixLookUpPinned = NULL;
  lostWarning = false;

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("GPUEngine: SetPrefix (large): %s\n", cudaGetErrorString(err));
  }
}

bool GPUEngine::callKernel() {

  // Reset nbFound
  cudaMemset(outputPrefix, 0, 4);

  // Call the kernel (Perform STEP_SIZE keys per thread)
  if (hasPattern) {
    printf("GPUEngine: (TODO) unsupported pattern\n");
    return false;
    if (searchType == BECH32) {
      // TODO
      printf("GPUEngine: (TODO) BECH32 not yet supported with wildard\n");
      return false;
    }
    comp_keys_pattern<<<nbThread / nbThreadPerGroup, nbThreadPerGroup>>>(
      searchMode, inputPrefix, inputKey, maxFound, outputPrefix);
  } else {
    if (searchMode == SEARCH_COMPRESSED) {
      printf("GPUEngine: (TODO) unsupported search compressed\n");
      return false;
    } else {
      comp_keys<<<nbThread / nbThreadPerGroup, nbThreadPerGroup>>>(
        searchMode, inputPrefix, inputPrefixLookUp, inputKey, maxFound, outputPrefix);
    }
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("GPUEngine: Kernel: %s\n", cudaGetErrorString(err));
    return false;
  }
  return true;
}

bool GPUEngine::SetKeys(Point *p) {

  // Sets the starting keys for each thread
  // p must contains nbThread public keys
  for (int i = 0; i < nbThread; i += nbThreadPerGroup) {
    for (int j = 0; j < nbThreadPerGroup; j++) {

      inputKeyPinned[8 * i + j + 0 * nbThreadPerGroup] = p[i + j].x.bits64[0];
      inputKeyPinned[8 * i + j + 1 * nbThreadPerGroup] = p[i + j].x.bits64[1];
      inputKeyPinned[8 * i + j + 2 * nbThreadPerGroup] = p[i + j].x.bits64[2];
      inputKeyPinned[8 * i + j + 3 * nbThreadPerGroup] = p[i + j].x.bits64[3];

      inputKeyPinned[8 * i + j + 4 * nbThreadPerGroup] = p[i + j].y.bits64[0];
      inputKeyPinned[8 * i + j + 5 * nbThreadPerGroup] = p[i + j].y.bits64[1];
      inputKeyPinned[8 * i + j + 6 * nbThreadPerGroup] = p[i + j].y.bits64[2];
      inputKeyPinned[8 * i + j + 7 * nbThreadPerGroup] = p[i + j].y.bits64[3];
    }
  }

  // Fill device memory
  cudaMemcpy(inputKey, inputKeyPinned, nbThread * 32 * 2, cudaMemcpyHostToDevice);

  if (!rekey) {
    // We do not need the input pinned memory anymore
    cudaFreeHost(inputKeyPinned);
    inputKeyPinned = NULL;
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("GPUEngine: SetKeys: %s\n", cudaGetErrorString(err));
  }

  return callKernel();
}

bool GPUEngine::Launch(CHECK_PREFIXES *output, bool spinWait) {
  if (spinWait) {
    cudaMemcpy(outputPrefixPinned, outputPrefix, outputSize, cudaMemcpyDeviceToHost);
  } else {
    // Use cudaMemcpyAsync to avoid default spin wait of cudaMemcpy which takes 100% CPU
    cudaEvent_t evt;
    cudaEventCreate(&evt);
    cudaMemcpyAsync(outputPrefixPinned, outputPrefix, 4, cudaMemcpyDeviceToHost, 0);
    cudaEventRecord(evt, 0);
    while (cudaEventQuery(evt) == cudaErrorNotReady) {
      // Sleep 1 ms to free the CPU
      Timer::SleepMillis(1);
    }
    cudaEventDestroy(evt);
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("GPUEngine: Launch: %s\n", cudaGetErrorString(err));
    return false;
  }

  // Look for prefix found
  uint32_t nbFound = outputPrefixPinned[0];
  if (nbFound > maxFound) {
    // prefix has been lost
    if (!lostWarning) {
      printf(
        "\nWarning, %d items lost\nHint: Search with less prefixes, less threads (-g) or increase maxFound (-m)\n",
        (nbFound - maxFound));
      lostWarning = true;
    }
    nbFound = maxFound;
  }
  auto size = nbFound * ITEM_SIZE + 4;
  uint32_t *out = (uint32_t *)malloc(size);
  
  // Copy data
  cudaMemcpy(outputPrefixPinned, outputPrefix, size, cudaMemcpyDeviceToHost);
  memcpy(out, outputPrefixPinned, size);
  output->raw = out;
  output->size = nbFound;

  // The TSQueue's push will handle backpressure automatically
  return callKernel();
}

bool GPUEngine::CheckPoint(Point *pt, vector<ITEM> &found, int tid, int incr, int endo, int *nbOK) {

  bool ok = true;
  //  printf("%s\n", pt->x.GetBase16().c_str());
  // Search in found by GPU
  bool f = false;
  int l = 0;
  uint64_t *p64;
  Point temp;
  // printf("Search: %s\n", toHex(h,20).c_str());
  while (l < found.size() && !f) {
    p64 = (uint64_t *)found[l].hash;
    for (int x = 0; x < NB64BLOCK; x++) {
      temp.x.bits64[x] = p64[x];
    }
    f = temp.x.IsEqual(&(pt->x));
    if (!f)
      l++;
  }
  if (f) {
    found.erase(found.begin() + l);
    *nbOK = *nbOK + 1;
  } else {
    ok = false;
    printf("Expected item not found %s (thread=%d, incr=%d, endo=%d)\n", pt->x.GetBase16().c_str(), tid, incr, endo);
  }

  return ok;
}

void printfound(vector<ITEM> &found) {
  int l = 0;
  uint8_t *p64;
  Point temp;
  // printf("Search: %s\n", toHex(h,20).c_str());
  while (l < found.size()) {
    p64 = (uint8_t *)found[l].hash;
    for (int x = 0; x < NB08BLOCK; x++) {
      temp.x.bits08[x] = found[l].hash[x];
    }
    l++;
    printf("Found: %s\n", temp.x.GetBase16().c_str());
  }
}

bool GPUEngine::Check(Secp224R1 *secp) {

  uint8_t h[20];
  int i = 0;
  int j = 0;
  bool ok = true;

  if (!initialised)
    return false;

  printf("GPU: %s\n", deviceName.c_str());

#ifdef FULLCHECK

  // Get endianess
  get_endianness<<<1, 1>>>(outputPrefix);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("GPUEngine: get_endianness: %s\n", cudaGetErrorString(err));
    return false;
  }
  cudaMemcpy(outputPrefixPinned, outputPrefix, 1, cudaMemcpyDeviceToHost);
  littleEndian = *outputPrefixPinned != 0;
  printf("Endianness: %s\n", (littleEndian ? "Little" : "Big"));

  // Check modular mult
  Int a;
  Int b;
  Int r;
  Int c;
  a.SetBase16("49C5C08402A02494ED104ADBF426F0F43BE1C152F42160751CAA7E7E");
  b.Rand(224);
  c.ModMul(&a, &b);
  memcpy(inputKeyPinned, a.bits64, BIFULLSIZE);
  memcpy(inputKeyPinned + 5, b.bits64, BIFULLSIZE);
  cudaMemcpy(inputKey, inputKeyPinned, BIFULLSIZE * 2, cudaMemcpyHostToDevice);
  chekc_mult<<<1, 1>>>(inputKey, inputKey + 5, (uint64_t *)outputPrefix);
  cudaMemcpy(outputPrefixPinned, outputPrefix, BIFULLSIZE, cudaMemcpyDeviceToHost);
  memcpy(r.bits64, outputPrefixPinned, BIFULLSIZE);

  if (!c.IsEqual(&r)) {
    printf("r=%llu,%llu,%llu,%llu,%llu\n", r.bits64[0], r.bits64[1], r.bits64[2], r.bits64[3], r.bits64[4]);
    printf(
      "\nModular Mult wrong:\nA=0x%s\nB=0x%s\nR=0x%s\nC=0x%s\n",
      a.GetBase16().c_str(),
      b.GetBase16().c_str(),
      r.GetBase16().c_str(),
      c.GetBase16().c_str());
    return false;
  }

  a.ModInv();
  r.SetBase16("49C5C08402A02494ED104ADBF426F0F43BE1C152F42160751CAA7E7E");
  memcpy(inputKeyPinned, r.bits64, BIFULLSIZE);
  cudaMemcpy(inputKey, inputKeyPinned, BIFULLSIZE, cudaMemcpyHostToDevice);
  check_mod_inv<<<1, 1>>>(inputKey, (uint64_t *)outputPrefix);
  cudaMemcpy(outputPrefixPinned, outputPrefix, BIFULLSIZE, cudaMemcpyDeviceToHost);
  memcpy(r.bits64, outputPrefixPinned, BIFULLSIZE);
  printf("DONE\n");
  if (!a.IsEqual(&r)) {
    printf("r=%llu,%llu,%llu,%llu,%llu\n", r.bits64[0], r.bits64[1], r.bits64[2], r.bits64[3], r.bits64[4]);
    printf("\nModular Inv wrong:\nA=0x%s\nR=0x%s\n", a.GetBase16().c_str(), r.GetBase16().c_str());
    return false;
  }
  return false;

#endif // FULLCHECK

  Point *p = new Point[nbThread];
  Point *p2 = new Point[nbThread];
  Int k;

  // Check kernel
  int nbFoundCPU[1];
  int nbOK[1];
  vector<ITEM> found;
  bool searchComp;

  if (searchMode == SEARCH_BOTH) {
    printf("Warning, Check function does not support BOTH_MODE, use either compressed or uncompressed");
    return true;
  }

  searchComp = SEARCH_PUBLICKEYS;

  uint32_t seed = 1710463954;
  printf("Seed: %u\n", seed);
  rseed(seed);
  memset(nbOK, 0, sizeof(nbOK));
  memset(nbFoundCPU, 0, sizeof(nbFoundCPU));
  //  k.SetBase16("49C5C08402A02494ED104ADBF426F0F43BE1C152F42160751CAA7E7E");
  //  printf("K: %s\nPub: %s\n", k.GetBase16().c_str(),
  //         secp->ComputePublicKey(&k).toString().c_str());
  //  Point p0 = secp->ComputePublicKey(&k);
  //  p[0] = p0;
  //  p2[0] = p0;
  for (int i = 0; i < nbThread; i++) {
    k.Rand(256);
    p[i] = secp->ComputePublicKey(&k);
    // Group starts at the middle
    k.Add((uint64_t)GRP_SIZE / 2);
    p2[i] = secp->ComputePublicKey(&k);
    //    p0 = secp->NextKey(p0);
  }

  std::vector<prefix_t> prefs;
  prefs.push_back(0x6371);
  prefs.push_back(0x1234);
  SetPrefix(prefs);
  SetKeys(p2);
  double t0 = Timer::get_tick();
  //  Launch(found, true);
  double t1 = Timer::get_tick();
  Timer::printResult((char *)"Key", 6 * STEP_SIZE * nbThread, t0, t1);

  // for (int i = 0; i < found.size(); i++) {
  //   printf("[%d]: thId=%d incr=%d\n", i, found[i].thId,found[i].incr);
  //   printf("[%d]: %s\n", i,toHex(found[i].hash,20).c_str());
  // }

  printf("ComputeKeys() found %d items , CPU check...\n", (int)found.size());

  Int beta, beta2;
  beta.SetBase16((char *)"94353937171b8337606664f1900be8995691cf49934d5551cc5ace29");
  beta2.SetBase16((char *)"94353937171b8337606664f1900be8995691cf49934d5551cc5ace29");

  if (found.size() == 0) {
    ok = false;
    printf("Expected to find at least one item !\n");
  }

  delete[] p;
  return ok;
}

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

#ifndef GPUENGINEH
#define GPUENGINEH

#include <atomic>
#include <vector>
#include <string>
#include "../SECP224r1.h"

#define SEARCH_COMPRESSED 0
#define SEARCH_UNCOMPRESSED 1
#define SEARCH_BOTH 2
#define SEARCH_PUBLICKEYS 3

static const char *searchModes[] = {"Compressed", "Uncompressed", "Compressed or Uncompressed", "PublicKeys"};

// Number of key per thread (must be a multiple of GRP_SIZE) per kernel call
#define STEP_SIZE 1024

// Number of thread per block
#define ITEM_SIZE 48
#define ITEM_SIZE32 (ITEM_SIZE / 4)
#define _64K 65536

typedef uint16_t prefix_t;
typedef uint32_t prefixl_t;

typedef struct {
  uint32_t thId;
  int16_t incr;
  int16_t endo;
  uint8_t *hash;
  bool mode;
} ITEM;

// Second level lookup
typedef struct {
  prefix_t sPrefix;
  std::vector<prefixl_t> lPrefixes;
} LPREFIX;

typedef struct {
  //  std::vector<ITEM> found;
  uint32_t *raw;
  uint32_t size;
  std::vector<Int> keys;
} CHECK_PREFIXES;

struct GPUEngineCounters {
    std::atomic<uint64_t> totalProcessed{0};  // Total keys processed
    std::atomic<double> currentSpeed{0.0};    // Current keys/sec
    std::atomic<double> averageSpeed{0.0};    // Average keys/sec
};

class GPUEngine {

public:
  GPUEngine(int nbThreadGroup, int nbThreadPerGroup, int gpuId, uint32_t maxFound, bool rekey);
  ~GPUEngine();
  void SetPrefix(std::vector<prefix_t> prefixes);
  void SetPrefix(std::vector<LPREFIX> prefixes, uint32_t totalPrefix);
  bool SetKeys(Point *p);
  void SetSearchMode(int searchMode);
  void SetSearchType(int searchType);
  void SetPattern(const char *pattern);
  bool Launch(CHECK_PREFIXES *output, bool spinWait = false);
  int GetNbThread();
  int GetGroupSize();

  bool Check(Secp224R1 *secp);
  std::string deviceName;

  static void PrintCudaInfo();
  static void GenerateCode(Secp224R1 *secp, int size);

private:
  GPUEngineCounters counters;
  bool callKernel();
  static void ComputeIndex(std::vector<int> &s, int depth, int n);
  static void Browse(FILE *f, int depth, int max, int s);
  bool CheckHash(uint8_t *h, std::vector<ITEM> &found, int tid, int incr, int endo, int *ok);
  bool CheckPoint(Point *px, std::vector<ITEM> &found, int tid, int incr, int endo, int *ok);
  int nbThread;
  int nbThreadPerGroup;
  prefix_t *inputPrefix;
  prefix_t *inputPrefixPinned;
  uint32_t *inputPrefixLookUp;
  uint32_t *inputPrefixLookUpPinned;
  uint64_t *inputKey;
  uint64_t *inputKeyPinned;
  uint32_t *outputPrefix;
  uint32_t *outputPrefixPinned;
  bool initialised;
  uint32_t searchMode;
  uint32_t searchType;
  bool littleEndian;
  bool lostWarning;
  bool rekey;
  uint32_t maxFound;
  uint32_t outputSize;
  std::string pattern;
  bool hasPattern;
};

#endif // GPUENGINEH

/*
 * This file is part of the VanitySearch distribution (https://github.com/JeanLucPons/VanitySearch).
 * Copyright (c) 2019 Jean Luc PONS.
 * Copyright (c) 2025 Chapoly1305.
 * 
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

#ifndef VANITYH
#define VANITYH

#include <string>
#include <vector>
#include <mutex>
#include "SECP224r1.h"
#include "GPU/GPUEngine.h"
#include "tsqueue.h"
#include <thread>
#include "PrefixThreadPool.h"

#ifdef WIN64
#include <Windows.h>
#endif

#define CPU_GRP_SIZE 1024
class VanitySearch;

typedef struct {

  VanitySearch *obj;
  int threadId;
  bool isRunning;
  bool hasStarted;
  bool rekeyRequest;
  int gridSizeX;
  int gridSizeY;
  int gpuId;

} TH_PARAM;

typedef struct {

  VanitySearch *obj;

} CP_PARAM;

typedef struct {

  char *prefix;
  int prefixLength;
  prefix_t sPrefix;
  bool *found;

  // For dreamer ;)
  bool isFull;
  prefixl_t lPrefix;
  uint8_t hash160[20];

  Point pubkey;
  int pubkeylen;

} PREFIX_ITEM;

typedef struct {

  std::vector<PREFIX_ITEM> *items;
  bool found;

} PREFIX_TABLE_ITEM;

class PrefixThreadPool;

class VanitySearch {

public:
  VanitySearch(
    Secp224R1 *secp,
    std::vector<std::string> &prefix,
    std::string seed,
    int searchMode,
    bool useGpu,
    bool stop,
    std::string outputFile,
    bool useSSE,
    uint32_t maxFound,
    uint64_t rekey,
    bool caseSensitive,
    Point &startPubKey,
    bool paranoiacSeed);

  void Search(int nbThread, std::vector<int> gpuId, std::vector<int> gridSize);
  void FindKeyCPU(TH_PARAM *p);
  void FindKeyGPU(TH_PARAM *p);
  void CheckPrefixes(CP_PARAM *p);

  std::mutex mtx;
  std::vector<std::string> lineBuffer;

private:
  double lastFlushTime;
  static const double FLUSH_INTERVAL;
  bool passThrough;
  double gpuKPS;         // Track GPU keys per second
  double gpuAvgKPS;      // Track average GPU keys per second
  uint64_t lastGPUCount; // Last GPU key count
  double lastGPUTime;     // Last GPU time check
  friend class PrefixThreadPool;
  PrefixThreadPool *prefixPool;
  void getGPUStartingKeysThread(int start, int end, int thId, int groupSize, Int *keys, Point *p);
  std::string GetHex(std::vector<unsigned char> &buffer);
  void checkPubKey(int pi, Int &key, int32_t incr, int endomorphism, Point &pt);
  void checkAddr(int prefIdx, uint8_t *hash160, Int &key, int32_t incr, int endomorphism, bool mode);
  void checkAddrSSE(
    uint8_t *h1,
    uint8_t *h2,
    uint8_t *h3,
    uint8_t *h4,
    int32_t incr1,
    int32_t incr2,
    int32_t incr3,
    int32_t incr4,
    Int &key,
    int endomorphism,
    bool mode);
  void checkAddresses(bool compressed, Int key, int i, Point p1);
  void checkPublicKeys(const Int &key, int i, const Point &p1, const Point &p2, const Point &p3, const Point &p4);
  void checkAddressesSSE(bool compressed, Int key, int i, Point p1, Point p2, Point p3, Point p4);
  void output(std::string pPubKey, std::string pAddrHex);
  void writeToFile();
  bool isAlive(TH_PARAM *p);
  bool isSingularPrefix(std::string pref);
  bool hasStarted(TH_PARAM *p);
  void rekeyRequest(TH_PARAM *p);
  uint64_t getGPUCount();
  uint64_t getCPUCount();
  bool initPrefix(std::string &prefix, PREFIX_ITEM *it);
  void dumpPrefixes();
  void getCPUStartingKey(int thId, Int &key, Point &startP);
  void getGPUStartingKeys(int thId, int groupSize, int nbThread, Int *keys, Point *p);
  void enumCaseUnsentivePrefix(std::string s, std::vector<std::string> &list);
  bool prefixMatch(char *prefix, char *addr);
  size_t totalPrefixLength;

  Secp224R1 *secp;
  Int startKey;
  Point startPubKey;
  bool startPubKeySpecified;
  uint64_t counters[256];
  double startTime;
  int searchType;
  int searchMode;
  bool hasPattern;
  bool caseSensitive;
  bool useGpu;
  bool stopWhenFound;
  bool endOfSearch;
  int nbCPUThread;
  int nbGPUThread;
  int nbFoundKey;
  int nbFoundKeyLast;
  double accumulatedTime;
  double KPS;
  double avgKPS;
  uint64_t rekey;
  uint64_t lastRekey;
  uint32_t nbPrefix;
  std::string outputFile;
  bool useSSE;
  bool onlyFull;
  uint32_t maxFound;
  bool *patternFound;
  std::vector<PREFIX_TABLE_ITEM> prefixes;
  std::vector<prefix_t> usedPrefix;
  std::vector<LPREFIX> usedPrefixL;
  std::vector<std::string> &inputPrefixes;

  Int beta;
  Int lambda;
  Int beta2;
  Int lambda2;
  TSQueue<CHECK_PREFIXES> *queue;

#ifdef WIN64
  HANDLE ghMutex;
#else
  pthread_mutex_t ghMutex;
#endif
};

#endif // VANITYH

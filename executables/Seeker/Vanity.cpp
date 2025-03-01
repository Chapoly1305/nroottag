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
#include "GPU/GPUEngine.h"
#include "Vanity.h"
#include "hash/sha256.h"
#include "hash/sha512.h"
#include "IntGroup.h"
#include "Wildcard.h"
#include "Timer.h"
#include <string.h>
#include <math.h>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <atomic>

#ifndef WIN64

#include <pthread.h>

#endif

using namespace std;
std::atomic<size_t> totalAllocated{0};
std::atomic<size_t> currentAllocations{0};
Point Gn[CPU_GRP_SIZE / 2];
Point _2Gn;
const double VanitySearch::FLUSH_INTERVAL = 3.0;
// ----------------------------------------------------------------------------

VanitySearch::VanitySearch(
  Secp224R1 *secp,
  vector<std::string> &inputPrefixes,
  string seed,
  int searchMode,
  bool useGpu,
  bool stop,
  string outputFile,
  bool useSSE,
  uint32_t maxFound,
  uint64_t rekey,
  bool caseSensitive,
  Point &startPubKey,
  bool paranoiacSeed)
    : inputPrefixes(inputPrefixes) {
  lastFlushTime = Timer::get_tick();
  passThrough = inputPrefixes.empty();

  totalPrefixLength = 0;
  if (!passThrough) {
    for (const auto &prefix : inputPrefixes) {
      totalPrefixLength += prefix.length();
    }
    totalPrefixLength /= 2;
  }

  this->secp = secp;
  this->searchMode = searchMode;
  this->useGpu = useGpu;
  this->stopWhenFound = stop;
  this->outputFile = outputFile;
  this->useSSE = useSSE;
  this->nbGPUThread = 0;
  this->maxFound = maxFound;
  this->rekey = rekey;
  this->startPubKey = startPubKey;
  this->hasPattern = false;
  this->caseSensitive = caseSensitive;
  this->startPubKeySpecified = !startPubKey.isZero();
  if (searchMode == SEARCH_PUBLICKEYS) {
    this->searchType = P2PKH;
  } else {
    this->searchType = -1;
  }

  lastRekey = 0;
  prefixes.clear();

  size_t numThreads = std::max(1u, std::thread::hardware_concurrency() - nbGPUThread);
  prefixPool = new PrefixThreadPool(this, numThreads);

  if (inputPrefixes.size() > 0) {
    // Create a 65536 items lookup table
    PREFIX_TABLE_ITEM t;
    t.found = true;
    t.items = NULL;
    for (int i = 0; i < 65536; i++)
      prefixes.push_back(t);
    // Check is inputPrefixes contains wildcard character
    for (int i = 0; i < (int)inputPrefixes.size() && !hasPattern; i++) {
      hasPattern =
        ((inputPrefixes[i].find('*') != std::string::npos) || (inputPrefixes[i].find('?') != std::string::npos));
    }

    if (!hasPattern) {

      // No wildcard used, standard search
      // Insert prefixes
      bool loadingProgress = (inputPrefixes.size() > 1000);
      if (loadingProgress)
        printf("[Building lookup16   0.0%%]\r");

      nbPrefix = 0;
      onlyFull = true;
      for (int i = 0; i < (int)inputPrefixes.size(); i++) {

        PREFIX_ITEM it;
        std::vector<PREFIX_ITEM> itPrefixes;

        if (!caseSensitive) {

          // For caseunsensitive search, loop through all possible combination
          // and fill up lookup table
          vector<string> subList;
          enumCaseUnsentivePrefix(inputPrefixes[i], subList);

          bool *found = new bool;
          *found = false;

          for (int j = 0; j < (int)subList.size(); j++) {
            if (initPrefix(subList[j], &it)) {
              it.found = found;
              it.prefix = strdup(it.prefix); // We need to allocate here, subList will be destroyed
              itPrefixes.push_back(it);
            }
          }

        } else {

          if (initPrefix(inputPrefixes[i], &it)) {
            bool *found = new bool;
            *found = false;
            it.found = found;
            itPrefixes.push_back(it);
          }
        }

        if (itPrefixes.size() > 0) {

          // Add the item to all correspoding prefixes in the lookup table
          for (int j = 0; j < (int)itPrefixes.size(); j++) {

            prefix_t p = itPrefixes[j].sPrefix;

            if (prefixes[p].items == NULL) {
              prefixes[p].items = new vector<PREFIX_ITEM>();
              prefixes[p].found = false;
              usedPrefix.push_back(p);
            }
            (*prefixes[p].items).push_back(itPrefixes[j]);
          }

          onlyFull &= it.isFull;
          nbPrefix++;
        }

        if (loadingProgress && i % 1000 == 0)
          printf("[Building lookup16 %5.1f%%]\r", (((double)i) / (double)(inputPrefixes.size() - 1)) * 100.0);
      }

      if (loadingProgress)
        printf("\n");

      //     dumpPrefixes();

      if (!caseSensitive && searchType == BECH32) {
        printf("Error, case unsensitive search with BECH32 not allowed.\n");
        exit(1);
      }

      if (nbPrefix == 0) {
        printf("VanitySearch: nothing to search !\n");
        exit(1);
      }

      // Second level lookup
      uint32_t unique_sPrefix = 0;
      uint32_t minI = 0xFFFFFFFF;
      uint32_t maxI = 0;
      for (int i = 0; i < (int)prefixes.size(); i++) {
        if (prefixes[i].items) {
          LPREFIX lit;
          lit.sPrefix = i;
          if (prefixes[i].items) {
            for (int j = 0; j < (int)prefixes[i].items->size(); j++) {
              lit.lPrefixes.push_back((*prefixes[i].items)[j].lPrefix);
            }
          }
          sort(lit.lPrefixes.begin(), lit.lPrefixes.end());
          usedPrefixL.push_back(lit);
          if ((uint32_t)lit.lPrefixes.size() > maxI)
            maxI = (uint32_t)lit.lPrefixes.size();
          if ((uint32_t)lit.lPrefixes.size() < minI)
            minI = (uint32_t)lit.lPrefixes.size();
          unique_sPrefix++;
        }
        if (loadingProgress)
          printf("[Building lookup32 %.1f%%]\r", ((double)i * 100.0) / (double)prefixes.size());
      }

      if (loadingProgress)
        printf("\n");

      string seachInfo = string(searchModes[searchMode]) + (startPubKeySpecified ? ", with public key" : "");
      if (nbPrefix == 1) {
        if (!caseSensitive) {
          // Case unsensitive search
          printf(
            "Search: %s [%s, Case unsensitive] (Lookup size %d)\n",
            inputPrefixes[0].c_str(),
            seachInfo.c_str(),
            unique_sPrefix);
        } else {
          printf("Search: %s [%s]\n", inputPrefixes[0].c_str(), seachInfo.c_str());
        }
      } else {
        if (onlyFull) {
          printf(
            "Search: %d addresses (Lookup size %d,[%d,%d]) [%s]\n",
            nbPrefix,
            unique_sPrefix,
            minI,
            maxI,
            seachInfo.c_str());
        } else {
          printf("Search: %d prefixes (Lookup size %d) [%s]\n", nbPrefix, unique_sPrefix, seachInfo.c_str());
        }
      }

    } else {
      string searchInfo = string(searchModes[searchMode]) + (startPubKeySpecified ? ", with public key" : "");
      if (inputPrefixes.size() == 1) {
        printf("Search: %s [%s]\n", inputPrefixes[0].c_str(), searchInfo.c_str());
      } else {
        printf("Search: %d patterns [%s]\n", (int)inputPrefixes.size(), searchInfo.c_str());
      }

      patternFound = (bool *)malloc(inputPrefixes.size() * sizeof(bool));
      memset(patternFound, 0, inputPrefixes.size() * sizeof(bool));
    }
  } else {
    printf("No prefix specified - passthrough mode enabled\n");
    nbPrefix = 1; // Just to indicate we have something to search for
    onlyFull = false;
  }

  // Compute Generator table G[n] = (n+1)*G

  Point g = secp->G;
  Gn[0] = g;
  g = secp->DoubleDirect(g);
  Gn[1] = g;
  for (int i = 2; i < CPU_GRP_SIZE / 2; i++) {
    g = secp->AddDirect(g, secp->G);
    Gn[i] = g;
  }
  // _2Gn = CPU_GRP_SIZE*G
  _2Gn = secp->DoubleDirect(Gn[CPU_GRP_SIZE / 2 - 1]);

  // Constant for endomorphism (secp224r1 does not have any endomorphism)
  // if a is a nth primitive root of unity, a^-1 is also a nth primitive root.
  // beta^3 = 1 mod p implies also beta^2 = beta^-1 mop (by multiplying both side by beta^-1)
  // (beta^3 = 1 mod p),  beta2 = beta^-1 = beta^2
  // (lambda^3 = 1 mod n), lamba2 = lamba^-1 = lamba^2
  // beta.SetBase16("7ae96a2b657c07106e64479eac3434e99cf0497512f58995c1396c28719501ee");
  // lambda.SetBase16("5363ad4cc05c30e0a5261c028812645a122e22ea20816678df02967c1b23bd72");
  // beta2.SetBase16("851695d49a83f8ef919bb86153cbcb16630fb68aed0a766a3ec693d68e6afa40");
  // lambda2.SetBase16("ac9c52b33fa3cf1f5ad9e3fd77ed9ba4a880b9fc8ec739c2e0cfc810b51283ce");

  // Seed
  if (seed.length() == 0) {
    // Default seed
    seed = Timer::getSeed(32);
  }

  if (paranoiacSeed) {
    seed += Timer::getSeed(32);
  }
  startKey.SetInt32(0);
  // Create a base with pbkdf2_hmac_sha512 then sha256
  // Only need to run once unless rekey is enabled.
  string salt = "VanitySearch";
  unsigned char hseed[64];
  pbkdf2_hmac_sha512(
    hseed, 64, (const uint8_t *)seed.c_str(), seed.length(), (const uint8_t *)salt.c_str(), salt.length(), 2048);

  sha256(hseed, 64, (unsigned char *)startKey.bits64);
  startKey.ShiftR(8 * 4);
  //    startKey.Rand(224);

  char *ctimeBuff;
  time_t now = time(NULL);
  ctimeBuff = ctime(&now);
  printf("Start %s", ctimeBuff);

  if (rekey > 0) {
    printf("Base Key: Randomly changed every %.0f Mkeys\n", (double)rekey);
  } else {
    printf("Base Key: %s\n", startKey.GetBase16().c_str());
  }

  this->queue = new TSQueue<CHECK_PREFIXES>();
}

// ----------------------------------------------------------------------------

bool VanitySearch::isSingularPrefix(std::string pref) {

  // check is the given prefix contains only 1
  bool only1 = true;
  int i = 0;
  while (only1 && i < (int)pref.length()) {
    only1 = pref.data()[i] == '1';
    i++;
  }
  return only1;
}

// ----------------------------------------------------------------------------
bool VanitySearch::initPrefix(std::string &prefix, PREFIX_ITEM *it) {

  std::vector<unsigned char> result;
  string dummy1 = prefix;
  int nbDigit = 0;
  bool wrong = false;

  if (prefix.length() < 2) {
    printf("Ignoring prefix \"%s\" (too short)\n", prefix.c_str());
    return false;
  }

  int aType = -1;

  if (searchMode == SEARCH_PUBLICKEYS) {
    std::transform(prefix.begin(), prefix.end(), prefix.begin(), ::toupper);

    if (prefix.size() > 112) {
      printf("Ignoring prefix \"%s\" 112 characters max\n", prefix.c_str());
      return false;
    }

    if (prefix.size() % 2 != 0) {
      printf("Ignoring prefix \"%s\" length of prefix should be even (full bytes)\n", prefix.c_str());
      return false;
    }

    Point pt;
    pt.Clear();
    if (prefix.size() <= 56) {
      pt.x.SetBase16((char *)prefix.c_str());
      pt.x.ShiftL((56 - prefix.size()) * 4);
    } else {
      pt.x.SetBase16((char *)prefix.substr(0, 56).c_str());
      pt.y.SetBase16((char *)prefix.substr(56).c_str());
      pt.y.ShiftL((56 - (prefix.size() - 56)) * 4);
    }

    it->sPrefix = *(prefix_t *)&pt.x.bits16[NB16BLOCK - 7];
    it->isFull = false;
    it->lPrefix = *(prefixl_t *)&pt.x.bits32[NB32BLOCK - 4];
    it->prefix = (char *)prefix.c_str();
    it->prefixLength = (int)prefix.length();

    it->pubkeylen = prefix.size() / 2;
    it->pubkey = pt;

    return true;
  } else {
    printf("Not supported\n");
    throw std::string("Not supported");
  }
}

// ----------------------------------------------------------------------------

void VanitySearch::dumpPrefixes() {

  for (int i = 0; i < 0xFFFF; i++) {
    if (prefixes[i].items) {
      printf("%04X\n", i);
      for (int j = 0; j < (int)prefixes[i].items->size(); j++) {
        printf("  %d\n", (*prefixes[i].items)[j].sPrefix);
        printf("  %s\n", (*prefixes[i].items)[j].prefix);
      }
    }
  }
}
// ----------------------------------------------------------------------------

void VanitySearch::enumCaseUnsentivePrefix(std::string s, std::vector<std::string> &list) {

  char letter[64];
  int letterpos[64];
  int nbLetter = 0;
  int length = (int)s.length();

  for (int i = 1; i < length; i++) {
    char c = s.data()[i];
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) {
      letter[nbLetter] = tolower(c);
      letterpos[nbLetter] = i;
      nbLetter++;
    }
  }

  int total = 1 << nbLetter;

  for (int i = 0; i < total; i++) {

    char tmp[64];
    strcpy(tmp, s.c_str());

    for (int j = 0; j < nbLetter; j++) {
      int mask = 1 << j;
      if (mask & i)
        tmp[letterpos[j]] = toupper(letter[j]);
      else
        tmp[letterpos[j]] = letter[j];
    }

    list.push_back(string(tmp));
  }
}

// ----------------------------------------------------------------------------

double log1(double x) {
  // Use taylor series to approximate log(1-x)
  return -x - (x * x) / 2.0 - (x * x * x) / 3.0 - (x * x * x * x) / 4.0;
}

// ----------------------------------------------------------------------------
void VanitySearch::output(string pPubKey, string pAddrHex) {
  std::unique_lock<std::mutex> lock(mtx);

  // Always pad the private key to 56 characters
  std::stringstream ss;
  ss << std::setw(56) << std::setfill('0') << std::right << pAddrHex;
  string paddedAddrHex = ss.str();

  if (!outputFile.empty()) {
    lineBuffer.emplace_back(pPubKey + ":" + paddedAddrHex);

    // Check if enough time has elapsed since last flush
    double currentTime = Timer::get_tick();
    if (currentTime - lastFlushTime >= FLUSH_INTERVAL) {
      writeToFile();
      lastFlushTime = currentTime;
    }
  } else {
    printf("\nPubKey: %s\n", pPubKey.c_str());
    printf("Priv (HEX): %s\n", paddedAddrHex.c_str());
  }
}

void VanitySearch::writeToFile() {
  if (lineBuffer.empty()) {
    return; // Nothing to write
  }

  FILE *f = fopen(outputFile.c_str(), "a");
  if (f) {
    for (const auto &line : lineBuffer) {
      fprintf(f, "%s\n", line.c_str());
    }
    fclose(f);
    lineBuffer.clear();
  } else {
    printf("Cannot open %s for writing\n", outputFile.c_str());
  }
}
// ----------------------------------------------------------------------------

bool pubKeyCompare(const Point &pt1, const Point &pt2, int len) {
  int idx = 0;
  while (idx < len) {
    if (pt1.x.bits08[27 - idx] != pt2.x.bits08[27 - idx]) {
      return false;
    }
    ++idx;
  }
  len -= 32;
  while (idx < len) {
    if (pt1.y.bits08[27 - idx] != pt2.y.bits08[27 - idx]) {
      return false;
    }
    ++idx;
  }
  return true;
}

void VanitySearch::checkPubKey(int pi, Int &key, int32_t incr, int endomorphism, Point &pt) {
  if (passThrough || (prefixes[pi].items->size() == 1 && (*prefixes[pi].items)[0].prefixLength <= 2)) {
    // Skip verification - trust GPU result
    Int k(&key);
    if (incr < 0) {
      k.Add((uint64_t)(-incr));
      k.Neg();
      k.Add(&secp->order);
    } else {
      k.Add((uint64_t)incr);
    }
    nbFoundKey++;
    output(pt.toString(), k.GetBase16());
    return;
  }

  for (int i = 0; i < prefixes[pi].items->size(); ++i) {
    PREFIX_ITEM *preitm = &(*prefixes[pi].items)[i];
    if (pubKeyCompare(pt, preitm->pubkey, preitm->pubkeylen)) {
      Int k(&key);

      Point sp = startPubKey;

      if (incr < 0) {
        k.Add((uint64_t)(-incr));
        k.Neg();
        k.Add(&secp->order);
        if (startPubKeySpecified)
          sp.y.ModNeg();
      } else {
        k.Add((uint64_t)incr);
      }

      // Check addresses
      Point p = secp->ComputePublicKey(&k);
      if (startPubKeySpecified)
        p = secp->AddDirect(p, sp);
      if (p.x.GetBase16() == pt.x.GetBase16()) {
        nbFoundKey++;
        output(p.toString(), k.GetBase16());
        return;
      } else {
        printf("Warn: Find 1 public key mismatch\n");
      }
    }
  }
}

// ----------------------------------------------------------------------------

#ifdef WIN64
DWORD WINAPI _FindKey(LPVOID lpParam) {
#else

void *_FindKey(void *lpParam) {
#endif
  TH_PARAM *p = (TH_PARAM *)lpParam;
  p->obj->FindKeyCPU(p);
  return 0;
}

#ifdef WIN64
DWORD WINAPI _FindKeyGPU(LPVOID lpParam) {
#else

void *_FindKeyGPU(void *lpParam) {
#endif
  TH_PARAM *p = (TH_PARAM *)lpParam;
  p->obj->FindKeyGPU(p);
  return 0;
}

#ifdef WIN64
DWORD WINAPI _FindKeyGPU(LPVOID lpParam) {
#else

void *_CheckPrefixes(void *lpParam) {
#endif
  CP_PARAM *p = (CP_PARAM *)lpParam;
  p->obj->CheckPrefixes(p);
  return 0;
}

// ----------------------------------------------------------------------------

void VanitySearch::checkPublicKeys(
  const Int &key,
  int i,
  const Point &p1,
  const Point &p2,
  const Point &p3,
  const Point &p4) {

  // Regular prefix-based checking for non-pass-through mode
  prefix_t pr0 = *(prefix_t *)&p1.x.bits16[NB16BLOCK - 7];
  prefix_t pr1 = *(prefix_t *)&p2.x.bits16[NB16BLOCK - 7];
  prefix_t pr2 = *(prefix_t *)&p3.x.bits16[NB16BLOCK - 7];
  prefix_t pr3 = *(prefix_t *)&p4.x.bits16[NB16BLOCK - 7];

  if (prefixes[pr0].items)
    checkPubKey(pr0, const_cast<Int &>(key), i, 0, const_cast<Point &>(p1));
  if (prefixes[pr1].items)
    checkPubKey(pr1, const_cast<Int &>(key), i + 1, 0, const_cast<Point &>(p2));
  if (prefixes[pr2].items)
    checkPubKey(pr2, const_cast<Int &>(key), i + 2, 0, const_cast<Point &>(p3));
  if (prefixes[pr3].items)
    checkPubKey(pr3, const_cast<Int &>(key), i + 3, 0, const_cast<Point &>(p4));
}

// ----------------------------------------------------------------------------
void VanitySearch::getCPUStartingKey(int thId, Int &key, Point &startP) {

  if (rekey > 0) {
    key.Rand(224);
  } else {
    Int one((uint64_t)1);
    key.Set(&one);
    key.Set(&startKey);
    Int off((int64_t)thId);
    off.ShiftL(56);
    key.Add(&off);
  }
  Int km(&key);
  km.Add((uint64_t)CPU_GRP_SIZE / 2);
  startP = secp->ComputePublicKey(&km);
  if (startPubKeySpecified)
    startP = secp->AddDirect(startP, startPubKey);
}

void VanitySearch::FindKeyCPU(TH_PARAM *ph) {

  // Initialize thread-specific variables
  int thId = ph->threadId;
  counters[thId] = 0;

  // Create a group of integers for batch operations
  // This improves performance by allowing multiple operations in parallel
  IntGroup *grp = new IntGroup(CPU_GRP_SIZE / 2 + 1);

  // Initialize the starting key and point for this thread
  Int key;
  Point startP;
  getCPUStartingKey(thId, key, startP);

  // Prepare arrays for storing intermediate results
  Int dx[CPU_GRP_SIZE / 2 + 1];
  Point pts[CPU_GRP_SIZE];

  // Variables for elliptic curve computations
  Int dy;
  Int dyn;
  Int _s;
  Int _p;
  Point pp;
  Point pn;

  // Set up the group for batch inverse calculations
  grp->Set(dx);

  // Signal that the thread has started
  ph->hasStarted = true;
  ph->rekeyRequest = false;

  while (!endOfSearch) {

    // Check if a new key is requested (for periodic key changes)
    if (ph->rekeyRequest) {
      getCPUStartingKey(thId, key, startP);
      ph->rekeyRequest = false;
    }

    // Fill the group with x-coordinates differences
    // This is done to prepare for a batch inverse operation
    int i;
    int hLength = (CPU_GRP_SIZE / 2 - 1);

    for (i = 0; i < hLength; i++) {
      dx[i].ModSub(&Gn[i].x, &startP.x);
    }
    dx[i].ModSub(&Gn[i].x, &startP.x);    // For the first point
    dx[i + 1].ModSub(&_2Gn.x, &startP.x); // For the next center point

    // Perform a batch modular inverse operation
    // This is more efficient than individual inverses
    grp->ModInv();

    // We use the fact that P + i*G and P - i*G have the same deltax, so the same inverse
    // We compute keys in both positive and negative directions from the center of the group

    // Set the center point of the group
    pts[CPU_GRP_SIZE / 2] = startP;

    for (i = 0; i < hLength && !endOfSearch; i++) {

      pp = startP;
      pn = startP;

      // Compute P = startP + i*G
      dy.ModSub(&Gn[i].y, &pp.y);

      _s.ModMul(&dy, &dx[i]); // s = (p2.y-p1.y)*inverse(p2.x-p1.x);
      _p.ModSquare(&_s);      // _p = s^2

      pp.x.ModNeg();
      pp.x.ModAdd(&_p);
      pp.x.ModSub(&Gn[i].x); // rx = s^2 - p1.x - p2.x;

      pp.y.ModSub(&Gn[i].x, &pp.x);
      pp.y.ModMul(&_s);
      pp.y.ModSub(&Gn[i].y); // ry = - p2.y - s*(ret.x-p2.x);

      // Compute P = startP - i*G  , if (x,y) = i*G then (x,-y) = -i*G
      dyn.Set(&Gn[i].y);
      dyn.ModNeg();
      dyn.ModSub(&pn.y);

      _s.ModMul(&dyn, &dx[i]); // s = (p2.y-p1.y)*inverse(p2.x-p1.x);
      _p.ModSquare(&_s);       // _p = s^2

      pn.x.ModNeg();
      pn.x.ModAdd(&_p);
      pn.x.ModSub(&Gn[i].x); // rx = s^2 - p1.x - p2.x;

      pn.y.ModSub(&Gn[i].x, &pn.x);
      pn.y.ModMul(&_s);
      pn.y.ModAdd(&Gn[i].y); // ry = - p2.y - s*(ret.x-p2.x);

      // Store the computed points
      pts[CPU_GRP_SIZE / 2 + (i + 1)] = pp;
      pts[CPU_GRP_SIZE / 2 - (i + 1)] = pn;
    }

    // Compute the first point (startP - (GRP_SZIE/2)*G)
    pn = startP;
    dyn.Set(&Gn[i].y);
    dyn.ModNeg();
    dyn.ModSub(&pn.y);

    _s.ModMul(&dyn, &dx[i]);
    _p.ModSquare(&_s);

    pn.x.ModNeg();
    pn.x.ModAdd(&_p);
    pn.x.ModSub(&Gn[i].x);

    pn.y.ModSub(&Gn[i].x, &pn.x);
    pn.y.ModMul(&_s);
    pn.y.ModAdd(&Gn[i].y);

    pts[0] = pn;

    // Compute the next start point (startP + GRP_SIZE*G)
    pp = startP;
    dy.ModSub(&_2Gn.y, &pp.y);

    _s.ModMul(&dy, &dx[i + 1]);
    _p.ModSquare(&_s);

    pp.x.ModNeg();
    pp.x.ModAdd(&_p);
    pp.x.ModSub(&_2Gn.x);

    pp.y.ModSub(&_2Gn.x, &pp.x);
    pp.y.ModMul(&_s);
    pp.y.ModSub(&_2Gn.y);
    startP = pp;

    // Check the generated public keys against the target prefixes
    for (int i = 0; i < CPU_GRP_SIZE && !endOfSearch; i += 4) {
      checkPublicKeys(key, i, pts[i], pts[i + 1], pts[i + 2], pts[i + 3]);
    }

    // Move to the next batch of keys
    key.Add((uint64_t)CPU_GRP_SIZE);
    counters[thId] += CPU_GRP_SIZE;
  }

  // Signal that the thread has finished
  ph->isRunning = false;
}
// ----------------------------------------------------------------------------

void VanitySearch::getGPUStartingKeysThread(int start, int end, int thId, int groupSize, Int *keys, Point *p) {
  for (int i = start; i < end; i++) {
    if (rekey > 0) {
      keys[i].Rand(224);
    } else {
      keys[i].Set(&startKey);
      Int offT((uint64_t)i);
      offT.ShiftL(80);
      Int offG((uint64_t)thId);
      offG.ShiftL(112);
      keys[i].Add(&offT);
      keys[i].Add(&offG);
    }
    Int k(keys + i);
    // Starting key is at the middle of the group
    k.Add((uint64_t)(groupSize / 2));
    p[i] = secp->ComputePublicKey(&k);
    if (startPubKeySpecified)
      p[i] = secp->AddDirect(p[i], startPubKey);
  }
}

void VanitySearch::getGPUStartingKeys(int thId, int groupSize, int nbThread, Int *keys, Point *p) {
  // Get number of CPU cores
  int numCores = std::thread::hardware_concurrency();
  std::vector<std::thread> threads;

  // Ensure we use at least one core
  numCores = std::max(1, numCores);

  // Split work among cores
  int keysPerThread = nbThread / numCores;
  int remainder = nbThread % numCores;

  // Launch threads
  int start = 0;
  for (int i = 0; i < numCores; i++) {
    int count = keysPerThread + (i < remainder ? 1 : 0);
    if (count > 0) {
      threads.emplace_back(
        &VanitySearch::getGPUStartingKeysThread, this, start, start + count, thId, groupSize, keys, p);
      start += count;
    }
  }

  // Wait for all threads to complete
  for (auto &t : threads) {
    t.join();
  }
}

void VanitySearch::FindKeyGPU(TH_PARAM *ph) {

  bool ok = true;

#ifdef WITHGPU

  // Global init
  int thId = ph->threadId;
  GPUEngine g(ph->gridSizeX, ph->gridSizeY, ph->gpuId, maxFound, (rekey != 0));
  int nbThread = g.GetNbThread();
  Point *p = new Point[nbThread];
  Int *keys = new Int[nbThread];
  vector<ITEM> found;

  printf("GPU: %s\n", g.deviceName.c_str());

  counters[thId] = 0;

  getGPUStartingKeys(thId, g.GetGroupSize(), nbThread, keys, p);

  g.SetSearchMode(searchMode);
  g.SetSearchType(searchType);
  if (onlyFull) {
    g.SetPrefix(usedPrefixL, nbPrefix);
  } else {
    if (hasPattern)
      g.SetPattern(inputPrefixes[0].c_str());
    else
      g.SetPrefix(usedPrefix);
  }

  getGPUStartingKeys(thId, g.GetGroupSize(), nbThread, keys, p);
  ok = g.SetKeys(p);
  ph->rekeyRequest = false;

  ph->hasStarted = true;
  CHECK_PREFIXES r = CHECK_PREFIXES{};

  // GPU Thread
  while (ok && !endOfSearch) {

    if (ph->rekeyRequest) {
      getGPUStartingKeys(thId, g.GetGroupSize(), nbThread, keys, p);
      ok = g.SetKeys(p);
      ph->rekeyRequest = false;
    }
    // Call kernel
    ok = g.Launch(&r);

    //    auto vec = vector<ITEM>(found);
    auto keyCopy = vector<Int>(nbThread);
    std::copy(keys, keys + nbThread, keyCopy.begin());
    queue->push(CHECK_PREFIXES{r.raw, r.size, std::move(keyCopy)});

    if (ok) {
      for (int i = 0; i < nbThread; i++) {
        keys[i].Add((uint64_t)STEP_SIZE);
      }
      counters[thId] += 1ULL * STEP_SIZE * nbThread; // Point +  endo1 + endo2 + symetrics
    }
  }

  delete[] keys;
  delete[] p;

#else
  ph->hasStarted = true;
  printf("GPU code not compiled, use -DWITHGPU when compiling.\n");
#endif

  ph->isRunning = false;
}

void VanitySearch::CheckPrefixes(CP_PARAM *p) {
  const size_t numThreads = prefixPool->getNumThreads();

  CHECK_PREFIXES result;
  while (!queue->done()) {
    auto valid = queue->pop(result); // Will block until queue has items
    if (!valid) {
      break;
    }

    // RAII guard for memory cleanup
    struct RawMemGuard {
      uint32_t *raw;
      size_t alloc_size;

      RawMemGuard(uint32_t *p, uint32_t count) : raw(p), alloc_size(count * ITEM_SIZE + 4) {}

      ~RawMemGuard() {
        if (raw) {
          free(raw);
          raw = nullptr;
        }
      }

      // Prevent copying
      RawMemGuard(const RawMemGuard &) = delete;
      RawMemGuard &operator=(const RawMemGuard &) = delete;
    } guard(result.raw, result.size);

    // Process batch
    uint32_t itemsPerThread = (result.size + numThreads - 1) / numThreads;

    // Submit work items directly without intermed1iate vector
    for (size_t i = 0; i < numThreads; i++) {
      uint32_t startIdx = i * itemsPerThread;
      if (startIdx >= result.size)
        break;

      uint32_t endIdx = std::min(startIdx + itemsPerThread, result.size);

      CheckPrefixesWorkItem work{startIdx, endIdx, &result, &result.keys};

      prefixPool->addTask(std::move(work));
    }

    prefixPool->waitForCompletion();
  }
}

// ----------------------------------------------------------------------------

bool VanitySearch::isAlive(TH_PARAM *p) {

  bool isAlive = true;
  int total = nbCPUThread + nbGPUThread;
  for (int i = 0; i < total; i++)
    isAlive = isAlive && p[i].isRunning;

  return isAlive;
}

// ----------------------------------------------------------------------------

bool VanitySearch::hasStarted(TH_PARAM *p) {

  bool hasStarted = true;
  int total = nbCPUThread + nbGPUThread;
  for (int i = 0; i < total; i++)
    hasStarted = hasStarted && p[i].hasStarted;

  return hasStarted;
}

// ----------------------------------------------------------------------------

void VanitySearch::rekeyRequest(TH_PARAM *p) {

  bool hasStarted = true;
  int total = nbCPUThread + nbGPUThread;
  for (int i = 0; i < total; i++)
    p[i].rekeyRequest = true;
}

// ----------------------------------------------------------------------------

uint64_t VanitySearch::getGPUCount() {

  uint64_t count = 0;
  for (int i = 0; i < nbGPUThread; i++)
    count += counters[0x80L + i];
  return count;
}

uint64_t VanitySearch::getCPUCount() {

  uint64_t count = 0;
  for (int i = 0; i < nbCPUThread; i++)
    count += counters[i];
  return count;
}

// ----------------------------------------------------------------------------

void VanitySearch::Search(int nbThread, std::vector<int> gpuId, std::vector<int> gridSize) {

  double t0;
  double t1;
  endOfSearch = false;
  nbCPUThread = nbThread;
  nbGPUThread = (useGpu ? (int)gpuId.size() : 0);
  nbFoundKey = 0;
  nbFoundKeyLast = 0;
  gpuKPS = 0;
  gpuAvgKPS = 0;
  lastGPUCount = 0;
  accumulatedTime = 0;
  memset(counters, 0, sizeof(counters));

  printf("Number of CPU thread: %d\n", nbCPUThread);

  TH_PARAM *params = (TH_PARAM *)malloc((nbCPUThread + nbGPUThread) * sizeof(TH_PARAM));
  memset(params, 0, (nbCPUThread + nbGPUThread) * sizeof(TH_PARAM));

  // Launch CPU threads
  for (int i = 0; i < nbCPUThread; i++) {
    params[i].obj = this;
    params[i].threadId = i;
    params[i].isRunning = true;

#ifdef WIN64
    DWORD thread_id;
    CreateThread(NULL, 0, _FindKey, (void *)(params + i), 0, &thread_id);
    ghMutex = CreateMutex(NULL, FALSE, NULL);
#else
    pthread_t thread_id;
    pthread_create(&thread_id, NULL, &_FindKey, (void *)(params + i));
    ghMutex = PTHREAD_MUTEX_INITIALIZER;
#endif
  }

  // Launch GPU threads
  for (int i = 0; i < nbGPUThread; i++) {
    params[nbCPUThread + i].obj = this;
    params[nbCPUThread + i].threadId = 0x80L + i;
    params[nbCPUThread + i].isRunning = true;
    params[nbCPUThread + i].gpuId = gpuId[i];
    params[nbCPUThread + i].gridSizeX = gridSize[2 * i];
    params[nbCPUThread + i].gridSizeY = gridSize[2 * i + 1];
#ifdef WIN64
    DWORD thread_id;
    CreateThread(NULL, 0, _FindKeyGPU, (void *)(params + (nbCPUThread + i)), 0, &thread_id);
#else
    pthread_t thread_id;
    pthread_create(&thread_id, NULL, &_FindKeyGPU, (void *)(params + (nbCPUThread + i)));
#endif
  }
  CP_PARAM cparams = {this};
  if (nbGPUThread > 0) {
#ifdef WIN64
    DWORD thread_id;
    CreateThread(NULL, 0, _CheckPrefixes, (void *)(&cparams), 0, &thread_id);
#else
    pthread_t thread_id;
    pthread_create(&thread_id, NULL, &_CheckPrefixes, (void *)(&cparams));
#endif
  }

#ifndef WIN64
  setvbuf(stdout, NULL, _IONBF, 0);
#endif

  uint64_t lastCount = 0;
  uint64_t gpuCount = 0;
  uint64_t lastGPUCount = 0;

  // Key rate smoothing filter
#define FILTER_SIZE 8
  double lastkeyRate[FILTER_SIZE];
  double lastGpukeyRate[FILTER_SIZE];
  uint32_t filterPos = 0;

  memset(lastkeyRate, 0, sizeof(lastkeyRate));
  memset(lastGpukeyRate, 0, sizeof(lastkeyRate));

  // Wait that all threads have started
  while (!hasStarted(params)) {
    Timer::SleepMillis(500);
  }

  t0 = Timer::get_tick();
  startTime = t0;
  lastGPUTime = t0;

  while (isAlive(params)) {

    int delay = 2000;
    while (isAlive(params) && delay > 0) {
      Timer::SleepMillis(500);
      delay -= 500;
    }

    gpuCount = getGPUCount();
    uint64_t count = getCPUCount() + gpuCount;

    t1 = Timer::get_tick();
    double elapsed = t1 - t0;
    double totalElapsed = t1 - startTime;

    if (isAlive(params)) {
      // Calculate overall key finding rate
      KPS = (nbFoundKey - nbFoundKeyLast) / elapsed;
      avgKPS = nbFoundKey / totalElapsed;
      nbFoundKeyLast = nbFoundKey;

      // Calculate GPU key processing rate
      if (nbGPUThread > 0) {
        gpuKPS = (gpuCount - lastGPUCount) / elapsed;
        gpuAvgKPS = gpuCount / totalElapsed;
        lastGPUCount = gpuCount;
      }

      printf(
        "\r[Time %.1fs][%.2f Gkey/s][GPU %.2f Gkey/s][Found %d][%.2f KPS][Avg %.2f key/s]",
        totalElapsed,
        gpuKPS / 1e9,
        gpuAvgKPS / 1e9,
        nbFoundKey,
        KPS,
        avgKPS);
      fflush(stdout);
    }

    if (rekey > 0) {
      if ((count - lastRekey) > (1000000 * rekey)) {
        rekeyRequest(params);
        lastRekey = count;
      }
    }

    lastCount = count;
    lastGPUCount = gpuCount;
    t0 = t1;
  }

  free(params);
}
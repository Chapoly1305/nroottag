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

#include "Timer.h"
#include "Vanity.h"
#include "SECP224r1.h"
#include <fstream>
#include <string>
#include <string.h>
#include <stdexcept>
#include "hash/sha512.h"
#include "hash/sha256.h"

#define RELEASE "1.0"

using namespace std;

// ------------------------------------------------------------------------------------------

void printUsage() {

  printf("Seeker [-check] [-v] [-u] [-b] [-c] [-gpu] [-stop] [-i inputfile]\n");
  printf("             [-gpuId gpuId1[,gpuId2,...]] [-g g1x,g1y,[,g2x,g2y,...]]\n");
  printf("             [-o outputfile] [-m maxFound] [-ps seed] [-s seed] [-t nbThread]\n");
  printf("             [-nosse] [-r rekey] [-check] [-kp] [-sp startPubKey]\n");
  printf("             [prefix]\n\n");
  printf(" prefix: prefix to search (Can contains wildcard '?' or '*')\n");
  printf(" -v: Print version\n");
  printf(" -u: Search uncompressed addresses\n");
  printf(" -b: Search both uncompressed or compressed addresses\n");
  printf(" -p: Search public keys\n");
  printf(" -c: Case unsensitive search\n");
  printf(" -gpu: Enable gpu calculation\n");
  printf(" -stop: Stop when all prefixes are found\n");
  printf(" -i inputfile: Get list of prefixes to search from specified file\n");
  printf(" -o outputfile: Output results to the specified file\n");
  printf(" -gpu gpuId1,gpuId2,...: List of GPU(s) to use, default is 0\n");
  printf(" -g g1x,g1y,g2x,g2y, ...: Specify GPU(s) kernel gridsize, default is 8*(MP number),128\n");
  printf(" -m: Specify maximun number of prefixes found by each kernel call\n");
  printf(" -s seed: Specify a seed for the base key, default is random\n");
  printf(" -ps seed: Specify a seed concatened with a crypto secure random seed\n");
  printf(" -t threadNumber: Specify number of CPU thread, default is number of core\n");
  printf(" -nosse: Disable SSE hash function\n");
  printf(" -l: List cuda enabled devices\n");
  printf(" -check: Check CPU and GPU kernel vs CPU\n");
  printf(" -cp privKey: Compute public key (privKey in hex format)\n");
  printf(" -ca pubKey: Compute address (pubKey in hex format)\n");
  printf(" -kp: Generate key pair\n");
  printf(" -sp startPubKey: Start the search with a pubKey (for private key splitting)\n");
  printf(" -r rekey: Rekey interval in MegaKey, default is disabled\n");
  exit(0);
}

// ------------------------------------------------------------------------------------------

int getInt(string name, char *v) {

  int r;

  try {

    r = std::stoi(string(v));

  } catch (std::invalid_argument &) {

    printf("Invalid %s argument, number expected\n", name.c_str());
    exit(-1);
  }

  return r;
}

// ------------------------------------------------------------------------------------------

void getInts(string name, vector<int> &tokens, const string &text, char sep) {

  size_t start = 0, end = 0;
  tokens.clear();
  int item;

  try {

    while ((end = text.find(sep, start)) != string::npos) {
      item = std::stoi(text.substr(start, end - start));
      tokens.push_back(item);
      start = end + 1;
    }

    item = std::stoi(text.substr(start));
    tokens.push_back(item);

  } catch (std::invalid_argument &) {

    printf("Invalid %s argument, number expected\n", name.c_str());
    exit(-1);
  }
}

// ------------------------------------------------------------------------------------------

void parseFile(string fileName, vector<string> &lines) {

  // Get file size
  FILE *fp = fopen(fileName.c_str(), "rb");
  if (fp == NULL) {
    printf("Error: Cannot open %s %s\n", fileName.c_str(), strerror(errno));
    exit(-1);
  }
  fseek(fp, 0L, SEEK_END);
  size_t sz = ftell(fp);
  size_t nbAddr = sz / 33; /* Upper approximation */
  bool loaddingProgress = sz > 100000;
  fclose(fp);

  // Parse file
  int nbLine = 0;
  string line;
  ifstream inFile(fileName);
  lines.reserve(nbAddr);
  while (getline(inFile, line)) {

    // Remove ending \r\n
    int l = (int)line.length() - 1;
    while (l >= 0 && isspace(line.at(l))) {
      line.pop_back();
      l--;
    }

    if (line.length() > 0) {
      lines.push_back(line);
      nbLine++;
      if (loaddingProgress) {
        if ((nbLine % 50000) == 0)
          printf("[Loading input file %5.1f%%]\r", ((double)nbLine * 100.0) / ((double)(nbAddr) * 33.0 / 34.0));
      }
    }
  }

  if (loaddingProgress)
    printf("[Loading input file 100.0%%]\n");
}

// ------------------------------------------------------------------------------------------

void generateKeyPair(Secp224R1 *secp, string seed, int searchMode, bool paranoiacSeed) {

  if (seed.length() < 8) {
    printf("Error: Use a seed of at least 8 characters to generate a key pair\n");
    printf("Ex: VanitySearch -s \"A Strong Password\" -kp\n");
    exit(-1);
  }

  if (paranoiacSeed)
    seed = seed + Timer::getSeed(32);

  if (searchMode == SEARCH_BOTH) {
    printf("Error: Use compressed or uncompressed to generate a key pair\n");
    exit(-1);
  }

  bool compressed = (searchMode == SEARCH_COMPRESSED);

  string salt = "VanitySearch";
  unsigned char hseed[64];
  pbkdf2_hmac_sha512(
    hseed, 64, (const uint8_t *)seed.c_str(), seed.length(), (const uint8_t *)salt.c_str(), salt.length(), 2048);

  Int privKey;
  privKey.SetInt32(0);
  sha256(hseed, 64, (unsigned char *)privKey.bits64);
  Point p = secp->ComputePublicKey(&privKey);
//  printf("Priv : %s\n", secp->GetPrivAddress(compressed, privKey).c_str());
  printf("Pub  : %s\n", secp->GetPublicKeyHex(compressed, p).c_str());
}

// ------------------------------------------------------------------------------------------

// void outputAdd(string outputFile, int addrType, string addr, string pAddr, string pAddrHex) {

//   FILE *f = stdout;
//   bool needToClose = false;

//   if (outputFile.length() > 0) {
//     f = fopen(outputFile.c_str(), "a");
//     if (f == NULL) {
//       printf("Cannot open %s for writing\n", outputFile.c_str());
//       f = stdout;
//     } else {
//       needToClose = true;
//     }
//   }

//   fprintf(f, "\nPub Addr: %s\n", addr.c_str());

//   switch (addrType) {
//     case P2PKH:
//       fprintf(f, "Priv (WIF): p2pkh:%s\n", pAddr.c_str());
//       break;
//     case P2SH:
//       fprintf(f, "Priv (WIF): p2wpkh-p2sh:%s\n", pAddr.c_str());
//       break;
//     case BECH32:
//       fprintf(f, "Priv (WIF): p2wpkh:%s\n", pAddr.c_str());
//       break;
//   }
//   fprintf(f, "Priv (HEX): 0x%s\n", pAddrHex.c_str());

//   if (needToClose)
//     fclose(f);
// }

// ------------------------------------------------------------------------------------------

int main(int argc, char *argv[]) {

  // Global Init
  Timer::Init();
  rseed(Timer::getSeed32());

  // Init SecpK1
  Secp224R1 *secp = new Secp224R1();
  secp->Init();

  // Browse arguments
  if (argc < 2) {
    printf("Error: No arguments (use -h for help)\n");
    exit(-1);
  }

  int a = 1;
  bool gpuEnable = false;
  bool stop = false;
  int searchMode = SEARCH_COMPRESSED;
  vector<int> gpuId = {0};
  vector<int> gridSize;
  string seed = "";
  vector<string> prefix;
  string outputFile = "";
  int nbCPUThread = Timer::getCoreNumber();
  bool tSpecified = false;
  bool sse = true;
  uint32_t maxFound = 6553500; //Affects usage of GPU memory. Compute as 48 bytes * maxFound
  uint64_t rekey = 0;
  Point startPuKey;
  startPuKey.Clear();
  bool startPubKeyCompressed;
  bool caseSensitive = true;
  bool paranoiacSeed = false;

  while (a < argc) {

    if (strcmp(argv[a], "-gpu") == 0) {
      gpuEnable = true;
      a++;
    } else if (strcmp(argv[a], "-gpuId") == 0) {
      a++;
      getInts("gpuId", gpuId, string(argv[a]), ',');
      a++;
    } else if (strcmp(argv[a], "-stop") == 0) {
      stop = true;
      a++;
    } else if (strcmp(argv[a], "-c") == 0) {
      caseSensitive = false;
      a++;
    } else if (strcmp(argv[a], "-v") == 0) {
      printf("%s\n", RELEASE);
      exit(0);
    } else if (strcmp(argv[a], "-check") == 0) {
      // Check modular mult
      Int a;
      Int b;
      Int c;
      Int d;
      a.SetBase16("9EABF932365A7215BA638A8F11E5E4A6F5AC7CC42082B3160895FC8F");
      b.SetBase16("A6445595DDD941827134B378868A79D11914240B9F21E3522C8333A6");
      c.ModMul(&a, &b);
      d.ModMulR1(&a, &b);
      Int::InitR1();
      printf(
        "A=0x%s\nB=0x%s\nC=0x%s\nD=0x%s\n",
        a.GetBase16().c_str(),
        b.GetBase16().c_str(),
        c.GetBase16().c_str(),
        d.GetBase16().c_str());
      //      Int::Check();
      //      secp->Check();
      printf("R1=%s\n", Int::GetR()->GetBase16().c_str());
      printf("R2=%s\n", Int::GetR2()->GetBase16().c_str());
      printf("R3=%s\n", Int::GetR3()->GetBase16().c_str());
      printf("R4=%s\n", Int::GetR4()->GetBase16().c_str());
      printf("MM64=%#016lx\n", Int::GetMM64());

#ifdef WITHGPU
      if (gridSize.size() == 0) {
        gridSize.push_back(-1); // gridSize X, Let CUDA Determine based on num of SM, 32*SM is found best. 
        gridSize.push_back(512); // gridSize Y. We tested the optimal is 512, greater will crash. 
      }
      GPUEngine g(gridSize[0], gridSize[1], gpuId[0], maxFound, false);
      g.SetSearchMode(searchMode);
      g.Check(secp);
#else
      printf("GPU code not compiled, use -DWITHGPU when compiling.\n");
#endif
      exit(0);
    } else if (strcmp(argv[a], "-l") == 0) {

#ifdef WITHGPU
      GPUEngine::PrintCudaInfo();
#else
      printf("GPU code not compiled, use -DWITHGPU when compiling.\n");
#endif
      exit(0);

    } else if (strcmp(argv[a], "-kp") == 0) {
      generateKeyPair(secp, seed, searchMode, paranoiacSeed);
      exit(0);
    } else if (strcmp(argv[a], "-sp") == 0) {
      a++;
      string pub = string(argv[a]);
      startPuKey = secp->ParsePublicKeyHex(pub, startPubKeyCompressed);
      a++;
    } else if (strcmp(argv[a], "-ca") == 0) {
      a++;
      string pub = string(argv[a]);
      bool isComp;
      Point p = secp->ParsePublicKeyHex(pub, isComp);
      exit(0);
    } else if (strcmp(argv[a], "-cp") == 0) {
      a++;
      string priv = string(argv[a]);
      Int k;
      bool isComp = true;
      if (priv[0] == '5' || priv[0] == 'K' || priv[0] == 'L') {
        k = secp->DecodePrivateKey((char *)priv.c_str(), &isComp);
      } else {
        k.SetBase16(argv[a]);
      }
      Point p = secp->ComputePublicKey(&k);
      printf("PubKey: %s\n", secp->GetPublicKeyHex(isComp, p).c_str());
      printf("Private Number: %s\n", k.GetBase10().c_str());
      printf("X: %s\n", p.x.GetBase10().c_str());
      printf("Y: %s\n", p.y.GetBase10().c_str());
      exit(0);
    } else if (strcmp(argv[a], "-u") == 0) {
      searchMode = SEARCH_UNCOMPRESSED;
      a++;
    } else if (strcmp(argv[a], "-b") == 0) {
      searchMode = SEARCH_BOTH;
      a++;
    } else if (strcmp(argv[a], "-p") == 0) {
      searchMode = SEARCH_PUBLICKEYS;
      a++;
    } else if (strcmp(argv[a], "-nosse") == 0) {
      sse = false;
      a++;
    } else if (strcmp(argv[a], "-g") == 0) {
      a++;
      getInts("gridSize", gridSize, string(argv[a]), ',');
      a++;
    } else if (strcmp(argv[a], "-s") == 0) {
      a++;
      seed = string(argv[a]);
      a++;
    } else if (strcmp(argv[a], "-ps") == 0) {
      a++;
      seed = string(argv[a]);
      paranoiacSeed = true;
      a++;
    } else if (strcmp(argv[a], "-o") == 0) {
      a++;
      outputFile = string(argv[a]);
      a++;
    } else if (strcmp(argv[a], "-i") == 0) {
      a++;
      parseFile(string(argv[a]), prefix);
      a++;
    } else if (strcmp(argv[a], "-t") == 0) {
      a++;
      nbCPUThread = getInt("nbCPUThread", argv[a]);
      a++;
      tSpecified = true;
    } else if (strcmp(argv[a], "-m") == 0) {
      a++;
      maxFound = getInt("maxFound", argv[a]);
      a++;
    } else if (strcmp(argv[a], "-r") == 0) {
      a++;
      rekey = (uint64_t)getInt("rekey", argv[a]);
      a++;
    } else if (strcmp(argv[a], "-generateCode") == 0) {
      GPUEngine::GenerateCode(secp, 1024);
      return 0;
    } else if (strcmp(argv[a], "-h") == 0) {
      printUsage();
    } else if (a == argc - 1) {
      prefix.push_back(string(argv[a]));
      a++;
    } else {
      printf("Unexpected %s argument\n", argv[a]);
      exit(-1);
    }
  }
printf("   _____           _             \n"
       "  / ____|         | |            \n"
       " | (___   ___  ___| | _____ _ __ \n"
       "  \\___ \\ / _ \\/ _ \\ |/ / _ \\ '__|\n"
       "  ____) |  __/  __/   <  __/ |   \n"
       " |_____/ \\___|\\___|\\_\\_\\___|_|   \n"
       "                                 \n"
       "                                 \n");
printf("Release " RELEASE "\n");

  if (gridSize.size() == 0) {
    for (int i = 0; i < gpuId.size(); i++) {
      gridSize.push_back(-1);
      gridSize.push_back(512);
    }
  } else if (gridSize.size() != gpuId.size() * 2) {
    printf("Invalid gridSize or gpuId argument, must have coherent size\n");
    exit(-1);
  }

  // Let one CPU core free per gpu is gpu is enabled
  // It will avoid to hang the system
  if (!tSpecified && nbCPUThread > 1 && gpuEnable)
    nbCPUThread -= (int)gpuId.size();
  if (nbCPUThread < 0)
    nbCPUThread = 0;

  // If a starting public key is specified, force the search mode according to the key
  if (!startPuKey.isZero()) {
    searchMode = (startPubKeyCompressed) ? SEARCH_COMPRESSED : SEARCH_UNCOMPRESSED;
  }

  VanitySearch *v = new VanitySearch(
    secp,
    prefix,
    seed,
    searchMode,
    gpuEnable,
    stop,
    outputFile,
    sse,
    maxFound,
    rekey,
    caseSensitive,
    startPuKey,
    paranoiacSeed);
  v->Search(nbCPUThread, gpuId, gridSize);

  return 0;
}

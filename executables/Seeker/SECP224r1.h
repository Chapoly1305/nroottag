/*
 * Copyright (c) 2025 Chapoly1305, William Flores
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

#ifndef SECP256R1H
#define SECP256R1H

#include "Point.h"
#include <string>
#include <vector>

// Address type
#define P2PKH 0
#define P2SH 1
#define BECH32 2
#define PUB 3

class Secp224R1 {
public:
  Secp224R1();
  ~Secp224R1();
  void Init();
  Point ComputePublicKey(Int *privKey);
  Point NextKey(Point &key);
  void Check();
  bool EC(Point &p);

  void GetHash160(
    int type,
    bool compressed,
    Point &k0,
    Point &k1,
    Point &k2,
    Point &k3,
    uint8_t *h0,
    uint8_t *h1,
    uint8_t *h2,
    uint8_t *h3);

  void GetHash160(int type, bool compressed, const Point &pubKey, unsigned char *hash);

  std::string GetPublicKeyHex(bool compressed, Point &p);
  Point ParsePublicKeyHex(std::string str, bool &isCompressed);


  static Int DecodePrivateKey(char *key, bool *compressed);

  Point Add(Point &p1, Point &p2);
  Point Add2(Point &p1, Point &p2);
  Point AddDirect(Point &p1, Point &p2);
  Point Double(Point &p);
  Point DoubleDirect(Point &p);

  Point G;   // Generator
  Int order; // Curve order

private:
  uint8_t GetByte(std::string &str, int idx);

  Int GetY(Int x, bool isEven);
  Point GTable[256 * 28]; // Generator table
  Int a, b;
};

#endif // SECP256KR1H

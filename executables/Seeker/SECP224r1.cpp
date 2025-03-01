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

#include "SECP224r1.h"
#include "hash/sha256.h"
#include <string.h>

Secp224R1::Secp224R1() {}

void Secp224R1::Init() {

  // Prime for the finite field
  Int P;
  P.SetBase16("ffffffffffffffffffffffffffffffff000000000000000000000001");

  // Set up field
  Int::SetupField(&P);

  // Generator point and order
  G.x.SetBase16("b70e0cbd6bb4bf7f321390b94a03c1d356c21122343280d6115c1d21");
  G.y.SetBase16("bd376388b5f723fb4c22dfe6cd4375a05a07476444d5819985007e34");
  a.SetBase16("fffffffffffffffffffffffffffffffefffffffffffffffffffffffe");
  b.SetBase16("b4050a850c04b3abf54132565044b0b7d7bfd8ba270b39432355ffb4");
  G.z.SetInt32(1);
  order.SetBase16("10000000000000000000000000001dce8d2ec6184caf0a971769fb1f7");
  Int o;
  o.SetBase16("ffffffffffffffffffffffffffff16a2e0b8f03e13dd29455c5c2a3d");
  //    o.ModNeg();
  //    o.ModInv();
  //    printf("mod %s\n", o.GetBase16().c_str());

  Int::InitK1(&order);

  // Compute Generator table
  Point N(G);
  for (int i = 0; i < 28; i++) {
    GTable[i * 256] = N;
    //        printf("%s,%s\n", N.x.GetBase10().c_str(), N.y.GetBase10().c_str());
    N = DoubleDirect(N);
    for (int j = 1; j < 255; j++) {
      GTable[i * 256 + j] = N;
      N = AddDirect(N, GTable[i * 256]);
    }
    GTable[i * 256 + 255] = N; // Dummy point for check function
  }
}

Secp224R1::~Secp224R1() {}

void PrintResult(bool ok) {
  if (ok) {
    printf("OK\n");
  } else {
    printf("Failed !\n");
  }
}

void Secp224R1::Check() {

  printf("Check Generator :");

  bool ok = true;
  int i = 0;
  while (i < 256 * 28 && EC(GTable[i])) {
    i++;
  }
  printf("%d\n", i);
  PrintResult(i == 256 * 28);

  printf("Check Double :");
  Point Pt(G);
  Point R1;
  Point R2;
  Point R3;
  R1 = Double(G);
  R1.Reduce();
  PrintResult(EC(R1));

  printf("Check Add :");
  R2 = Add(G, R1);
  R3 = Add(R1, R2);
  R3.Reduce();
  PrintResult(EC(R3));

  printf("Check GenKey :");
  Int privKey;
  privKey.SetBase16("46b9e861b63d3509c88b7817275a30d22d62c8cd8fa6486ddee35ef0d8e0495f");
  Point pub = ComputePublicKey(&privKey);
  Point expectedPubKey;
  expectedPubKey.x.SetBase16("2500e7f3fbddf2842903f544ddc87494ce95029ace4e257d54ba77f2bc1f3a88");
  expectedPubKey.y.SetBase16("37a9461c4f1c57fecc499753381e772a128a5820a924a2fa05162eb662987a9f");
  expectedPubKey.z.SetInt32(1);

  PrintResult(pub.equals(expectedPubKey));
}

// Compute the public x and y coordinates of
// a private key, it uses the precomputed generator table
// based off the G of the curve, to compute the public key
// https://crypto.stackexchange.com/questions/86242/what-is-the-use-of-pre-computed-points-in-ecc
Point Secp224R1::ComputePublicKey(Int *privKey) {

  int i = 0;
  uint8_t b;
  Point Q;
  Q.Clear();

  // Search first significant byte
  for (i = 0; i < 28; i++) {
    b = privKey->GetByte(i);
    if (b)
      break;
  }
  Q = GTable[256 * i + (b - 1)];
  i++;

  for (; i < 28; i++) {
    b = privKey->GetByte(i);
    if (b)
      Q = Add2(Q, GTable[256 * i + (b - 1)]);
  }

  Q.Reduce();
  return Q;
}

Point Secp224R1::NextKey(Point &key) {
  // Input key must be reduced and different from G
  // in order to use AddDirect
  return AddDirect(key, G);
}

Int Secp224R1::DecodePrivateKey(char *key, bool *compressed) {

  Int ret;
  ret.SetInt32(0);
  std::vector<unsigned char> privKey;

  printf("Invalid private key, not supported !\n");
  ret.SetInt32(-1);
  return ret;
}

uint8_t Secp224R1::GetByte(std::string &str, int idx) {

  char tmp[3];
  int val;

  tmp[0] = str.data()[2 * idx];
  tmp[1] = str.data()[2 * idx + 1];
  tmp[2] = 0;

  if (sscanf(tmp, "%X", &val) != 1) {
    printf("ParsePublicKeyHex: Error invalid public key specified (unexpected hexadecimal digit)\n");
    exit(-1);
  }

  return (uint8_t)val;
}

Point Secp224R1::ParsePublicKeyHex(std::string str, bool &isCompressed) {

  Point ret;
  ret.Clear();

  if (str.length() < 2) {
    printf("ParsePublicKeyHex: Error invalid public key specified (66 or 130 character length)\n");
    exit(-1);
  }

  uint8_t type = GetByte(str, 0);

  switch (type) {

    case 0x02:
      if (str.length() != 66) {
        printf("ParsePublicKeyHex: Error invalid public key specified (66 character length)\n");
        exit(-1);
      }
      for (int i = 0; i < 32; i++)
        ret.x.SetByte(31 - i, GetByte(str, i + 1));
      ret.y = GetY(ret.x, true);
      isCompressed = true;
      break;

    case 0x03:
      if (str.length() != 66) {
        printf("ParsePublicKeyHex: Error invalid public key specified (66 character length)\n");
        exit(-1);
      }
      for (int i = 0; i < 32; i++)
        ret.x.SetByte(31 - i, GetByte(str, i + 1));
      ret.y = GetY(ret.x, false);
      isCompressed = true;
      break;

    case 0x04:
      if (str.length() != 130) {
        printf("ParsePublicKeyHex: Error invalid public key specified (130 character length)\n");
        exit(-1);
      }
      for (int i = 0; i < 32; i++)
        ret.x.SetByte(31 - i, GetByte(str, i + 1));
      for (int i = 0; i < 32; i++)
        ret.y.SetByte(31 - i, GetByte(str, i + 33));
      isCompressed = false;
      break;

    default:
      printf("ParsePublicKeyHex: Error invalid public key specified (Unexpected prefix (only 02,03 or 04 allowed)\n");
      exit(-1);
  }

  ret.z.SetInt32(1);

  if (!EC(ret)) {
    printf("ParsePublicKeyHex: Error invalid public key specified (Not lie on elliptic curve)\n");
    exit(-1);
  }

  return ret;
}

std::string Secp224R1::GetPublicKeyHex(bool compressed, Point &pubKey) {

  unsigned char publicKeyBytes[128];
  char tmp[3];
  std::string ret;

  if (!compressed) {

    // Full public key
    publicKeyBytes[0] = 0x4;
    pubKey.x.Get32Bytes(publicKeyBytes + 1);
    pubKey.y.Get32Bytes(publicKeyBytes + 33);

    for (int i = 0; i < 65; i++) {
      sprintf(tmp, "%02X", (int)publicKeyBytes[i]);
      ret.append(tmp);
    }

  } else {

    // Compressed public key
    publicKeyBytes[0] = pubKey.y.IsEven() ? 0x2 : 0x3;
    pubKey.x.Get32Bytes(publicKeyBytes + 1);

    for (int i = 0; i < 33; i++) {
      sprintf(tmp, "%02X", (int)publicKeyBytes[i]);
      ret.append(tmp);
    }
  }

  return ret;
}

// Add two points, Z = 1
Point Secp224R1::AddDirect(Point &p1, Point &p2) {

  Int _s;
  Int _p;
  Int dy;
  Int dx;
  Point r;
  r.z.SetInt32(1);

  dy.ModSub(&p2.y, &p1.y);
  dx.ModSub(&p2.x, &p1.x);
  dx.ModInv();
  _s.ModMul(&dy, &dx); // s = (p2.y-p1.y)*inverse(p2.x-p1.x);

  _p.ModSquare(&_s); // _p = pow2(s)

  r.x.ModSub(&_p, &p1.x);
  r.x.ModSub(&p2.x); // rx = pow2(s) - p1.x - p2.x;

  r.y.ModSub(&p2.x, &r.x);
  r.y.ModMul(&_s);
  r.y.ModSub(&p2.y); // ry = - p2.y - s*(ret.x-p2.x);

  return r;
}

// Add two points
Point Secp224R1::Add2(Point &p1, Point &p2) {

  // P2.z = 1

  Int u;
  Int v;
  Int u1;
  Int v1;
  Int vs2;
  Int vs3;
  Int us2;
  Int a;
  Int us2w;
  Int vs2v2;
  Int vs3u2;
  Int _2vs2v2;
  Point r;

  u1.ModMul(&p2.y, &p1.z);
  v1.ModMul(&p2.x, &p1.z);
  u.ModSub(&u1, &p1.y);
  v.ModSub(&v1, &p1.x);
  us2.ModSquare(&u);
  vs2.ModSquare(&v);
  vs3.ModMul(&vs2, &v);
  us2w.ModMul(&us2, &p1.z);
  vs2v2.ModMul(&vs2, &p1.x);
  _2vs2v2.ModAdd(&vs2v2, &vs2v2);
  a.ModSub(&us2w, &vs3);
  a.ModSub(&_2vs2v2);

  r.x.ModMul(&v, &a);

  vs3u2.ModMul(&vs3, &p1.y);
  r.y.ModSub(&vs2v2, &a);
  r.y.ModMul(&r.y, &u);
  r.y.ModSub(&vs3u2);

  r.z.ModMul(&vs3, &p1.z);

  return r;
}

// Add two points
Point Secp224R1::Add(Point &p1, Point &p2) {

  Int u;
  Int v;
  Int u1;
  Int u2;
  Int v1;
  Int v2;
  Int vs2;
  Int vs3;
  Int us2;
  Int w;
  Int a;
  Int us2w;
  Int vs2v2;
  Int vs3u2;
  Int _2vs2v2;
  Int x3;
  Int vs3y1;
  Point r;

  /*
  U1 = Y2 * Z1
  U2 = Y1 * Z2
  V1 = X2 * Z1
  V2 = X1 * Z2
  if (V1 == V2)
    if (U1 != U2)
      return POINT_AT_INFINITY
    else
      return POINT_DOUBLE(X1, Y1, Z1)
  U = U1 - U2
  V = V1 - V2
  W = Z1 * Z2
  A = U ^ 2 * W - V ^ 3 - 2 * V ^ 2 * V2
  X3 = V * A
  Y3 = U * (V ^ 2 * V2 - A) - V ^ 3 * U2
  Z3 = V ^ 3 * W
  return (X3, Y3, Z3)
  */

  u1.ModMul(&p2.y, &p1.z);
  u2.ModMul(&p1.y, &p2.z);
  v1.ModMul(&p2.x, &p1.z);
  v2.ModMul(&p1.x, &p2.z);
  u.ModSub(&u1, &u2);
  v.ModSub(&v1, &v2);
  w.ModMul(&p1.z, &p2.z);
  us2.ModSquare(&u);
  vs2.ModSquare(&v);
  vs3.ModMul(&vs2, &v);
  us2w.ModMul(&us2, &w);
  vs2v2.ModMul(&vs2, &v2);
  _2vs2v2.ModAdd(&vs2v2, &vs2v2);
  a.ModSub(&us2w, &vs3);
  a.ModSub(&_2vs2v2);

  r.x.ModMul(&v, &a);

  vs3u2.ModMul(&vs3, &u2);
  r.y.ModSub(&vs2v2, &a);
  r.y.ModMul(&r.y, &u);
  r.y.ModSub(&vs3u2);

  r.z.ModMul(&vs3, &w);

  return r;
}

// optimized double since Z = 1
Point Secp224R1::DoubleDirect(Point &p) {

  Int _s;
  Int _p;
  Int a;
  Point r;
  r.z.SetInt32(1);

  _s.ModMul(&p.x, &p.x);
  _p.ModAdd(&_s, &_s);
  _p.ModAdd(&_s);
  _p.ModAdd(&this->a);

  a.ModAdd(&p.y, &p.y);
  a.ModInv();
  _s.ModMul(&_p, &a); // s = (3*pow2(p.x) + a)*inverse(2*p.y);

  _p.ModMul(&_s, &_s);
  a.ModAdd(&p.x, &p.x);
  a.ModNeg();
  r.x.ModAdd(&a, &_p); // rx = pow2(s) + neg(2*p.x);

  a.ModSub(&r.x, &p.x);

  _p.ModMul(&a, &_s);
  r.y.ModAdd(&_p, &p.y);
  r.y.ModNeg(); // ry = neg(p.y + s*(ret.x+neg(p.x)));

  return r;
}

// double a point
Point Secp224R1::Double(Point &p) {

  /*
  if (Y == 0)
    return POINT_AT_INFINITY
    W = a * Z ^ 2 + 3 * X ^ 2
    S = Y * Z
    B = X * Y*S
    H = W ^ 2 - 8 * B
    X' = 2*H*S
    Y' = W*(4*B - H) - 8*Y^2*S^2
    Z' = 8*S^3
    return (X', Y', Z')
  */

  Int z2;
  Int x2;
  Int _3x2;
  Int w;
  Int s;
  Int s2;
  Int b;
  Int _8b;
  Int _8y2s2;
  Int y2;
  Int h;
  Point r;

  z2.ModSquare(&p.z);
  z2.ModMul(&this->a);
  x2.ModSquare(&p.x);
  _3x2.ModAdd(&x2, &x2);
  _3x2.ModAdd(&x2);
  w.ModAdd(&z2, &_3x2);
  s.ModMul(&p.y, &p.z);
  b.ModMul(&p.y, &s);
  b.ModMul(&p.x);
  h.ModSquare(&w);
  _8b.ModAdd(&b, &b);
  _8b.ModDouble();
  _8b.ModDouble();
  h.ModSub(&_8b);

  r.x.ModMul(&h, &s);
  r.x.ModAdd(&r.x);

  s2.ModSquare(&s);
  y2.ModSquare(&p.y);
  _8y2s2.ModMul(&y2, &s2);
  _8y2s2.ModDouble();
  _8y2s2.ModDouble();
  _8y2s2.ModDouble();

  r.y.ModAdd(&b, &b);
  r.y.ModAdd(&r.y, &r.y);
  r.y.ModSub(&h);
  r.y.ModMul(&w);
  r.y.ModSub(&_8y2s2);

  r.z.ModMul(&s2, &s);
  r.z.ModDouble();
  r.z.ModDouble();
  r.z.ModDouble();

  return r;
}

// not used
Int Secp224R1::GetY(Int x, bool isEven) {

  Int _s;
  Int _p;

  _s.ModSquare(&x);
  _p.ModMul(&_s, &x);
  _p.ModAdd(5);
  _p.ModSqrt();

  if (!_p.IsEven() && isEven) {
    _p.ModNeg();
  } else if (_p.IsEven() && !isEven) {
    _p.ModNeg();
  }

  return _p;
}

// Compute wither the point lies on the curve
bool Secp224R1::EC(Point &p) {

  Int _s;
  Int _p;
  Int _ax;

  _s.ModSquare(&p.x);
  _p.ModMul(&_s, &p.x);
  _p.ModAdd(&this->b);
  _ax.ModMul(&p.x, &this->a);
  _p.ModAdd(&_ax);
  _s.ModMul(&p.y, &p.y);
  _s.ModSub(&_p);

  return _s.IsZero(); // ( ((pow2(y) - (pow3(x) + b + (a*x))) % P) == 0 );
}

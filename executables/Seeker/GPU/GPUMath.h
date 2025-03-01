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

// ---------------------------------------------------------------------------------
// 256(+64) bits integer CUDA libray for SECPK1
// ---------------------------------------------------------------------------------

// We need 1 extra block for ModInv
#define NBBLOCK 5
#define BIFULLSIZE 40

// Constants for 0 and 1
__device__ __constant__ uint64_t _0[] = {0ULL, 0ULL, 0ULL, 0ULL, 0ULL};
__device__ __constant__ uint64_t _1[] = {1ULL, 0ULL, 0ULL, 0ULL, 0ULL};

// Assembly directives
// Assembly directives for addition
#define UADDO(c, a, b) asm volatile("add.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b) : "memory");
#define UADDC(c, a, b) asm volatile("addc.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b) : "memory");
#define UADD(c, a, b) asm volatile("addc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b));

#define UADDO1(c, a) asm volatile("add.cc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) : "memory");
#define UADDC1(c, a) asm volatile("addc.cc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) : "memory");
#define UADD1(c, a) asm volatile("addc.u64 %0, %0, %1;" : "+l"(c) : "l"(a));

// Assembly directives for subtraction
#define USUBO(c, a, b) asm volatile("sub.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b) : "memory");
#define USUBC(c, a, b) asm volatile("subc.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b) : "memory");
#define USUB(c, a, b) asm volatile("subc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b));

#define USUBO1(c, a) asm volatile("sub.cc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) : "memory");
#define USUBC1(c, a) asm volatile("subc.cc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) : "memory");
#define USUB1(c, a) asm volatile("subc.u64 %0, %0, %1;" : "+l"(c) : "l"(a));

// Assembly directives for multiplication
#define UMULLO(lo, a, b) asm volatile("mul.lo.u64 %0, %1, %2;" : "=l"(lo) : "l"(a), "l"(b));
#define UMULHI(hi, a, b) asm volatile("mul.hi.u64 %0, %1, %2;" : "=l"(hi) : "l"(a), "l"(b));
#define MADDO(r, a, b, c) asm volatile("mad.hi.cc.u64 %0, %1, %2, %3;" : "=l"(r) : "l"(a), "l"(b), "l"(c) : "memory");
#define MADDC(r, a, b, c) asm volatile("madc.hi.cc.u64 %0, %1, %2, %3;" : "=l"(r) : "l"(a), "l"(b), "l"(c) : "memory");
#define MADD(r, a, b, c) asm volatile("madc.hi.u64 %0, %1, %2, %3;" : "=l"(r) : "l"(a), "l"(b), "l"(c));
#define MADDS(r, a, b, c) asm volatile("madc.hi.s64 %0, %1, %2, %3;" : "=l"(r) : "l"(a), "l"(b), "l"(c));

__device__ void _ModMult(uint64_t *r, uint64_t *a, uint64_t *b);
__device__ void _ModMult(uint64_t *r, uint64_t *a);

// SECPK1 constants
//__device__ __constant__ uint64_t
//  _beta[] = {0xC1396C28719501EEULL, 0x9CF0497512F58995ULL, 0x6E64479EAC3434E9ULL, 0x7AE96A2B657C0710ULL};
//__device__ __constant__ uint64_t
//  _beta2[] = {0x3EC693D68E6AFA40ULL, 0x630FB68AED0A766AULL, 0x919BB86153CBCB16ULL, 0x851695D49A83F8EFULL};
__device__ __constant__ uint64_t
  _P[] = {0x0000000000000001ULL, 0xffffffff00000000ULL, 0xffffffffffffffffULL, 0x00000000ffffffffULL, 0x0ULL};
__device__ __constant__ uint64_t MM64 = 0xffffffffffffffff; // 64bits lsb negative inverse of P (mod 2^64)
__device__ __constant__ uint64_t
  _R2[] = {0xffffffff00000001ULL, 0xffffffff00000000ULL, 0xfffffffe00000000ULL, 0x00000000ffffffffULL, 0x0ULL};
#define HSIZE (GRP_SIZE / 2 - 1)

// 64bits lsb negative inverse of P (mod 2^64)
#define MM64 0xffffffffffffffff
// ---------------------------------------------------------------------------------------

#define _IsPositive(x) (((int64_t)(x[4])) >= 0LL)
#define _IsNegative(x) (((int64_t)(x[4])) < 0LL)
#define _IsEqual(a, b) ((a[4] == b[4]) && (a[3] == b[3]) && (a[2] == b[2]) && (a[1] == b[1]) && (a[0] == b[0]))
#define _IsZero(a) ((a[4] | a[3] | a[2] | a[1] | a[0]) == 0ULL)
#define _IsOne(a) ((a[4] == 0ULL) && (a[3] == 0ULL) && (a[2] == 0ULL) && (a[1] == 0ULL) && (a[0] == 1ULL))

#define IDX threadIdx.x

// Macro for right shift of 128-bit numbers
#define __sright128(a, b, n) ((a) >> (n)) | ((b) << (64 - (n)))
// Macro for left shift of 128-bit numbers
#define __sleft128(a, b, n) ((b) << (n)) | ((a) >> (64 - (n)))

// ---------------------------------------------------------------------------------------

// Macro for adding the prime number to a number
#define AddP(r)                                                                                                        \
  {                                                                                                                    \
    UADDO1(r[0], 0x0000000000000001ULL);                                                                               \
    UADDC1(r[1], 0xffffffff00000000ULL);                                                                               \
    UADDC1(r[2], 0xffffffffffffffffULL);                                                                               \
    UADDC1(r[3], 0x00000000ffffffffULL);                                                                               \
    UADD1(r[4], 0ULL);                                                                                                 \
  }

// ---------------------------------------------------------------------------------------

// Macro for subtracting the prime number from a number
#define SubP(r)                                                                                                        \
  {                                                                                                                    \
    USUBO1(r[0], 0x0000000000000001ULL);                                                                               \
    USUBC1(r[1], 0xffffffff00000000ULL);                                                                               \
    USUBC1(r[2], 0xffffffffffffffffULL);                                                                               \
    USUBC1(r[3], 0x00000000ffffffffULL);                                                                               \
    USUB1(r[4], 0ULL);                                                                                                 \
  }

// ---------------------------------------------------------------------------------------

// Macro for subtracting two 256-bit numbers
#define Sub2(r, a, b)                                                                                                  \
  {                                                                                                                    \
    USUBO(r[0], a[0], b[0]);                                                                                           \
    USUBC(r[1], a[1], b[1]);                                                                                           \
    USUBC(r[2], a[2], b[2]);                                                                                           \
    USUBC(r[3], a[3], b[3]);                                                                                           \
    USUB(r[4], a[4], b[4]);                                                                                            \
  }

// ---------------------------------------------------------------------------------------

// Macro for subtracting two 256-bit numbers, with the result being store in the first parameter
#define Sub1(r, a)                                                                                                     \
  {                                                                                                                    \
    USUBO1(r[0], a[0]);                                                                                                \
    USUBC1(r[1], a[1]);                                                                                                \
    USUBC1(r[2], a[2]);                                                                                                \
    USUBC1(r[3], a[3]);                                                                                                \
    USUB1(r[4], a[4]);                                                                                                 \
  }

// ---------------------------------------------------------------------------------------

// Macro for negating a 256-bit number
#define Neg(r)                                                                                                         \
  {                                                                                                                    \
    USUBO(r[0], 0ULL, r[0]);                                                                                           \
    USUBC(r[1], 0ULL, r[1]);                                                                                           \
    USUBC(r[2], 0ULL, r[2]);                                                                                           \
    USUBC(r[3], 0ULL, r[3]);                                                                                           \
    USUB(r[4], 0ULL, r[4]);                                                                                            \
  }

// ---------------------------------------------------------------------------------------
// Macro for multiplying two 256-bit numbers
#define UMult(r, a, b)                                                                                                 \
  {                                                                                                                    \
    UMULLO(r[0], a[0], b);                                                                                             \
    UMULLO(r[1], a[1], b);                                                                                             \
    MADDO(r[1], a[0], b, r[1]);                                                                                        \
    UMULLO(r[2], a[2], b);                                                                                             \
    MADDC(r[2], a[1], b, r[2]);                                                                                        \
    UMULLO(r[3], a[3], b);                                                                                             \
    MADDC(r[3], a[2], b, r[3]);                                                                                        \
    MADD(r[4], a[3], b, 0ULL);                                                                                         \
  }

// ---------------------------------------------------------------------------------------

// Macro for loading a 256-bit number
#define Load(r, a)                                                                                                     \
  {                                                                                                                    \
    (r)[0] = (a)[0];                                                                                                   \
    (r)[1] = (a)[1];                                                                                                   \
    (r)[2] = (a)[2];                                                                                                   \
    (r)[3] = (a)[3];                                                                                                   \
    (r)[4] = (a)[4];                                                                                                   \
  }

// ---------------------------------------------------------------------------------------

// Macro for loading a 64-bit number into the lower part of a 256-bit number
#define _LoadI64(r, a)                                                                                                 \
  {                                                                                                                    \
    (r)[0] = a;                                                                                                        \
    (r)[1] = a >> 63;                                                                                                  \
    (r)[2] = (r)[1];                                                                                                   \
    (r)[3] = (r)[1];                                                                                                   \
    (r)[4] = (r)[1];                                                                                                   \
  }
// ---------------------------------------------------------------------------------------

// Macro for loading a 256-bit number from another 256-bit number
#define Load256(r, a)                                                                                                  \
  {                                                                                                                    \
    (r)[0] = (a)[0];                                                                                                   \
    (r)[1] = (a)[1];                                                                                                   \
    (r)[2] = (a)[2];                                                                                                   \
    (r)[3] = (a)[3];                                                                                                   \
  }

// ---------------------------------------------------------------------------------------

// Macro for loading a 256-bit number from memory in a parallelized manner
#define Load256A(r, a)                                                                                                 \
  {                                                                                                                    \
    (r)[0] = (a)[IDX];                                                                                                 \
    (r)[1] = (a)[IDX + blockDim.x];                                                                                    \
    (r)[2] = (a)[IDX + 2 * blockDim.x];                                                                                \
    (r)[3] = (a)[IDX + 3 * blockDim.x];                                                                                \
  }

// ---------------------------------------------------------------------------------------

// Macro for storing a 256-bit number into memory in a parallelized manner
#define Store256A(r, a)                                                                                                \
  {                                                                                                                    \
    (r)[IDX] = (a)[0];                                                                                                 \
    (r)[IDX + blockDim.x] = (a)[1];                                                                                    \
    (r)[IDX + 2 * blockDim.x] = (a)[2];                                                                                \
    (r)[IDX + 3 * blockDim.x] = (a)[3];                                                                                \
  }

// ---------------------------------------------------------------------------------------

// Macro for adding two 256-bit numbers
#define Add2(r, a, b)                                                                                                  \
  {                                                                                                                    \
    UADDO(r[0], a[0], b[0]);                                                                                           \
    UADDC(r[1], a[1], b[1]);                                                                                           \
    UADDC(r[2], a[2], b[2]);                                                                                           \
    UADDC(r[3], a[3], b[3]);                                                                                           \
    UADD(r[4], a[4], b[4]);                                                                                            \
  }

// ---------------------------------------------------------------------------------------
// Macro for adding two 256-bit number, being stored in the first parameter
#define Add1(r, a)                                                                                                     \
  {                                                                                                                    \
    UADDO1(r[0], a[0]);                                                                                                \
    UADDC1(r[1], a[1]);                                                                                                \
    UADDC1(r[2], a[2]);                                                                                                \
    UADDC1(r[3], a[3]);                                                                                                \
    UADD1(r[4], a[4]);                                                                                                 \
  }

// Macro for adding two numbers with carry
#define AddC(r, a, carry)                                                                                              \
  {                                                                                                                    \
    UADDO1(r[0], a[0]);                                                                                                \
    UADDC1(r[1], a[1]);                                                                                                \
    UADDC1(r[2], a[2]);                                                                                                \
    UADDC1(r[3], a[3]);                                                                                                \
    UADDC1(r[4], a[4]);                                                                                                \
    UADDC(carry, 0ULL, 0ULL);                                                                                          \
  }

// Macro for adding two numbers and shifting
#define AddAndShift(r, a, b, cH)                                                                                       \
  {                                                                                                                    \
    UADDO(r[0], a[0], b[0]);                                                                                           \
    UADDC(r[0], a[1], b[1]);                                                                                           \
    UADDC(r[1], a[2], b[2]);                                                                                           \
    UADDC(r[2], a[3], b[3]);                                                                                           \
    UADDC(r[3], a[4], b[4]);                                                                                           \
    UADD(r[4], 0ULL, cH);                                                                                              \
  }

// Macro for shifting a 256-bit number by 64 bits
#define Shift64(r, a, cH)                                                                                              \
  {                                                                                                                    \
    r[0] = a[1];                                                                                                       \
    r[1] = a[2];                                                                                                       \
    r[2] = a[3];                                                                                                       \
    r[3] = a[4];                                                                                                       \
    r[4] = cH;                                                                                                         \
  }

// for convert a 256 bit number to hex
__device__ void toHex(char *v, uint64_t *data) {

  printf("%s=%lx%016lx%016lx%016lx\n", v, data[3], data[2], data[1], data[0]);
}

// Shifts a 256-bit number right by 62 bits
__device__ void ShiftR62(uint64_t *r) {

  r[0] = (r[1] << 2) | (r[0] >> 62);
  r[1] = (r[2] << 2) | (r[1] >> 62);
  r[2] = (r[3] << 2) | (r[2] >> 62);
  r[3] = (r[4] << 2) | (r[3] >> 62);
  // With sign extent
  r[4] = (int64_t)(r[4]) >> 62;
}

// Shifts a 256-bit number right by 62 bits with carry
// Similar to ShiftR62, but with an additional carry value
__device__ void ShiftR62(uint64_t dest[5], uint64_t r[5], uint64_t carry) {

  dest[0] = (r[1] << 2) | (r[0] >> 62);
  dest[1] = (r[2] << 2) | (r[1] >> 62);
  dest[2] = (r[3] << 2) | (r[2] >> 62);
  dest[3] = (r[4] << 2) | (r[3] >> 62);
  dest[4] = (carry << 2) | (r[4] >> 62);
}

// ---------------------------------------------------------------------------------------

// Multiplies a 256-bit number by a 64-bit signed integer
__device__ void IMult(uint64_t *r, uint64_t *a, int64_t b) {

  uint64_t t[NBBLOCK];

  // Make b positive
  if (b < 0) {
    b = -b;
    USUBO(t[0], 0ULL, a[0]);
    USUBC(t[1], 0ULL, a[1]);
    USUBC(t[2], 0ULL, a[2]);
    USUBC(t[3], 0ULL, a[3]);
    USUB(t[4], 0ULL, a[4]);
  } else {
    Load(t, a);
  }

  UMULLO(r[0], t[0], b);
  UMULLO(r[1], t[1], b);
  MADDO(r[1], t[0], b, r[1]);
  UMULLO(r[2], t[2], b);
  MADDC(r[2], t[1], b, r[2]);
  UMULLO(r[3], t[3], b);
  MADDC(r[3], t[2], b, r[3]);
  UMULLO(r[4], t[4], b);
  MADD(r[4], t[3], b, r[4]);
}

// Multiplies a 256-bit number by a 64-bit signed integer with carry
__device__ uint64_t IMultC(uint64_t *r, uint64_t *a, int64_t b) {

  uint64_t t[NBBLOCK];
  uint64_t carry;

  // Make b positive
  if (b < 0) {
    b = -b;
    USUBO(t[0], 0ULL, a[0]);
    USUBC(t[1], 0ULL, a[1]);
    USUBC(t[2], 0ULL, a[2]);
    USUBC(t[3], 0ULL, a[3]);
    USUB(t[4], 0ULL, a[4]);
  } else {
    Load(t, a);
  }

  UMULLO(r[0], t[0], b);
  UMULLO(r[1], t[1], b);
  MADDO(r[1], t[0], b, r[1]);
  UMULLO(r[2], t[2], b);
  MADDC(r[2], t[1], b, r[2]);
  UMULLO(r[3], t[3], b);
  MADDC(r[3], t[2], b, r[3]);
  UMULLO(r[4], t[4], b);
  MADDC(r[4], t[3], b, r[4]);
  MADDS(carry, t[4], b, 0ULL);

  return carry;
}

// ---------------------------------------------------------------------------------------
// Multiplies a 256-bit number by a 64-bit unsigned integer a
__device__ void MulP(uint64_t *r, uint64_t a) {

  uint64_t ah;
  uint64_t al;

  UMULLO(al, a, 0x01ULL);
  UMULHI(ah, a, 0x01ULL);

  USUBO(r[0], 0ULL, al);
  USUBC(r[1], 0ULL, ah);
  USUBC(r[2], 0ULL, 0ULL);
  USUBC(r[3], 0ULL, 0ULL);
  USUB(r[4], a, 0ULL);
}

// ---------------------------------------------------------------------------------------

// Computes the negative of a 256-bit number a modulo P
__device__ void ModNeg256(uint64_t *r, uint64_t *a) {

  uint64_t t[4];
  USUBO(t[0], 0ULL, a[0]);
  USUBC(t[1], 0ULL, a[1]);
  USUBC(t[2], 0ULL, a[2]);
  USUBC(t[3], 0ULL, a[3]);
  UADDO(r[0], t[0], 0x0000000000000001ULL);
  UADDC(r[1], t[1], 0xffffffff00000000ULL);
  UADDC(r[2], t[2], 0xffffffffffffffffULL);
  UADD(r[3], t[3], 0x00000000ffffffffULL);
}

// ---------------------------------------------------------------------------------------

// Computes the negative of a 256-bit number r modulo P
__device__ void ModNeg256(uint64_t *r) {

  uint64_t t[4];
  USUBO(t[0], 0ULL, r[0]);
  USUBC(t[1], 0ULL, r[1]);
  USUBC(t[2], 0ULL, r[2]);
  USUBC(t[3], 0ULL, r[3]);
  UADDO(r[0], t[0], 0x0000000000000001ULL);
  UADDC(r[1], t[1], 0xffffffff00000000ULL);
  UADDC(r[2], t[2], 0xffffffffffffffffULL);
  UADD(r[3], t[3], 0x00000000ffffffffULL);
}

// ---------------------------------------------------------------------------------------
// Computes r = (a - b) mod 2^256
__device__ void ModSub256(uint64_t *r, uint64_t *a, uint64_t *b) {

  uint64_t t;
  uint64_t T[4];
  USUBO(r[0], a[0], b[0]);
  USUBC(r[1], a[1], b[1]);
  USUBC(r[2], a[2], b[2]);
  USUBC(r[3], a[3], b[3]);
  USUB(t, 0ULL, 0ULL);
  T[0] = 0x0000000000000001ULL & t;
  T[1] = 0xffffffff00000000ULL & t;
  T[2] = 0xffffffffffffffffULL & t;
  T[3] = 0x00000000ffffffffULL & t;
  UADDO1(r[0], T[0]);
  UADDC1(r[1], T[1]);
  UADDC1(r[2], T[2]);
  UADD1(r[3], T[3]);
}

// ---------------------------------------------------------------------------------------
// Computes r = (r - b) mod 2^256
__device__ void ModSub256(uint64_t *r, uint64_t *b) {

  uint64_t t;
  uint64_t T[4];
  USUBO(r[0], r[0], b[0]);
  USUBC(r[1], r[1], b[1]);
  USUBC(r[2], r[2], b[2]);
  USUBC(r[3], r[3], b[3]);
  USUB(t, 0ULL, 0ULL);
  T[0] = 0x0000000000000001ULL & t;
  T[1] = 0xffffffff00000000ULL & t;
  T[2] = 0xffffffffffffffffULL & t;
  T[3] = 0x00000000ffffffffULL & t;
  UADDO1(r[0], T[0]);
  UADDC1(r[1], T[1]);
  UADDC1(r[2], T[2]);
  UADD1(r[3], T[3]);
}

// ---------------------------------------------------------------------------------------
// Counts the trailing zero bits in a 64-bit unsigned integer x
__device__ __forceinline__ uint32_t ctz(uint64_t x) {
  uint32_t n;

  // Inline assembly to perform bit manipulation
  asm("{\n\t"
      " .reg .u64 tmp;\n\t"       // Declare a temporary 64-bit register
      " brev.b64 tmp, %1;\n\t"    // Reverse the bits of x and store in tmp
      " clz.b64 %0, tmp;\n\t"     // Count leading zeros in tmp and store in n
      "}"
      : "=r"(n)                   // Output constraint: n is a register
      : "l"(x));                  // Input constraint: x is a register

  return n;                       // Return the count of trailing zero bits
}
// ---------------------------------------------------------------------------------------
#define SWAP(tmp, x, y)                                                                                                \
  tmp = x;                                                                                                             \
  x = y;                                                                                                               \
  y = tmp;
// max value of a 62 bit value
#define MSK62 0x3FFFFFFFFFFFFFFF

__device__ void
_DivStep62(uint64_t u[5], uint64_t v[5], int32_t *pos, int64_t *uu, int64_t *uv, int64_t *vu, int64_t *vv) {

  // u' = (uu*u + uv*v) >> bitCount
  // v' = (vu*u + vv*v) >> bitCount
  // Do not maintain a matrix for r and s, the number of
  // 'added P' can be easily calculated

  *uu = 1;
  *uv = 0;
  *vu = 0;
  *vv = 1;

  uint32_t bitCount = 62;
  uint32_t zeros;
  uint64_t u0 = u[0];
  uint64_t v0 = v[0];

  // Extract 64 MSB of u and v
  // u and v must be positive
  uint64_t uh, vh;
  int64_t w, x, y, z;
  bitCount = 62;

  while (*pos > 0 && (u[*pos] | v[*pos]) == 0)
    (*pos)--;
  if (*pos == 0) {

    uh = u[0];
    vh = v[0];

  } else {

    uint32_t s = __clzll(u[*pos] | v[*pos]);
    if (s == 0) {
      uh = u[*pos];
      vh = v[*pos];
    } else {
      uh = __sleft128(u[*pos - 1], u[*pos], s);
      vh = __sleft128(v[*pos - 1], v[*pos], s);
    }
  }

  while (true) {

    // Use a sentinel bit to count zeros only up to bitCount
    zeros = ctz(v0 | (1ULL << bitCount));

    v0 >>= zeros;
    vh >>= zeros;
    *uu <<= zeros;
    *uv <<= zeros;
    bitCount -= zeros;

    if (bitCount == 0)
      break;

    if (vh < uh) {
      SWAP(w, uh, vh);
      SWAP(x, u0, v0);
      SWAP(y, *uu, *vu);
      SWAP(z, *uv, *vv);
    }

    vh -= uh;
    v0 -= u0;
    *vv -= *uv;
    *vu -= *uu;
  }
}

__device__ void
MatrixVecMulHalf(uint64_t dest[5], uint64_t u[5], uint64_t v[5], int64_t _11, int64_t _12, uint64_t *carry) {

  uint64_t t1[NBBLOCK];
  uint64_t t2[NBBLOCK];
  uint64_t c1, c2;

  c1 = IMultC(t1, u, _11);
  c2 = IMultC(t2, v, _12);

  UADDO(dest[0], t1[0], t2[0]);
  UADDC(dest[1], t1[1], t2[1]);
  UADDC(dest[2], t1[2], t2[2]);
  UADDC(dest[3], t1[3], t2[3]);
  UADDC(dest[4], t1[4], t2[4]);
  UADD(*carry, c1, c2);
}

__device__ void MatrixVecMul(uint64_t u[5], uint64_t v[5], int64_t _11, int64_t _12, int64_t _21, int64_t _22) {

  uint64_t t1[NBBLOCK];
  uint64_t t2[NBBLOCK];
  uint64_t t3[NBBLOCK];
  uint64_t t4[NBBLOCK];

  IMult(t1, u, _11);
  IMult(t2, v, _12);
  IMult(t3, u, _21);
  IMult(t4, v, _22);

  UADDO(u[0], t1[0], t2[0]);
  UADDC(u[1], t1[1], t2[1]);
  UADDC(u[2], t1[2], t2[2]);
  UADDC(u[3], t1[3], t2[3]);
  UADD(u[4], t1[4], t2[4]);

  UADDO(v[0], t3[0], t4[0]);
  UADDC(v[1], t3[1], t4[1]);
  UADDC(v[2], t3[2], t4[2]);
  UADDC(v[3], t3[3], t4[3]);
  UADD(v[4], t3[4], t4[4]);
}

// 320 bits Function to perform addition with carry
__device__ uint64_t AddCh(uint64_t r[5], uint64_t a[5], uint64_t carry) {

  uint64_t carryOut;

  UADDO1(r[0], a[0]);
  UADDC1(r[1], a[1]);
  UADDC1(r[2], a[2]);
  UADDC1(r[3], a[3]);
  UADDC1(r[4], a[4]);
  UADD(carryOut, carry, 0ULL);

  return carryOut;
}

#define SWAP_ADD(x, y)                                                                                                 \
  x += y;                                                                                                              \
  y -= x;
#define SWAP_SUB(x, y)                                                                                                 \
  x -= y;                                                                                                              \
  y += x;
#define IS_EVEN(x) ((x & 1LL) == 0)

__device__ void _ModInv(uint64_t *R) {

  // Compute modular inverse of R mop _P (using 320bits signed integer)
  // 0 < this < P  , P must be odd
  // Return 0 if no inverse

  int64_t bitCount;
  int64_t uu, uv, vu, vv;
  int64_t v0, u0;
  uint64_t r0, s0;
  int64_t nb0;

  uint64_t u[NBBLOCK];
  uint64_t v[NBBLOCK];
  uint64_t r[NBBLOCK];
  uint64_t s[NBBLOCK];
  uint64_t t1[NBBLOCK];
  uint64_t t2[NBBLOCK];
  uint64_t t3[NBBLOCK];
  uint64_t t4[NBBLOCK];

  Load(u, _P);
  Load(v, R);
  Load(r, _0);
  Load(s, _1);

  // Delayed right shift 62bits

  while (!_IsZero(u)) {

    // u' = (uu*u + uv*v) >> bitCount
    // v' = (vu*u + vv*v) >> bitCount
    // Do not maintain a matrix for r and s, the number of
    // 'added P' can be easily calculated
    uu = 1;
    uv = 0;
    vu = 0;
    vv = 1;

    bitCount = 0LL;
    u0 = (int64_t)u[0];
    v0 = (int64_t)v[0];

    // Slightly optimized Binary XCD loop on native signed integers
    // Stop at 62 bits to avoid uv matrix overfow and loss of sign bit
    while (true) {

      while (IS_EVEN(u0) && (bitCount < 62)) {

        bitCount++;
        u0 >>= 1;
        vu <<= 1;
        vv <<= 1;
      }

      if (bitCount == 62)
        break;

      nb0 = (v0 + u0) & 0x3;
      if (nb0 == 0) {
        SWAP_ADD(uv, vv);
        SWAP_ADD(uu, vu);
        SWAP_ADD(u0, v0);
      } else {
        SWAP_SUB(uv, vv);
        SWAP_SUB(uu, vu);
        SWAP_SUB(u0, v0);
      }
    }

    // Now update BigInt variables

    IMult(t1, u, uu);
    IMult(t2, v, uv);
    IMult(t3, u, vu);
    IMult(t4, v, vv);

    // u = (uu*u + uv*v)
    Add2(u, t1, t2);
    // v = (vu*u + vv*v)
    Add2(v, t3, t4);

    IMult(t1, r, uu);
    IMult(t2, s, uv);
    IMult(t3, r, vu);
    IMult(t4, s, vv);

    // Compute multiple of P to add to s and r to make them multiple of 2^62
    r0 = ((t1[0] + t2[0]) * MM64) & MSK62;
    s0 = ((t3[0] + t4[0]) * MM64) & MSK62;
    // r = (uu*r + uv*s + r0*P)
    UMult(r, _P, r0);
    Add1(r, t1);
    Add1(r, t2);

    // s = (vu*r + vv*s + s0*P)
    UMult(s, _P, s0);
    Add1(s, t3);
    Add1(s, t4);

    // Right shift all variables by 62bits
    ShiftR62(u);
    ShiftR62(v);
    ShiftR62(r);
    ShiftR62(s);
  }

  // v ends with -1 or 1
  if (_IsNegative(v)) {
    // V = -1
    Sub2(s, _P, s);
    Neg(v);
  }

  if (!_IsOne(v)) {
    // No inverse
    Load(R, _0);
    return;
  }

  // In very rare case |s|>2P
  while (_IsNegative(s))
    AddP(s);
  while (!_IsNegative(s))
    SubP(s);
  AddP(s);

  Load(R, s);
}


// Shift r left by 64 bits
__device__ void ShiftL64(uint64_t *r) {

  r[4] = r[3];
  r[3] = r[2];
  r[2] = r[1];
  r[1] = r[0];
  r[0] = 0;
}

// Shift r left by 32 bits
__device__ void ShiftL32(uint32_t *r) {

  r[9] = r[8];
  r[8] = r[7];
  r[7] = r[6];
  r[6] = r[5];
  r[5] = r[4];
  r[4] = r[3];
  r[3] = r[2];
  r[2] = r[1];
  r[1] = r[0];
  r[0] = 0;
}

// an optimized mod multiplication of r = a * b mod P
// this uses a two step-bit folding from 512 -> 320 -> 224
// the constant that used to reduce the result is
// defined as (2 ^ 224) - P
// for Secp224k1 this comes out to be 0xffffffffffffffffffffffff
// since this is one away from a simple bit shift
// this gets implemented as a bit shift and a subtraction
__device__ void _ModMult(uint64_t *r, uint64_t *a, uint64_t *b) {

  uint64_t r512[8];
  uint64_t t[NBBLOCK], tt[NBBLOCK], sub[NBBLOCK];

  r512[5] = 0;
  r512[6] = 0;
  r512[7] = 0;

  // 256*256 multiplier
  UMult(r512, a, b[0]);
  UMult(t, a, b[1]);
  UADDO1(r512[1], t[0]);
  UADDC1(r512[2], t[1]);
  UADDC1(r512[3], t[2]);
  UADDC1(r512[4], t[3]);
  UADD1(r512[5], t[4]);
  UMult(t, a, b[2]);
  UADDO1(r512[2], t[0]);
  UADDC1(r512[3], t[1]);
  UADDC1(r512[4], t[2]);
  UADDC1(r512[5], t[3]);
  UADD1(r512[6], t[4]);
  UMult(t, a, b[3]);
  UADDO1(r512[3], t[0]);
  UADDC1(r512[4], t[1]);
  UADDC1(r512[5], t[2]);
  UADDC1(r512[6], t[3]);
  UADD1(r512[7], t[4]);

  // Reduce from 512 to 320
  uint32_t *r51232 = (uint32_t *)(r512);
  tt[0] = ((uint64_t) * (r51232 + 8) << 32) | (uint64_t) * (r51232 + 7);
  tt[1] = ((uint64_t) * (r51232 + 10) << 32) | (uint64_t) * (r51232 + 9);
  tt[2] = ((uint64_t) * (r51232 + 12) << 32) | (uint64_t) * (r51232 + 11);
  tt[3] = ((uint64_t) * (r51232 + 14) << 32) | (uint64_t) * (r51232 + 13);
  tt[4] = 0;
  sub[0] = tt[0];
  sub[1] = tt[1];
  sub[2] = tt[2];
  sub[3] = tt[3];
  sub[4] = tt[4];
  ShiftL64(tt);
  ShiftL32((uint32_t *)(tt));
  Sub1(tt, sub);

  UADDO1(r512[0], tt[0]);
  UADDC1(r512[1], tt[1]);
  UADDC1(r512[2], tt[2]);
  UADDC(r512[3], (uint64_t)((uint32_t)(r512[3])), (uint64_t)((uint32_t)(tt[3])));

  uint32_t *tt32 = (uint32_t *)(tt);
  UADDO(tt[0], ((uint64_t) * (tt32 + 8) << 32) | (uint64_t) * (tt32 + 7), (uint64_t)(((uint8_t *)(r512 + 3))[4]));
  UADDC(tt[1], 0ULL, (uint64_t) * (tt32 + 9));
  tt[2] = 0;
  tt[3] = 0;
  tt[4] = 0;
  sub[0] = tt[0];
  sub[1] = tt[1];
  sub[2] = tt[2];
  sub[3] = tt[3];
  sub[4] = tt[4];
  ShiftL64(tt);
  ShiftL32((uint32_t *)(tt));
  Sub1(tt, sub);

  // Reduce from 320 to 224
  UADDO(r[0], r512[0], tt[0]);
  UADDC(r[1], r512[1], tt[1]);
  UADDC(r[2], r512[2], tt[2]);
  UADDC(r[3], (uint64_t)((uint32_t)(r512[3])), tt[3]);
}

// ---------------------------------------------------------------------------------------
// _MontgomeryMult
// Compute a*b*R^-1 (mod n),  R=2^256 (mod n)
// a and b must be lower than n
// ---------------------------------------------------------------------------------------
__device__ void _MontgomeryMult(uint64_t *r, uint64_t *a, uint64_t *b) {

  uint64_t pr[NBBLOCK];
  uint64_t p[NBBLOCK];
  uint64_t t[NBBLOCK];
  uint64_t ML;
  uint64_t c;
  UMult(pr, a, b[0]);

  ML = pr[0] * MM64;
  UMult(p, _P, ML);
  AddC(pr, p, c);
  Shift64(t, pr, c);

  UMult(pr, a, b[1]);
  ML = (pr[0] + t[0]) * MM64;
  UMult(p, _P, ML);
  AddC(pr, p, c);
  AddAndShift(t, pr, t, c);

  UMult(pr, a, b[2]);
  ML = (pr[0] + t[0]) * MM64;
  UMult(p, _P, ML);
  AddC(pr, p, c);
  AddAndShift(t, pr, t, c);

  UMult(pr, a, b[3]);
  ML = (pr[0] + t[0]) * MM64;
  UMult(p, _P, ML);
  AddC(pr, p, c);
  AddAndShift(t, pr, t, c);

  //  ML = t[0] * MM64;
  //  UMult(p, _P, ML);
  //  AddAndShift(t, p, t, 0ULL);

  Sub2(p, t, _P);
  if (_IsPositive(p))
    Load256(r, p) else Load256(r, t)
}

// same as _MontgomeryMult but r= r * a mod P
__device__ void _MontgomeryMult(uint64_t *r, uint64_t *a) {

  uint64_t pr[NBBLOCK];
  uint64_t p[NBBLOCK];
  uint64_t t[NBBLOCK];
  uint64_t ML;
  uint64_t c;

  UMult(pr, a, r[0]);
  ML = pr[0] * MM64;
  MulP(p, ML);
  AddC(pr, p, c);
  Shift64(t, pr, c);

  UMult(pr, a, r[1]);
  ML = (pr[0] + t[0]) * MM64;
  MulP(p, ML);
  AddC(pr, p, c);
  AddAndShift(t, pr, t, c);

  UMult(pr, a, r[2]);
  ML = (pr[0] + t[0]) * MM64;
  MulP(p, ML);
  AddC(pr, p, c);
  AddAndShift(t, pr, t, c);

  UMult(pr, a, r[3]);
  ML = (pr[0] + t[0]) * MM64;
  MulP(p, ML);
  AddC(pr, p, c);
  AddAndShift(t, pr, t, c);

  Sub2(p, t, _P);
  if (_IsPositive(p))
    Load256(r, p) else Load256(r, t)
}

// r = r ^ 3 mod p Montgomery based
__device__ void _MontgomeryMultR3(uint64_t *r) {

  // R3 = (2^256)^3 mod p
  // R3 = { 0x002BB1E33795F671ULL, 0x0000000100000B73ULL, 0ULL, 0ULL };

  uint64_t pr[NBBLOCK];
  uint64_t p[NBBLOCK];
  uint64_t t[NBBLOCK];
  uint64_t ML;
  uint64_t c;

  UMult(pr, r, 0x002BB1E33795F671ULL);
  ML = pr[0] * MM64;
  MulP(p, ML);
  AddC(pr, p, c);
  Shift64(t, pr, c);

  UMult(pr, r, 0x0000000100000B73ULL);
  ML = (pr[0] + t[0]) * MM64;
  MulP(p, ML);
  AddC(pr, p, c);
  AddAndShift(t, pr, t, c);

  ML = t[0] * MM64;
  MulP(p, ML);
  AddC(t, p, c);
  Shift64(t, t, c);

  ML = t[0] * MM64;
  MulP(p, ML);
  AddC(t, p, c);
  Shift64(t, t, c);

  Sub2(p, t, _P);
  if (_IsPositive(p))
    Load256(r, p) else Load256(r, t)
}

__device__ void _MontgomeryMultR4(uint64_t *r) {

  // R4 = (2^256)^4 mod p
  // R4 = { 0xDE57DA9823518541ULL,0x00000F44005763C6ULL,0x1ULL,0ULL };

  uint64_t pr[NBBLOCK];
  uint64_t p[NBBLOCK];
  uint64_t t[NBBLOCK];
  uint64_t ML;
  uint64_t c;

  UMult(pr, r, 0xDE57DA9823518541ULL);
  ML = pr[0] * MM64;
  MulP(p, ML);
  AddC(pr, p, c);
  Shift64(t, pr, c);

  UMult(pr, r, 0x00000F44005763C6ULL);
  ML = (pr[0] + t[0]) * MM64;
  MulP(p, ML);
  AddC(pr, p, c);
  AddAndShift(t, pr, t, c);

  Load256(pr, r);
  pr[4] = 0ULL;
  ML = (pr[0] + t[0]) * MM64;
  MulP(p, ML);
  AddC(pr, p, c);
  AddAndShift(t, pr, t, c);

  ML = t[0] * MM64;
  MulP(p, ML);
  AddC(t, p, c);
  Shift64(t, t, c);

  Sub2(p, t, _P);
  if (_IsPositive(p))
    Load256(r, p) else Load256(r, t)
}

// ---------------------------------------------------------------------------------------
// Compute all ModInv of the group
// ---------------------------------------------------------------------------------------
__device__ void _ModInvGrouped(uint64_t r[GRP_SIZE / 2 + 1][4]) {

  uint64_t subp[GRP_SIZE / 2 + 1][4];
  uint64_t newValue[4];
  uint64_t inverse[5];

  // Number of _MontgomeryMult must be equal before and after the inversion of each item
  // in order that power of R vanish

  Load256(subp[0], r[0]);
  for (uint32_t i = 1; i < (GRP_SIZE / 2 + 1); i++) {
    _ModMult(subp[i], subp[i - 1], r[i]);
  }

  // We need 320bit signed int for ModInv
  Load256(inverse, subp[(GRP_SIZE / 2 + 1) - 1]);
  inverse[4] = 0;
  _ModInv(inverse);

  for (uint32_t i = (GRP_SIZE / 2 + 1) - 1; i > 0; i--) {
    _ModMult(newValue, subp[i - 1], inverse);
    _ModMult(inverse, r[i]);
    Load256(r[i], newValue);
  }

  Load256(r[0], inverse);
}

//__device__ void _ModMult(uint64_t *r, uint64_t *a, uint64_t *b) {
//  uint64_t p[NBBLOCK];
//  _MontgomeryMult(p, a, b);
//  _MontgomeryMult(r, (uint64_t *)&_R2, (uint64_t *)&p);
//}

__device__ void _ModMult(uint64_t *r, uint64_t *a) { _ModMult(r, r, a); }

__device__ void _ModSqr(uint64_t *rp, uint64_t *up) { _ModMult(rp, up, up); }
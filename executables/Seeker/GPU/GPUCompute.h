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

// CUDA Kernel main function
// Compute Secp224R1 keys and then check prefix
// For the kernel, we use a 16 bits prefix lookup table which correspond to 2 bytes prefix for an uncompressed x
// coordinate A second level lookup table contains 32 bits prefix (if used) (The CPU computes the full address and check
// the full prefix)
//
// We use affine coordinates for elliptic curve point (ie Z=1)

// Check that the x coordinate matches one of the prefixes that we are lookings for
__device__ __noinline__ void CheckPointPub(
  uint16_t *px,
  int32_t incr,
  int32_t endo,
  int32_t mode,
  prefix_t *prefix,
  uint32_t *lookup32,
  uint32_t maxFound,
  uint32_t *out,
  int type) {

  uint32_t off;
  prefixl_t l32;
  prefix_t pr0;
  prefix_t hit;
  uint32_t pos;
  uint32_t st;
  uint32_t ed;
  uint32_t mi;
  uint32_t lmi;
  uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  uint32_t *p32x;

  {
    // If prefix is NULL, bypass prefix checking and make hit always true
    if (prefix == NULL) {
      hit = true;
    } else {
      // Normal prefix lookup
      pr0 = px[13];
      hit = prefix[pr0];
    }

    if (hit) {
      //      if (lookup32) {
      //        off = lookup32[pr0];
      //        l32 = px[13];
      //        st = off;
      //        ed = off + hit - 1;
      //        while (st <= ed) {
      //          mi = (st + ed) / 2;
      //          lmi = lookup32[mi];
      //          if (l32 < lmi) {
      //            ed = mi - 1;
      //          } else if (l32 == lmi) {
      //            // found
      //            goto addItem;
      //          } else {
      //            st = mi + 1;
      //          }
      //        }
      //        return;
      //      }
      pos = atomicAdd(out, 1);
      if (pos < maxFound) {
        p32x = (uint32_t *)(px);
        out[pos * ITEM_SIZE32 + 1] = tid;
        out[pos * ITEM_SIZE32 + 2] = (uint32_t)(incr << 16) | (uint32_t)(mode << 15) | (uint32_t)(endo);
        out[pos * ITEM_SIZE32 + 3] = p32x[0];
        out[pos * ITEM_SIZE32 + 4] = p32x[1];
        out[pos * ITEM_SIZE32 + 5] = p32x[2];
        out[pos * ITEM_SIZE32 + 6] = p32x[3];
        out[pos * ITEM_SIZE32 + 7] = p32x[4];
        out[pos * ITEM_SIZE32 + 8] = p32x[5];
        out[pos * ITEM_SIZE32 + 9] = p32x[6];
        out[pos * ITEM_SIZE32 + 10] = p32x[7];
        out[pos * ITEM_SIZE32 + 11] = 0;
        out[pos * ITEM_SIZE32 + 12] = 0;
      }
    }
  }
}


#define CHECK_POINT_PUB(px, incr, endo, mode) CheckPointPub(px, incr, endo, mode, prefix, lookup32, maxFound, out, PUB)

// Public Key Check for Secp224r1
// -----------------------------------------------------------------------------------------
__device__ __noinline__ void CheckPublicKey(
  prefix_t *prefix,
  uint64_t *px,
  uint64_t *py,
  int32_t incr,
  uint32_t *lookup32,
  uint32_t maxFound,
  uint32_t *out) {

  CHECK_POINT_PUB((uint16_t *)px, incr, 0, false);
}

// -----------------------------------------------------------------------------------------
// Check the prefix of a given x and y
__device__ __noinline__ void CheckPrefix(
  uint32_t mode,
  prefix_t *prefix,
  uint64_t *px,
  uint64_t *py,
  int32_t incr,
  uint32_t *lookup32,
  uint32_t maxFound,
  uint32_t *out) {

  CheckPublicKey(prefix, px, py, incr, lookup32, maxFound, out);
}

// marco to help reduce parameters
#define CHECK_PREFIX(incr) CheckPrefix(mode, sPrefix, px, py, j *GRP_SIZE + (incr), lookup32, maxFound, out)

// -----------------------------------------------------------------------------------------
// Compute the x and y coordinates given a starting point
// the amount of points computed is based on GRP_SIZE
__device__ void ComputeKeys(
  uint32_t mode,
  uint64_t *startx,
  uint64_t *starty,
  prefix_t *sPrefix,
  uint32_t *lookup32,
  uint32_t maxFound,
  uint32_t *out) {

  uint64_t dx[GRP_SIZE / 2 + 1][4];
  uint64_t px[4];
  uint64_t py[4];
  uint64_t pyn[4];
  uint64_t sx[4];
  uint64_t sy[4];
  uint64_t dy[4];
  uint64_t _s[4];
  uint64_t _p2[4];
  char pattern[48];

  // Load starting key
  __syncthreads();
  Load256A(sx, startx);
  Load256A(sy, starty);
  Load256(px, sx);
  Load256(py, sy);

  if (sPrefix == NULL) {
    memcpy(pattern, lookup32, 48);
    lookup32 = (uint32_t *)pattern;
  }

  for (uint32_t j = 0; j < STEP_SIZE / GRP_SIZE; j++) {

    // Fill group with delta x
    uint32_t i;
    for (i = 0; i < HSIZE; i++)
      ModSub256(dx[i], Gx[i], sx);
    ModSub256(dx[i], Gx[i], sx);     // For the first point
    ModSub256(dx[i + 1], _2Gnx, sx); // For the next center point

    // Compute modular inverse
    _ModInvGrouped(dx);

    // We use the fact that P + i*G and P - i*G has the same deltax, so the same inverse
    // We compute key in the positive and negative way from the center of the group

    // Check starting point
    CHECK_PREFIX(GRP_SIZE / 2);

    ModNeg256(pyn, py);

    for (i = 0; i < HSIZE; i++) {

      // P = StartPoint + i*G
      Load256(px, sx);
      Load256(py, sy);
      ModSub256(dy, Gy[i], py);

      _ModMult(_s, dy, dx[i]); //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
      _ModSqr(_p2, _s);        // _p2 = pow2(s)

      ModSub256(px, _p2, px);
      ModSub256(px, Gx[i]); // px = pow2(s) - p1.x - p2.x;

      ModSub256(py, Gx[i], px);
      _ModMult(py, _s);     // py = - s*(ret.x-p2.x)
      ModSub256(py, Gy[i]); // py = - p2.y - s*(ret.x-p2.x);

      CHECK_PREFIX(GRP_SIZE / 2 + (i + 1));

      // P = StartPoint - i*G, if (x,y) = i*G then (x,-y) = -i*G
      Load256(px, sx);
      ModSub256(dy, pyn, Gy[i]);

      _ModMult(_s, dy, dx[i]); //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
      _ModSqr(_p2, _s);        // _p = pow2(s)

      ModSub256(px, _p2, px);
      ModSub256(px, Gx[i]); // px = pow2(s) - p1.x - p2.x;

      ModSub256(py, px, Gx[i]);
      _ModMult(py, _s);         // py = s*(ret.x-p2.x)
      ModSub256(py, Gy[i], py); // py = - p2.y - s*(ret.x-p2.x);

      CHECK_PREFIX(GRP_SIZE / 2 - (i + 1));
    }

    // First point (startP - (GRP_SZIE/2)*G)
    Load256(px, sx);
    Load256(py, sy);
    ModNeg256(dy, Gy[i]);
    ModSub256(dy, py);

    _ModMult(_s, dy, dx[i]); //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
    _ModSqr(_p2, _s);        // _p = pow2(s)

    ModSub256(px, _p2, px);
    ModSub256(px, Gx[i]); // px = pow2(s) - p1.x - p2.x;

    ModSub256(py, px, Gx[i]);
    _ModMult(py, _s);         // py = s*(ret.x-p2.x)
    ModSub256(py, Gy[i], py); // py = - p2.y - s*(ret.x-p2.x);

    CHECK_PREFIX(0);

    i++;

    // Next start point (startP + GRP_SIZE*G)
    Load256(px, sx);
    Load256(py, sy);
    ModSub256(dy, _2Gny, py);

    _ModMult(_s, dy, dx[i]); //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
    _ModSqr(_p2, _s);        // _p2 = pow2(s)

    ModSub256(px, _p2, px);
    ModSub256(px, _2Gnx); // px = pow2(s) - p1.x - p2.x;

    ModSub256(py, _2Gnx, px);
    _ModMult(py, _s);     // py = - s*(ret.x-p2.x)
    ModSub256(py, _2Gny); // py = - p2.y - s*(ret.x-p2.x);
  }

  // Update starting point
  __syncthreads();
  Store256A(startx, px);
  Store256A(starty, py);
}
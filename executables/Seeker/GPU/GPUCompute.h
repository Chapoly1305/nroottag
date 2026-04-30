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
// For the kernel, we use a 16-bit prefix lookup table for the first two bytes of the uncompressed x coordinate.
// When lookup32 is present, each 16-bit bucket contains value/mask pairs for exact GPU-side filtering of the
// remaining requested bytes. CPU validation still recomputes and checks GPU-filtered candidates before output.
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
  uint8_t *p8x;
  uint32_t remainingPrefix;

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
      if (lookup32) {
        off = lookup32[pr0];
        p8x = (uint8_t *)px;
        remainingPrefix =
          ((uint32_t)p8x[25] << 24) |
          ((uint32_t)p8x[24] << 16) |
          ((uint32_t)p8x[23] << 8) |
          ((uint32_t)p8x[22]);

        for (uint32_t idx = 0; idx < hit; idx++) {
          l32 = lookup32[off + idx * 2];
          lmi = lookup32[off + idx * 2 + 1];
          if (((remainingPrefix ^ l32) & lmi) == 0)
            goto addItem;
        }
        return;
      }

    addItem:
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
// Window size for Montgomery batch inversion inside each CUDA thread.
// This is not tied to SM count or warp-lane mapping. It trades fewer _ModInv()
// calls against larger per-thread local arrays:
//   dx + temp = 2 * INV_WINDOW * 4 * sizeof(uint64_t) = 64 * INV_WINDOW bytes.
// RTX 4090 tuning showed 16 under-amortizes inversions, while 64/96/128 grow
// the local stack enough to hurt cache/occupancy; 32 was the best observed
// balance with GRP_SIZE=1024 and -g 4096,384.
#define INV_WINDOW 32

__device__ void _ModInvWindowed(uint64_t r[INV_WINDOW][4], uint64_t temp[INV_WINDOW][4], uint32_t count) {

  uint64_t inverse[NBBLOCK];

  Load256(temp[0], r[0]);
  for (uint32_t i = 1; i < count; i++)
    _ModMult(temp[i], temp[i - 1], r[i]);

  Load256(inverse, temp[count - 1]);
  inverse[4] = 0;
  _ModInv(inverse);

  for (int32_t i = (int32_t)count - 1; i > 0; i--) {
    uint64_t newValue[4];
    _ModMult(newValue, temp[i - 1], inverse);
    _ModMult(inverse, r[i]);
    Load256(r[i], newValue);
  }

  Load256(r[0], inverse);
}

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

  uint64_t dx[INV_WINDOW][4];
  uint64_t temp[INV_WINDOW][4];
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

    // We use the fact that P + i*G and P - i*G has the same deltax, so the same inverse
    // We compute key in the positive and negative way from the center of the group

    // Check starting point
    CHECK_PREFIX(GRP_SIZE / 2);

    ModNeg256(pyn, py);

    for (uint32_t base = 0; base < HSIZE; base += INV_WINDOW) {

      uint32_t windowSize = INV_WINDOW;
      if (base + windowSize > HSIZE)
        windowSize = HSIZE - base;

      for (uint32_t k = 0; k < windowSize; k++)
        ModSub256(dx[k], Gx[base + k], sx);

      _ModInvWindowed(dx, temp, windowSize);

      for (uint32_t k = 0; k < windowSize; k++) {

        uint32_t i = base + k;

        // P = StartPoint + i*G
        Load256(px, sx);
        Load256(py, sy);
        ModSub256(dy, Gy[i], py);

        _ModMult(_s, dy, dx[k]); //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
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

        _ModMult(_s, dy, dx[k]); //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
        _ModSqr(_p2, _s);        // _p = pow2(s)

        ModSub256(px, _p2, px);
        ModSub256(px, Gx[i]); // px = pow2(s) - p1.x - p2.x;

        ModSub256(py, px, Gx[i]);
        _ModMult(py, _s);         // py = s*(ret.x-p2.x)
        ModSub256(py, Gy[i], py); // py = - p2.y - s*(ret.x-p2.x);

        CHECK_PREFIX(GRP_SIZE / 2 - (i + 1));
      }
    }

    ModSub256(dx[0], Gx[HSIZE], sx); // For the first point
    ModSub256(dx[1], _2Gnx, sx);     // For the next center point
    _ModInvWindowed(dx, temp, 2);

    // First point (startP - (GRP_SZIE/2)*G)
    Load256(px, sx);
    Load256(py, sy);
    ModNeg256(dy, Gy[HSIZE]);
    ModSub256(dy, py);

    _ModMult(_s, dy, dx[0]); //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
    _ModSqr(_p2, _s);        // _p = pow2(s)

    ModSub256(px, _p2, px);
    ModSub256(px, Gx[HSIZE]); // px = pow2(s) - p1.x - p2.x;

    ModSub256(py, px, Gx[HSIZE]);
    _ModMult(py, _s);         // py = s*(ret.x-p2.x)
    ModSub256(py, Gy[HSIZE], py); // py = - p2.y - s*(ret.x-p2.x);

    CHECK_PREFIX(0);

    // Next start point (startP + GRP_SIZE*G)
    Load256(px, sx);
    Load256(py, sy);
    ModSub256(dy, _2Gny, py);

    _ModMult(_s, dy, dx[1]); //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
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

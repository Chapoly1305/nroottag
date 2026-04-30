# Seeker CUDA Performance Revision

## Scope and corrections

This document supersedes the earlier analysis. After deeper investigation, several
initial proposals were found to be incorrect or inapplicable.

### Withdrawn proposals (with reasons)

| Proposal | Why withdrawn |
|----------|--------------|
| GLV endomorphism `φ(x,y) = (βx, y)` | SECP224R1 does **not** have complex multiplication. j-invariant = `0xc5563016...` is not a CM value. The cube roots of unity β, λ exist in F_p and F_n respectively, but do **not** define a curve endomorphism when a ≠ 0. The developers' comment "secp224r1 does not have any endomorphism" (Vanity.cpp:275) is correct. |
| Switch to Montgomery multiplication | For SECP224R1's prime (P = 2^224 − 2^96 + 1), the two-step folding reduction in `_ModMult` exploits the fact that 2^224 − P = 2^96 − 1, making reduction extremely cheap. Montgomery does 4 full rounds with no comparable shortcut. The current approach is optimal for this prime. |
| Change memory layout for coalescing | The current `Load256A`/`Store256A` pattern loads limb `k` from a contiguous range of `blockDim.x` elements — perfectly coalesced. The AoS-within-group layout was designed for this. |
| Manually hoist `Load256(px, sx)` | The compiler keeps `sx` (4 uint64_t) in registers. The "reload" is a register-to-register copy. Not a bottleneck. |

---

## Proposal 1 — Remove unnecessary `__syncthreads()` barriers

**Location:** `GPU/GPUCompute.h` lines 159 and 263, inside `ComputeKeys()`.

**Change:** Delete both `__syncthreads()` calls.

**Before:**
```c
  // line 158-161
  __syncthreads();
  Load256A(sx, startx);
  Load256A(sy, starty);
```
```c
  // line 262-265
  __syncthreads();
  Store256A(startx, px);
  Store256A(starty, py);
```

**After:**
```c
  Load256A(sx, startx);
  Load256A(sy, starty);
```
```c
  Store256A(startx, px);
  Store256A(starty, py);
```

**Reasoning:** `Load256A` and `Store256A` access only global memory using
per-thread offsets (`threadIdx.x`). Each thread reads and writes disjoint
addresses. There is zero shared-memory access and zero inter-thread data
dependency at either site. Each barrier wastes ~30–60 cycles.

**Expected impact:** ~1–3% kernel throughput improvement. Zero risk.

---

## Proposal 2 — Enable read-only data cache for the prefix lookup table

**Location:**
- Kernel declarations: `GPU/GPUEngine.cu` lines 59–74
- `CheckPointPub`: `GPU/GPUCompute.h` line 57
- `CheckPrefix` / `CheckPublicKey` / `ComputeKeys` signatures

**Change:**

**Step A** — Add `const __restrict__` to kernel pointer parameters:

```c
__global__ void
comp_keys(uint32_t mode,
          const prefix_t * __restrict__ prefix,
          const uint32_t * __restrict__ lookup32,
          uint64_t * __restrict__ keys,
          uint32_t maxFound,
          uint32_t * __restrict__ found) {
```

Do the same for `comp_keys_pattern`.

**Step B** — Propagate `const` through the call chain (`CheckPublicKey`,
`CheckPrefix`, `ComputeKeys` — all take `const prefix_t * __restrict__ sPrefix`).

**Step C** — Use `__ldg()` for the actual load:

```c
// line 56-57 — was:
      pr0 = px[13];
      hit = prefix[pr0];
// becomes:
      pr0 = px[13];
      hit = __ldg(&prefix[pr0]);
```

**Reasoning:** The `prefix` table is 128 KB (65536 entries × uint16_t), accessed
with a **random index** 1024 times per thread per kernel call (once per computed
key). Without `const __restrict__`, NVCC cannot prove the pointer doesn't alias
with `keys`, `lookup32`, or `found` — forcing all `prefix[pr0]` loads through
the generic L1 cache path. Each 2-byte read pulls a full 128-byte cache line,
wasting 98% of the fetched data and thrashing L1.

`const __restrict__` + `__ldg()` routes these loads through the **read-only
data cache** (separate 32–48 KB, ~50 cycle latency vs ~80 for L1 / ~200+ for
L2). The RO cache uses an eviction policy suited to random-read workloads and
is not polluted by writes to other pointers.

**Expected impact:** 5–15% kernel throughput improvement. Zero risk.

---

## Proposal 3 — Re-enable second-level prefix lookup (32-bit GPU-side filter)

**Location:** `GPU/GPUCompute.h` lines 60–79, the commented-out `lookup32` block.

**Current state:** The code contains a fully-designed but commented-out
second-level prefix lookup using binary search on a 32-bit prefix table.
This would allow up to **48 bits of GPU-side filtering** (16-bit first level
+ 32-bit second level). Currently only the 16-bit first level is active,
meaning the GPU reports matches on the first 2 bytes and the CPU validates the
full prefix. For a 6-byte prefix search, this produces ~16 false positives per
~1M keys, each consuming CPU time.

**Proposed change:** Uncomment and fix the second-level lookup.

The commented-out code at lines 61–79:
```c
//      if (lookup32) {
//        off = lookup32[pr0];
//        l32 = px[13];              // BUG: should read a different 32-bit slice
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
```

Two bugs to fix:
1. `l32 = px[13]` reads the same 16-bit slice as `pr0`. Should instead read a
   different portion of the x-coordinate — e.g., 32 bits from the middle of the
   224-bit value: `l32 = p32x[4]` (bits 128–159 of the x-coordinate). This
   gives an independent 32-bit hash, providing 48 total bits of discrimination.
2. The `lookup32` table needs to be structured as: for each 16-bit prefix bucket,
   store a sorted list of 32-bit sub-prefixes. The `hit` value (currently a
   boolean) needs to be changed to a count for binary search bounds.

After fixing:
```c
    if (hit) {
      if (lookup32) {
        // Second-level 32-bit filter
        uint32_t off = lookup32[pr0];            // start index in level-2 table
        uint32_t count = lookup32[pr0 + 1] - off; // entry count (requires adjusted table layout)
        uint32_t l32 = p32x[4];                   // bits 128-159 of x-coordinate
        uint32_t lo = off;
        uint32_t hi = off + count;
        while (lo < hi) {
          uint32_t mi = (lo + hi) / 2;
          uint32_t lmi = lookup32[2 + mi];        // 2-entry header per bucket
          if (l32 < lmi) {
            hi = mi;
          } else if (l32 == lmi) {
            goto addItem;
          } else {
            lo = mi + 1;
          }
        }
        return;  // 16-bit match but 32-bit mismatch — false positive filtered on GPU
      }
      // addItem: (existing code)
      pos = atomicAdd(out, 1);
      ...
    }
```

The table format change requires updating `GPUEngine::SetPrefix` (the
`LPREFIX` overload) to build the combined 16+32 bit lookup structure. The
`inputPrefixLookUp` allocation at line 449 of GPUEngine.cu already exists
and is wired through — it's just the kernel-side consumer that's disabled.

**Expected impact:** Eliminates CPU validation overhead for prefixes up to
~48 bits. For a 6-byte prefix, GPU false-positive rate drops from 1/65536
to ~1/2^48, meaning essentially zero CPU validation traffic. The binary
search cost is only paid on 16-bit matches (1 in 65536), so the average
overhead per key is negligible.

**Risk:** Requires fixing the table layout in `GPUEngine::SetPrefix` to match
the new lookup structure. This is a data-structure change, not a cryptographic
change — correctness is easy to verify by running `./Seeker -check`.

---

## Proposal 4 — Sliding-window batch inversion (reduce local memory 16×)

**Location:** `_ModInvGrouped` and the `dx` array in `GPUCompute.h`.

### Problem

Each thread allocates `dx[513][4]` (16.4 KB) and `_ModInvGrouped` internally
allocates `subp[513][4]` (another 16.4 KB). Total: **~33 KB local memory per
thread**. With 128 threads/block = **~4 MB per block** — far exceeding the L1
cache (128 KB/SM on Ada Lovelace). This is the primary occupancy killer.

### Solution

Split the single 513-element batch inversion into **8 windows of 64 elements
each** (plus one window of 2 for the tail). Each window is batch-inverted
independently. Window-local storage drops to `2 × 64 × 4 × 8 = 4 KB`.

```
                    Current (W=513)     Proposed (W=64)
─────────────────────────────────────────────────────────
dx local mem        16.4 KB             0.5 KB  (32× less)
subp local mem      16.4 KB             0.5 KB  (32× less)
Total local/thread  ~33 KB              ~2 KB   (16× less)
Total local/block   ~4 MB               ~256 KB
# of _ModInv()      1                   10 (8 windows + 2 tail)
# of _ModMult()     ~1026               ~1026  (identical total)
```

The total number of modular multiplications is **unchanged** — the product
scan is the same number of operations, just split across windows. The only
extra cost is 9 additional `_ModInv()` calls (~3000–5000 cycles each ≈
27K–45K extra cycles per thread). This is traded against:

- 16× less local memory traffic
- Higher occupancy (more warps to hide latency)
- Less L1/L2 thrashing
- The inner loop (`ModSub256` + point addition + prefix check) runs identically

### Implementation

Add a configurable window constant:

```c
#define INV_WINDOW 64
```

Add a windowed inversion helper (replaces `_ModInvGrouped`):

```c
__device__ void _ModInvWindowed(
    uint64_t r[][4],     // in/out: values to invert
    uint64_t temp[][4],  // scratch space, same dimensions as r
    uint32_t count)      // number of elements (<= INV_WINDOW or 2)
{
    uint64_t inverse[5];

    // Forward product scan
    Load256(temp[0], r[0]);
    for (uint32_t i = 1; i < count; i++)
        _ModMult(temp[i], temp[i - 1], r[i]);

    // Single modular inverse
    Load256(inverse, temp[count - 1]);
    inverse[4] = 0;
    _ModInv(inverse);

    // Backward scan
    for (uint32_t i = count - 1; i > 0; i--) {
        uint64_t newValue[4];
        _ModMult(newValue, temp[i - 1], inverse);
        _ModMult(inverse, r[i]);
        Load256(r[i], newValue);
    }
    Load256(r[0], inverse);
}
```

Restructure `ComputeKeys` to process windows:

```c
__device__ void ComputeKeys(...) {
    // Dramatically reduced local arrays
    uint64_t dx[INV_WINDOW][4];      // was [513][4]
    uint64_t temp[INV_WINDOW][4];    // was subp[513][4]
    uint64_t px[4], py[4], pyn[4];
    uint64_t sx[4], sy[4];
    uint64_t dy[4], _s[4], _p2[4];
    char pattern[48];

    Load256A(sx, startx);
    Load256A(sy, starty);
    Load256(px, sx);
    Load256(py, sy);

    if (sPrefix == NULL) {
        memcpy(pattern, lookup32, 48);
        lookup32 = (uint32_t *)pattern;
    }

    // Check center point
    CHECK_PREFIX(GRP_SIZE / 2);
    ModNeg256(pyn, py);

    // Process in windows of INV_WINDOW
    for (uint32_t base = 0; base < HSIZE; base += INV_WINDOW) {
        uint32_t wEnd = min(base + INV_WINDOW, (uint32_t)HSIZE);
        uint32_t W = wEnd - base;

        // Fill this window's dx values: dx[i] = Gx[base+i] - sx
        for (uint32_t i = 0; i < W; i++)
            ModSub256(dx[i], Gx[base + i], sx);

        // Batch-invert this window
        _ModInvWindowed(dx, temp, W);

        // Compute points P+(base+i+1)*G and P-(base+i+1)*G
        for (uint32_t i = 0; i < W; i++) {
            // P + (base+i+1)*G
            Load256(px, sx);
            Load256(py, sy);
            ModSub256(dy, Gy[base + i], py);
            _ModMult(_s, dy, dx[i]);
            _ModSqr(_p2, _s);
            ModSub256(px, _p2, px);
            ModSub256(px, Gx[base + i]);
            ModSub256(py, Gx[base + i], px);
            _ModMult(py, _s);
            ModSub256(py, Gy[base + i]);
            CHECK_PREFIX(GRP_SIZE / 2 + (base + i + 1));

            // P - (base+i+1)*G
            Load256(px, sx);
            ModSub256(dy, pyn, Gy[base + i]);
            _ModMult(_s, dy, dx[i]);
            _ModSqr(_p2, _s);
            ModSub256(px, _p2, px);
            ModSub256(px, Gx[base + i]);
            ModSub256(py, px, Gx[base + i]);
            _ModMult(py, _s);
            ModSub256(py, Gy[base + i], py);
            CHECK_PREFIX(GRP_SIZE / 2 - (base + i + 1));
        }
    }

    // Tail: dx[HSIZE] and dx[HSIZE+1] (invert 2 elements)
    ModSub256(dx[0], Gx[HSIZE], sx);
    ModSub256(dx[1], _2Gnx, sx);
    _ModInvWindowed(dx, temp, 2);

    // First point: P - GRP_SIZE/2*G
    {
        Load256(px, sx); Load256(py, sy);
        ModNeg256(dy, Gy[HSIZE]); ModSub256(dy, py);
        _ModMult(_s, dy, dx[0]); _ModSqr(_p2, _s);
        ModSub256(px, _p2, px); ModSub256(px, Gx[HSIZE]);
        ModSub256(py, px, Gx[HSIZE]); _ModMult(py, _s);
        ModSub256(py, Gy[HSIZE], py);
        CHECK_PREFIX(0);
    }

    // Next starting point: P + GRP_SIZE*G
    {
        Load256(px, sx); Load256(py, sy);
        ModSub256(dy, _2Gny, py);
        _ModMult(_s, dy, dx[1]); _ModSqr(_p2, _s);
        ModSub256(px, _p2, px); ModSub256(px, _2Gnx);
        ModSub256(py, _2Gnx, px); _ModMult(py, _s);
        ModSub256(py, _2Gny);
    }

    Store256A(startx, px);
    Store256A(starty, py);
}
```

### Tradeoff analysis

The 9 extra modular inverses cost ~27K–45K cycles, but the inner loop (which
dominates runtime) is **unchanged**. The benefit is from occupancy: if
occupancy doubles (from ~25% to ~50%), twice as many warps are available to
hide the latency of constant-cache reads and prefix-table lookups.

For a kernel bottlenecked by memory latency (constant cache + L2), higher
occupancy directly translates to higher throughput. The extra inversions are
pure arithmetic that runs in otherwise-idle execution slots.

### Measurement protocol

```bash
for W in 32 64 128 256; do
    # Set INV_WINDOW, rebuild, benchmark
    make clean && make gpu=1 CCAP=89 ... all
    ./Seeker -t 0 -gpu -g 1024,128 -p deadbe 2>&1 | grep "Key"
done
```

### Risk

Medium — this is the most invasive code change. The windowing must correctly
handle all boundary conditions (HSIZE = 511, tail window of 2 elements).
Validation: run `./Seeker -check` to verify GPU results match CPU, and test
with known prefixes.

---

## Proposal 5 — Occupancy tuning via `__launch_bounds__` and register cap

**Location:** `GPU/GPUEngine.cu` line 59 and `Makefile` line 58.

**Change:**

Add `__launch_bounds__` to the kernel:
```c
__global__ __launch_bounds__(128, 4)
void comp_keys(...)
```

Set a register cap in the Makefile (value determined by benchmarking):
```makefile
-maxrregcount=80
```

**Reasoning:** The kernel's massive local arrays force heavy register usage
for scalar temporaries. Capping registers improves occupancy. The
`__launch_bounds__` annotation tells the compiler the expected block size
(128) and minimum blocks per SM (4), helping it make informed register-
allocation decisions.

On Ada Lovelace (CC 8.9, 65536 registers/SM):

| maxrregcount | Max warps/SM | Occupancy |
|-------------:|-------------:|----------:|
| 64           | 32           | ~67%      |
| 80           | 24           | ~50%      |
| 96           | 20           | ~42%      |
| 128          | 16           | ~33%      |
| unlimited    | 12–16        | ~25–33%   |

**Expected impact:** 10–30% with the right cap. Requires benchmarking.

**Risk:** Wrong value degrades performance. This is GPU-specific and must be
measured per target architecture.

---

## Proposal 6 — Remove `__noinline__` from `CheckPointPub`

**Location:** `GPU/GPUCompute.h` line 27.

**Change:** Remove `__noinline__`, or replace with `__forceinline__`.

**Reasoning:** `CheckPointPub` is called ~2550 times per thread per kernel
call (via `CHECK_PREFIX` at 5 call sites × 511 iterations). With
`__noinline__`, each call has full function-call overhead — save/restore
registers, stack-frame setup. The common path (no match) is just:
```c
pr0 = px[13];
hit = prefix[pr0];
if (!hit) return;  // 99.998% of the time
```
3 instructions whose cost is dwarfed by call overhead.

Inlining eliminates ~2550 call sequences per thread. The tradeoff is higher
register pressure in `ComputeKeys`. If Proposal 5 (register cap) is applied,
inlining may push some scalars to local memory. Test with and without.

**Expected impact:** Moderate (5–10%), but must be tested in combination with
Proposal 5. If register pressure is already critical, leave `__noinline__` in
place.

---

## Proposal 7 — Flatten vestigial outer loop

**Location:** `GPU/GPUCompute.h` lines 170–260.

**Change:** Remove the `for (uint32_t j = 0; j < STEP_SIZE / GRP_SIZE; j++)`
loop since `STEP_SIZE == GRP_SIZE == 1024`, making it a single-iteration loop.

**Reasoning:** Simplifies the IR seen by the optimizer. The `j * GRP_SIZE`
multiplier inside `CHECK_PREFIX` is always zero. Removing the loop structure
may improve register allocation and inlining decisions in combination with
Proposals 5 and 6.

**Risk:** None.

---

## Proposal 8 — Build: emit PTX/SASS for inspection

**Location:** `Makefile` line 58.

**Change:** Add `--keep --keep-dir=$(OBJDIR)/ptx` to NVCC flags.

**Reasoning:** The `--ptxas-options=-v` flag already prints register usage.
Adding `--keep` preserves the intermediate files so you can directly inspect:
- Per-function register count
- Per-function local-memory allocation (lmem)
- Whether `CheckPointPub` was inlined
- Exact instruction count for the inner loop

This is essential for validating Proposals 4–7. Example of what to look for:
```
ptxas info: Used 96 registers, 16384 bytes lmem, 0 bytes smem
```

---

## Dependency ordering

```
Proposal 1  (sync barriers)      ← independent, do immediately
Proposal 7  (flatten j loop)     ← independent, do immediately
Proposal 8  (emit PTX)           ← independent, helps diagnose others
Proposal 2  (const __restrict__) ← independent, compile-time only
Proposal 6  (inline CheckPointPub) ← do after Proposal 5 for register mgmt
Proposal 5  (register tuning)    ← requires benchmarking
Proposal 3  (second-level lookup) ← independent data-structure change
Proposal 4  (sliding window)     ← do LAST, requires most validation
```

## Summary of expected impact

| Proposal | Expected gain | Certainty | Effort |
|----------|--------------|-----------|--------|
| 1  Remove sync barriers        | ~1–3%    | Certain   | Trivial |
| 2  Read-only cache for prefix  | ~5–15%   | High      | Low |
| 3  Second-level prefix lookup  | Variable* | High     | Medium |
| 4  Sliding-window inversion    | ~20–50%  | Medium    | High |
| 5  Register tuning             | ~10–30%  | Medium    | Low (needs benchmark) |
| 6  Inline CheckPointPub        | ~5–10%   | Medium    | Trivial |
| 7  Flatten vestigial loop      | ~0–2%    | Certain   | Trivial |

\* Proposal 3 eliminates CPU validation overhead. For short prefixes (≤2 bytes)
it has no effect. For 4–6 byte prefixes running against many CPU threads, it
eliminates a significant bottleneck.

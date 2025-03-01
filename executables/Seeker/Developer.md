# Design Overview
1. Initial Setup (CPU):
```
a. Generate/Set Base Private Keys (one per GPU thread)
   For each GPU thread i:
   - Either random: keys[i].Rand(224)
   - Or from start key: keys[i].Set(&startKey) + thread offsets
   - Each thread gets unique key space by adding offsets:
     * Thread offset: shifts left 80 bits
     * GPU ID offset: shifts left 112 bits

    For instance, RTX 4090 may use Grid(4096x512), means:
    - Grid Size: 4096 blocks x 512 threads per block
    - Total Threads = 4096 * 512 = 2,097,152 threads

    Each thread:
    - Has its own unique starting point
    - Works on its own sequence of points:
    P -> P±512G -> P+1024G -> etc.
    - Gets unique thread ID = (blockIdx.x * blockDim.x) + threadIdx.x
    Where:
    - blockIdx.x ranges from 0 to 4095
    - blockDim.x = 512
    - threadIdx.x ranges from 0 to 511

    Example thread IDs:
    - Block 0, Thread 0: (0 * 512) + 0 = 0
    - Block 0, Thread 1: (0 * 512) + 1 = 1
    - Block 1, Thread 0: (1 * 512) + 0 = 512
    - Block 4095, Thread 511: (4095 * 512) + 511 = 2,097,151

b. Convert to Base Public Keys
   For each thread:
   - startP[i] = secp->ComputePublicKey(&basePrivateKey[i])
   - Each thread works with its own base point

c. Prepare GPU Data
   - Copy thread's public key coordinates to GPU memory
   - Set up prefix lookup tables
   - Configure search parameters
```

2. Transfer to GPU:
```
- x,y coordinates of base public key
- Prefix lookup tables
- Search configuration
```

3. GPU Computation (ComputeKeys):
```
a. Configuration:
   - GRP_SIZE = 1024 points per batch
   - Each batch centers around point P
   - n = GRP_SIZE/2 = 512

b. Point Generation Per Batch:
                P
                |
    -512G ... -2G -G  +G +2G ... +511G
    |____________|  |   |___________|
      Negative      P    Positive
       Half              Half

c. Processing:
   - For center point P: Check prefix
   - Loop i from 1 to 512:
     * Check P + iG (positive side)
     * Check P - iG (negative side)
   - If match found, store:
     * Thread ID
     * Increment value (±i)
     * Point coordinates

d. Next Batch:
   - New P = Current P + 1024G
   - Ensures continuous coverage:
     Batch 1: P±512G  = [P-512G to P+511G]
     Batch 2: (P+1024G)±512G = [P+512G to P+1535G]
     Batch 3: (P+2048G)±512G = [P+1536G to P+2559G]
```

Key points:
- Each batch processes 1024 points symmetrically around P
- Next batch jumps 1024G forward to avoid gaps
- GPU computes points in parallel using point addition
- Each match returns how many G were added/subtracted from base point


4. Transfer back to CPU when match found:
```
- Thread ID that found the match
- Increment value
- Matching point coordinates
```

5. CPU Final Processing:
```
a. Calculate actual private key:
   finalPrivateKey = basePrivateKey + increment

b. Verify result (optional):
   - Compute public key from final private key
   - Check if it matches target prefix

c. Output/Store result:
   - Private key in hex format
   - Corresponding public key/address
```

Key Points:
- CPU handles private key operations
- GPU focuses on efficient point addition/subtraction
- Communication between CPU/GPU is minimal
- Only successful matches trigger CPU computation
- The process is highly parallel on GPU side

This architecture takes advantage of:
- GPU's parallel processing for point operations
- Minimal data transfer between CPU/GPU
- Efficient modular arithmetic on GPU

# GPU Performance Design
GPU may generate too many keys per sec. For example, RTX 4090 may have 8 billion keys generated internally. However, when transfer it to host and compute for private key, the process is slown down. Therefore, the GPU employeed two byte filter using lookup table in GPU. When a byte or two bytes prefixes given, the GPU will firstly compare with the lookup table and discard mismatch, reduce pressure on transfer and CPU.


### Transfer Efficiency
The GPU stores discovered matches in a structure of 48 bytes per item, containing a thread ID (4 bytes), search metadata including increment value (4 bytes), and the public key's x coordinate (32 bytes), followed by padding (8 bytes) for optimal memory alignment.

You may notice the GPU does not transfer private keys to the CPU. Instead, it transfers the Thread ID and increment value. The CPU then reconstructs the private key by taking the starting key associated with that thread and adding the increment value. This approach significantly reduces data transfer between GPU and CPU.

The `maxFound` parameter determines how many items can be transferred from GPU to CPU in a single batch. This value directly impacts VRAM usage - a higher number requires more GPU memory but can handle more matches before overflow. When this buffer is insufficient, the transfer may not keep up with discovered matches, leading to result loss. You can adjust the buffer size using the `-m` parameter - for example, `-m 2000000`. The optimal value should be adjusted based on your needs and hardware capabilities, finding a balance between preventing "items lost" warnings and staying within your GPU's memory constraints.

An important consideration is that larger buffer sizes provide better handling of match bursts but require more VRAM, while smaller sizes use less memory but may lose results if matches are found faster than they can be transferred. Monitor for warning messages (usually show at start up) and memory usage to find the right balance for your specific situation.

# Things you probably should not do
### Don't get rid of Y coordinates

A public key in elliptic curve cryptography consists of coordinates (x,y). While our vanity search only checks the x-coordinate for matches, the y-coordinate cannot be eliminated from calculations due to the mathematics of point addition on elliptic curves.

Consider how we compute the next point R(x₃,y₃) from P(x₁,y₁) and Q(x₂,y₂):
```
s = (y₂ - y₁)/(x₂ - x₁)     // Need y-coordinates for slope
x₃ = s² - x₁ - x₂           // New x depends on slope
```

As shown above, we need y-coordinates to calculate the slope s, which is required to compute the next x-coordinate. Even though y values aren't used in our final prefix matching, they are essential intermediate values in the point arithmetic chain. There is no mathematical way to compute successive x-coordinates without tracking y values.

The code maintains both coordinates throughout the computation pipeline:
```cpp
ModSub256(dy, Gy[i], py);    // Calculate using y
_ModMult(_s, dy, dx[i]);     // To get slope
_ModSqr(_p2, _s);            // Which gives us
ModSub256(px, _p2, px);      // The next x
```

Therefore, while we only care about x in the end, y must remain part of our calculations for mathematical correctness.



# Modifications from SECP256K1 to SECP224R1


To be able to get this to works from Secp256k1 to Secp224r1
many modifications where made to the program to accommodate the new curve.

The first assumption that this program makes is the fixed bit size width
that all operations perform on. secp256 requires 4 64 bit of memory to do operations.
Some of the algorithms require to an extra bits to allow the overflow, so it bring it up to 320.

While secp224r1 is a 32 bits lower i kept most of the same bit lengths
for memory alignment. Targeting curves with a higher bit order of 256 will
require significant operations on the GPU side.

## Modifications


Change Msize in `IntMod.cpp` this is required for multiplication to operate properly.

Added the `-generateCode` flag to the program to generate new G constants gpu file

Updated point buffer to accommodate new bit size.

Used new class to represent Secp224r1 with all the constants defined,
also updated the iterations required for the generator table as the iterations
is defined as = bit_length / 8

Updated the some math operations method to include the `a` component of the curve.
DoubleDirect, Double, EC

Update methods in `Vanity.cpp` that operate on 256 bit assumptions to be 224.
+ The prefix shifting
+ The public key comparison

Implement ModMulR1 for the curve, this is an optimized multiplication operation
for the curve instead of a general use case.
The python code of how this works:

```python
p = 26959946667150639794667015087019630673557916260026308143510066298881
mul = 0xFFFFFFFFFFFFFFFFFFFFFFFF # defined as 2**224 - p
max_size = 224
maxsize = 2**max_size - 1


def calculate(a, b):
    expected = (a * b) % p
    r512 = a * b
    r512High = r512 >> max_size
    r512Low = r512 & maxsize
    res = r512High * mul
    print("First reduction", res)
    resLow = res & maxsize
    resHigh = res >> max_size

    r512Sum = r512Low + resLow
    r512SumLow = r512Sum & maxsize
    print("low sum", r512Sum)
    c = r512Sum >> max_size
    print("Res High", resHigh, c)
    al = (resHigh + c) * mul
    print("Second Reduction", al)
    r512SumLow = r512SumLow + al

    assert r512SumLow == expected, (r512SumLow, expected)
    return expected


# A = 0xAAA76B475B49CE46B36DDE5110EF8979D02B49370B33B030B4D7BE55
# B = 0xDBB45F7DACF80C3329B14C0873DDE41C033DB578B553176F26728E4B
# A = 21527200547432688033328474624722878788527285424879059932277260738698
# B = 1928016179827zz2891101689416591597159709208639520924814540571416132115
A = 0x9EABF932365A7215BA638A8F11E5E4A6F5AC7CC42082B3160895FC8F
B = 0xA6445595DDD941827134B378868A79D11914240B9F21E3522C8333A6
calculate(A, B)
```

Update Key Rate for the speed of which these are calculated since Secp224r1 does not have
any known endomorphisms.

GPU checking is offloaded entirely to the CPU in another thread so that it doesn't block the GPU thread
from call the GPU.

Many of the same operations done as above, also ported over to GPU cuda.

## Other Curve Implementations
You may compare our implementation verse [c8d48ce5f03f5357c0e87cbdb3e1e93cd50af88b](https://github.com/JeanLucPons/VanitySearch/commit/c8d48ce5f03f5357c0e87cbdb3e1e93cd50af88b).


### Implement CPU Modification first

Before implementing anything for the GPU, implement the CPU side of things first.
This is required because the `GPUGroup.h` is built off the CPU implementation.
It uses these computed G constants for point multiplication later in the code.

### Account for bitsize

The gpu is optimized for 256/320 operations, going to a lower bit curve will probably work
but will be wasting cycles or the upper bits.
Going to a higher bit size curve, will require modifications to account for the bit size.

Some of the CPU code has implemented up to 512 bits, but some are optimized for 256.



## Performance Improvements

The best performance improvement is finding any known endomorphisms,
depending on the complexity of the endomorphism, it can be as simple as a constant
which can double the performance practically. In the case of Secp224r1, there is no
known endomorphisms.

For secp224r1, there might be some improvement to use 224/288 instead of 256/320 bit size.
This would result in one less operation for each instruction set across addition, subtraction,
and multiplication. CUDA emulates 64 bits operations, but this would increase code complexity.






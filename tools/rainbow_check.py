import os
import glob
import multiprocessing as mp
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import ec
from tqdm import tqdm
import math

# Constants
TOTAL_KEYS = 2 ^ 24  # Total number of keys
BYTES_PER_KEY = 28  # Each key is 28 bytes
TOTAL_FILE_SIZE = 448 * 1024 * 1024  # 448MB in bytes
# Calculate chunk size that's a multiple of BYTES_PER_KEY (28 bytes)
KEYS_PER_CHUNK = 37449 * 2
CHUNK_SIZE = KEYS_PER_CHUNK * BYTES_PER_KEY


def prefix_check(priv_hex, db_file_name, offset):
    prefix = db_file_name.split(".")[0]
    # hex to bytes
    prefix_bytes = bytearray.fromhex(prefix)
    prefix_bytes[0] &= 0x3F

    # prefix_bytes[0] or 0x00, 0x40, 0x80, 0xC0
    prefix_mutations = [prefix_bytes[0], prefix_bytes[0] | 0x40, prefix_bytes[0] | 0x80, prefix_bytes[0] | 0xC0]

    private_key = ec.derive_private_key(
        int(priv_hex, 16),
        ec.SECP224R1(),
        default_backend()
    )

    public_key = private_key.public_key()
    public_key_bytes = public_key.public_numbers().x.to_bytes(28, byteorder='big')

    if int.from_bytes(public_key_bytes[0:1], 'big') in prefix_mutations:
        if public_key_bytes[1:3] == prefix_bytes[1:3]:
            if offset == int(public_key_bytes[3:6].hex(), 16) * 28:
                return True

    # print(f"Prefix check failed for {priv_hex}")
    return False


def validate_chunk(args):
    file_path, start_offset, chunk_size = args
    file_name = os.path.basename(file_path)
    mismatches = []
    skipped = 0
    try:
        with open(file_path, 'rb') as file:
            file.seek(start_offset)
            chunk = file.read(chunk_size)

            # Process each block in the chunk
            for i in range(0, len(chunk), BYTES_PER_KEY):
                block = chunk[i:i + BYTES_PER_KEY]

                if len(block) < BYTES_PER_KEY:
                    break

                # Skip padding blocks
                if block == b'\x00' * BYTES_PER_KEY:
                    skipped += 1
                    continue

                # Convert block to hex and validate
                block_hex = block.hex()
                # the offset of this key in the file
                offset = start_offset + i
                if start_offset + i < 469762048:  # 448MB
                    if not prefix_check(block_hex, file_name, offset):
                            mismatches.append(start_offset + i)

        return mismatches, skipped

    except Exception as e:
        print(f"\nError processing chunk at offset {start_offset} in {file_path}: {str(e)}")
        return []


def validate_dat_file(file_path):
    file_name = os.path.basename(file_path).lower()
    file_size = os.path.getsize(file_path)

    # Calculate number of chunks
    num_chunks = math.ceil(file_size / CHUNK_SIZE)

    # Prepare chunk arguments
    chunk_args = [
        (file_path, offset, min(CHUNK_SIZE, file_size - offset))
        for offset in range(0, file_size, CHUNK_SIZE)
    ]

    # Create a pool with number of available CPUs
    num_cpus = mp.cpu_count()
    pool = mp.Pool(num_cpus)

    # Process chunks in parallel with progress bar
    mismatches = []
    skipped = 0
    with tqdm(total=num_chunks, desc=f"Validating {file_name}", unit="chunk") as pbar:
        for chunk_mismatches, chunk_skipped in pool.imap_unordered(validate_chunk, chunk_args):
            mismatches.extend(chunk_mismatches)
            skipped += chunk_skipped
            pbar.update(1)

    pool.close()
    pool.join()

    # Report results
    if mismatches:
        print(f"\nValidation failed for {file_name}")
        print(f"Found {len(mismatches)} mismatches.")
        if len(mismatches) < 10:
            print(f"Mismatched offsets: {mismatches}")
        else:
            print(f"First 10 mismatched offsets: {mismatches[:10]}")
    else:
        print(f"\nValidation successful for {file_name}")
        coverage = (2 ** 24 - skipped) / 2 ** 24 * 100
        print(f"Skipped {skipped} padding blocks. Coverage: {coverage:.2f}%")

    return len(mismatches)


def process_dat_files():
    # Get the absolute path to the collections directory
    collections_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'collections'))

    # Use glob to find all .dat files
    dat_files = glob.glob(os.path.join(collections_path, '*.dat'))

    total_mismatches = 0

    # Process each .dat file (one at a time, but each file is processed in parallel)
    with tqdm(total=len(dat_files), desc="Overall progress", unit="file") as overall_pbar:
        for file_path in dat_files:
            mismatches = validate_dat_file(file_path)
            total_mismatches += mismatches
            overall_pbar.update(1)

    # Print summary
    print(f"\nProcessing complete:")
    print(f"Total files processed: {len(dat_files)}")
    print(f"Total mismatches found: {total_mismatches}")


if __name__ == "__main__":
    # Set start method for multiprocessing
    mp.set_start_method('spawn')
    process_dat_files()

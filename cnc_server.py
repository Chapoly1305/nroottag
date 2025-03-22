# /* nRootTag - Track device utilizing Find My network
#  * Copyright (c) 2025 Chapoly1305
#  *
#  * This program is free software: you can redistribute it and/or modify
#  * it under the terms of the GNU General Public License as published by
#  * the Free Software Foundation, version 3.
#  *
#  * This program is distributed in the hope that it will be useful, but
#  * WITHOUT ANY WARRANTY; without even the implied warranty of
#  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
#  * General Public License for more details.
#  *
#  * You should have received a copy of the GNU General Public License
#  * along with this program. If not, see <http://www.gnu.org/licenses/>.
#  */
#

import base64
import hashlib
import io
import logging
import mmap
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import List, BinaryIO
import queue
import json
import multiprocessing as mp
import requests

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
import uvicorn
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import ec

# Configure logging with timestamp, logger name, level, and message
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI application with metadata
app = FastAPI(
    title="nRootTag Management API",
    description="""
    Management API for nRootTag operations.
    Supports:
    - Storing private keys using prefix/suffix addressing
    - Retrieving public keys from stored private keys
    - Searching keys by address components
    - Managing key coverage and statistics
    """,
    version="1.0.0"
)

# Initialize thread-safe queue for batch processing inserts
insert_queue = queue.Queue()

# Create a manager for sharing file handles between processes
manager = mp.Manager()
file_handles = {}

# Global configuration
should_continue = True
record_size = 28  # Size of each key record in bytes
total_records = 2 ** 24  # Total number of possible records (16M)
data_size = total_records * record_size  # Total size of data file
num_workers = 4  # Number of parallel workers for coverage calculation
chunk_size = total_records // num_workers  # Size of chunks for parallel processing

VAST_API_BASE_URL = "https://console.vast.ai/api/v0"


class PrefixRequest(BaseModel):
    """
    Request model for operations that only require a prefix.
    Example:
        {
            "prefix": "aaaaaa"  # 6-character hex string
        }
    """
    prefix: str


class InsertRequest(BaseModel):
    """
    Request model for inserting private keys.
    Example:
        {
            "pairs": [
                "aaaaaa1111110000...000",  # 56-character hex string
                "bbbbbb2222220000...000"   # First 6 chars are prefix, next 50 are key data
            ]
        }
    """
    pairs: List[str]


class GetPublicKey(BaseModel):
    """
    Request model for searching a public key by address components.
    Example:
        {
            "prefix": "aaaaaa",  # First 6 chars of address
            "suffix": "111111"   # Next 6 chars of address
        }
    """
    prefix: str
    suffix: str


class PublicKeyRequest(BaseModel):
    """
    Request model for retrieving a public key using full address.
    Example:
        {
            "address": "aaaaaa111111"  # 12-character hex string (prefix + suffix)
        }
    """
    address: str


class ApiKey(BaseModel):
    """
    Request model for API key operations.
    """
    vastai_api: str
    cnc_server_url: str


class DeleteRequest(BaseModel):
    """
    Request model for deleting storage records.
    Example:
        {
            "key": "abc123"  # Key to delete, or "*" to clear all
        }
    """
    key: str


class Storage:
    """
    Persistent storage handler for managing JSON-based data.
    Used for storing configuration and metadata.
    """

    def __init__(self, file_path="storage.json"):
        self.file_path = file_path
        self.data = self.load_or_create()

    def load_or_create(self):
        """Load existing storage file or create new if doesn't exist."""
        if not os.path.exists(self.file_path):
            default_data = {}
            self.save(default_data)
            logger.info("Created new storage.json file")
            return default_data

        with open(self.file_path, 'r') as file:
            data = json.load(file)
            logger.info("Loaded existing storage.json file")
            return data

    def save(self, data=None):
        """Save current data to storage file."""
        if data is not None:
            self.data = data
        with open(self.file_path, 'w') as file:
            json.dump(self.data, file)

    def get(self, key, default=None):
        """Retrieve value for key with optional default."""
        return self.data.get(key, default)

    def set(self, key, value):
        """Set value for key and persist to storage."""
        self.data[key] = value
        self.save()

    def get_empty(self):
        # Get a list of prefixes that have value ""
        empty_prefixes = []
        for prefix in self.data:
            if self.data[prefix] == "":
                empty_prefixes.append(prefix)
        return empty_prefixes


# File Management Functions
def ensure_dat_file(prefix: str):
    """
    Ensure data file exists for given prefix with correct size.
    Creates new file if doesn't exist.

    Args:
        prefix: 6-character hex string identifying the data file
    """
    if not os.path.exists(f"collections/{prefix}.dat"):
        with open(f"collections/{prefix}.dat", 'w') as f:
            f.seek(28 * 256 ** 3)  # Pre-allocate space for all possible keys
            f.write('\0')


def if_dat_exists(prefix: str) -> bool:
    """
    Check if data file exists for given prefix.

    Args:
        prefix: 6-character hex string identifying the data file
    Returns:
        bool: True if file exists, False otherwise
    """
    return os.path.exists(f"collections/{prefix}.dat")


def create_empty_dat(prefix: str):
    """
    Create new empty data file for prefix.

    Args:
        prefix: 6-character hex string identifying the data file
    """
    with open(f"collections/{prefix}.dat", 'w') as f:
        f.seek(28 * 256 ** 3)  # Allocate space for 16M records of 28 bytes each
        f.write('\0')


def get_value_from_dat(f: BinaryIO, index: int):
    """
    Retrieve private key value from data file at specific index.

    Args:
        f: File handle for data file
        index: Position of key in file (0-16M)
    Returns:
        bytes: 28-byte key value
    """
    f.seek(index * 28)
    return f.read(28)


def get_dat_handle(prefix: str) -> BinaryIO:
    """
    Get or create file handle for data file.
    Maintains cache of open file handles.

    Args:
        prefix: 6-character hex string identifying the data file
    Returns:
        BinaryIO: File handle for reading/writing
    """
    if prefix not in file_handles:
        ensure_dat_file(prefix)
        file_handles[prefix] = open(f"collections/{prefix}.dat", 'r+b')
    return file_handles[prefix]


# Background Processing Functions
def process_insert_queue():
    """
    Background worker that processes queued insert requests.
    Runs continuously, pulling requests from queue and processing them.
    """
    while True:
        try:
            request = insert_queue.get(timeout=1)
            insert_data(request)
            insert_queue.task_done()
        except queue.Empty:
            time.sleep(1)
            continue
        except Exception as e:
            logger.error(f"Error processing insert request: {e}")


def insert_data(request: InsertRequest):
    """
    Process insertion of private keys into data files.

    Algorithm:
    1. For each key pair:
        - Derive private key from hex string
        - Generate public key
        - Calculate prefix and suffix for storage
        - Store in appropriate data file

    Args:
        request: InsertRequest containing list of key pairs to insert
    """
    try:
        for pair in request.pairs:
            try:
                if len(pair) != 56:
                    logger.error(f"Invalid pair length: {pair}")
                    continue

                # Convert hex string to private key object
                private_key = ec.derive_private_key(
                    int(pair, 16),
                    ec.SECP224R1(),
                    default_backend()
                )
                priv_key = private_key.private_numbers().private_value.to_bytes(28, byteorder='big')

                # Generate public key and calculate address components
                public_key = private_key.public_key()
                public_key_bytes = public_key.public_numbers().x.to_bytes(28, byteorder='big')
                suffix = public_key_bytes[3:6]
                addr_int = int.from_bytes(suffix, byteorder='big')

                # Calculate prefix (first 3 bytes with high bits masked)
                prefix_bytes = bytearray(public_key_bytes[:3])
                prefix_bytes[0] &= 0x3F  # Mask high bits to save space for Attack-II
                prefix = prefix_bytes.hex()

                # Get or create file handle
                if prefix not in file_handles:
                    if not if_dat_exists(f"{prefix}"):
                        create_empty_dat(f"{prefix}")
                    file_handles[prefix] = get_dat_handle(f"{prefix}")
                f = file_handles[prefix]

            except ValueError:
                logger.error(f"Invalid hex string: {pair}")
                continue

            try:
                existing_value = get_value_from_dat(f, addr_int)
                if existing_value != b'\0' * 28:
                    continue

                # Write private key to file
                f.seek(addr_int * 28)
                f.write(priv_key)
                f.flush()  # Ensure data is written to disk
            except io.UnsupportedOperation:
                # Reopen file in read-write mode if needed
                f.close()
                f = open(f"collections/{prefix}.dat", 'r+b')
                file_handles[prefix] = f
                f.seek(addr_int * 28)
                f.write(priv_key)
                f.flush()

    except Exception as e:
        logger.error(f"Error in insert_data: {str(e)}", exc_info=True)
        raise


def address_mutate(address: str):
    address_collection = []
    address_bytes = bytearray.fromhex(address)

    address_bytes[0] = address_bytes[0] & 0x3F
    address_collection.append(address_bytes.hex())

    address_bytes[0] = address_bytes[0] & 0x3F | 0x40
    address_collection.append(address_bytes.hex())

    address_bytes[0] = address_bytes[0] & 0x3F | 0x80
    address_collection.append(address_bytes.hex())

    address_bytes[0] = address_bytes[0] & 0x3F | 0xC0
    address_collection.append(address_bytes.hex())

    return address_collection


async def record_unfound_request(prefix: str, suffix: str):
    """
    Record unsuccessful key lookups for analysis.
    Maintains log of missing keys with timestamps.

    Args:
        prefix: 6-character hex prefix of missing key
        suffix: 6-character hex suffix of missing key
    """
    UNFOUND_REQUESTS_FILE = "unfound_requests.txt"
    request_entry = f"{prefix}+{suffix}"

    # Create log file if needed
    if not os.path.exists(UNFOUND_REQUESTS_FILE):
        with open(UNFOUND_REQUESTS_FILE, "w") as f:
            f.write("Address,Timestamp\n")

    # Skip if already logged
    with open(UNFOUND_REQUESTS_FILE, "r") as f:
        if request_entry in f.read():
            return

    # Add new entry with timestamp
    with open(UNFOUND_REQUESTS_FILE, "a") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{request_entry},{timestamp}\n")


# API Endpoints
@app.post("/insert-data", tags=["Data Insertion"])
async def insert_data_route(request: InsertRequest):
    """
    This api is for seeker to insert private keys in batch.

    Example Request:
    ```json
    {
        "pairs": [
            "aaaaaa1111110000...000",
            "bbbbbb2222220000...000"
        ]
    }
    ```

    Returns:
        dict: Status message indicating request was received
    """
    global should_continue
    try:
        if not should_continue:
            return {"error": "STOP"}

        insert_queue.put(request)
        return {"message": "received"}
    except Exception as e:
        logger.error(f"Error inserting data: {e}")
        raise HTTPException(status_code=500, detail=f"Error inserting data: {str(e)}")


def normalize_address(address: str):
    """
    Normalize an address by masking the high bits to 0b00.

    Args:
        address: Hex string address

    Returns:
        str: Normalized address with high bits set to 0b00
    """
    try:
        address_bytes = bytearray.fromhex(address)
        # Mask high bits to 00
        address_bytes[0] &= 0x3F
        return address_bytes.hex()
    except Exception as e:
        logger.error(f"Error normalizing address {address}: {e}")
        return address  # Return original if error


@app.post("/public-key", tags=["Key Retrieval"])
async def get_public_key(request: PublicKeyRequest = PublicKeyRequest(address="1eadbe112233")):
    """
    Retrieve public key for given address.
    If given address is not found, it will be added to the task list.

    Example Request:
    ```json
    {
        "address": "1eadbe112233"
    }
    ```

    Returns:
        dict: Public key in hex format
    """
    try:
        if not request.address:
            raise HTTPException(status_code=400, detail="Missing required fields")

        # Normalize the address first (mask high bits to 00)
        normalized_address = normalize_address(request.address[:12])
        normalized_prefix = normalized_address[:6]
        normalized_suffix = normalized_address[6:12]

        try:
            suffix_int = int(normalized_suffix, 16)
        except ValueError:
            logger.error(f"Invalid suffix format")
            raise HTTPException(status_code=400, detail="Invalid suffix format")

        # Check if normalized data file exists
        if not if_dat_exists(normalized_prefix):
            logger.info(
                f"Key File Not Found: {normalized_prefix} {normalized_suffix} (original: {request.address[:12]})")
            await record_unfound_request(normalized_prefix, normalized_suffix)
            # Store original address variants for task list
            address_collection = address_mutate(request.address[:12])
            for address in address_collection:
                storage.set(address, "")
            raise HTTPException(status_code=404, detail="Key File Not Found, added for task")

        # Get file handle for normalized address
        if normalized_prefix not in file_handles:
            file_handles[normalized_prefix] = get_dat_handle(normalized_prefix)
        file_handle = file_handles[normalized_prefix]

        # Read private key
        priv_key = get_value_from_dat(file_handle, suffix_int)

        # Check if key exists
        if priv_key == b'\0' * 28:
            await record_unfound_request(normalized_prefix, normalized_suffix)
            # Store original address variants for task list
            address_collection = address_mutate(request.address[:12])
            for address in address_collection:
                storage.set(address, "")
            raise HTTPException(status_code=404, detail="Key Record Not Found, added for task")

        logger.debug(f"Private key found: {priv_key.hex()}")

        # Generate public key
        try:
            private_key = ec.derive_private_key(
                int.from_bytes(priv_key, byteorder='big'),
                ec.SECP224R1(),
                default_backend()
            )
            public_key = private_key.public_key()
            public_hex_str = public_key.public_numbers().x.to_bytes(28, byteorder='big').hex()
            storage.set(request.address[:12], priv_key.hex())
            return {"public_key": public_hex_str}

        except Exception as e:
            logger.error(f"Error generating public key: {e}")
            raise HTTPException(
                status_code=500,
                detail="Error generating public key"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Key Not Found")


@app.post("/review-key", tags=["Key Retrieval"])
async def review_key(request: GetPublicKey = GetPublicKey(prefix="1eadbe", suffix="112233")):
    """
    Examine the public and private key pair, also the hashed public key of a given address.

    Example Request:
    ```json
    {
        "prefix": "1eadbe",
        "suffix": "112233"
    }
    ```

    Returns:
        dict: Public key, private key, and SHA256 hash of public key
    """
    # Get file handle
    if request.prefix in file_handles:
        f = file_handles[request.prefix]
    else:
        if not if_dat_exists(f"{request.prefix}"):
            await record_unfound_request(request.prefix, request.suffix)
            return {"message": f"NOK, {request.prefix} Table not found"}
        f = get_dat_handle(f"{request.prefix}")
        file_handles[request.prefix] = f

    # Read private key
    priv_key = get_value_from_dat(f, int(request.suffix, 16))
    if priv_key == b'\0' * 28:
        await record_unfound_request(request.prefix, request.suffix)
        return {"message": "NOK, Private key not found"}

    # Generate key pair and hash
    private_key = ec.derive_private_key(
        int.from_bytes(priv_key, byteorder='big'),
        ec.SECP224R1(),
        default_backend()
    )
    public_key = private_key.public_key()
    public_hex_str = public_key.public_numbers().x.to_bytes(28, byteorder='big').hex()

    public_key_bytes = public_key.public_numbers().x.to_bytes(28, byteorder='big')
    hash_object = hashlib.sha256(public_key_bytes)
    hex_dig = hash_object.hexdigest()

    return {
        "public_key": public_hex_str,
        "private_key": priv_key.hex(),
        "public_key_sha256": base64.b64encode(bytearray.fromhex(hex_dig)).decode()
    }


@app.get("/review-storage", tags=["Key Retrieval"])
async def review_storage():
    """
    Retrieve all stored keys from storage file.

    Returns:
        dict: Dictionary of all stored keys. Key is address, value is private key.
    """
    return storage.data


@app.get("/random-key", tags=["Key Retrieval"])
async def get_random_key(prefix: str):
    """
    Retrieve random existing key from specified prefix table.

    Args:
        prefix: 6-character hex prefix to search in

    Returns:
        dict: Random public/private key pair from the table
    """
    # Get file handle
    if prefix in file_handles:
        f = file_handles[prefix]
    else:
        if not if_dat_exists(f"{prefix}"):
            return {"message": "NOK, Table not found"}
        f = get_dat_handle(f"{prefix}")
        file_handles[prefix] = f

    # Keep trying random indices until we find a non-empty key
    priv_key = None
    while not priv_key:
        index = os.urandom(3)  # Generate random 3-byte index

        priv_key = get_value_from_dat(f, int.from_bytes(index, byteorder='big'))
        if priv_key == b'\0' * 28:  # Skip empty records
            priv_key = None
            continue

        # Generate public key
        private_key = ec.derive_private_key(
            int.from_bytes(priv_key, byteorder='big'),
            ec.SECP224R1(),
            default_backend()
        )
        public_key = private_key.public_key()
        public_hex_str = public_key.public_numbers().x.to_bytes(28, byteorder='big').hex()

        return {
            "public_key": public_hex_str,
            "private_key": priv_key.hex()
        }


# Add these endpoints to your FastAPI app
@app.get("/status", tags=["Server Control"])
async def get_status():
    """
    Get current server status.

    Returns:
        dict: Current server status
        {
            "continue": true/false  # Whether server should continue processing
        }
    """
    global should_continue
    return {"continue": should_continue}


@app.post("/status", tags=["Server Control"])
async def set_status(status: bool):
    """
    Update server operation status.
    
    Example Request:
    ```json
    {
        "continue_flag": true  # Set to false to stop server
    }
    ```

    Returns:
        dict: Updated server status
    """
    global should_continue
    try:
        logger.debug(f"Server status updated: continue={status}")
        should_continue = status
        return {"continue": status, "message": "Status updated successfully"}
    except Exception as e:
        logger.error(f"Error updating status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# @app.post("/vastai-trigger-search", tags=["Integration"])
async def trigger_search(vastai_token: str, cnc_server_url: str, quantity: int = 1, ):
    """
    Trigger remote search task on Vast.ai GPU instance using direct API calls.
    """
    if quantity <= 0:
        raise HTTPException(status_code=400, detail="Quantity must be greater than 0")

    try:
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {vastai_token}'
        }

        commited_quantity = 0
        while commited_quantity < quantity:
            # Search for available GPU instances
            search_url = f"{VAST_API_BASE_URL}/search/asks/"
            search_payload = {
                "q": {
                    "gpu_name": {"in": ["RTX 3080"]},
                    "inet_down": {"gte": 300},
                    "geolocation": {"in": ["US", "CA"]},
                    "type": "on-demand"
                }
            }

            logger.debug(f"Searching for GPU instances with params: {search_payload}")
            response = requests.put(
                search_url,
                headers=headers,
                data=json.dumps(search_payload)
            )

            if response.status_code != 200:
                logger.error(f"Search API error: {response.text}")
                raise HTTPException(status_code=response.status_code,
                                    detail=f"Error searching for instances: {response.text}")

            try:
                results = response.json()
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {response.text}")
                raise HTTPException(status_code=500, detail="Invalid JSON response from API")

            # Handle various response formats
            if isinstance(results, dict):
                if results.get('message'):
                    logger.error(f"API Error in response: {results}")
                    raise HTTPException(status_code=500, detail=f"API Error: {results['message']}")
                offers = results.get('offers', [])
            elif isinstance(results, list):
                offers = results
            else:
                logger.error(f"Unexpected response format: {results}")
                raise HTTPException(status_code=500, detail="Unexpected response format from API")

            if not offers:
                logger.error("No available devices found")
                raise HTTPException(status_code=500, detail="No available devices found")

            available_devices = []
            for offer in offers:
                if len(available_devices) >= (quantity - commited_quantity):
                    break
                if isinstance(offer, dict):
                    offer_id = offer.get("id")
                    if offer_id:
                        available_devices.append(offer_id)
                elif isinstance(offer, str):
                    available_devices.append(offer)

            logger.debug(f"Available Devices: {available_devices}")

            # Create instances for available devices
            for device_id in available_devices:
                logger.debug(f"Creating instance for device: {device_id}")

                create_url = f"{VAST_API_BASE_URL}/asks/{device_id}/"
                create_payload = {
                    "image": "chiba765/nroottag-seeker:latest",
                    "disk": 8,
                    "extra_env": {
                        "CNC_SERVER_URL": cnc_server_url
                    },
                    "runtype": "command",
                    "target_state": "running",
                    "cancel_unavail": True
                }

                response = requests.put(
                    create_url,
                    headers=headers,
                    json=create_payload
                )

                if response.status_code == 200 and response.json().get('success', False):
                    logger.debug(f"Successfully created instance: {response.json()}")
                    commited_quantity += 1
                else:
                    logger.error(f"Failed to create instance: {response.text}")
                    return {"message": f"Created {commited_quantity} instances, failed to create more"}

        return {"message": "Search triggered successfully"}

    except Exception as e:
        logger.error(f"Error creating instance: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating instance: {str(e)}")


# @app.post("/vastai-destroy-all-instances", tags=["Integration"])
async def destroy_instances(vastai_token: str):
    """
    Destroy all running instances on Vast.ai GPU cluster.
    
    Returns:
        dict: Message indicating success/failure and count of destroyed instances
    """
    try:
        import vastai
        APIKey = vastai_token
        server = vastai.VastAI(APIKey, raw=True)

        # Get list of all running instances
        instances_json = server.show_instances()
        instances = json.loads(instances_json)

        if not instances:
            return {"message": "No running instances found"}

        destroyed_count = 0
        failed_instances = []

        # Iterate through each instance and destroy it
        for instance in instances:
            instance_id = instance.get('id')
            if not instance_id:
                continue

            try:
                result = server.destroy_instance(id=instance_id)
                result = json.loads(result)

                if result.get('success'):
                    destroyed_count += 1
                    logger.debug(f"Successfully destroyed instance {instance_id}")
                else:
                    failed_instances.append(instance_id)
                    logger.error(f"Failed to destroy instance {instance_id}: {result}")

            except Exception as inner_e:
                failed_instances.append(instance_id)
                logger.error(f"Error destroying instance {instance_id}: {inner_e}")

        # Prepare response message
        if failed_instances:
            return {
                "message": f"Partially successful: Destroyed {destroyed_count} instances, failed to destroy {len(failed_instances)} instances",
                "destroyed_count": destroyed_count,
                "failed_instances": failed_instances
            }
        else:
            return {
                "message": f"Successfully destroyed {destroyed_count} instances",
                "destroyed_count": destroyed_count
            }

    except Exception as e:
        logger.error(f"Error destroying instances: {e}")
        raise HTTPException(status_code=500, detail=f"Error destroying instances")


@app.post("/saladcloud-start-containers", tags=["Integration"])
async def start_containers(
        organization_name: str,
        project_name: str,
        container_group_name: str,
        salad_api_key: str,
):
    """
    Start containers on Saladcloud using their public API.
    
    Args:
        organization_name: Name of the organization
        project_name: Name of the project
        container_group_name: Name of the container group
        salad_api_key: Saladcloud API key
    
    Returns:
        dict: Message indicating success/failure and count of started containers
    """

    try:
        headers = {
            'Salad-Api-Key': salad_api_key,
            'Content-Type': 'application/json'
        }

        base_url = "https://api.salad.com/api/public"
        start_url = f"{base_url}/organizations/{organization_name}/projects/{project_name}/containers/{container_group_name}/start"

        response = requests.post(
            start_url,
            headers=headers
        )

        if response.status_code == 202:
            return {
                "message": f"Successfully started containers",
            }
        else:
            logger.error(f"Failed to start containers: {response.text}")
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Failed to start containers: {response.text}"
            )

    except Exception as e:
        logger.error(f"Error starting containers: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting containers")


@app.post("/saladcloud-stop-containers", tags=["Integration"])
async def stop_containers(
        organization_name: str,
        project_name: str,
        container_group_name: str,
        salad_api_key: str
):
    """
    Stop all containers in a container group on Saladcloud.
    
    Args:
        organization_name: Name of the organization
        project_name: Name of the project
        container_group_name: Name of the container group
        salad_api_key: Saladcloud API key
    
    Returns:
        dict: Message indicating success/failure and count of stopped containers
    """
    try:
        headers = {
            'Salad-Api-Key': salad_api_key,
            'Content-Type': 'application/json'
        }

        base_url = "https://api.salad.com/api/public"
        stop_url = f"{base_url}/organizations/{organization_name}/projects/{project_name}/containers/{container_group_name}/stop"

        response = requests.post(
            stop_url,
            headers=headers
        )

        if response.status_code == 202:
            return {
                "message": f"Successfully stopped containers",
            }
        else:
            logger.error(f"Failed to stop containers: {response.text}")
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Failed to stop containers: {response.text}"
            )

    except Exception as e:
        logger.error(f"Error stopping containers: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error stopping containers")


# Coverage Calculation Functions
def process_chunk(filename, start_idx, num_records):
    """
    Process a chunk of the data file to count non-empty records.
    Used for parallel coverage calculation.

    Args:
        filename: Path to data file
        start_idx: Starting index in file
        num_records: Number of records to process

    Returns:
        int: Number of non-empty records in chunk
    """
    with open(filename, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            # Read chunk into numpy array for efficient processing
            arr = np.frombuffer(mm, dtype=np.uint8, count=data_size)
            arr = arr.reshape((total_records, record_size))

            # Count non-zero records in chunk
            arr_chunk = arr[start_idx:start_idx + num_records]
            count = np.count_nonzero(np.any(arr_chunk != 0, axis=1))

            # Clean up arrays
            arr_chunk = None
            arr = None
        finally:
            mm.close()

    return count


# This endpoint is working, but it is not used in the current version of the API
# If you want to use it, you need to uncomment it.
# It reads the end of the file and returns the coverage count, which might be outdated.
# @app.get("/coverage-cached", tags=["Coverage"])
# async def get_cached_coverage(prefix: str):
#     """
#     Get cached coverage count for prefix.
#     Coverage is stored at end of data file.
#
#     Args:
#         prefix: 6-character hex prefix to check
#
#     Returns:
#         dict: Number of non-empty records in table
#     """
#     if prefix not in file_handles:
#         if not if_dat_exists(f"{prefix}"):
#             return {"message": "NOK, Table not found"}
#
#     # Get file handle
#     if prefix in file_handles:
#         f = file_handles[prefix]
#     else:
#         f = get_dat_handle(f"{prefix}")
#
#     # Read coverage from end of file
#     f.seek(28 * 256 ** 3)
#     coverage = int.from_bytes(f.read(28), byteorder='big')
#     return {"coverage": coverage}


@app.get("/coverage-reliable", tags=["Coverage"])
async def get_coverage(prefix: str):
    """
    Calculate current coverage using parallel processing.
    More accurate but slower than cached coverage.

    Args:
        prefix: 6-character hex prefix to analyze

    Returns:
        dict: Number of non-empty records in table
    """
    # Check if data file exists
    if prefix not in file_handles:
        if not if_dat_exists(f"{prefix}"):
            return {"message": "NOK, Table not found"}

    # Get file handle
    if prefix in file_handles:
        f = file_handles[prefix]
    else:
        f = get_dat_handle(f"{prefix}")

    filename = f.name
    futures = []

    # Process file in parallel chunks
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for i in range(num_workers):
            start_idx = i * chunk_size
            if i == num_workers - 1:
                records_to_process = total_records - start_idx
            else:
                records_to_process = chunk_size

            futures.append(executor.submit(
                process_chunk,
                filename,
                start_idx,
                records_to_process
            ))

        # Sum results from all chunks
        partial_counts = [future.result() for future in futures]
    total_count = sum(partial_counts)

    # Update cached coverage
    with open(filename, "r+b") as ff:
        ff.seek(data_size)
        ff.write(total_count.to_bytes(record_size, byteorder='big'))

    return {"coverage": total_count}


@app.post("/delete-search-task", tags=["Search Task"])
async def delete_storage(key: str):
    """
    Delete specific storage record or clear all records.
    
    Example Request:
    ```json
    {
        "key": "abc123"  # Specific key to delete
        # OR
        "key": "*"      # Clear all records
    }
    ```

    Returns:
        dict: Status message indicating what was deleted
    """
    try:
        if key == "*":
            # Clear all records
            storage.data.clear()
            storage.save()
            logger.debug("Cleared all storage records")
            return {"message": "All storage records cleared successfully"}

        # Delete specific key
        if key in storage.data:
            del storage.data[key]
            storage.save()
            logger.debug(f"Deleted storage record for key: {key}")
            return {"message": f"Storage record for key '{key}' deleted successfully"}
        else:
            logger.debug(f"Key not found in storage: {key}")
            return {"message": "Key not found in storage"}

    except Exception as e:
        logger.error(f"Error deleting storage record: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# File Download Endpoints
@app.get("/executable", tags=["Download"])
async def get_executable(version: str = "11"):
    """Download Seeker_CUDA executable."""
    return FileResponse(
        f"executables/Seeker_CUDA_{version}",
        filename="Seeker_CUDA",
        media_type="application/octet-stream"
    )


@app.get("/search-task", tags=["Search Task"])
async def get_search_task():
    """
    Get search tasks as a newline-separated file.
    Each line contains a 6-character prefix representing a task.

    Returns:
        Response: Plain text file with one task per line
    """
    candidates = storage.get_empty()
    tasks = set()
    for candidate in candidates:
        tasks.add(candidate)

    # Convert set to newline-separated string
    task_content = '\n'.join(sorted(tasks))
    task_content += '\n'
    # Return as a downloadable text file
    return Response(
        content=task_content,
        media_type="text/plain",
        headers={
            "Content-Disposition": "attachment; filename=filters.txt"
        }
    )


@app.post("/add-search-task", tags=["Search Task"])
async def add_search_task(prefix: PrefixRequest):
    """
    This api adds a new search task to the storage.json.

    Args:
        prefix: 6-character hex prefix (OUI) to add as a task

    Returns:
        dict: Status message indicating task was added
    """
    address_collection = address_mutate(prefix.prefix)
    for address in address_collection:
        storage.set(address, "")

    return {"message": "Task added"}


# Main Entry Point
if __name__ == "__main__":
    # Start background processing thread
    insert_thread = threading.Thread(target=process_insert_queue, daemon=True)
    insert_thread.start()

    # Initialize storage
    storage = Storage()

    banner = '''
 ██████ ███    ██  ██████     ███████ ███████ ██████  ██    ██ ███████ ██████  
██      ████   ██ ██          ██      ██      ██   ██ ██    ██ ██      ██   ██ 
██      ██ ██  ██ ██          ███████ █████   ██████  ██    ██ █████   ██████  
██      ██  ██ ██ ██               ██ ██      ██   ██  ██  ██  ██      ██   ██ 
 ██████ ██   ████  ██████     ███████ ███████ ██   ██   ████   ███████ ██   ██ 
                                                                               
                                                                               
                                                                               '''
    print(banner)

    # Start FastAPI server
    uvicorn.run(app, host="localhost", port=7898, log_level="debug")

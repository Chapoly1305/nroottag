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

import logging
import os
import secrets
import subprocess
import sys
import tempfile
import time
import threading
from dataclasses import dataclass
from typing import List, Optional

# Third-party imports
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

# Configure logging with timestamp and appropriate format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [%(name)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("node_runner")

import logging
import os
import secrets
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import List, Optional

# Third-party imports
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry


# [Previous logging setup remains the same...]

@dataclass
class Config:
    """Configuration class for the node runner application."""
    cnc_server_url: str = None  # Command & Control server URL from environment
    batch_size: int = 200000  # Number of items to process in each batch
    retry_attempts: int = 30  # Number of HTTP retry attempts
    backoff_factor: int = 2  # Exponential backoff factor for retries
    status_forcelist: List[int] = None  # HTTP status codes to retry on
    status_check_interval: int = 30  # Interval in seconds for checking server status

    def __post_init__(self):
        if self.status_forcelist is None:
            self.status_forcelist = [500, 502, 503, 504]

        if self.cnc_server_url is None:
            self.cnc_server_url = os.environ.get('CNC_SERVER_URL', None)
            if self.cnc_server_url is None:
                logger.error("CNC_SERVER_URL environment variable is not set")
                self.cnc_server_url = 'http://localhost:7898'
            logger.info(f"Using C&C server URL: {self.cnc_server_url}")


class HTTPClient:
    """
    Handles all HTTP communications with the C&C server.
    Implements retry logic and connection pooling.
    """

    def __init__(self, config: Config):
        """
        Initialize HTTP client with retry strategy and connection pooling.
        Args:
            config (Config): Application configuration
        """
        self.config = config
        # Configure retry strategy for resilient HTTP connections
        retry_strategy = Retry(
            total=config.retry_attempts,
            backoff_factor=config.backoff_factor,
            status_forcelist=config.status_forcelist
        )
        self.session = requests.Session()
        # Apply retry strategy to both HTTP and HTTPS connections
        self.session.mount('http://', HTTPAdapter(max_retries=retry_strategy))
        self.session.mount('https://', HTTPAdapter(max_retries=retry_strategy))

    def check_status(self) -> bool:
        """
        Check server status.
        Returns:
            bool: True if should continue, False if should stop
        """
        try:
            response = self.session.get(
                f'{self.config.cnc_server_url}/status',
                timeout=5
            )
            response.raise_for_status()
            status = response.json()
            return status.get('continue', True)
        except requests.exceptions.RequestException as e:
            logger.error(f"Status check failed: {e}")
            return True

    def post_data(self, key_pairs: List[str]) -> dict:
        """
        Post processed data back to the C&C server.
        Args:
            key_pairs (List[str]): List of processed key pairs to submit
        Returns:
            dict: Server response or error information
        """
        try:
            response = self.session.post(
                f'{self.config.cnc_server_url}/insert-data',
                json={"pairs": key_pairs},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return {"error": str(e)}

    def download_executable(self, version) -> bytes:
        """
        Download the executable file from the C&C server.
        Returns:
            bytes: The downloaded executable content
        Raises:
            requests.exceptions.RequestException: If download fails
        """
        try:
            response = self.session.get(
                f'{self.config.cnc_server_url}/executable?version={version}',
                timeout=60
            )
            response.raise_for_status()
            return response.content
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download executable: {e}")
            raise

    def get_search_prefix(self) -> Optional[str]:
        """
        Retrieve search prefix from the C&C server.
        Returns:
            Optional[str]: Search prefix if successful, None otherwise
        """
        try:
            response = self.session.get(f'{self.config.cnc_server_url}/search-task')
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException:
            logger.error("Server Connection Issue. No search prefix download.")
            return None

    def close(self):
        """Clean up resources by closing the HTTP session"""
        self.session.close()


class GPUManager:
    """
    Manages GPU detection and environment setup.
    Provides static methods for GPU-related operations.
    """

    @staticmethod
    def get_gpu_count() -> int:
        """
        Detect number of available NVIDIA GPUs using nvidia-smi.
        Returns:
            int: Number of GPUs detected, 0 if none found or error occurs
        """
        try:
            env = os.environ.copy()
            if 'FORCE_GPU_ID' in env:
                return 1

            process = subprocess.run(
                ["nvidia-smi", "-L"],
                capture_output=True,
                text=True,
                check=True
            )
            return sum(1 for line in process.stdout.splitlines() if line.strip().startswith("GPU "))
        except subprocess.SubprocessError:
            logger.warning("Failed to count GPUs")
            return 0

    @staticmethod
    def get_gpu_env() -> dict:
        """
        Set up GPU environment variables for CUDA.
        Returns:
            dict: Environment variables including CUDA configuration
        """
        env = os.environ.copy()
        if 'FORCE_GPU_ID' in env:
            env['CUDA_VISIBLE_DEVICES'] = env['FORCE_GPU_ID']
            logger.info(f"Set CUDA_VISIBLE_DEVICES to {env['CUDA_VISIBLE_DEVICES']}")
        else:
            gpu_count = GPUManager.get_gpu_count()
            # Configure CUDA environment if GPUs are available
            if gpu_count > 0 and not env.get('CUDA_VISIBLE_DEVICES'):
                env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, range(gpu_count)))
                logger.info(f"Set CUDA_VISIBLE_DEVICES to {env['CUDA_VISIBLE_DEVICES']}")
        return env


def get_cuda_ver() -> str:
    try:
        output = subprocess.check_output(["nvidia-smi"])
        if "CUDA Version: 12" in output.decode():
            cuda_version = "12"
        elif "CUDA Version: 11" in output.decode():
            cuda_version = "11"
    except:
        print("nvidia-smi command not found")
    print("Identified CUDA Version:", cuda_version)
    return cuda_version


class NodeRunner:
    """
    Main class for running the node operations.
    Handles process management, data processing, and coordination.
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize NodeRunner with configuration.
        Args:
            config (Optional[Config]): Configuration object, creates default if None
        """
        self.config = config or Config()
        self.http_client = HTTPClient(self.config)
        self.gpu_processes: List[subprocess.Popen] = []
        self.stop_requested = False

    def setup_executable(self, filename) -> str:
        """
        Ensure executable is available, downloading if necessary.
        Args:
            filename (str): Name of the executable file
        Returns:
            str: Path to the executable
        """
        if os.path.exists(filename):
            logger.info(f"Using existing executable at {filename}")
            return filename

        logger.info(f"Downloading executable {filename}")
        content = self.http_client.download_executable(get_cuda_ver())
        with tempfile.NamedTemporaryFile(delete=False, suffix='.exe') as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name

        # Make executable file executable
        os.chmod(temp_file_path, os.stat(temp_file_path).st_mode | 0o755)
        logger.info(f"Executable saved to {temp_file_path}")
        return temp_file_path

    def launch_gpu_processes(self, executable_path: str, output_file: str):
        """
        Launch GPU processes for parallel processing.
        Args:
            executable_path (str): Path to the executable
            output_file (str): Path to the output file
        """
        gpu_count = GPUManager.get_gpu_count()
        if gpu_count == 0:
            logger.error("No GPUs detected")
            return

        env = GPUManager.get_gpu_env()
        # Launch process for each GPU
        for i in range(gpu_count):
            process = subprocess.Popen(
                [
                    executable_path,
                    '-gpu',
                    '-gpuId', str(i),
                    '-ps', secrets.token_urlsafe(16),  # Generate random process secret
                    '-t', '0',
                    '-i', 'prefixes.txt',
                    '-m', '6553500',
                    '-o', output_file,
                    '-p',
                ],
                env=env
            )
            self.gpu_processes.append(process)
            time.sleep(0.1)  # Small delay between process launches

        logger.info(f"Launched {len(self.gpu_processes)} GPU process(es)")

    def process_output(self, output_file: str):
        """
        Process the output file and send results to server.
        Args:
            output_file (str): Path to the output file to process
        """
        # Wait for output file to be created
        while not os.path.exists(output_file) and not self.stop_requested:
            logger.info(f"Waiting for {output_file} to be created...")
            time.sleep(5)

        if self.stop_requested:
            return

        data_batch = []
        checkpoint_time = time.time()
        logger.info(f"Processing output from {output_file}")

        with open(output_file, 'r') as file:
            while not self.stop_requested:
                if self.should_stop_processing():
                    break

                line = file.readline()
                if not line:
                    if self.should_submit_batch(data_batch, checkpoint_time):
                        self.submit_batch(data_batch)
                        data_batch = []
                        checkpoint_time = time.time()
                    time.sleep(10)
                    continue

                # Extract private key from line
                priv = line.strip().split(':')[3] if len(line.strip().split(':')) >= 4 else None
                if priv:
                    data_batch.append(priv)

                if self.should_submit_batch(data_batch, checkpoint_time):
                    self.submit_batch(data_batch)
                    data_batch = []
                    checkpoint_time = time.time()

        # Submit any remaining data
        if data_batch:
            self.submit_batch(data_batch)

    def should_stop_processing(self) -> bool:
        """
        Check if processing should stop based on GPU process status.
        Returns:
            bool: True if all GPU processes have completed
        """
        return all(p.poll() is not None for p in self.gpu_processes)

    def should_submit_batch(self, batch: List[str], checkpoint_time: float) -> bool:
        """
        Determine if current batch should be submitted based on size or time.
        Args:
            batch (List[str]): Current batch of data
            checkpoint_time (float): Time of last submission
        Returns:
            bool: True if batch should be submitted
        """
        return len(batch) >= self.config.batch_size or (time.time() - checkpoint_time >= 10)

    def submit_batch(self, batch: List[str]):
        """
        Submit a batch of processed data to the server.
        Args:
            batch (List[str]): Batch of data to submit
        """
        if not batch:
            return

        logger.info(f"Submitting batch of {len(batch)} items")
        result = self.http_client.post_data(batch)
    
    def stop_processes(self):
        """Terminate all GPU processes and clean up"""
        for process in self.gpu_processes:
            if process.poll() is None:
                process.terminate()
        self.gpu_processes = []
        logger.info("All GPU processes terminated")

    def run(self, filename: str) -> Optional[str]:
        """
        Main execution method for the node runner.
        Args:
            filename (str): Name of the executable to run
        Returns:
            Optional[str]: Error message if any, None on success
        """
        output_file = "output.txt"
        try:
            executable_path = "/app/Seeker_CUDA_12"
            self.launch_gpu_processes(executable_path, output_file)

            if not self.gpu_processes:
                return "No GPU processes were launched because no GPUs were detected"

            self.process_output(output_file)
            return None
        except Exception as e:
            error_msg = f"Error in node_runner: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg
        finally:
            self.stop_processes()
            self.http_client.close()


def write_search_prefixes(prefixes: str):
    """
    Write search prefixes to the input file.
    Args:
        prefixes (str): Comma-separated list of prefixes
    """
    prefix_list = prefixes.split(',')
    with open('prefixes.txt', 'w') as file:
        for prefix in prefix_list:
            file.write(f"{prefix}\n")


class Watchdog:
    """
    Watchdog class to monitor server status and search tasks.
    Controls the GPU processes based on server conditions.
    """
    def __init__(self, config: Config, http_client: HTTPClient):
        self.config = config
        self.http_client = http_client
        self.current_search_prefix = None
        self.runner = None
        self.runner_thread = None
        self.stop_requested = False

    def check_server_status(self) -> bool:
        """Check server status and return whether to continue"""
        try:
            return self.http_client.check_status()
        except Exception as e:
            logger.error(f"Error checking server status: {e}")
            return False

    def get_current_task(self) -> Optional[str]:
        """Get current search task from server"""
        try:
            return self.http_client.get_search_prefix()
        except Exception as e:
            logger.error(f"Error getting search task: {e}")
            return None

    def start_gpu_search(self, search_prefix: str):
        """Start GPU search with given prefix"""
        if search_prefix and len(search_prefix.split(',')) > 0:
            write_search_prefixes(search_prefix)
            self.current_search_prefix = search_prefix
            self.runner = NodeRunner(self.config)
            self.runner_thread = threading.Thread(
                target=self.runner.run,
                args=("Seeker_CUDA",)
            )
            self.runner_thread.start()
            logger.info(f"Started GPU search with prefix: {search_prefix}")
        else:
            logger.warning("Invalid search prefix received")

    def stop_gpu_search(self):
        """Stop current GPU search if running"""
        if self.runner:
            self.runner.stop_requested = True
            self.runner.stop_processes()
            if self.runner_thread:
                self.runner_thread.join(timeout=5)
            self.runner = None
            self.runner_thread = None
            logger.info("Stopped GPU search")

    def run(self):
        """Main watchdog loop"""
        while not self.stop_requested:
            # Check server status first
            status = self.check_server_status()
            if not status:
                logger.info("Server status is False, stopping GPU search")
                self.stop_gpu_search()
                time.sleep(3)
                continue

            # Get current search task
            new_search_prefix = self.get_current_task()

            # Skip if it's just a newline character (0x0a) or invalid
            if (not new_search_prefix) or (len(new_search_prefix.strip()) == 0):
                logger.info("No valid search task available or received empty task")
                self.stop_gpu_search()
                time.sleep(3)
            # If we have a valid new task and it's different from current
            elif new_search_prefix != self.current_search_prefix:
                logger.info("New search task received")
                self.stop_gpu_search()  # Stop current search if running
                self.start_gpu_search(new_search_prefix)
            
            # Brief sleep before next check
            time.sleep(self.config.status_check_interval)

    def stop(self):
        """Stop the watchdog"""
        self.stop_requested = True
        self.stop_gpu_search()


def main():
    """Main entry point for the application"""
    try:
        config = Config()
        client = HTTPClient(config)
        
        # Initialize and start watchdog
        watchdog = Watchdog(config, client)
        
        try:
            watchdog.run()
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")
        finally:
            watchdog.stop()
            client.close()

    except Exception as e:
        logger.error("Main process error", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
# gemini_parallel_api.py

import os
import time
import logging
import threading
import concurrent.futures
import traceback
from google import genai
from google.genai import types
from google.api_core import exceptions as google_exceptions
import dotenv

dotenv.load_dotenv()


# --- Constants ---
# Markers for internal communication about API call outcomes
EXHAUSTED_MARKER = "RESOURCE_EXHAUSTED"
PERSISTENT_ERROR_MARKER = "PERSISTENT_ERROR"
ALL_KEYS_WAITING_MARKER = "ALL_KEYS_WAITING"

# Key status markers for the manager
KEY_STATUS_AVAILABLE = "AVAILABLE"
KEY_STATUS_COOLDOWN = "COOLDOWN"  # New status: cooling down after use
KEY_STATUS_TEMPORARILY_EXHAUSTED = "TEMPORARILY_EXHAUSTED"  # New status: temporarily exhausted
KEY_STATUS_FULLY_EXHAUSTED = "FULLY_EXHAUSTED"  # New status: fully exhausted
KEY_STATUS_FAILED_INIT = "FAILED_INIT"

# Default configuration for the parallel processor
DEFAULT_MAX_WORKERS = 4
DEFAULT_WORKER_WAIT_SECONDS = 10 # How long workers wait when all keys are exhausted
DEFAULT_API_CALL_RETRIES = 3 # Retries for non-exhaustion errors within a single API call attempt
DEFAULT_KEY_COOLDOWN_SECONDS = 60  # Cooldown time after key usage (1 minute)
DEFAULT_EXHAUSTED_WAIT_SECONDS = 60  # Wait time for temporary exhaustion (1 minute)
DEFAULT_FULLY_EXHAUSTED_WAIT_SECONDS = 12 * 3600  # Wait time for full exhaustion (12 hours)
DEFAULT_MAX_EXHAUSTED_RETRIES = 3  # Maximum retry count before becoming fully exhausted

# Configure logging for the module
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s",
)


class AdvancedApiKeyManager:
    """
    Advanced API key manager: supports cooldown, staged exhausted states, and time-based recovery.
    """
    def __init__(self, keylist_names, 
                 key_cooldown_seconds=DEFAULT_KEY_COOLDOWN_SECONDS,
                 exhausted_wait_seconds=DEFAULT_EXHAUSTED_WAIT_SECONDS,
                 fully_exhausted_wait_seconds=DEFAULT_FULLY_EXHAUSTED_WAIT_SECONDS,
                 max_exhausted_retries=DEFAULT_MAX_EXHAUSTED_RETRIES):
        """
        Initialize the advanced API key manager.

        Args:
            keylist_names (list[str]): List of environment variable names containing API keys
            key_cooldown_seconds (int): Cooldown time after key usage (seconds)
            exhausted_wait_seconds (int): Wait time for temporary exhaustion (seconds)
            fully_exhausted_wait_seconds (int): Wait time for full exhaustion (seconds)
            max_exhausted_retries (int): Maximum retry count before becoming fully exhausted
        """
        self.key_cooldown_seconds = key_cooldown_seconds
        self.exhausted_wait_seconds = exhausted_wait_seconds
        self.fully_exhausted_wait_seconds = fully_exhausted_wait_seconds
        self.max_exhausted_retries = max_exhausted_retries
        
        self.api_keys = self._load_keys(keylist_names)
        if not self.api_keys:
            raise ValueError("No valid API keys found from provided environment variables.")

        # Track detailed information for each key
        self.key_info = {}
        for key in self.api_keys:
            self.key_info[key] = {
                'status': KEY_STATUS_AVAILABLE,
                'last_used_time': 0,  # Last usage time
                'status_change_time': 0,  # Status change time
                'exhausted_count': 0,  # Consecutive exhausted count
                'total_exhausted_count': 0,  # Total exhausted count
            }
        
        self.num_keys = len(self.api_keys)
        self._lock = threading.Lock()
        self._next_key_index = 0

        logging.info(f"AdvancedApiKeyManager initialized with {self.num_keys} keys.")
        logging.info(f"Settings - Cooldown: {self.key_cooldown_seconds}s, "
                    f"Exhausted wait: {self.exhausted_wait_seconds}s, "
                    f"Fully exhausted wait: {self.fully_exhausted_wait_seconds}s")

    def _load_keys(self, keylist_names):
        """Load API keys from environment variables."""
        keys = []
        for key_name in keylist_names:
            key = os.getenv(key_name)
            if key and len(key) > 10:
                keys.append(key)
                logging.debug(f"Loaded key from {key_name}.")
            else:
                logging.warning(f"Environment variable '{key_name}' not found or invalid.")
        logging.info(f"Successfully loaded {len(keys)} valid API keys.")
        return keys

    def _update_key_status_based_on_time(self):
        """Update key statuses based on time."""
        current_time = time.time()
        
        for key, info in self.key_info.items():
            if info['status'] == KEY_STATUS_COOLDOWN:
                # Check if cooldown time has passed
                if current_time - info['status_change_time'] >= self.key_cooldown_seconds:
                    info['status'] = KEY_STATUS_AVAILABLE
                    logging.debug(f"Key ...{key[-4:]} cooldown finished, now AVAILABLE")
            
            elif info['status'] == KEY_STATUS_TEMPORARILY_EXHAUSTED:
                # Check if temporary exhaustion time has passed
                if current_time - info['status_change_time'] >= self.exhausted_wait_seconds:
                    info['status'] = KEY_STATUS_AVAILABLE
                    logging.info(f"Key ...{key[-4:]} temporary exhaustion recovered, now AVAILABLE")
            
            elif info['status'] == KEY_STATUS_FULLY_EXHAUSTED:
                # Check if full exhaustion time has passed
                if current_time - info['status_change_time'] >= self.fully_exhausted_wait_seconds:
                    info['status'] = KEY_STATUS_AVAILABLE
                    info['exhausted_count'] = 0  # Reset count
                    logging.info(f"Key ...{key[-4:]} full exhaustion recovered, now AVAILABLE")

    def get_next_available_key(self):
        """
        Return the next available API key.
        Considers cooldown and exhausted states for selection.

        Returns:
            str: Available API key
            str: ALL_KEYS_WAITING_MARKER if all keys are temporarily unavailable
            None: if no usable keys are available
        """
        with self._lock:
            # Time-based status update
            self._update_key_status_based_on_time()
            
            # Find available key (round robin)
            available_key_found = None
            initial_index = self._next_key_index
            
            for i in range(self.num_keys):
                current_index = (initial_index + i) % self.num_keys
                key = self.api_keys[current_index]
                info = self.key_info[key]
                
                if info['status'] == KEY_STATUS_AVAILABLE:
                    available_key_found = key
                    self._next_key_index = (current_index + 1) % self.num_keys
                    
                    # Start key usage - change to cooldown status
                    info['last_used_time'] = time.time()
                    info['status'] = KEY_STATUS_COOLDOWN
                    info['status_change_time'] = time.time()
                    
                    logging.debug(f"Providing key ...{key[-4:]} (now in COOLDOWN for {self.key_cooldown_seconds}s)")
                    break

            if available_key_found is None:
                # Analyze when no available keys exist
                status_counts = {}
                for info in self.key_info.values():
                    status = info['status']
                    status_counts[status] = status_counts.get(status, 0) + 1
                
                if status_counts.get(KEY_STATUS_FAILED_INIT, 0) == self.num_keys:
                    logging.error("FATAL: All API keys failed initialization.")
                    return None
                
                # Temporarily unavailable situation
                logging.info(f"No available keys. Status: {status_counts}")
                return ALL_KEYS_WAITING_MARKER

            return available_key_found

    def mark_key_exhausted(self, api_key):
        """
        Mark key as exhausted.
        Classify as temporary or full exhaustion based on consecutive exhausted count.
        """
        with self._lock:
            if api_key not in self.key_info:
                logging.error(f"Unknown key marked as exhausted: {api_key}")
                return
            
            info = self.key_info[api_key]
            info['exhausted_count'] += 1
            info['total_exhausted_count'] += 1
            current_time = time.time()
            
            masked_key = f"...{api_key[-4:]}" if len(api_key) > 4 else "invalid key"
            
            if info['exhausted_count'] >= self.max_exhausted_retries:
                # Change to fully exhausted status
                info['status'] = KEY_STATUS_FULLY_EXHAUSTED
                info['status_change_time'] = current_time
                wait_hours = self.fully_exhausted_wait_seconds / 3600
                logging.warning(
                    f"Key {masked_key} marked as FULLY_EXHAUSTED "
                    f"(count: {info['exhausted_count']}) - waiting {wait_hours:.1f}h"
                )
            else:
                # Change to temporarily exhausted status
                info['status'] = KEY_STATUS_TEMPORARILY_EXHAUSTED
                info['status_change_time'] = current_time
                wait_minutes = self.exhausted_wait_seconds / 60
                logging.warning(
                    f"Key {masked_key} marked as TEMPORARILY_EXHAUSTED "
                    f"(count: {info['exhausted_count']}) - waiting {wait_minutes:.1f}m"
                )

    def mark_key_successful(self, api_key):
        """
        Called when key usage is successful.
        Resets exhausted count.
        """
        with self._lock:
            if api_key not in self.key_info:
                return
            
            info = self.key_info[api_key]
            if info['exhausted_count'] > 0:
                logging.info(f"Key ...{api_key[-4:]} successful, resetting exhausted count from {info['exhausted_count']} to 0")
                info['exhausted_count'] = 0

    def mark_key_failed_init(self, api_key):
        """Mark key initialization failure."""
        with self._lock:
            if api_key not in self.key_info:
                return
            
            self.key_info[api_key]['status'] = KEY_STATUS_FAILED_INIT
            masked_key = f"...{api_key[-4:]}" if len(api_key) > 4 else "invalid key"
            logging.error(f"Key {masked_key} marked as FAILED_INIT")

    def get_keys_status_summary(self):
        """Return status summary of all keys."""
        with self._lock:
            self._update_key_status_based_on_time()
            
            summary = {}
            for key, info in self.key_info.items():
                masked_key = f"...{key[-4:]}"
                summary[masked_key] = {
                    'status': info['status'],
                    'exhausted_count': info['exhausted_count'],
                    'total_exhausted_count': info['total_exhausted_count']
                }
            
            return summary

class GeminiParallelProcessor:
    """
    Manages parallel calls to the Gemini API using a DynamicApiKeyManager.
    It handles API key rotation, resource exhaustion retries, and general API errors.
    """
    def __init__(self, key_manager: AdvancedApiKeyManager, model_name: str,
                 api_call_interval: float = 0.0, max_workers: int = DEFAULT_MAX_WORKERS):
        """
        Initializes the parallel processor.

        Args:
            key_manager (AdvancedApiKeyManager): An instance of the API key manager.
            model_name (str): The name of the Gemini model to use (e.g., "gemini-2.0-flash-001").
            api_call_interval (float): Minimum time (in seconds) to wait between
                                       consecutive API calls made by a single worker.
                                       Helps with per-key rate limits.
            max_workers (int): The maximum number of parallel threads to use.
        """
        self.key_manager = key_manager
        self.model_name = model_name
        self.api_call_interval = api_call_interval
        self.max_workers = max_workers
        logging.info(
            f"GeminiParallelProcessor initialized for model '{self.model_name}' "
            f"with {self.max_workers} workers and interval {self.api_call_interval}s."
        )

    def _make_single_api_call(self, client_instance, prompt_data: dict) -> str:
        """
        Executes a single API call to the Gemini model.
        Handles retries for non-quota errors.
        Supports both text-only and text+audio prompts.

        Args:
            client_instance: An initialized genai.Client instance.
            prompt_data (dict): Dictionary containing:
                - 'prompt' (str): The text prompt
                - 'audio_path' (str, optional): Path to audio file
                - 'audio_bytes' (bytes, optional): Audio bytes
                - 'video_url' (str, optional): URL of video file
                - 'video_path' (str, optional): Path to video file
                - 'video_bytes' (bytes, optional): Video bytes
                - 'audio_mime_type' (str, optional): MIME type of audio file (e.g., 'audio/mp3')
                - 'video_mime_type' (str, optional): MIME type of video file (e.g., 'video/mp4')
                - 'video_metadata' (dict, optional): Metadata for video file (e.g., {start_offset='1250s', end_offset='1570s', fps=5})
                - 'generation_config' (dict, optional): Generation config for the API call (refer to "https://ai.google.dev/api/generate-content?#generationconfig")
        Instructions:
            videos and audios bigger than 20MB are recommended to be uploaded with video_path, else use video_bytes.

        Returns:
            str: The raw text response from the Gemini model on success.
            str: `EXHAUSTED_MARKER` if a ResourceExhausted error occurs.
            str: `PERSISTENT_ERROR_MARKER` if other errors persist after retries.
        """
        prompt = prompt_data.get('prompt', '')
        audio_path = prompt_data.get('audio_path')
        audio_bytes = prompt_data.get('audio_bytes')
        audio_mime_type = prompt_data.get('audio_mime_type', 'audio/mp3')
        video_url = prompt_data.get('video_url')
        video_path = prompt_data.get('video_path')
        video_bytes = prompt_data.get('video_bytes')
        video_mime_type = prompt_data.get('video_mime_type', 'video/mp4')
        video_metadata = prompt_data.get('video_metadata', {})
        generation_config = prompt_data.get('generation_config', {})

        # Prepare contents for API call
        contents = []

        if video_url:
            try:
                video_part = types.Part(
                    file_data=types.FileData(file_url=video_url),
                    video_metadata=types.VideoMetadata(**video_metadata)
                )
                contents.append(video_part)
                logging.debug(f"Added video file: {video_url}")
            except Exception as e:
                logging.error(f"Failed to read video file {video_url}: {e}")
                return PERSISTENT_ERROR_MARKER
        elif video_path and os.path.exists(video_path):
            try:
                video_file = client_instance.files.upload(file=video_path)
                contents.append(video_file)
                logging.debug(f"Added video file: {video_path}")
            except Exception as e:
                logging.error(f"Failed to upload video file {video_path}: {e}")
                return PERSISTENT_ERROR_MARKER
        elif video_bytes:
            try:
                video_part = types.Part(
                    inline_data=types.Blob(data=video_bytes, mime_type=video_mime_type),
                    video_metadata=types.VideoMetadata(**video_metadata)
                )
                contents.append(video_part)
                logging.debug(f"Added video bytes: {video_mime_type}")
            except Exception as e:
                logging.error(f"Failed to create video part from bytes: {e}")
                return PERSISTENT_ERROR_MARKER

        # Add audio if provided
        if audio_path and os.path.exists(audio_path):
            try:
                audio_file = client_instance.files.upload(file=audio_path)
                contents.append(audio_file)
                logging.debug(f"Added audio file: {audio_path})")
                
            except Exception as e:
                logging.error(f"Failed to read audio file {audio_path}: {e}")
                return PERSISTENT_ERROR_MARKER
        elif audio_bytes:
            try:
                audio_part = types.Part.from_bytes(
                    data=audio_bytes,
                    mime_type=audio_mime_type
                )
                contents.append(audio_part)
                logging.debug(f"Added audio bytes: {audio_mime_type}")
            except Exception as e:
                logging.error(f"Failed to create audio part from bytes: {e}")
                return PERSISTENT_ERROR_MARKER

        # Add text prompt if provided
        if prompt:
            contents.append(prompt)
        
        # Ensure we have some content
        if not contents:
            logging.error("No content provided (neither prompt nor audio)")
            return PERSISTENT_ERROR_MARKER
        
        # Perform API call with retries
        retries = 0
        while retries < DEFAULT_API_CALL_RETRIES:
            response = None
            try:
                response = client_instance.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=genai.types.GenerateContentConfig(**generation_config) # Example config if needed
                )
                response_text = response.text.strip()
                content_type = "text+audio" if audio_path else "text-only"
                logging.debug(f"API call successful ({content_type}). Response length: {len(response_text)}.")
                return response_text

            except google_exceptions.ResourceExhausted as e:
                logging.warning(f"ResourceExhausted error: {e}. Signaling exhaustion.")
                return EXHAUSTED_MARKER
            except (google_exceptions.GoogleAPIError, ValueError, AttributeError) as e:
                logging.warning(
                    f"{type(e).__name__} during API call: {e}. "
                    f"Retry {retries + 1}/{DEFAULT_API_CALL_RETRIES}..."
                )
                if isinstance(e, AttributeError) and response is not None and not hasattr(response, "text"):
                    logging.error(
                        "Response object does not have a 'text' attribute. Available attributes: %s",
                        dir(response)
                    )
            except Exception as e:
                logging.error(
                    f"Unexpected error during API call: {type(e).__name__} - {e}. "
                    f"Traceback: {traceback.format_exc()}. "
                    f"Retry {retries + 1}/{DEFAULT_API_CALL_RETRIES}..."
                )

            retries += 1
            if retries < DEFAULT_API_CALL_RETRIES:
                wait_time = 2**retries # Exponential backoff
                logging.info(f"Waiting {wait_time}s before retrying API call...")
                time.sleep(wait_time)
            else:
                logging.error(
                    f"Failed API call after {DEFAULT_API_CALL_RETRIES} retries."
                )
                return PERSISTENT_ERROR_MARKER

        return PERSISTENT_ERROR_MARKER # Should not be reached if loop conditions are correct

    def _worker_task(self, prompt_data: dict) -> tuple:
        """
        Worker function executed by the thread pool.
        Gets an API key, initializes a client, makes the API call,
        and handles key-specific exhaustion retries with the new advanced key management.

        Args:
            prompt_data (dict): A dictionary containing 'prompt' (str) and
                                'metadata' (dict) for the task.

        Returns:
            tuple: (metadata_dict, api_response_text_or_marker, error_message_str)
        """
        prompt = prompt_data['prompt']
        metadata = prompt_data['metadata']
        task_id = metadata.get('task_id', 'unknown_task')

        if not prompt:
            logging.warning(f"Skipping task {task_id} due to empty prompt.")
            return metadata, None, "Empty prompt provided."

        # Key acquisition and retry loop
        max_key_switches = 5  # Maximum number of key switches
        key_switch_count = 0
        
        while key_switch_count < max_key_switches:
            api_key = self.key_manager.get_next_available_key()

            if api_key == ALL_KEYS_WAITING_MARKER:
                logging.info(f"Worker for {task_id} waiting - no available keys")
                time.sleep(DEFAULT_WORKER_WAIT_SECONDS)
                continue
            elif api_key is None:
                logging.error(f"Worker for {task_id} - FATAL: No usable keys")
                return metadata, None, "Fatal: No usable API keys available."

            # Client initialization
            masked_key = f"...{api_key[-4:]}" if len(api_key) > 4 else "invalid key"
            try:
                logging.debug(f"Worker for {task_id} using key {masked_key}")
                client_instance = genai.Client(api_key=api_key)

                if self.api_call_interval > 0:
                    time.sleep(self.api_call_interval)

            except Exception as e:
                logging.error(f"Failed to initialize client for {task_id} with key {masked_key}: {e}")
                self.key_manager.mark_key_failed_init(api_key)
                key_switch_count += 1
                continue

            # Perform API call
            result = self._make_single_api_call(client_instance, prompt_data)
            
            if result == EXHAUSTED_MARKER:
                # Key is exhausted - notify key manager and retry with different key
                logging.warning(f"Key {masked_key} exhausted for task {task_id}, trying different key")
                self.key_manager.mark_key_exhausted(api_key)
                key_switch_count += 1
                continue
            elif result == PERSISTENT_ERROR_MARKER:
                # Persistent error - treat this task as failed
                logging.error(f"Persistent error for {task_id} with key {masked_key}")
                return metadata, None, "Persistent API call error."
            else:
                # Success! - reset exhausted count and return result
                logging.debug(f"Success for {task_id} with key {masked_key}")
                self.key_manager.mark_key_successful(api_key)
                return metadata, result, None

        # Maximum key switch count exceeded
        logging.error(f"Task {task_id} failed after {max_key_switches} key switches")
        return metadata, None, f"Failed after {max_key_switches} key switches"

    def process_prompts(self, prompts_with_metadata: list[dict]) -> list[tuple]:
        """
        Processes a list of prompts in parallel using the managed API keys.
        Supports both text-only and text+audio inputs.

        Args:
            prompts_with_metadata (list[dict]): A list of dictionaries, where each
                                                dictionary can contain:
                                                - 'prompt' (str): The text prompt to send to Gemini.
                                                - 'audio_path' (str, optional): Path to audio file.
                                                - 'audio_mime_type' (str, optional): MIME type of audio (default: 'audio/mp3').
                                                - 'video_url' (str, optional): URL of video file.
                                                - 'video_path' (str, optional): Path to video file.
                                                - 'video_bytes' (bytes, optional): Video bytes.
                                                - 'video_mime_type' (str, optional): MIME type of video (default: 'video/mp4').
                                                - 'video_metadata' (dict, optional): Metadata for video file (e.g., {'start_offset='1250s', end_offset='1570s', fps=5}).
                                                - 'generation_config' (dict, optional): Generation config for the API call (refer to "https://ai.google.dev/api/generate-content?#generationconfig").
                                                - 'metadata' (dict): A dictionary of any
                                                  additional data associated with this prompt
                                                  (e.g., original line index, task info).
                                                  It's recommended to include a 'task_id' for logging.

        Returns:
            list[tuple]: A list of tuples, where each tuple contains:
                         (metadata_dict, api_response_text_or_none, error_message_str_or_none).
                         The `metadata_dict` is the original metadata passed in.
                         `api_response_text_or_none` is the raw text response from Gemini on success,
                         or None if an error occurred.
                         `error_message_str_or_none` is None on success, or a string describing the error.
        """
        if not prompts_with_metadata:
            logging.info("No prompts to process.")
            return []

        # Determine actual number of workers based on available keys and max_workers setting
        # We need at least one key for any worker to start.
        actual_workers = min(self.max_workers, len(prompts_with_metadata), self.key_manager.num_keys)
        if actual_workers == 0:
            logging.error("No workers can be started: No prompts, no keys, or max_workers is 0.")
            return []

        logging.info(f"Starting parallel processing with {actual_workers} workers for {len(prompts_with_metadata)} prompts.")
        results = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=actual_workers, thread_name_prefix="GeminiAPIWorker"
        ) as executor:
            # Map each prompt_data dictionary to the _worker_task
            futures = {
                executor.submit(self._worker_task, prompt_data): prompt_data['metadata']
                for prompt_data in prompts_with_metadata
            }

            processed_count = 0
            last_log_time = time.time()
            for future in concurrent.futures.as_completed(futures):
                original_metadata = futures[future]
                task_id = original_metadata.get('task_id', 'unknown_task')
                try:
                    metadata_res, api_response_text, error_msg = future.result()
                    results.append((metadata_res, api_response_text, error_msg))
                except Exception as exc:
                    logging.error(
                        f"Task for {task_id} generated an unhandled exception: {exc}",
                        exc_info=True,
                    )
                    results.append((original_metadata, None, f"Unhandled worker exception: {exc}"))

                processed_count += 1
                current_time = time.time()
                # Log progress periodically
                if processed_count % 50 == 0 or current_time - last_log_time > 30:
                    logging.info(
                        f"Progress: {processed_count}/{len(prompts_with_metadata)} tasks completed."
                    )
                    last_log_time = current_time

        logging.info(f"Parallel processing finished. Processed {processed_count} tasks.")
        return results
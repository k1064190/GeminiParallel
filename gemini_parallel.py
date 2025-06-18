# gemini_parallel_api.py

import os
import time
import logging
import threading
import concurrent.futures
import traceback
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from google.api_core import exceptions as google_exceptions
import dotenv
import queue
import uuid

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
DEFAULT_KEY_COOLDOWN_SECONDS = 30  # Cooldown time after key usage (30 seconds)
DEFAULT_EXHAUSTED_WAIT_SECONDS = 120  # Wait time for temporary exhaustion (120 seconds)
DEFAULT_FULLY_EXHAUSTED_WAIT_SECONDS = 12 * 3600  # Wait time for full exhaustion (12 hours)
DEFAULT_MAX_EXHAUSTED_RETRIES = 3  # Maximum retry count before becoming fully exhausted
DEFAULT_ERROR_RETRY_DELAY = 30 # Delay between error retries (30 seconds)

# Configure logging for the module
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s",
)


class AdvancedApiKeyManager:
    """
    Advanced API key manager: supports worker-specific key assignment with cooldown, 
    staged exhausted states, and time-based recovery.
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
                'assigned_worker': None,  # Which worker is using this key
            }
        
        self.num_keys = len(self.api_keys)
        
        # Worker-specific key assignments
        self.worker_assignments = {}  # worker_id -> api_key
        self.available_keys = set(self.api_keys)  # Keys not assigned to any worker
        
        self._lock = threading.Lock()

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

    def assign_key_to_worker(self, worker_id: str):
        """
        Assign a key to a specific worker. Each worker gets a dedicated key.
        
        Args:
            worker_id (str): Unique identifier for the worker
            
        Returns:
            str: API key assigned to the worker
            str: ALL_KEYS_WAITING_MARKER if no keys are available
            None: if no usable keys exist
        """
        with self._lock:
            # Update key statuses based on time
            self._update_key_status_based_on_time()
            
            # Check if worker already has a key assigned
            if worker_id in self.worker_assignments:
                assigned_key = self.worker_assignments[worker_id]
                key_info = self.key_info[assigned_key]
                
                # If assigned key is still usable (not FULLY_EXHAUSTED or FAILED_INIT), keep it
                if key_info['status'] not in [KEY_STATUS_FULLY_EXHAUSTED, KEY_STATUS_FAILED_INIT]:
                    logging.debug(f"Worker {worker_id} keeping assigned key ...{assigned_key[-4:]} (status: {key_info['status']})")
                    return assigned_key
                else:
                    # Release the unusable key and find a new one
                    logging.info(f"Worker {worker_id} releasing unusable key ...{assigned_key[-4:]} (status: {key_info['status']})")
                    self._release_key_from_worker(worker_id, assigned_key)
            
            # Find an available key for assignment
            available_key = None
            for key in self.available_keys.copy():
                key_info = self.key_info[key]
                if key_info['status'] not in [KEY_STATUS_FULLY_EXHAUSTED, KEY_STATUS_FAILED_INIT]:
                    available_key = key
                    break
            
            if available_key is None:
                # Check if any assigned keys can be reassigned (if their worker is done)
                for key, info in self.key_info.items():
                    if (info['assigned_worker'] is None and 
                        info['status'] not in [KEY_STATUS_FULLY_EXHAUSTED, KEY_STATUS_FAILED_INIT]):
                        available_key = key
                        self.available_keys.add(key)
                        break
            
            if available_key is None:
                # No usable keys available
                status_counts = {}
                for info in self.key_info.values():
                    status = info['status']
                    status_counts[status] = status_counts.get(status, 0) + 1
                
                if status_counts.get(KEY_STATUS_FAILED_INIT, 0) == self.num_keys:
                    logging.error("FATAL: All API keys failed initialization.")
                    return None
                
                logging.info(f"Worker {worker_id} waiting - no available keys. Status: {status_counts}")
                return ALL_KEYS_WAITING_MARKER
            
            # Assign the key to the worker
            self.worker_assignments[worker_id] = available_key
            self.key_info[available_key]['assigned_worker'] = worker_id
            self.available_keys.discard(available_key)
            
            logging.info(f"Assigned key ...{available_key[-4:]} to worker {worker_id}")
            return available_key
    
    def _release_key_from_worker(self, worker_id: str, api_key: str):
        """Internal method to release a key from a worker."""
        if worker_id in self.worker_assignments and self.worker_assignments[worker_id] == api_key:
            del self.worker_assignments[worker_id]
            self.key_info[api_key]['assigned_worker'] = None
            self.available_keys.add(api_key)
            logging.debug(f"Released key ...{api_key[-4:]} from worker {worker_id}")

    def release_key_from_worker(self, worker_id: str, api_key: str):
        """
        Release a key from a worker (public method).
        
        Args:
            worker_id (str): Worker identifier
            api_key (str): API key to release
        """
        with self._lock:
            self._release_key_from_worker(worker_id, api_key)

    def check_key_status(self, api_key: str) -> str:
        """
        Check the current status of a specific API key.
        
        Args:
            api_key (str): The API key to check
            
        Returns:
            str: Current status of the key
        """
        with self._lock:
            self._update_key_status_based_on_time()
            if api_key in self.key_info:
                return self.key_info[api_key]['status']
            return KEY_STATUS_FAILED_INIT

    def can_use_key_now(self, api_key: str) -> bool:
        """
        Check if a key can be used immediately (not in cooldown or temporarily exhausted).
        
        Args:
            api_key (str): The API key to check
            
        Returns:
            bool: True if key can be used now, False otherwise
        """
        status = self.check_key_status(api_key)
        return status == KEY_STATUS_AVAILABLE

    def mark_key_used(self, api_key: str):
        """
        Mark a key as just used (put it in cooldown).
        
        Args:
            api_key (str): The API key that was used
        """
        with self._lock:
            if api_key not in self.key_info:
                logging.error(f"Unknown key marked as used: {api_key}")
                return
            
            info = self.key_info[api_key]
            info['last_used_time'] = time.time()
            info['status'] = KEY_STATUS_COOLDOWN
            info['status_change_time'] = time.time()
            
            logging.debug(f"Key ...{api_key[-4:]} marked as used, now in COOLDOWN for {self.key_cooldown_seconds}s")

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
                    'total_exhausted_count': info['total_exhausted_count'],
                    'assigned_worker': info['assigned_worker']
                }
            
            return summary

class GeminiParallelProcessor:
    """
    Manages parallel calls to the Gemini API using a DynamicApiKeyManager.
    It handles API key rotation, resource exhaustion retries, and general API errors.
    """
    def __init__(self, key_manager: AdvancedApiKeyManager, model_name: str,
                 api_call_interval: float = 2.0, max_workers: int = DEFAULT_MAX_WORKERS):
        """
        Initializes the parallel processor.

        Args:
            key_manager (AdvancedApiKeyManager): An instance of the API key manager.
            model_name (str): The name of the Gemini model to use (e.g., "gemini-2.0-flash-001").
            api_call_interval (float): Minimum time (in seconds) to wait between
                                       consecutive API calls made by ANY worker.
                                       Ensures sequential timing across all workers.
            max_workers (int): The maximum number of parallel threads to use. Recommended to be less or equal to 4.
        """
        self.key_manager = key_manager
        self.model_name = model_name
        self.api_call_interval = api_call_interval
        self.max_workers = max_workers
        
        # Global API call timing control
        self._last_api_call_time = 0.0
        self._api_call_lock = threading.Lock()
        
        logging.info(
            f"GeminiParallelProcessor initialized for model '{self.model_name}' "
            f"with {self.max_workers} workers and global interval {self.api_call_interval}s."
        )

    def _parse_prompt_with_media_tokens(self, prompt: str, audio_files: list, video_files: list) -> list:
        """
        Parse prompt containing <audio> and <video> tokens and construct a content sequence.
        
        Args:
            prompt (str): Text prompt containing <audio> and <video> tokens
            audio_files: List of audio file objects/parts
            video_files: List of video file objects/parts
            
        Returns:
            list: Ordered list of content parts (text and media)
        """
        import re
        
        contents = []
        audio_index = 0
        video_index = 0
        
        # Find all tokens with their positions
        tokens = []
        for match in re.finditer(r'<(audio|video)>', prompt):
            tokens.append({
                'type': match.group(1),
                'start': match.start(),
                'end': match.end()
            })
        
        # Sort tokens by position
        tokens.sort(key=lambda x: x['start'])
        
        # Split text and insert media
        current_pos = 0
        
        for token in tokens:
            # Add text before token (if any)
            text_before = prompt[current_pos:token['start']].strip()
            if text_before:
                contents.append(text_before)
            
            # Add corresponding media file
            if token['type'] == 'audio' and audio_index < len(audio_files):
                contents.append(audio_files[audio_index])
                audio_index += 1
                logging.debug(f"Added audio file at position {len(contents)-1}")
            elif token['type'] == 'video' and video_index < len(video_files):
                contents.append(video_files[video_index])
                video_index += 1
                logging.debug(f"Added video file at position {len(contents)-1}")
            else:
                logging.warning(f"No {token['type']} file available for token at position {token['start']}")
            
            current_pos = token['end']
        
        # Add remaining text after last token
        remaining_text = prompt[current_pos:].strip()
        if remaining_text:
            contents.append(remaining_text)
        
        # Add any unused audio files at the end
        while audio_index < len(audio_files):
            contents.append(audio_files[audio_index])
            audio_index += 1
            logging.debug(f"Added unused audio file at end")
        
        # Add any unused video files at the end
        while video_index < len(video_files):
            contents.append(video_files[video_index])
            video_index += 1
            logging.debug(f"Added unused video file at end")
        
        return contents

    def _make_single_api_call(self, client_instance, prompt_data: dict) -> str:
        """
        Executes a single API call to the Gemini model.
        Handles retries for non-quota errors.
        Supports both text-only and text+audio/video prompts with position specification.

        Args:
            client_instance: An initialized genai.Client instance.
            prompt_data (dict): Dictionary containing:
                - 'prompt' (str): The text prompt (can contain <audio> and <video> tokens for positioning)
                - 'audio_path' (str or list[str], optional): Path(s) to audio file(s)
                - 'audio_bytes' (bytes or list[bytes], optional): Audio bytes
                - 'video_url' (str or list[str], optional): URL(s) of video file(s)
                - 'video_path' (str or list[str], optional): Path(s) to video file(s)
                - 'video_bytes' (bytes or list[bytes], optional): Video bytes
                - 'audio_mime_type' (str or list[str], optional): MIME type(s) of audio file(s) (e.g., 'audio/mp3')
                - 'video_mime_type' (str or list[str], optional): MIME type(s) of video file(s) (e.g., 'video/mp4')
                - 'video_metadata' (dict or list[dict], optional): Metadata for video file(s)
                - 'generation_config' (dict, optional): Generation config for the API call
        
        Instructions:
            - Use <audio> and <video> tokens in prompt to specify positioning
            - Multiple tokens are supported and will be matched with files in order
            - Videos and audios bigger than 20MB are recommended to be uploaded with paths
            
        Returns:
            str: The raw text response from the Gemini model on success.
            str: `EXHAUSTED_MARKER` if a ResourceExhausted error occurs.
            str: `PERSISTENT_ERROR_MARKER` if other errors persist after retries.
        """
        prompt = prompt_data.get('prompt', '')
        
        # Handle both single values and lists for all media parameters
        def ensure_list(value):
            if value is None:
                return []
            return value if isinstance(value, list) else [value]
        
        audio_paths = ensure_list(prompt_data.get('audio_path'))
        audio_bytes_list = ensure_list(prompt_data.get('audio_bytes'))
        audio_mime_types = ensure_list(prompt_data.get('audio_mime_type', 'audio/mp3'))
        video_urls = ensure_list(prompt_data.get('video_url'))
        video_paths = ensure_list(prompt_data.get('video_path'))
        video_bytes_list = ensure_list(prompt_data.get('video_bytes'))
        video_mime_types = ensure_list(prompt_data.get('video_mime_type', 'video/mp4'))
        video_metadata_list = ensure_list(prompt_data.get('video_metadata', {}))
        generation_config = prompt_data.get('generation_config', {})

        # Prepare video files
        video_files = []
        
        # Process video URLs
        for i, video_url in enumerate(video_urls):
            try:
                video_metadata = video_metadata_list[i] if i < len(video_metadata_list) else {}
                video_part = types.Part(
                    file_data=types.FileData(file_url=video_url),
                    video_metadata=types.VideoMetadata(**video_metadata)
                )
                video_files.append(video_part)
                logging.debug(f"Added video URL: {video_url}")
            except Exception as e:
                logging.error(f"Failed to process video URL {video_url}: {e}")
                return PERSISTENT_ERROR_MARKER
        
        # Process video paths
        for i, video_path in enumerate(video_paths):
            if os.path.exists(video_path):
                try:
                    video_file = client_instance.files.upload(file=video_path)
                    video_files.append(video_file)
                    logging.debug(f"Added video file: {video_path}")
                except Exception as e:
                    logging.error(f"Failed to upload video file {video_path}: {e}")
                    return PERSISTENT_ERROR_MARKER
            else:
                logging.error(f"Video file not found: {video_path}")
                return PERSISTENT_ERROR_MARKER
        
        # Process video bytes
        for i, video_bytes in enumerate(video_bytes_list):
            try:
                video_mime_type = video_mime_types[i] if i < len(video_mime_types) else 'video/mp4'
                video_metadata = video_metadata_list[i] if i < len(video_metadata_list) else {}
                video_part = types.Part(
                    inline_data=types.Blob(data=video_bytes, mime_type=video_mime_type),
                    video_metadata=types.VideoMetadata(**video_metadata)
                )
                video_files.append(video_part)
                logging.debug(f"Added video bytes: {video_mime_type}")
            except Exception as e:
                logging.error(f"Failed to create video part from bytes: {e}")
                return PERSISTENT_ERROR_MARKER

        # Prepare audio files
        audio_files = []
        
        # Process audio paths
        for i, audio_path in enumerate(audio_paths):
            if os.path.exists(audio_path):
                try:
                    audio_file = client_instance.files.upload(file=audio_path)
                    audio_files.append(audio_file)
                    logging.debug(f"Added audio file: {audio_path}")
                except Exception as e:
                    logging.error(f"Failed to upload audio file {audio_path}: {e}")
                    return PERSISTENT_ERROR_MARKER
            else:
                logging.error(f"Audio file not found: {audio_path}")
                return PERSISTENT_ERROR_MARKER
        
        # Process audio bytes
        for i, audio_bytes in enumerate(audio_bytes_list):
            try:
                audio_mime_type = audio_mime_types[i] if i < len(audio_mime_types) else 'audio/mp3'
                audio_part = types.Part.from_bytes(
                    data=audio_bytes,
                    mime_type=audio_mime_type
                )
                audio_files.append(audio_part)
                logging.debug(f"Added audio bytes: {audio_mime_type}")
            except Exception as e:
                logging.error(f"Failed to create audio part from bytes: {e}")
                return PERSISTENT_ERROR_MARKER

        # Parse prompt and construct contents with proper positioning
        if prompt and ('<audio>' in prompt or '<video>' in prompt):
            # Use token-based positioning
            contents = self._parse_prompt_with_media_tokens(prompt, audio_files, video_files)
        else:
            # Fallback to original behavior: video + audio + text
            contents = []
            contents.extend(video_files)
            contents.extend(audio_files)
            if prompt:
                contents.append(prompt)
        
        # Ensure we have some content
        if not contents:
            logging.error("No content provided (neither prompt nor media files)")
            return PERSISTENT_ERROR_MARKER
        
        # Perform API call with retries
        retries = 0
        wait_time = DEFAULT_ERROR_RETRY_DELAY
        while retries < DEFAULT_API_CALL_RETRIES:
            response = None
            try:
                # Global API call interval control - ensure sequential timing across all workers
                with self._api_call_lock:
                    current_time = time.time()
                    time_since_last_call = current_time - self._last_api_call_time
                    
                    if time_since_last_call < self.api_call_interval:
                        wait_time = self.api_call_interval - time_since_last_call
                        worker_id = threading.current_thread().name
                        logging.debug(f"Worker {worker_id} waiting {wait_time:.2f}s for global API interval")
                        time.sleep(wait_time)
                    
                    # Update last API call time before making the call
                    self._last_api_call_time = time.time()
                
                response = client_instance.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=genai.types.GenerateContentConfig(**generation_config)
                )
                response_text = response.text.strip()
                
                # Log content types for debugging
                media_count = len(audio_files) + len(video_files)
                content_type = f"text+{media_count}media" if media_count > 0 else "text-only"
                logging.debug(f"API call successful ({content_type}). Response length: {len(response_text)}.")
                return response_text
            except genai_errors.APIError as e:
                # Handle different error codes based on official Gemini API documentation
                error_code = getattr(e, 'code', None) or getattr(e, 'status_code', None)
                
                if error_code == 429:  # RESOURCE_EXHAUSTED
                    logging.warning(f"RESOURCE_EXHAUSTED (429): {e}. Signaling exhaustion.")
                    return EXHAUSTED_MARKER
                    
                elif error_code in [400, 403, 404]:  # Non-retryable errors
                    # 400: INVALID_ARGUMENT, FAILED_PRECONDITION
                    # 403: PERMISSION_DENIED  
                    # 404: NOT_FOUND
                    logging.error(f"Non-retryable error ({error_code}): {e}. Signaling persistent error.")
                    return PERSISTENT_ERROR_MARKER
                    
                elif error_code in [500, 503, 504]:  # Retryable server errors
                    # 500: INTERNAL - Google internal error
                    # 503: UNAVAILABLE - Service temporarily overloaded/down
                    # 504: DEADLINE_EXCEEDED - Service couldn't complete in time
                    logging.warning(
                        f"Retryable server error ({error_code}): {e}. "
                        f"Retry {retries + 1}/{DEFAULT_API_CALL_RETRIES}..."
                    )
                else:
                    # Unknown error code - treat as retryable
                    logging.warning(
                        f"Unknown APIError ({error_code}): {e}. "
                        f"Retry {retries + 1}/{DEFAULT_API_CALL_RETRIES}..."
                    )
            except Exception as e:
                logging.error(
                    f"Unexpected error during API call: {type(e).__name__} - {e}. "
                    f"Traceback: {traceback.format_exc()}. "
                    f"Retry {retries + 1}/{DEFAULT_API_CALL_RETRIES}..."
                )

            retries += 1
            if retries < DEFAULT_API_CALL_RETRIES:
                logging.info(f"Waiting {wait_time}s before retrying API call...")
                time.sleep(wait_time)
                wait_time = wait_time * 2**retries # Exponential backoff
            else:
                logging.error(
                    f"Failed API call after {DEFAULT_API_CALL_RETRIES} retries."
                )
                return PERSISTENT_ERROR_MARKER

        return PERSISTENT_ERROR_MARKER # Should not be reached if loop conditions are correct

    def _worker_task(self, prompt_data: dict) -> tuple:
        """
        Worker function executed by the thread pool.
        Each worker gets a dedicated API key and keeps using it until it becomes unusable.
        Workers wait for COOLDOWN and TEMPORARILY_EXHAUSTED keys, but switch when keys are
        FULLY_EXHAUSTED or FAILED_INIT.

        Args:
            prompt_data (dict): A dictionary containing 'prompt' (str) and
                                'metadata' (dict) for the task.

        Returns:
            tuple: (metadata_dict, api_response_text_or_marker, error_message_str)
        """
        prompt = prompt_data['prompt']
        metadata = prompt_data['metadata']
        task_id = metadata.get('task_id', 'unknown_task')
        worker_id = threading.current_thread().name

        if not prompt:
            logging.warning(f"Skipping task {task_id} due to empty prompt.")
            return metadata, None, "Empty prompt provided."

        # Get or maintain worker's assigned key
        max_key_switches = 5  # Maximum number of key switches
        key_switch_count = 0
        current_api_key = None
        
        while key_switch_count < max_key_switches:
            # Get assigned key for this worker
            current_api_key = self.key_manager.assign_key_to_worker(worker_id)

            if current_api_key == ALL_KEYS_WAITING_MARKER:
                # logging.info(f"Worker {worker_id} for task {task_id} waiting - no available keys")
                time.sleep(DEFAULT_WORKER_WAIT_SECONDS)
                continue
            elif current_api_key is None:
                logging.error(f"Worker {worker_id} for task {task_id} - FATAL: No usable keys")
                return metadata, None, "Fatal: No usable API keys available."

            masked_key = f"...{current_api_key[-4:]}" if len(current_api_key) > 4 else "invalid key"
            
            # Check if key can be used now
            if not self.key_manager.can_use_key_now(current_api_key):
                
                key_status = self.key_manager.check_key_status(current_api_key)
                if key_status in [KEY_STATUS_COOLDOWN, KEY_STATUS_TEMPORARILY_EXHAUSTED]:
                    # Wait for the key to become available
                    if key_status == KEY_STATUS_COOLDOWN:
                        wait_time = self.key_manager.key_cooldown_seconds
                        # logging.debug(f"Worker {worker_id} waiting {wait_time}s for key {masked_key} cooldown")
                    else:  # TEMPORARILY_EXHAUSTED
                        wait_time = self.key_manager.exhausted_wait_seconds
                        # logging.info(f"Worker {worker_id} waiting {wait_time}s for key {masked_key} temporary exhaustion")
                    
                    time.sleep(min(wait_time, DEFAULT_WORKER_WAIT_SECONDS))
                    continue
                    
                elif key_status in [KEY_STATUS_FULLY_EXHAUSTED, KEY_STATUS_FAILED_INIT]:
                    # Key is unusable, need to switch
                    logging.warning(f"Worker {worker_id} key {masked_key} is unusable ({key_status}), switching")
                    self.key_manager.release_key_from_worker(worker_id, current_api_key)
                    key_switch_count += 1
                    continue

            # Initialize client with the assigned key
            try:
                logging.debug(f"Worker {worker_id} using key {masked_key} for task {task_id}")
                client_instance = genai.Client(api_key=current_api_key)

            except Exception as e:
                logging.error(f"Failed to initialize client for {task_id} with key {masked_key}: {e}")
                self.key_manager.mark_key_failed_init(current_api_key)
                key_switch_count += 1
                continue

            # Perform API call
            result = self._make_single_api_call(client_instance, prompt_data)
            
            if result == EXHAUSTED_MARKER:
                # Key is exhausted - notify key manager
                logging.warning(f"Key {masked_key} exhausted for task {task_id}")
                self.key_manager.mark_key_exhausted(current_api_key)
                # Don't switch key immediately - let the key manager handle the state change
                # Next iteration will check the key status and decide whether to wait or switch
                continue
            elif result == PERSISTENT_ERROR_MARKER:
                # Persistent error - treat this task as failed
                logging.error(f"Persistent error for {task_id} with key {masked_key}")
                return metadata, None, "Persistent API call error."
            else:
                # Success! - mark key as used (cooldown) and reset exhausted count
                logging.debug(f"Success for {task_id} with key {masked_key}")
                self.key_manager.mark_key_successful(current_api_key)
                self.key_manager.mark_key_used(current_api_key)
                return metadata, result, None

        # Maximum key switch count exceeded
        logging.error(f"Task {task_id} failed after {max_key_switches} key switches")
        return metadata, None, f"Failed after {max_key_switches} key switches"

    def process_prompts(self, prompts_with_metadata: list[dict]) -> list[tuple]:
        """
        Processes a list of prompts in parallel using the managed API keys.
        Each worker is assigned a dedicated key and keeps using it throughout their tasks.
        Supports text-only and multimedia inputs with flexible positioning.

        Args:
            prompts_with_metadata (list[dict]): A list of dictionaries, where each
                                                dictionary can contain:
                                                - 'prompt' (str): The text prompt to send to Gemini.
                                                  Can contain <audio> and <video> tokens for positioning.
                                                - 'audio_path' (str or list[str], optional): Path(s) to audio file(s).
                                                - 'audio_bytes' (bytes or list[bytes], optional): Audio bytes.
                                                - 'audio_mime_type' (str or list[str], optional): MIME type(s) of audio (default: 'audio/mp3').
                                                - 'video_url' (str or list[str], optional): URL(s) of video file(s).
                                                - 'video_path' (str or list[str], optional): Path(s) to video file(s).
                                                - 'video_bytes' (bytes or list[bytes], optional): Video bytes.
                                                - 'video_mime_type' (str or list[str], optional): MIME type(s) of video (default: 'video/mp4').
                                                - 'video_metadata' (dict or list[dict], optional): Metadata for video file(s).
                                                - 'generation_config' (dict, optional): Generation config for the API call.
                                                - 'metadata' (dict): A dictionary of any
                                                  additional data associated with this prompt
                                                  (e.g., original line index, task info).
                                                  It's recommended to include a 'task_id' for logging.

        Example usage with positioning:
            prompts_with_metadata = [{
                'prompt': 'Analyze this audio: <audio> Then compare with this video: <video> What do you think about <audio>?',
                'audio_path': ['audio1.mp3', 'audio2.mp3'],
                'video_path': ['video1.mp4'],
                'metadata': {'task_id': 'multimedia_analysis_1'}
            }]

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
        errors = []
        active_workers = set()  # Track active worker IDs for key cleanup
        
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
                    if error_msg is None:
                        results.append((metadata_res, api_response_text, error_msg))
                    else:
                        errors.append((metadata_res, error_msg))
                except Exception as exc:
                    errors.append((original_metadata, exc))
                    logging.error(
                        f"Task for {task_id} generated an unhandled exception: {exc}",
                        exc_info=True,
                    )

                processed_count += 1
                current_time = time.time()
                # Log progress periodically
                if processed_count % 50 == 0 or current_time - last_log_time > 30:
                    logging.info(
                        f"Progress: {processed_count}/{len(prompts_with_metadata)} tasks completed."
                    )
                    last_log_time = current_time

        # Clean up worker key assignments after all tasks are completed
        # Get a snapshot of current worker assignments to clean up
        worker_assignments_snapshot = dict(self.key_manager.worker_assignments)
        for worker_id, api_key in worker_assignments_snapshot.items():
            if worker_id.startswith("GeminiAPIWorker"):  # Only clean up our workers
                logging.debug(f"Releasing key ...{api_key[-4:]} from completed worker {worker_id}")
                self.key_manager.release_key_from_worker(worker_id, api_key)

        logging.info(f"Parallel processing finished. Processed {processed_count} tasks.")
        logging.info(f"Key status summary: {self.key_manager.get_keys_status_summary()}")
        return results

class GeminiStreamingProcessor:
    """
    Streaming version of GeminiParallelProcessor that maintains persistent workers
    and processes single requests on-demand.
    
    Usage:
        processor = GeminiStreamingProcessor(key_manager, model_name)
        processor.start()  # Start persistent workers
        
        # Process single requests
        result = processor.process_single(prompt_data)
        
        processor.stop()  # Stop workers when done
    """
    def __init__(self, key_manager: AdvancedApiKeyManager, model_name: str,
                 api_call_interval: float = 2.0, max_workers: int = DEFAULT_MAX_WORKERS):
        """
        Initialize the streaming processor.
        
        Args:
            key_manager (AdvancedApiKeyManager): An instance of the API key manager.
            model_name (str): The name of the Gemini model to use.
            api_call_interval (float): Minimum time between API calls.
            max_workers (int): Maximum number of persistent worker threads.
        """
        self.key_manager = key_manager
        self.model_name = model_name
        self.api_call_interval = api_call_interval
        self.max_workers = max_workers
        
        # Global API call timing control (shared with batch processor if needed)
        self._last_api_call_time = 0.0
        self._api_call_lock = threading.Lock()
        
        # Streaming-specific components
        self.request_queue = queue.Queue()  # Queue for incoming requests
        self.result_dict = {}  # Dictionary to store results by request_id
        self.result_events = {}  # Events to signal when results are ready
        
        self.executor = None
        self.workers_running = False
        self.worker_futures = []
        
        logging.info(
            f"GeminiStreamingProcessor initialized for model '{self.model_name}' "
            f"with {self.max_workers} workers and global interval {self.api_call_interval}s."
        )

    def start(self):
        """Start the persistent worker pool."""
        if self.workers_running:
            logging.warning("Workers are already running.")
            return
        
        # Determine actual number of workers
        actual_workers = min(self.max_workers, self.key_manager.num_keys)
        if actual_workers == 0:
            raise ValueError("No workers can be started: no keys available.")
        
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=actual_workers, 
            thread_name_prefix="GeminiStreamWorker"
        )
        
        # Start persistent workers
        self.workers_running = True
        for i in range(actual_workers):
            future = self.executor.submit(self._persistent_worker, i)
            self.worker_futures.append(future)
        
        logging.info(f"Started {actual_workers} persistent workers for streaming processing.")

    def stop(self):
        """Stop the persistent worker pool."""
        if not self.workers_running:
            logging.warning("Workers are not running.")
            return
        
        logging.info("Stopping persistent workers...")
        self.workers_running = False
        
        # Send stop signals to all workers
        for _ in self.worker_futures:
            self.request_queue.put(None)  # None is the stop signal
        
        # Wait for all workers to finish
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None
        
        self.worker_futures.clear()
        
        # Clean up worker key assignments
        worker_assignments_snapshot = dict(self.key_manager.worker_assignments)
        for worker_id, api_key in worker_assignments_snapshot.items():
            if worker_id.startswith("GeminiStreamWorker"):
                logging.debug(f"Releasing key ...{api_key[-4:]} from stopped worker {worker_id}")
                self.key_manager.release_key_from_worker(worker_id, api_key)
        
        logging.info("All persistent workers stopped.")

    def process_single(self, prompt_data: dict, timeout: float = 300.0) -> tuple:
        """
        Process a single prompt and return the result.
        
        Args:
            prompt_data (dict): Dictionary containing prompt and metadata
            timeout (float): Maximum time to wait for result (default: 5 minutes)
            
        Returns:
            tuple: (metadata_dict, api_response_text_or_none, error_message_str_or_none)
            
        Raises:
            RuntimeError: If workers are not running
            TimeoutError: If processing takes longer than timeout
        """
        if not self.workers_running:
            raise RuntimeError("Workers are not running. Call start() first.")
        
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        
        # Create result event
        result_event = threading.Event()
        self.result_events[request_id] = result_event
        
        # Add task ID if not present
        if 'metadata' not in prompt_data:
            prompt_data['metadata'] = {}
        if 'task_id' not in prompt_data['metadata']:
            prompt_data['metadata']['task_id'] = f"stream_{request_id[:8]}"
        
        # Submit request to queue
        request = {
            'request_id': request_id,
            'prompt_data': prompt_data
        }
        self.request_queue.put(request)
        
        # Wait for result
        if result_event.wait(timeout=timeout):
            result = self.result_dict.pop(request_id)
            self.result_events.pop(request_id)
            return result
        else:
            # Timeout occurred
            self.result_events.pop(request_id, None)
            self.result_dict.pop(request_id, None)
            raise TimeoutError(f"Processing timed out after {timeout} seconds")

    def _persistent_worker(self, worker_index: int):
        """
        Persistent worker that continuously processes requests from the queue.
        
        Args:
            worker_index (int): Index of this worker for identification
        """
        worker_id = f"{threading.current_thread().name}_{worker_index}"
        logging.debug(f"Persistent worker {worker_id} started.")
        
        while self.workers_running:
            try:
                # Get request from queue (blocking with timeout)
                request = self.request_queue.get(timeout=1.0)
                
                # Check for stop signal
                if request is None:
                    break
                
                request_id = request['request_id']
                prompt_data = request['prompt_data']
                
                # Process the request using existing worker logic
                result = self._process_single_request(prompt_data, worker_id)
                
                # Store result and signal completion
                self.result_dict[request_id] = result
                if request_id in self.result_events:
                    self.result_events[request_id].set()
                
                # Mark task as done
                self.request_queue.task_done()
                
            except queue.Empty:
                # Timeout on queue.get - continue if still running
                continue
            except Exception as e:
                logging.error(f"Persistent worker {worker_id} error: {e}", exc_info=True)
        
        logging.debug(f"Persistent worker {worker_id} stopped.")

    def _process_single_request(self, prompt_data: dict, worker_id: str) -> tuple:
        """
        Process a single request using the existing worker task logic.
        This is essentially the same as _worker_task but adapted for streaming.
        
        Args:
            prompt_data (dict): The prompt data to process
            worker_id (str): Worker identifier
            
        Returns:
            tuple: (metadata_dict, api_response_text_or_marker, error_message_str)
        """
        prompt = prompt_data.get('prompt', '')
        metadata = prompt_data.get('metadata', {})
        task_id = metadata.get('task_id', 'unknown_task')

        if not prompt:
            logging.warning(f"Skipping task {task_id} due to empty prompt.")
            return metadata, None, "Empty prompt provided."

        # Get or maintain worker's assigned key
        max_key_switches = 5
        key_switch_count = 0
        current_api_key = None
        
        while key_switch_count < max_key_switches:
            # Get assigned key for this worker
            current_api_key = self.key_manager.assign_key_to_worker(worker_id)

            if current_api_key == ALL_KEYS_WAITING_MARKER:
                time.sleep(DEFAULT_WORKER_WAIT_SECONDS)
                continue
            elif current_api_key is None:
                logging.error(f"Worker {worker_id} for task {task_id} - FATAL: No usable keys")
                return metadata, None, "Fatal: No usable API keys available."

            masked_key = f"...{current_api_key[-4:]}" if len(current_api_key) > 4 else "invalid key"
            
            # Check if key can be used now
            if not self.key_manager.can_use_key_now(current_api_key):
                key_status = self.key_manager.check_key_status(current_api_key)
                
                if key_status in [KEY_STATUS_COOLDOWN, KEY_STATUS_TEMPORARILY_EXHAUSTED]:
                    # Wait for the key to become available
                    if key_status == KEY_STATUS_COOLDOWN:
                        wait_time = self.key_manager.key_cooldown_seconds
                    else:  # TEMPORARILY_EXHAUSTED
                        wait_time = self.key_manager.exhausted_wait_seconds
                    
                    time.sleep(min(wait_time, DEFAULT_WORKER_WAIT_SECONDS))
                    continue
                    
                elif key_status in [KEY_STATUS_FULLY_EXHAUSTED, KEY_STATUS_FAILED_INIT]:
                    # Key is unusable, need to switch
                    logging.warning(f"Worker {worker_id} key {masked_key} is unusable ({key_status}), switching")
                    self.key_manager.release_key_from_worker(worker_id, current_api_key)
                    key_switch_count += 1
                    continue

            # Initialize client with the assigned key
            try:
                logging.debug(f"Worker {worker_id} using key {masked_key} for task {task_id}")
                client_instance = genai.Client(api_key=current_api_key)

            except Exception as e:
                logging.error(f"Failed to initialize client for {task_id} with key {masked_key}: {e}")
                self.key_manager.mark_key_failed_init(current_api_key)
                key_switch_count += 1
                continue

            # Perform API call using existing logic
            result = self._make_single_api_call(client_instance, prompt_data)
            
            if result == EXHAUSTED_MARKER:
                # Key is exhausted - notify key manager
                logging.warning(f"Key {masked_key} exhausted for task {task_id}")
                self.key_manager.mark_key_exhausted(current_api_key)
                continue
            elif result == PERSISTENT_ERROR_MARKER:
                # Persistent error - treat this task as failed
                logging.error(f"Persistent error for {task_id} with key {masked_key}")
                return metadata, None, "Persistent API call error."
            else:
                # Success!
                logging.debug(f"Success for {task_id} with key {masked_key}")
                self.key_manager.mark_key_successful(current_api_key)
                self.key_manager.mark_key_used(current_api_key)
                return metadata, result, None

        # Maximum key switch count exceeded
        logging.error(f"Task {task_id} failed after {max_key_switches} key switches")
        return metadata, None, f"Failed after {max_key_switches} key switches"

    def _parse_prompt_with_media_tokens(self, prompt: str, audio_files: list, video_files: list) -> list:
        """Reuse the media parsing logic from GeminiParallelProcessor."""
        temp_processor = GeminiParallelProcessor(self.key_manager, self.model_name, self.api_call_interval, 1)
        return temp_processor._parse_prompt_with_media_tokens(prompt, audio_files, video_files)
    
    def _make_single_api_call(self, client_instance, prompt_data: dict) -> str:
        """
        Executes a single API call - same logic as GeminiParallelProcessor but optimized.
        """
        prompt = prompt_data.get('prompt', '')
        
        # Handle both single values and lists for all media parameters
        def ensure_list(value):
            if value is None:
                return []
            return value if isinstance(value, list) else [value]
        
        audio_paths = ensure_list(prompt_data.get('audio_path'))
        audio_bytes_list = ensure_list(prompt_data.get('audio_bytes'))
        audio_mime_types = ensure_list(prompt_data.get('audio_mime_type', 'audio/mp3'))
        video_urls = ensure_list(prompt_data.get('video_url'))
        video_paths = ensure_list(prompt_data.get('video_path'))
        video_bytes_list = ensure_list(prompt_data.get('video_bytes'))
        video_mime_types = ensure_list(prompt_data.get('video_mime_type', 'video/mp4'))
        video_metadata_list = ensure_list(prompt_data.get('video_metadata', {}))
        generation_config = prompt_data.get('generation_config', {})

        # Prepare video files
        video_files = []
        
        # Process video URLs
        for i, video_url in enumerate(video_urls):
            try:
                video_metadata = video_metadata_list[i] if i < len(video_metadata_list) else {}
                video_part = types.Part(
                    file_data=types.FileData(file_url=video_url),
                    video_metadata=types.VideoMetadata(**video_metadata)
                )
                video_files.append(video_part)
                logging.debug(f"Added video URL: {video_url}")
            except Exception as e:
                logging.error(f"Failed to process video URL {video_url}: {e}")
                return PERSISTENT_ERROR_MARKER
        
        # Process video paths
        for i, video_path in enumerate(video_paths):
            if os.path.exists(video_path):
                try:
                    video_file = client_instance.files.upload(file=video_path)
                    video_files.append(video_file)
                    logging.debug(f"Added video file: {video_path}")
                except Exception as e:
                    logging.error(f"Failed to upload video file {video_path}: {e}")
                    return PERSISTENT_ERROR_MARKER
            else:
                logging.error(f"Video file not found: {video_path}")
                return PERSISTENT_ERROR_MARKER
        
        # Process video bytes
        for i, video_bytes in enumerate(video_bytes_list):
            try:
                video_mime_type = video_mime_types[i] if i < len(video_mime_types) else 'video/mp4'
                video_metadata = video_metadata_list[i] if i < len(video_metadata_list) else {}
                video_part = types.Part(
                    inline_data=types.Blob(data=video_bytes, mime_type=video_mime_type),
                    video_metadata=types.VideoMetadata(**video_metadata)
                )
                video_files.append(video_part)
                logging.debug(f"Added video bytes: {video_mime_type}")
            except Exception as e:
                logging.error(f"Failed to create video part from bytes: {e}")
                return PERSISTENT_ERROR_MARKER

        # Prepare audio files
        audio_files = []
        
        # Process audio paths
        for i, audio_path in enumerate(audio_paths):
            if os.path.exists(audio_path):
                try:
                    audio_file = client_instance.files.upload(file=audio_path)
                    audio_files.append(audio_file)
                    logging.debug(f"Added audio file: {audio_path}")
                except Exception as e:
                    logging.error(f"Failed to upload audio file {audio_path}: {e}")
                    return PERSISTENT_ERROR_MARKER
            else:
                logging.error(f"Audio file not found: {audio_path}")
                return PERSISTENT_ERROR_MARKER
        
        # Process audio bytes
        for i, audio_bytes in enumerate(audio_bytes_list):
            try:
                audio_mime_type = audio_mime_types[i] if i < len(audio_mime_types) else 'audio/mp3'
                audio_part = types.Part.from_bytes(
                    data=audio_bytes,
                    mime_type=audio_mime_type
                )
                audio_files.append(audio_part)
                logging.debug(f"Added audio bytes: {audio_mime_type}")
            except Exception as e:
                logging.error(f"Failed to create audio part from bytes: {e}")
                return PERSISTENT_ERROR_MARKER

        # Parse prompt and construct contents with proper positioning
        if prompt and ('<audio>' in prompt or '<video>' in prompt):
            # Use token-based positioning
            contents = self._parse_prompt_with_media_tokens(prompt, audio_files, video_files)
        else:
            # Fallback to original behavior: video + audio + text
            contents = []
            contents.extend(video_files)
            contents.extend(audio_files)
            if prompt:
                contents.append(prompt)
        
        # Ensure we have some content
        if not contents:
            logging.error("No content provided (neither prompt nor media files)")
            return PERSISTENT_ERROR_MARKER
        
        # Perform API call with retries
        retries = 0
        wait_time = DEFAULT_ERROR_RETRY_DELAY
        while retries < DEFAULT_API_CALL_RETRIES:
            response = None
            try:
                # Global API call interval control - ensure sequential timing across all workers
                with self._api_call_lock:
                    current_time = time.time()
                    time_since_last_call = current_time - self._last_api_call_time
                    
                    if time_since_last_call < self.api_call_interval:
                        sleep_time = self.api_call_interval - time_since_last_call
                        worker_id = threading.current_thread().name
                        logging.debug(f"Worker {worker_id} waiting {sleep_time:.2f}s for global API interval")
                        time.sleep(sleep_time)
                    
                    # Update last API call time before making the call
                    self._last_api_call_time = time.time()
                
                response = client_instance.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=genai.types.GenerateContentConfig(**generation_config)
                )
                response_text = response.text.strip()
                
                # Log content types for debugging
                media_count = len(audio_files) + len(video_files)
                content_type = f"text+{media_count}media" if media_count > 0 else "text-only"
                logging.debug(f"API call successful ({content_type}). Response length: {len(response_text)}.")
                return response_text

            except genai_errors.APIError as e:
                # Handle different error codes based on official Gemini API documentation
                error_code = getattr(e, 'code', None) or getattr(e, 'status_code', None)
                
                if error_code == 429:  # RESOURCE_EXHAUSTED
                    logging.warning(f"RESOURCE_EXHAUSTED (429): {e}. Signaling exhaustion.")
                    return EXHAUSTED_MARKER
                    
                elif error_code in [400, 403, 404]:  # Non-retryable errors
                    # 400: INVALID_ARGUMENT, FAILED_PRECONDITION
                    # 403: PERMISSION_DENIED  
                    # 404: NOT_FOUND
                    logging.error(f"Non-retryable error ({error_code}): {e}. Signaling persistent error.")
                    return PERSISTENT_ERROR_MARKER
                    
                elif error_code in [500, 503, 504]:  # Retryable server errors
                    # 500: INTERNAL - Google internal error
                    # 503: UNAVAILABLE - Service temporarily overloaded/down
                    # 504: DEADLINE_EXCEEDED - Service couldn't complete in time
                    logging.warning(
                        f"Retryable server error ({error_code}): {e}. "
                        f"Retry {retries + 1}/{DEFAULT_API_CALL_RETRIES}..."
                    )
                else:
                    # Unknown error code - treat as retryable
                    logging.warning(
                        f"Unknown APIError ({error_code}): {e}. "
                        f"Retry {retries + 1}/{DEFAULT_API_CALL_RETRIES}..."
                    )
            except Exception as e:
                logging.error(
                    f"Unexpected error during API call: {type(e).__name__} - {e}. "
                    f"Traceback: {traceback.format_exc()}. "
                    f"Retry {retries + 1}/{DEFAULT_API_CALL_RETRIES}..."
                )

            retries += 1
            if retries < DEFAULT_API_CALL_RETRIES:
                logging.info(f"Waiting {wait_time}s before retrying API call...")
                time.sleep(wait_time)
                wait_time = wait_time * 2**retries # Exponential backoff
            else:
                logging.error(
                    f"Failed API call after {DEFAULT_API_CALL_RETRIES} retries."
                )
                return PERSISTENT_ERROR_MARKER

        return PERSISTENT_ERROR_MARKER

    def get_queue_size(self) -> int:
        """Get the current size of the request queue."""
        return self.request_queue.qsize()

    def is_running(self) -> bool:
        """Check if the workers are currently running."""
        return self.workers_running

    def get_worker_status(self) -> dict:
        """Get status information about the workers and keys."""
        return {
            'workers_running': self.workers_running,
            'queue_size': self.get_queue_size(),
            'max_workers': self.max_workers,
            'active_workers': len(self.worker_futures),
            'key_status': self.key_manager.get_keys_status_summary()
        }

"""
Usage Examples:

# === Batch Processing (Existing Method) ===
key_manager = AdvancedApiKeyManager(['GEMINI_API_KEY_1', 'GEMINI_API_KEY_2'])
batch_processor = GeminiParallelProcessor(key_manager, "gemini-2.0-flash-001")

prompts_with_metadata = [
    {
        'prompt': 'What is the capital of France?',
        'metadata': {'task_id': 'question_1'}
    },
    {
        'prompt': 'Explain quantum physics.',
        'metadata': {'task_id': 'question_2'}
    }
]

results = batch_processor.process_prompts(prompts_with_metadata)
for metadata, response, error in results:
    if error is None:
        print(f"Task {metadata['task_id']}: {response}")
    else:
        print(f"Task {metadata['task_id']} failed: {error}")

# === Streaming Processing (New Method) ===
key_manager = AdvancedApiKeyManager(['GEMINI_API_KEY_1', 'GEMINI_API_KEY_2'])
stream_processor = GeminiStreamingProcessor(key_manager, "gemini-2.0-flash-001")

# Start persistent workers
stream_processor.start()

try:
    # Process individual requests
    prompt_data_1 = {
        'prompt': 'What is the capital of France?',
        'metadata': {'task_id': 'question_1'}
    }
    
    # This call blocks until result is ready
    metadata, response, error = stream_processor.process_single(prompt_data_1)
    if error is None:
        print(f"Response: {response}")
    else:
        print(f"Error: {error}")
    
    # Process another request immediately
    prompt_data_2 = {
        'prompt': 'Explain quantum physics.',
        'metadata': {'task_id': 'question_2'}
    }
    
    metadata, response, error = stream_processor.process_single(prompt_data_2)
    if error is None:
        print(f"Response: {response}")
    else:
        print(f"Error: {error}")
    
    # Check worker status
    status = stream_processor.get_worker_status()
    print(f"Worker Status: {status}")
    
finally:
    # Always stop the workers when done
    stream_processor.stop()

# === Multimedia Processing with Streaming ===
multimedia_prompt = {
    'prompt': 'Analyze this audio: <audio> What do you hear?',
    'audio_path': '/path/to/audio.mp3',
    'metadata': {'task_id': 'audio_analysis'}
}

metadata, response, error = stream_processor.process_single(multimedia_prompt)

# === Advantages of Streaming Processor ===
# 1. Persistent workers - no overhead of creating/destroying threads
# 2. Immediate response - process one request at a time
# 3. Flexible usage - can be called from anywhere in your code
# 4. Same key management and error handling as batch processor
# 5. Perfect for web services, interactive applications, or incremental processing
"""
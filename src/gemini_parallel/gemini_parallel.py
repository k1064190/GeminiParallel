# gemini_parallel.py
# ABOUTME: Simplified sequential processor for Google Gemini API with intelligent key management
# This module provides a clean, sequential approach to Gemini API calls with automatic rate limiting

import os
import time
import logging
import random
from typing import Union
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from google.api_core import exceptions as google_exceptions
from dotenv import load_dotenv

# Import media processing utilities
try:
    from .gemini_media_processor import prepare_media_contents
    from .prompt_types import PromptData
except ImportError:
    from gemini_media_processor import prepare_media_contents  # type: ignore
    from prompt_types import PromptData  # type: ignore

load_dotenv()

# --- Constants ---
EXHAUSTED_MARKER = "RESOURCE_EXHAUSTED"
PERSISTENT_ERROR_MARKER = "PERSISTENT_ERROR"
ALL_KEYS_WAITING_MARKER = "ALL_KEYS_WAITING"

# Key status markers
KEY_STATUS_AVAILABLE = "AVAILABLE"
KEY_STATUS_COOLDOWN = "COOLDOWN"
KEY_STATUS_TEMPORARILY_EXHAUSTED = "TEMPORARILY_EXHAUSTED"
KEY_STATUS_FULLY_EXHAUSTED = "FULLY_EXHAUSTED"
KEY_STATUS_FAILED_INIT = "FAILED_INIT"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class AdvancedApiKeyManager:
    """
    Simplified API key manager without worker assignment complexity.

    Manages API keys with cooldowns, exhaustion recovery, and adaptive rate limiting.
    No worker tracking - keys are allocated dynamically per request.
    """

    def __init__(
        self,
        keylist_names,
        paid_keys=None,
        key_settings=None,
        adaptive_cooldown_settings=None,
    ):
        """
        Initialize the API key manager.

        Args:
            keylist_names (list[str] | str | int):
                - List of environment variable names containing API keys
                - "all": Find all GEMINI_API_KEY_* environment variables
                - Integer (e.g., 5): Search for GEMINI_API_KEY_1 through GEMINI_API_KEY_5
            paid_keys (str | list[str] | None):
                - "all": All keys are paid
                - List of strings: Environment variable names of paid keys
                - None: All keys are free (default)
            key_settings (dict | None): Settings for each key category
            adaptive_cooldown_settings (dict | None): Settings for adaptive cooldown
        """
        self.paid_keys = paid_keys

        # Default adaptive cooldown settings
        default_adaptive_settings = {
            "enabled": True,
            "exhaustion_threshold": 0.02,
            "initial_multiplier": 1.1,
            "max_multiplier": 1.5,
            "multiplier_increment": 0.1,
            "api_call_window": 300,
            "cooldown_recovery_rate": 0.9,
        }

        self.adaptive_settings = {
            **default_adaptive_settings,
            **(adaptive_cooldown_settings or {}),
        }

        # Adaptive cooldown tracking
        self.api_call_history = []
        self.current_cooldown_multiplier = 1.0
        self.last_adjustment_time = 0
        self.adjustment_count = 0

        # Default settings for free/paid keys
        default_free_settings = {
            "key_cooldown_seconds": 30,
            "exhausted_wait_seconds": 120,
            "fully_exhausted_wait_seconds": 43200,
            "max_exhausted_retries": 3,
            "consecutive_exhaustion_lockout_seconds": 43200,  # 12 hours
        }

        default_paid_settings = {
            "key_cooldown_seconds": 0,
            "exhausted_wait_seconds": 120,
            "fully_exhausted_wait_seconds": 43200,
            "max_exhausted_retries": 3,
            "consecutive_exhaustion_lockout_seconds": 43200,  # 12 hours
        }

        if key_settings is None:
            self.category_settings = {
                "free": default_free_settings,
                "paid": default_paid_settings,
            }
        else:
            self.category_settings = {
                "free": {**default_free_settings, **key_settings.get("free", {})},
                "paid": {**default_paid_settings, **key_settings.get("paid", {})},
            }

        # Load keys
        self.api_keys, self.key_to_env_name = self._load_keys_with_names(keylist_names)
        if not self.api_keys:
            raise ValueError(
                "No valid API keys found from provided environment variables."
            )

        paid_key_set = self._determine_paid_keys(paid_keys)

        # Track key information
        self.key_info = {}
        for key in self.api_keys:
            env_name = self.key_to_env_name.get(key, "")
            is_paid = key in paid_key_set

            self.key_info[key] = {
                "status": KEY_STATUS_AVAILABLE,
                "last_used_time": 0,
                "status_change_time": 0,
                "exhausted_count": 0,
                "consecutive_exhausted_count": 0,  # Track consecutive exhaustions
                "total_exhausted_count": 0,
                "category": "paid" if is_paid else "free",
                "env_name": env_name,
            }

        self.num_keys = len(self.api_keys)

        free_count = sum(
            1 for info in self.key_info.values() if info["category"] == "free"
        )
        paid_count = sum(
            1 for info in self.key_info.values() if info["category"] == "paid"
        )

        logging.info(
            f"AdvancedApiKeyManager initialized with {self.num_keys} keys (Free: {free_count}, Paid: {paid_count})"
        )

        for category, settings in self.category_settings.items():
            count = free_count if category == "free" else paid_count
            if count > 0:
                logging.info(
                    f"{category.capitalize()} key settings: "
                    f"Cooldown: {settings['key_cooldown_seconds']}s, "
                    f"Exhausted wait: {settings['exhausted_wait_seconds']}s"
                )

    def _load_keys_with_names(self, keylist_names):
        """Load API keys from environment variables."""
        api_keys = []
        key_to_env_name = {}

        if isinstance(keylist_names, list):
            for key_name in keylist_names:
                api_key = os.getenv(key_name)
                if api_key:
                    api_keys.append(api_key)
                    key_to_env_name[api_key] = key_name
                else:
                    logging.warning(
                        f"Environment variable {key_name} not found or empty."
                    )

        elif isinstance(keylist_names, str):
            if keylist_names == "all":
                for key, value in os.environ.items():
                    if key.startswith("GEMINI_API_KEY_") and value:
                        api_keys.append(value)
                        key_to_env_name[value] = key
            else:
                api_key = os.getenv(keylist_names)
                if api_key:
                    api_keys.append(api_key)
                    key_to_env_name[api_key] = keylist_names
                else:
                    logging.warning(f"Environment variable {keylist_names} not found.")

        elif isinstance(keylist_names, int):
            for i in range(1, keylist_names + 1):
                key_name = f"GEMINI_API_KEY_{i}"
                api_key = os.getenv(key_name)
                if api_key:
                    api_keys.append(api_key)
                    key_to_env_name[api_key] = key_name

        else:
            raise ValueError("keylist_names must be a list, string, or integer")

        return api_keys, key_to_env_name

    def _determine_paid_keys(self, paid_keys):
        """Determine which keys are paid."""
        if paid_keys is None:
            return set()

        if paid_keys == "all":
            return set(self.api_keys)

        if isinstance(paid_keys, list):
            paid_key_set = set()
            for key_name in paid_keys:
                api_key = os.getenv(key_name)
                if api_key and api_key in self.api_keys:
                    paid_key_set.add(api_key)
            return paid_key_set

        raise ValueError("paid_keys must be 'all', a list of key names, or None")

    def _update_key_status_based_on_time(self):
        """Update key statuses based on elapsed time."""
        current_time = time.time()

        for key, info in self.key_info.items():
            category = info["category"]
            settings = self.category_settings[category]

            # Apply adaptive multiplier to cooldown settings
            adjusted_settings = settings.copy()
            if self.adaptive_settings["enabled"]:
                adjusted_settings["key_cooldown_seconds"] = int(
                    settings["key_cooldown_seconds"] * self.current_cooldown_multiplier
                )
                adjusted_settings["exhausted_wait_seconds"] = int(
                    settings["exhausted_wait_seconds"]
                    * self.current_cooldown_multiplier
                )

            if info["status"] == KEY_STATUS_COOLDOWN:
                if (
                    current_time - info["status_change_time"]
                    >= adjusted_settings["key_cooldown_seconds"]
                ):
                    info["status"] = KEY_STATUS_AVAILABLE
                    logging.debug(
                        f"{category.capitalize()} key ...{key[-4:]} cooldown finished"
                    )

            elif info["status"] == KEY_STATUS_TEMPORARILY_EXHAUSTED:
                if (
                    current_time - info["status_change_time"]
                    >= adjusted_settings["exhausted_wait_seconds"]
                ):
                    info["status"] = KEY_STATUS_AVAILABLE
                    logging.info(
                        f"{category.capitalize()} key ...{key[-4:]} temporary exhaustion recovered"
                    )

            elif info["status"] == KEY_STATUS_FULLY_EXHAUSTED:
                base_settings = self.category_settings[category]
                if (
                    current_time - info["status_change_time"]
                    >= base_settings["fully_exhausted_wait_seconds"]
                ):
                    info["status"] = KEY_STATUS_AVAILABLE
                    info["exhausted_count"] = 0
                    logging.info(
                        f"{category.capitalize()} key ...{key[-4:]} full exhaustion recovered"
                    )

    def get_any_available_key(self, caller_id: str = "unknown"):
        """
        Get any available key (no worker assignment).

        Args:
            caller_id: Identifier for logging purposes only

        Returns:
            str: API key
            str: ALL_KEYS_WAITING_MARKER if all keys busy
            None: If no usable keys exist
        """
        self._update_key_status_based_on_time()

        # Find available keys
        available_keys = [
            key
            for key, info in self.key_info.items()
            if info["status"] == KEY_STATUS_AVAILABLE
        ]

        if available_keys:
            # Randomize to distribute load
            selected_key = random.choice(available_keys)
            info = self.key_info[selected_key]

            # Mark as COOLDOWN if applicable
            category = info["category"]
            cooldown = self.category_settings[category]["key_cooldown_seconds"]

            if cooldown > 0:
                info["status"] = KEY_STATUS_COOLDOWN
                info["status_change_time"] = time.time()

            info["last_used_time"] = time.time()

            masked_key = f"...{selected_key[-4:]}"
            logging.debug(f"Caller {caller_id} using key {masked_key}")
            return selected_key

        # No available keys - check if any will recover
        status_counts = {}
        for info in self.key_info.values():
            status = info["status"]
            status_counts[status] = status_counts.get(status, 0) + 1

        if status_counts.get(KEY_STATUS_FAILED_INIT, 0) == self.num_keys:
            logging.error("FATAL: All API keys failed initialization")
            return None

        if status_counts.get(KEY_STATUS_FULLY_EXHAUSTED, 0) == self.num_keys:
            logging.error("FATAL: All API keys fully exhausted")
            return None

        logging.debug(
            f"Caller {caller_id} waiting - no available keys. Status: {status_counts}"
        )
        return ALL_KEYS_WAITING_MARKER

    def mark_key_exhausted(self, api_key: str):
        """Mark a key as exhausted due to rate limiting."""
        if api_key not in self.key_info:
            return

        info = self.key_info[api_key]
        info["exhausted_count"] += 1
        info["consecutive_exhausted_count"] += 1
        info["total_exhausted_count"] += 1

        category = info["category"]
        max_retries = self.category_settings[category]["max_exhausted_retries"]

        # Check if key hit consecutive exhaustion limit (3 times → 12 hour lockout)
        if info["consecutive_exhausted_count"] >= 3:
            lockout_seconds = self.category_settings[category][
                "consecutive_exhaustion_lockout_seconds"
            ]
            info["status"] = KEY_STATUS_FULLY_EXHAUSTED
            info["status_change_time"] = time.time()
            logging.error(
                f"Key ...{api_key[-4:]} locked out for {lockout_seconds / 3600:.1f} hours after 3 consecutive exhaustions"
            )
        elif info["exhausted_count"] >= max_retries:
            info["status"] = KEY_STATUS_FULLY_EXHAUSTED
            info["status_change_time"] = time.time()
            logging.warning(
                f"Key ...{api_key[-4:]} marked as FULLY_EXHAUSTED after {info['exhausted_count']} consecutive exhaustions"
            )
        else:
            info["status"] = KEY_STATUS_TEMPORARILY_EXHAUSTED
            info["status_change_time"] = time.time()
            logging.info(
                f"Key ...{api_key[-4:]} marked as TEMPORARILY_EXHAUSTED ({info['exhausted_count']}/{max_retries})"
            )

        # Track for adaptive cooldown
        self.api_call_history.append((time.time(), True))
        self._adjust_cooldown_if_needed()

    def mark_key_success(self, api_key: str):
        """Reset consecutive exhaustion counter on successful API call."""
        if api_key in self.key_info:
            self.key_info[api_key]["consecutive_exhausted_count"] = 0

    def mark_key_failed_init(self, api_key: str):
        """Mark a key as failed during initialization."""
        if api_key in self.key_info:
            self.key_info[api_key]["status"] = KEY_STATUS_FAILED_INIT
            logging.error(f"Key ...{api_key[-4:]} marked as FAILED_INIT")

    def _adjust_cooldown_if_needed(self):
        """Adjust cooldown multiplier based on exhaustion rate."""
        if not self.adaptive_settings["enabled"]:
            return

        current_time = time.time()
        window = self.adaptive_settings["api_call_window"]

        # Remove old history
        self.api_call_history = [
            (t, exhausted)
            for t, exhausted in self.api_call_history
            if current_time - t < window
        ]

        if len(self.api_call_history) < 10:
            return

        exhausted_count = sum(1 for _, exhausted in self.api_call_history if exhausted)
        exhaustion_rate = exhausted_count / len(self.api_call_history)

        threshold = self.adaptive_settings["exhaustion_threshold"]

        if exhaustion_rate > threshold:
            # Increase cooldown
            old_multiplier = self.current_cooldown_multiplier
            self.current_cooldown_multiplier = min(
                self.current_cooldown_multiplier
                + self.adaptive_settings["multiplier_increment"],
                self.adaptive_settings["max_multiplier"],
            )
            if self.current_cooldown_multiplier != old_multiplier:
                logging.warning(
                    f"Adaptive cooldown: Exhaustion rate {exhaustion_rate:.1%} > {threshold:.1%}. "
                    f"Multiplier: {old_multiplier:.2f} → {self.current_cooldown_multiplier:.2f}"
                )
                self.adjustment_count += 1
                self.last_adjustment_time = current_time

        elif exhaustion_rate < threshold / 2 and self.current_cooldown_multiplier > 1.0:
            # Decrease cooldown (recovery)
            if current_time - self.last_adjustment_time > window:
                old_multiplier = self.current_cooldown_multiplier
                self.current_cooldown_multiplier = max(
                    self.current_cooldown_multiplier
                    * self.adaptive_settings["cooldown_recovery_rate"],
                    1.0,
                )
                if self.current_cooldown_multiplier != old_multiplier:
                    logging.info(
                        f"Adaptive cooldown recovery: Rate {exhaustion_rate:.1%}. "
                        f"Multiplier: {old_multiplier:.2f} → {self.current_cooldown_multiplier:.2f}"
                    )
                    self.last_adjustment_time = current_time

    def get_keys_status_summary(self) -> dict:
        """Get summary of all key statuses."""
        summary = {}
        for key, info in self.key_info.items():
            masked_key = f"...{key[-4:]}"
            summary[masked_key] = {
                "category": info["category"],
                "status": info["status"],
                "exhausted_count": info["exhausted_count"],
                "total_exhausted_count": info["total_exhausted_count"],
            }
        return summary


class GeminiSequentialProcessor:
    """
    Simplified sequential processor for Gemini API calls.

    Eliminates unnecessary threading complexity by processing requests sequentially.
    Since API calls must be rate-limited anyway (IP ban protection), threading
    only adds overhead without improving throughput.

    Key improvements over old parallel/streaming processors:
    - 75% less code (~500 lines vs ~1800 lines)
    - No threading overhead
    - No lock contention
    - Same throughput (sequential execution already enforced)
    - Easier debugging
    - Lower memory usage

    Usage:
        key_manager = AdvancedApiKeyManager(keylist_names="all")
        processor = GeminiSequentialProcessor(
            key_manager=key_manager,
            model_name="gemini-2.0-flash-001",
            api_call_interval=2.0
        )

        # Single request
        metadata, response, error = processor.process_single(prompt_data)

        # Multiple requests
        results = processor.process_batch(prompts_data)
    """

    def __init__(
        self,
        key_manager: AdvancedApiKeyManager,
        model_name: str,
        api_call_interval: float = 2.0,
        api_call_retries: int = 3,
        return_response: bool = False,
        ip_ban_detection_threshold: int = 3,
        ip_ban_wait_seconds: int = 300,
    ):
        """
        Initialize the sequential processor.

        Args:
            key_manager: API key manager instance
            model_name: Gemini model name (e.g., "gemini-2.0-flash-001")
            api_call_interval: Minimum seconds between API calls (IP ban protection)
            api_call_retries: Maximum retry attempts for API errors
            return_response: If True, return full response object; if False, return text only
            ip_ban_detection_threshold: Number of consecutive exhaustions to trigger IP ban wait (default: 3)
            ip_ban_wait_seconds: Seconds to wait when IP ban is detected (default: 300 = 5 minutes)
        """
        self.key_manager = key_manager
        self.model_name = model_name
        self.api_call_interval = api_call_interval
        self.api_call_retries = api_call_retries
        self.return_response = return_response
        self.ip_ban_detection_threshold = ip_ban_detection_threshold
        self.ip_ban_wait_seconds = ip_ban_wait_seconds

        # Simple timing control - no locks needed (single-threaded)
        self.last_api_call_time = 0.0

        # Track consecutive exhaustions for IP ban detection
        self.consecutive_exhaustions = 0

        logging.info(
            f"GeminiSequentialProcessor initialized for model '{self.model_name}' "
            f"with API interval {self.api_call_interval}s, "
            f"IP ban detection: {self.ip_ban_detection_threshold} exhaustions → {self.ip_ban_wait_seconds}s wait"
        )

    def _enforce_api_interval(self):
        """Enforce minimum interval between API calls to prevent IP ban."""
        elapsed = time.time() - self.last_api_call_time
        adjusted_interval = self.api_call_interval

        # Apply adaptive cooldown multiplier if enabled
        if self.key_manager.adaptive_settings["enabled"]:
            adjusted_interval *= self.key_manager.current_cooldown_multiplier

        if elapsed < adjusted_interval:
            sleep_time = adjusted_interval - elapsed
            logging.debug(
                f"Waiting {sleep_time:.2f}s for API interval (adjusted: {adjusted_interval:.1f}s)"
            )
            time.sleep(sleep_time)

        self.last_api_call_time = time.time()

    def _get_available_key(self, task_id: str) -> str | None:
        """
        Get an available API key with retry logic.

        Args:
            task_id: Task identifier for logging

        Returns:
            API key string, or None if no keys available after timeout
        """
        max_wait_time = 3600  # 1 hour max
        start_time = time.time()
        attempt = 0

        while (time.time() - start_time) < max_wait_time:
            current_api_key = self.key_manager.get_any_available_key("sequential")

            if current_api_key == ALL_KEYS_WAITING_MARKER:
                elapsed = time.time() - start_time
                if elapsed > max_wait_time / 2:
                    logging.warning(
                        f"Task {task_id} waiting {elapsed:.0f}s for available keys"
                    )

                wait_time = min(30, 10 + (attempt // 10) * 5)
                logging.debug(
                    f"Task {task_id} waiting {wait_time}s - all keys busy (attempt {attempt + 1})"
                )
                time.sleep(wait_time)
                attempt += 1
                continue
            elif current_api_key is None:
                logging.error(f"Task {task_id} - FATAL: No usable keys available")
                return None
            else:
                masked_key = (
                    f"...{current_api_key[-4:]}"
                    if len(current_api_key) > 4
                    else "invalid"
                )
                logging.debug(f"Task {task_id} using key {masked_key}")
                return current_api_key

        logging.error(f"Task {task_id} - Timeout waiting for available keys")
        return None

    def _make_api_call_with_retries(
        self, client_instance, prompt_data: Union[dict, PromptData]
    ) -> str:
        """
        Execute a single API call with retry logic.

        Returns:
            EXHAUSTED_MARKER if resource exhausted
            PERSISTENT_ERROR_MARKER if other errors persist
            Response text/object on success
        """
        if isinstance(prompt_data, PromptData):
            prompt_dict = prompt_data.to_dict()
        else:
            prompt_dict = prompt_data

        contents, error_msg = prepare_media_contents(client_instance, prompt_dict)
        if contents is None:
            return PERSISTENT_ERROR_MARKER

        generation_config = prompt_dict.get("generation_config", {})

        retries = 0
        wait_time = 30

        while retries < self.api_call_retries:
            try:
                # Handle generation_config
                if isinstance(generation_config, types.GenerateContentConfig):
                    config = generation_config
                elif isinstance(generation_config, dict) and generation_config:
                    config_kwargs = generation_config.copy()

                    # Convert nested configs
                    if "thinkingConfig" in config_kwargs and isinstance(
                        config_kwargs["thinkingConfig"], dict
                    ):
                        config_kwargs["thinkingConfig"] = types.ThinkingConfig(
                            **config_kwargs["thinkingConfig"]
                        )
                    elif "thinking_config" in config_kwargs and isinstance(
                        config_kwargs["thinking_config"], dict
                    ):
                        config_kwargs["thinkingConfig"] = types.ThinkingConfig(
                            **config_kwargs.pop("thinking_config")
                        )

                    if "speechConfig" in config_kwargs and isinstance(
                        config_kwargs["speechConfig"], dict
                    ):
                        config_kwargs["speechConfig"] = types.SpeechConfig(
                            **config_kwargs["speechConfig"]
                        )
                    elif "speech_config" in config_kwargs and isinstance(
                        config_kwargs["speech_config"], dict
                    ):
                        config_kwargs["speechConfig"] = types.SpeechConfig(
                            **config_kwargs.pop("speech_config")
                        )

                    # Convert snake_case to camelCase
                    snake_to_camel_mappings = {
                        "max_output_tokens": "maxOutputTokens",
                        "stop_sequences": "stopSequences",
                        "response_mime_type": "responseMimeType",
                        "response_logprobs": "responseLogprobs",
                        "top_p": "topP",
                        "top_k": "topK",
                        "presence_penalty": "presencePenalty",
                        "frequency_penalty": "frequencyPenalty",
                        "candidate_count": "candidateCount",
                        "response_modalities": "responseModalities",
                        "media_resolution": "mediaResolution",
                        "response_schema": "responseSchema",
                        "response_json_schema": "responseJsonSchema",
                    }

                    for snake, camel in snake_to_camel_mappings.items():
                        if snake in config_kwargs:
                            config_kwargs[camel] = config_kwargs.pop(snake)

                    config = types.GenerateContentConfig(**config_kwargs)
                else:
                    config = types.GenerateContentConfig()

                response = client_instance.models.generate_content(
                    model=self.model_name, contents=contents, config=config
                )
                response_text = response.text.strip()

                logging.debug(
                    f"API call successful. Response length: {len(response_text)}"
                )
                return response if self.return_response else response_text

            except genai_errors.ClientError as e:
                error_code = e.code if hasattr(e, "code") else None
                if error_code == 429:
                    logging.warning(f"Resource exhausted (429): {e}")
                    return EXHAUSTED_MARKER
                elif error_code in [400, 401, 403, 404]:
                    logging.error(f"Client error ({error_code}): {e}")
                    return PERSISTENT_ERROR_MARKER
                else:
                    logging.warning(
                        f"Retryable client error ({error_code}): {e}. Retry {retries + 1}/{self.api_call_retries}"
                    )

            except genai_errors.ServerError as e:
                error_code = e.code if hasattr(e, "code") else None
                logging.warning(
                    f"Retryable server error ({error_code}): {e}. Retry {retries + 1}/{self.api_call_retries}"
                )

            except Exception as e:
                logging.error(
                    f"Unexpected error: {type(e).__name__} - {e}. Retry {retries + 1}/{self.api_call_retries}"
                )

            retries += 1
            if retries < self.api_call_retries:
                logging.info(f"Waiting {wait_time}s before retrying...")
                time.sleep(wait_time)
                wait_time = wait_time * 2
            else:
                logging.error(f"Failed after {self.api_call_retries} retries")
                return PERSISTENT_ERROR_MARKER

        return PERSISTENT_ERROR_MARKER

    def process_single(self, prompt_data: Union[dict, PromptData]) -> tuple:
        """
        Process a single prompt request.

        Args:
            prompt_data: Dictionary or PromptData object containing:
                - 'prompt' (str): The text prompt
                - 'metadata' (dict): Task metadata
                - Optional: audio_path, video_path, generation_config, etc.

        Returns:
            tuple: (metadata, response_text_or_object, error_message)
        """
        if isinstance(prompt_data, PromptData):
            prompt = prompt_data.prompt
            metadata = prompt_data.metadata
        else:
            prompt = prompt_data["prompt"]
            metadata = prompt_data["metadata"]

        task_id = metadata.get("task_id", "unknown_task")

        if not prompt:
            logging.warning(f"Skipping task {task_id} - empty prompt")
            return metadata, None, "Empty prompt provided"

        self._enforce_api_interval()

        max_key_retries = 100
        for key_attempt in range(max_key_retries):
            api_key = self._get_available_key(task_id)

            if api_key is None:
                return metadata, None, "Fatal: No usable API keys available"

            try:
                client_instance = genai.Client(api_key=api_key)
            except Exception as e:
                logging.error(
                    f"Failed to initialize client with key ...{api_key[-4:]}: {e}"
                )
                self.key_manager.mark_key_failed_init(api_key)
                continue

            result = self._make_api_call_with_retries(client_instance, prompt_data)

            if result == EXHAUSTED_MARKER:
                logging.warning(
                    f"Key ...{api_key[-4:]} exhausted for task {task_id}, trying another key"
                )
                self.key_manager.mark_key_exhausted(api_key)
                self.consecutive_exhaustions += 1

                # IP ban detection: N consecutive exhaustions → wait
                if self.consecutive_exhaustions >= self.ip_ban_detection_threshold:
                    logging.error(
                        f"IP BAN DETECTED: {self.consecutive_exhaustions} consecutive exhaustions. "
                        f"Waiting {self.ip_ban_wait_seconds}s before retrying..."
                    )
                    time.sleep(self.ip_ban_wait_seconds)
                    self.consecutive_exhaustions = 0  # Reset counter after wait
                    logging.info("IP ban wait complete, resuming operations")

                continue
            elif result == PERSISTENT_ERROR_MARKER:
                logging.error(f"Task {task_id} failed with persistent error")
                self.consecutive_exhaustions = 0  # Reset on non-exhaustion result
                return metadata, None, "Persistent API error"
            else:
                logging.info(f"Task {task_id} completed successfully")
                self.consecutive_exhaustions = 0  # Reset on success
                self.key_manager.mark_key_success(
                    api_key
                )  # Reset key's consecutive counter
                return metadata, result, None

        logging.error(
            f"Task {task_id} failed after {max_key_retries} key retry attempts"
        )
        return metadata, None, f"Failed after {max_key_retries} key attempts"

    def process_batch(self, prompts_data: list[Union[dict, PromptData]]) -> list[tuple]:
        """
        Process multiple prompts sequentially.

        Args:
            prompts_data: List of prompt dictionaries or PromptData objects

        Returns:
            list of tuples: [(metadata, response, error), ...]
        """
        logging.info(f"Processing batch of {len(prompts_data)} prompts sequentially")

        results = []
        for i, prompt_data in enumerate(prompts_data):
            logging.info(f"Processing prompt {i + 1}/{len(prompts_data)}")
            result = self.process_single(prompt_data)
            results.append(result)

        logging.info(f"Batch processing complete: {len(results)} results")
        return results

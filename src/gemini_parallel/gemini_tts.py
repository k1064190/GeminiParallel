# gemini_tts.py

import os
import time
import logging
import wave
import threading
import concurrent.futures
from typing import Optional, Union, Dict, List, Tuple, Any
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from dotenv import load_dotenv

# Import the key manager from main module
try:
    from .gemini_parallel import AdvancedApiKeyManager, EXHAUSTED_MARKER, PERSISTENT_ERROR_MARKER, ALL_KEYS_WAITING_MARKER
except ImportError:
    from gemini_parallel import AdvancedApiKeyManager, EXHAUSTED_MARKER, PERSISTENT_ERROR_MARKER, ALL_KEYS_WAITING_MARKER  # type: ignore

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s",
)

# Available TTS voices with descriptions
TTS_VOICES = {
    # Bright/Upbeat voices
    "Zephyr": "Bright",
    "Puck": "Upbeat", 
    "Autonoe": "Bright",
    "Laomedeia": "Upbeat",
    
    # Firm/Clear voices
    "Kore": "Firm",
    "Orus": "Firm",
    "Alnilam": "Firm",
    "Erinome": "Clear",
    "Iapetus": "Clear",
    
    # Informative/Knowledgeable
    "Charon": "Informative",
    "Rasalgethi": "Informative",
    "Sadaltager": "Knowledgeable",
    
    # Smooth/Soft voices
    "Algieba": "Smooth",
    "Despina": "Smooth",
    "Achernar": "Soft",
    
    # Easy-going/Casual
    "Callirrhoe": "Easy-going",
    "Umbriel": "Easy-going",
    "Zubenelgenubi": "Casual",
    
    # Other styles
    "Fenrir": "Excitable",
    "Leda": "Youthful",
    "Aoede": "Breezy",
    "Enceladus": "Breathy",
    "Algenib": "Gravelly",
    "Gacrux": "Mature",
    "Pulcherrima": "Forward",
    "Achird": "Friendly",
    "Vindemiatrix": "Gentle",
    "Sadachbia": "Lively",
    "Schedar": "Even",
    "Sulafat": "Warm",
}

# TTS model names
TTS_MODELS = {
    "flash": "gemini-2.5-flash-preview-tts",
    "pro": "gemini-2.5-pro-preview-tts"
}


class GeminiTTSProcessor:
    """
    Gemini TTS Processor for text-to-speech generation.
    Supports single-speaker and multi-speaker audio generation with style control.
    """
    
    def __init__(self, 
                 key_manager: Optional[AdvancedApiKeyManager] = None,
                 model: str = "flash",
                 default_voice: str = "Kore",
                 output_sample_rate: int = 24000,
                 output_channels: int = 1,
                 output_sample_width: int = 2):
        """
        Initialize the TTS processor.
        
        Args:
            key_manager: API key manager instance. If None, creates one with GEMINI_API_KEY
            model: Model to use - "flash" or "pro" 
            default_voice: Default voice to use for TTS
            output_sample_rate: Sample rate for output audio (default 24kHz)
            output_channels: Number of channels (default 1 for mono)
            output_sample_width: Sample width in bytes (default 2 for 16-bit)
        """
        # Initialize key manager if not provided
        if key_manager is None:
            # Try to find API keys
            if os.getenv("GEMINI_API_KEY"):
                self.key_manager = AdvancedApiKeyManager(["GEMINI_API_KEY"])
            else:
                # Try to find numbered keys
                self.key_manager = AdvancedApiKeyManager("all")
        else:
            self.key_manager = key_manager
            
        # Set model
        if model in TTS_MODELS:
            self.model_name = TTS_MODELS[model]
        else:
            self.model_name = model
            
        # Audio settings
        self.default_voice = default_voice
        self.output_sample_rate = output_sample_rate
        self.output_channels = output_channels
        self.output_sample_width = output_sample_width
        
        # API call control
        self._last_api_call_time = 0.0
        self._api_call_lock = threading.Lock()
        self.api_call_interval = 2.0  # Minimum seconds between API calls
        
        logging.info(f"GeminiTTSProcessor initialized with model '{self.model_name}' and default voice '{self.default_voice}'")
    
    def _save_audio_to_file(self, audio_data: bytes, filename: str) -> None:
        """
        Save audio data to a WAV file.
        
        Args:
            audio_data: PCM audio data bytes
            filename: Output filename
        """
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(self.output_channels)
            wf.setsampwidth(self.output_sample_width)
            wf.setframerate(self.output_sample_rate)
            wf.writeframes(audio_data)
        logging.info(f"Audio saved to {filename}")
    
    def _make_tts_api_call(self, 
                          client: genai.Client, 
                          contents: str,
                          speech_config: types.SpeechConfig) -> Union[bytes, str]:
        """
        Make a single TTS API call with retry logic.
        
        Args:
            client: Gemini client instance
            contents: Text content to convert to speech
            speech_config: Speech configuration
            
        Returns:
            bytes: Audio data on success
            str: Error marker on failure
        """
        retries = 0
        wait_time = 30
        
        while retries < 3:
            try:
                # API call interval control
                with self._api_call_lock:
                    current_time = time.time()
                    time_since_last_call = current_time - self._last_api_call_time
                    
                    if time_since_last_call < self.api_call_interval:
                        sleep_time = self.api_call_interval - time_since_last_call
                        logging.debug(f"Waiting {sleep_time:.2f}s for API interval")
                        time.sleep(sleep_time)
                    
                    self._last_api_call_time = time.time()
                
                # Make TTS API call
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        response_modalities=["AUDIO"],
                        speech_config=speech_config
                    )
                )
                
                # Extract audio data
                audio_data = response.candidates[0].content.parts[0].inline_data.data
                logging.debug(f"TTS API call successful, generated {len(audio_data)} bytes of audio")
                return audio_data
                
            except genai_errors.APIError as e:
                error_code = getattr(e, 'code', None) or getattr(e, 'status_code', None)
                
                if error_code == 429:  # Rate limit
                    logging.warning(f"Rate limit hit: {e}")
                    return EXHAUSTED_MARKER
                elif error_code in [400, 403, 404]:  # Non-retryable
                    logging.error(f"Non-retryable error ({error_code}): {e}")
                    return PERSISTENT_ERROR_MARKER
                elif error_code in [500, 503, 504]:  # Retryable
                    logging.warning(f"Retryable error ({error_code}): {e}. Retry {retries + 1}/3")
                else:
                    logging.warning(f"Unknown error ({error_code}): {e}. Retry {retries + 1}/3")
                    
            except Exception as e:
                logging.error(f"Unexpected error: {type(e).__name__} - {e}. Retry {retries + 1}/3")
            
            retries += 1
            if retries < 3:
                logging.info(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                wait_time = wait_time * 2 ** retries
        
        return PERSISTENT_ERROR_MARKER
    
    def generate_speech(self,
                       text: str,
                       voice: Optional[str] = None,
                       output_file: Optional[str] = None,
                       style_prompt: Optional[str] = None) -> Optional[bytes]:
        """
        Generate speech from text using a single speaker.
        
        Args:
            text: Text to convert to speech
            voice: Voice name (from TTS_VOICES). Uses default if not specified
            output_file: Optional filename to save audio to
            style_prompt: Optional style instruction (e.g., "Say cheerfully", "Whisper mysteriously")
            
        Returns:
            bytes: Audio data, or None on error
            
        Example:
            audio = tts.generate_speech(
                "Hello world!", 
                voice="Puck",
                style_prompt="Say excitedly with enthusiasm"
            )
        """
        # Prepare voice
        voice = voice or self.default_voice
        if voice not in TTS_VOICES:
            logging.warning(f"Unknown voice '{voice}', using default '{self.default_voice}'")
            voice = self.default_voice
        
        # Prepare prompt with style if provided
        if style_prompt:
            contents = f"{style_prompt}: {text}"
        else:
            contents = text
            
        # Create speech config
        speech_config = types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=voice
                )
            )
        )
        
        # Dynamic key allocation with retry logic (like GeminiStreamingProcessor)
        max_attempts = 10000
        attempt_count = 0
        
        while attempt_count < max_attempts:
            # Get any available key
            api_key = self.key_manager.get_any_available_key("tts_single")
            
            if api_key == ALL_KEYS_WAITING_MARKER:
                # All keys are busy, wait and retry
                logging.debug("All keys busy, waiting...")
                time.sleep(10)
                attempt_count += 1
                continue
            elif api_key is None:
                logging.error("No usable API keys available")
                return None
            
            masked_key = f"...{api_key[-4:]}" if len(api_key) > 4 else "invalid key"
            
            try:
                # Create client
                logging.debug(f"Using key {masked_key} for TTS")
                client = genai.Client(api_key=api_key)
                
            except Exception as e:
                logging.error(f"Failed to initialize client with key {masked_key}: {e}")
                self.key_manager.mark_key_failed_init(api_key)
                continue
            
            # Make API call
            result = self._make_tts_api_call(client, contents, speech_config)
            
            if result == EXHAUSTED_MARKER:
                # Key is exhausted - mark it and try again with another key
                logging.warning(f"Key {masked_key} exhausted, trying another key")
                self.key_manager.mark_key_exhausted(api_key)
                continue
            elif result == PERSISTENT_ERROR_MARKER:
                # Persistent error - fail this request
                logging.error(f"Persistent error with key {masked_key}")
                return None
            elif isinstance(result, bytes):
                # Success!
                logging.debug(f"TTS generation successful with key {masked_key}")
                self.key_manager.mark_key_successful(api_key)
                self.key_manager.mark_key_returned(api_key, "tts_single")
                
                # Save to file if requested
                if output_file:
                    self._save_audio_to_file(result, output_file)
                
                return result
            
            attempt_count += 1
        
        # Maximum attempts exceeded
        logging.error(f"TTS generation failed after {max_attempts} attempts")
        return None
    
    def generate_multi_speaker_dialogue(self,
                                       dialogue: Union[Dict[str, str], List[Tuple[str, str]]],
                                       voices: Dict[str, str],
                                       output_file: Optional[str] = None,
                                       style_prompt: Optional[str] = None) -> Optional[bytes]:
        """
        Generate multi-speaker dialogue audio.
        
        Args:
            dialogue: Either a dict {speaker: text} or list of (speaker, text) tuples
            voices: Dict mapping speaker names to voice names
            output_file: Optional filename to save audio to
            style_prompt: Optional overall style instruction
            
        Returns:
            bytes: Audio data, or None on error
            
        Example:
            audio = tts.generate_multi_speaker_dialogue(
                dialogue=[
                    ("Host", "Welcome to our podcast!"),
                    ("Guest", "Thanks for having me!"),
                    ("Host", "Let's dive into today's topic...")
                ],
                voices={"Host": "Kore", "Guest": "Puck"},
                style_prompt="Make Host sound professional and Guest sound enthusiastic"
            )
        """
        # Convert dialogue to formatted text
        if isinstance(dialogue, dict):
            dialogue_items = list(dialogue.items())
        else:
            dialogue_items = dialogue
            
        # Validate speakers
        speakers = list(set(speaker for speaker, _ in dialogue_items))
        if len(speakers) > 2:
            logging.error("Multi-speaker TTS supports maximum 2 speakers")
            return None
            
        for speaker in speakers:
            if speaker not in voices:
                logging.error(f"No voice specified for speaker '{speaker}'")
                return None
                
        # Build prompt
        dialogue_text = "\n".join([f"{speaker}: {text}" for speaker, text in dialogue_items])
        
        if style_prompt:
            contents = f"{style_prompt}\n\nTTS the following conversation:\n{dialogue_text}"
        else:
            contents = f"TTS the following conversation:\n{dialogue_text}"
            
        # Create multi-speaker config
        speaker_configs = []
        for speaker in speakers:
            voice_name = voices[speaker]
            if voice_name not in TTS_VOICES:
                logging.warning(f"Unknown voice '{voice_name}' for speaker '{speaker}'")
                voice_name = self.default_voice
                
            speaker_configs.append(
                types.SpeakerVoiceConfig(
                    speaker=speaker,
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=voice_name
                        )
                    )
                )
            )
        
        speech_config = types.SpeechConfig(
            multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                speaker_voice_configs=speaker_configs
            )
        )
        
        # Dynamic key allocation with retry logic (like GeminiStreamingProcessor)
        max_attempts = 10000
        attempt_count = 0
        
        while attempt_count < max_attempts:
            # Get any available key
            api_key = self.key_manager.get_any_available_key("tts_multi")
            
            if api_key == ALL_KEYS_WAITING_MARKER:
                # All keys are busy, wait and retry
                logging.debug("All keys busy for multi-speaker, waiting...")
                time.sleep(10)
                attempt_count += 1
                continue
            elif api_key is None:
                logging.error("No usable API keys available")
                return None
            
            masked_key = f"...{api_key[-4:]}" if len(api_key) > 4 else "invalid key"
            
            try:
                # Create client
                logging.debug(f"Using key {masked_key} for multi-speaker TTS")
                client = genai.Client(api_key=api_key)
                
            except Exception as e:
                logging.error(f"Failed to initialize client with key {masked_key}: {e}")
                self.key_manager.mark_key_failed_init(api_key)
                continue
            
            # Make API call
            result = self._make_tts_api_call(client, contents, speech_config)
            
            if result == EXHAUSTED_MARKER:
                # Key is exhausted - mark it and try again with another key
                logging.warning(f"Key {masked_key} exhausted, trying another key")
                self.key_manager.mark_key_exhausted(api_key)
                continue
            elif result == PERSISTENT_ERROR_MARKER:
                # Persistent error - fail this request
                logging.error(f"Persistent error with key {masked_key}")
                return None
            elif isinstance(result, bytes):
                # Success!
                logging.debug(f"Multi-speaker TTS generation successful with key {masked_key}")
                self.key_manager.mark_key_successful(api_key)
                self.key_manager.mark_key_returned(api_key, "tts_multi")
                
                # Save to file if requested
                if output_file:
                    self._save_audio_to_file(result, output_file)
                
                return result
            
            attempt_count += 1
        
        # Maximum attempts exceeded
        logging.error(f"Multi-speaker TTS generation failed after {max_attempts} attempts")
        return None
    
    def generate_from_prompt(self,
                            model: str = "gemini-2.0-flash",
                            prompt: str = None,
                            voice: Optional[str] = None,
                            output_file: Optional[str] = None) -> Optional[bytes]:
        """
        Generate a text transcript using another Gemini model, then convert to speech.
        
        Args:
            model: Gemini model to use for text generation
            prompt: Prompt for generating the transcript
            voice: Voice to use for TTS
            output_file: Optional filename to save audio to
            
        Returns:
            bytes: Audio data, or None on error
            
        Example:
            audio = tts.generate_from_prompt(
                prompt="Generate a 30-second radio ad for a coffee shop",
                voice="Puck"
            )
        """
        # Dynamic key allocation with retry logic for text generation
        max_attempts = 10000
        attempt_count = 0
        transcript = None
        
        while attempt_count < max_attempts:
            # Get any available key for text generation
            api_key = self.key_manager.get_any_available_key("tts_generate")
            
            if api_key == ALL_KEYS_WAITING_MARKER:
                # All keys are busy, wait and retry
                logging.debug("All keys busy for text generation, waiting...")
                time.sleep(10)
                attempt_count += 1
                continue
            elif api_key is None:
                logging.error("No usable API keys available")
                return None
            
            masked_key = f"...{api_key[-4:]}" if len(api_key) > 4 else "invalid key"
            
            try:
                # Create client
                logging.debug(f"Using key {masked_key} for text generation")
                client = genai.Client(api_key=api_key)
                
                # Generate transcript with API call interval control
                with self._api_call_lock:
                    current_time = time.time()
                    time_since_last_call = current_time - self._last_api_call_time
                    
                    if time_since_last_call < self.api_call_interval:
                        sleep_time = self.api_call_interval - time_since_last_call
                        logging.debug(f"Waiting {sleep_time:.2f}s for API interval")
                        time.sleep(sleep_time)
                    
                    self._last_api_call_time = time.time()
                
                response = client.models.generate_content(
                    model=model,
                    contents=prompt
                )
                transcript = response.text.strip()
                logging.info(f"Generated transcript: {transcript[:100]}...")
                
                # Mark key as successful
                self.key_manager.mark_key_successful(api_key)
                self.key_manager.mark_key_returned(api_key, "tts_generate")
                break
                
            except Exception as e:
                if "429" in str(e) or "exhausted" in str(e).lower():
                    logging.warning(f"Key {masked_key} exhausted during text generation")
                    self.key_manager.mark_key_exhausted(api_key)
                    continue
                else:
                    logging.error(f"Error generating transcript with key {masked_key}: {e}")
                    return None
            
            attempt_count += 1
        
        if transcript is None:
            logging.error(f"Text generation failed after {max_attempts} attempts")
            return None
        
        # Convert to speech
        return self.generate_speech(
            text=transcript,
            voice=voice,
            output_file=output_file
        )
    
    def batch_generate_speech(self,
                            texts: List[Dict[str, Any]],
                            max_workers: int = 4) -> List[Tuple[Dict, Optional[bytes], Optional[str]]]:
        """
        Generate speech for multiple texts in parallel.
        
        Args:
            texts: List of dicts with keys:
                - 'text': Text to convert
                - 'voice': Optional voice name
                - 'output_file': Optional output filename
                - 'style_prompt': Optional style instruction
                - 'metadata': Optional metadata dict
            max_workers: Maximum parallel workers
            
        Returns:
            List of tuples (metadata, audio_bytes, error_msg)
            
        Example:
            results = tts.batch_generate_speech([
                {'text': 'Hello', 'voice': 'Kore', 'metadata': {'id': 1}},
                {'text': 'World', 'voice': 'Puck', 'metadata': {'id': 2}}
            ])
        """
        results = []
        
        def process_single(item):
            metadata = item.get('metadata', {})
            try:
                audio = self.generate_speech(
                    text=item['text'],
                    voice=item.get('voice'),
                    output_file=item.get('output_file'),
                    style_prompt=item.get('style_prompt')
                )
                if audio:
                    return (metadata, audio, None)
                else:
                    return (metadata, None, "TTS generation failed")
            except Exception as e:
                return (metadata, None, str(e))
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_single, item) for item in texts]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logging.error(f"Batch processing error: {e}")
                    results.append(({}, None, str(e)))
        
        return results
    
    def list_voices(self, style: Optional[str] = None) -> Dict[str, str]:
        """
        List available voices, optionally filtered by style.
        
        Args:
            style: Optional style to filter by (e.g., "Bright", "Firm")
            
        Returns:
            Dict of voice_name: description
        """
        if style:
            return {k: v for k, v in TTS_VOICES.items() if v.lower() == style.lower()}
        return TTS_VOICES.copy()


# Convenience function for quick TTS
def text_to_speech(text: str, 
                   voice: str = "Kore",
                   output_file: Optional[str] = None,
                   api_key: Optional[str] = None) -> Optional[bytes]:
    """
    Quick function to convert text to speech.
    
    Args:
        text: Text to convert
        voice: Voice name
        output_file: Optional output filename
        api_key: Optional API key (uses env var if not provided)
        
    Returns:
        Audio bytes or None on error
    """
    if api_key:
        key_manager = AdvancedApiKeyManager(["CUSTOM_KEY"])
        os.environ["CUSTOM_KEY"] = api_key
    else:
        key_manager = None
    
    tts = GeminiTTSProcessor(key_manager=key_manager)
    return tts.generate_speech(text, voice, output_file)
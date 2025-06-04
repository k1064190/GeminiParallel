# Gemini Parallel API Processor

A robust Python library for making parallel API calls to Google's Gemini AI models with advanced API key management, automatic retry mechanisms, and comprehensive error handling.

## Features

- **Parallel Processing**: Execute multiple Gemini API calls simultaneously with configurable worker threads
- **Advanced API Key Management**: Intelligent key rotation with cooldown periods and exhaustion recovery
- **Multi-Modal Support**: Process text, audio, and video inputs seamlessly
- **Resilient Error Handling**: Automatic retries, exponential backoff, and graceful degradation
- **Resource Management**: Smart handling of rate limits and quota exhaustion
- **Comprehensive Logging**: Detailed logging for monitoring and debugging

## Installation
### Using pip

```bash
pip install -r requirements.txt
or
pip install google-genai google-api-core python-dotenv
```

## Setup

1. Clone or download `gemini_parallel.py` to your project directory

2. Create a `.env` file in your project root and add your Gemini API keys:

```bash
# .env file
GEMINI_API_KEY_1=your_first_api_key_here
GEMINI_API_KEY_2=your_second_api_key_here
GEMINI_API_KEY_3=your_third_api_key_here
GEMINI_API_KEY_4=your_fourth_api_key_here
# Add as many keys as you have
```

3. Make sure to add `.env` to your `.gitignore` file:
```bash
echo ".env" >> .gitignore
```

## Quick Start

```python
from gemini_parallel import AdvancedApiKeyManager, GeminiParallelProcessor

# Initialize the API key manager (will automatically load from .env file)
key_manager = AdvancedApiKeyManager([
    "GEMINI_API_KEY_1",
    "GEMINI_API_KEY_2", 
    "GEMINI_API_KEY_3"
])

# Create the parallel processor
processor = GeminiParallelProcessor(
    key_manager=key_manager,
    model_name="gemini-2.0-flash-001",
    max_workers=4
)

# Prepare your prompts
prompts_data = [
    {
        "prompt": "What is the capital of France?",
        "metadata": {"task_id": "task_1", "category": "geography"}
    },
    {
        "prompt": "Explain quantum computing in simple terms.",
        "metadata": {"task_id": "task_2", "category": "science"}
    }
]

# Process in parallel
results = processor.process_prompts(prompts_data)

# Handle results
for metadata, response, error in results:
    if error:
        print(f"Task {metadata['task_id']} failed: {error}")
    else:
        print(f"Task {metadata['task_id']} result: {response[:100]}...")
```

## Multi-Modal Usage

Make sure to use audio_path when videos or audios are above 20MB.
### Text + Audio Processing
```python
prompts_data = [
    {
        "prompt": "Transcribe and summarize this audio:",
        "audio_path": "/path/to/audio/file.mp3",
        "audio_mime_type": "audio/mp3",
        "metadata": {"task_id": "audio_task_1"}
    },
    {
        "prompt": "What is being discussed in this audio?",
        "audio_bytes": open("audio.wav", "rb").read(),
        "audio_mime_type": "audio/wav",
        "metadata": {"task_id": "audio_task_2"}
    }
]

results = processor.process_prompts(prompts_data)
```

### Video Processing

```python
prompts_data = [
    {
        "prompt": "Describe what happens in this video:",
        "video_path": "/path/to/video.mp4",
        "video_mime_type": "video/mp4",
        "metadata": {"task_id": "video_task_1"}
    },
    {
        "prompt": "Analyze this video content:",
        "video_url": "https://youtube.com/video.mp4",
        "video_metadata": {"fps": 5},
        "metadata": {"task_id": "video_task_2"}
    },
    {
        "prompt": "Analyze this video content:",
        "video_url": "https://youtube.com/video.mp4",
        "video_metadata": {"start_offset"="1250s", "end_offset"="1570s"},
        "metadata": {"task_id": "video_task_2"}
    },
]

results = processor.process_prompts(prompts_data)
```

## Configuration Options

### AdvancedApiKeyManager Parameters

```python
key_manager = AdvancedApiKeyManager(
    keylist_names=["GEMINI_API_KEY_1", "GEMINI_API_KEY_2"],
    key_cooldown_seconds=60,           # Cooldown after each key use
    exhausted_wait_seconds=60,         # Wait time for temporary exhaustion
    fully_exhausted_wait_seconds=43200, # Wait time for full exhaustion (12 hours)
    max_exhausted_retries=3            # Max retries before marking fully exhausted
)
```

### GeminiParallelProcessor Parameters

```python
processor = GeminiParallelProcessor(
    key_manager=key_manager,
    model_name="gemini-2.0-flash-001",  # Gemini model to use
    api_call_interval=0.5,              # Minimum interval between API calls
    max_workers=4                       # Maximum concurrent workers
)
```

### Generation Configuration

You can customize the AI response generation:

```python
prompts_data = [
    {
        "prompt": "Write a creative story:",
        "generation_config": {
            "temperature": 0.9,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 1000,
            "candidate_count": 1
        },
        "metadata": {"task_id": "creative_task"}
    }
]
```

## API Key Management States

The system manages API keys through several states:

- **AVAILABLE**: Key is ready to use
- **COOLDOWN**: Key is in cooldown period after recent use
- **TEMPORARILY_EXHAUSTED**: Key hit rate limits, temporary wait
- **FULLY_EXHAUSTED**: Key exceeded retry limits, long wait period
- **FAILED_INIT**: Key failed initialization, won't be used

### Monitoring Key Status

```python
# Get current status of all keys
status_summary = key_manager.get_keys_status_summary()
for key_id, status_info in status_summary.items():
    print(f"Key {key_id}: {status_info['status']} "
          f"(exhausted: {status_info['exhausted_count']})")
```

## Error Handling

The system handles various types of errors:

1. **Resource Exhaustion**: Automatic key rotation and retry
2. **API Errors**: Exponential backoff and retry
3. **Network Issues**: Graceful degradation
4. **Invalid Inputs**: Clear error messages

### Result Processing

```python
for metadata, response, error in results:
    task_id = metadata.get('task_id', 'unknown')
    
    if error:
        if "exhausted" in error.lower():
            print(f"Task {task_id}: All keys exhausted, try again later")
        elif "persistent" in error.lower():
            print(f"Task {task_id}: Persistent API error")
        else:
            print(f"Task {task_id}: {error}")
    else:
        print(f"Task {task_id}: Success - {len(response)} characters")
```

## Logging Configuration

The library uses Python's logging module. Customize as needed:

```python
import logging

# Set log level
logging.getLogger().setLevel(logging.DEBUG)

# Custom format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s",
    handlers=[
        logging.FileHandler("gemini_parallel.log"),
        logging.StreamHandler()
    ]
)
```

## Common Issues

1. **"No valid API keys found"**
   - Check if `.env` file exists in your project root
   - Verify API key variable names in `.env` file
   - Ensure `.env` file is in the same directory as your script
   - Check if API keys are valid and active

2. **"All keys exhausted"**
   - Wait for cooldown periods to expire
   - Add more API keys to your `.env` file
   - Reduce request rate

3. **"Failed to initialize client"**
   - Check API key validity in `.env` file
   - Verify network connectivity
   - Check Google AI service status

4. **"Module not found" errors**
   - Make sure you've run `pip install google-genai google-api-core python-dotenv`

### Debug Mode

Enable debug logging for detailed information:

```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

## License

This project is provided as-is for educational and development purposes. Please comply with Google's Gemini API terms of service when using this library.

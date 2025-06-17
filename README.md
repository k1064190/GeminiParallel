# Gemini Parallel API Processor

A robust Python library for making parallel API calls to Google's Gemini AI models with advanced API key management, automatic retry mechanisms, and comprehensive error handling.

## Features

- **Parallel Processing**: Execute multiple Gemini API calls simultaneously with configurable worker threads
- **Advanced API Key Management**: Intelligent key rotation with cooldown periods and exhaustion recovery
- **Multi-Modal Support**: Process text, audio, and video inputs with flexible positioning using `<audio>` and `<video>` tokens
- **Multiple Media Files**: Support for multiple audio/video files per prompt with precise positioning control
- **Flexible Input Methods**: Support file paths, raw bytes, and URLs for media content
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

The library supports flexible positioning of multimedia content using `<audio>` and `<video>` tokens in your prompts. This allows you to specify exactly where each media file should appear in the context.

**Note**: Use file paths (`audio_path`, `video_path`) when files are above 20MB for better performance.

### Basic Multi-Modal Processing

```python
# Simple text + audio
prompts_data = [
    {
        "prompt": "Transcribe and summarize this audio:",
        "audio_path": "/path/to/audio/file.mp3",
        "audio_mime_type": "audio/mp3",
        "metadata": {"task_id": "audio_task_1"}
    }
]

results = processor.process_prompts(prompts_data)
```

### Advanced Positioning with Tokens

Use `<audio>` and `<video>` tokens to specify exact placement of media files:

```python
prompts_data = [
    {
        "prompt": "First, analyze this audio: <audio> Now compare it with this video: <video> Finally, what do you think about this second audio: <audio>",
        "audio_path": ["audio1.mp3", "audio2.wav"],
        "video_path": ["video1.mp4"],
        "metadata": {"task_id": "multimedia_analysis"}
    }
]

# This creates the sequence: text → audio1.mp3 → text → video1.mp4 → text → audio2.wav
```

### Multiple Files with Mixed Types

```python
prompts_data = [
    {
        "prompt": "Compare these recordings: <audio> <audio> with this visual content: <video>",
        "audio_path": ["interview1.mp3", "interview2.mp3"],
        "video_path": ["presentation.mp4"],
        "audio_mime_type": ["audio/mp3", "audio/mp3"],
        "video_mime_type": ["video/mp4"],
        "metadata": {"task_id": "comparison_task"}
    }
]
```

### Using Different Input Methods

```python
# Mix of paths, bytes, and URLs
prompts_data = [
    {
        "prompt": "Analyze this audio file: <audio> and this video: <video> then this audio data: <audio>",
        "audio_path": ["/path/to/audio1.mp3"],  # File path
        "audio_bytes": [open("audio2.wav", "rb").read()],  # Raw bytes
        "video_url": ["https://example.com/video.mp4"],  # URL
        "audio_mime_type": ["audio/mp3", "audio/wav"],
        "video_mime_type": ["video/mp4"],
        "metadata": {"task_id": "mixed_input_task"}
    }
]
```

### Video Processing with Metadata

```python
prompts_data = [
    {
        "prompt": "Analyze this video segment: <video> What are the key points?",
        "video_path": ["/path/to/video.mp4"],
        "video_metadata": [{"start_offset": "1250s", "end_offset": "1570s", "fps": 5}],
        "metadata": {"task_id": "video_segment_analysis"}
    },
    {
        "prompt": "Compare these two video clips: <video> <video>",
        "video_url": ["https://example.com/video1.mp4", "https://example.com/video2.mp4"],
        "video_metadata": [{"fps": 10}, {"start_offset": "30s", "end_offset": "60s"}],
        "metadata": {"task_id": "video_comparison"}
    }
]
```

### Complex Multi-Modal Scenarios

```python
# Advanced example with multiple media types and precise positioning
prompts_data = [
    {
        "prompt": """
        I need you to analyze this presentation. 
        
        First, listen to the introduction: <audio>
        
        Now watch the main content: <video>
        
        Then listen to the Q&A section: <audio>
        
        Finally, review this supplementary video: <video>
        
        Please provide a comprehensive summary covering all aspects.
        """,
        "audio_path": ["intro.mp3", "qa_session.mp3"],
        "video_path": ["main_presentation.mp4", "supplementary.mp4"],
        "audio_mime_type": ["audio/mp3", "audio/mp3"],
        "video_mime_type": ["video/mp4", "video/mp4"],
        "video_metadata": [{"fps": 5}, {"start_offset": "0s", "end_offset": "300s"}],
        "metadata": {"task_id": "comprehensive_analysis"}
    }
]
```

### Fallback Behavior

If you don't use positioning tokens, the library falls back to the original behavior:

```python
# Without tokens - media files are added before the text
prompts_data = [
    {
        "prompt": "Analyze the provided media files.",
        "audio_path": ["audio1.mp3", "audio2.mp3"],
        "video_path": ["video1.mp4"],
        "metadata": {"task_id": "fallback_example"}
    }
]
# Order: video1.mp4 → audio1.mp3 → audio2.mp3 → text
```

## Media Token Reference

### Supported Tokens

- `<audio>`: Insert audio file at this position
- `<video>`: Insert video file at this position

### Token Matching Rules

1. **Sequential Matching**: Tokens are matched with media files in the order they appear
2. **Type-Specific**: `<audio>` tokens match with audio files, `<video>` tokens with video files
3. **Unused Files**: Any unused media files are automatically appended at the end
4. **Missing Files**: If more tokens exist than available files, a warning is logged

### Supported Media Parameters

| Parameter | Single Value | Multiple Values | Description |
|-----------|--------------|-----------------|-------------|
| `audio_path` | `str` | `list[str]` | Path(s) to audio file(s) |
| `audio_bytes` | `bytes` | `list[bytes]` | Raw audio data |
| `audio_mime_type` | `str` | `list[str]` | MIME type(s) (default: `audio/mp3`) |
| `video_path` | `str` | `list[str]` | Path(s) to video file(s) |
| `video_bytes` | `bytes` | `list[bytes]` | Raw video data |
| `video_url` | `str` | `list[str]` | Video URL(s) |
| `video_mime_type` | `str` | `list[str]` | MIME type(s) (default: `video/mp4`) |
| `video_metadata` | `dict` | `list[dict]` | Video processing metadata |

### Example Token Usage

```python
# Complex positioning example
prompt = """
Analyze this conversation:

Speaker A: <audio>
Speaker B: <audio>

Now watch the presentation: <video>

Final thoughts on Speaker A: <audio>
"""

# Files will be matched in order:
# <audio> (1st) → audio_files[0]
# <audio> (2nd) → audio_files[1] 
# <video> (1st) → video_files[0]
# <audio> (3rd) → audio_files[2]
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
Look for "https://ai.google.dev/api/generate-content?#generationconfig" for more information about config parameters.

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

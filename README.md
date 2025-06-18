# Gemini Parallel API Processor

A robust Python library for making parallel API calls to Google's Gemini AI models with advanced API key management, automatic retry mechanisms, and comprehensive error handling. Supports both **batch processing** and **streaming processing** modes.

## Features

- **Dual Processing Modes**: Choose between batch processing (multiple requests at once) or streaming processing (persistent workers for real-time requests)
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

## Processing Modes

### 🔄 Batch Processing

Perfect for processing large amounts of data at once. All requests are submitted together and results are returned when all are complete.

```python
from gemini_parallel import AdvancedApiKeyManager, GeminiParallelProcessor

# Initialize the API key manager
key_manager = AdvancedApiKeyManager([
    "GEMINI_API_KEY_1",
    "GEMINI_API_KEY_2", 
    "GEMINI_API_KEY_3"
])

# Create the batch processor
processor = GeminiParallelProcessor(
    key_manager=key_manager,
    model_name="gemini-2.0-flash-001",
    worker_cooldown_seconds=5.0,  # Worker pacing
    api_call_interval=2.0,        # IP ban protection
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

# Process all at once
results = processor.process_prompts(prompts_data)

# Handle results
for metadata, response, error in results:
    if error:
        print(f"Task {metadata['task_id']} failed: {error}")
    else:
        print(f"Task {metadata['task_id']} result: {response[:100]}...")
```

### 🚀 Streaming Processing

Perfect for real-time applications, web services, and interactive systems. Maintains persistent workers and processes requests one by one as they come.

```python
from gemini_parallel import AdvancedApiKeyManager, GeminiStreamingProcessor

# Initialize the API key manager
key_manager = AdvancedApiKeyManager([
    "GEMINI_API_KEY_1",
    "GEMINI_API_KEY_2"
])

# Create the streaming processor
stream_processor = GeminiStreamingProcessor(
    key_manager=key_manager,
    model_name="gemini-2.0-flash-001",
    api_call_interval=2.0,
    max_workers=4
)

# Start persistent workers (do this once)
stream_processor.start()

try:
    # Process individual requests (can be called many times)
    prompt_data = {
        'prompt': 'What is the capital of France?',
        'metadata': {'task_id': 'question_1'}
    }
    
    # This call blocks until result is ready
    metadata, response, error = stream_processor.process_single(prompt_data)
    
    if error is None:
        print(f"Response: {response}")
    else:
        print(f"Error: {error}")
    
    # Process another request immediately
    prompt_data2 = {
        'prompt': 'Explain quantum physics briefly.',
        'metadata': {'task_id': 'question_2'}
    }
    
    metadata, response, error = stream_processor.process_single(prompt_data2)
    print(f"Second response: {response}")
    
    # Check worker status anytime
    status = stream_processor.get_worker_status()
    print(f"Queue size: {status['queue_size']}")
    print(f"Active workers: {status['active_workers']}")
    
finally:
    # Always stop workers when done
    stream_processor.stop()
```

### 🎯 When to Use Each Mode

| Use Case | Batch Processing | Streaming Processing |
|----------|-----------------|---------------------|
| **Large Dataset Processing** | ✅ Perfect | ✅ Good |
| **Web API Backend** | ❌ Too slow | ✅ Perfect |
| **Interactive Chatbot** | ❌ Poor UX | ✅ Perfect |
| **Batch Analysis Jobs** | ✅ Perfect | ✅ Good |
| **Real-time Applications** | ❌ Too slow | ✅ Perfect |
| **Jupyter Notebooks** | ✅ Good | ✅ Great for testing |

## Streaming Processor Advanced Usage

### Error Handling with Timeout

```python
try:
    # Set custom timeout (default: 300 seconds)
    metadata, response, error = stream_processor.process_single(
        prompt_data, 
        timeout=60.0  # 1 minute timeout
    )
except TimeoutError:
    print("Request timed out!")
except RuntimeError:
    print("Workers not running - call start() first!")
```

### Multimedia Processing with Streaming

```python
# Audio analysis
audio_prompt = {
    'prompt': 'Transcribe and summarize: <audio>',
    'audio_path': '/path/to/audio.mp3',
    'metadata': {'task_id': 'audio_analysis'}
}

metadata, response, error = stream_processor.process_single(audio_prompt)

# Video analysis
video_prompt = {
    'prompt': 'What happens in this video: <video>',
    'video_path': '/path/to/video.mp4',
    'metadata': {'task_id': 'video_analysis'}
}

metadata, response, error = stream_processor.process_single(video_prompt)
```

### Worker Management

```python
# Check if workers are running
if not stream_processor.is_running():
    stream_processor.start()

# Monitor queue
print(f"Requests in queue: {stream_processor.get_queue_size()}")

# Get detailed status
status = stream_processor.get_worker_status()
print(f"Workers running: {status['workers_running']}")
print(f"Max workers: {status['max_workers']}")
print(f"Key status: {status['key_status']}")

# Stop and restart if needed
stream_processor.stop()
stream_processor.start()
```

### Integration with Web Frameworks

#### Flask Example

```python
from flask import Flask, request, jsonify
from gemini_parallel import AdvancedApiKeyManager, GeminiStreamingProcessor

app = Flask(__name__)

# Initialize once at startup
key_manager = AdvancedApiKeyManager(["GEMINI_API_KEY_1", "GEMINI_API_KEY_2"])
processor = GeminiStreamingProcessor(key_manager, "gemini-2.0-flash-001")
processor.start()

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    
    prompt_data = {
        'prompt': user_message,
        'metadata': {'task_id': f'web_request_{request.remote_addr}'}
    }
    
    try:
        metadata, response, error = processor.process_single(prompt_data, timeout=30.0)
        
        if error:
            return jsonify({'error': error}), 500
        else:
            return jsonify({'response': response})
            
    except TimeoutError:
        return jsonify({'error': 'Request timeout'}), 408

@app.teardown_appcontext
def shutdown_processor(exception):
    processor.stop()

if __name__ == '__main__':
    app.run()
```

#### FastAPI Example

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from gemini_parallel import AdvancedApiKeyManager, GeminiStreamingProcessor

app = FastAPI()

# Initialize at startup
key_manager = AdvancedApiKeyManager(["GEMINI_API_KEY_1", "GEMINI_API_KEY_2"])
processor = GeminiStreamingProcessor(key_manager, "gemini-2.0-flash-001")

class ChatRequest(BaseModel):
    message: str
    task_id: str = None

@app.on_event("startup")
async def startup_event():
    processor.start()

@app.on_event("shutdown")
async def shutdown_event():
    processor.stop()

@app.post("/chat")
async def chat(request: ChatRequest):
    prompt_data = {
        'prompt': request.message,
        'metadata': {'task_id': request.task_id or 'api_request'}
    }
    
    try:
        metadata, response, error = processor.process_single(prompt_data, timeout=30.0)
        
        if error:
            raise HTTPException(status_code=500, detail=error)
        
        return {"response": response, "task_id": metadata.get('task_id')}
        
    except TimeoutError:
        raise HTTPException(status_code=408, detail="Request timeout")

@app.get("/status")
async def get_status():
    return processor.get_worker_status()
```

## Multi-Modal Usage

The library supports flexible positioning of multimedia content using `<audio>` and `<video>` tokens in your prompts. This allows you to specify exactly where each media file should appear in the context.

**Note**: Use file paths (`audio_path`, `video_path`) when files are above 20MB for better performance.

### Basic Multi-Modal Processing

```python
# Works with both batch and streaming processors
prompts_data = [
    {
        "prompt": "Transcribe and summarize this audio:",
        "audio_path": "/path/to/audio/file.mp3",
        "audio_mime_type": "audio/mp3",
        "metadata": {"task_id": "audio_task_1"}
    }
]

# Batch processing
results = processor.process_prompts(prompts_data)

# OR Streaming processing
metadata, response, error = stream_processor.process_single(prompts_data[0])
```

### Advanced Positioning with Tokens

Use `<audio>` and `<video>` tokens to specify exact placement of media files:

```python
prompt_data = {
    "prompt": "First, analyze this audio: <audio> Now compare it with this video: <video> Finally, what do you think about this second audio: <audio>",
    "audio_path": ["audio1.mp3", "audio2.wav"],
    "video_path": ["video1.mp4"],
    "metadata": {"task_id": "multimedia_analysis"}
}

# This creates the sequence: text → audio1.mp3 → text → video1.mp4 → text → audio2.wav

# Use with either processor
metadata, response, error = stream_processor.process_single(prompt_data)
```

### Multiple Files with Mixed Types

```python
prompt_data = {
    "prompt": "Compare these recordings: <audio> <audio> with this visual content: <video>",
    "audio_path": ["interview1.mp3", "interview2.mp3"],
    "video_path": ["presentation.mp4"],
    "audio_mime_type": ["audio/mp3", "audio/mp3"],
    "video_mime_type": ["video/mp4"],
    "metadata": {"task_id": "comparison_task"}
}
```

### Using Different Input Methods

```python
# Mix of paths, bytes, and URLs
prompt_data = {
    "prompt": "Analyze this audio file: <audio> and this video: <video> then this audio data: <audio>",
    "audio_path": ["/path/to/audio1.mp3"],  # File path
    "audio_bytes": [open("audio2.wav", "rb").read()],  # Raw bytes
    "video_url": ["https://example.com/video.mp4"],  # URL
    "audio_mime_type": ["audio/mp3", "audio/wav"],
    "video_mime_type": ["video/mp4"],
    "metadata": {"task_id": "mixed_input_task"}
}
```

### Video Processing with Metadata

```python
prompt_data = {
    "prompt": "Analyze this video segment: <video> What are the key points?",
    "video_path": ["/path/to/video.mp4"],
    "video_metadata": [{"start_offset": "1250s", "end_offset": "1570s", "fps": 5}],
    "metadata": {"task_id": "video_segment_analysis"}
}
```

### Fallback Behavior

If you don't use positioning tokens, the library falls back to the original behavior:

```python
# Without tokens - media files are added before the text
prompt_data = {
    "prompt": "Analyze the provided media files.",
    "audio_path": ["audio1.mp3", "audio2.mp3"],
    "video_path": ["video1.mp4"],
    "metadata": {"task_id": "fallback_example"}
}
# Order: video1.mp4 → audio1.mp3 → audio2.mp3 → text
```

## Configuration Options

### AdvancedApiKeyManager Parameters

```python
key_manager = AdvancedApiKeyManager(
    keylist_names=["GEMINI_API_KEY_1", "GEMINI_API_KEY_2"],
    key_cooldown_seconds=30,           # Cooldown after each key use (default: 30s)
    exhausted_wait_seconds=120,        # Wait time for temporary exhaustion (default: 120s)
    fully_exhausted_wait_seconds=43200, # Wait time for full exhaustion (default: 12 hours)
    max_exhausted_retries=3            # Max retries before marking fully exhausted (default: 3)
)
```

### Processor Parameters

```python
# Batch Processor
processor = GeminiParallelProcessor(
    key_manager=key_manager,
    model_name="gemini-2.0-flash-001",      # Gemini model to use
    worker_cooldown_seconds=5.0,            # Worker cooldown between API calls (default: 5s)
    api_call_interval=2.0,                  # Global interval between ANY API calls (default: 2s)
    max_workers=4                           # Maximum concurrent workers
)

# Streaming Processor (same parameters except no worker_cooldown_seconds)
stream_processor = GeminiStreamingProcessor(
    key_manager=key_manager,
    model_name="gemini-2.0-flash-001",
    api_call_interval=2.0,                  # Global interval to prevent IP ban
    max_workers=4
)
```

### Generation Configuration

You can customize the AI response generation:

```python
prompt_data = {
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

# Works with both processors
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

# For streaming processor, also check worker status
if hasattr(processor, 'get_worker_status'):  # Streaming processor
    worker_status = processor.get_worker_status()
    print(f"Workers running: {worker_status['workers_running']}")
    print(f"Queue size: {worker_status['queue_size']}")
```

## Error Handling

The system handles various types of errors:

1. **Resource Exhaustion**: Automatic key rotation and retry
2. **API Errors**: Exponential backoff and retry
3. **Network Issues**: Graceful degradation
4. **Invalid Inputs**: Clear error messages

### Result Processing

```python
# Batch processing
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

# Streaming processing
try:
    metadata, response, error = stream_processor.process_single(prompt_data, timeout=60)
    if error:
        print(f"Error: {error}")
    else:
        print(f"Success: {response}")
except TimeoutError:
    print("Request timed out")
except RuntimeError as e:
    print(f"Runtime error: {e}")
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

3. **"Workers not running" (Streaming only)**
   - Call `stream_processor.start()` before processing requests
   - Check if `stop()` was called accidentally

4. **"Request timeout" (Streaming only)**
   - Increase timeout parameter in `process_single()`
   - Check if workers are stuck due to key exhaustion

5. **"Failed to initialize client"**
   - Check API key validity in `.env` file
   - Verify network connectivity
   - Check Google AI service status

### Debug Mode

Enable debug logging for detailed information:

```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

## License

This project is provided as-is for educational and development purposes. Please comply with Google's Gemini API terms of service when using this library.

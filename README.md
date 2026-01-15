# Gemini Parallel

A clean, simple Python library for Google Gemini API with intelligent key management and automatic rate limiting.

## Features

- **Simple Sequential Processing** - No threading complexity, just straightforward API calls
- **Smart Key Management** - Automatic key rotation, cooldown, and exhaustion recovery
- **CLI Key Manager** - Easy command-line tool to manage your API keys
- **Multi-Modal Support** - Text, audio, and video inputs with flexible positioning
- **Auto Rate Limiting** - Built-in protection against IP bans
- **Adaptive Cooldown** - Automatically adjusts to API rate limits

## Installation

```bash
pip install gemini-parallel
```

Or install from source:

```bash
git clone https://github.com/your-repo/gemini-parallel
cd gemini-parallel
pip install -e .
```

## Quick Start

### 1. Set up your API keys

```bash
# Initialize .env file
geminiparallel init

# Add your API key(s)
geminiparallel add YOUR_GEMINI_API_KEY_HERE

# List your keys (masked)
geminiparallel list

# Test your keys
geminiparallel test
```

### 2. Use in your code

```python
from gemini_parallel import GeminiSequentialProcessor, AdvancedApiKeyManager

# Initialize key manager (reads from .env)
key_manager = AdvancedApiKeyManager(keylist_names="all")

# Create processor
processor = GeminiSequentialProcessor(
    key_manager=key_manager,
    model_name="gemini-3-flash-preview",
    api_call_interval=4.0  # IP ban protection (default: 4.0)
)

# Process a single request
result = processor.process_single({
    'prompt': 'What is the capital of France?',
    'metadata': {'task_id': 'question_1'}
})

metadata, response, error = result
if error is None:
    print(response)
```

### 3. Process multiple requests

```python
prompts = [
    {'prompt': 'What is AI?', 'metadata': {'task_id': 'q1'}},
    {'prompt': 'Explain quantum computing', 'metadata': {'task_id': 'q2'}},
    {'prompt': 'What is Python?', 'metadata': {'task_id': 'q3'}}
]

results = processor.process_batch(prompts)

for metadata, response, error in results:
    if error:
        print(f"{metadata['task_id']} failed: {error}")
    else:
        print(f"{metadata['task_id']}: {response[:100]}...")
```

## CLI Commands

Manage your API keys easily with the `geminiparallel` command:

| Command | Description |
|---------|-------------|
| `geminiparallel init` | Initialize .env file |
| `geminiparallel add KEY` | Add a new API key |
| `geminiparallel list` | List all keys (masked) |
| `geminiparallel remove KEY_NAME` | Remove a specific key |
| `geminiparallel test` | Test all keys |

**Example workflow:**

```bash
# Set up new project
cd my-project
geminiparallel init

# Add your keys
geminiparallel add AIza...your_key_1
geminiparallel add AIza...your_key_2

# Verify they work
geminiparallel test

# Check what you have
geminiparallel list
```

## Multi-Modal Usage

### Image Processing

```python
# Single image from file
result = processor.process_single({
    'prompt': 'Describe this image: <image>',
    'image_path': '/path/to/image.jpg',
    'metadata': {'task_id': 'image_1'}
})

# Image from URL
result = processor.process_single({
    'prompt': 'What is in this image?',
    'image_url': 'https://example.com/image.jpg',
    'metadata': {'task_id': 'image_2'}
})

# Image from bytes
with open('/path/to/image.png', 'rb') as f:
    image_bytes = f.read()

result = processor.process_single({
    'prompt': 'Analyze this image: <image>',
    'image_bytes': image_bytes,
    'image_mime_type': 'image/png',
    'metadata': {'task_id': 'image_3'}
})

# Multiple images
result = processor.process_single({
    'prompt': 'Compare <image> with <image>',
    'image_path': ['/path/to/image1.jpg', '/path/to/image2.jpg'],
    'metadata': {'task_id': 'multi_image'}
})
```

### Audio Processing

```python
result = processor.process_single({
    'prompt': 'Transcribe and summarize: <audio>',
    'audio_path': '/path/to/audio.mp3',
    'metadata': {'task_id': 'audio_1'}
})
```

### Video Processing

```python
result = processor.process_single({
    'prompt': 'What happens in this video: <video>',
    'video_path': '/path/to/video.mp4',
    'metadata': {'task_id': 'video_1'}
})
```

### Multiple Media Files

```python
result = processor.process_single({
    'prompt': 'Describe <image>, then transcribe <audio>, and summarize <video>',
    'image_path': '/path/to/image.jpg',
    'audio_path': 'audio1.mp3',
    'video_path': 'video1.mp4',
    'metadata': {'task_id': 'multi_1'}
})
```

## Key Management

### Basic Setup

```python
# Load all GEMINI_API_KEY_* from .env
key_manager = AdvancedApiKeyManager(keylist_names="all")

# Or specify exact keys
key_manager = AdvancedApiKeyManager(
    keylist_names=["GEMINI_API_KEY_1", "GEMINI_API_KEY_2"]
)

# Or use first N keys
key_manager = AdvancedApiKeyManager(keylist_names=5)
```

### Paid vs Free Keys

```python
# Mark some keys as paid (no cooldown)
key_manager = AdvancedApiKeyManager(
    keylist_names="all",
    paid_keys=["GEMINI_API_KEY_1", "GEMINI_API_KEY_2"]
)

# Or mark all as paid
key_manager = AdvancedApiKeyManager(
    keylist_names="all",
    paid_keys="all"
)
```

### Custom Settings

```python
key_manager = AdvancedApiKeyManager(
    keylist_names="all",
    key_settings={
        "free": {
            "key_cooldown_seconds": 30,      # 30s cooldown for free keys
            "exhausted_wait_seconds": 120,   # 2 min wait on rate limit
        },
        "paid": {
            "key_cooldown_seconds": 0,       # No cooldown for paid keys
            "exhausted_wait_seconds": 60,    # 1 min wait on rate limit
        }
    }
)
```

## Generation Configuration

Customize AI responses:

```python
result = processor.process_single({
    'prompt': 'Write a creative story',
    'generation_config': {
        'temperature': 1.0,  # Gemini 3: Keep at 1.0 (default, recommended)
        'top_p': 0.8,
        'max_output_tokens': 1000,
        'thinking_config': {
            'thinking_level': 'high'  # minimal, low, medium, high
        }
    },
    'metadata': {'task_id': 'creative_1'}
})
```

### Gemini 3 Best Practices

**Model Selection**:
- `gemini-3-flash-preview`: Fast, cost-effective, Pro-level intelligence
- `gemini-3-pro-preview`: Complex tasks requiring advanced reasoning

**Temperature**: Keep at **1.0** (default). Gemini 3's reasoning is optimized for this value. Lower values may cause looping or degraded performance.

**Thinking Level**: Controls reasoning depth
- `minimal`: Fastest, minimal thinking (Flash only)
- `low`: Simple tasks, low latency
- `medium`: Balanced (Flash only)  
- `high`: Maximum reasoning (default, dynamic)

**Prompting Tips**:
- Be **concise and direct** - Gemini 3 prefers clear instructions
- For large context (books, codebases): place questions **at the end**
- For verbose responses: explicitly request conversational style
- Anchor reasoning: "Based on the information above..."

### Accessing Full Response (Advanced)

By default, `process_single()` returns only the text response. To access the full response object (including thinking process, metadata, etc.):

```python
# Enable full response mode
processor = GeminiSequentialProcessor(
    key_manager=key_manager,
    model_name="gemini-3-flash-preview",
    return_response=True  # Return full response object
)

metadata, response_obj, error = processor.process_single({
    'prompt': 'Solve this problem step by step',
    'generation_config': {
        'thinking_config': {
            'include_thoughts': True  # Enable thinking process in response
        }
    },
    'metadata': {'task_id': 'advanced_1'}
})

if not error:
    # Access thinking process
    for part in response_obj.candidates[0].content.parts:
        if hasattr(part, 'thought') and part.thought:
            print(f"Thinking: {part.thought}")
        if part.text:
            print(f"Answer: {part.text}")
    
    # Or just get the text
    print(response_obj.text)
```

**Note**: The thinking process is only included if `include_thoughts=True` is set in `thinking_config`. By default, only the final answer is returned.

## Image Generation (Nano Banana)

Generate and edit images using the `GeminiImageGenerator`.

### Text-to-Image

```python
from gemini_parallel import GeminiImageGenerator, AdvancedApiKeyManager

key_manager = AdvancedApiKeyManager(keylist_names="all")
generator = GeminiImageGenerator(
    key_manager=key_manager,
    model_name="gemini-3-pro-image-preview"  # or gemini-2.5-flash-image
)

# Generate an image
image = generator.generate_image(
    prompt="A futuristic city with flying cars at sunset, cyberpunk style",
    output_file="city.png",
    aspect_ratio="16:9",
    image_size="2K"  # 1K, 2K, 4K (Pro only)
)
```

### Image Editing

```python
# Edit an existing image
edited_image = generator.edit_image(
    prompt="Make it snowy",
    input_image="city.png",
    output_file="city_snowy.png"
)
```

## Error Handling

The library handles errors automatically:

- **Resource Exhaustion (429)** - Tries another key automatically
- **API Errors** - Retries with exponential backoff
- **Network Issues** - Graceful degradation

```python
metadata, response, error = processor.process_single(prompt_data)

if error:
    if "Fatal: No usable API keys" in error:
        print("All keys are exhausted or invalid")
    elif "Persistent API error" in error:
        print("API call failed after retries")
    else:
        print(f"Error: {error}")
else:
    print(f"Success: {response}")
```

## Rate Limiting

The library protects you from IP bans:

- **API Call Interval** (default 4s) - Minimum time between ANY API calls
- **Key Cooldown** - Per-key cooldown after use (configurable)
- **Adaptive Cooldown** - Automatically increases intervals if too many 429 errors

```python
processor = GeminiSequentialProcessor(
    key_manager=key_manager,
    model_name="gemini-3-flash-preview",
    api_call_interval=4.0,      # Global rate limit (IP ban protection, default: 4.0)
    api_call_retries=3          # Max retries per request
)
```

## Text-to-Speech (TTS)

Generate speech from text:

```python
from gemini_parallel import GeminiTTSProcessor

tts = GeminiTTSProcessor(
    key_manager=key_manager,
    model="flash"
)

# Generate speech
audio = tts.generate_speech(
    text="Hello, world!",
    voice="Kore",
    output_file="hello.wav"
)

# Multi-speaker dialogue
dialogue = [
    ("Speaker1", "Hello there!"),
    ("Speaker2", "Hi, how are you?")
]

audio = tts.generate_multi_speaker_dialogue(
    dialogue=dialogue,
    voices={"Speaker1": "Kore", "Speaker2": "Puck"},
    output_file="conversation.wav"
)
```

## Environment Variables

Your `.env` file (managed by `geminiparallel` CLI):

```bash
# Gemini API Keys
GEMINI_API_KEY_1=your_first_key_here
GEMINI_API_KEY_2=your_second_key_here
GEMINI_API_KEY_3=your_third_key_here
```

**Important:** Add `.env` to your `.gitignore`!

```bash
echo ".env" >> .gitignore
```
## License

MIT License - See LICENSE file for details.

## Contributing

Contributions welcome! Please open an issue or PR on GitHub.

"""
Advanced Gemini API Parallel Processing System Usage Example

New features:
- Key cooldown after use (1 minute default)  
- Gradual exhausted management (temporary → complete)
- Continuous exhausted count tracking
- Automatic recovery system
"""

import os
import time
import logging
from gemini_parallel import AdvancedApiKeyManager, GeminiParallelProcessor
import dotenv

dotenv.load_dotenv()

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

KEY_NAMES = [f"GEMINI_API_KEY_{i}" for i in range(1, 17)]

def main():
    print("🚀 Advanced Gemini API Parallel Processing System Example")
    print("=" * 60)
    
    # 1. Initialize advanced API key manager
    print("\n📋 1. Initialize API key manager")
    
    try:
        key_manager = AdvancedApiKeyManager(
            keylist_names=KEY_NAMES,
        )
        print(f"✅ {len(key_manager.api_keys)} keys initialized")
        
    except ValueError as e:
        print(f"❌ Failed to initialize key manager: {e}")
        print("Check your environment variables:")
        for key_name in KEY_NAMES:
            value = os.getenv(key_name)
            status = "✅ Set" if value else "❌ Not set"
            print(f"  {key_name}: {status}")
        return

    # 2. Initialize parallel processor
    print("\n🔧 2. Initialize parallel processor")
    processor = GeminiParallelProcessor(
        key_manager=key_manager,
        model_name="gemini-2.0-flash-001",
        api_call_interval=0.5,  # 0.5 second interval
        max_workers=4           # 4 workers
    )
    print("✅ Processor initialized")

    # 3. Prepare test prompts
    print("\n📝 3. Prepare test prompts")
    test_prompts = [
        {
            'prompt': "Analyze the sentiment of the following sentence: 'Today was a great day!'",
            'metadata': {'task_id': 'sentiment_1', 'type': 'sentiment_analysis'}
        },
        {
            'prompt': "Show a simple example of using list comprehension in Python.",
            'metadata': {'task_id': 'coding_1', 'type': 'coding_help'}
        },
        {
            'prompt': "Explain the future of artificial intelligence in one paragraph.",
            'metadata': {'task_id': 'essay_1', 'type': 'essay'}
        },
        {
            'prompt': "Translate the following English sentence into Korean: 'Artificial intelligence is transforming our world.'",
            'metadata': {'task_id': 'translate_1', 'type': 'translation'}
        },
        {
            'prompt': "Recommend 3 healthy breakfast menus.",
            'metadata': {'task_id': 'recommend_1', 'type': 'recommendation'}
        }
    ]
    print(f"✅ {len(test_prompts)} prompts prepared")

    # 4. Check initial key status
    print("\n🔍 4. Check initial key status")
    status_summary = key_manager.get_keys_status_summary()
    for key_id, info in status_summary.items():
        print(f"  Key {key_id}: {info['status']} (exhausted: {info['exhausted_count']}/{info['total_exhausted_count']})")

    # 5. Execute parallel processing
    print("\n🚀 5. Execute parallel processing")
    start_time = time.time()
    
    results = processor.process_prompts(test_prompts)
    
    end_time = time.time()
    processing_time = end_time - start_time

    # 6. Analyze results
    print(f"\n📊 6. Analyze results (processing time: {processing_time:.2f}s)")
    success_count = 0
    error_count = 0
    
    for metadata, response, error in results:
        task_id = metadata['task_id']
        task_type = metadata['type']
        
        if error is None:
            success_count += 1
            response_preview = response[:100] + "..." if len(response) > 100 else response
            print(f"✅ {task_id} ({task_type}): {response_preview}")
        else:
            error_count += 1
            print(f"❌ {task_id} ({task_type}): {error}")

    print(f"\nSuccess: {success_count}/{len(results)} | Failure: {error_count}/{len(results)}")

    # 7. Check final key status
    print("\n🔍 7. Check final key status")
    final_status = key_manager.get_keys_status_summary()
    for key_id, info in final_status.items():
        print(f"  Key {key_id}: {info['status']} (exhausted: {info['exhausted_count']}/{info['total_exhausted_count']})")

    # 8. Key status change simulation (optional)
    if input("\n🔄 Check key status change in real-time? (y/n): ").lower() == 'y':
        demonstrate_key_recovery(key_manager)

def demonstrate_key_recovery(key_manager):
    """Demonstrates the key recovery process in real-time."""
    print("\n⏰ Monitor key status change (30 seconds)")
    print("Check cooldown and recovery process in real-time...")
    
    for i in range(30):
        status = key_manager.get_keys_status_summary()
        available_count = sum(1 for info in status.values() if info['status'] == 'AVAILABLE')
        cooldown_count = sum(1 for info in status.values() if info['status'] == 'COOLDOWN')
        exhausted_count = sum(1 for info in status.values() if 'EXHAUSTED' in info['status'])
        
        print(f"⏱️  {i+1:2d}s: AVAILABLE({available_count}) | COOLDOWN({cooldown_count}) | EXHAUSTED({exhausted_count})")
        time.sleep(1)
    
    print("✅ Monitoring completed!")

def stress_test():
    """Stress test: Test key management system with many requests"""
    print("\n🔥 Start stress test")
    
    # Generate more prompts
    stress_prompts = []
    for i in range(20):
        stress_prompts.append({
            'prompt': f"Explain the number {i+1} in one sentence.",
            'metadata': {'task_id': f'stress_{i+1}', 'type': 'stress_test'}
        })
    
    key_manager = AdvancedApiKeyManager(
        keylist_names=KEY_NAMES,
        key_cooldown_seconds=10,    # Short cooldown for quick testing
        exhausted_wait_seconds=30,
        fully_exhausted_wait_seconds=60,
        max_exhausted_retries=1     # Quick exhausted transition
    )
    
    processor = GeminiParallelProcessor(
        key_manager=key_manager,
        model_name="gemini-2.0-flash-001",
        max_workers=4
    )
    
    start_time = time.time()
    results = processor.process_prompts(stress_prompts)
    end_time = time.time()
    
    success_count = sum(1 for _, _, error in results if error is None)
    print(f"Result: {success_count}/{len(results)} success ({end_time - start_time:.2f}s)")

if __name__ == "__main__":
    main()
    
    # Check whether to run stress test
    if input("\n🔥 Will you run stress test? (y/n): ").lower() == 'y':
        stress_test() 
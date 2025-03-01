#!/usr/bin/env python3
# cost: 0.05 / 1M tokens input, 0.25 / 1M tokens output
"""
Test script for streaming with Replicate's Llama 3 model.
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv
import time

# Add the parent directory to the path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
load_dotenv()

import replicate
from core.llm_backends import create_llm_backend

def test_streaming_direct():
    """
    Test streaming directly with Replicate's API using the example from their website.
    """
    # Check if REPLICATE_API_TOKEN is set
    if not os.environ.get("REPLICATE_API_TOKEN"):
        print("ERROR: REPLICATE_API_TOKEN environment variable not found. Please set it before running.")
        sys.exit(1)
    
    print("Testing streaming with Llama 3 8B model directly...")
    print("=" * 80)
    
    # The prompt
    prompt = "Johnny has 8 billion parameters. His friend Tommy has 70 billion parameters. What does this mean when it comes to speed?"
    system_prompt = "You are a helpful assistant"
    
    print(f"Prompt: {prompt}")
    print("Response: ", end="", flush=True)
    
    # Track timing
    start_time = time.time()
    
    for event in replicate.stream(
        "meta/meta-llama-3-8b-instruct",
        input={
            "top_k": 0,
            "top_p": 0.95,
            "prompt": prompt,
            "max_tokens": 512,
            "temperature": 0.7,
            "system_prompt": system_prompt,
            "length_penalty": 1,
            "max_new_tokens": 512,
            "stop_sequences": "<|end_of_text|>,<|eot_id|>",
            "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            "presence_penalty": 0,
            "log_performance_metrics": False
        },
    ):
        print(str(event), end="", flush=True)
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print(f"Response time: {elapsed_time:.2f} seconds")
    print("=" * 80)

def test_streaming_with_backend():
    """
    Test streaming using our ReplicateBackend class.
    """
    try:
        # Create the backend using the factory function
        backend = create_llm_backend("replicate")
        
        print("Testing streaming with Llama 3 8B model using our backend...")
        print("=" * 80)
        
        # Test prompt
        prompt = "What are the key differences between Llama 3 and GPT-4? Please list 3 main points."
        system_prompt = "You are a helpful AI assistant that provides concise, accurate information."
        
        print(f"Prompt: {prompt}")
        print("Response: ", end="", flush=True)
        
        # Track timing
        start_time = time.time()
        
        # Call the API with streaming
        stream_iterator = backend.call(
            model="llama3:8b",
            prompt=prompt,
            temperature=0.7,
            max_tokens=512,
            system_prompt=system_prompt,
            stream=True
        )
        
        # Print the streaming response
        for chunk in stream_iterator:
            print(str(chunk), end="", flush=True)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        print("\n" + "=" * 80)
        print(f"Response time: {elapsed_time:.2f} seconds")
        print("=" * 80)
        
    except ValueError as e:
        print(f"ERROR: {str(e)}")
        sys.exit(1)

def test_fact_extraction_prompt():
    """
    Test a fact extraction prompt with streaming to see how it performs.
    """
    try:
        # Create the backend using the factory function
        backend = create_llm_backend("replicate")
        
        print("Testing fact extraction with Llama 3 8B model...")
        print("=" * 80)
        
        # Sample text
        text = """
        Jennifer grew up in Colorado and graduated from a small high school with only 79 people in her class.
        She joined the military at 17 to escape her small town. She served for five years, describing it as
        "the best of times, worst of times." Her childhood was difficult, with money being tight and limited
        educational opportunities. After the military, Jennifer got married and pursued acting as a way to
        deal with her experiences, finding it more rewarding than seeking help through the VA.
        """
        
        # Fact extraction prompt
        prompt = f"""
        Extract atomic facts from the following text. An atomic fact is a single, self-contained piece of information that:
        1. Contains exactly ONE piece of information
        2. Can stand alone without context
        3. Has no ambiguous pronouns (replace them with actual names/entities)
        4. Includes relevant temporal information if available
        
        Text to process:
        {text}
        
        Format each fact as a numbered list.
        """
        
        system_prompt = "You are a helpful assistant that extracts factual information from text. You analyze text carefully and identify individual, atomic facts."
        
        print(f"Text to analyze: {text}\n")
        print("Extracting facts... (streaming response)")
        print("-" * 40)
        
        # Track timing
        start_time = time.time()
        
        # Call the API with streaming
        stream_iterator = backend.call(
            model="llama3:8b",
            prompt=prompt,
            temperature=0.7,
            max_tokens=1024,
            system_prompt=system_prompt,
            stream=True
        )
        
        # Print the streaming response
        fact_count = 0
        output_text = ""
        for chunk in stream_iterator:
            chunk_str = str(chunk)
            output_text += chunk_str
            print(chunk_str, end="", flush=True)
        
        # Count facts (rough estimate)
        fact_count = output_text.count("1.") or output_text.count("\n1 ") or 1
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        print("\n" + "=" * 80)
        print(f"Response time: {elapsed_time:.2f} seconds")
        print(f"Approximate fact count: {fact_count}")
        print("=" * 80)
        
    except ValueError as e:
        print(f"ERROR: {str(e)}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Test Replicate streaming with Llama 3")
    parser.add_argument("--direct", action="store_true", help="Test direct streaming with Replicate API")
    parser.add_argument("--backend", action="store_true", help="Test streaming with our backend class")
    parser.add_argument("--facts", action="store_true", help="Test fact extraction with streaming")
    
    args = parser.parse_args()
    
    # Track total timing
    start_time = time.time()
    
    if args.direct:
        test_streaming_direct()
    elif args.backend:
        test_streaming_with_backend()
    elif args.facts:
        test_fact_extraction_prompt()
    else:
        # Run all tests by default
        test_streaming_direct()
        print("\n")
        test_streaming_with_backend()
        print("\n")
        test_fact_extraction_prompt()
    
    # Calculate total elapsed time
    total_time = time.time() - start_time
    
    # If we ran multiple tests, show a summary
    if not (args.direct or args.backend or args.facts):
        print("\n" + "=" * 80)
        print(f"All tests completed in {total_time:.2f} seconds")
        print("Cost information is being tracked by the ReplicateBackend and will be")
        print("summarized at the end of the session and logged to CSV.")
        print("=" * 80)

if __name__ == "__main__":
    main() 
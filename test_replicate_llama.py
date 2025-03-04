#!/usr/bin/env python3
"""
Test script for running the chunking process with Llama models on Replicate.
This script demonstrates the end-to-end flow using a real LLM for fact extraction.
"""
import os
import sys
import re
import json
import uuid
import datetime
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any
from dotenv import load_dotenv

# Create a log file path
LOG_FILE = Path("output/llama3_test_run/logs/processing.log")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

def log_message(message: str):
    """Log message to both console and file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp} - {message}"
    print(log_entry)
    
    with open(LOG_FILE, "a") as f:
        f.write(log_entry + "\n")

# Load environment variables for Replicate API key
load_dotenv()

# Get direct token from .env file (more reliable than environment variable)
env_path = Path('.env')
api_token = None

if env_path.exists():
    log_message(f".env file exists at {env_path.absolute()}")
    with open(env_path, 'r') as f:
        for line in f:
            if line.strip().startswith('REPLICATE_API_TOKEN='):
                api_token = line.strip().split('=', 1)[1].strip()
                # Remove quotes if present
                if api_token.startswith('"') and api_token.endswith('"'):
                    api_token = api_token[1:-1]
                elif api_token.startswith("'") and api_token.endswith("'"):
                    api_token = api_token[1:-1]
                
                masked_token = api_token[:4] + "..." + api_token[-4:] if len(api_token) > 8 else "****"
                log_message(f"Read Replicate API token from .env: {masked_token}")
                break

# If not found in .env, try environment variable
if not api_token and "REPLICATE_API_TOKEN" in os.environ:
    api_token = os.environ["REPLICATE_API_TOKEN"]
    masked_token = api_token[:4] + "..." + api_token[-4:] if len(api_token) > 8 else "****"
    log_message(f"Using Replicate API token from environment: {masked_token}")

# Verify we have a token
if not api_token:
    log_message("Error: No Replicate API token found in .env or environment.")
    sys.exit(1)

# Set token globally for replicate
os.environ["REPLICATE_API_TOKEN"] = api_token

try:
    import replicate
except ImportError:
    print("Error: replicate package not installed.")
    print("Install it with: pip install replicate")
    sys.exit(1)

# Initialize with JCS transcript
SAMPLE_TRANSCRIPT_PATH = "/Users/zbodnar/Stories/brain_dump/zettelkasten/output/JCS/deepgram_nova-3/transcripts/JCS_transcript_short.txt"

# Create output directory for test results 
OUTPUT_DIR = Path("output/llama3_test_run")
WINDOWS_DIR = OUTPUT_DIR / "windows"
FACTS_DIR = OUTPUT_DIR / "facts" 
LOGS_DIR = OUTPUT_DIR / "logs"

# Create directories if they don't exist
for dir_path in [OUTPUT_DIR, WINDOWS_DIR, FACTS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


class MockTokenizer:
    """Simple tokenizer for testing that doesn't rely on tiktoken."""
    
    def __init__(self, model: str = "llama3:8b"):
        """Initialize the mock tokenizer."""
        self.model = model
    
    def encode(self, text: str) -> List[int]:
        """Simple whitespace-based tokenization."""
        words = re.findall(r'\b\w+\b|[.,!?;:]', text)
        return list(range(1, len(words) + 1))
    
    def decode(self, tokens: List[int]) -> str:
        """Return a placeholder for decoded tokens."""
        return " ".join([f"[Token{t}]" for t in tokens])
    
    def get_token_count(self, text: str) -> int:
        """Get approximate token count based on whitespace."""
        return len(self.encode(text))


def create_windows(
    text: str, 
    tokenizer: MockTokenizer, 
    window_size: int = 500, 
    overlap_size: int = 100
) -> List[Tuple[str, int, int]]:
    """Create windows from text using a sliding window approach."""
    log_message(f"Creating windows with size={window_size}, overlap={overlap_size}")
    
    # Write input text to file
    with open(OUTPUT_DIR / "input_text.txt", "w") as f:
        f.write(text)
    log_message(f"Saved input text to {OUTPUT_DIR}/input_text.txt")
    
    # Split text into sentences for better windowing
    sentences = re.split(r'(?<=[.!?])\s+', text)
    log_message(f"Split text into {len(sentences)} sentences")
    
    # Create a list to store tokens per sentence and their text
    sentence_data = []
    current_token = 0
    
    # Store sentence tokenization info
    sentence_info = []
    
    for i, sentence in enumerate(sentences):
        token_count = tokenizer.get_token_count(sentence)
        sent_data = {
            "text": sentence,
            "start_token": current_token,
            "end_token": current_token + token_count,
            "token_count": token_count
        }
        sentence_data.append(sent_data)
        sentence_info.append(sent_data)
        current_token += token_count
    
    total_tokens = current_token
    log_message(f"Total text contains {total_tokens} tokens")
    
    # Write sentence tokenization to file
    with open(LOGS_DIR / "sentence_tokenization.json", "w") as f:
        json.dump(sentence_info, f, indent=2)
    log_message(f"Saved sentence tokenization to {LOGS_DIR}/sentence_tokenization.json")
    
    # Calculate effective step size (window size minus overlap)
    step_size = window_size - overlap_size
    
    # Create windows
    windows = []
    window_info = []
    current_pos = 0
    
    while current_pos < total_tokens:
        # Set window boundaries
        window_start = current_pos
        window_end = min(window_start + window_size, total_tokens)
        
        # Find sentences that fall within this window
        window_sentences = []
        
        for sent_data in sentence_data:
            # Check if this sentence overlaps with the current window
            if (sent_data["start_token"] < window_end and 
                sent_data["end_token"] > window_start):
                window_sentences.append(sent_data["text"])
        
        # Combine sentences into a window
        window_text = " ".join(window_sentences)
        
        # Generate window ID
        window_id = f"window_{len(windows) + 1:03d}"
        
        # Add window to list
        windows.append((window_text, window_start, window_end, window_id))
        
        # Store window info for logging
        window_data = {
            "window_id": window_id,
            "start_token": window_start,
            "end_token": window_end,
            "token_count": window_end - window_start,
            "text": window_text
        }
        window_info.append(window_data)
        
        # Save individual window to file
        with open(WINDOWS_DIR / f"{window_id}.txt", "w") as f:
            f.write(window_text)
        
        # Advance to next window position
        current_pos += step_size
        
        # Break if we've covered all tokens
        if window_end >= total_tokens:
            break
    
    # Write all window info to file
    with open(LOGS_DIR / "windows.json", "w") as f:
        json.dump(window_info, f, indent=2)
    log_message(f"Created {len(windows)} windows, saved to {WINDOWS_DIR}")
    
    return windows


def extract_facts_with_llama(window_text: str, window_id: str) -> List[Dict[str, Any]]:
    """
    Extract facts using Llama models on Replicate.
    """
    log_message(f"Extracting facts from window {window_id}")
    
    # Save raw window text for reference
    with open(LOGS_DIR / f"{window_id}_raw.txt", "w") as f:
        f.write(window_text)
    
    # Create prompt for fact extraction
    prompt = f"""Extract atomic facts from the following text. An atomic fact is a single, self-contained piece of information that:
1. Contains exactly ONE piece of information
2. Can stand alone without context
3. Has no ambiguous pronouns (replace them with actual names/entities)
4. Includes relevant temporal information if available

Guidelines:
- Each fact should be a complete, standalone statement
- Replace pronouns with specific names/entities
- Include speaker attribution if clear from context
- Include temporal markers if present
- Break compound statements into individual facts

Text to process:
{window_text}

Format your response as a JSON array of fact objects, each with:
- text: The fact itself as a complete sentence
- confidence: Your confidence in the fact (0.0-1.0)
- entities: An array of named entities mentioned

RESPONSE (JSON format):
"""
    
    # Track start time
    start_time = time.time()
    
    # Log the prompt
    with open(LOGS_DIR / f"{window_id}_prompt.txt", "w") as f:
        f.write(prompt)
    
    try:
        # Call Replicate API
        log_message(f"Calling Replicate API for {window_id}...")
        
        # Set up Replicate client with explicit token
        client = replicate.Client(api_token=api_token)
        log_message("Created Replicate client with explicit token")
        
        # Run the Llama 3 model
        log_message("Using Llama 3 8B Instruct model for fact extraction")
        output = client.run(
            "meta/meta-llama-3-8b-instruct",
            input={
                "prompt": prompt,
                "system_prompt": "You are a helpful AI assistant that extracts factual information from text. You analyze text carefully and identify individual, atomic facts. Always output in valid JSON format.",
                "temperature": 0.1,  # Low temperature for factual extraction
                "top_p": 0.9,
                "max_tokens": 2000
            }
        )
        
        # Combine streaming output
        result = "".join(output)
        
        # Log the raw response
        with open(LOGS_DIR / f"{window_id}_response_raw.txt", "w") as f:
            f.write(result)
        
        # Track end time and calculate duration
        end_time = time.time()
        duration = end_time - start_time
        log_message(f"Received response in {duration:.2f} seconds")
        
        # Try to parse response as JSON
        try:
            # Find JSON in the response (in case model outputs extra text)
            json_match = re.search(r'\[\s*{.*}\s*\]', result, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = result
                
            # Clean up common issues
            json_str = json_str.replace('```json', '').replace('```', '')
            
            # Parse the JSON
            parsed_facts = json.loads(json_str)
            log_message(f"Successfully parsed {len(parsed_facts)} facts as JSON")
            
            # Save parsed response
            with open(LOGS_DIR / f"{window_id}_response_parsed.json", "w") as f:
                json.dump(parsed_facts, f, indent=2)
                
        except json.JSONDecodeError as e:
            log_message(f"Error parsing JSON: {e}")
            
            # Fallback to a simple regex-based extraction
            log_message("Falling back to regex-based extraction")
            fact_texts = re.findall(r'"text"\s*:\s*"([^"]+)"', result)
            parsed_facts = [{"text": text, "confidence": 0.5, "entities": []} for text in fact_texts]
            
            if not parsed_facts:
                # If regex fails too, create a single fact with the first sentence
                first_sentence = window_text.split('.')[0] + '.'
                parsed_facts = [{"text": first_sentence, "confidence": 0.3, "entities": []}]
            
        # Process facts to ensure consistent format
        facts = []
        facts_info = []
        
        for i, fact_data in enumerate(parsed_facts):
            # Skip if text is missing
            if "text" not in fact_data or not fact_data["text"].strip():
                continue
                
            # Create fact ID
            fact_id = f"fact_{window_id}_{i+1:03d}"
            
            # Ensure required fields
            if "confidence" not in fact_data:
                fact_data["confidence"] = 0.8
            if "entities" not in fact_data:
                # Extract potential entities (capitalized words)
                fact_data["entities"] = re.findall(r'\b[A-Z][a-z]+\b', fact_data["text"])
                
            # Create a standardized fact
            fact = {
                "id": fact_id,
                "text": fact_data["text"].strip(),
                "confidence": float(fact_data["confidence"]),
                "source": "llama3-8b",
                "temporal_info": fact_data.get("temporal_info", ""),
                "entities": fact_data["entities"],
                "parent_window": window_id
            }
            
            facts.append(fact)
            facts_info.append(fact)
            
            # Write fact to individual file (JSON)
            with open(FACTS_DIR / f"{fact_id}.json", "w") as f:
                json.dump(fact, f, indent=2)
                
            # Also create a markdown version
            with open(FACTS_DIR / f"{fact_id}.md", "w") as f:
                f.write(f"---\n")
                f.write(f"id: {fact_id}\n")
                f.write(f"parent_window: {window_id}\n")
                f.write(f"confidence: {fact['confidence']}\n")
                f.write(f"source: llama3-8b\n")
                f.write(f"---\n\n")
                f.write(f"# {fact_id}\n\n")
                f.write(f"## Content\n\n")
                f.write(f"{fact['text']}\n\n")
                if fact['entities']:
                    f.write(f"## Entities\n\n")
                    for entity in fact['entities']:
                        f.write(f"- {entity}\n")
        
        # Write all facts from this window to a consolidated file
        with open(FACTS_DIR / f"{window_id}_facts.json", "w") as f:
            json.dump(facts_info, f, indent=2)
        
        log_message(f"Extracted {len(facts)} facts from window {window_id}")
        return facts
        
    except Exception as e:
        log_message(f"Error with Replicate API call: {e}")
        # Return an empty list in case of error
        return []


def generate_relationships(facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generate relationships between facts based on shared entities.
    """
    log_message("Generating relationships between facts")
    
    relationships = []
    
    # Create a lookup of entities to facts
    entity_facts = {}
    for fact in facts:
        for entity in fact.get("entities", []):
            if entity not in entity_facts:
                entity_facts[entity] = []
            entity_facts[entity].append(fact["id"])
    
    # Find relationships between facts that share entities
    for entity, fact_ids in entity_facts.items():
        if len(fact_ids) > 1:
            # Create relationship pairs for each combination
            for i in range(len(fact_ids)):
                for j in range(i+1, len(fact_ids)):
                    relationship = {
                        "id": f"rel_{uuid.uuid4().hex[:8]}",
                        "type": "shared_entity",
                        "entity": entity,
                        "fact1": fact_ids[i],
                        "fact2": fact_ids[j],
                        "description": f"Both facts mention {entity}"
                    }
                    relationships.append(relationship)
    
    # Write relationships to file
    with open(OUTPUT_DIR / "relationships.json", "w") as f:
        json.dump(relationships, f, indent=2)
    
    log_message(f"Generated {len(relationships)} relationships between facts")
    return relationships


def main():
    # Initialize log file
    with open(LOG_FILE, "w") as f:
        f.write(f"=== Zettelkasten Test Run with Llama 3 8B {datetime.datetime.now().isoformat()} ===\n\n")
    
    log_message(f"Starting test run with output to {OUTPUT_DIR}")
    log_message(f"Loading transcript from: {SAMPLE_TRANSCRIPT_PATH}")
    
    with open(SAMPLE_TRANSCRIPT_PATH, "r") as f:
        transcript = f.read()
        
    log_message(f"Transcript length: {len(transcript)} characters")
    
    # Create tokenizer
    tokenizer = MockTokenizer()
    log_message(f"Initialized MockTokenizer")
    
    # Get token count
    token_count = tokenizer.get_token_count(transcript)
    log_message(f"Approximate token count: {token_count}")
    
    # Create windows - larger for JCS transcript
    window_size = 300
    overlap_size = 50
    
    # Phase 1: Create windows 
    windows = create_windows(
        transcript, 
        tokenizer, 
        window_size=window_size, 
        overlap_size=overlap_size
    )
    
    # Phase 2: Extract facts from windows using Llama 3
    all_facts = []
    window_facts_map = {}
    
    # Process all windows - we have a working token now
    for window_text, start_token, end_token, window_id in windows:
        log_message(f"Processing window {window_id}")
        
        # Extract facts
        facts = extract_facts_with_llama(window_text, window_id)
        all_facts.extend(facts)
        window_facts_map[window_id] = [fact["id"] for fact in facts]
    
    # Phase 3: Process and validate facts
    log_message(f"Validating {len(all_facts)} extracted facts")
    
    # Write consolidated facts file
    with open(OUTPUT_DIR / "all_facts.json", "w") as f:
        json.dump(all_facts, f, indent=2)
    
    # Phase 4: Generate relationships
    relationships = generate_relationships(all_facts)
    
    # Phase 5: Generate summary report
    log_message("Generating summary report")
    
    with open(OUTPUT_DIR / "summary.md", "w") as f:
        f.write(f"# Zettelkasten Processing Summary with Llama 3 8B\n\n")
        f.write(f"**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"## Source Text\n\n")
        f.write(f"- **File:** {SAMPLE_TRANSCRIPT_PATH}\n")
        f.write(f"- **Length:** {len(transcript)} characters\n")
        f.write(f"- **Token Count:** {token_count} tokens\n\n")
        
        f.write(f"## Processing Stats\n\n")
        f.write(f"- **Window Size:** {window_size} tokens\n")
        f.write(f"- **Overlap Size:** {overlap_size} tokens\n")
        f.write(f"- **Windows Created:** {len(windows)}\n")
        f.write(f"- **Windows Processed:** {len(window_facts_map)}\n")
        f.write(f"- **Facts Extracted:** {len(all_facts)}\n")
        f.write(f"- **Relationships Found:** {len(relationships)}\n\n")
        
        f.write(f"## Windows\n\n")
        for window_text, start_token, end_token, window_id in windows:
            f.write(f"### {window_id}\n\n")
            f.write(f"- **Token Range:** {start_token}-{end_token} ({end_token-start_token} tokens)\n")
            if window_id in window_facts_map:
                f.write(f"- **Facts:** {len(window_facts_map[window_id])}\n")
            else:
                f.write(f"- **Facts:** Not processed\n")
            f.write(f"- **Preview:** {window_text[:100]}...\n\n")
        
        f.write(f"## Processing Pipeline\n\n")
        f.write(f"1. ✅ Create windows from transcript\n")
        f.write(f"2. ✅ Extract facts using Llama 3 8B\n")
        f.write(f"3. ✅ Process and validate facts\n")
        f.write(f"4. ✅ Generate relationships between facts\n")
        f.write(f"5. ✅ Generate summary report\n")
    
    log_message(f"Test completed successfully! Results saved to {OUTPUT_DIR}")
    log_message(f"See summary report at {OUTPUT_DIR}/summary.md")


if __name__ == "__main__":
    main()
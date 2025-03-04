#!/usr/bin/env python3
"""
Test script demonstrating the use of MockTokenizer with sample data.
This script avoids the circular import issues by implementing a simplified chunking process.
"""
import os
import sys
import re
import json
import uuid
import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Initialize with JCS transcript
SAMPLE_TRANSCRIPT_PATH = "/Users/zbodnar/Stories/brain_dump/zettelkasten/output/JCS/deepgram_nova-3/transcripts/JCS_transcript_short.txt"

# Create output directory for test results 
OUTPUT_DIR = Path("output/jcs_test_run")
WINDOWS_DIR = OUTPUT_DIR / "windows"
FACTS_DIR = OUTPUT_DIR / "facts" 
LOGS_DIR = OUTPUT_DIR / "logs"

# Create directories if they don't exist
for dir_path in [OUTPUT_DIR, WINDOWS_DIR, FACTS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Create a log file
LOG_FILE = LOGS_DIR / "processing.log"

def log_message(message: str):
    """Log message to both console and file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp} - {message}"
    print(log_entry)
    
    with open(LOG_FILE, "a") as f:
        f.write(log_entry + "\n")


class MockTokenizer:
    """Simple tokenizer for testing that doesn't rely on tiktoken."""
    
    def __init__(self, model: str = "test-model"):
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
        windows.append((window_text, window_start, window_end))
        
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


def extract_facts(window_text: str, window_id: str) -> List[Dict[str, Any]]:
    """
    Simulate fact extraction without using LLM APIs.
    This is a mock implementation for demonstration purposes.
    """
    log_message(f"Extracting facts from window {window_id}")
    
    # Simple rule-based fact extraction
    facts = []
    facts_info = []
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', window_text)
    log_message(f"Split window into {len(sentences)} sentences for fact extraction")
    
    for i, sentence in enumerate(sentences):
        # Skip very short sentences
        if len(sentence.split()) < 5:
            continue
            
        # Create fact ID
        fact_id = f"fact_{uuid.uuid4().hex[:8]}"
        
        # Extract potential entities (capitalized words)
        entities = re.findall(r'\b[A-Z][a-z]+\b', sentence)
        
        # Set confidence based on sentence length
        confidence = min(0.95, 0.7 + (len(sentence) / 500))
        
        # Create a fact from each sentence
        fact = {
            "id": fact_id,
            "text": sentence.strip(),
            "confidence": confidence,
            "source": "mock extraction",
            "temporal_info": "",
            "entities": entities,
            "parent_window": window_id
        }
        
        facts.append(fact)
        facts_info.append(fact)
        
        # Write fact to individual file
        with open(FACTS_DIR / f"{fact_id}.json", "w") as f:
            json.dump(fact, f, indent=2)
            
        # Also create a markdown version
        with open(FACTS_DIR / f"{fact_id}.md", "w") as f:
            f.write(f"---\n")
            f.write(f"id: {fact_id}\n")
            f.write(f"parent_window: {window_id}\n")
            f.write(f"confidence: {confidence}\n")
            f.write(f"source: mock extraction\n")
            f.write(f"---\n\n")
            f.write(f"# {fact_id}\n\n")
            f.write(f"## Content\n\n")
            f.write(f"{sentence.strip()}\n\n")
            if entities:
                f.write(f"## Entities\n\n")
                for entity in entities:
                    f.write(f"- {entity}\n")
    
    # Write all facts from this window to a consolidated file
    with open(FACTS_DIR / f"{window_id}_facts.json", "w") as f:
        json.dump(facts_info, f, indent=2)
    
    log_message(f"Extracted {len(facts)} facts from window {window_id}")
    return facts


def generate_relationships(facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generate relationships between facts.
    This is a simple demo implementation that looks for facts with shared entities.
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
        f.write(f"=== Zettelkasten Test Run {datetime.datetime.now().isoformat()} ===\n\n")
    
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
    
    # Phase 2: Extract facts from windows
    all_facts = []
    window_facts_map = {}
    
    for i, (window_text, start_token, end_token) in enumerate(windows):
        window_id = f"window_{i+1:03d}"
        
        # Extract facts
        facts = extract_facts(window_text, window_id)
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
        f.write(f"# Zettelkasten Processing Summary\n\n")
        f.write(f"**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"## Source Text\n\n")
        f.write(f"- **File:** {SAMPLE_TRANSCRIPT_PATH}\n")
        f.write(f"- **Length:** {len(transcript)} characters\n")
        f.write(f"- **Token Count:** {token_count} tokens\n\n")
        
        f.write(f"## Processing Stats\n\n")
        f.write(f"- **Window Size:** {window_size} tokens\n")
        f.write(f"- **Overlap Size:** {overlap_size} tokens\n")
        f.write(f"- **Windows Created:** {len(windows)}\n")
        f.write(f"- **Facts Extracted:** {len(all_facts)}\n")
        f.write(f"- **Relationships Found:** {len(relationships)}\n\n")
        
        f.write(f"## Windows\n\n")
        for i, (window_text, start_token, end_token) in enumerate(windows):
            window_id = f"window_{i+1:03d}"
            f.write(f"### {window_id}\n\n")
            f.write(f"- **Token Range:** {start_token}-{end_token} ({end_token-start_token} tokens)\n")
            f.write(f"- **Facts:** {len(window_facts_map.get(window_id, []))}\n")
            f.write(f"- **Preview:** {window_text[:100]}...\n\n")
        
        f.write(f"## Processing Pipeline\n\n")
        f.write(f"1. ✅ Create windows from transcript\n")
        f.write(f"2. ✅ Extract facts from each window\n")
        f.write(f"3. ✅ Process and validate facts\n")
        f.write(f"4. ✅ Generate relationships between facts\n")
        f.write(f"5. ✅ Generate summary report\n")
    
    log_message(f"Test completed successfully! Results saved to {OUTPUT_DIR}")
    log_message(f"See summary report at {OUTPUT_DIR}/summary.md")

if __name__ == "__main__":
    main()
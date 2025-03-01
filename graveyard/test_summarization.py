import os
from graveyard.zettel import generate_summary_and_topics
from pathlib import Path
import yaml
from datetime import datetime
import pytz
from tqdm import tqdm

def get_timestamp():
    """Get current timestamp in local timezone"""
    return datetime.now(pytz.timezone('America/New_York')).strftime('%Y-%m-%d %H:%M:%S %Z')

def log_event(log_file, message, level="INFO"):
    """Log an event with timestamp"""
    timestamp = get_timestamp()
    log_entry = f"[{timestamp}] {level}: {message}\n"
    with open(log_file, "a") as f:
        f.write(log_entry)
    print(log_entry.strip())

def process_summaries():
    # Setup paths
    chunks_dir = Path("chunks")
    log_file = Path("transcripts/JCS_base_summarization.log")
    
    try:
        log_event(log_file, "Starting summarization process")
        
        # Get all markdown files in chunks directory
        chunk_files = list(chunks_dir.glob("*.md"))
        log_event(log_file, f"Found {len(chunk_files)} chunk files to process")
        
        # Process each chunk
        with tqdm(total=len(chunk_files), desc="Generating summaries") as pbar:
            for chunk_file in chunk_files:
                log_event(log_file, f"Processing {chunk_file.name}")
                
                # Read the markdown file
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Split YAML front matter from content
                try:
                    _, yaml_text, chunk_text = content.split('---', 2)
                    metadata = yaml.safe_load(yaml_text)
                except Exception as e:
                    log_event(log_file, f"Error parsing {chunk_file.name}: {str(e)}", "ERROR")
                    continue
                
                # Generate summary and topics
                try:
                    summary, topics = generate_summary_and_topics(chunk_text.strip())
                    log_event(log_file, f"Generated summary and {len(topics)} topics")
                    
                    # Update metadata
                    metadata['summary'] = summary
                    metadata['topics'] = topics
                    
                    # Write updated markdown file
                    updated_content = f"""---
{yaml.dump(metadata, sort_keys=False)}---
{chunk_text}"""
                    
                    with open(chunk_file, 'w', encoding='utf-8') as f:
                        f.write(updated_content)
                    
                    log_event(log_file, f"Updated {chunk_file.name}")
                except Exception as e:
                    log_event(log_file, f"Error processing {chunk_file.name}: {str(e)}", "ERROR")
                    continue
                
                pbar.update(1)
        
        log_event(log_file, "Summarization process completed successfully")
        
    except Exception as e:
        error_msg = f"Summarization process failed: {str(e)}"
        log_event(log_file, error_msg, "ERROR")
        raise Exception(error_msg)

if __name__ == "__main__":
    process_summaries() 
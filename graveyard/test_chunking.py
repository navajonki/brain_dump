import os
from graveyard.zettel import chunk_text as split_text
from graveyard.zettel import create_markdown_chunk
import uuid
from pathlib import Path
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

def process_chunks():
    # Setup input/output paths
    transcript_file = "transcripts/JCS_base_transcript.txt"
    chunks_dir = Path("chunks")
    chunks_dir.mkdir(exist_ok=True)
    
    # Create log file
    log_file = Path("transcripts/JCS_base_chunking.log")
    
    try:
        log_event(log_file, f"Starting chunking process for {transcript_file}")
        
        # Read transcript
        log_event(log_file, "Reading transcript file...")
        with open(transcript_file, 'r') as f:
            transcript_text = f.read()
        log_event(log_file, f"Read {len(transcript_text)} characters from transcript")
        
        # Chunk the transcript
        log_event(log_file, "Chunking transcript...")
        chunks = split_text(transcript_text, max_tokens=300)
        log_event(log_file, f"Created {len(chunks)} chunks")
        
        # Create markdown files for each chunk
        log_event(log_file, "Creating markdown files...")
        with tqdm(total=len(chunks), desc="Creating chunk files") as pbar:
            for i, chunk_content in enumerate(chunks):
                chunk_id = str(uuid.uuid4())
                metadata = {
                    "id": chunk_id,
                    "source_file": "JCS.mp3",
                    "timestamp_range": f"{i}-{i+1}",  # Basic timestamp for now
                    "session_id": "jcs_analysis",
                    "summary": "",  # We'll add summaries later
                    "topics": [],   # We'll add topics later
                    "related_chunks": []
                }
                
                # Create markdown file
                create_markdown_chunk(
                    chunk_id=chunk_id,
                    text=chunk_content,
                    metadata=metadata,
                    output_dir="chunks"
                )
                
                log_event(log_file, f"Created chunk {i+1}/{len(chunks)}: chunks/{chunk_id}.md")
                pbar.update(1)
        
        log_event(log_file, "Chunking process completed successfully")
        
    except Exception as e:
        error_msg = f"Chunking process failed: {str(e)}"
        log_event(log_file, error_msg, "ERROR")
        raise Exception(error_msg)

if __name__ == "__main__":
    process_chunks() 
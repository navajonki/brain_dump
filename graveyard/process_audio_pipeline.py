import os
from pathlib import Path
from datetime import datetime
import pytz
from tqdm import tqdm
import yaml
from graveyard.zettel import (
    transcribe_audio,
    chunk_text as split_text,
    create_markdown_chunk,
    generate_summary_and_topics
)
from config import config

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

def setup_output_directories(audio_file: str, model_name: str):
    """Create output directory structure for a given audio file"""
    base_name = Path(audio_file).stem
    
    # Create main output directory using config
    output_dir = Path(config.OUTPUT_DIR) / base_name
    
    # Create subdirectories using config paths
    dirs = {
        'root': output_dir,
        'transcripts': output_dir / config.TRANSCRIPTS_DIR,
        'chunks': output_dir / config.CHUNKS_DIR,
        'logs': output_dir / "logs",
        'topics': output_dir / config.TOPIC_SUMMARIES_DIR
    }
    
    # Create all directories
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Define output files
    files = {
        'transcript': dirs['transcripts'] / f"{base_name}_{model_name}_transcript.txt",
        'log': dirs['logs'] / f"{base_name}_{model_name}_pipeline.log"
    }
    
    return dirs, files

def process_audio_pipeline(audio_file: str, model_name: str = None):
    # Use model name from config if not specified
    model_name = model_name or config.WHISPER_MODEL
    
    # Setup directory structure
    dirs, files = setup_output_directories(audio_file, model_name)
    
    try:
        log_event(files['log'], f"Starting pipeline for {audio_file}")
        
        # Step 1: Transcription
        log_event(files['log'], "Step 1: Transcribing audio...")
        transcript = transcribe_audio(
            audio_file, 
            model_name=model_name,
            output_file=files['transcript']
        )
        log_event(files['log'], f"Transcription complete: {len(transcript)} characters")
        
        # Step 2: Chunking
        log_event(files['log'], "Step 2: Chunking transcript...")
        chunks = split_text(transcript, max_tokens=config.MAX_CHUNK_TOKENS)
        log_event(files['log'], f"Created {len(chunks)} chunks")
        
        # Create markdown files for each chunk
        log_event(files['log'], "Creating markdown files...")
        chunk_files = []
        with tqdm(total=len(chunks), desc="Creating chunk files") as pbar:
            for i, chunk_content in enumerate(chunks):
                chunk_id = f"{Path(audio_file).stem}_chunk_{i:03d}"
                metadata = {
                    "id": chunk_id,
                    "source_file": audio_file,
                    "timestamp_range": f"{i}-{i+1}",
                    "session_id": f"{Path(audio_file).stem}_{model_name}",
                    "summary": "",
                    "topics": [],
                    "related_chunks": []
                }
                
                chunk_file = dirs['chunks'] / f"{chunk_id}.md"
                create_markdown_chunk(
                    chunk_id=chunk_id,
                    text=chunk_content,
                    metadata=metadata,
                    output_dir=str(dirs['chunks'])
                )
                chunk_files.append(chunk_file)
                log_event(files['log'], f"Created chunk file: {chunk_file.name}")
                pbar.update(1)
        
        # Step 3: Summarization
        log_event(files['log'], "Step 3: Generating summaries and topics...")
        with tqdm(total=len(chunk_files), desc="Generating summaries") as pbar:
            for chunk_file in chunk_files:
                log_event(files['log'], f"Processing {chunk_file.name}")
                
                # Read the markdown file
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Split YAML front matter from content
                try:
                    _, yaml_text, chunk_text = content.split('---', 2)
                    metadata = yaml.safe_load(yaml_text)
                    
                    # Generate summary and topics
                    summary, topics = generate_summary_and_topics(chunk_text.strip())
                    log_event(files['log'], f"Generated summary and {len(topics)} topics")
                    
                    # Update metadata
                    metadata['summary'] = summary
                    metadata['topics'] = topics
                    
                    # Write updated markdown file
                    updated_content = f"""---
{yaml.dump(metadata, sort_keys=False)}---
{chunk_text}"""
                    
                    with open(chunk_file, 'w', encoding='utf-8') as f:
                        f.write(updated_content)
                    
                    log_event(files['log'], f"Updated {chunk_file.name}")
                except Exception as e:
                    log_event(files['log'], f"Error processing {chunk_file.name}: {str(e)}", "ERROR")
                    continue
                
                pbar.update(1)
        
        log_event(files['log'], "Pipeline completed successfully")
        
    except Exception as e:
        error_msg = f"Pipeline failed: {str(e)}"
        log_event(files['log'], error_msg, "ERROR")
        raise Exception(error_msg)

if __name__ == "__main__":
    audio_file = "/Users/zbodnar/python/stories/Conversations/techcrunch.mp3"
    process_audio_pipeline(audio_file)  # Uses default model from config 
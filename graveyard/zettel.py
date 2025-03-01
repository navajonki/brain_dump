import os
import uuid
import yaml
import re
import openai
import whisper
# import spacy  # Comment out
import tiktoken
import asyncio
import logging
from config import Config
from typing import List, Dict, Any, Optional, Tuple
from neo4j import GraphDatabase
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import chromadb
from dotenv import load_dotenv
import torch
from tqdm import tqdm  # Add this import at the top
from datetime import datetime, timedelta
import pytz
import json
from pathlib import Path
from config.prompts import SUMMARY_AND_TOPICS_PROMPT, TOPIC_AGGREGATION_PROMPT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load config
load_dotenv()
config = Config()
config.validate()

# Comment out spacy model load
# nlp = spacy.load("en_core_web_sm")  # Comment out

openai.api_key = os.getenv("OPENAI_API_KEY")

def clean_text_for_markdown(text: str) -> str:
    """
    Removes or replaces problematic characters for Markdown.
    """
    text = text.replace("```", "` ` `")
    return text

def num_tokens(text: str, model_name="gpt-3.5-turbo") -> int:
    """
    Count tokens in a text string using OpenAI tiktoken.
    Adjust for your model or approximate.
    """
    enc = tiktoken.encoding_for_model(model_name)
    return len(enc.encode(text))

def chunk_text(text: str, max_tokens: int = 300) -> List[str]:
    """
    Chunk text based on a token limit (roughly 200-500 range is typical).
    """
    words = text.split()
    chunks = []
    current_chunk_words = []
    current_chunk_token_count = 0

    for word in words:
        # Approx token count by word length or do a more accurate approach
        test_chunk = " ".join(current_chunk_words + [word])
        if num_tokens(test_chunk) > max_tokens:
            # Save current chunk
            chunks.append(" ".join(current_chunk_words))
            current_chunk_words = [word]
        else:
            current_chunk_words.append(word)

    # Last chunk
    if current_chunk_words:
        chunks.append(" ".join(current_chunk_words))

    return chunks

def get_timestamp():
    """Get current timestamp in local timezone"""
    return datetime.now(pytz.timezone('America/New_York')).strftime('%Y-%m-%d %H:%M:%S %Z')

def log_event(log_file, message, level="INFO"):
    """Log an event with timestamp"""
    timestamp = get_timestamp()
    log_entry = f"[{timestamp}] {level}: {message}\n"
    with open(log_file, "a") as f:
        f.write(log_entry)
    if level == "ERROR":
        logger.error(message)
    else:
        logger.info(message)

def transcribe_audio(audio_file_path: str, model_name=None, output_file=None) -> str:
    """
    Transcribe audio using OpenAI Whisper locally with optimized M2 GPU acceleration.
    Outputs transcription with timestamps and detailed logs.
    """
    model_name = model_name or config.WHISPER_MODEL
    
    # Create output paths
    audio_file_name = Path(audio_file_path).stem
    output_dir = Path(config.TRANSCRIPTS_DIR)
    output_dir.mkdir(exist_ok=True)
    
    # Use audio filename and model name as prefix for both files
    file_prefix = f"{audio_file_name}_{model_name}"
    transcript_file = output_file or (output_dir / f"{file_prefix}_transcript.txt")
    log_file = output_dir / f"{file_prefix}_log.txt"
    
    if not os.path.exists(audio_file_path):
        log_event(log_file, f"Audio file not found: {audio_file_path}", "ERROR")
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
    
    log_event(log_file, f"Starting transcription of {audio_file_path}")
    log_event(log_file, f"Using model: {model_name}")
    
    try:
        # Load model
        log_event(log_file, "Loading Whisper model...")
        model = whisper.load_model(model_name)
        log_event(log_file, f"Model {model_name} loaded successfully")
        
        # Move to MPS (M2 GPU) if available
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            try:
                log_event(log_file, "Attempting to use M2 GPU...")
                
                # Set default device to MPS
                device = torch.device("mps")
                
                # Move model components to GPU explicitly
                model.encoder = model.encoder.to(device)
                model.decoder = model.decoder.to(device)
                if hasattr(model, 'mel_filters'):
                    model.mel_filters = model.mel_filters.to(device)
                
                log_event(log_file, "Successfully initialized M2 GPU acceleration")
            except Exception as e:
                log_event(log_file, f"GPU acceleration failed, falling back to CPU: {str(e)}", "WARNING")
                model = model.to("cpu")
        else:
            log_event(log_file, "Using CPU (MPS not available)")
        
        # Get audio duration for progress calculation
        import soundfile as sf
        audio_info = sf.info(audio_file_path)
        total_duration = audio_info.duration
        log_event(log_file, f"Audio duration: {total_duration:.2f} seconds")
        
        # Transcribe with settings from config
        result = model.transcribe(
            audio_file_path,
            **config.WHISPER_SETTINGS
        )
        
        # Write transcription with timestamps incrementally
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                for segment in result["segments"]:
                    start_time = str(timedelta(seconds=int(segment["start"])))
                    end_time = str(timedelta(seconds=int(segment["end"])))
                    line = f"[{start_time} --> {end_time}] {segment['text'].strip()}\n"
                    f.write(line)
                    f.flush()  # Ensure it's written to disk
                    log_event(log_file, f"Transcribed segment: {start_time} --> {end_time}")
        
        return result["text"]
        
    except Exception as e:
        error_msg = f"Transcription failed: {str(e)}"
        log_event(log_file, error_msg, "ERROR")
        raise Exception(error_msg)

def generate_summary_and_topics(chunk_text: str) -> tuple[str, List[str]]:
    """
    Use an LLM (OpenAI) to create a short summary and a list of topics/keywords.
    """
    prompt = SUMMARY_AND_TOPICS_PROMPT.format(text=chunk_text)
    
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=config.GPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=config.GPT_TEMPERATURE
    )
    content = response.choices[0].message.content

    # Attempt to parse JSON
    try:
        import json
        data = json.loads(content)
        summary = data.get("summary", "")
        topics = data.get("topics", [])
    except:
        # Fallback if it doesn't parse well
        summary = "No summary found"
        topics = []

    return summary.strip(), topics

# Replace spacy_topics with a simpler implementation
def spacy_topics(chunk_text: str, top_n=5) -> List[str]:
    """Simplified topic extraction without spaCy"""
    # Simple word frequency-based approach
    words = chunk_text.lower().split()
    from collections import Counter
    word_freq = Counter(words)
    # Filter out common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
    keywords = [word for word, _ in word_freq.most_common(top_n * 2) 
               if word not in stop_words][:top_n]
    return keywords

def create_markdown_chunk(chunk_id: str,
                          text: str,
                          metadata: Dict[str, Any],
                          output_dir: str = None):
    """
    Creates a .md file with YAML front matter for the chunk.
    """
    output_dir = output_dir or config.CHUNKS_DIR
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{chunk_id}.md")

    # Build YAML front matter
    yaml_front_matter = {
        "id": chunk_id,
        "source_file": metadata.get("source_file", ""),
        "timestamp_range": metadata.get("timestamp_range", ""),
        "session_id": metadata.get("session_id", ""),
        "topics": metadata.get("topics", []),
        "summary": metadata.get("summary", ""),
        "related_chunks": metadata.get("related_chunks", []),
    }

    yaml_str = yaml.dump(yaml_front_matter, sort_keys=False)
    text_clean = clean_text_for_markdown(text)

    md_content = f"""---
{yaml_str}---
{text_clean}
"""

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(md_content)

def ingest_into_vector_db(chunks_data: List[Dict[str, Any]], 
                         collection_name=None, 
                         persist_dir=None):
    """
    Ingest chunk data into Chroma.
    Each chunk_data element has: {id, text, metadata}.
    """
    collection_name = collection_name or config.VECTOR_DB_SETTINGS["collection_name"]
    persist_dir = persist_dir or config.VECTOR_DB_SETTINGS["persist_dir"]
    
    client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=persist_dir
    ))

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"), 
        model_name=config.EMBEDDING_MODEL
    )

    collection = client.get_or_create_collection(
        name=collection_name, 
        embedding_function=openai_ef
    )

    documents = [cd["text"] for cd in chunks_data]
    metadatas = [cd["metadata"] for cd in chunks_data]
    ids = [cd["id"] for cd in chunks_data]

    collection.add(documents=documents, metadatas=metadatas, ids=ids)

    # Optionally persist changes
    # (Chroma automatically persists if configured)

def ingest_into_graph_db(chunks_data: List[Dict[str, Any]]):
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "test")

    driver = GraphDatabase.driver(uri, auth=(user, password))

    with driver.session() as session:
        for cd in chunks_data:
            chunk_id = cd["id"]
            text = cd["text"]
            meta = cd["metadata"]

            # Create or update the Chunk node
            session.run("""
            MERGE (c:Chunk {id: $id})
            SET c.text = $text,
                c.source_file = $source_file,
                c.timestamp_range = $timestamp_range,
                c.session_id = $session_id,
                c.summary = $summary
            """, 
            id=chunk_id,
            text=text,
            source_file=meta.get("source_file", ""),
            timestamp_range=meta.get("timestamp_range", ""),
            session_id=meta.get("session_id", ""),
            summary=meta.get("summary", "")
            )

            # For each topic, MERGE a Topic node and create relationship
            topics = meta.get("topics", [])
            for topic in topics:
                session.run("""
                MERGE (t:Topic {name: $topicName})
                MERGE (c:Chunk {id: $chunkId})
                MERGE (c)-[:HAS_TOPIC]->(t)
                """,
                topicName=topic,
                chunkId=chunk_id)
    driver.close()

def update_topic_summaries(chunks_data: List[Dict[str, Any]],
                           topic_summaries_dir: str = None):
    """
    For each topic in the chunk, update or create a summary .md file:
    1. The file name: e.g. "topic_<topicName>.md"
    2. YAML front matter with 'topic_name' and 'chunks' references
    3. Append or update a short textual summary
    """
    topic_summaries_dir = topic_summaries_dir or config.TOPIC_SUMMARIES_DIR
    os.makedirs(topic_summaries_dir, exist_ok=True)

    # Let's group new chunks by topic
    from collections import defaultdict
    topic_to_new_chunks = defaultdict(list)
    for cd in chunks_data:
        for topic in cd["metadata"].get("topics", []):
            topic_to_new_chunks[topic].append(cd)

    for topic, chunk_list in topic_to_new_chunks.items():
        # Build a file path for this topic
        # Slugify the topic name for safe filename
        topic_slug = re.sub(r"[^a-zA-Z0-9_\-]+", "_", topic.lower())
        filename = os.path.join(topic_summaries_dir, f"topic_{topic_slug}.md")

        # Load existing content if the file exists
        existing_yaml = {}
        existing_body = ""
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                content = f.read()
                match = re.search(r"---(.*?)---(.*)", content, re.DOTALL)
                if match:
                    front_matter = match.group(1).strip()
                    existing_body = match.group(2).strip()
                    existing_yaml = yaml.safe_load(front_matter) or {}

        # Update YAML front matter
        current_chunks = existing_yaml.get("chunks", [])
        for cd in chunk_list:
            # Add reference if not already present
            if cd["id"] not in [c["id"] for c in current_chunks]:
                # store chunk id + short snippet
                current_chunks.append({
                    "id": cd["id"],
                    "summary": cd["metadata"].get("summary", ""),
                    "source_file": cd["metadata"].get("source_file", "")
                })

        existing_yaml["topic_name"] = topic
        existing_yaml["chunks"] = current_chunks

        # Optionally, generate or update an aggregated summary
        # We'll do a simple placeholder approach here:
        # you could feed the chunk summaries into an LLM to produce a new combined summary
        updated_summary_text = existing_yaml.get("summary", "")
        if not updated_summary_text:
            # If there's no existing summary for this topic, generate one from the chunk summaries
            updated_summary_text = create_aggregated_summary(topic, current_chunks)
        else:
            # Could re-generate the summary each time. For simplicity, let's leave as is.
            pass
        
        existing_yaml["summary"] = updated_summary_text

        # Reconstruct new file content
        new_yaml_str = yaml.dump(existing_yaml, sort_keys=False)
        new_md = f"""---
{new_yaml_str}---
{existing_body}
"""
        # (We're not automatically rewriting the existing body in this example—just preserving it.)
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write(new_md)

def create_aggregated_summary(topic: str, chunks: List[Dict]) -> str:
    """
    Optionally call an LLM to produce an aggregated summary for a topic.
    """
    chunk_summaries = [c["summary"] for c in chunks if c["summary"]]
    joined_summaries = "\n".join(chunk_summaries)

    prompt = TOPIC_AGGREGATION_PROMPT.format(
        topic=topic,
        summaries=joined_summaries
    )
    
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=config.GPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=config.GPT_TEMPERATURE
    )
    return response.choices[0].message.content.strip()

import uuid

class AudioProcessingError(Exception):
    """Custom exception for audio processing errors"""
    pass

async def process_audio_chunk(chunk_str: str) -> Tuple[str, List[str]]:
    """Async wrapper for GPT processing"""
    try:
        return await asyncio.to_thread(generate_summary_and_topics, chunk_str)
    except Exception as e:
        logger.error(f"Error processing chunk: {str(e)}")
        raise AudioProcessingError(f"Failed to process chunk: {str(e)}")

async def process_audio_note(
    audio_file: str,
    session_id: str,
    output_dir: Optional[str] = None,
    topic_summaries_dir: Optional[str] = None,
    max_chunk_tokens: Optional[int] = None
) -> None:
    """
    Async pipeline for processing audio notes
    """
    output_dir = output_dir or config.OUTPUT_DIR
    topic_summaries_dir = topic_summaries_dir or config.TOPIC_SUMMARIES_DIR
    max_chunk_tokens = max_chunk_tokens or config.MAX_CHUNK_TOKENS

    try:
        # 1. Transcribe
        logger.info(f"Transcribing {audio_file}")
        transcript_text = transcribe_audio(audio_file, model_name=config.WHISPER_MODEL)

        # 2. Chunk
        chunk_texts = chunk_text(transcript_text, max_tokens=max_chunk_tokens)
        logger.info(f"Created {len(chunk_texts)} chunks")

        # 3. Process chunks concurrently with a progress bar
        chunks_data = []
        tasks = []
        for i, chunk_str in enumerate(chunk_texts):
            chunk_id = str(uuid.uuid4())
            task = process_audio_chunk(chunk_str)
            tasks.append((chunk_id, i, chunk_str, task))

        # Use tqdm to show progress
        for chunk_id, i, chunk_str, task in tqdm(tasks, desc="Processing chunks", unit="chunk"):
            try:
                summary, topics = await task
                
                metadata = {
                    "id": chunk_id,
                    "source_file": audio_file,
                    "timestamp_range": f"{i}-{i+1}",  # TODO: Implement proper timestamps
                    "session_id": session_id,
                    "summary": summary,
                    "topics": topics,
                    "related_chunks": []
                }

                chunks_data.append({
                    "id": chunk_id,
                    "text": chunk_str,
                    "metadata": metadata
                })

                # 4. Create Markdown (can be done in parallel)
                await asyncio.to_thread(
                    create_markdown_chunk,
                    chunk_id=chunk_id,
                    text=chunk_str,
                    metadata=metadata,
                    output_dir=output_dir
                )

            except AudioProcessingError as e:
                logger.error(f"Failed to process chunk {chunk_id}: {str(e)}")
                continue

        # 5. Batch database operations
        if chunks_data:
            try:
                await asyncio.gather(
                    asyncio.to_thread(ingest_into_vector_db, chunks_data),
                    asyncio.to_thread(ingest_into_graph_db, chunks_data)
                )
            except Exception as e:
                logger.error(f"Database ingestion failed: {str(e)}")
                raise

            # 6. Update topic summaries
            try:
                await asyncio.to_thread(update_topic_summaries, chunks_data, topic_summaries_dir)
            except Exception as e:
                logger.error(f"Topic summary update failed: {str(e)}")
                raise

        logger.info(f"Successfully processed {audio_file}. Created {len(chunks_data)} chunks.")

    except Exception as e:
        logger.error(f"Processing failed for {audio_file}: {str(e)}")
        raise

def get_audio_timestamps(audio_file: str, chunk_index: int, total_chunks: int) -> str:
    """
    Calculate actual timestamp ranges for audio chunks.
    TODO: Implement proper audio duration detection and chunking
    """
    from pydub import AudioSegment
    try:
        audio = AudioSegment.from_file(audio_file)
        total_duration = len(audio) / 1000  # Convert to seconds
        chunk_duration = total_duration / total_chunks
        start_time = chunk_duration * chunk_index
        end_time = chunk_duration * (chunk_index + 1)
        return f"{start_time:.2f}-{end_time:.2f}"
    except Exception as e:
        logger.warning(f"Could not calculate timestamps: {str(e)}")
        return f"{chunk_index}-{chunk_index+1}"

# Example usage:
if __name__ == "__main__":
    AUDIO_FILE = "example_audio.wav"
    SESSION_ID = "session_2025_02_15"
    process_audio_note(AUDIO_FILE, SESSION_ID)

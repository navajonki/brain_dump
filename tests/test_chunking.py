import argparse
import yaml
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.chunking import TextChunker
from config.chunking.chunking_config import ChunkingConfig
from utils.logging import get_logger

logger = get_logger(__name__)

def test_chunking(transcript_file: str, config_file: str = None):
    """
    Test the chunking process with a transcript file
    
    Args:
        transcript_file: Path to the transcript file
        config_file: Path to the config file (optional)
    """
    # Load config if provided
    config_data = {}
    if config_file:
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
    
    # Initialize chunker with config
    config = ChunkingConfig(**config_data)
    chunker = AtomicChunker(config)
    
    # Read transcript
    with open(transcript_file, 'r') as f:
        text = f.read()
    
    # Process text
    chunks = chunker.process(text, transcript_file)
    
    print(f"Processed {len(chunks)} chunks")
    
    logger.info(f"\nChunking complete. Check the output/chunks directory for results.")
    return chunks

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('transcript_file', help='Path to transcript file')
    parser.add_argument('--config', help='Path to config file')
    args = parser.parse_args()
    
    test_chunking(args.transcript_file, args.config) 
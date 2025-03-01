import sys
from pathlib import Path
import argparse

# Add project root to Python path
project_root = Path(__file__).parents[1]
sys.path.append(str(project_root))

from core.summarization import Summarizer
from utils.logging import get_logger

logger = get_logger(__name__)

def test_summarization(chunk_dir: str):
    """Test summarization of existing chunks"""
    logger.info(f"Testing summarization of chunks in: {chunk_dir}")
    
    # Get all chunk files
    chunk_dir = Path(chunk_dir)
    chunk_files = sorted(chunk_dir.glob("*.md"))
    logger.info(f"Found {len(chunk_files)} chunk files")
    
    # Initialize summarizer
    summarizer = Summarizer()
    
    # Process chunks
    processed = summarizer.process_chunks(chunk_files)
    
    # Print results
    for chunk in processed:
        logger.info(f"\nChunk {chunk['id']}:")
        logger.info(f"Summary: {chunk['metadata']['summary']}")
        logger.info(f"Topics: {', '.join(chunk['metadata']['topics'])}")
        logger.info("-" * 50)
    
    return processed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("chunk_dir", help="Directory containing chunk files to summarize")
    args = parser.parse_args()
    
    test_summarization(args.chunk_dir) 
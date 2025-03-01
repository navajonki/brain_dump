import sys
from pathlib import Path
import argparse
import yaml
import os

# Add project root to Python path
project_root = Path(__file__).parents[1]
sys.path.append(str(project_root))

from core.chunking_v2 import AtomicChunker
from utils.logging import get_logger
from config.chunking.chunking_config import ChunkingConfig

logger = get_logger(__name__)

def test_atomic_chunking(
    transcript_file: str,
    config_file: str,
    output_dir: str = None,
    window_size: int = None,
    overlap_size: int = None
):
    """Test atomic fact extraction from a transcript"""
    # Load config
    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Override config with command line arguments if provided
    if window_size is not None:
        config_data["window_size"] = window_size
    if overlap_size is not None:
        config_data["overlap_size"] = overlap_size
    
    # Initialize chunker with config
    config = ChunkingConfig(**config_data)
    chunker = AtomicChunker(config)
    
    # Read transcript
    with open(transcript_file, 'r') as f:
        transcript = f.read()
    
    # Process transcript
    enhanced_chunks = chunker.process(transcript, transcript_file)
    
    # Get the chunk file paths from the output manager
    chunk_files = []
    if hasattr(chunker, 'output_mgr'):
        output_dir = os.path.dirname(chunker.output_mgr.get_chunk_path(1))
        # Updated to match the new naming convention with model name prefix
        model_prefix = config.model.replace(':', '_')  # Replace colons with underscores for file system compatibility
        chunk_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) 
                      if f.endswith('.md') and f.startswith(f"{model_prefix}_chunk_")]
    
    # Print results
    print(f"\nProcessed {len(enhanced_chunks)} chunks:")
    for i, chunk in enumerate(enhanced_chunks, 1):
        chunk_id = chunk.get('chunk_id', f'chunk_{i}')
        print(f"  - {chunk_id}: {len(chunk.get('facts', []))} facts")
        
    return chunk_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract atomic facts from a transcript using a sliding window approach")
    parser.add_argument("transcript_file", help="Path to transcript file to process")
    parser.add_argument("--config", help="Path to chunking config YAML file")
    parser.add_argument("--window-size", type=int, help="Size of sliding window in tokens (default: 1000)")
    parser.add_argument("--overlap-size", type=int, help="Size of overlap between windows in tokens (default: 100)")
    args = parser.parse_args()
    
    test_atomic_chunking(
        args.transcript_file, 
        args.config,
        window_size=args.window_size,
        overlap_size=args.overlap_size
    ) 
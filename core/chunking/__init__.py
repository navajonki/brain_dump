"""
Chunking module for processing text into atomic information units.

This module provides classes for splitting text into atomic chunks based on 
semantic content rather than just token counts. It uses a sliding window approach
with multiple processing passes to ensure complete and accurate information extraction.
"""

# Import main chunking classes
from core.chunking.text_chunker import TextChunker
from core.chunking.interfaces import (
    ILLMBackend, 
    ITokenizer, 
    IOutputManager, 
    ISessionProvider
)
from core.chunking.providers import (
    DefaultTokenizer,
    DefaultOutputManager,
    DefaultSessionProvider
)
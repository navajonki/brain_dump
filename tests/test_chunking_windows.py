"""
Tests for window creation and management in the chunking process.
This includes window creation, overlap handling, and token management.
"""

import pytest
from typing import List, Tuple, Dict
import re

from core.chunking import TextChunker
from config.chunking.chunking_config import ChunkingConfig
from tests.fixtures.fixtures import (
    load_transcript, load_config, load_windows
)

class TestWindowCreation:
    """Tests for window creation in the chunking process."""
    
    def test_create_windows_short_text(self):
        """Test window creation with text smaller than one window."""
        # Arrange
        config = ChunkingConfig(**load_config("test_config_basic"))
        chunker = TextChunker(config)
        transcript = load_transcript("short")
        
        # Act
        windows = chunker._create_windows(transcript)
        
        # Assert
        assert isinstance(windows, list)
        assert len(windows) == 1
        
        window_text, start_token, end_token = windows[0]
        assert start_token == 0
        assert isinstance(window_text, str)
        assert len(window_text) > 0

    def test_create_windows_medium_text(self):
        """Test window creation with text spanning multiple windows."""
        # Arrange
        config = ChunkingConfig(**load_config("test_config_basic"))
        chunker = TextChunker(config)
        transcript = load_transcript("medium")
        
        # Act
        windows = chunker._create_windows(transcript)
        
        # Assert
        assert isinstance(windows, list)
        assert len(windows) > 1
        
        # Check first window
        window_text, start_token, end_token = windows[0]
        assert start_token == 0
        assert end_token > start_token
        
        # Check last window
        window_text, start_token, end_token = windows[-1]
        assert start_token > 0
        assert end_token > start_token

    def test_window_overlap(self):
        """Test that windows have the configured overlap."""
        # Arrange
        config_data = load_config("test_config_basic")
        config_data["window_size"] = 500
        config_data["overlap_size"] = 100
        config = ChunkingConfig(**config_data)
        chunker = TextChunker(config)
        transcript = load_transcript("medium")
        
        # Act
        windows = chunker._create_windows(transcript)
        
        # Assert
        assert len(windows) > 1
        
        # Check overlap between windows
        for i in range(len(windows) - 1):
            current_window, current_start, current_end = windows[i]
            next_window, next_start, next_end = windows[i + 1]
            
            # Next window should start before the current window ends
            assert next_start < current_end
            
            # The overlap should be approximately the configured size
            overlap_size = current_end - next_start
            # Allow for slight variations due to token boundaries
            assert abs(overlap_size - config.overlap_size) <= 20

    def test_window_boundaries_at_sentence_breaks(self):
        """Test that windows try to break at sentence boundaries when possible."""
        # Arrange
        config = ChunkingConfig(**load_config("test_config_basic"))
        chunker = TextChunker(config)
        transcript = load_transcript("medium")
        
        # Act
        windows = chunker._create_windows(transcript)
        
        # Assert
        # This is a heuristic test, as we can't guarantee sentence breaks
        # We check if most windows end with sentence-ending punctuation
        sentence_ends = 0
        paragraph_ends = 0
        
        for window_text, _, _ in windows:
            if re.search(r'[.!?]\s*$', window_text):
                sentence_ends += 1
            if window_text.endswith('\n\n'):
                paragraph_ends += 1
        
        # Majority of windows should end with sentence punctuation or paragraphs
        # This is a probabilistic test, not deterministic
        assert (sentence_ends + paragraph_ends) / len(windows) >= 0.5

    def test_window_creation_with_small_overlap(self):
        """Test window creation with a very small overlap."""
        # Arrange
        config_data = load_config("test_config_basic")
        config_data["window_size"] = 500
        config_data["overlap_size"] = 20
        config = ChunkingConfig(**config_data)
        chunker = TextChunker(config)
        transcript = load_transcript("medium")
        
        # Act
        windows = chunker._create_windows(transcript)
        
        # Assert
        assert len(windows) > 1
        
        # Check overlap between windows
        for i in range(len(windows) - 1):
            current_window, current_start, current_end = windows[i]
            next_window, next_start, next_end = windows[i + 1]
            
            # Next window should start before the current window ends
            assert next_start < current_end
            
            # The overlap should be approximately the configured size
            overlap_size = current_end - next_start
            # Allow for slight variations due to token boundaries
            assert abs(overlap_size - config.overlap_size) <= 10

    def test_window_creation_with_large_overlap(self):
        """Test window creation with a very large overlap."""
        # Arrange
        config_data = load_config("test_config_basic")
        config_data["window_size"] = 500
        config_data["overlap_size"] = 250  # 50% overlap
        config = ChunkingConfig(**config_data)
        chunker = TextChunker(config)
        transcript = load_transcript("medium")
        
        # Act
        windows = chunker._create_windows(transcript)
        
        # Assert
        assert len(windows) > 1
        
        # Check overlap between windows
        for i in range(len(windows) - 1):
            current_window, current_start, current_end = windows[i]
            next_window, next_start, next_end = windows[i + 1]
            
            # Next window should start before the current window ends
            assert next_start < current_end
            
            # The overlap should be approximately the configured size
            overlap_size = current_end - next_start
            # Allow for slight variations due to token boundaries
            assert abs(overlap_size - config.overlap_size) <= 30

    def test_window_indices_are_sequential(self):
        """Test that window token indices are sequential and non-overlapping."""
        # Arrange
        config = ChunkingConfig(**load_config("test_config_basic"))
        chunker = TextChunker(config)
        transcript = load_transcript("medium")
        
        # Act
        windows = chunker._create_windows(transcript)
        
        # Assert
        assert len(windows) > 1
        
        # Check that indices are sequential
        for i in range(len(windows) - 1):
            current_window, current_start, current_end = windows[i]
            next_window, next_start, next_end = windows[i + 1]
            
            # Next window should start after the current window starts
            assert next_start > current_start
            
            # End index should be greater than start index
            assert current_end > current_start
            assert next_end > next_start

    def test_expected_window_count(self):
        """Test that the expected number of windows is created based on text length and window size."""
        # Arrange
        config_data = load_config("test_config_basic")
        window_size = 500
        overlap_size = 100
        config_data["window_size"] = window_size
        config_data["overlap_size"] = overlap_size
        config = ChunkingConfig(**config_data)
        chunker = TextChunker(config)
        transcript = load_transcript("medium")
        
        # Calculate approximate expected window count
        # This is a rough estimation since token counts don't directly map to character counts
        approx_tokens = len(transcript) / 4  # Rough estimate: 4 chars per token
        effective_window_size = window_size - overlap_size
        expected_windows = max(1, int(approx_tokens / effective_window_size) + 1)
        
        # Act
        windows = chunker._create_windows(transcript)
        
        # Assert
        # Allow for some variation due to token vs character differences
        assert abs(len(windows) - expected_windows) <= 2

    def test_window_creation_matches_expected(self):
        """Test that window creation matches expected output for a reference input."""
        # Arrange
        config = ChunkingConfig(**load_config("test_config_basic"))
        chunker = TextChunker(config)
        transcript = load_transcript("short")
        expected_windows = load_windows("short")
        
        # Act
        windows = chunker._create_windows(transcript)
        
        # Assert
        assert len(windows) == len(expected_windows)
        
        for i, (window_text, start_token, end_token) in enumerate(windows):
            expected = expected_windows[i]
            assert window_text == expected["window_text"]
            assert start_token == expected["start_token"]
            assert end_token == expected["end_token"]
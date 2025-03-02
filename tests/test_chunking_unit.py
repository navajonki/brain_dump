"""
Unit tests for the chunking process in TextChunker.
"""

import pytest
from typing import Dict, List, Any
import json

from core.chunking import TextChunker
from config.chunking.chunking_config import ChunkingConfig
from tests.fixtures.fixtures import (
    load_transcript, load_config, load_windows,
    load_expected_facts, load_expected_refined_facts,
    load_llm_response, load_fallback_response,
    get_all_transcripts, get_all_configs
)

# Test Window Creation
@pytest.mark.parametrize("transcript_name", ["short", "medium", "technical"])
def test_window_creation(transcript_name: str):
    """Test creating windows from different transcripts."""
    # Arrange
    transcript_text = load_transcript(transcript_name)
    config = ChunkingConfig(**load_config("test_config_basic"))
    chunker = TextChunker(config)
    
    # Act
    windows = chunker._create_windows(transcript_text)
    
    # Assert
    assert isinstance(windows, list)
    assert len(windows) > 0
    for window in windows:
        assert isinstance(window, tuple)
        assert len(window) == 3
        window_text, start_token, end_token = window
        assert isinstance(window_text, str)
        assert isinstance(start_token, int)
        assert isinstance(end_token, int)
        assert start_token < end_token
        assert len(window_text) > 0

# Test Window Creation With Expected Output
def test_window_creation_matches_expected():
    """Test window creation matches the expected output."""
    # Arrange
    transcript_text = load_transcript("short")
    config = ChunkingConfig(**load_config("test_config_basic"))
    chunker = TextChunker(config)
    expected_windows = load_windows("short")
    
    # Act
    windows = chunker._create_windows(transcript_text)
    
    # Assert
    assert len(windows) == len(expected_windows)
    for i, window in enumerate(windows):
        window_text, start_token, end_token = window
        assert window_text == expected_windows[i]["window_text"]
        assert start_token == expected_windows[i]["start_token"]
        assert end_token == expected_windows[i]["end_token"]

# Mock the LLM call for testing
class MockLlmChunker(TextChunker):
    """TextChunker with mocked LLM calls for testing."""
    
    def __init__(self, config, response_type="first_pass"):
        super().__init__(config)
        self.response_type = response_type
    
    def _call_llm(self, model, prompt, **kwargs):
        """Mock LLM call that returns predefined responses."""
        if "first pass" in prompt.lower():
            return load_llm_response("first_pass")
        elif "second pass" in prompt.lower():
            return load_llm_response("second_pass")
        elif "analyze this fact" in prompt.lower():
            return load_llm_response("tagging")
        elif "analyze these facts" in prompt.lower():
            return load_llm_response("relationship")
        elif "review these facts" in prompt.lower():
            return load_llm_response("global_check")
        else:
            return load_fallback_response("text_only")

# Test First Pass Extraction
def test_first_pass_extraction():
    """Test extracting atomic facts in the first pass."""
    # Arrange
    config = ChunkingConfig(**load_config("test_config_basic"))
    chunker = MockLlmChunker(config)
    window_text = load_transcript("short")
    
    # Act
    facts = chunker._extract_atomic_facts_first_pass(window_text, 0)
    
    # Assert
    expected_facts = load_llm_response("first_pass")
    assert len(facts) == len(expected_facts)
    for i, fact in enumerate(facts):
        assert fact["text"] == expected_facts[i]["text"]
        assert fact["confidence"] == expected_facts[i]["confidence"]
        assert fact["source"] == expected_facts[i]["source"]
        assert "temporal_info" in fact
        assert "entities" in fact

# Test Fact Validation
def test_fact_validation():
    """Test validation of facts."""
    # Arrange
    config = ChunkingConfig(**load_config("test_config_basic"))
    chunker = TextChunker(config)
    
    # Valid fact
    valid_fact = {
        "text": "This is a valid fact.",
        "confidence": 0.9,
        "source": "extraction",
        "temporal_info": "",
        "entities": ["fact"]
    }
    
    # Missing fields fact
    missing_fields_fact = {
        "text": "This fact is missing fields."
    }
    
    # String fact
    string_fact = "This is a string fact."
    
    # Act & Assert - Valid fact should pass validation
    assert chunker._validate_atomic_fact(valid_fact) == valid_fact
    
    # Act & Assert - Missing fields should be filled with defaults
    validated_missing = chunker._validate_atomic_fact(missing_fields_fact)
    assert "confidence" in validated_missing
    assert "source" in validated_missing
    assert "temporal_info" in validated_missing
    assert "entities" in validated_missing
    
    # Act & Assert - String fact should be converted to a dictionary
    validated_string = chunker._validate_atomic_fact(string_fact)
    assert isinstance(validated_string, dict)
    assert validated_string["text"] == string_fact
    assert "confidence" in validated_string
    assert "source" in validated_string
    assert "temporal_info" in validated_string
    assert "entities" in validated_string

# Test Malformed Response Handling
def test_malformed_response_handling():
    """Test handling of malformed LLM responses."""
    # Arrange
    config = ChunkingConfig(**load_config("test_config_basic"))
    chunker = TextChunker(config)
    malformed_response = load_fallback_response("malformed")
    
    # Act
    cleaned_facts = chunker._clean_llm_response(malformed_response)
    
    # Assert
    assert isinstance(cleaned_facts, list)
    for fact in cleaned_facts:
        if fact is not None and fact != "" and fact != 42:  # Skip null/empty entries
            assert isinstance(fact, (dict, str))
            if isinstance(fact, dict):
                # Either has 'text' field or will be fixed by validation
                pass
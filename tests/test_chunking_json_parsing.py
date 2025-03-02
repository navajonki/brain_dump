"""
Tests for JSON parsing and handling in the chunking process.
"""

import pytest
import json
import os
from pathlib import Path

from core.chunking import TextChunker
from config.chunking.chunking_config import ChunkingConfig
from tests.fixtures.fixtures import (
    load_config,
    load_fallback_response,
    load_llm_response
)

# Basic config for testing
TEST_CONFIG = load_config("test_config_basic")


def test_clean_json_simple():
    """Test cleaning a simple well-formed JSON array."""
    # Arrange
    config = ChunkingConfig(**TEST_CONFIG)
    chunker = TextChunker(config)
    well_formed_json = load_llm_response("first_pass")
    
    # Act
    cleaned_json = chunker._clean_llm_response(well_formed_json)
    
    # Assert
    assert cleaned_json == well_formed_json
    assert isinstance(cleaned_json, list)
    assert len(cleaned_json) > 0
    for item in cleaned_json:
        assert isinstance(item, dict)
        assert "text" in item
        assert "confidence" in item


def test_clean_json_nested():
    """Test extracting facts from a nested JSON structure."""
    # Arrange
    config = ChunkingConfig(**TEST_CONFIG)
    chunker = TextChunker(config)
    nested_json = load_fallback_response("nested_json")
    
    # Act
    cleaned_json = chunker._clean_llm_response(nested_json)
    
    # Assert
    assert isinstance(cleaned_json, list)
    assert len(cleaned_json) > 0
    for item in cleaned_json:
        assert isinstance(item, dict)
        assert "text" in item
        assert "confidence" in item
    
    # Verify it correctly extracted the facts array
    assert len(cleaned_json) == len(nested_json["results"]["facts"])


def test_clean_json_malformed():
    """Test repairing a malformed JSON response."""
    # Arrange
    config = ChunkingConfig(**TEST_CONFIG)
    chunker = TextChunker(config)
    invalid_json_text = load_fallback_response("invalid_json")
    
    # Act
    cleaned_json = chunker._clean_llm_response(invalid_json_text)
    
    # Assert
    assert isinstance(cleaned_json, list)
    assert len(cleaned_json) > 0
    
    # Check if the essential data was recovered
    facts_text = [fact.get("text", "") if isinstance(fact, dict) else fact for fact in cleaned_json if fact]
    expected_texts = [
        "The Zettelkasten Method is a personal knowledge management system",
        "Niklas Luhmann developed the Zettelkasten Method",
        "Luhmann published more than 70 books and nearly 400 scholarly articles"
    ]
    
    # Check that all expected texts are found in the cleaned results
    for expected in expected_texts:
        assert any(expected in text for text in facts_text if isinstance(text, str))


def test_clean_json_code_block():
    """Test extracting JSON from a code block in a text response."""
    # Arrange
    config = ChunkingConfig(**TEST_CONFIG)
    chunker = TextChunker(config)
    code_block = load_fallback_response("code_block_json")
    
    # Act
    cleaned_json = chunker._clean_llm_response(code_block)
    
    # Assert
    assert isinstance(cleaned_json, list)
    assert len(cleaned_json) > 0
    
    # Should have extracted the JSON array from the code block
    for item in cleaned_json:
        assert isinstance(item, dict)
        assert "text" in item
        assert "confidence" in item


def test_clean_json_text_only():
    """Test handling plain text responses with no JSON structure."""
    # Arrange
    config = ChunkingConfig(**TEST_CONFIG)
    chunker = TextChunker(config)
    text_only = load_fallback_response("text_only")
    
    # Act
    cleaned_json = chunker._clean_llm_response(text_only)
    
    # Assert
    assert isinstance(cleaned_json, list)
    assert len(cleaned_json) > 0
    
    # Should have converted bullet points to facts
    for item in cleaned_json:
        if isinstance(item, str):
            # If it's a string, should start with a dash or bullet
            assert any(item.lstrip().startswith(prefix) for prefix in ["- ", "• "])
        elif isinstance(item, dict):
            # If already converted to dict, should have text field
            assert "text" in item


def test_validate_fact_complete():
    """Test validating a complete fact with all required fields."""
    # Arrange
    config = ChunkingConfig(**TEST_CONFIG)
    chunker = TextChunker(config)
    complete_fact = {
        "text": "This is a complete fact",
        "confidence": 0.95,
        "source": "extraction",
        "temporal_info": "2023",
        "entities": ["entity1", "entity2"]
    }
    
    # Act
    validated_fact = chunker._validate_atomic_fact(complete_fact)
    
    # Assert
    assert validated_fact == complete_fact


def test_validate_fact_missing_fields():
    """Test validating a fact with missing fields."""
    # Arrange
    config = ChunkingConfig(**TEST_CONFIG)
    chunker = TextChunker(config)
    incomplete_fact = {
        "text": "This fact is missing fields"
    }
    
    # Act
    validated_fact = chunker._validate_atomic_fact(incomplete_fact)
    
    # Assert
    assert validated_fact["text"] == incomplete_fact["text"]
    assert "confidence" in validated_fact
    assert "source" in validated_fact
    assert "temporal_info" in validated_fact
    assert "entities" in validated_fact
    assert isinstance(validated_fact["entities"], list)


def test_validate_string_fact():
    """Test validating a fact that's just a string."""
    # Arrange
    config = ChunkingConfig(**TEST_CONFIG)
    chunker = TextChunker(config)
    string_fact = "This is just a string fact"
    
    # Act
    validated_fact = chunker._validate_atomic_fact(string_fact)
    
    # Assert
    assert isinstance(validated_fact, dict)
    assert validated_fact["text"] == string_fact
    assert "confidence" in validated_fact
    assert "source" in validated_fact
    assert "temporal_info" in validated_fact
    assert "entities" in validated_fact


def test_validate_fact_with_entity_object():
    """Test validating a fact where entities is an object instead of an array."""
    # Arrange
    config = ChunkingConfig(**TEST_CONFIG)
    chunker = TextChunker(config)
    fact_with_entity_object = {
        "text": "This fact has an entity object",
        "confidence": 0.9,
        "source": "extraction",
        "temporal_info": "",
        "entities": {
            "people": ["Person 1", "Person 2"],
            "places": ["Place 1"],
            "organizations": [],
            "other": ["Thing 1"]
        }
    }
    
    # Act
    validated_fact = chunker._validate_atomic_fact(fact_with_entity_object)
    
    # Assert
    assert validated_fact["text"] == fact_with_entity_object["text"]
    assert isinstance(validated_fact["entities"], dict)
    assert "people" in validated_fact["entities"]
    assert "places" in validated_fact["entities"]
    assert "organizations" in validated_fact["entities"]
    assert "other" in validated_fact["entities"]


def test_validate_fact_with_confidence_string():
    """Test validating a fact where confidence is a string instead of a number."""
    # Arrange
    config = ChunkingConfig(**TEST_CONFIG)
    chunker = TextChunker(config)
    fact_with_string_confidence = {
        "text": "This fact has confidence as a string",
        "confidence": "high",
        "source": "extraction",
        "temporal_info": "",
        "entities": ["entity1"]
    }
    
    # Act
    validated_fact = chunker._validate_atomic_fact(fact_with_string_confidence)
    
    # Assert
    assert validated_fact["text"] == fact_with_string_confidence["text"]
    assert isinstance(validated_fact["confidence"], float)
    # Non-numeric confidence should be converted to a default value (1.0)
    assert validated_fact["confidence"] == 1.0


def test_validate_fact_with_invalid_entities():
    """Test validating a fact with invalid entities field (not a list or dict)."""
    # Arrange
    config = ChunkingConfig(**TEST_CONFIG)
    chunker = TextChunker(config)
    fact_with_invalid_entities = {
        "text": "This fact has invalid entities",
        "confidence": 0.9,
        "source": "extraction",
        "temporal_info": "",
        "entities": "single entity"  # String instead of list
    }
    
    # Act
    validated_fact = chunker._validate_atomic_fact(fact_with_invalid_entities)
    
    # Assert
    assert validated_fact["text"] == fact_with_invalid_entities["text"]
    assert isinstance(validated_fact["entities"], list)
    # Should have converted the string to a one-item list
    assert validated_fact["entities"] == ["single entity"]


def test_validate_fact_with_placeholder_text():
    """Test validating a fact with placeholder text that should be rejected."""
    # Arrange
    config = ChunkingConfig(**TEST_CONFIG)
    chunker = TextChunker(config)
    placeholder_facts = [
        {"text": "Fact 1: The Zettelkasten Method is a personal knowledge management system"},
        {"text": "Here are the facts from the text:"},
        {"text": "I couldn't extract any facts from this text."},
        {"text": "No clear facts were present in the provided text."}
    ]
    
    # Act & Assert
    for fact in placeholder_facts:
        validated_fact = chunker._validate_atomic_fact(fact)
        # Placeholder facts should be rejected (None returned)
        assert validated_fact is None
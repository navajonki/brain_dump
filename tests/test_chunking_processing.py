"""
Tests for the processing components of the chunking pipeline.
This includes fact extraction, tagging, second pass refinement, 
relationship analysis, and global consistency checking.
"""

import pytest
from typing import Dict, List, Any
import json

from core.chunking import TextChunker
from config.chunking.chunking_config import ChunkingConfig
from tests.fixtures.fixtures import (
    load_transcript, load_config, load_windows,
    load_expected_facts, load_expected_refined_facts,
    load_llm_response, load_fallback_response
)

# Mock the LLM call for testing
class MockLlmChunker(TextChunker):
    """TextChunker with mocked LLM calls for testing."""
    
    def __init__(self, config):
        super().__init__(config)
    
    def _call_llm(self, model, prompt, **kwargs):
        """Mock LLM call that returns predefined responses."""
        if "first pass" in prompt.lower() or "list all distinct" in prompt.lower():
            return load_llm_response("first_pass")
        elif "second pass" in prompt.lower() or "additional context" in prompt.lower():
            return load_llm_response("second_pass")
        elif "analyze this fact" in prompt.lower() or "identify 3-5 relevant tags" in prompt.lower():
            return load_llm_response("tagging")
        elif "analyze these facts and identify relationships" in prompt.lower():
            return load_llm_response("relationship")
        elif "review these facts" in prompt.lower():
            return load_llm_response("global_check")
        else:
            return load_fallback_response("text_only")


class TestFactExtraction:
    """Tests for the fact extraction process."""
    
    def test_extract_facts_first_pass(self):
        """Test extracting facts in the first pass."""
        # Arrange
        config = ChunkingConfig(**load_config("test_config_basic"))
        chunker = MockLlmChunker(config)
        transcript = load_transcript("short")
        
        # Act
        windows = chunker._create_windows(transcript)
        window_text = windows[0][0]
        facts = chunker._extract_atomic_facts_first_pass(window_text, 0)
        
        # Assert
        assert isinstance(facts, list)
        assert len(facts) > 0
        for fact in facts:
            assert isinstance(fact, dict)
            assert "text" in fact
            assert "confidence" in fact
            assert "source" in fact
            assert "temporal_info" in fact
            assert "entities" in fact
    
    def test_extract_facts_second_pass(self):
        """Test refining facts in the second pass."""
        # Arrange
        config = ChunkingConfig(**load_config("test_config_full"))
        chunker = MockLlmChunker(config)
        transcript = load_transcript("short")
        first_pass_facts = load_llm_response("first_pass")
        
        # Act
        windows = chunker._create_windows(transcript)
        window_text = windows[0][0]
        refined_facts = chunker._extract_atomic_facts_second_pass(first_pass_facts, window_text)
        
        # Assert
        assert isinstance(refined_facts, list)
        assert len(refined_facts) > 0
        
        # Refined facts should have more details
        for i, refined_fact in enumerate(refined_facts):
            if i < len(first_pass_facts):
                first_pass = first_pass_facts[i]
                # Text should be either the same or longer
                assert len(refined_fact["text"]) >= len(first_pass["text"])
                # Confidence should be at least as high
                assert refined_fact["confidence"] >= first_pass["confidence"]
                # Entities list should be the same or longer
                assert len(refined_fact["entities"]) >= len(first_pass["entities"])


class TestTagging:
    """Tests for the fact tagging process."""
    
    def test_tag_fact(self):
        """Test tagging a single fact."""
        # Arrange
        config = ChunkingConfig(**load_config("test_config_full"))
        chunker = MockLlmChunker(config)
        fact = {
            "text": "The Zettelkasten Method is a personal knowledge management system developed by German sociologist Niklas Luhmann.",
            "confidence": 0.95,
            "source": "extraction",
            "temporal_info": "",
            "entities": ["Zettelkasten Method", "Niklas Luhmann", "German sociologist"]
        }
        
        # Act
        tagged_fact = chunker._tag_fact(fact)
        
        # Assert
        assert isinstance(tagged_fact, dict)
        assert tagged_fact["text"] == fact["text"]
        assert "tags" in tagged_fact
        assert "topics" in tagged_fact
        assert "entities" in tagged_fact
        
        # Entities should now be a dictionary with categories
        assert isinstance(tagged_fact["entities"], dict)
        assert "people" in tagged_fact["entities"]
        assert "places" in tagged_fact["entities"]
        assert "organizations" in tagged_fact["entities"]
        assert "other" in tagged_fact["entities"]
        
        # Should have sentiment and importance
        assert "sentiment" in tagged_fact
        assert "importance" in tagged_fact
        assert tagged_fact["sentiment"] in ["positive", "negative", "neutral"]
        assert 1 <= tagged_fact["importance"] <= 10
    
    def test_tag_facts_in_window(self):
        """Test tagging all facts in a window."""
        # Arrange
        config = ChunkingConfig(**load_config("test_config_full"))
        chunker = MockLlmChunker(config)
        facts = load_llm_response("first_pass")
        
        # Act
        tagged_facts = chunker._tag_facts_in_window(facts)
        
        # Assert
        assert isinstance(tagged_facts, list)
        assert len(tagged_facts) == len(facts)
        
        for tagged_fact in tagged_facts:
            assert "tags" in tagged_fact
            assert "topics" in tagged_fact
            assert "sentiment" in tagged_fact
            assert "importance" in tagged_fact
            assert isinstance(tagged_fact["entities"], dict)


class TestRelationshipAnalysis:
    """Tests for relationship analysis between facts."""
    
    def test_analyze_relationships_pairwise(self):
        """Test analyzing relationships between pairs of facts."""
        # Arrange
        config = ChunkingConfig(**load_config("test_config_full"))
        chunker = MockLlmChunker(config)
        facts = load_llm_response("first_pass")
        
        # Only test first two facts for efficiency
        fact1 = facts[0]
        fact2 = facts[1]
        
        # Act
        relationship = chunker._analyze_relationship_pairwise(fact1, fact2)
        
        # Assert
        assert isinstance(relationship, dict)
        assert "relationship_type" in relationship
        assert "confidence" in relationship
        assert relationship["relationship_type"] in [
            "supports", "contradicts", "elaborates", "unrelated", 
            "precedes", "follows", "causes", "leads_to"
        ]
        assert 0 <= relationship["confidence"] <= 1
    
    def test_analyze_relationships_global(self):
        """Test analyzing relationships across all facts."""
        # Arrange
        config = ChunkingConfig(**load_config("test_config_full"))
        chunker = MockLlmChunker(config)
        facts = load_llm_response("first_pass")
        
        # Act
        relationships = chunker._analyze_relationships_global(facts)
        
        # Assert
        assert isinstance(relationships, list)
        assert len(relationships) > 0
        
        for relationship in relationships:
            assert "fact_idx" in relationship
            assert "related_fact_idx" in relationship
            assert "relationship_type" in relationship
            assert "confidence" in relationship
            assert relationship["fact_idx"] != relationship["related_fact_idx"]
            assert 0 <= relationship["confidence"] <= 1


class TestGlobalConsistencyCheck:
    """Tests for the global consistency check."""
    
    def test_global_consistency_check(self):
        """Test checking for global consistency issues."""
        # Arrange
        config = ChunkingConfig(**load_config("test_config_full"))
        chunker = MockLlmChunker(config)
        all_facts = []
        
        # Add some facts from first pass and some from second pass
        all_facts.extend(load_llm_response("first_pass"))
        all_facts.extend(load_llm_response("second_pass"))
        
        # Act
        changes = chunker._global_consistency_check(all_facts)
        
        # Assert
        assert isinstance(changes, dict)
        assert "redundancies" in changes
        assert "contradictions" in changes
        assert "timeline_issues" in changes
        
        # If there are redundancies, check their structure
        if changes["redundancies"]:
            for redundancy in changes["redundancies"]:
                assert "fact_idx" in redundancy
        
        # If there are contradictions, check their structure
        if changes["contradictions"]:
            for contradiction in changes["contradictions"]:
                assert "fact_idx" in contradiction
                assert "correction" in contradiction
                assert isinstance(contradiction["correction"], str)
        
        # If there are timeline issues, check their structure
        if changes["timeline_issues"]:
            for issue in changes["timeline_issues"]:
                assert "fact_idx" in issue
                assert "correction" in issue
                assert isinstance(issue["correction"], str)
    
    def test_apply_global_consistency_changes(self):
        """Test applying changes from global consistency check."""
        # Arrange
        config = ChunkingConfig(**load_config("test_config_full"))
        chunker = MockLlmChunker(config)
        facts = load_llm_response("second_pass")
        
        changes = {
            "redundancies": [{"fact_idx": 1}],
            "contradictions": [
                {"fact_idx": 0, "correction": "Corrected fact text."}
            ],
            "timeline_issues": []
        }
        
        # Act
        corrected_facts = chunker._apply_global_consistency_changes(facts, changes)
        
        # Assert
        assert isinstance(corrected_facts, list)
        
        # Should have one less fact (removed redundancy)
        assert len(corrected_facts) == len(facts) - 1
        
        # First fact should be corrected
        assert corrected_facts[0]["text"] == "Corrected fact text."
        
        # Other facts should remain unchanged
        for i in range(1, len(corrected_facts)):
            original_idx = i + 1  # Skip the redundant fact
            assert corrected_facts[i]["text"] == facts[original_idx]["text"]


class TestEndToEnd:
    """End-to-end tests for the chunking process."""
    
    def test_process_single_window(self):
        """Test processing a single window through the entire pipeline."""
        # Arrange
        config = ChunkingConfig(**load_config("test_config_full"))
        chunker = MockLlmChunker(config)
        transcript = load_transcript("short")
        
        # Act
        windows = chunker._create_windows(transcript)
        window_facts = chunker._process_window(windows[0], 0)
        
        # Assert
        assert isinstance(window_facts, list)
        assert len(window_facts) > 0
        
        for fact in window_facts:
            assert "text" in fact
            assert "confidence" in fact
            assert "source" in fact
            assert "temporal_info" in fact
            assert "entities" in fact
            assert "tags" in fact
            assert "topics" in fact
            assert "sentiment" in fact
            assert "importance" in fact
    
    def test_full_process(self):
        """Test the full chunking process with a short transcript."""
        # Arrange
        config = ChunkingConfig(**load_config("test_config_full"))
        chunker = MockLlmChunker(config)
        transcript = load_transcript("short")
        
        # Act
        result = chunker.process(transcript, "test_transcript")
        
        # Assert
        assert isinstance(result, list)
        assert len(result) > 0
        
        # Check structure of the result chunks
        for chunk in result:
            assert "chunk_id" in chunk
            assert "start_idx" in chunk
            assert "end_idx" in chunk
            assert "text" in chunk
            assert "facts" in chunk
            assert isinstance(chunk["facts"], list)
            
            # Check structure of facts in the chunk
            for fact in chunk["facts"]:
                assert "text" in fact
                assert "confidence" in fact
                assert "tags" in fact
                assert "topics" in fact
                assert "sentiment" in fact
                assert "importance" in fact
                assert isinstance(fact["entities"], dict)
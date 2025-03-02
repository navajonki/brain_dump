"""
Response parsing module for handling LLM responses.

This module provides classes and utilities for parsing, validating, and normalizing
responses from LLMs, with robust error handling and schema validation.
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional, Union, Type, TypeVar, Callable
from pydantic import BaseModel, Field, ValidationError, create_model
from pydantic.error_wrappers import ErrorWrapper

from utils.logging import get_logger

logger = get_logger(__name__)

# Type for schema models
T = TypeVar('T', bound=BaseModel)


class JSONParser:
    """
    Parser for JSON responses from LLMs.
    
    This class handles extraction, cleaning, and validation of JSON from LLM responses,
    with robust error handling for various formats and issues.
    """
    
    def __init__(self, schema: Optional[Type[BaseModel]] = None, logger=None):
        """
        Initialize the JSON parser.
        
        Args:
            schema: Optional Pydantic model for validating responses
            logger: Optional logger instance
        """
        self.schema = schema
        self.logger = logger or get_logger(__name__)
    
    def parse(self, response: Any, schema: Optional[Type[BaseModel]] = None) -> Union[Dict, List, Any]:
        """
        Parse a response from an LLM into a structured object.
        
        Args:
            response: The raw response from the LLM
            schema: Optional schema to use for this specific parse (overrides default)
            
        Returns:
            Parsed and normalized response object
        """
        if response is None:
            return None
        
        # Use provided schema or default
        active_schema = schema or self.schema
        
        # Extract JSON from the response
        try:
            extracted_json = self._extract_json(response)
            
            # If we have a schema, validate against it
            if active_schema:
                try:
                    validated = active_schema.parse_obj(extracted_json)
                    return validated.dict()
                except ValidationError as e:
                    self.logger.warning(f"Schema validation failed: {str(e)}")
                    # Try to extract useful data even if validation fails
                    return extracted_json
            
            return extracted_json
        
        except Exception as e:
            self.logger.error(f"Failed to parse response: {str(e)}")
            return None
    
    def _extract_json(self, response: Any) -> Union[Dict, List, Any]:
        """
        Extract JSON from a response, handling various formats.
        
        Args:
            response: The raw response to extract JSON from
            
        Returns:
            Extracted JSON object or the original response
        """
        # If already a dict or list, just return it
        if isinstance(response, (dict, list)):
            return response
        
        # If a string, try to parse as JSON
        if isinstance(response, str):
            # Try to clean and parse as JSON
            cleaned_str = self._clean_json_string(response)
            try:
                return json.loads(cleaned_str)
            except json.JSONDecodeError:
                # If that fails, look for bullets or structured text
                return self._extract_structured_content(response)
        
        # If none of the above, return as is
        return response
    
    def _clean_json_string(self, json_str: str) -> str:
        """
        Clean a string to prepare it for JSON parsing.
        
        Args:
            json_str: JSON string to clean
            
        Returns:
            Cleaned JSON string
        """
        if not json_str or not isinstance(json_str, str):
            return "{}"
        
        # Extract JSON from code blocks (common in LLM responses)
        code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', json_str)
        if code_block_match:
            json_str = code_block_match.group(1)
        
        # Remove markdown formatting and explanations
        # Common patterns in LLM responses
        json_str = re.sub(r'^.*?(\[|\{)', r'\1', json_str, flags=re.DOTALL)
        json_str = re.sub(r'(\]|\}).*$', r'\1', json_str, flags=re.DOTALL)
        
        # Fix common JSON errors
        
        # Fix trailing commas before closing brackets
        json_str = re.sub(r',\s*(\}|\])', r'\1', json_str)
        
        # Fix missing commas between objects
        json_str = re.sub(r'(\}|\])\s*(\{|\[)', r'\1,\2', json_str)
        
        # Fix unquoted keys
        json_str = re.sub(r'(\{|\,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)
        
        # Replace JS-style comments
        json_str = re.sub(r'//.*?(\n|$)', r'\1', json_str)
        json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
        
        # Fix single quotes to double quotes (but not within already double-quoted strings)
        in_string = False
        result = []
        i = 0
        while i < len(json_str):
            if json_str[i] == '"' and (i == 0 or json_str[i-1] != '\\'):
                in_string = not in_string
                result.append('"')
            elif json_str[i] == "'" and not in_string:
                result.append('"')
            else:
                result.append(json_str[i])
            i += 1
        
        json_str = ''.join(result)
        
        return json_str
    
    def _extract_structured_content(self, text: str) -> Union[List[str], List[Dict], str]:
        """
        Extract structured content from text, like bullet points or lines.
        
        Args:
            text: Text to extract content from
            
        Returns:
            Extracted content as a list or the original text
        """
        if not text or not isinstance(text, str):
            return text
        
        # Try to extract bullet points
        bullet_points = re.findall(r'(?:^|\n)[•\-\*]\s*(.*?)(?=\n[•\-\*]|\n\n|$)', text, re.DOTALL)
        if bullet_points:
            return [point.strip() for point in bullet_points if point.strip()]
        
        # Try to extract numbered points
        numbered_points = re.findall(r'(?:^|\n)\d+\.?\s*(.*?)(?=\n\d+\.?|\n\n|$)', text, re.DOTALL)
        if numbered_points:
            return [point.strip() for point in numbered_points if point.strip()]
        
        # If no structured content found, split by lines as a fallback
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if lines and len(lines) > 1:
            return lines
        
        # If all else fails, return the original text
        return text


# Schema models for different response types

class AtomicFact(BaseModel):
    """Schema for an atomic fact."""
    text: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source: str = "extraction"
    temporal_info: str = ""
    entities: Union[List[str], Dict[str, List[str]]] = []


class FactTag(BaseModel):
    """Schema for fact tagging metadata."""
    tags: List[str] = []
    topics: List[str] = []
    sentiment: str = "neutral"
    importance: int = Field(default=5, ge=1, le=10)
    entities: Optional[Union[List[str], Dict[str, List[str]]]] = None


class Relationship(BaseModel):
    """Schema for a relationship between facts."""
    fact_idx: int
    related_fact_idx: int
    relationship_type: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ConsistencyChange(BaseModel):
    """Schema for consistency changes to facts."""
    fact_idx: int
    correction: Optional[str] = None


class ConsistencyChanges(BaseModel):
    """Schema for all consistency changes."""
    redundancies: List[Dict[str, int]] = []
    contradictions: List[ConsistencyChange] = []
    timeline_issues: List[ConsistencyChange] = []


# Factory for creating parsers with specific schemas

class ResponseParserFactory:
    """Factory for creating response parsers with specific schemas."""
    
    @staticmethod
    def get_first_pass_parser() -> JSONParser:
        """Get a parser for first pass extraction responses."""
        return JSONParser(schema=create_model('FirstPassResponse', facts=(List[AtomicFact], ...)))
    
    @staticmethod
    def get_second_pass_parser() -> JSONParser:
        """Get a parser for second pass refinement responses."""
        return JSONParser(schema=create_model('SecondPassResponse', facts=(List[AtomicFact], ...)))
    
    @staticmethod
    def get_tagging_parser() -> JSONParser:
        """Get a parser for tagging responses."""
        return JSONParser(schema=FactTag)
    
    @staticmethod
    def get_batch_tagging_parser() -> JSONParser:
        """Get a parser for batch tagging responses."""
        return JSONParser(schema=create_model('BatchTaggingResponse', fact_tags=(Dict[str, FactTag], ...)))
    
    @staticmethod
    def get_relationship_parser() -> JSONParser:
        """Get a parser for relationship analysis responses."""
        return JSONParser(schema=create_model('RelationshipResponse', relationships=(List[Relationship], ...)))
    
    @staticmethod
    def get_global_check_parser() -> JSONParser:
        """Get a parser for global consistency check responses."""
        return JSONParser(schema=create_model('GlobalCheckResponse', changes=(ConsistencyChanges, ...)))
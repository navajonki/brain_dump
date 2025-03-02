"""
Main text chunking implementation with dependency injection.

This module provides the TextChunker class, which extracts atomic units of information
from text using LLMs. The class is designed with testability in mind, using dependency
injection to allow easy mocking and unit testing.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import re
import json
import time
import os
import logging
from datetime import datetime

from utils.logging import get_logger
from core.llm_backends import create_llm_backend
from config.chunking.chunking_config import ChunkingConfig

# Import interfaces
from core.chunking.interfaces import (
    ILLMBackend, 
    ITokenizer, 
    IOutputManager, 
    ISessionProvider,
    IPromptManager
)

# Import default providers
from core.chunking.providers import (
    DefaultTokenizer,
    DefaultOutputManager,
    DefaultSessionProvider,
    DefaultPromptManager
)


class TextChunker:
    """
    A chunker that extracts atomic units of information from transcripts
    using a sliding window approach with multiple passes.
    
    This is the main chunking implementation, focusing on atomic fact extraction
    rather than simple token-based splitting. It uses LLMs to identify discrete
    facts within text and preserves semantic relationships.
    """
    
    def __init__(
        self, 
        config: ChunkingConfig = None,
        llm_backend: ILLMBackend = None,
        tokenizer: ITokenizer = None,
        output_manager: IOutputManager = None,
        session_provider: ISessionProvider = None,
        prompt_manager: IPromptManager = None,
        logger = None
    ):
        """
        Initialize the chunker with configuration and dependencies.
        
        Args:
            config: Configuration object
            llm_backend: LLM backend implementation
            tokenizer: Tokenizer implementation
            output_manager: Output manager implementation
            session_provider: Session provider implementation
            prompt_manager: Prompt manager implementation
            logger: Logger instance
        """
        # Initialize configuration
        self.config = config or ChunkingConfig()
        
        # Initialize logger
        self.logger = logger or get_logger(__name__)
        
        # Initialize tokenizer (default uses tiktoken)
        self.tokenizer = tokenizer or DefaultTokenizer(self.config.model)
        
        # Initialize session provider and get session ID
        self.session_provider = session_provider or DefaultSessionProvider()
        self.session_id = self.session_provider.get_session_id()
        
        # Initialize output manager
        self.output_manager = output_manager or DefaultOutputManager(self.logger)
        self.output_manager.initialize(self.session_id, self.config)
        
        # Initialize prompt manager
        self.prompt_manager = prompt_manager or DefaultPromptManager(self.config)
        
        # Initialize LLM backend
        self.llm_backend = llm_backend or self._create_default_llm_backend()
        
        # Log initialization
        self.logger.info(f"Initialized TextChunker with session ID: {self.session_id}")
        self.logger.info(f"Using model: {self.config.model}")
    
    def _create_default_llm_backend(self) -> ILLMBackend:
        """Create and initialize the default LLM backend based on configuration."""
        # Get the backend type from config
        backend_type = self.config.llm_backend
        
        if not backend_type:
            # Default to OpenAI if not specified
            backend_type = "openai"
            self.logger.warning(f"No LLM backend specified, defaulting to {backend_type}")
        
        try:
            # Create the backend using the factory function
            backend = create_llm_backend(backend_type)
            
            # Initialize backend with config if needed
            if hasattr(backend, 'initialize'):
                backend.initialize(self.config)
                
            self.logger.info(f"Initialized {backend_type} LLM backend")
            return backend
        except Exception as e:
            self.logger.error(f"Failed to initialize {backend_type} LLM backend: {str(e)}")
            raise
    
    def process(self, text: str, source_path: str = None) -> List[Dict[str, Any]]:
        """
        Process a text into atomic chunks with metadata.
        
        Args:
            text: The text to process
            source_path: The source path or identifier (optional)
            
        Returns:
            A list of chunks, each with extracted facts and metadata
        """
        if not text:
            self.logger.warning("Empty text provided, nothing to process.")
            return []
        
        self.logger.info(f"Processing text from {source_path or 'unknown source'}")
        self.logger.info(f"Text length: {len(text)} characters")
        
        # Create windows from the text
        windows = self.create_windows(text)
        self.logger.info(f"Created {len(windows)} windows")
        
        # Process each window to extract facts
        processed_windows = []
        for i, window in enumerate(windows):
            self.logger.info(f"Processing window {i+1}/{len(windows)}")
            window_facts = self.process_window(window, i)
            
            # Create chunk data
            window_text, start_idx, end_idx = window
            chunk_id = f"chunk_{i+1:03d}"
            
            chunk = {
                "chunk_id": chunk_id,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "text": window_text,
                "facts": window_facts,
                "source": source_path,
                "processed_at": datetime.now().isoformat()
            }
            
            # Write chunk and facts to storage
            self.output_manager.write_chunk(chunk_id, chunk)
            self.output_manager.write_facts(chunk_id, window_facts)
            
            processed_windows.append(chunk)
        
        # Perform global consistency check if enabled
        if self.config.global_check_enabled and len(processed_windows) > 1:
            self.logger.info("Performing global consistency check")
            self._perform_global_consistency_check(processed_windows)
        
        self.logger.info("Processing complete")
        return processed_windows
    
    def create_windows(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Create overlapping windows from the text.
        
        This is an exposed version of the internal _create_windows method for testing.
        
        Args:
            text: The text to split into windows
            
        Returns:
            List of window tuples (text, start_token, end_token)
        """
        return self._create_windows(text)
    
    def process_window(self, window: Tuple[str, int, int], window_index: int) -> List[Dict[str, Any]]:
        """
        Process a single window to extract facts.
        
        This method oversees the complete processing pipeline for a single window.
        
        Args:
            window: A tuple of (window_text, start_token, end_token)
            window_index: The index of the window in the sequence
            
        Returns:
            A list of extracted and processed facts
        """
        window_text, start_token, end_token = window
        
        # First pass: extract atomic facts
        if self.config.first_pass_enabled:
            facts = self._extract_atomic_facts_first_pass(window_text, window_index)
        else:
            self.logger.warning("First pass disabled, no facts will be extracted")
            return []
        
        # Second pass: improve and refine facts
        if self.config.second_pass_enabled:
            facts = self._extract_atomic_facts_second_pass(facts, window_text)
        
        # Tag facts with additional metadata
        if self.config.tagging_enabled:
            facts = self._tag_facts_in_window(facts)
        
        # Analyze relationships
        if self.config.relationship_analysis_enabled:
            if self.config.use_global_relationship_analysis:
                self._analyze_relationships_global(facts)
            else:
                self._analyze_relationships_pairwise(facts)
        
        return facts
    
    def _create_windows(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Split text into overlapping windows based on configuration.
        
        Args:
            text: The text to split into windows
            
        Returns:
            List of window tuples (text, start_token, end_token)
        """
        # Encode text into tokens
        tokens = self.tokenizer.encode(text)
        
        if not tokens:
            self.logger.warning("Text encoded to empty token list")
            return []
        
        # If text is smaller than window size, use the entire text
        if len(tokens) <= self.config.window_size:
            return [(text, 0, len(tokens))]
        
        windows = []
        
        # Calculate window boundaries
        start_idx = 0
        
        while start_idx < len(tokens):
            # Calculate end index for this window
            end_idx = min(start_idx + self.config.window_size, len(tokens))
            
            # Get the text for this window
            window_tokens = tokens[start_idx:end_idx]
            window_text = self.tokenizer.decode(window_tokens)
            
            # Add to windows list
            windows.append((window_text, start_idx, end_idx))
            
            # If we've reached the end, break
            if end_idx >= len(tokens):
                break
            
            # Calculate next start index (with overlap)
            next_start = end_idx - self.config.overlap_size
            
            # Adjust to try to break at sentence boundaries where possible
            sentence_boundary = self._find_sentence_boundary(window_text, 0.8)
            if sentence_boundary and sentence_boundary > 0.5:
                # Convert the percentage through the text to a token offset
                offset = int(len(window_tokens) * sentence_boundary)
                # Adjust the start index to try to align with sentence boundaries
                adjusted_start = start_idx + offset
                
                # Only use if it's within a reasonable range of the calculated overlap
                if abs(adjusted_start - next_start) < self.config.overlap_size * 0.5:
                    next_start = adjusted_start
            
            # Ensure start_idx actually advances
            start_idx = max(next_start, start_idx + 1)
        
        return windows
    
    def _find_sentence_boundary(self, text: str, preferred_position: float = 0.8) -> Optional[float]:
        """
        Find a good sentence boundary in text.
        
        Args:
            text: The text to analyze
            preferred_position: Preferred position (0-1) in the text
            
        Returns:
            Position (0-1) of a suitable break point, or None if not found
        """
        if not text or len(text) < 10:
            return None
        
        # Look for sentence-ending punctuation followed by space
        # Find all matches
        boundaries = []
        
        for match in re.finditer(r'[.!?]\s+', text):
            position = match.end() / len(text)
            # Store position and distance from preferred position
            boundaries.append((position, abs(position - preferred_position)))
        
        # Also look for paragraph breaks
        for match in re.finditer(r'\n\n', text):
            position = match.end() / len(text)
            # Paragraph breaks are preferred, so reduce the distance artificially
            boundaries.append((position, abs(position - preferred_position) * 0.8))
        
        if not boundaries:
            return None
        
        # Sort by distance from preferred position
        boundaries.sort(key=lambda x: x[1])
        
        # Return the closest boundary position
        return boundaries[0][0]
    
    def _call_llm(self, prompt: str, max_retries: int = 3) -> Any:
        """
        Call the LLM with the given prompt and handle the response.
        
        Args:
            prompt: The prompt to send to the LLM
            max_retries: Maximum number of retries if the call fails
            
        Returns:
            The parsed response from the LLM
        """
        if not self.llm_backend:
            self.logger.error("LLM backend not initialized")
            return None
            
        if not prompt or not isinstance(prompt, str):
            self.logger.error(f"Invalid prompt provided: {type(prompt)}")
            return None
            
        # Log the complete prompt for debugging
        self.output_manager.log_llm_interaction(prompt, None)
        
        # Try to call the LLM with retries
        for attempt in range(max_retries):
            try:
                # Ensure prompt is a string and not empty
                if not prompt.strip():
                    raise ValueError("Empty prompt provided")
                    
                # Call the backend with the prompt using the correct interface
                response = self.llm_backend.call(
                    model=self.config.model,
                    prompt=prompt,
                    temperature=getattr(self.config, 'temperature', 0.7),
                    max_tokens=getattr(self.config, 'max_response_tokens', 1000),
                    expected_format="json"
                )
                
                # Log the response
                self.output_manager.log_llm_interaction(prompt, response)
                
                # Handle different response types
                if isinstance(response, dict):
                    if response.get("error"):
                        raise ValueError(f"LLM error: {response['error']}")
                    
                    # Extract the actual response content
                    if "parsed" in response:
                        parsed = response["parsed"]
                        self.logger.debug(f"Parsed response received")
                        return parsed
                    elif "response" in response:
                        # Try to parse the response as JSON if it's a string
                        if isinstance(response["response"], str):
                            try:
                                # Clean the string before parsing
                                cleaned_response = self._clean_json_string(response["response"])
                                parsed = json.loads(cleaned_response)
                                self.logger.debug(f"Parsed JSON from response")
                                return parsed
                            except json.JSONDecodeError as e:
                                self.logger.warning(f"JSON parsing failed: {str(e)}")
                                return response["response"]
                        return response["response"]
                    else:
                        return response
                elif isinstance(response, str):
                    # Try to parse string response as JSON
                    try:
                        # Clean the string before parsing
                        cleaned_response = self._clean_json_string(response)
                        parsed = json.loads(cleaned_response)
                        self.logger.debug(f"Parsed JSON from string response")
                        return parsed
                    except json.JSONDecodeError:
                        # If not JSON, return the string
                        self.logger.debug(f"Returning raw string response")
                        return response
                else:
                    # Return the raw response for other types
                    self.logger.debug(f"Returning raw response of type {type(response)}")
                    return response
            
            except Exception as e:
                self.logger.warning(f"LLM call attempt {attempt+1}/{max_retries} failed: {str(e)}")
                if attempt < max_retries - 1:
                    # Exponential backoff
                    sleep_time = 2 ** attempt
                    self.logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    self.logger.error(f"All LLM call attempts failed")
                    # Return None on complete failure
                    return None
    
    def __init__(
        self, 
        config: ChunkingConfig = None,
        llm_backend: ILLMBackend = None,
        tokenizer: ITokenizer = None,
        output_manager: IOutputManager = None,
        session_provider: ISessionProvider = None,
        prompt_manager: IPromptManager = None,
        logger = None
    ):
        """
        Initialize the chunker with configuration and dependencies.
        
        Args:
            config: Configuration object
            llm_backend: LLM backend implementation
            tokenizer: Tokenizer implementation
            output_manager: Output manager implementation
            session_provider: Session provider implementation
            prompt_manager: Prompt manager implementation
            logger: Logger instance
        """
        # Initialize configuration
        self.config = config or ChunkingConfig()
        
        # Initialize logger
        self.logger = logger or get_logger(__name__)
        
        # Initialize tokenizer (default uses tiktoken)
        self.tokenizer = tokenizer or DefaultTokenizer(self.config.model)
        
        # Initialize session provider and get session ID
        self.session_provider = session_provider or DefaultSessionProvider()
        self.session_id = self.session_provider.get_session_id()
        
        # Initialize output manager
        self.output_manager = output_manager or DefaultOutputManager(self.logger)
        self.output_manager.initialize(self.session_id, self.config)
        
        # Initialize prompt manager
        self.prompt_manager = prompt_manager or DefaultPromptManager(self.config)
        
        # Initialize LLM backend
        self.llm_backend = llm_backend or self._create_default_llm_backend()
        
        # Initialize parsers
        from core.chunking.parsers import ResponseParserFactory
        self.parsers = {
            'first_pass': ResponseParserFactory.get_first_pass_parser(),
            'second_pass': ResponseParserFactory.get_second_pass_parser(),
            'tagging': ResponseParserFactory.get_tagging_parser(),
            'batch_tagging': ResponseParserFactory.get_batch_tagging_parser(),
            'relationship': ResponseParserFactory.get_relationship_parser(),
            'global_check': ResponseParserFactory.get_global_check_parser(),
            'generic': ResponseParserFactory.get_first_pass_parser(),
        }
        
        # Log initialization
        self.logger.info(f"Initialized TextChunker with session ID: {self.session_id}")
        self.logger.info(f"Using model: {self.config.model}")
    
    def _parse_response(self, response: Any, parser_type: str = 'generic') -> Any:
        """
        Parse a response using the appropriate parser.
        
        Args:
            response: The response to parse
            parser_type: Type of parser to use
            
        Returns:
            Parsed response
        """
        parser = self.parsers.get(parser_type, self.parsers['generic'])
        return parser.parse(response)
    
    def _extract_atomic_facts_first_pass(
        self, window_text: str, window_index: int
    ) -> List[Dict[str, Any]]:
        """
        Extract atomic facts from a window (first pass).
        
        Args:
            window_text: The text of the window
            window_index: The index of the window
            
        Returns:
            List of extracted facts
        """
        self.logger.info(f"Extracting atomic facts from window {window_index+1}")
        
        # Format the prompt using the prompt manager
        prompt = self.prompt_manager.get_prompt("first_pass", window_text=window_text)
        
        # Call the LLM with the prompt
        response = self._call_llm(prompt)
        
        if not response:
            self.logger.warning(f"No response from LLM for window {window_index+1}")
            # Create a fallback fact from the window text
            fallback = [{
                "text": window_text[:300] + "..." if len(window_text) > 300 else window_text,
                "confidence": 0.5,
                "source": "fallback",
                "temporal_info": "",
                "entities": ["window content"]
            }]
            return fallback
        
        # Parse and normalize the response
        facts = self._parse_response(response, 'first_pass')
        
        if not facts or not isinstance(facts, list) or len(facts) == 0:
            self.logger.warning(f"No facts extracted from window {window_index+1}")
            # Try again with a more explicit prompt
            self.logger.info(f"Attempting extraction with alternative prompt")
            alternative_prompt = self.prompt_manager.get_prompt(
                "alternative_first_pass", window_text=window_text
            )
            alternative_response = self._call_llm(alternative_prompt)
            
            if alternative_response:
                facts = self._parse_response(alternative_response, 'first_pass')
        
        # Validate each fact and filter out invalid ones
        valid_facts = []
        if isinstance(facts, list):
            for fact in facts:
                validated_fact = self._validate_atomic_fact(fact)
                if validated_fact:
                    valid_facts.append(validated_fact)
        
        # If no valid facts, create a fallback
        if not valid_facts:
            self.logger.warning(f"No valid facts extracted from window {window_index+1}")
            fallback = [{
                "text": window_text[:300] + "..." if len(window_text) > 300 else window_text,
                "confidence": 0.5,
                "source": "fallback",
                "temporal_info": "",
                "entities": ["window content"]
            }]
            return fallback
        
        self.logger.info(f"Extracted {len(valid_facts)} facts from window {window_index+1}")
        return valid_facts
    
    def _extract_atomic_facts_second_pass(
        self, facts: List[Dict[str, Any]], window_text: str
    ) -> List[Dict[str, Any]]:
        """
        Refine and improve extracted facts (second pass).
        
        Args:
            facts: Facts from the first pass
            window_text: The original window text for context
            
        Returns:
            List of refined facts
        """
        self.logger.info(f"Refining {len(facts)} facts in second pass")
        
        # Skip if no facts to refine
        if not facts:
            return []
        
        # Prepare facts for the prompt
        facts_text = json.dumps(facts, indent=2)
        
        # Format the prompt using the prompt manager
        prompt = self.prompt_manager.get_prompt(
            "second_pass", facts=facts_text, window_text=window_text
        )
        
        # Call the LLM with the prompt
        response = self._call_llm(prompt)
        
        if not response:
            self.logger.warning("No response from LLM for second pass")
            # Return original facts if second pass fails
            return facts
        
        # Parse and normalize the response
        refined_facts = self._parse_response(response, 'second_pass')
        
        if not refined_facts or not isinstance(refined_facts, list) or len(refined_facts) == 0:
            self.logger.warning("No facts extracted from second pass")
            # Try again with a more explicit prompt
            self.logger.info("Attempting refinement with alternative prompt")
            alternative_prompt = self.prompt_manager.get_prompt(
                "alternative_second_pass", facts=facts_text, window_text=window_text
            )
            alternative_response = self._call_llm(alternative_prompt)
            
            if alternative_response:
                refined_facts = self._parse_response(alternative_response, 'second_pass')
            else:
                # Return original facts if alternative prompt fails
                return facts
        
        # Validate each fact and filter out invalid ones
        valid_facts = []
        if isinstance(refined_facts, list):
            for fact in refined_facts:
                validated_fact = self._validate_atomic_fact(fact)
                if validated_fact:
                    valid_facts.append(validated_fact)
        
        # If no valid facts, return original facts
        if not valid_facts:
            self.logger.warning("No valid facts extracted from second pass")
            return facts
        
        self.logger.info(f"Refined {len(valid_facts)} facts in second pass")
        return valid_facts
    
    def _validate_atomic_fact(self, fact: Any) -> Optional[Dict[str, Any]]:
        """
        Validate and normalize an atomic fact.
        
        Args:
            fact: The fact to validate (may be a dict or string)
            
        Returns:
            Validated fact dictionary or None if invalid
        """
        # Handle string facts
        if isinstance(fact, str):
            # Skip empty strings or placeholder text
            if not fact.strip() or self._is_placeholder_text(fact):
                return None
            
            # Convert to dictionary format
            return {
                "text": fact,
                "confidence": 1.0,
                "source": "extraction",
                "temporal_info": "",
                "entities": []
            }
        
        # Handle non-dictionary facts
        if not isinstance(fact, dict):
            return None
        
        # Check for required text field
        if "text" not in fact or not fact["text"]:
            return None
        
        # Check if text is a placeholder
        if self._is_placeholder_text(fact["text"]):
            return None
        
        # Normalize fact structure
        normalized_fact = {
            "text": fact["text"],
            "confidence": 1.0,
            "source": "extraction",
            "temporal_info": "",
            "entities": []
        }
        
        # Copy existing fields if present
        if "confidence" in fact:
            # Ensure confidence is a float between 0 and 1
            try:
                confidence = float(fact["confidence"])
                normalized_fact["confidence"] = max(0.0, min(1.0, confidence))
            except (ValueError, TypeError):
                # Keep default if conversion fails
                pass
        
        if "source" in fact and fact["source"]:
            normalized_fact["source"] = fact["source"]
        
        if "temporal_info" in fact:
            normalized_fact["temporal_info"] = fact["temporal_info"] or ""
        
        # Normalize entities
        if "entities" in fact:
            entities = fact["entities"]
            
            # Handle dictionary-style entities
            if isinstance(entities, dict):
                normalized_fact["entities"] = entities
            # Handle list-style entities
            elif isinstance(entities, list):
                normalized_fact["entities"] = entities
            # Handle string entities (convert to single-item list)
            elif isinstance(entities, str):
                normalized_fact["entities"] = [entities]
            # Default to empty list for other types
            else:
                normalized_fact["entities"] = []
        
        return normalized_fact
    
    def _is_placeholder_text(self, text: str) -> bool:
        """
        Check if text appears to be a placeholder rather than a real fact.
        
        Args:
            text: The text to check
            
        Returns:
            True if the text is a placeholder, False otherwise
        """
        if not text or not isinstance(text, str):
            return True
        
        # Common placeholder patterns in LLM outputs
        placeholder_patterns = [
            r'^Fact \d+:',
            r'^Here are the facts',
            r'^I (couldn\'t|could not) extract',
            r'^No (clear |distinct |)facts',
            r'^The text does not contain',
            r'Unable to determine'
        ]
        
        for pattern in placeholder_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def _tag_facts_in_window(self, facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Add tags and additional metadata to facts in a window.
        
        This method batches facts together based on token count to reduce
        the number of LLM calls needed for tagging.
        
        Args:
            facts: List of facts to tag
            
        Returns:
            List of tagged facts
        """
        self.logger.info(f"Tagging {len(facts)} facts")
        
        # Skip if no facts to tag
        if not facts:
            return []
        
        # Determine batch size based on configuration or default
        max_batch_tokens = getattr(self.config, 'tagging_batch_max_tokens', 4000)
        
        # If batch tagging is disabled or only one fact, use individual tagging
        if getattr(self.config, 'tagging_batch_enabled', True) is False or len(facts) == 1:
            tagged_facts = []
            for fact in facts:
                tagged_fact = self._tag_fact(fact)
                if tagged_fact:
                    tagged_facts.append(tagged_fact)
            return tagged_facts
        
        # Prepare for batch processing
        all_tagged_facts = []
        current_batch = []
        current_batch_tokens = 0
        
        # Calculate tokens for each fact and create batches
        for fact in facts:
            if not fact or "text" not in fact or not fact["text"]:
                continue
                
            # Estimate token count for this fact (approximately)
            fact_tokens = self.tokenizer.get_token_count(fact["text"])
            
            # If this fact would make the batch too large, process the current batch
            if current_batch and current_batch_tokens + fact_tokens > max_batch_tokens:
                # Process the current batch
                batch_tagged_facts = self._tag_facts_batch(current_batch)
                all_tagged_facts.extend(batch_tagged_facts)
                
                # Start a new batch with this fact
                current_batch = [fact]
                current_batch_tokens = fact_tokens
            else:
                # Add to the current batch
                current_batch.append(fact)
                current_batch_tokens += fact_tokens
        
        # Process any remaining facts in the last batch
        if current_batch:
            batch_tagged_facts = self._tag_facts_batch(current_batch)
            all_tagged_facts.extend(batch_tagged_facts)
        
        self.logger.info(f"Tagged {len(all_tagged_facts)} facts in {len(facts) // max(1, len(current_batch))} batches")
        return all_tagged_facts
    
    def _tag_facts_batch(self, facts_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of facts to tag them in a single LLM call.
        
        Args:
            facts_batch: List of facts to tag together
            
        Returns:
            List of tagged facts
        """
        self.logger.debug(f"Batch tagging {len(facts_batch)} facts")
        
        # Skip empty batches
        if not facts_batch:
            return []
            
        # If only one fact, use the single fact tagging method
        if len(facts_batch) == 1:
            tagged_fact = self._tag_fact(facts_batch[0])
            return [tagged_fact] if tagged_fact else []
        
        # Prepare facts for batch prompt
        facts_text = []
        for i, fact in enumerate(facts_batch):
            facts_text.append(f"Fact {i+1}: {fact['text']}")
        
        batch_facts_text = "\n\n".join(facts_text)
        
        # Format the batch prompt
        prompt = self.prompt_manager.get_prompt("batch_tagging", facts=batch_facts_text)
        
        # Call the LLM to get tags for all facts
        response = self._call_llm(prompt)
        
        if not response:
            self.logger.warning(f"No batch tagging response from LLM")
            # Add default tags to all facts in the batch
            tagged_batch = []
            for fact in facts_batch:
                fact_copy = fact.copy()
                self._apply_default_tags(fact_copy)
                tagged_batch.append(fact_copy)
            return tagged_batch
        
        # Parse the response with the batch tagging parser
        parsed_response = self._parse_response(response, 'batch_tagging')
        
        # Process the parsed response
        tagged_batch = []
        
        # If the parsed response has fact_tags
        if isinstance(parsed_response, dict) and "fact_tags" in parsed_response:
            fact_tags = parsed_response["fact_tags"]
            
            # Match tags to facts based on index
            for i, fact in enumerate(facts_batch):
                fact_copy = fact.copy()
                
                # Find tags for this fact index
                fact_idx_str = str(i + 1)  # 1-based indexing in the response
                if fact_idx_str in fact_tags:
                    tag_info = fact_tags[fact_idx_str]
                    self._apply_tag_info_to_fact(fact_copy, tag_info)
                else:
                    # Use default tags if no tags found for this fact
                    self._apply_default_tags(fact_copy)
                
                tagged_batch.append(fact_copy)
        
        # If response is not in the expected format, try alternative parsing
        else:
            # If we got a list of tag objects
            if isinstance(parsed_response, list) and len(parsed_response) > 0:
                # Check if the list items have fact_idx field
                if all(isinstance(item, dict) and "fact_idx" in item for item in parsed_response):
                    # Create a map of fact_idx to tag info
                    tag_map = {item.get("fact_idx", i+1): item for i, item in enumerate(parsed_response)}
                    
                    # Apply tags to each fact
                    for i, fact in enumerate(facts_batch):
                        fact_copy = fact.copy()
                        fact_idx = i + 1  # 1-based indexing
                        
                        if fact_idx in tag_map:
                            tag_info = tag_map[fact_idx]
                            self._apply_tag_info_to_fact(fact_copy, tag_info)
                        else:
                            # Use default tags if no tags found for this fact
                            self._apply_default_tags(fact_copy)
                        
                        tagged_batch.append(fact_copy)
                else:
                    # Try to match list items to facts by position
                    for i, fact in enumerate(facts_batch):
                        fact_copy = fact.copy()
                        
                        if i < len(parsed_response) and isinstance(parsed_response[i], dict):
                            tag_info = parsed_response[i]
                            self._apply_tag_info_to_fact(fact_copy, tag_info)
                        else:
                            # Use default tags if no tags found for this fact
                            self._apply_default_tags(fact_copy)
                        
                        tagged_batch.append(fact_copy)
            else:
                # Fallback to individual tagging
                self.logger.warning(f"Batch tagging response format not recognized, falling back to individual tagging")
                for fact in facts_batch:
                    tagged_fact = self._tag_fact(fact)
                    if tagged_fact:
                        tagged_batch.append(tagged_fact)
        
        return tagged_batch
    
    def _apply_tag_info_to_fact(self, fact: Dict[str, Any], tag_info: Dict[str, Any]) -> None:
        """
        Apply tag information to a fact.
        
        Args:
            fact: The fact to update
            tag_info: The tag information to apply
        """
        # Copy tags if present
        if "tags" in tag_info:
            fact["tags"] = tag_info["tags"]
        else:
            fact["tags"] = []
        
        # Handle topics field (might be 'topics' or 'topic')
        if "topics" in tag_info:
            fact["topics"] = tag_info["topics"]
        elif "topic" in tag_info:
            if isinstance(tag_info["topic"], list):
                fact["topics"] = tag_info["topic"]
            else:
                fact["topics"] = [tag_info["topic"]]
        else:
            fact["topics"] = []
        
        # Handle entities
        if "entities" in tag_info:
            # If the response has structured entities, use them
            fact["entities"] = tag_info["entities"]
        
        # Handle sentiment
        if "sentiment" in tag_info:
            fact["sentiment"] = tag_info["sentiment"]
        else:
            fact["sentiment"] = "neutral"
        
        # Handle importance
        if "importance" in tag_info:
            try:
                importance = int(tag_info["importance"])
                fact["importance"] = max(1, min(10, importance))
            except (ValueError, TypeError):
                fact["importance"] = 5
        else:
            fact["importance"] = 5
    
    def _apply_default_tags(self, fact: Dict[str, Any]) -> None:
        """
        Apply default tag fields to a fact.
        
        Args:
            fact: The fact to update
        """
        fact["tags"] = []
        fact["topics"] = []
        fact["sentiment"] = "neutral"
        fact["importance"] = 5
    
    def _tag_fact(self, fact: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add tags and metadata to a single fact.
        
        Args:
            fact: The fact to tag
            
        Returns:
            Tagged fact
        """
        # Skip if fact is missing required fields
        if not fact or "text" not in fact or not fact["text"]:
            return fact
        
        # Make a copy of the fact to avoid modifying the original
        fact_copy = fact.copy()
        
        # Format the prompt
        prompt = self.prompt_manager.get_prompt("tagging", fact_text=fact["text"])
        
        # Call the LLM to get tags
        response = self._call_llm(prompt)
        
        if not response:
            self.logger.warning(f"No tagging response from LLM for fact: {fact['text'][:50]}...")
            # Add empty tags
            self._apply_default_tags(fact_copy)
            return fact_copy
        
        # Parse the response
        tag_info = self._parse_response(response, 'tagging')
        
        # Apply tags to the fact
        if isinstance(tag_info, dict) and len(tag_info) > 0:
            # Apply tags from response
            self._apply_tag_info_to_fact(fact_copy, tag_info)
        else:
            # If parsing failed, add empty tags
            self._apply_default_tags(fact_copy)
        
        return fact_copy
    
    def _analyze_relationships_pairwise(self, fact1: Dict[str, Any], fact2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the relationship between two facts.
        
        Args:
            fact1: First fact
            fact2: Second fact
            
        Returns:
            Relationship information
        """
        # Skip if either fact is missing required fields
        if (not fact1 or "text" not in fact1 or not fact1["text"] or
            not fact2 or "text" not in fact2 or not fact2["text"]):
            return {"relationship_type": "unrelated", "confidence": 0.0}
        
        # Format the prompt
        prompt = self.prompt_manager.get_prompt(
            "relationship_pairwise",
            fact1=fact1["text"],
            fact2=fact2["text"]
        )
        
        # Call the LLM to analyze the relationship
        response = self._call_llm(prompt)
        
        if not response:
            self.logger.warning(f"No relationship analysis response from LLM")
            # Return default relationship
            return {"relationship_type": "unrelated", "confidence": 0.0}
        
        # Parse the response
        relationship = self._parse_response(response, 'relationship')
        
        # Ensure we have a valid relationship object
        if isinstance(relationship, dict) and "relationship_type" in relationship:
            # Ensure confidence is in bounds
            if "confidence" in relationship:
                try:
                    confidence = float(relationship["confidence"])
                    relationship["confidence"] = max(0.0, min(1.0, confidence))
                except (ValueError, TypeError):
                    relationship["confidence"] = 0.0
            else:
                relationship["confidence"] = 0.0
                
            return relationship
        else:
            # If parsing failed, return default relationship
            return {"relationship_type": "unrelated", "confidence": 0.0}
    
    def _analyze_relationships_global(self, facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze relationships across all facts in a window.
        
        Args:
            facts: List of facts to analyze
            
        Returns:
            List of relationship data
        """
        # Skip if fewer than 2 facts
        if not facts or len(facts) < 2:
            return []
        
        # Prepare facts for the prompt
        facts_text = []
        for i, fact in enumerate(facts):
            if "text" in fact and fact["text"]:
                facts_text.append(f"Fact {i+1}: {fact['text']}")
        
        facts_str = "\n".join(facts_text)
        
        # Format the prompt
        prompt = self.prompt_manager.get_prompt("relationship_global", facts=facts_str)
        
        # Call the LLM to analyze relationships
        response = self._call_llm(prompt)
        
        if not response:
            self.logger.warning(f"No global relationship analysis response from LLM")
            return []
        
        # Parse the response
        parsed = self._parse_response(response, 'relationship')
        
        # Extract relationships
        relationships = []
        
        if isinstance(parsed, dict) and "relationships" in parsed:
            # If we got the expected format with a relationships array
            rels = parsed["relationships"]
            if isinstance(rels, list):
                for rel in rels:
                    if isinstance(rel, dict) and "fact_idx" in rel and "related_fact_idx" in rel:
                        relationships.append(rel)
        elif isinstance(parsed, list):
            # If we got a list directly, check each item
            for rel in parsed:
                if isinstance(rel, dict) and "fact_idx" in rel and "related_fact_idx" in rel:
                    relationships.append(rel)
        
        return relationships
    
    def _global_consistency_check(self, all_facts: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Check for global consistency issues across all facts.
        
        Args:
            all_facts: List of all facts
            
        Returns:
            Dictionary of changes to apply
        """
        # Skip if too few facts
        if not all_facts or len(all_facts) < 2:
            return {"redundancies": [], "contradictions": [], "timeline_issues": []}
        
        # Prepare facts for the prompt
        facts_text = []
        for i, fact in enumerate(all_facts):
            if "text" in fact and fact["text"]:
                facts_text.append(f"Fact {i+1}: {fact['text']}")
        
        facts_str = "\n".join(facts_text)
        
        # Format the prompt
        prompt = self.prompt_manager.get_prompt("global_check", all_facts=facts_str)
        
        # Call the LLM to check global consistency
        response = self._call_llm(prompt)
        
        if not response:
            self.logger.warning(f"No global consistency check response from LLM")
            return {"redundancies": [], "contradictions": [], "timeline_issues": []}
        
        # Parse the response
        parsed = self._parse_response(response, 'global_check')
        
        # Extract changes
        default_changes = {"redundancies": [], "contradictions": [], "timeline_issues": []}
        
        if isinstance(parsed, dict):
            # If we got changes directly
            if all(key in parsed for key in ["redundancies", "contradictions", "timeline_issues"]):
                return parsed
            
            # If we got a changes object
            if "changes" in parsed:
                changes = parsed["changes"]
                if isinstance(changes, dict):
                    # Copy each change category if present
                    for category in ["redundancies", "contradictions", "timeline_issues"]:
                        if category in changes and isinstance(changes[category], list):
                            default_changes[category] = changes[category]
        
        return default_changes
    
    def _apply_global_consistency_changes(
        self, facts: List[Dict[str, Any]], changes: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Apply changes from global consistency check.
        
        Args:
            facts: The facts to modify
            changes: The changes to apply
            
        Returns:
            Updated list of facts
        """
        if not facts:
            return []
        
        if not changes:
            return facts
        
        # Create a copy of the facts to modify
        updated_facts = facts.copy()
        
        # Track which facts to remove
        to_remove = set()
        
        # Apply redundancy changes
        if "redundancies" in changes and changes["redundancies"]:
            for redundancy in changes["redundancies"]:
                if "fact_idx" in redundancy:
                    fact_idx = redundancy["fact_idx"]
                    if isinstance(fact_idx, int) and 0 <= fact_idx < len(updated_facts):
                        to_remove.add(fact_idx)
        
        # Apply contradiction corrections
        if "contradictions" in changes and changes["contradictions"]:
            for contradiction in changes["contradictions"]:
                if "fact_idx" in contradiction and "correction" in contradiction:
                    fact_idx = contradiction["fact_idx"]
                    correction = contradiction["correction"]
                    if (isinstance(fact_idx, int) and 0 <= fact_idx < len(updated_facts)
                            and isinstance(correction, str) and correction.strip()):
                        updated_facts[fact_idx]["text"] = correction.strip()
        
        # Apply timeline issue corrections
        if "timeline_issues" in changes and changes["timeline_issues"]:
            for issue in changes["timeline_issues"]:
                if "fact_idx" in issue and "correction" in issue:
                    fact_idx = issue["fact_idx"]
                    correction = issue["correction"]
                    if (isinstance(fact_idx, int) and 0 <= fact_idx < len(updated_facts)
                            and isinstance(correction, str) and correction.strip()):
                        updated_facts[fact_idx]["text"] = correction.strip()
        
        # Remove redundant facts (in reverse order to avoid index issues)
        for fact_idx in sorted(to_remove, reverse=True):
            del updated_facts[fact_idx]
        
        return updated_facts
    
    def _perform_global_consistency_check(self, processed_windows: List[Dict[str, Any]]) -> None:
        """
        Perform global consistency check across all windows and update facts.
        
        Args:
            processed_windows: List of processed window chunks
        """
        # Extract all facts from all windows
        all_facts = []
        for window in processed_windows:
            if "facts" in window and window["facts"]:
                all_facts.extend(window["facts"])
        
        # Skip if too few facts
        if len(all_facts) < 2:
            self.logger.info("Too few facts for global consistency check")
            return
        
        # Perform consistency check
        changes = self._global_consistency_check(all_facts)
        
        # Skip if no changes
        if not changes or all(not changes[key] for key in changes):
            self.logger.info("No global consistency issues found")
            return
        
        # Log the changes
        self.logger.info(f"Global consistency check found:")
        self.logger.info(f"- {len(changes.get('redundancies', []))} redundancies")
        self.logger.info(f"- {len(changes.get('contradictions', []))} contradictions")
        self.logger.info(f"- {len(changes.get('timeline_issues', []))} timeline issues")
        
        # Apply changes to all facts
        updated_facts = self._apply_global_consistency_changes(all_facts, changes)
        
        # Update facts in windows
        fact_index = 0
        for window in processed_windows:
            if "facts" in window and window["facts"]:
                window_fact_count = len(window["facts"])
                
                # Replace with updated facts
                end_index = min(fact_index + window_fact_count, len(updated_facts))
                window["facts"] = updated_facts[fact_index:end_index]
                
                # Update output files
                self.output_manager.write_chunk(window["chunk_id"], window)
                self.output_manager.write_facts(window["chunk_id"], window["facts"])
                
                fact_index = end_index
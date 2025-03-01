from typing import List, Dict, Any, Optional, Tuple
import tiktoken
import re
import openai
import time
import requests
from pathlib import Path
from datetime import datetime, timedelta
import json
import os
import sys
from dotenv import load_dotenv
from utils.logging import get_logger
from utils.file_ops import OutputManager
from config.chunking.chunking_config import ChunkingConfig
# No longer importing prompts directly, using prompt registry instead
import jsonschema
from core.llm_backends import create_llm_backend
import uuid
import logging

# Load environment variables
load_dotenv()

class AtomicChunker:
    """
    A chunker that extracts atomic units of information from transcripts
    using a sliding window approach with multiple passes.
    """
    
    def __init__(self, config: ChunkingConfig = None):
        """
        Initialize the chunker with configuration.
        
        Args:
            config: Configuration object (optional)
        """
        self.config = config or ChunkingConfig()
        self.encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.logger = get_logger(__name__)
        
        # Generate a unique session ID
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_id = f"{timestamp}_{uuid.uuid4().hex[:8]}"
        
        # Initialize output paths
        model_name = self.config.model.replace("/", "_").replace(":", "_")
        self.output_base_dir = os.path.join("output", model_name, self.session_id)
        self.chunks_dir = os.path.join(self.output_base_dir, "chunks")
        self.facts_dir = os.path.join(self.output_base_dir, "facts")
        self.logs_dir = os.path.join(self.output_base_dir, "logs")
        
        # Create directory structure
        for directory in [self.chunks_dir, self.facts_dir, self.logs_dir]:
            os.makedirs(directory, exist_ok=True)
            
        # Initialize logging files
        self.llm_interactions_log = os.path.join(self.logs_dir, "llm_interactions.log")
        self.llm_metrics_log = os.path.join(self.logs_dir, "llm_metrics.log")
        
        # Set up file handlers for logging
        interactions_handler = logging.FileHandler(self.llm_interactions_log)
        interactions_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(interactions_handler)
        
        metrics_handler = logging.FileHandler(self.llm_metrics_log)
        metrics_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(metrics_handler)
        
        # Initialize the LLM backend
        self._initialize_llm_backend()
        
        # Log initialization
        self.logger.info(f"Initialized AtomicChunker with session ID: {self.session_id}")
        self.logger.info(f"Output directory: {self.output_base_dir}")
        self.logger.info(f"Using model: {self.config.model}")
    
    def _initialize_llm_backend(self):
        """Initialize the LLM backend based on configuration."""
        from core.llm_backends import create_llm_backend
        
        # Get the backend type from config
        backend_type = self.config.llm_backend
        
        if not backend_type:
            # Default to OpenAI if not specified
            backend_type = "openai"
            self.logger.warning(f"No LLM backend specified, defaulting to {backend_type}")
        
        try:
            # Create the backend using the factory function
            self.llm_backend = create_llm_backend(backend_type)
            
            # Initialize backend with config if needed
            if hasattr(self.llm_backend, 'initialize'):
                self.llm_backend.initialize(self.config)
                
            self.logger.info(f"Initialized {backend_type} LLM backend")
        except Exception as e:
            self.logger.error(f"Failed to initialize {backend_type} LLM backend: {str(e)}")
            raise
    
    def _get_metadata(self) -> dict:
        """Get metadata about the current processing run"""
        return {
            "model": self.config.model,
            "llm_backend": getattr(self.config, 'llm_backend', 'unknown'),
            "max_tokens": getattr(self.config, 'max_tokens', None),
            "use_semantic_chunking": getattr(self.config, 'use_semantic_chunking', False),
            "temperature": getattr(self.config, 'temperature', None),
            "max_response_tokens": getattr(self.config, 'max_response_tokens', None),
            "ollama_url": getattr(self.config, 'ollama_url', None),
            "window_size": getattr(self.config, 'window_size', 1000),
            "overlap_size": getattr(self.config, 'overlap_size', 100)
        }
    
    def _call_llm(self, prompt, max_retries=3):
        """
        Call the LLM with the given prompt and handle the response.
        
        Args:
            prompt (str): The prompt to send to the LLM
            max_retries (int): Maximum number of retries if the call fails
            
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
        self.logger.debug(f"Complete LLM prompt:\n{prompt}\n")
        
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
                
                # Log the complete raw response
                self.logger.debug(f"Raw LLM response:\n{response}\n")
                
                # Handle different response types
                if isinstance(response, dict):
                    if response.get("error"):
                        raise ValueError(f"LLM error: {response['error']}")
                    
                    # Extract the actual response content
                    if "parsed" in response:
                        parsed = response["parsed"]
                        self.logger.debug(f"Parsed response:\n{json.dumps(parsed, indent=2)}\n")
                        return parsed
                    elif "response" in response:
                        # Try to parse the response as JSON if it's a string
                        if isinstance(response["response"], str):
                            try:
                                # Clean the string before parsing
                                cleaned_response = self._clean_json_string(response["response"])
                                parsed = json.loads(cleaned_response)
                                self.logger.debug(f"Parsed JSON from response:\n{json.dumps(parsed, indent=2)}\n")
                                return parsed
                            except json.JSONDecodeError as e:
                                self.logger.warning(f"JSON parsing failed: {str(e)}\nFailed JSON:\n{response['response']}\n")
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
                        self.logger.debug(f"Parsed JSON from string:\n{json.dumps(parsed, indent=2)}\n")
                        return parsed
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"JSON parsing failed: {str(e)}\nFailed JSON:\n{response}\n")
                        return response
                
                return response
                
            except Exception as e:
                self.logger.error(f"Error calling LLM (attempt {attempt+1}/{max_retries}): {str(e)}")
                if attempt == max_retries - 1:
                    # Last attempt failed
                    self.logger.error(f"Failed to call LLM after {max_retries} attempts")
                    return None
                    
                # Wait before retrying
                time.sleep(1)
                
        # Should not reach here, but just in case
        return None

    def _clean_json_string(self, json_str: str) -> str:
        """
        Clean a JSON string to make it more parseable using a streamlined approach.
        
        Efficiently extracts and repairs JSON structures from LLM responses.
        """
        if not isinstance(json_str, str):
            return json_str
        
        # Log input for debugging (truncated)
        self.logger.debug(f"Cleaning JSON string: {json_str[:100]}...")
        
        # Try direct parsing first for already valid JSON
        try:
            json.loads(json_str)
            return json_str  # Already valid JSON
        except json.JSONDecodeError:
            pass  # Continue with cleaning
        
        # Find the first { or [ and the last } or ]
        start_indices = [json_str.find(c) for c in "{[" if json_str.find(c) != -1]
        end_indices = [json_str.rfind(c) for c in "}]" if json_str.rfind(c) != -1]
        
        if start_indices and end_indices:
            start_idx = min(start_indices)
            end_idx = max(end_indices)
            if start_idx < end_idx:
                json_str = json_str[start_idx:end_idx+1]
                self.logger.debug(f"Extracted potential JSON: {json_str[:50]}...")
        
        # Apply essential fixes
        # 1. Remove control characters and non-printable characters
        json_str = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', json_str)
        
        # 2. Fix trailing commas before closing brackets/braces
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        # 3. Fix missing quotes around property names
        json_str = re.sub(r'(\{|\,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)
        
        # 4. Fix missing quotes around string values
        json_str = re.sub(r':\s*([a-zA-Z][a-zA-Z0-9_]*)\s*([,}])', r':"\1"\2', json_str)
        
        # 5. Fix unclosed quotes in strings 
        # This regex finds unclosed quotes in array elements (a common issue)
        json_str = re.sub(r'(:\s*\[\s*\".+?)(?=,\s*\])', r'\1"', json_str)
        
        # Check balance of braces and fix
        opening_to_closing = {'{': '}', '[': ']'}
        stack = []
        fixed_str = ""
        
        # First-pass balance check with stack-based approach
        for char in json_str:
            if char in '{[':
                stack.append(char)
                fixed_str += char
            elif char in '}]':
                # Check if this is a valid closing bracket
                if stack and opening_to_closing[stack[-1]] == char:
                    stack.pop()
                    fixed_str += char
                else:
                    # Skip invalid closing bracket
                    continue
            else:
                fixed_str += char
                
        # Add any missing closing brackets/braces
        while stack:
            opener = stack.pop()
            fixed_str += opening_to_closing[opener]
            self.logger.debug(f"Added missing closing {opening_to_closing[opener]}")
        
        json_str = fixed_str
        
        # Try to parse with our fixed JSON
        try:
            json.loads(json_str)
            self.logger.debug("JSON successfully fixed and validated")
            return json_str
        except json.JSONDecodeError as e:
            self.logger.debug(f"JSON still invalid after cleanup: {str(e)}")
            
            # Advanced fix for mismatched quotes in nested structures
            # Look for any unbalanced quotes
            try:
                # Count quotes outside of string literals
                in_string = False
                escape_next = False
                quotes = []
                
                for i, char in enumerate(json_str):
                    if char == '\\' and not escape_next:
                        escape_next = True
                        continue
                    
                    if char == '"' and not escape_next:
                        if not in_string:
                            in_string = True
                            quotes.append(i)  # Record opening quote position
                        else:
                            in_string = False
                            if quotes:  # Remove matched opening quote
                                quotes.pop()
                    
                    escape_next = False
                
                # Fix unbalanced quotes by adding missing quotes
                if quotes:
                    for pos in quotes:
                        # Add closing quote at a reasonable position
                        # Look for a comma, brace or bracket after the opening quote
                        end_pos = json_str.find(',', pos)
                        if end_pos == -1:
                            end_pos = min(pos + 50, len(json_str))  # Limit to 50 chars
                        
                        json_str = json_str[:end_pos] + '"' + json_str[end_pos:]
                        self.logger.debug(f"Added missing quote at position {end_pos}")
            
            except Exception as quote_error:
                self.logger.debug(f"Error fixing quotes: {str(quote_error)}")
            
            # Try again after quote fixes
            try:
                json.loads(json_str)
                return json_str
            except json.JSONDecodeError:
                # More aggressive cleanup
                json_str = re.sub(r'[^\[\]{}",:\s\w._\-#&\'%+*!@()$]', '', json_str)
                
                # Replace any malformed "key": [incomplete... with proper array
                json_str = re.sub(r'"\w+":\s*\[\s*[^]]*(?!\])', r'":[]', json_str)
                
                # Try one final time
                try:
                    json.loads(json_str) 
                    return json_str
                except json.JSONDecodeError:
                    # Last resort: create a minimum valid JSON
                    if json_str.startswith('{'):
                        self.logger.warning("Creating minimum valid JSON object")
                        return "{}"
                    elif json_str.startswith('['):
                        self.logger.warning("Creating minimum valid JSON array")
                        return "[]"
                    else:
                        self.logger.warning("Failed to repair JSON, returning minimal valid JSON")
                        return "{}"
    
    def _split_text_into_windows(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Split text into overlapping windows.
        Returns a list of tuples (window_text, start_token, end_token)
        
        This is a wrapper around _create_windows for backward compatibility.
        """
        return self._create_windows(text)
    
    def _create_windows(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Split text into overlapping windows.
        Returns a list of tuples (window_text, start_token, end_token)
        
        Improved to better handle small texts and ensure meaningful content in the final window.
        """
        try:
            # Debug: Log tokenization start
            debug_msg = f"DEBUG: Starting text tokenization, text length: {len(text)} characters"
            self.logger.info(debug_msg)
            if hasattr(self, 'output_mgr') and self.output_mgr:
                self.output_mgr.log_debug(debug_msg)
            
            # Tokenize with timeout protection
            tokenization_start = time.time()
            tokens = self.encoder.encode(text)
            tokenization_time = time.time() - tokenization_start
            
            # Debug: Log tokenization completion
            debug_msg = f"DEBUG: Tokenization complete, got {len(tokens)} tokens in {tokenization_time:.2f}s"
            self.logger.info(debug_msg)
            if hasattr(self, 'output_mgr') and self.output_mgr:
                self.output_mgr.log_debug(debug_msg)
            
            # Get window size and overlap from config
            window_size = self.config.window_size
            overlap_size = self.config.overlap_size
            
            # Handle case where text is small enough to fit in a single window
            if len(tokens) <= window_size:
                self.logger.info(f"Text is small enough to fit in a single window ({len(tokens)} tokens <= {window_size})")
                window_text = self.encoder.decode(tokens)
                return [(window_text, 0, len(tokens))]
            
            # For small texts with multiple windows, reduce overlap to ensure meaningful content
            if len(tokens) < window_size * 2 and overlap_size > window_size // 4:
                # Reduce overlap for small texts to get more meaningful windows
                adjusted_overlap = window_size // 4
                self.logger.info(f"Adjusted overlap size from {overlap_size} to {adjusted_overlap} for small text")
                overlap_size = adjusted_overlap
            
            windows = []
            
            # Debug: Log window creation start
            debug_msg = f"DEBUG: Starting window creation with window_size={window_size}, overlap={overlap_size}"
            self.logger.info(debug_msg)
            if hasattr(self, 'output_mgr') and self.output_mgr:
                self.output_mgr.log_debug(debug_msg)
            
            # Limit the number of windows for safety
            max_windows = 100
            window_count = 0
            
            # Create windows with the configured size and overlap
            start_idx = 0
            while start_idx < len(tokens):
                # Calculate end index for this window
                end_idx = min(start_idx + window_size, len(tokens))
                
                # Debug: Log window indices
                debug_msg = f"DEBUG: Window {window_count+1} indices: {start_idx}-{end_idx}, length={end_idx-start_idx} tokens"
                self.logger.info(debug_msg)
                if hasattr(self, 'output_mgr') and self.output_mgr:
                    self.output_mgr.log_debug(debug_msg)
                
                # Get tokens for this window
                window_tokens = tokens[start_idx:end_idx]
                
                # Check for minimum window size to ensure meaningful content
                min_window_size = window_size // 3
                if len(window_tokens) < min_window_size and len(windows) > 0:
                    # If window is too small and not the first window, merge with previous window
                    self.logger.info(f"Window {window_count+1} too small ({len(window_tokens)} < {min_window_size}), skipping")
                    break
                
                # Decode tokens to text
                decode_start = time.time()
                window_text = self.encoder.decode(window_tokens)
                decode_time = time.time() - decode_start
                
                # Debug: Log decoding completion
                debug_msg = f"DEBUG: Decoded window {window_count+1}, got {len(window_text)} characters in {decode_time:.2f}s"
                self.logger.info(debug_msg)
                if hasattr(self, 'output_mgr') and self.output_mgr:
                    self.output_mgr.log_debug(debug_msg)
                
                # Special handling for the final window
                is_final_window = end_idx >= len(tokens)
                if is_final_window and len(windows) > 0:
                    # For the final window, check for minimum unique content
                    prev_window_text = windows[-1][0]
                    min_unique_ratio = 0.3  # At least 30% unique content
                    
                    # Check how much unique content is in this window compared to previous
                    unique_chars = sum(1 for c in window_text if c not in prev_window_text[-overlap_size*4:])
                    unique_ratio = unique_chars / len(window_text) if window_text else 0
                    
                    self.logger.info(f"Final window has {unique_ratio:.2f} unique content ratio")
                    
                    if unique_ratio < min_unique_ratio:
                        # If final window has too much overlap with previous, expand previous window instead
                        self.logger.info(f"Final window has insufficient unique content, extending previous window")
                        
                        # Replace previous window with extended one
                        prev_window_text, prev_start, _ = windows.pop()
                        extended_tokens = tokens[prev_start:len(tokens)]
                        extended_text = self.encoder.decode(extended_tokens)
                        windows.append((extended_text, prev_start, len(tokens)))
                        
                        # Skip adding this window
                        break
                
                # Add window to list
                windows.append((window_text, start_idx, end_idx))
                
                # Move to next window with overlap
                prev_start_idx = start_idx
                start_idx = end_idx - overlap_size
                
                # Safety check to prevent infinite loops
                if start_idx <= prev_start_idx or start_idx >= len(tokens):
                    debug_msg = f"DEBUG: Breaking window creation loop, start_idx={start_idx}, prev_start_idx={prev_start_idx}, len(tokens)={len(tokens)}"
                    self.logger.info(debug_msg)
                    if hasattr(self, 'output_mgr') and self.output_mgr:
                        self.output_mgr.log_debug(debug_msg)
                    break
                
                window_count += 1
                
                # Safety limit on number of windows
                if window_count >= max_windows:
                    debug_msg = f"DEBUG: Reached maximum window count ({max_windows}), stopping window creation"
                    self.logger.warning(debug_msg)
                    if hasattr(self, 'output_mgr') and self.output_mgr:
                        self.output_mgr.log_debug(debug_msg)
                    break
            
            # Debug: Log window creation completion
            debug_msg = f"DEBUG: Window creation complete, created {len(windows)} windows"
            self.logger.info(debug_msg)
            if hasattr(self, 'output_mgr') and self.output_mgr:
                self.output_mgr.log_debug(debug_msg)
            
            # Sanity check: ensure we have at least one window
            if not windows:
                self.logger.warning("No windows created, falling back to single window")
                window_text = self.encoder.decode(tokens)
                return [(window_text, 0, len(tokens))]
            
            return windows
            
        except Exception as e:
            error_msg = f"ERROR in _create_windows: {str(e)}"
            self.logger.error(error_msg)
            if hasattr(self, 'output_mgr') and self.output_mgr:
                self.output_mgr.log_debug(error_msg)
            
            # Return a single window with the entire text as fallback
            if hasattr(self, 'output_mgr') and self.output_mgr:
                self.output_mgr.log_debug("DEBUG: Falling back to single window with entire text")
            
            try:
                tokens = self.encoder.encode(text)
                return [(text, 0, len(tokens))]
            except:
                # Last resort fallback
                if hasattr(self, 'output_mgr') and self.output_mgr:
                    self.output_mgr.log_debug("DEBUG: Complete fallback, returning empty window list")
                return []
    
    def _extract_atomic_facts_first_pass(self, window_text, window_num):
        """
        Extract atomic facts from the window text using the first pass prompt.
        
        Args:
            window_text (str): The text window to extract facts from
            window_num (int): The window number for logging purposes
            
        Returns:
            list: A list of extracted facts
        """
        self.logger.debug(f"Starting first pass fact extraction for window {window_num}")
        
        # Get the first pass prompt from the centralized registry
        try:
            # Use the new get_prompt method to get the formatted prompt
            prompt = self.config.get_prompt("first_pass", window_text=window_text)
            self.logger.debug("Using first_pass prompt from registry")
        except (ValueError, KeyError) as e:
            self.logger.warning(f"Error getting first_pass prompt from registry: {str(e)}")
            
            # Fall back to legacy prompt access methods
            prompt_template = None
            
            # Try different ways to access the prompt (legacy)
            if hasattr(self.config, 'first_pass_prompt') and self.config.first_pass_prompt:
                prompt_template = self.config.first_pass_prompt
                self.logger.debug("Using first_pass_prompt from config (legacy)")
            elif hasattr(self.config.prompts, 'FIRST_PASS_PROMPT') and self.config.prompts.FIRST_PASS_PROMPT:
                prompt_template = self.config.prompts.FIRST_PASS_PROMPT
                self.logger.debug("Using FIRST_PASS_PROMPT from config.prompts (legacy)")
            
            if not prompt_template:
                self.logger.warning("First pass prompt template is empty. No facts will be extracted.")
                return []
                
            # Format the prompt with the window text (legacy method)
            try:
                # Clean up the template to ensure proper formatting
                if '{text}' in prompt_template:
                    prompt = prompt_template.replace('{text}', window_text)
                elif '{window_text}' in prompt_template:
                    prompt = prompt_template.replace('{window_text}', window_text)
                else:
                    # If no placeholder is found, append the text to the prompt
                    prompt = f"{prompt_template}\n\nText: {window_text}"
            except Exception as e:
                self.logger.error(f"Error formatting prompt: {str(e)}")
                # Fallback to a simple prompt
                prompt = f"""
                Extract atomic facts from the following text. Return a JSON array of fact objects.
                
                Text:
                {window_text}
                
                Return ONLY a JSON array of facts with these fields: text, confidence, source, temporal_info, entities.
                """
            self.logger.debug("Using fallback prompt due to formatting error")
        
        self.logger.debug(f"First pass prompt (first 200 chars): {prompt[:200]}...")
        self.logger.debug(f"Complete LLM prompt:\n{prompt}\n")
        
        # Use retry mechanism for robust fact extraction
        max_extraction_attempts = 2
        for attempt in range(max_extraction_attempts):
            # Call the LLM with the prompt
            response = self._call_llm(prompt)
            
            # Log the raw response
            self.logger.debug(f"Raw LLM response for window {window_num} (attempt {attempt+1}):\n{response}\n")
            
            # Process the response
            facts = []
            if not response:
                self.logger.warning(f"Empty response from LLM for window {window_num}")
                if attempt < max_extraction_attempts - 1:
                    self.logger.info(f"Retrying fact extraction for window {window_num}")
                    continue
                else:
                    return []
            
            # Handle different response formats
            try:
                if isinstance(response, dict):
                    # Log the dictionary structure
                    self.logger.debug(f"Response is a dictionary with keys: {list(response.keys())}")
                    
                    # If response is already a dictionary
                    if 'facts' in response:
                        facts = response['facts']
                        self.logger.debug(f"Extracted facts from 'facts' key: {len(facts)} facts")
                    elif 'parsed' in response:
                        parsed_data = response['parsed']
                        self.logger.debug(f"Found 'parsed' key with type: {type(parsed_data)}")
                        if isinstance(parsed_data, list):
                            facts = parsed_data
                            self.logger.debug(f"Using parsed list directly: {len(facts)} facts")
                        elif isinstance(parsed_data, dict) and 'facts' in parsed_data:
                            facts = parsed_data['facts']
                            self.logger.debug(f"Extracted facts from parsed dictionary: {len(facts)} facts")
                    elif 'response' in response:
                        # Try to parse the response string
                        response_str = response['response']
                        self.logger.debug(f"Found 'response' key with content:\n{response_str}\n")
                        try:
                            # Use the enhanced cleaning function for better JSON parsing
                            cleaned_response = self._clean_json_string(response_str)
                            self.logger.debug(f"Cleaned response string:\n{cleaned_response}\n")
                            
                            parsed = json.loads(cleaned_response)
                            if isinstance(parsed, list):
                                facts = parsed
                            elif isinstance(parsed, dict) and 'facts' in parsed:
                                facts = parsed['facts']
                            self.logger.debug(f"Parsed response string into {len(facts)} facts")
                        except json.JSONDecodeError as e:
                            self.logger.warning(f"Failed to parse response as JSON: {str(e)}")
                            
                            # Look for markdown lists or bullet points if JSON parsing fails
                            if '- ' in response_str or '* ' in response_str:
                                # Extract facts from markdown-style bullet points
                                facts_lines = re.findall(r'(?:^|\n)[\s]*[-*][\s]*(.*?)(?=\n[\s]*[-*]|$)', response_str)
                                facts = [{'text': line.strip(), 'confidence': 1.0, 'source': 'extraction', 
                                          'temporal_info': '', 'entities': []} for line in facts_lines if line.strip()]
                                self.logger.debug(f"Extracted {len(facts)} facts from markdown bullet points")
                            else:
                                # If not bullet points, split by newlines
                                facts = [line.strip() for line in response_str.split('\n') if line.strip() and len(line.strip()) > 10]
                                self.logger.debug(f"Split response into {len(facts)} line-based facts")
                                
                                # Convert string facts to proper fact objects
                                facts = [{'text': line, 'confidence': 1.0, 'source': 'extraction', 
                                         'temporal_info': '', 'entities': []} for line in facts]
                            
                            # Retry with a more explicit format instruction if parsing failed but we have content
                            if len(facts) == 0 and attempt < max_extraction_attempts - 1:
                                self.logger.warning(f"Failed to extract structured facts, retrying with clearer instructions")
                                # Add a more explicit JSON format instruction
                                prompt += "\n\nIMPORTANT: Please format your response ONLY as a JSON array of objects with these exact fields: text, confidence, source, temporal_info, entities. Do not include any explanatory text before or after the JSON."
                                continue
                elif isinstance(response, str):
                    # Log the string response
                    self.logger.debug(f"Response is a string:\n{response}\n")
                    
                    # Try to parse as JSON first
                    try:
                        # Use the enhanced cleaning function
                        cleaned_response = self._clean_json_string(response)
                        self.logger.debug(f"Cleaned response string:\n{cleaned_response}\n")
                        
                        parsed = json.loads(cleaned_response)
                        if isinstance(parsed, list):
                            facts = parsed
                            self.logger.debug(f"Parsed string as JSON array: {len(facts)} facts")
                        elif isinstance(parsed, dict) and 'facts' in parsed:
                            facts = parsed['facts']
                            self.logger.debug(f"Parsed string as JSON object with facts: {len(facts)} facts")
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Failed to parse string as JSON: {str(e)}")
                        
                        # Look for markdown lists or bullet points
                        if '- ' in response or '* ' in response:
                            # Extract facts from markdown-style bullet points
                            facts_lines = re.findall(r'(?:^|\n)[\s]*[-*][\s]*(.*?)(?=\n[\s]*[-*]|$)', response)
                            facts = [{'text': line.strip(), 'confidence': 1.0, 'source': 'extraction', 
                                      'temporal_info': '', 'entities': []} for line in facts_lines if line.strip()]
                            self.logger.debug(f"Extracted {len(facts)} facts from markdown bullet points")
                        else:
                            # If not bullet points, split by newlines and filter out short lines
                            facts = [line.strip() for line in response.split('\n') if line.strip() and len(line.strip()) > 10]
                            self.logger.debug(f"Split string into {len(facts)} line-based facts")
                            
                            # Convert string facts to proper fact objects
                            facts = [{'text': line, 'confidence': 1.0, 'source': 'extraction', 
                                     'temporal_info': '', 'entities': []} for line in facts]
                        
                        # Retry with a more explicit format instruction if parsing failed but we have text content
                        if len(facts) == 0 and attempt < max_extraction_attempts - 1:
                            self.logger.warning(f"Failed to extract structured facts, retrying with clearer instructions")
                            # Add a more explicit JSON format instruction
                            prompt += "\n\nIMPORTANT: Please format your response ONLY as a JSON array of objects with these exact fields: text, confidence, source, temporal_info, entities. Do not include any explanatory text before or after the JSON."
                            continue
                elif isinstance(response, list):
                    facts = response
                    self.logger.debug(f"Response is already a list with {len(facts)} facts")
                    
            except Exception as e:
                self.logger.error(f"Error processing response: {str(e)}")
                if attempt < max_extraction_attempts - 1:
                    self.logger.info(f"Retrying fact extraction for window {window_num} due to processing error")
                    continue
                else:
                    facts = []
            
            # If we have facts, break out of retry loop
            if facts:
                break
        
        # Log the final extracted facts
        self.logger.debug(f"Final extracted facts for window {window_num}:")
        for i, fact in enumerate(facts):
            self.logger.debug(f"Fact {i+1}: {fact}")
        
        # Log the number of facts extracted
        self.logger.debug(f"Extracted {len(facts)} facts from first pass for window {window_num}")
        
        # Validate and normalize facts
        valid_facts = []
        for fact in facts:
            if isinstance(fact, str) and fact.strip():
                # Convert string facts to dict format
                valid_facts.append({
                    'text': fact.strip(),
                    'confidence': 1.0,
                    'source': 'extraction',
                    'temporal_info': '',
                    'entities': []
                })
            elif isinstance(fact, dict) and fact.get('text', '').strip():
                # Ensure all required fields are present
                if not 'confidence' in fact:
                    fact['confidence'] = 1.0
                if not 'source' in fact:
                    fact['source'] = 'extraction'
                if not 'temporal_info' in fact:
                    fact['temporal_info'] = ''
                if not 'entities' in fact:
                    fact['entities'] = []
                valid_facts.append(fact)
        
        # Check if we got valid facts
        if not valid_facts and window_text.strip():
            self.logger.warning(f"No valid facts extracted from window {window_num}, creating a fallback fact")
            # Create a fallback fact with the beginning of the window text
            first_sentence = re.split(r'[.!?]', window_text)[0] + "."
            if len(first_sentence) > 20:  # Only if it's a substantial sentence
                fallback_fact = {
                    'text': first_sentence.strip(),
                    'confidence': 0.5,
                    'source': 'fallback',
                    'temporal_info': '',
                    'entities': []
                }
                valid_facts = [fallback_fact]
                
        return valid_facts

    def _extract_atomic_facts_second_pass(self, window_text, first_pass_facts, window_num):
        """
        Refine the facts extracted in the first pass using the second pass prompt.
        
        Args:
            window_text (str): The text of the window to extract facts from
            first_pass_facts (list): The facts extracted in the first pass
            window_num (int): The window number for logging purposes
            
        Returns:
            list: A list of refined facts
        """
        self.logger.debug(f"Starting second pass fact extraction for window {window_num}")
        
        # If no facts were extracted in the first pass, return an empty list
        if not first_pass_facts:
            self.logger.warning(f"No facts to refine in second pass for window {window_num}")
            return []
            
        # Convert facts to a string if they are dictionaries
        facts_text = ""
        if isinstance(first_pass_facts[0], dict):
            facts_text = "\n".join([f"{i+1}. {fact.get('text', '')}" for i, fact in enumerate(first_pass_facts)])
        else:
            facts_text = "\n".join([f"{i+1}. {fact}" for i, fact in enumerate(first_pass_facts)])
            
        # Get the second pass prompt from the centralized registry
        try:
            # Use the new get_prompt method to get the formatted prompt
            prompt = self.config.get_prompt("second_pass", window_text=window_text, facts_text=facts_text)
            self.logger.debug("Using second_pass prompt from registry")
        except (ValueError, KeyError) as e:
            self.logger.warning(f"Error getting second_pass prompt from registry: {str(e)}")
            
            # Fall back to legacy prompt access methods
            prompt_template = None
            
            # Try different ways to access the prompt (legacy)
            if hasattr(self.config, 'second_pass_prompt') and self.config.second_pass_prompt:
                prompt_template = self.config.second_pass_prompt
                self.logger.debug("Using second_pass_prompt from config (legacy)")
            elif hasattr(self.config.prompts, 'SECOND_PASS_PROMPT') and self.config.prompts.SECOND_PASS_PROMPT:
                prompt_template = self.config.prompts.SECOND_PASS_PROMPT
                self.logger.debug("Using SECOND_PASS_PROMPT from config.prompts (legacy)")
            
            if not prompt_template:
                self.logger.warning("Second pass prompt template is empty. Using first pass facts as is.")
                return first_pass_facts
                
            # Format the prompt with the window text and facts (legacy method)
            try:
                # Clean up the template to ensure proper formatting
                if '{text}' in prompt_template and '{facts}' in prompt_template:
                    prompt = prompt_template.replace('{text}', window_text).replace('{facts}', facts_text)
                elif '{window_text}' in prompt_template and '{facts_text}' in prompt_template:
                    prompt = prompt_template.replace('{window_text}', window_text).replace('{facts_text}', facts_text)
                else:
                    # If no placeholder is found, append the text and facts to the prompt
                    prompt = f"{prompt_template}\n\nText: {window_text}\n\nFacts:\n{facts_text}"
            except Exception as e:
                self.logger.error(f"Error formatting prompt: {str(e)}")
                # Fallback to a simple prompt
                prompt = f"""
                Review and improve these extracted facts to ensure they are atomic and self-contained.
                
                Original text:
                {window_text}
                
                Facts to review:
                {facts_text}
                
                Return ONLY a JSON array of facts with these fields: text, confidence, source, temporal_info, entities.
                """
            self.logger.debug("Using fallback prompt due to formatting error")
        
        self.logger.debug(f"Second pass prompt (first 200 chars): {prompt[:200]}...")
        
        # Call the LLM with the prompt
        response = self._call_llm(prompt)
        
        # Process the response
        refined_facts = []
        if not response:
            self.logger.warning(f"Empty response from LLM for second pass in window {window_num}")
            return first_pass_facts  # Return original facts if no response
            
        try:
            # Handle different response formats
            if isinstance(response, dict):
                if 'facts' in response:
                    refined_facts = response['facts']
                elif 'parsed' in response:
                    parsed_data = response['parsed']
                    if isinstance(parsed_data, list):
                        refined_facts = parsed_data
                    elif isinstance(parsed_data, dict) and 'facts' in parsed_data:
                        refined_facts = parsed_data['facts']
                elif 'response' in response:
                    # Try to parse the response string
                    try:
                        parsed = json.loads(response['response'])
                        if isinstance(parsed, list):
                            refined_facts = parsed
                        elif isinstance(parsed, dict) and 'facts' in parsed:
                            refined_facts = parsed['facts']
                    except json.JSONDecodeError:
                        # If not JSON, split by newlines
                        lines = response['response'].split('\n') if isinstance(response['response'], str) else []
                        refined_facts = [line.strip() for line in lines if line.strip()]
                else:
                    # Treat the entire response as facts
                    refined_facts = [response]
            elif isinstance(response, str):
                # Try to parse as JSON first
                try:
                    parsed = json.loads(response)
                    if isinstance(parsed, list):
                        refined_facts = parsed
                    elif isinstance(parsed, dict) and 'facts' in parsed:
                        refined_facts = parsed['facts']
                    else:
                        refined_facts = [parsed]
                except json.JSONDecodeError:
                    # If not JSON, split by newlines
                    lines = response.split('\n')
                    refined_facts = [line.strip() for line in lines if line.strip()]
            elif isinstance(response, list):
                refined_facts = response
            else:
                self.logger.warning(f"Unexpected response type: {type(response)}")
                return first_pass_facts
                
        except Exception as e:
            self.logger.error(f"Error processing second pass response: {str(e)}")
            return first_pass_facts
            
        # Validate the facts
        valid_facts = []
        for fact in refined_facts:
            if isinstance(fact, str) and fact.strip():
                # Convert string facts to dict format
                valid_facts.append({
                    'text': fact.strip(),
                    'confidence': 1.0,
                    'source': 'extraction',
                    'temporal_info': '',
                    'entities': []
                })
            elif isinstance(fact, dict) and fact.get('text', '').strip():
                valid_facts.append(fact)
                
        if not valid_facts:
            self.logger.warning(f"No valid facts after second pass for window {window_num}, using first pass facts")
            return first_pass_facts
            
        self.logger.debug(f"Validated {len(valid_facts)} facts from second pass")
        return valid_facts

    def _deduplicate_facts(self, all_facts: List[str]) -> List[str]:
        """Deduplicate and merge similar facts"""
        if not all_facts:
            return []
        
        # First, normalize facts for comparison
        normalized_facts = [fact.lower().strip() for fact in all_facts]
        
        # Use a set to track which facts we've already processed
        processed_indices = set()
        deduplicated_facts = []
        
        for i, fact in enumerate(all_facts):
            if i in processed_indices:
                continue
                
            processed_indices.add(i)
            deduplicated_facts.append(fact)
            
            # Check for similar facts
            for j, other_fact in enumerate(all_facts):
                if i != j and j not in processed_indices:
                    # Simple similarity check - could be improved
                    if normalized_facts[i] in normalized_facts[j] or normalized_facts[j] in normalized_facts[i]:
                        processed_indices.add(j)
                        # We could merge facts here if needed
        
        return deduplicated_facts
    
    def _global_check(self, facts):
        """
        Perform a global check to ensure continuity and resolve contradictions.
        
        Args:
            facts (list): List of facts to check
            
        Returns:
            list: List of facts after global check
        """
        # Check if global check is enabled
        if not getattr(self.config, 'global_check_enabled', True):
            self.logger.debug("Global check is disabled")
            return facts
            
        # Check if there are enough facts to analyze
        if len(facts) < 2:
            self.logger.debug("Not enough facts for global check")
            return facts
            
        self.logger.debug(f"Performing global check on {len(facts)} facts")
        
        # Create a string with all facts
        facts_text = "\n".join([f"{i+1}. {fact.get('text', '') if isinstance(fact, dict) else fact}" for i, fact in enumerate(facts)])
        
        # Get the global check prompt from the centralized registry
        try:
            # Use the new get_prompt method to get the formatted prompt
            prompt = self.config.get_prompt("global_check", facts_text=facts_text)
            self.logger.debug("Using global_check prompt from registry")
        except (ValueError, KeyError) as e:
            self.logger.warning(f"Error getting global_check prompt from registry: {str(e)}")
            
            # Fall back to legacy prompt access methods
            prompt_template = None
            
            # Try different ways to access the prompt (legacy)
            if hasattr(self.config, 'global_check_prompt') and self.config.global_check_prompt:
                prompt_template = self.config.global_check_prompt
                self.logger.debug("Using global_check_prompt from config (legacy)")
            elif hasattr(self.config.prompts, 'GLOBAL_CHECK_PROMPT') and self.config.prompts.GLOBAL_CHECK_PROMPT:
                prompt_template = self.config.prompts.GLOBAL_CHECK_PROMPT
                self.logger.debug("Using GLOBAL_CHECK_PROMPT from config.prompts (legacy)")
            
            if not prompt_template:
                self.logger.warning("Global check prompt template is empty. No global check will be performed.")
                return facts
                
            # Format the prompt (legacy method)
            try:
                if '{facts_text}' in prompt_template:
                    prompt = prompt_template.replace('{facts_text}', facts_text)
                else:
                    prompt = f"{prompt_template}\n\nFacts:\n{facts_text}"
            except Exception as e:
                self.logger.error(f"Error formatting global check prompt: {str(e)}")
                return facts
                
        self.logger.debug(f"Global check prompt (first 200 chars): {prompt[:200]}...")
            
        # Call the LLM with the prompt
        response = self._call_llm(prompt)
        
        # Process the response
        if not response:
            self.logger.warning("Empty response from LLM for global check")
            return facts
            
        # Try to parse the response as JSON
        try:
            if isinstance(response, dict) and 'changes' in response:
                # If the response is already a dictionary with a 'changes' key
                changes = response['changes']
                self.logger.debug(f"Found {len(changes)} changes in global check response")
            elif isinstance(response, str):
                # Try to parse the response as a JSON string
                import json
                parsed = json.loads(response)
                if isinstance(parsed, dict) and 'changes' in parsed:
                    changes = parsed['changes']
                else:
                    changes = parsed  # Assume the whole response is the changes object
                self.logger.debug(f"Parsed {len(changes)} changes from global check response")
            else:
                self.logger.warning("Unknown response format from global check")
                return facts
                
            # Apply the changes to the facts
            updated_facts = facts.copy()
            
            # Process redundancies
            if 'redundancies' in changes:
                redundancies = changes['redundancies']
                self.logger.debug(f"Processing {len(redundancies)} redundancies")
                # Mark redundant facts for removal
                for redundancy in redundancies:
                    fact_idx = redundancy.get('fact_idx')
                    if fact_idx is not None and 0 <= fact_idx < len(updated_facts):
                        updated_facts[fact_idx] = None  # Mark for removal
                        
            # Process contradictions
            if 'contradictions' in changes:
                contradictions = changes['contradictions']
                self.logger.debug(f"Processing {len(contradictions)} contradictions")
                # Update contradicting facts
                for contradiction in contradictions:
                    fact_idx = contradiction.get('fact_idx')
                    correction = contradiction.get('correction')
                    if fact_idx is not None and correction and 0 <= fact_idx < len(updated_facts):
                        if isinstance(updated_facts[fact_idx], dict):
                            updated_facts[fact_idx]['text'] = correction
                        else:
                            updated_facts[fact_idx] = correction
                            
            # Process timeline issues
            if 'timeline_issues' in changes:
                timeline_issues = changes['timeline_issues']
                self.logger.debug(f"Processing {len(timeline_issues)} timeline issues")
                # Update facts with timeline issues
                for issue in timeline_issues:
                    fact_idx = issue.get('fact_idx')
                    correction = issue.get('correction')
                    if fact_idx is not None and correction and 0 <= fact_idx < len(updated_facts):
                        if isinstance(updated_facts[fact_idx], dict):
                            updated_facts[fact_idx]['text'] = correction
                        else:
                            updated_facts[fact_idx] = correction
                            
            # Remove None values (marked for removal)
            updated_facts = [fact for fact in updated_facts if fact is not None]
            
            self.logger.debug(f"Global check complete, {len(facts) - len(updated_facts)} facts removed")
            return updated_facts
            
        except Exception as e:
            self.logger.error(f"Error processing global check response: {str(e)}")
            return facts
    
    def _tag_fact(self, fact: Dict, fact_num: int = None) -> Dict:
        """
        Add tags, topics, and entity information to a fact
        
        Args:
            fact: The atomic fact dictionary
            fact_num: The fact number for logging (optional)
            
        Returns:
            Updated fact dictionary with tags, topic, and entities
        """
        # Use fact_id as fact_num if not provided
        if fact_num is None and "id" in fact:
            fact_num = fact["id"].replace("fact_", "")
        elif fact_num is None:
            fact_num = 0
            
        # Debug: Log tagging start
        debug_msg = f"DEBUG: Starting tagging for fact {fact_num}"
        self.logger.info(debug_msg)
        if hasattr(self, 'output_mgr') and self.output_mgr:
            self.output_mgr.log_debug(debug_msg)
        
        # Get the fact text
        fact_text = fact.get("text", "")
        if not fact_text and isinstance(fact, str):
            fact_text = fact
            
        if not fact_text:
            self.logger.warning(f"No text found in fact {fact_num}")
            fact["tags"] = []
            fact["topic"] = "unknown"
            fact["entities"] = {"people": [], "places": [], "organizations": [], "other": []}
            return fact
        
        # Get the tagging prompt from the centralized registry
        try:
            # Use the new get_prompt method to get the formatted prompt
            prompt = self.config.get_prompt("tagging", fact=fact_text)
            self.logger.debug("Using tagging prompt from registry")
        except (ValueError, KeyError) as e:
            self.logger.warning(f"Error getting tagging prompt from registry: {str(e)}")
            
            # Fall back to legacy prompt access methods
            prompt_template = None
            
            # Try different ways to access the prompt (legacy)
            if hasattr(self.config, 'tagging_prompt') and self.config.tagging_prompt:
                prompt_template = self.config.tagging_prompt
                self.logger.debug("Using tagging_prompt from config (legacy)")
            elif hasattr(self.config.prompts, 'TAGGING_PROMPT') and self.config.prompts.TAGGING_PROMPT:
                prompt_template = self.config.prompts.TAGGING_PROMPT
                self.logger.debug("Using TAGGING_PROMPT from config.prompts (legacy)")
            
            if not prompt_template:
                # Use a default prompt if none is found
                prompt_template = """
                Analyze this fact and categorize its entities:

                Fact: {fact_text}

                Return a valid JSON object with this EXACT structure:
                {
                    "tags": ["keyword1", "keyword2"],
                    "topics": ["topic1", "topic2"],
                    "entities": {
                        "people": ["person names only"],
                        "places": ["location names only"],
                        "organizations": ["organization names only"],
                        "other": ["anything else"]
                    },
                    "sentiment": "positive|negative|neutral",
                    "importance": 1-10
                }

                IMPORTANT:
                - Put all person names in "people"
                - Put all location names in "places"
                - Put all organization/institution names in "organizations"
                - Put remaining entities in "other"
                - Ensure proper JSON formatting with quotes around all strings
                - Rate importance on a scale of 1-10
                """
                
            # Format the prompt (legacy method)
            try:
                # Try different placeholder formats
                if '{fact_text}' in prompt_template:
                    prompt = prompt_template.replace('{fact_text}', fact_text)
                elif '{text}' in prompt_template:
                    prompt = prompt_template.replace('{text}', fact_text)
                elif '{fact}' in prompt_template:
                    prompt = prompt_template.replace('{fact}', fact_text)
                else:
                    prompt = f"{prompt_template}\n\nFact: {fact_text}"
            except Exception as e:
                self.logger.error(f"Error formatting tagging prompt: {str(e)}")
                # Use a simple fallback prompt
                prompt = f"""
                Analyze this fact and return JSON with tags, topic, and categorized entities:
                {fact_text}
                """
        
        self.logger.debug(f"Tagging prompt for fact {fact_num}: {prompt[:200]}...")
        
        # Call LLM with expected JSON format
        response = self._call_llm(prompt)
        
        # Debug: Log raw response
        self.logger.info(f"Raw tagging response for fact {fact_num}: {response}")
        
        try:
            # Process the response
            if isinstance(response, dict):
                # If we already have a dictionary, use it directly
                self.logger.info(f"Response is already a dictionary: {response}")
                result = response
            else:
                # Try to parse as JSON if it's a string
                try:
                    # Clean the JSON string first
                    if isinstance(response, str):
                        # Use the enhanced cleaning function
                        cleaned_response = self._clean_json_string(response)
                        self.logger.debug(f"Cleaned JSON string: {cleaned_response}")
                        
                        result = json.loads(cleaned_response)
                    else:
                        result = response
                        
                    self.logger.info(f"Successfully parsed JSON: {result}")
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse tagging response as JSON for fact {fact_num}: {str(e)}")
                    # Try to extract JSON-like structure using regex
                    try:
                        import re
                        # Look for tags array
                        tags_match = re.search(r'"tags"\s*:\s*\[(.*?)\]', response)
                        tags = []
                        if tags_match:
                            tags_str = tags_match.group(1)
                            tags = [tag.strip().strip('"\'') for tag in tags_str.split(',')]
                            
                        # Look for topic/topics
                        topics_match = re.search(r'"topics"\s*:\s*\[(.*?)\]', response)
                        topics = []
                        if topics_match:
                            topics_str = topics_match.group(1)
                            topics = [topic.strip().strip('"\'') for topic in topics_str.split(',')]
                        else:
                            topic_match = re.search(r'"topic"\s*:\s*"([^"]+)"', response)
                            if topic_match:
                                topics = [topic_match.group(1)]
                            
                        # Look for sentiment
                        sentiment_match = re.search(r'"sentiment"\s*:\s*"([^"]+)"', response)
                        sentiment = sentiment_match.group(1) if sentiment_match else "neutral"
                        
                        # Look for importance
                        importance_match = re.search(r'"importance"\s*:\s*(\d+)', response)
                        importance = int(importance_match.group(1)) if importance_match else 5
                        
                        # Look for entities
                        entities_match = re.search(r'"entities"\s*:\s*({.*?})', response)
                        entities = {"people": [], "places": [], "organizations": [], "other": []}
                        if entities_match:
                            try:
                                entities_str = entities_match.group(1)
                                entities_str = re.sub(r',\s*}', '}', entities_str)
                                entities = json.loads(entities_str)
                            except:
                                pass
                                
                        result = {
                            "tags": tags,
                            "topics": topics,
                            "sentiment": sentiment,
                            "importance": importance,
                            "entities": entities
                        }
                        self.logger.info(f"Extracted partial structure: {result}")
                    except Exception as regex_error:
                        self.logger.error(f"Failed to extract structure: {str(regex_error)}")
                        result = {
                            "tags": [],
                            "topics": [],
                            "sentiment": "neutral",
                            "importance": 5,
                            "entities": {"people": [], "places": [], "organizations": [], "other": []}
                        }
            
            # Save the complete tagging response for debugging
            self.logger.debug(f"Tagging result object for fact {fact_num}: {json.dumps(result)}")
            
            # Update the fact with tagging information
            # Handle tags (accept both "tags" and single tags in "tag")
            if "tags" in result:
                fact["tags"] = result["tags"]
                self.logger.debug(f"Added {len(result['tags'])} tags from tagging result")
            elif "tag" in result:
                fact["tags"] = [result["tag"]] if isinstance(result["tag"], str) else result["tag"]
                self.logger.debug(f"Added tag from tagging result: {result['tag']}")
            else:
                fact["tags"] = []
                self.logger.debug("No tags found in tagging result")
                
            # Handle topics (accept both "topics" array and single "topic")
            if "topics" in result:
                fact["topics"] = result["topics"]
                # Also set the singular topic field for backward compatibility
                fact["topic"] = result["topics"][0] if result["topics"] else "unknown"
                self.logger.debug(f"Added {len(result['topics'])} topics from tagging result")
            elif "topic" in result:
                fact["topic"] = result["topic"]
                fact["topics"] = [result["topic"]] if result["topic"] != "unknown" else []
                self.logger.debug(f"Added topic from tagging result: {result['topic']}")
            else:
                fact["topic"] = "unknown"
                fact["topics"] = []
                self.logger.debug("No topics found in tagging result")
                
            # Handle sentiment if available
            if "sentiment" in result:
                fact["sentiment"] = result["sentiment"]
                self.logger.debug(f"Added sentiment from tagging result: {result['sentiment']}")
                
            # Handle importance if available
            if "importance" in result:
                fact["importance"] = result["importance"]
                self.logger.debug(f"Added importance from tagging result: {result['importance']}")
                
            # Directly transfer the entire tagging result
            # This ensures we capture all fields returned by the LLM
            for key, value in result.items():
                if key not in fact and key not in ['text', 'confidence', 'source', 'temporal_info']:
                    fact[key] = value
                    self.logger.debug(f"Added additional field '{key}' from tagging result")
            
            # Handle entities field
            if "entities" in result:
                # Store the original entities array for reference
                if isinstance(result["entities"], list):
                    fact["entity_list"] = result["entities"]
                
                if isinstance(result["entities"], list):
                    # Attempt to categorize entities if they come as a list
                    categorized = {
                        "people": [],
                        "places": [],
                        "organizations": [],
                        "other": []
                    }
                    
                    for entity in result["entities"]:
                        if entity.lower() == "none" or entity.lower() == "n/a":
                            continue  # Skip placeholder entities
                            
                        # Simple heuristics for categorization
                        entity_lower = entity.lower()
                        if any(term in entity_lower for term in ["military", "va", "company", "inc", "corp", "organization"]):
                            categorized["organizations"].append(entity)
                        elif any(term in entity_lower for term in ["street", "road", "city", "state", "country", "town"]):
                            categorized["places"].append(entity)
                        elif any(term in entity_lower for term in ["mr", "mrs", "ms", "dr", "prof", "mother", "father", "dad", "mom"]):
                            categorized["people"].append(entity)
                        else:
                            # Check if it looks like a person's name (capitalized words)
                            words = entity.split()
                            if all(word[0].isupper() for word in words if word):
                                categorized["people"].append(entity)
                            else:
                                categorized["other"].append(entity)
                    
                    fact["entities"] = categorized
                else:
                    fact["entities"] = result["entities"]
            else:
                fact["entities"] = {"people": [], "places": [], "organizations": [], "other": []}
            
            # Add temporal context if available
            if "temporal_context" in result:
                fact["temporal_context"] = result["temporal_context"]
            elif "temporal_info" in fact and fact["temporal_info"]:
                # Move existing temporal_info to temporal_context for consistency
                fact["temporal_context"] = fact["temporal_info"]
            
            # Add relationships if available
            if "relationships" in result:
                fact["relationships"] = result["relationships"]
            
            # Debug: Log tagging completion
            debug_msg = f"DEBUG: Completed tagging for fact {fact_num}: {len(fact.get('tags', []))} tags identified"
            self.logger.info(debug_msg)
            if hasattr(self, 'output_mgr') and self.output_mgr:
                self.output_mgr.log_debug(debug_msg)
            
            # Debug: Log all keys in the updated fact
            self.logger.debug(f"Final fact {fact_num} keys: {list(fact.keys())}")
            
            return fact
            
        except Exception as e:
            error_msg = f"Error processing tagging response: {str(e)}"
            self.logger.error(error_msg)
            if hasattr(self, 'output_mgr') and self.output_mgr:
                self.output_mgr.log_debug(f"DEBUG ERROR: {error_msg}")
            
            # Return fact with empty tags as fallback
            fact["tags"] = []
            fact["topic"] = "unknown"
            fact["entities"] = {"people": [], "places": [], "organizations": [], "other": []}
            return fact

    def _analyze_relationships(self, facts):
        """
        Analyze relationships between facts.
        
        Args:
            facts (list): List of facts to analyze
            
        Returns:
            dict: Dictionary of relationships between facts
        """
        # Check if relationships are enabled
        if not getattr(self.config, 'relationships_enabled', True):
            self.logger.debug("Relationship analysis is disabled")
            return {}
            
        # Check if there are enough facts to analyze
        if len(facts) < 2:
            self.logger.debug("Not enough facts to analyze relationships")
            return {}
            
        self.logger.debug(f"Analyzing relationships between {len(facts)} facts")
        
        # Create a numbered list of facts for the prompt
        facts_text = ""
        for i, fact in enumerate(facts):
            if isinstance(fact, dict):
                facts_text += f"{i+1}. {fact.get('text', '')}\n"
            else:
                facts_text += f"{i+1}. {fact}\n"
        
        # Get the relationship prompt from the centralized registry
        try:
            # Use the new get_prompt method to get the formatted prompt
            prompt = self.config.get_prompt("relationship", facts_text=facts_text)
            self.logger.debug("Using relationship prompt from registry")
            
            # Check if we need pairwise analysis based on prompt parameters
            template = self.config.prompt_registry.get("relationship", self.config.model)
            if "fact1" in template.required_params and "fact2" in template.required_params:
                # Pairwise analysis
                self.logger.debug("Using pairwise relationship analysis based on prompt requirements")
                return self._analyze_pairwise_relationships(facts)
            
            # Continue with global analysis
        except (ValueError, KeyError) as e:
            self.logger.warning(f"Error getting relationship prompt from registry: {str(e)}")
            
            # Fall back to legacy prompt access methods
            prompt_template = None
            
            # Try different ways to access the prompt (legacy)
            if hasattr(self.config, 'relationship_prompt') and self.config.relationship_prompt:
                prompt_template = self.config.relationship_prompt
                self.logger.debug("Using relationship_prompt from config (legacy)")
            elif hasattr(self.config.prompts, 'RELATIONSHIP_PROMPT') and self.config.prompts.RELATIONSHIP_PROMPT:
                prompt_template = self.config.prompts.RELATIONSHIP_PROMPT
                self.logger.debug("Using RELATIONSHIP_PROMPT from config.prompts (legacy)")
            
            if not prompt_template:
                self.logger.warning("Relationship prompt template is empty. No relationships will be analyzed.")
                return {}
                
            # Check if the prompt is for pairwise analysis or global analysis
            if '{fact1}' in prompt_template and '{fact2}' in prompt_template:
                # Pairwise analysis
                self.logger.debug("Using pairwise relationship analysis")
                return self._analyze_pairwise_relationships(facts)
            
            # Format the prompt (legacy method)
            try:
                if '{facts_text}' in prompt_template:
                    prompt = prompt_template.replace('{facts_text}', facts_text)
                else:
                    prompt = f"{prompt_template}\n\nFacts:\n{facts_text}"
            except Exception as e:
                self.logger.error(f"Error formatting relationship prompt: {str(e)}")
                return {}
        
        self.logger.debug(f"Relationship prompt (first 200 chars): {prompt[:200]}...")
        
        # Call the LLM with the prompt
        response = self._call_llm(prompt)
        
        # Process the response
        relationships = {}
        if not response:
            self.logger.warning("Empty response from LLM for relationship analysis")
            return relationships
        
        # Try to parse the response as JSON
        try:
            if isinstance(response, dict) and 'relationships' in response:
                # If the response is already a dictionary with a 'relationships' key
                relationships = response['relationships']
            elif isinstance(response, str):
                # Try to parse the response as a JSON string
                import json
                parsed = json.loads(response)
                if isinstance(parsed, dict) and 'relationships' in parsed:
                    relationships = parsed['relationships']
                else:
                    relationships = parsed  # Assume the whole response is the relationships object
        except Exception as e:
            self.logger.error(f"Error parsing relationship response: {str(e)}")
            
        self.logger.debug(f"Analyzed {len(relationships)} relationships")
        return relationships
            
    def _analyze_pairwise_relationships(self, facts):
        """
        Analyze relationships between facts in a pairwise manner.
        
        Args:
            facts (list): List of facts to analyze
            
        Returns:
            dict: Dictionary of relationships between facts
        """
        if len(facts) < 2:
            self.logger.debug("Not enough facts to analyze relationships")
            return {}
            
        # Get the relationship prompt template
        prompt_template = self.config.relationship_prompt
        
        # Initialize relationships dictionary
        relationships = {}
        
        # Limit the number of pairs to analyze to avoid excessive API calls
        max_pairs = min(100, len(facts) * (len(facts) - 1) // 2)
        
        # Create pairs of facts to analyze
        pairs = []
        chunk_ids = list(range(len(facts)))
        
        # Focus on nearby facts first
        for i in range(len(facts)):
            # Get indices of nearby chunks
            start = max(0, i - 3)
            end = min(len(facts), i + 4)
            nearby_indices = list(range(start, end))
            
            # Add a few random chunks for broader relationships
            import random
            random_indices = random.sample(range(len(chunk_ids)), min(3, len(chunk_ids)))
            comparison_indices = list(set(nearby_indices + random_indices) - {i})
            
            # Prepare chunks for comparison
            for j in comparison_indices:
                if i < j:  # Avoid duplicates
                    pairs.append((i, j))
        
        # Shuffle and limit pairs
        random.shuffle(pairs)
        pairs = pairs[:max_pairs]
        
        # Analyze each pair
        for i, j in pairs:
            fact1 = facts[i]
            fact2 = facts[j]
            
            # Format the prompt
            try:
                if isinstance(fact1, dict):
                    fact1_text = fact1.get('text', '')
                else:
                    fact1_text = str(fact1)
                    
                if isinstance(fact2, dict):
                    fact2_text = fact2.get('text', '')
                else:
                    fact2_text = str(fact2)
                    
                prompt = prompt_template.replace('{fact1}', fact1_text).replace('{fact2}', fact2_text)
                
                self.logger.debug(f"Relationship prompt for facts {i} and {j} (first 100 chars): {prompt[:100]}...")
            except Exception as e:
                self.logger.error(f"Error formatting relationship prompt for facts {i} and {j}: {str(e)}")
                continue
                
            # Call the LLM with the prompt
            response = self._call_llm(prompt)
            
            # Process the response
            try:
                # Check if there's a relationship
                if isinstance(response, dict) and 'relationship_type' in response:
                    relationship_type = response['relationship_type']
                    confidence = response.get('confidence', 0.5)
                    
                    # Add the relationship if confidence is high enough
                    if confidence > 0.3:
                        # Add relationship from fact1 to fact2
                        if i not in relationships:
                            relationships[i] = {}
                        relationships[i][j] = {
                            'type': relationship_type,
                            'confidence': confidence
                        }
                        
                        # Add reverse relationship from fact2 to fact1
                        if relationship_type in ['contradicts', 'supports', 'related']:
                            if j not in relationships:
                                relationships[j] = {}
                            relationships[j][i] = {
                                'type': relationship_type,
                                'confidence': confidence
                            }
            except Exception as e:
                self.logger.error(f"Error analyzing relationship between facts {i} and {j}: {str(e)}")
                
        self.logger.debug(f"Analyzed {len(relationships)} pairwise relationships")
        return relationships

    def _track_transcript_positions(self, facts: List[str], windows: List[Tuple[str, int, int]]) -> List[Dict]:
        """
        Track which parts of the transcript each fact came from
        
        Args:
            facts: List of atomic facts
            windows: List of window tuples (text, start_token, end_token)
            
        Returns:
            List of dictionaries with transcript position information for each fact
        """
        # Debug: Log transcript position tracking start
        debug_msg = f"DEBUG: Starting transcript position tracking for {len(facts)} facts"
        self.logger.info(debug_msg)
        if hasattr(self, 'output_mgr') and self.output_mgr:
            self.output_mgr.log_debug(debug_msg)
        
        if not self.config.track_transcript_positions:
            return [{"windows": []} for _ in facts]
        
        # For each fact, find which windows it likely came from
        fact_positions = []
        
        for fact_idx, fact in enumerate(facts):
            fact_windows = []
            fact_lower = fact.lower()
            
            # Tokenize the fact for comparison
            fact_tokens = set(self.encoder.encode(fact_lower))
            
            for window_idx, (window_text, start_token, end_token) in enumerate(windows, 1):
                window_lower = window_text.lower()
                window_tokens = set(self.encoder.encode(window_lower))
                
                # Calculate token overlap
                token_overlap = len(fact_tokens.intersection(window_tokens))
                token_overlap_ratio = token_overlap / len(fact_tokens)
                
                # Check for significant overlap (more than 30% of fact tokens found in window)
                if token_overlap_ratio > 0.3:
                    # Find the specific text spans that match
                    text_spans = []
                    words = [w for w in fact_lower.split() if len(w) > 4]  # Only check significant words
                    
                    for word in words:
                        for match in re.finditer(re.escape(word), window_lower):
                            text_spans.append({
                                "start": match.start(),
                                "end": match.end(),
                                "text": window_text[match.start():match.end()]
                            })
                    
                    if text_spans:
                        fact_windows.append({
                            "window_num": window_idx,
                            "token_range": [start_token, end_token],
                            "overlap_ratio": round(token_overlap_ratio, 2),
                            "text_spans": text_spans,
                            "window_excerpt": window_text[
                                max(0, min(s["start"] for s in text_spans) - 50):
                                min(len(window_text), max(s["end"] for s in text_spans) + 50)
                            ]
                        })
            
            fact_positions.append({
                "fact_index": fact_idx + 1,
                "windows": sorted(fact_windows, key=lambda w: w["overlap_ratio"], reverse=True)
            })
            
            # Debug: Log progress periodically
            if (fact_idx + 1) % 10 == 0 or fact_idx == len(facts) - 1:
                debug_msg = f"DEBUG: Processed transcript positions for {fact_idx + 1}/{len(facts)} facts"
                self.logger.info(debug_msg)
                if hasattr(self, 'output_mgr') and self.output_mgr:
                    self.output_mgr.log_debug(debug_msg)
        
        return fact_positions

    def _estimate_completion_time(self, current_step: int, total_steps: int, elapsed_time: float) -> str:
        """
        Estimate time to completion based on current progress
        
        Args:
            current_step: Current step number
            total_steps: Total number of steps
            elapsed_time: Time elapsed so far in seconds
            
        Returns:
            String with estimated time to completion
        """
        if current_step == 0:
            return "Unknown"
        
        # Calculate estimated time to completion
        time_per_step = elapsed_time / current_step
        remaining_steps = total_steps - current_step
        estimated_remaining_time = time_per_step * remaining_steps
        
        # Format as human-readable time
        eta = timedelta(seconds=int(estimated_remaining_time))
        
        return str(eta)
    
    def _save_fact_to_file(self, fact: Dict, chunk_id: str, output_dir: str) -> str:
        """
        Save a single fact to its own JSON and MD files.
        
        Args:
            fact: The fact dictionary to save
            chunk_id: ID of the parent chunk
            output_dir: Base output directory
            
        Returns:
            Path to the created fact JSON file
        """
        # Create fact-specific directories
        facts_dir = os.path.join(output_dir, "facts")
        os.makedirs(facts_dir, exist_ok=True)
        
        # Generate a unique filename using the fact ID
        fact_id = fact.get("id", f"fact_{uuid.uuid4().hex[:8]}")
        if "id" not in fact:
            fact["id"] = fact_id
            
        # Add reference to parent chunk
        fact["parent_chunk"] = chunk_id
        
        # Debug: Log the fact being saved
        self.logger.debug(f"Saving fact {fact_id} with keys: {list(fact.keys())}")
        
        # Save JSON version
        json_path = os.path.join(facts_dir, f"{fact_id}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(fact, f, indent=2, ensure_ascii=False)
            
        # Save MD version
        md_path = os.path.join(facts_dir, f"{fact_id}.md")
        with open(md_path, 'w', encoding='utf-8') as f:
            # Write YAML frontmatter
            f.write("---\n")
            f.write(f"id: {fact_id}\n")
            f.write(f"parent_chunk: {chunk_id}\n")
            f.write(f"confidence: {fact.get('confidence', 0.0)}\n")
            f.write(f"source: {fact.get('source', 'unknown')}\n")
            f.write(f"created_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            # Write tags section
            if fact.get('tags'):
                f.write("tags:\n")
                for tag in fact['tags']:
                    f.write(f"  - {tag}\n")
            
            # Write topic and topics
            if fact.get('topic'):
                f.write(f"topic: {fact['topic']}\n")
            if fact.get('topics'):
                f.write("topics:\n")
                for topic in fact['topics']:
                    f.write(f"  - {topic}\n")
            
            # Write sentiment if available
            if fact.get('sentiment'):
                f.write(f"sentiment: {fact['sentiment']}\n")
                
            # Write importance if available
            if fact.get('importance'):
                f.write(f"importance: {fact.get('importance')}\n")
                
            f.write("---\n\n")
            
            # Write fact content
            f.write(f"# {fact_id}\n\n")
            f.write("## Content\n\n")
            f.write(f"{fact.get('text', '')}\n\n")
            
            # Write metadata sections
            if fact.get('temporal_info') or fact.get('temporal_context'):
                f.write("## Temporal Context\n\n")
                f.write(f"{fact.get('temporal_context', fact.get('temporal_info', ''))}\n\n")
                
            if fact.get('entities'):
                f.write("## Entities\n\n")
                entities = fact['entities']
                if isinstance(entities, list):
                    # If entities is a list, write them all under "other"
                    f.write("**other:**\n")
                    for entity in entities:
                        f.write(f"- {entity}\n")
                    f.write("\n")
                elif isinstance(entities, dict):
                    # If entities is a dictionary, write each category
                    for entity_type, entity_list in entities.items():
                        if entity_list:
                            f.write(f"**{entity_type}:**\n")
                            for entity in entity_list:
                                f.write(f"- {entity}\n")
                    f.write("\n")
                
            # Write tags section in markdown body
            if fact.get('tags'):
                f.write("## Tags\n\n")
                for tag in fact['tags']:
                    f.write(f"- {tag}\n")
                f.write("\n")
                
            # Write topics section in markdown body
            if fact.get('topics'):
                f.write("## Topics\n\n")
                for topic in fact['topics']:
                    f.write(f"- {topic}\n")
                f.write("\n")
            elif fact.get('topic') and fact.get('topic') != "unknown":
                f.write("## Topic\n\n")
                f.write(f"{fact['topic']}\n\n")
                
            # Write sentiment section if available
            if fact.get('sentiment'):
                f.write("## Sentiment\n\n")
                f.write(f"{fact['sentiment']}\n\n")
                
            # Write importance section if available
            if fact.get('importance'):
                f.write("## Importance\n\n")
                f.write(f"{fact['importance']}/10\n\n")
                
            if fact.get('transcript_positions'):
                f.write("## Source Context\n\n")
                f.write("<details>\n<summary>Transcript Positions</summary>\n\n")
                f.write("```json\n")
                f.write(json.dumps(fact['transcript_positions'], indent=2))
                f.write("\n```\n</details>\n")
        
        return json_path

    def _create_chunk_markdown(self, chunk_data: Dict, output_path: str) -> None:
        """
        Create a markdown file for a chunk with its facts and metadata.
        
        Args:
            chunk_data: Dictionary containing chunk information
            output_path: Path where to save the markdown file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            # Write YAML frontmatter
            f.write("---\n")
            f.write(f"chunk_id: {chunk_data['chunk_id']}\n")
            f.write(f"source_file: {chunk_data['source_file']}\n")
            f.write(f"window_num: {chunk_data['window_num']}\n")
            f.write(f"total_windows: {chunk_data['total_windows']}\n")
            f.write(f"token_range: {chunk_data['token_range']}\n")
            f.write(f"processing_time: {chunk_data['processing_time']:.2f}\n")
            f.write("---\n\n")
            
            # Write chunk content
            f.write(f"# Chunk {chunk_data['chunk_id']}\n\n")
            
            # Write original text
            f.write("## Original Text\n\n")
            f.write("```text\n")
            f.write(chunk_data['text'])
            f.write("\n```\n\n")
            
            # Write facts section
            f.write("## Facts\n\n")
            if chunk_data['facts']:
                for fact_id in chunk_data['facts']:
                    f.write(f"- [{fact_id}](../facts/{fact_id}.md)\n")
            else:
                f.write("No facts extracted from this chunk.\n")
            
            # Write metadata section
            f.write("\n## Metadata\n\n")
            f.write("```json\n")
            metadata = {
                "source_file": chunk_data['source_file'],
                "window_num": chunk_data['window_num'],
                "total_windows": chunk_data['total_windows'],
                "token_range": chunk_data['token_range'],
                "processing_time": f"{chunk_data['processing_time']:.2f}s"
            }
            f.write(json.dumps(metadata, indent=2))
            f.write("\n```\n")

    def process(self, text: str, source_file: str = None) -> List[Dict]:
        """
        Process text into atomic facts and relationships.
        
        Args:
            text: The text to process
            source_file: Path to the source file (optional)
            
        Returns:
            List of enhanced chunks with atomic facts
        """
        start_time = time.time()
        self.source_file = source_file or "unknown_source"
        
        # Log processing start
        self.logger.info(f"Starting processing of text ({len(text)} chars) from {self.source_file}")
        self.logger.info(f"Using session ID: {self.session_id}")
        self.logger.info(f"Output directories:")
        self.logger.info(f"  - Chunks: {self.chunks_dir}")
        self.logger.info(f"  - Facts: {self.facts_dir}")
        self.logger.info(f"  - Logs: {self.logs_dir}")
        
        # Split text into windows
        windows = self._create_windows(text)
        self.logger.info(f"Created {len(windows)} windows for processing")
        
        # Initialize tracking variables
        all_facts = []
        window_facts_map = {}
        total_facts = 0
        facts_per_second = 0
        
        # Process each window
        for i, (window_text, start_token, end_token) in enumerate(windows, 1):
            window_start_time = time.time()
            window_num = i
            
            # Extract facts from window
            try:
                facts = self._extract_facts_from_window(window_text, window_num)
                
                # Track transcript positions
                fact_positions = self._track_transcript_positions(
                    [f.get("text", "") for f in facts],
                    [(window_text, start_token, end_token)]
                )
                
                # Update facts with position information
                for fact, positions in zip(facts, fact_positions):
                    fact["transcript_positions"] = positions
                
                # Save facts and update tracking
                all_facts.extend(facts)
                window_facts_map[window_num] = facts
                total_facts = len(all_facts)
                
                # Calculate metrics
                window_elapsed = time.time() - window_start_time
                total_elapsed = time.time() - start_time
                facts_per_second = total_facts / total_elapsed if total_elapsed > 0 else 0
                
                # Log progress
                self.logger.info(f"Window {window_num}/{len(windows)}: Extracted {len(facts)} facts")
                self.logger.info(f"Progress: {total_facts} total facts, {facts_per_second:.2f} facts/s")
                
                # Save window results
                chunk_id = f"chunk_{window_num:03d}"
                chunk_data = {
                    "chunk_id": chunk_id,
                    "source_file": self.source_file,
                    "text": window_text,
                    "facts": [],  # Will store fact IDs
                    "window_num": window_num,
                    "total_windows": len(windows),
                    "token_range": [start_token, end_token],
                    "processing_time": window_elapsed
                }
                
                # Process and save each fact
                for fact in facts:
                    fact_path = self._save_fact_to_file(fact, chunk_id, self.facts_dir)
                    chunk_data["facts"].append(fact["id"])
                
                # Save chunk data
                chunk_json_path = os.path.join(self.chunks_dir, f"{chunk_id}.json")
                with open(chunk_json_path, 'w', encoding='utf-8') as f:
                    json.dump(chunk_data, f, indent=2, ensure_ascii=False)
                
                # Create markdown version
                chunk_md_path = os.path.join(self.chunks_dir, f"{chunk_id}.md")
                self._create_chunk_markdown(chunk_data, chunk_md_path)
                
            except Exception as e:
                self.logger.error(f"Error processing window {window_num}: {str(e)}")
                self.logger.exception("Full traceback:")
                continue
        
        # Apply tagging if enabled (always enabled by default)
        tagging_enabled = getattr(self.config, 'tagging_enabled', True)
        if tagging_enabled:
            self.logger.info(f"Applying tagging to {len(all_facts)} facts")
            tagged_facts = []
            for i, fact in enumerate(all_facts):
                try:
                    tagged_fact = self._tag_fact(fact, i+1)
                    # Update the fact file with tagged information
                    if "id" in tagged_fact:
                        # Get the fact's parent chunk
                        parent_chunk = tagged_fact.get("parent_chunk", "unknown")
                        
                        # Print the keys in tagged_fact for debugging
                        self.logger.debug(f"Tagged fact {i+1} has keys: {list(tagged_fact.keys())}")
                        
                        # Update the fact files with the new metadata
                        self._save_fact_to_file(tagged_fact, parent_chunk, self.facts_dir)
                        
                    tagged_facts.append(tagged_fact)
                except Exception as e:
                    self.logger.error(f"Error tagging fact {i+1}: {str(e)}")
                    self.logger.exception("Full tagging error traceback:")
                    tagged_facts.append(fact)  # Keep original fact if tagging fails
            all_facts = tagged_facts
            self.logger.info(f"Completed tagging for {len(tagged_facts)} facts")
        else:
            self.logger.info("Tagging is disabled, skipping tagging step")
        
        # Process relationships between chunks
        try:
            self.logger.debug("Starting chunk relationship analysis")
            chunk_relationships = self._analyze_chunk_relationships(windows, window_facts_map)
            
            # Save relationships to file
            relationships_path = os.path.join(self.output_base_dir, "relationships.json")
            self.logger.debug(f"Attempting to save relationships to {relationships_path}")
            try:
                self.logger.debug(f"Relationships data before serialization: {json.dumps(chunk_relationships, indent=2)[:500]}...")
                with open(relationships_path, 'w', encoding='utf-8') as f:
                    json.dump(chunk_relationships, f, indent=2, ensure_ascii=False)
                self.logger.debug("Successfully saved relationships to file")
            except json.JSONDecodeError as je:
                self.logger.error(f"JSON encoding error while saving relationships: {str(je)}")
                self.logger.error(f"Problematic data structure: {chunk_relationships}")
            except Exception as e:
                self.logger.error(f"Error saving relationships to file: {str(e)}")
                
        except Exception as e:
            self.logger.error(f"Error analyzing chunk relationships: {str(e)}")
            self.logger.error(f"Full error context: {str(e.__class__.__name__)}")
            self.logger.exception("Full traceback:")
        
        # Log completion
        total_time = time.time() - start_time
        self.logger.info(f"Processing complete in {total_time:.2f}s")
        self.logger.info(f"Extracted {total_facts} facts across {len(windows)} windows")
        self.logger.info(f"Average rate: {facts_per_second:.2f} facts/s")
        
        return all_facts

    def _extract_facts_from_window(self, window_text: str, window_num: int) -> List[Dict]:
        """
        Extract atomic facts from a window of text.
        
        Args:
            window_text: The text window to extract facts from
            window_num: The window number
            
        Returns:
            A list of atomic facts
        """
        self.logger.info(f"Extracting facts from window {window_num}")
        
        # First pass: Extract initial facts
        first_pass_facts = self._extract_atomic_facts_first_pass(window_text, window_num)
        
        # Check if we got any facts
        if not first_pass_facts:
            self.logger.warning(f"No facts extracted in first pass for window {window_num}")
            # Create a default fact to avoid empty results
            default_fact = {
                "text": f"No facts could be extracted from window {window_num}.",
                "confidence": 0.0,
                "source": "system",
                "temporal_info": "",
                "entities": []
            }
            return [default_fact]
        
        # Log the number of facts extracted in the first pass
        self.logger.info(f"Extracted {len(first_pass_facts)} facts in first pass for window {window_num}")
        
        # Second pass: Refine the facts
        second_pass_facts = self._extract_atomic_facts_second_pass(window_text, first_pass_facts, window_num)
        
        if not second_pass_facts:
            self.logger.warning(f"No facts extracted in second pass for window {window_num}, using first pass facts")
            second_pass_facts = first_pass_facts
        
        # Log the number of facts extracted in the second pass
        self.logger.info(f"Extracted {len(second_pass_facts)} facts in second pass for window {window_num}")
        
        # Validate and normalize the facts
        validated_facts = []
        for i, fact in enumerate(second_pass_facts):
            # Skip facts that are just placeholders or headers
            if isinstance(fact, dict) and fact.get('text', '').lower().startswith(('here are', 'atomic fact', 'fact:')):
                self.logger.warning(f"Skipping placeholder fact: {fact.get('text', '')[:50]}...")
                continue
            
            # Convert string facts to dictionaries
            if isinstance(fact, str):
                fact = {
                    'text': fact.strip(),
                    'confidence': 1.0,
                    'source': 'extraction',
                    'temporal_info': '',
                    'entities': []
                }
            
            # Ensure the fact has the required fields
            if not isinstance(fact, dict):
                self.logger.warning(f"Fact {i} is not a dictionary: {fact}")
                continue
            
            # Add missing fields with default values
            for field, default_value in [
                ('text', ''),
                ('confidence', 1.0),
                ('source', 'extraction'),
                ('temporal_info', ''),
                ('entities', [])
            ]:
                if field not in fact:
                    fact[field] = default_value
                    self.logger.warning(f"Added missing field '{field}' to fact {i}")
            
            # Validate the fact
            if self._validate_atomic_fact(fact):
                validated_facts.append(fact)
            else:
                self.logger.warning(f"Fact {i} failed validation: {fact}")
        
        # If no facts passed validation, create a default fact
        if not validated_facts:
            self.logger.warning(f"No facts passed validation for window {window_num}, creating default fact")
            default_fact = {
                "text": f"No valid facts could be extracted from window {window_num}.",
                "confidence": 0.0,
                "source": "system",
                "temporal_info": "",
                "entities": []
            }
            validated_facts = [default_fact]
        
        # Log the number of validated facts
        self.logger.info(f"Validated {len(validated_facts)} facts for window {window_num}")
        
        return validated_facts

    def _validate_atomic_fact(self, fact: Dict) -> bool:
        """
        Validate that a fact has all required fields and proper types.
        
        Args:
            fact: The fact to validate
            
        Returns:
            True if the fact is valid, False otherwise
        """
        # Check if fact is a dictionary
        if not isinstance(fact, dict):
            self.logger.warning(f"Fact is not a dictionary: {fact}")
            return False
        
        # Check for required fields
        required_fields = ['text', 'confidence', 'source', 'temporal_info', 'entities']
        for field in required_fields:
            if field not in fact:
                self.logger.warning(f"Fact missing required field '{field}': {fact}")
                # Add default values for missing fields
                if field == 'text':
                    fact[field] = "No text provided"
                elif field == 'confidence':
                    fact[field] = 0.5
                elif field == 'source':
                    fact[field] = "unknown"
                elif field == 'temporal_info':
                    fact[field] = ""
                elif field == 'entities':
                    fact[field] = []
                else:
                    return False
        
        # Check that text is a non-empty string
        if not isinstance(fact['text'], str) or not fact['text'].strip():
            self.logger.warning(f"Fact has invalid or empty text: {fact}")
            return False
        
        # Check that confidence is a number between 0 and 1
        if not isinstance(fact['confidence'], (int, float)):
            try:
                fact['confidence'] = float(fact['confidence'])
            except (ValueError, TypeError):
                self.logger.warning(f"Fact has invalid confidence value: {fact}")
                fact['confidence'] = 0.5
        
        # Ensure confidence is between 0 and 1
        if fact['confidence'] < 0 or fact['confidence'] > 1:
            self.logger.warning(f"Fact has confidence value outside range [0,1]: {fact}")
            fact['confidence'] = max(0, min(1, fact['confidence']))
        
        # Check that source is a string
        if not isinstance(fact['source'], str):
            self.logger.warning(f"Fact has invalid source: {fact}")
            fact['source'] = str(fact['source'])
        
        # Check that temporal_info is a string
        if not isinstance(fact['temporal_info'], str):
            self.logger.warning(f"Fact has invalid temporal_info: {fact}")
            fact['temporal_info'] = str(fact['temporal_info'])
        
        # Check that entities is a list or convert it to a list
        if not isinstance(fact['entities'], list):
            if isinstance(fact['entities'], dict):
                # Convert dict to list if it has the right structure
                if 'persons' in fact['entities'] or 'locations' in fact['entities'] or 'organizations' in fact['entities']:
                    entities_list = []
                    for entity_type, entities in fact['entities'].items():
                        if isinstance(entities, list):
                            entities_list.extend(entities)
                    fact['entities'] = entities_list
                else:
                    self.logger.warning(f"Fact has invalid entities dictionary: {fact}")
                    fact['entities'] = list(fact['entities'].values())
            else:
                self.logger.warning(f"Fact has invalid entities: {fact}")
                try:
                    # Try to convert to list if it's a string
                    if isinstance(fact['entities'], str):
                        if fact['entities'].startswith('[') and fact['entities'].endswith(']'):
                            # Looks like a JSON array string
                            try:
                                fact['entities'] = json.loads(fact['entities'])
                            except json.JSONDecodeError:
                                fact['entities'] = [e.strip() for e in fact['entities'][1:-1].split(',') if e.strip()]
                        else:
                            fact['entities'] = [e.strip() for e in fact['entities'].split(',') if e.strip()]
                    else:
                        fact['entities'] = []
                except Exception as e:
                    self.logger.warning(f"Error converting entities to list: {str(e)}")
                    fact['entities'] = []
        
        # Ensure all entities are strings
        fact['entities'] = [str(e) for e in fact['entities'] if e]
        
        # Check for placeholder text
        placeholder_patterns = [
            r'^fact\s*\d*\s*:',
            r'^atomic\s*fact\s*\d*\s*:',
            r'^here\s+are\s+the\s+facts',
            r'^extracted\s+facts',
            r'^no\s+facts\s+found',
            r'^i\s+couldn\'t\s+extract',
            r'^sorry,\s+i\s+couldn\'t',
        ]
        
        for pattern in placeholder_patterns:
            if re.search(pattern, fact['text'].lower()):
                self.logger.warning(f"Fact contains placeholder text: {fact['text']}")
                return False
        
        # All checks passed
        return True

    def _analyze_chunk_relationships(self, windows: List[Tuple[str, int, int]], window_facts_map: Dict[int, List[Dict]]) -> Dict[str, List[Dict]]:
        """Analyze relationships between chunks and their facts."""
        self.logger.debug(f"Starting chunk relationship analysis with {len(windows)} windows")
        chunk_ids = list(window_facts_map.keys())
        relationships = {}
        
        for i, chunk_id in enumerate(chunk_ids):
            self.logger.debug(f"Processing relationships for chunk {chunk_id}")
            # Get nearby chunks (2 before and 2 after)
            nearby_indices = list(range(max(0, i-2), min(len(chunk_ids), i+3)))
            
            # Add a few random chunks for broader relationships
            import random
            random_indices = random.sample(range(len(chunk_ids)), min(3, len(chunk_ids)))
            comparison_indices = list(set(nearby_indices + random_indices) - {i})
            
            # Prepare chunks for comparison
            chunks_to_compare = []
            for idx in comparison_indices:
                compare_id = chunk_ids[idx]
                chunks_to_compare.append({
                    "chunk_id": compare_id,
                    "facts": window_facts_map[compare_id]
                })
            
            # Skip if no chunks to compare
            if not chunks_to_compare:
                return
                
            # Get facts for current chunk
            current_facts = window_facts_map[chunk_id]
            
            # Skip if no facts in current chunk
            if not current_facts:
                return
                
            # Analyze relationships
            chunk_relationships = self._analyze_relationships(current_facts)
            
            # Store relationships
            relationships[str(chunk_id)] = {
                "chunk_id": chunk_id,
                "relationships": chunk_relationships
            }
            
        return relationships 
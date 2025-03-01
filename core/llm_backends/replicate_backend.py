import os
import time
import json
import logging
import re
import csv
import datetime
import sys
import atexit
from typing import Dict, Any, Optional, List, Iterator, Union
from dotenv import load_dotenv

import replicate
import tiktoken

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Constants for cost calculation
INPUT_COST_PER_MILLION = 0.05  # $0.05 per 1M input tokens
OUTPUT_COST_PER_MILLION = 0.25  # $0.25 per 1M output tokens

# Initialize tokenizer for counting tokens
try:
    # Use cl100k_base tokenizer (similar to what most models use)
    tokenizer = tiktoken.get_encoding("cl100k_base")
except:
    logger.warning("tiktoken not available, token counting will be estimated")
    tokenizer = None

class CostTracker:
    """
    Tracks costs for Replicate API calls across a session.
    """
    
    def __init__(self, csv_path="logs/replicate_costs.csv"):
        """
        Initialize the cost tracker.
        
        Args:
            csv_path: Path to the CSV file for cost logging
        """
        self.session_start = datetime.datetime.now()
        self.session_id = self.session_start.strftime("%Y%m%d_%H%M%S")
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.call_count = 0
        
        # Get the command that was used to start the script
        self.command = " ".join(sys.argv)
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        # Check if the CSV file exists
        file_exists = os.path.isfile(csv_path)
        
        # Open the CSV file in append mode
        self.csv_path = csv_path
        self.csv_file = open(csv_path, 'a', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        # Write header if the file is new
        if not file_exists:
            self.csv_writer.writerow([
                'Session ID', 
                'Start Time', 
                'End Time', 
                'Command', 
                'Input Tokens', 
                'Output Tokens', 
                'Input Cost ($)', 
                'Output Cost ($)', 
                'Total Cost ($)',
                'Call Count'
            ])
        
        # Register the write_summary method to be called when the program exits
        atexit.register(self.write_summary)
    
    def add_cost(self, input_tokens, output_tokens):
        """
        Add cost for a single API call.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
        """
        input_cost = (input_tokens / 1_000_000) * INPUT_COST_PER_MILLION
        output_cost = (output_tokens / 1_000_000) * OUTPUT_COST_PER_MILLION
        
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += (input_cost + output_cost)
        self.call_count += 1
    
    def write_summary(self):
        """
        Write the cost summary to the CSV file and close it.
        """
        # Calculate costs
        input_cost = (self.total_input_tokens / 1_000_000) * INPUT_COST_PER_MILLION
        output_cost = (self.total_output_tokens / 1_000_000) * OUTPUT_COST_PER_MILLION
        
        # Get the end time
        end_time = datetime.datetime.now()
        
        # Write to CSV
        self.csv_writer.writerow([
            self.session_id,
            self.session_start.strftime("%Y-%m-%d %H:%M:%S"),
            end_time.strftime("%Y-%m-%d %H:%M:%S"),
            self.command,
            self.total_input_tokens,
            self.total_output_tokens,
            f"{input_cost:.6f}",
            f"{output_cost:.6f}",
            f"{self.total_cost:.6f}",
            self.call_count
        ])
        
        # Close the CSV file
        self.csv_file.close()
        
        # Print summary to console
        logger.info(f"Replicate API Usage Summary:")
        logger.info(f"  Session ID: {self.session_id}")
        logger.info(f"  Duration: {(end_time - self.session_start).total_seconds():.2f} seconds")
        logger.info(f"  API Calls: {self.call_count}")
        logger.info(f"  Input Tokens: {self.total_input_tokens:,} (${input_cost:.6f})")
        logger.info(f"  Output Tokens: {self.total_output_tokens:,} (${output_cost:.6f})")
        logger.info(f"  Total Cost: ${self.total_cost:.6f}")
        logger.info(f"  Cost details saved to: {self.csv_path}")

def count_tokens(text):
    """Count the number of tokens in a text string."""
    if tokenizer:
        return len(tokenizer.encode(text))
    else:
        # Rough estimate: ~4 characters per token
        return len(text) // 4

class ReplicateBackend:
    """
    Backend for calling Replicate API to access Mistral and other models.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Replicate backend.
        
        Args:
            api_key: Replicate API key. If not provided, will look for REPLICATE_API_TOKEN env var.
        """
        # Try to get API key from parameter, then from environment variable
        self.api_key = api_key or os.environ.get("REPLICATE_API_TOKEN")
        if not self.api_key:
            raise ValueError(
                "Replicate API key not found. Please provide it as an argument or "
                "set the REPLICATE_API_TOKEN environment variable in your .env file."
            )
        
        # Set the API token for the replicate client
        os.environ["REPLICATE_API_TOKEN"] = self.api_key
        
        # Initialize cost tracker
        self.cost_tracker = CostTracker()
        
        # Model mappings for easy reference
        self.model_mappings = {
            "mistral:instruct": "mistralai/mistral-7b-instruct-v0.2",
            "mistral:medium": "mistralai/mistral-medium",
            "mistral:large": "mistralai/mistral-large-latest",
            "llama3:8b": "meta/meta-llama-3-8b-instruct",
            "llama3:70b": "meta/meta-llama-3-70b-instruct",
            "claude:3-haiku": "anthropic/claude-3-haiku:1.0",
            "claude:3-sonnet": "anthropic/claude-3-sonnet:1.0",
            "claude:3-opus": "anthropic/claude-3-opus:1.0",
        }
        
        # Default system prompts for different model families
        self.default_system_prompts = {
            "llama3": "You are a helpful assistant that extracts factual information from text.",
            "mistral": "You are a helpful assistant.",
            "claude": "You are a helpful assistant."
        }
    
    def get_model_id(self, model_name: str) -> str:
        """
        Get the Replicate model ID for a given model name.
        
        Args:
            model_name: The model name (e.g., "mistral:instruct")
            
        Returns:
            The Replicate model ID
        """
        if model_name in self.model_mappings:
            return self.model_mappings[model_name]
        
        # If not in mappings, assume it's already a valid Replicate model ID
        return model_name
    
    def get_model_family(self, model_id: str) -> str:
        """
        Determine the model family from the model ID.
        
        Args:
            model_id: The Replicate model ID
            
        Returns:
            The model family (e.g., "llama3", "mistral", "claude")
        """
        model_id_lower = model_id.lower()
        if "llama-3" in model_id_lower or "llama3" in model_id_lower:
            return "llama3"
        elif "mistral" in model_id_lower:
            return "mistral"
        elif "claude" in model_id_lower:
            return "claude"
        else:
            return "generic"
    
    def get_default_system_prompt(self, model_id: str) -> str:
        """
        Get the default system prompt for a model family.
        
        Args:
            model_id: The Replicate model ID
            
        Returns:
            The default system prompt for the model family
        """
        model_family = self.get_model_family(model_id)
        return self.default_system_prompts.get(model_family, "You are a helpful assistant.")
    
    def call(self, 
             model: str, 
             prompt: str, 
             temperature: float = 0.7, 
             max_tokens: int = 1000,
             system_prompt: Optional[str] = None,
             stream: bool = False,
             expected_format: str = "auto",
             **kwargs) -> Union[Dict[str, Any], Iterator[str]]:
        """
        Call the Replicate API with the given parameters.
        
        Args:
            model: Model name or ID
            prompt: The prompt to send
            temperature: Temperature parameter (0.0-1.0)
            max_tokens: Maximum number of tokens to generate
            system_prompt: System prompt to use (if None, will use default for model family)
            stream: Whether to stream the response
            expected_format: Expected format of the response ('json', 'list', 'text', or 'auto')
            **kwargs: Additional parameters to pass to the model
            
        Returns:
            Dict containing the response and metadata, or an iterator of response chunks if streaming
        """
        start_time = time.time()
        model_id = self.get_model_id(model)
        model_family = self.get_model_family(model_id)
        
        # Use provided system prompt or default for model family
        if system_prompt is None:
            system_prompt = self.get_default_system_prompt(model_id)
        
        logger.info(f"Calling Replicate API with model {model_id}")
        
        try:
            # Prepare input based on model type
            if model_family == "llama3":
                # Llama 3 models
                prompt_template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                
                input_params = {
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": kwargs.get("top_p", 0.95),
                    "system_prompt": system_prompt,
                    "prompt_template": prompt_template,
                    "stop_sequences": "<|end_of_text|>,<|eot_id|>",
                }
                
                # Count input tokens (including system prompt and template)
                full_prompt = prompt_template.format(system_prompt=system_prompt, prompt=prompt)
                input_tokens = count_tokens(full_prompt)
                
            elif model_family == "mistral":
                # Mistral models
                input_params = {
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_new_tokens": max_tokens,
                    "top_p": kwargs.get("top_p", 0.9),
                    "repetition_penalty": kwargs.get("repetition_penalty", 1.1),
                }
                
                # Count input tokens
                input_tokens = count_tokens(prompt)
                
            elif model_family == "claude":
                # Claude models
                full_prompt = f"<human>{prompt}</human><assistant>"
                input_params = {
                    "prompt": full_prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                
                # Count input tokens
                input_tokens = count_tokens(full_prompt)
                
            else:
                # Generic fallback
                input_params = {
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                
                # Count input tokens
                input_tokens = count_tokens(prompt)
            
            # Add any additional parameters
            for key, value in kwargs.items():
                if key not in input_params and key not in ["top_p", "repetition_penalty"]:
                    input_params[key] = value
            
            # Handle streaming vs. non-streaming
            if stream:
                # For streaming, we'll track tokens in a wrapper
                stream_iterator = replicate.stream(
                    model_id,
                    input=input_params
                )
                
                # Return a wrapped iterator that counts tokens
                def token_counting_stream():
                    output_text = ""
                    for chunk in stream_iterator:
                        chunk_str = str(chunk)
                        output_text += chunk_str
                        yield chunk
                    
                    # Count output tokens at the end
                    output_tokens = count_tokens(output_text)
                    
                    # Track cost
                    self.cost_tracker.add_cost(input_tokens, output_tokens)
                    
                return token_counting_stream()
                
            else:
                # Make the API call
                output = replicate.run(
                    model_id,
                    input=input_params
                )
                
                # Replicate returns a generator for streaming responses
                # We'll collect all the chunks into a single response
                response_chunks = []
                for chunk in output:
                    if chunk:
                        response_chunks.append(chunk)
                
                # Join all chunks
                response_text = "".join(response_chunks)
                
                # Count output tokens
                output_tokens = count_tokens(response_text)
                
                # Track cost
                self.cost_tracker.add_cost(input_tokens, output_tokens)
                
                # Format the response based on expected format
                formatted_response = self.format_response(response_text, expected_format)
                
                # Calculate elapsed time
                elapsed_time = time.time() - start_time
                
                # Add metadata to the response
                result = {
                    **formatted_response,
                    "model": model_id,
                    "elapsed_time": elapsed_time,
                    "success": True,
                    "error": None,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens
                }
                
                return result
                
        except Exception as e:
            elapsed_time = time.time() - start_time
            error_msg = f"Error calling Replicate API: {str(e)}"
            logger.error(error_msg)
            
            return {
                "response": "",
                "model": model_id,
                "elapsed_time": elapsed_time,
                "success": False,
                "error": error_msg,
                "input_tokens": input_tokens if 'input_tokens' in locals() else 0,
                "output_tokens": 0,
                "format": "error"
            }
    
    def parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse a JSON response from the model.
        
        Args:
            response_text: The text response from the model
            
        Returns:
            Parsed JSON as a dictionary, or a dictionary with the original response
        """
        if not response_text or response_text.isspace():
            logger.warning("Empty response received from LLM")
            return {"response": "", "error": "Empty response"}
        
        # Log the raw response for debugging
        logger.debug(f"Raw response to parse as JSON: {response_text[:500]}...")
        
        # Clean up the response text
        cleaned_text = response_text.strip()
        
        # Remove any markdown code block markers
        cleaned_text = re.sub(r'```(?:json)?\s*([\s\S]*?)\s*```', r'\1', cleaned_text)
        
        # Try to find JSON array or object
        json_array_match = re.search(r'\[\s*\{.*\}\s*\]', cleaned_text, re.DOTALL)
        json_object_match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
        
        if json_array_match:
            # Found a JSON array
            cleaned_text = json_array_match.group(0)
        elif json_object_match:
            # Found a JSON object
            cleaned_text = json_object_match.group(0)
        
        # Log the cleaned text
        logger.debug(f"Cleaned text for JSON parsing: {cleaned_text[:500]}...")
        
        # Try to parse the JSON
        try:
            parsed_json = json.loads(cleaned_text)
            logger.info(f"Successfully parsed JSON response")
            return parsed_json
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed: {str(e)}")
            
            # Try to repair the JSON
            fixed_text = self._repair_json(cleaned_text)
            try:
                parsed_json = json.loads(fixed_text)
                logger.info(f"Successfully parsed JSON after repair")
                return parsed_json
            except json.JSONDecodeError as repair_error:
                logger.error(f"JSON repair failed: {str(repair_error)}")
                
                # Try to extract structured data as a last resort
                structured_data = self._extract_structured_data(response_text)
                if structured_data.get("extracted"):
                    logger.info(f"Extracted structured data from response")
                    return structured_data.get("extracted", {})
                
                # If all parsing attempts fail, return a default response
                logger.error(f"All JSON parsing attempts failed, returning default response")
                return {
                    "facts": [
                        {
                            "text": "Failed to parse LLM response as JSON.",
                            "confidence": 0.0,
                            "source": "error",
                            "temporal_info": "",
                            "entities": []
                        }
                    ],
                    "error": f"JSON parsing failed: {str(e)}"
                }

    def format_response(self, response_text: str, expected_format: str = "auto") -> Dict[str, Any]:
        """
        Format and standardize the response from the LLM.
        
        Args:
            response_text: The raw text response from the LLM
            expected_format: The expected format of the response ('json', 'list', 'text', or 'auto')
            
        Returns:
            A dictionary containing the formatted response and metadata
        """
        if not response_text or response_text.isspace():
            return {"response": "", "error": "Empty response", "format": "empty"}
        
        # Log the raw response for debugging
        logger.debug(f"Raw response: {response_text[:100]}...")
        
        # Auto-detect format if not specified
        if expected_format == "auto":
            if (response_text.strip().startswith('{') and response_text.strip().endswith('}')) or \
               (response_text.strip().startswith('[') and response_text.strip().endswith(']')):
                expected_format = "json"
            elif re.search(r'^\s*\d+\.', response_text, re.MULTILINE):
                expected_format = "list"
            else:
                expected_format = "text"
        
        # Process based on expected format
        if expected_format == "json":
            # Try to parse as JSON
            try:
                # First try direct parsing
                try:
                    parsed_json = json.loads(response_text.strip())
                    return {"response": response_text, "parsed": parsed_json, "format": "json"}
                except json.JSONDecodeError:
                    # Try to extract and parse JSON from the response
                    parsed_result = self.parse_json_response(response_text)
                    
                    if "error" not in parsed_result:
                        return {"response": response_text, "parsed": parsed_result, "format": "json"}
                    
                    # If JSON parsing failed, try to repair common issues
                    fixed_text = self._repair_json(response_text)
                    try:
                        parsed_json = json.loads(fixed_text)
                        return {
                            "response": response_text, 
                            "parsed": parsed_json, 
                            "format": "json",
                            "repaired": True
                        }
                    except json.JSONDecodeError:
                        # If all JSON parsing attempts fail, extract structured data
                        structured_data = self._extract_structured_data(response_text)
                        structured_data["format"] = "extracted"
                        return structured_data
            except Exception as e:
                logger.warning(f"Failed to format JSON response: {str(e)}")
                return {"response": response_text, "error": str(e), "format": "text"}
        
        elif expected_format == "list":
            # Process as a numbered or bulleted list
            items = []
            current_item = ""
            
            for line in response_text.split('\n'):
                # Check if this line starts a new item
                if re.match(r'^\s*(\d+\.|[-•*])', line.strip()):
                    # Save the previous item if it exists
                    if current_item:
                        items.append(current_item.strip())
                    
                    # Start a new item (remove the number/bullet)
                    current_item = re.sub(r'^\s*(\d+\.|[-•*])\s*', '', line)
                else:
                    # Continue the current item
                    if current_item or line.strip():
                        current_item += " " + line.strip()
            
            # Add the last item
            if current_item:
                items.append(current_item.strip())
            
            return {"response": response_text, "items": items, "format": "list"}
        
        else:
            # Return as plain text
            return {"response": response_text, "format": "text"}

    def _repair_json(self, text: str) -> str:
        """
        Attempt to repair common JSON formatting issues.
        
        Args:
            text: The text to repair
            
        Returns:
            The repaired text
        """
        # Log the original text for debugging
        logger.debug(f"Attempting to repair JSON: {text[:100]}...")
        
        # Extract what looks like JSON
        json_match = re.search(r'({[\s\S]*}|\[[\s\S]*\])', text)
        if json_match:
            text = json_match.group(1)
        
        # Fix missing quotes around keys
        text = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', text)
        
        # Fix single quotes to double quotes
        # This is tricky because we need to avoid changing already escaped quotes
        # Simple approach: replace all single quotes with double quotes if there are more single than double
        if text.count("'") > text.count('"'):
            text = text.replace("'", '"')
        
        # Fix trailing commas in arrays and objects
        text = re.sub(r',\s*}', '}', text)
        text = re.sub(r',\s*]', ']', text)
        
        # Fix missing commas between array elements or object properties
        text = re.sub(r'}\s*{', '},{', text)
        text = re.sub(r'"\s*{', '",{', text)
        text = re.sub(r'}\s*"', '},"', text)
        text = re.sub(r']\s*{', '],{', text)
        text = re.sub(r'}\s*\[', '},\[', text)
        
        # Fix missing braces
        if text.count('{') > text.count('}'):
            text += '}'
        
        # Fix missing brackets
        if text.count('[') > text.count(']'):
            text += ']'
        
        # Log the repaired text for debugging
        logger.debug(f"Repaired JSON: {text[:100]}...")
        
        return text

    def _extract_structured_data(self, text: str) -> Dict[str, Any]:
        """
        Extract structured data from text when JSON parsing fails.
        
        Args:
            text: The text to extract data from
            
        Returns:
            A dictionary with the extracted data
        """
        result = {"response": text, "format": "extracted"}
        extracted = {}
        
        # Look for key-value pairs (Property: Value)
        for line in text.split('\n'):
            kv_match = re.match(r'^\s*([A-Za-z][A-Za-z0-9_\s]*?):\s*(.+)$', line)
            if kv_match:
                key = kv_match.group(1).strip().lower().replace(' ', '_')
                value = kv_match.group(2).strip()
                extracted[key] = value
        
        # Look for facts in a numbered list
        facts = []
        for line in text.split('\n'):
            fact_match = re.match(r'^\s*\d+\.\s*(.+)$', line)
            if fact_match:
                facts.append(fact_match.group(1).strip())
        
        if facts:
            extracted['facts'] = facts
        
        # Try to extract JSON-like structures
        json_like = []
        current_obj = {}
        for line in text.split('\n'):
            # Look for field definitions
            field_match = re.match(r'^\s*"?([a-zA-Z_][a-zA-Z0-9_]*)"?\s*:\s*(.+)$', line)
            if field_match:
                key = field_match.group(1).strip()
                value = field_match.group(2).strip().rstrip(',')
                
                # Try to parse value as JSON if it looks like an array or object
                if (value.startswith('[') and value.endswith(']')) or (value.startswith('{') and value.endswith('}')):
                    try:
                        value = json.loads(value)
                    except:
                        pass
                # Try to parse as number
                elif re.match(r'^-?\d+(\.\d+)?$', value):
                    try:
                        value = float(value) if '.' in value else int(value)
                    except:
                        pass
                # Remove quotes if it's a string
                elif (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]
                
                current_obj[key] = value
            
            # Check for object boundaries
            if line.strip() == '{':
                current_obj = {}
            elif line.strip() == '}':
                if current_obj:
                    json_like.append(current_obj)
                    current_obj = {}
        
        # Add the last object if it's not empty
        if current_obj:
            json_like.append(current_obj)
        
        if json_like:
            extracted['structured'] = json_like
        
        if extracted:
            result['extracted'] = extracted
        
        return result 
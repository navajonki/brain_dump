"""
Default implementations of chunking interfaces.

These classes provide the default implementations for the interfaces defined
in interfaces.py. They are used when no custom implementations are provided.
"""

import os
import uuid
import json
import tiktoken
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

from utils.logging import get_logger
from core.chunking.interfaces import ITokenizer, IOutputManager, ISessionProvider, IPromptManager
from config.chunking.chunking_config import ChunkingConfig
import logging


class DefaultTokenizer(ITokenizer):
    """Default implementation of the tokenizer interface using tiktoken."""
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        """Initialize the tokenizer for a specific model."""
        self.model = model
        self.encoder = tiktoken.encoding_for_model(model)
    
    def encode(self, text: str) -> List[int]:
        """Encode text into tokens."""
        return self.encoder.encode(text)
    
    def decode(self, tokens: List[int]) -> str:
        """Decode tokens into text."""
        return self.encoder.decode(tokens)
    
    def get_token_count(self, text: str) -> int:
        """Get the number of tokens in a text."""
        return len(self.encode(text))


class DefaultOutputManager(IOutputManager):
    """Default implementation of the output manager interface."""
    
    def __init__(self, logger=None):
        """Initialize the output manager."""
        self.logger = logger or get_logger(__name__)
        self.output_dirs = {}
    
    def initialize(self, session_id: str, config: Any) -> None:
        """Initialize the output manager with session and configuration."""
        # Format model name for file system
        model_name = config.model.replace("/", "_").replace(":", "_")
        
        # Set up output directories
        self.output_base_dir = os.path.join("output", model_name, session_id)
        self.output_dirs = {
            "base": self.output_base_dir,
            "chunks": os.path.join(self.output_base_dir, "chunks"),
            "facts": os.path.join(self.output_base_dir, "facts"),
            "logs": os.path.join(self.output_base_dir, "logs")
        }
        
        # Create directories
        for dir_path in self.output_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        # Set up logging files
        self.llm_interactions_log = os.path.join(self.output_dirs["logs"], "llm_interactions.log")
        self.llm_metrics_log = os.path.join(self.output_dirs["logs"], "llm_metrics.log")
        
        # Set up file handlers for logging if logger exists
        interactions_handler = logging.FileHandler(self.llm_interactions_log)
        interactions_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(interactions_handler)
        
        metrics_handler = logging.FileHandler(self.llm_metrics_log)
        metrics_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(metrics_handler)
        
        self.logger.info(f"Initialized output manager for session: {session_id}")
        self.logger.info(f"Output directory: {self.output_base_dir}")
    
    def get_output_dirs(self) -> Dict[str, str]:
        """Get output directories."""
        return self.output_dirs
    
    def write_chunk(self, chunk_id: str, chunk_data: Dict[str, Any]) -> str:
        """Write a chunk to storage and return its path."""
        filename = f"{chunk_id}.json"
        file_path = os.path.join(self.output_dirs["chunks"], filename)
        
        with open(file_path, 'w') as f:
            json.dump(chunk_data, f, indent=2)
        
        self.logger.debug(f"Wrote chunk to {file_path}")
        return file_path
    
    def write_facts(self, chunk_id: str, facts: List[Dict[str, Any]]) -> str:
        """Write facts to storage and return their path."""
        filename = f"{chunk_id}_facts.json"
        file_path = os.path.join(self.output_dirs["facts"], filename)
        
        with open(file_path, 'w') as f:
            json.dump(facts, f, indent=2)
        
        self.logger.debug(f"Wrote facts to {file_path}")
        return file_path
    
    def log_llm_interaction(self, prompt: str, response: Any) -> None:
        """Log an LLM interaction."""
        self.logger.debug(f"Prompt:\n{prompt}\n")
        self.logger.debug(f"Response:\n{response}\n")


class DefaultSessionProvider(ISessionProvider):
    """Default implementation of the session provider interface."""
    
    def get_session_id(self) -> str:
        """Generate and return a unique session ID."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{timestamp}_{uuid.uuid4().hex[:8]}"


class DefaultPromptManager(IPromptManager):
    """Default implementation of the prompt manager interface."""
    
    def __init__(self, config: ChunkingConfig):
        """Initialize the prompt manager with configuration."""
        self.config = config
    
    def get_prompt(self, prompt_name: str, **kwargs) -> str:
        """Get a formatted prompt with the given parameters."""
        # Use the config's get_prompt method
        return self.config.get_prompt(prompt_name, **kwargs)
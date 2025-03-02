"""
Interface definitions for chunking dependencies.

These interfaces define the contracts that must be satisfied by any dependency
implementations used by the TextChunker class. This allows for easy dependency
injection and testing.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union


class ITokenizer(ABC):
    """Interface for text tokenization functionality."""
    
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Encode text into tokens."""
        pass
    
    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        """Decode tokens into text."""
        pass
    
    @abstractmethod
    def get_token_count(self, text: str) -> int:
        """Get the number of tokens in a text."""
        pass


class ILLMBackend(ABC):
    """Interface for LLM interaction."""
    
    @abstractmethod
    def initialize(self, config: Any) -> None:
        """Initialize the backend with configuration."""
        pass
    
    @abstractmethod
    def call(self, model: str, prompt: str, **kwargs) -> Any:
        """Call the LLM with a prompt and return the response."""
        pass


class IOutputManager(ABC):
    """Interface for output management."""
    
    @abstractmethod
    def initialize(self, session_id: str, config: Any) -> None:
        """Initialize the output manager with session and configuration."""
        pass
    
    @abstractmethod
    def get_output_dirs(self) -> Dict[str, str]:
        """Get output directories."""
        pass
    
    @abstractmethod
    def write_chunk(self, chunk_id: str, chunk_data: Dict[str, Any]) -> str:
        """Write a chunk to storage and return its path."""
        pass
    
    @abstractmethod
    def write_facts(self, chunk_id: str, facts: List[Dict[str, Any]]) -> str:
        """Write facts to storage and return their path."""
        pass
    
    @abstractmethod
    def log_llm_interaction(self, prompt: str, response: Any) -> None:
        """Log an LLM interaction."""
        pass


class ISessionProvider(ABC):
    """Interface for session management."""
    
    @abstractmethod
    def get_session_id(self) -> str:
        """Generate and return a unique session ID."""
        pass


class IPromptManager(ABC):
    """Interface for prompt management."""
    
    @abstractmethod
    def get_prompt(self, prompt_name: str, **kwargs) -> str:
        """Get a formatted prompt with the given parameters."""
        pass
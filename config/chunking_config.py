from dataclasses import dataclass
from typing import Literal, Optional
from config.chunking.prompts import (
    FIRST_PASS_PROMPT, 
    SECOND_PASS_PROMPT, 
    GLOBAL_CHECK_PROMPT,
    TAGGING_PROMPT,
    RELATIONSHIP_PROMPT
)

@dataclass
class ChunkingConfig:
    # Token limits
    max_tokens: int = 300
    
    # LLM settings
    llm_backend: Literal["openai", "ollama"] = "openai"
    model: str = "gpt-3.5-turbo"  # or "mistral" for ollama
    
    # Ollama specific settings
    ollama_url: str = "http://localhost:11434/api/generate"
    
    # OpenAI specific settings
    temperature: float = 0
    max_response_tokens: int = 10
    
    # Chunking behavior
    use_semantic_chunking: bool = True  # If False, uses simple token-based chunking
    
    # Sliding window parameters for atomic chunking (v2)
    window_size: int = 1000  # Size of sliding window in tokens
    overlap_size: int = 100  # Size of overlap between windows in tokens
    
    # Enhanced features
    enable_tagging: bool = True  # Enable tagging and topic identification
    enable_relationships: bool = True  # Enable relationship analysis between chunks
    track_transcript_positions: bool = True  # Track which parts of transcript each chunk comes from
    
    # Prompt templates
    first_pass_prompt: str = FIRST_PASS_PROMPT
    second_pass_prompt: str = SECOND_PASS_PROMPT
    global_check_prompt: str = GLOBAL_CHECK_PROMPT
    tagging_prompt: str = TAGGING_PROMPT
    relationship_prompt: str = RELATIONSHIP_PROMPT 
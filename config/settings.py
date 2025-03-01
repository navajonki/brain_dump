from dataclasses import dataclass, field
from typing import Dict
import os
from dotenv import load_dotenv
from utils import get_logger

# Load environment variables at module import
load_dotenv()

# Validate required environment variables
def validate_env():
    required = [
        "OPENAI_API_KEY",
        "NEO4J_URI",
        "NEO4J_USER",
        "NEO4J_PASSWORD"
    ]
    missing = [var for var in required if not os.getenv(var)]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

# Validate on import
validate_env()

logger = get_logger(__name__)

@dataclass
class WhisperConfig:
    model: str = "base"
    settings: Dict = field(default_factory=lambda: {
        "language": "en",
        "beam_size": 5,
        "best_of": 5,
        "temperature": 0.0,
        "condition_on_previous_text": True,
        "initial_prompt": None,
        "fp16": False  # Ensure fp16 is disabled for MPS compatibility
    })

@dataclass
class OpenAIConfig:
    model: str = "gpt-3.5-turbo"
    temperature: float = 0
    embedding_model: str = "text-embedding-ada-002"

@dataclass
class PathConfig:
    output_dir: str = "output"
    transcripts_dir: str = "transcripts"
    chunks_dir: str = "chunks"
    topic_summaries_dir: str = "topic_summaries"

@dataclass
class AppConfig:
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    max_chunk_tokens: int = 300

    def __post_init__(self):
        self._load_env_vars()
    
    def _load_env_vars(self):
        # Remove debug logging
        self.whisper.model = os.getenv("WHISPER_MODEL", self.whisper.model)
        self.openai.model = os.getenv("GPT_MODEL", self.openai.model)
        # ... other env vars

config = AppConfig() 
import os
import logging
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Import backends conditionally to handle missing dependencies
try:
    from core.llm_backends.replicate_backend import ReplicateBackend
    REPLICATE_AVAILABLE = True
except ImportError:
    REPLICATE_AVAILABLE = False
    logger.warning("Replicate backend not available. Install with 'pip install replicate'")

# Check for OpenAI availability
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI backend not available. Install with 'pip install openai>=1.0.0'")

def create_llm_backend(backend_type: str, **kwargs) -> Any:
    """
    Factory function to create an LLM backend based on configuration.
    
    Args:
        backend_type: The type of backend to create ('replicate', 'openai', 'ollama')
        **kwargs: Additional arguments to pass to the backend constructor
        
    Returns:
        An instance of the requested backend
    
    Raises:
        ValueError: If the backend type is not supported or required dependencies are missing
    """
    if backend_type == "replicate":
        if not REPLICATE_AVAILABLE:
            raise ValueError("Replicate backend requested but not available. Install with 'pip install replicate'")
        
        api_key = kwargs.get('api_key') or os.environ.get("REPLICATE_API_TOKEN")
        if not api_key:
            raise ValueError("Replicate API key not found. Please set REPLICATE_API_TOKEN in your .env file")
        
        return ReplicateBackend(api_key=api_key)
    
    elif backend_type == "openai":
        if not OPENAI_AVAILABLE:
            raise ValueError("OpenAI backend requested but not available. Install with 'pip install openai>=1.0.0'")
        
        api_key = kwargs.get('api_key') or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file")
        
        # Set the API key in the environment
        os.environ["OPENAI_API_KEY"] = api_key
        
        # Return the OpenAI client
        return openai.OpenAI()
    
    elif backend_type == "ollama":
        # Ollama doesn't require a client object, just return the configuration
        return {
            "url": kwargs.get('ollama_url') or os.environ.get("OLLAMA_URL") or "http://localhost:11434/api/generate"
        }
    
    else:
        raise ValueError(f"Unsupported LLM backend type: {backend_type}")

__all__ = ["ReplicateBackend", "create_llm_backend"] 
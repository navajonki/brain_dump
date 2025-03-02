# Zettelkasten Project Guidelines

## Build & Run Commands
- Install dependencies: `pip install -e .` or `pip install -e ".[dev]"` for development
- Run tests: `pytest tests/`
- Run specific test: `pytest tests/test_chunking_atomic.py -v`
- Run test with specific config: `python tests/test_chunking_atomic.py <transcript_file> --config config/chunking/<config_file>.yaml`
- Run all tests with script: `./run_tests.sh`
- Test prompt system: `python tests/test_prompts.py`

## Architecture
This project is a Python-based system that:
- Transcribes audio using Whisper or Deepgram
- Segments transcriptions into atomic chunks using LLMs
- Processes and classifies knowledge using various backends
- Exports chunks as Markdown with YAML metadata
- Supports multiple LLM backends (OpenAI, Replicate, Ollama)

## Code Style Guidelines
- **Python**: Requires Python 3.10+
- **Type Hints**: Use comprehensive typing for functions and classes
- **Imports Order**: Standard → Third-party → Local
- **Error Handling**: Use try/except with specific error messages
- **Logging**: Use the provided logger from utils.logging
- **Config**: Use YAML files in config/ directory for different backends
- **Testing**: Use pytest for unit tests in tests/ directory
- **Documentation**: Use Google-style docstrings with Args, Returns, Raises sections

## Prompt System

The project uses a centralized prompt template system for managing prompts across different models and use cases:

```python
# Import the central prompt registry
from core.prompts import template_registry

# Get a prompt template
template = template_registry.get('first_pass')

# Format a prompt with parameters
prompt = template_registry.format('first_pass', window_text='...')

# Register a custom prompt
template_registry.register(
    name='custom_prompt',
    template='My prompt with {parameter}',
    description='Description of the prompt purpose',
    required_params=['parameter']
)
```

With ChunkingConfig:

```python
# Get a formatted prompt from config
prompt = config.get_prompt('first_pass', window_text='...')
```

### Prompt Template Structure
- Templates are stored in `core/prompts/defaults.py` with metadata
- Each template has a name, description, required parameters, and version
- Model-specific templates have a dedicated model attribute
- Templates are validated at runtime to ensure all required parameters are provided

### Loading Custom Prompts
- From YAML config: Define fields like `first_pass_prompt` in your YAML
- From module: Configure `prompt_template_module` in your config
- From registry: Use `template_registry.register()` in your code

## Batch Processing

The system supports batch processing for tagging facts to improve efficiency with LLM calls:

```python
# Configuration settings
config = ChunkingConfig(
    tagging_batch_enabled=True,     # Enable/disable batch processing (default: True)
    tagging_batch_max_tokens=4000,  # Max tokens per batch (adjust for model context size)
)
```

This feature:
- Automatically batches facts based on token count
- Adapts to different model context sizes
- Reduces API calls and costs
- Ensures proper handling of various response formats

## Response Parsing

The system uses a robust response parsing framework based on Pydantic schemas:

```python
from core.chunking.parsers import ResponseParserFactory, JSONParser

# Get a specialized parser for a specific response type
parser = ResponseParserFactory.get_first_pass_parser()

# Parse a response
parsed_data = parser.parse(llm_response)

# Or use the parser directly from TextChunker
chunker = TextChunker(config)
parsed_data = chunker._parse_response(response, 'first_pass')
```

Key features:
- Schema validation with Pydantic models
- Robust JSON cleaning and normalization
- Specialized parsers for different response types
- Graceful degradation with fallback strategies

See the [response parsing documentation](docs/response_parsing.md) for more details.
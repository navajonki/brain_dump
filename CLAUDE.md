# Zettelkasten Project Guidelines

## Build & Run Commands
- Install dependencies: `pip install -e .` or `pip install -e ".[dev]"` for development
- Run tests: `pytest tests/`
- Run specific test: `pytest tests/test_chunking_v2.py -v`
- Run test with specific config: `python tests/test_chunking_v2.py <transcript_file> --config config/chunking/<config_file>.yaml`
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
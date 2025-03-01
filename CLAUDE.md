# Zettelkasten Project Guidelines

## Build & Run Commands
- Install dependencies: `pip install -e .` or `pip install -e ".[dev]"` for development
- Run tests: `pytest tests/`
- Run specific test: `pytest tests/test_chunking_v2.py -v`
- Run test with specific config: `python tests/test_chunking_v2.py <transcript_file> --config config/chunking/<config_file>.yaml`
- Run all tests with script: `./run_tests.sh`

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
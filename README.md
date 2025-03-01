# Zettelkasten Audio Notes Processor

A system for processing audio notes into a searchable knowledge base with summaries and topic organization.

## Features

- Audio transcription with Whisper
- Automatic chunking and summarization
- Topic extraction and organization
- Vector and graph database storage
- GPU acceleration for Apple Silicon

## Installation

1. Prerequisites:
   - Python 3.10 or higher
   - FFmpeg
   - Neo4j (optional for graph storage)

2. Setup:
   ```bash
   # Create virtual environment
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows

   # Install package
   pip install -e .
   
   # For development
   pip install -e ".[dev]"
   ```

3. Configuration:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and settings
   ```

## Usage

```python
from pipelines import AudioPipeline

# Process an audio file
pipeline = AudioPipeline("path/to/audio.mp3")
pipeline.process()
```

## Project Structure

```
zettelkasten/
├── config/         # Configuration and prompts
├── core/           # Core processing modules
├── utils/          # Utility functions
├── pipelines/      # Processing pipelines
└── output/         # Generated content
```

## Configuration

Settings can be configured via:
1. Environment variables
2. `.env` file
3. Default values in `config/settings.py`

## Output Structure

```
output/
└── audio_name/
    ├── transcripts/  # Full transcriptions
    ├── chunks/       # Processed segments
    ├── topics/       # Topic summaries
    └── logs/         # Process logs
```

## Development

```bash
# Run tests
pytest

# Install dev dependencies
pip install -e ".[dev]"
```

## License

MIT
# Zettelkasten Audio Notes Processor

A comprehensive system for processing audio notes into a searchable knowledge base with atomic information chunking, intelligent summarization, and multi-modal storage.

## Core Features

- **Multi-provider Transcription**: Support for Whisper (local), Whisper API, and Deepgram
- **Intelligent Atomic Chunking**: Information-based segmentation (not just token-based)
- **Advanced Prompt Templates**: Centralized prompt management with model-specific variations
- **Multi-model LLM Support**: OpenAI, Replicate, and Ollama backends
- **Comprehensive Tagging**: Automatic entity extraction and relationship mapping
- **Multi-modal Storage**: Export to Markdown with YAML frontmatter + vector/graph database support

## Architecture Overview

The system follows a modular pipeline architecture:

1. **Audio Processing**: Convert and normalize audio files
2. **Transcription**: Convert speech to text with timestamps
3. **Chunking**: Segment text into atomic, self-contained knowledge units
4. **Enhancement**: Tagging, entity extraction, and relationship mapping
5. **Storage**: Export to markdown and databases for retrieval

## Installation

### Prerequisites:
- Python 3.10 or higher
- FFmpeg for audio processing
- (Optional) Neo4j for graph database storage
- (Optional) Ollama for local LLM inference

### Setup:
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install package
pip install -e .
   
# For development
pip install -e ".[dev]"
```

### Configuration:
```bash
cp .env.example .env
# Edit .env with your API keys and settings
```

## Usage

### Processing Audio Files
```python
from pipelines import AudioPipeline

# Process an audio file
pipeline = AudioPipeline("path/to/audio.mp3", config_path="config/chunking/openai.yaml")
pipeline.process()
```

### Using Different LLM Backends
The system supports multiple LLM backends that can be configured via YAML:

```bash
# Process using OpenAI
python tests/test_chunking_v2.py input.txt --config config/chunking/openai.yaml

# Process using Replicate
python tests/test_chunking_v2.py input.txt --config config/chunking/replicate_mistral_instruct.yaml

# Process using Ollama (local)
python tests/test_chunking_v2.py input.txt --config config/chunking/ollama_mistral.yaml
```

See [README_LLM_BACKENDS.md](README_LLM_BACKENDS.md) for detailed instructions on configuring different LLM providers.

## Advanced Features

### Atomic Chunking System
The system uses a sophisticated chunking approach based on information content rather than arbitrary token counts:

1. **Sliding Window**: Process text with overlapping windows to maintain context
2. **Two-Pass Extraction**:
   - First pass extracts atomic facts from each window
   - Second pass captures additional context and relationships
3. **Global Consistency**: Performs checks for redundancy, contradiction, and relationship mapping

### Centralized Prompt Template System
A unified prompt management system that:

- Standardizes parameter naming and validation
- Supports model-specific template variations
- Provides automatic fallbacks and error handling
- Enables loading from multiple sources (modules, YAML, registry)

```python
# Example: Access templates from code
from core.prompts import template_registry

# Format a template with parameters
prompt = template_registry.format("first_pass", window_text="...")
```

## Project Structure

```
zettelkasten/
├── config/                  # Configuration and prompts
│   ├── chunking/            # Chunking-specific configs and prompts
│   │   ├── openai.yaml      # OpenAI configuration
│   │   ├── prompts.py       # Default chunking prompts
│   │   └── prompts_mistral.py # Mistral-specific prompts
├── core/                    # Core processing modules
│   ├── chunking.py          # Advanced atomic chunking implementation
│   ├── llm_backends/        # LLM provider implementations
│   ├── prompts/             # Centralized prompt system
│   │   ├── registry.py      # Template registry
│   │   ├── template.py      # Template classes
│   │   └── defaults.py      # Default templates
│   ├── storage.py           # Storage interfaces
│   ├── summarization.py     # Summary generation
│   └── transcription.py     # Transcription services
├── pipelines/               # Processing pipelines
├── tests/                   # Test suite
├── utils/                   # Utility functions
└── output/                  # Generated content
```

## Output Structure

```
output/
└── model_name/
    └── session_id/
        ├── transcripts/     # Full transcriptions with timestamps
        ├── chunks/          # Processed atomic chunks as Markdown
        │   └── chunk_001.md # Individual chunks with YAML metadata
        ├── relationships/   # Relationship maps
        └── logs/            # Process logs and metrics
```

### Chunk Format

Each chunk is exported as a Markdown file with YAML frontmatter:

```markdown
---
id: "chunk_20250301_123456_abc123"
source: "meeting_recording.mp3"
timestamp: "00:12:34-00:12:58"
tags: ["project", "planning", "timeline"]
entities:
  people: ["Alice", "Bob"]
  places: ["Conference Room"]
  organizations: ["Acme Corp"]
relationships:
  - chunk_id: "chunk_20250301_123445_xyz789"
    type: "provides_context"
---

Alice mentioned that the Acme Corp project timeline needs to be adjusted to account for the new requirements from the client meeting last week.
```

## Development

```bash
# Run all tests
pytest

# Run specific test
pytest tests/test_chunking_atomic.py -v

# Run with specific config
python tests/test_chunking_atomic.py tests/data/sample_transcript.txt --config config/chunking/openai.yaml

# Test the prompt system
python tests/test_prompts.py
```

## Future Directions

- Enhanced real-time processing pipeline
- Web interface for browsing and querying chunks
- Integration with Obsidian and other knowledge management tools
- Support for more languages and audio formats
- Stream-based processing for long recordings

## License

MIT
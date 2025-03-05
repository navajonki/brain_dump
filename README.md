# Zettelkasten Audio Notes Processor

A comprehensive system for processing audio notes into a searchable knowledge base with atomic information chunking, intelligent summarization, and multi-modal storage.

## Overview

This project builds an end-to-end voice note taking system that captures audio recordings, transcribes them, and automatically segments the transcription into atomic "chunks" of information. The output is stored as Markdown (with YAML metadata) for human readability in tools like Obsidian while maintaining machine queryability through LLM-based retrieval systems. The system also integrates with both vector and graph databases and includes visualization capabilities.

## Core Features

- **Multi-provider Transcription**: Support for Whisper (local), Whisper API, and Deepgram
- **Intelligent Atomic Chunking**: Information-based segmentation (not just token-based)
- **Advanced Prompt Templates**: Centralized prompt management with model-specific variations
- **Multi-model LLM Support**: OpenAI, Replicate, and Ollama backends
- **Comprehensive Tagging**: Automatic entity extraction and relationship mapping
- **Multi-modal Storage**: Export to Markdown with YAML frontmatter + vector/graph database support
- **Batch Processing**: Token-based fact batching for efficient LLM usage
- **Robust Response Parsing**: Schema validation with Pydantic models

## Architecture Overview

The system follows a modular pipeline architecture:

1. **Audio Processing**: Convert and normalize audio files
2. **Transcription**: Convert speech to text with timestamps
3. **Chunking**: Segment text into atomic, self-contained knowledge units
   - Uses a sliding window approach with overlaps
   - Two-pass extraction (atomic facts + additional context)
   - Ensures global consistency and deduplication
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
# Clone the repository
git clone <repository-url>
cd zettelkasten

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install package
pip install -e .
   
# For development with test dependencies
pip install -e ".[dev]"
```

### Configuration:
```bash
# Create environment file for API keys
cp .env.example .env

# Edit .env with your API keys and settings
# Required keys depending on backends used:
# OPENAI_API_KEY=your-openai-key
# REPLICATE_API_TOKEN=your-replicate-token
# DEEPGRAM_API_KEY=your-deepgram-key
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
python tests/test_chunking_atomic.py input.txt --config config/chunking/openai.yaml

# Process using Replicate
python tests/test_chunking_atomic.py input.txt --config config/chunking/replicate_llama3_8b.yaml

# Process using Ollama (local)
python tests/test_chunking_atomic.py input.txt --config config/chunking/ollama_mistral.yaml
```

## LLM Backend Architecture

The system supports multiple LLM backends:

1. **OpenAI** - Cloud-based models from OpenAI (GPT-3.5, GPT-4, etc.)
2. **Replicate** - Cloud-based models hosted on Replicate (Mistral, Llama, Claude, etc.)
3. **Ollama** - Locally-run models via Ollama (Mistral, Llama, etc.)

Each backend is configured through YAML files in the `config/chunking/` directory. Example configuration files:

```yaml
# For OpenAI
llm_backend: openai
model: gpt-3.5-turbo

# For Replicate
llm_backend: replicate
model: mistral:instruct

# For Ollama
llm_backend: ollama
model: mistral
```

API keys are stored in the `.env` file:

```
# OpenAI API Key
OPENAI_API_KEY=your-openai-key-here

# Replicate API Token
REPLICATE_API_TOKEN=your-replicate-token-here
```

### Replicate Integration

For the Replicate integration:

1. Install the Replicate Python package:
   ```bash
   pip install replicate
   ```

2. Get a Replicate API token from [replicate.com](https://replicate.com)

Available models include:
- `mistral:instruct` - Mistral 7B Instruct
- `llama3:8b` - Meta's Llama 3 8B Instruct (recommended)
- `claude:3-haiku` - Fast Claude model
- And many more

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

### Response Parsing System
A robust schema-based parsing system that validates responses against expected schemas:

```python
from core.chunking.parsers import ResponseParserFactory

# Get a specialized parser
parser = ResponseParserFactory.get_first_pass_parser()

# Parse a response
parsed_data = parser.parse(llm_response)
```

### Batch Processing
Efficient batch processing of facts to reduce API calls:

```python
# Configuration settings
config = ChunkingConfig(
    tagging_batch_enabled=True,     # Enable/disable batch processing
    tagging_batch_max_tokens=4000,  # Max tokens per batch
)
```

## Chunking Module Architecture

The chunking module uses a dependency injection pattern for better testability:

```
TextChunker
├── ITokenizer (tokenization)
├── ILLMBackend (LLM interaction)
├── IOutputManager (file operations)
├── ISessionProvider (session management)
└── IPromptManager (prompt templating)
```

Basic usage:

```python
from core.chunking import TextChunker
from config.chunking.chunking_config import ChunkingConfig

# Create a chunker with default dependencies
config = ChunkingConfig()
chunker = TextChunker(config)

# Process a text
result = chunker.process("Your text to process", "source_identifier")
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

## Testing Guide

### Setting Up a Test Environment

1. **Ensure prerequisites are installed**:
   - Python 3.10+
   - Development dependencies: `pip install -e ".[dev]"`
   - Required API keys in .env file

2. **Verify the test environment**:
   ```bash
   # Check if pytest is installed
   pytest --version

   # Make sure test data exists
   ls tests/data
   ls tests/fixtures
   ```

### Running the Full Test Suite

You can run all tests using pytest:

```bash
# Run all tests
pytest

# Run with detailed output
pytest -v
```

Or use the provided script that runs specific test configurations:

```bash
./run_tests.sh
```

### Running Specific Tests

1. **Run specific test files**:
   ```bash
   # Run chunking unit tests
   pytest tests/test_chunking_unit.py

   # Run configuration tests
   pytest tests/test_config.py
   
   # Run a specific test within a file
   pytest tests/test_chunking_unit.py::test_create_windows
   ```

2. **Run tests with specific parameters**:
   ```bash
   # Testing with a specific transcript and config
   python tests/test_chunking_atomic.py tests/data/sample_transcript.txt --config config/chunking/openai.yaml
   ```

3. **Run tests for specific components**:
   ```bash
   # Test all transcription service integrations
   pytest tests/transcription/
   
   # Test prompts system
   python tests/test_prompts.py
   ```

### Creating New Tests

1. **Create a basic unit test**:
   ```python
   # In a new file tests/test_your_component.py
   import pytest
   from core.your_component import YourComponent

   def test_your_function():
       # 1. Set up the test
       component = YourComponent()
       
       # 2. Run the function
       result = component.your_function("test_input")
       
       # 3. Check the results
       assert result == "expected_output"
   ```

2. **Using test fixtures**:
   ```python
   # Import fixture utilities
   from tests.fixtures.fixtures import load_transcript, load_config
   
   def test_with_fixtures():
       # Load test data
       transcript = load_transcript("short")
       config_data = load_config("test_config_basic")
       
       # Run test with fixture data
       # ...
   ```

3. **Creating parameterized tests**:
   ```python
   @pytest.mark.parametrize("input_name", ["short", "medium", "technical"])
   def test_with_different_inputs(input_name):
       transcript = load_transcript(input_name)
       # Run the same test logic with different inputs
   ```

### Using Mock Dependencies

The project supports dependency injection for testing with mock objects:

```python
# Create mock dependencies
class MockLLMBackend:
    def call(self, model, prompt, **kwargs):
        # Return predetermined responses
        return "Mock response"

# Use in tests
chunker = TextChunker(
    config=config,
    llm_backend=MockLLMBackend(),  # Use mock instead of real API
    tokenizer=DefaultTokenizer()
)
```

### Understanding Test Results

When tests run, look for:

1. **Console output**: Passed tests show dots (`.`), failed tests show `F`
2. **Test summary**: Shows number of passed, failed, and skipped tests
3. **Error information**: For failed tests, shows the failure details with line numbers

### Working with Test Fixtures

The project organizes test fixtures in `tests/fixtures/`:

```
fixtures/
├── input/            # Input data for tests
│   ├── transcripts/  # Sample transcripts
│   └── config/       # Test configurations
├── expected/         # Expected outputs
├── responses/        # Mock LLM responses
└── fixtures.py       # Utility functions to load fixtures
```

To add new fixtures:
1. Add files to the appropriate directory
2. Update `fixtures.py` if needed for new fixture types
3. Reference the fixtures in your tests

## Project Structure

```
zettelkasten/
├── config/                  # Configuration and prompts
│   ├── chunking/            # Chunking-specific configs
│   │   ├── openai.yaml      # OpenAI configuration
│   │   ├── prompts.py       # Default chunking prompts
│   │   └── prompts_mistral.py # Mistral-specific prompts
├── core/                    # Core processing modules
│   ├── chunking/            # Advanced chunking implementation
│   │   ├── interfaces.py    # Dependency interfaces
│   │   ├── providers.py     # Default implementations
│   │   ├── parsers.py       # Response parsing system
│   │   └── text_chunker.py  # Main chunker implementation
│   ├── llm_backends/        # LLM provider implementations
│   │   ├── __init__.py      # Factory functions
│   │   └── replicate_backend.py # Replicate integration
│   ├── prompts/             # Centralized prompt system
│   │   ├── registry.py      # Template registry
│   │   ├── template.py      # Template classes
│   │   └── defaults.py      # Default templates
│   ├── transcription.py     # Transcription coordination
│   ├── transcription_services/ # Transcription implementations 
│   ├── storage.py           # Storage interfaces
│   └── summarization.py     # Summary generation
├── pipelines/               # Processing pipelines
├── tests/                   # Test suite
│   ├── fixtures/            # Test fixtures
│   │   ├── input/           # Input test data
│   │   ├── expected/        # Expected outputs
│   │   └── responses/       # Mock LLM responses
│   ├── data/                # Sample data for tests
│   ├── test_*.py            # Test files
│   └── transcription/       # Transcription tests
├── utils/                   # Utility functions
└── output/                  # Generated content
```

## Future Directions

- Enhanced real-time processing pipeline
- Parallel processing for large documents
- Advanced web interface for browsing and querying chunks
- Integration with Obsidian and other knowledge management tools
- Support for more languages and audio formats
- Stream-based processing for long recordings

## Web Comparison Tool

The project includes a web-based comparison tool for reviewing and comparing the results of different LLM pipeline runs:

```bash
# Install web dependencies
pip install flask flask-wtf

# Run the web app
python webapp.py
```

Then visit http://127.0.0.1:5000 in your browser.

Features:
- Compare facts, relationships, and chunks side-by-side
- Visualize relationship graphs
- Find similar facts across different LLM runs
- Review detailed metadata for each fact
- Interactive navigation between related components

## License

MIT
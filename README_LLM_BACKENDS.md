# LLM Backend Architecture

This document explains the LLM backend architecture implemented in the project, which allows for flexible switching between different language model providers through configuration.

## Overview

The system supports multiple LLM backends:

1. **OpenAI** - Cloud-based models from OpenAI (GPT-3.5, GPT-4, etc.)
2. **Replicate** - Cloud-based models hosted on Replicate (Mistral, Llama, Claude, etc.)
3. **Ollama** - Locally-run models via Ollama (Mistral, Llama, etc.)

The architecture is designed to make it easy to switch between these backends by simply changing the configuration file, without modifying any code.

## Configuration

Each backend is configured through YAML files in the `config/chunking/` directory. Example configuration files:

- `config/chunking/openai_gpt35.yaml` - Configuration for OpenAI GPT-3.5
- `config/chunking/replicate_mistral_instruct.yaml` - Configuration for Mistral Instruct via Replicate
- `config/chunking/ollama_mistral.yaml` - Configuration for local Mistral via Ollama

The key configuration parameter is `llm_backend`, which determines which backend to use:

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

## API Keys

API keys are stored in the `.env` file in the project root. Example:

```
# OpenAI API Key
OPENAI_API_KEY=your-openai-key-here

# Replicate API Token
REPLICATE_API_TOKEN=your-replicate-token-here
```

The system will automatically load these keys when needed based on the selected backend.

## Usage

To use a specific backend, simply specify the appropriate configuration file when running the chunking process:

```bash
# Run with OpenAI
python tests/test_chunking_v2.py input.txt --config config/chunking/openai_gpt35.yaml

# Run with Replicate
python tests/test_chunking_v2.py input.txt --config config/chunking/replicate_mistral_instruct.yaml

# Run with Ollama
python tests/test_chunking_v2.py input.txt --config config/chunking/ollama_mistral.yaml
```

## Implementation Details

The backend architecture is implemented through several key components:

1. **Factory Function** - `create_llm_backend()` in `core/llm_backends/__init__.py` creates the appropriate backend instance based on configuration.

2. **Backend Classes** - Each backend has its own implementation class (e.g., `ReplicateBackend`) that handles the specifics of calling that particular API.

3. **Environment Variables** - API keys are loaded from the `.env` file using python-dotenv.

4. **Chunker Integration** - The `AtomicChunker` class in `core/chunking_v2.py` uses the factory function to initialize the appropriate backend based on configuration.

## Adding New Backends

To add a new LLM backend:

1. Create a new backend class in `core/llm_backends/`
2. Update the factory function in `core/llm_backends/__init__.py`
3. Update the `_call_llm` method in `core/chunking_v2.py` to handle the new backend type
4. Create example configuration files in `config/chunking/`

## Testing

Each backend can be tested individually:

```bash
# Test OpenAI
python tests/test_openai.py

# Test Replicate
python tests/test_replicate.py --direct

# Test Ollama
python tests/test_ollama.py
``` 
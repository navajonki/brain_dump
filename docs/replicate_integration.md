# Replicate Integration for Fact Extraction

This document explains how to use the Replicate integration for fact extraction, which provides a faster and more reliable alternative to local Ollama models.

## Prerequisites

1. Install the Replicate Python package:
   ```bash
   pip install replicate
   ```

2. Get a Replicate API token:
   - Sign up at [replicate.com](https://replicate.com)
   - Go to your account settings to find your API token
   - Add your token to the `.env` file in the project root:
     ```
     REPLICATE_API_TOKEN=your_api_token_here
     ```

## Configuration

The Replicate integration is configured using a YAML file. Sample configurations are provided:
- `config/chunking/replicate_mistral_instruct.yaml` - For Mistral Instruct
- `config/chunking/replicate_llama3_8b.yaml` - For Llama 3 8B

Key configuration options:
- `llm_backend`: Set to `replicate` to use the Replicate backend
- `model`: The model to use (e.g., `llama3:8b`)
- `temperature`: Controls randomness (0.0-1.0)
- `max_tokens`: Maximum tokens to process
- `system_prompt`: System prompt to guide the model's behavior

## Available Models

The following models are available through the Replicate integration:

| Shorthand | Replicate Model ID | Notes |
|-----------|-------------------|-------|
| mistral:instruct | mistralai/mistral-7b-instruct-v0.2 | Good for general tasks |
| mistral:medium | mistralai/mistral-medium | More powerful than instruct |
| mistral:large | mistralai/mistral-large-latest | Most powerful Mistral model |
| llama3:8b | meta/meta-llama-3-8b-instruct | Fast, good quality |
| llama3:70b | meta/meta-llama-3-70b-instruct | High quality, slower |
| claude:3-haiku | anthropic/claude-3-haiku:1.0 | Fast Claude model |
| claude:3-sonnet | anthropic/claude-3-sonnet:1.0 | Balanced Claude model |
| claude:3-opus | anthropic/claude-3-opus:1.0 | Most powerful Claude model |

You can use either the shorthand (e.g., `llama3:8b`) or the full Replicate model ID in your configuration.

## Recommended Model

We recommend using `llama3:8b` (Meta's Llama 3 8B Instruct model) for fact extraction as it provides a good balance of speed and quality. This model is particularly well-suited for structured tasks like fact extraction.

## Testing the Integration

Several test scripts are provided to verify that the Replicate integration is working correctly:

```bash
# Test the Replicate backend directly
python tests/test_replicate.py --direct

# Test the AtomicChunker with Replicate backend
python tests/test_replicate.py --chunker --input output/JCS/deepgram_nova-3/transcripts/JCS_transcript_short.txt

# Test streaming with Llama 3
python tests/test_replicate_streaming.py
```

## Running Fact Extraction with Replicate

To run fact extraction using the Replicate backend with Llama 3:

```bash
python tests/test_chunking_v2.py output/JCS/deepgram_nova-3/transcripts/JCS_transcript_short.txt --config config/chunking/replicate_llama3_8b.yaml
```

Or use the convenience script that allows you to easily comment out tests you don't want to run:

```bash
./run_tests.sh
```

## Streaming Support

The Replicate backend supports streaming responses, which can be useful for interactive applications or for monitoring the progress of long responses. To use streaming:

```python
from core.llm_backends import create_llm_backend

# Create the backend
backend = create_llm_backend("replicate")

# Call with streaming enabled
stream_iterator = backend.call(
    model="llama3:8b",
    prompt="Your prompt here",
    temperature=0.7,
    max_tokens=512,
    system_prompt="You are a helpful assistant",
    stream=True  # Enable streaming
)

# Process the streaming response
for chunk in stream_iterator:
    print(str(chunk), end="", flush=True)
```

See `tests/test_replicate_streaming.py` for complete examples of streaming usage.

## Switching Between Backends

The system supports multiple LLM backends that can be selected through configuration:

1. **OpenAI**: Set `llm_backend: openai` in your config file and ensure `OPENAI_API_KEY` is in your `.env` file
2. **Replicate**: Set `llm_backend: replicate` in your config file and ensure `REPLICATE_API_TOKEN` is in your `.env` file
3. **Ollama**: Set `llm_backend: ollama` in your config file to use a local Ollama instance

Example of switching between backends:

```bash
# Run with OpenAI
python tests/test_chunking_v2.py input.txt --config config/chunking/openai_gpt35.yaml

# Run with Replicate (Llama 3)
python tests/test_chunking_v2.py input.txt --config config/chunking/replicate_llama3_8b.yaml

# Run with Ollama
python tests/test_chunking_v2.py input.txt --config config/chunking/ollama_mistral.yaml
```

## Troubleshooting

If you encounter issues with the Replicate integration:

1. Verify that your API token is set correctly in the `.env` file:
   ```bash
   cat .env | grep REPLICATE_API_TOKEN
   ```

2. Check your internet connection, as Replicate requires internet access.

3. Verify that the model you're trying to use is available on Replicate.

4. Check the logs for any error messages from the Replicate API.

5. If you're getting empty responses, try increasing the `max_tokens` parameter.

## Pricing

Note that using Replicate incurs costs based on the models you use and the amount of computation. Check the [Replicate pricing page](https://replicate.com/pricing) for current rates.

The Llama 3 8B model is one of the more cost-effective options, with pricing around $0.20 per million tokens (as of the time of writing).

## Extending the Integration

To add support for additional models:

1. Update the `model_mappings` dictionary in `core/llm_backends/replicate_backend.py`
2. Add appropriate input parameter handling in the `call` method if the model requires specific parameters 
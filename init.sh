#!/bin/bash

# Check Python version
python3 --version || { echo "Python 3.10+ required"; exit 1; }

# Check FFmpeg
ffmpeg -version || { echo "FFmpeg required"; exit 1; }

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -e .

# Create necessary directories
mkdir -p output

# Setup environment
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env file. Please update with your API keys."
fi

echo "Setup complete! Don't forget to:"
echo "1. Edit .env with your API keys"
echo "2. Activate the virtual environment: source .venv/bin/activate"

# Example dependencies:
pip install openai whisper chromadb pyyaml neo4j tiktoken spacy
# Or if you want LLM-based topic detection:
# pip install openai
# (Then load spaCy models if you're using spaCy for topic/keyword extraction)
python -m spacy download en_core_web_sm

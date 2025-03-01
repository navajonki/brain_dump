from pathlib import Path

# Test audio files
TEST_AUDIO = {
    "short": "/Users/zbodnar/python/stories/Conversations/techcrunch.mp3",
    "medium": "/Users/zbodnar/python/stories/Conversations/youtube/JCS.mp3",
    # Add more test files as needed
}

# Deepgram configurations
DEEPGRAM_CONFIGS = {
    "nova": {
        "model": "nova-3",
        "language": "en"
    },
    "base": {
        "model": "base",
        "language": "en"
    }
}

# Whisper API configurations
WHISPER_API_CONFIGS = {
    "default": {
        "model": "whisper-1",
        "response_format": "verbose_json",
        "language": "en"
    }
}

# Verify test files exist
for name, path in TEST_AUDIO.items():
    if not Path(path).exists():
        raise FileNotFoundError(f"Test audio file '{name}' not found at: {path}") 
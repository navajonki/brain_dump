import os
from graveyard.zettel import transcribe_audio
import torch
import asyncio
from graveyard.zettel import process_audio_note

# audio_file = "/Users/zbodnar/python/stories/Conversations/youtube/JCS.mp3"
# audio_file ="/Users/zbodnar/python/stories/Conversations/techcrunch.mp3"
audio_file = "/Users/zbodnar/Downloads/Voicenotes_2025-01-25_job_stuff.mp3"

assert os.path.exists(audio_file), "Audio file exists"

# Check available devices
print("CUDA available:", torch.cuda.is_available())
print("MPS available:", torch.backends.mps.is_available())
print("MPS built:", torch.backends.mps.is_built())
device = ("MPS (Apple Silicon)" if torch.backends.mps.is_available() and torch.backends.mps.is_built()
          else "CUDA" if torch.cuda.is_available() 
          else "CPU")
print(f"Using device: {device}")

# Test transcription
try:
    transcript = transcribe_audio(audio_file, model_name="base")
    # transcript = transcribe_audio(audio_file, model_name="tiny")
    print("Transcription successful!")
    print("First 200 characters:", transcript[:200])
except Exception as e:
    print(f"Transcription error: {str(e)}")

# Use the same audio file path
audio_file = "/Users/zbodnar/python/stories/Conversations/youtube/JCS.mp3"
session_id = "jcs_analysis"  # Give a meaningful session ID

# Run the processing pipeline
asyncio.run(process_audio_note(audio_file, session_id))
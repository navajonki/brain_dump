from pathlib import Path
from pipelines import AudioPipeline
from utils import get_logger

logger = get_logger(__name__)

def main():
    # Example usage
    audio_file = "/Users/zbodnar/python/stories/Conversations/techcrunch.mp3"
    pipeline = AudioPipeline(audio_file)
    pipeline.process()

if __name__ == "__main__":
    main() 
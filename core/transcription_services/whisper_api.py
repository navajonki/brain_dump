from typing import Optional
from pathlib import Path
import openai
from .base import TranscriptionService
from utils.logging import get_logger

class WhisperAPIService(TranscriptionService):
    def __init__(self, log_file: Optional[Path] = None):
        self.logger = get_logger(__name__, log_file)
        self.client = openai.OpenAI()
    
    @property
    def name(self) -> str:
        return "whisper_api"
        
    def transcribe(self, audio_path: str | Path, output_file: Optional[Path] = None) -> str:
        audio_path = Path(audio_path)
        self.logger.info(f"Transcribing {audio_path} with Whisper API")
        
        with open(audio_path, "rb") as audio_file:
            result = self.client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-1",
                response_format="verbose_json"
            )
            
        if output_file:
            with open(output_file, "w") as f:
                for segment in result.segments:
                    line = f"[{segment.start} --> {segment.end}] {segment.text}\n"
                    f.write(line)
                    self.logger.info(f"Transcribed: {line.strip()}")
        
        return result.text 
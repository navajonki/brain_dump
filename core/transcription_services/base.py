from abc import ABC, abstractmethod
from typing import Optional
from pathlib import Path

class TranscriptionService(ABC):
    @abstractmethod
    def transcribe(self, audio_path: str | Path, output_file: Optional[Path] = None) -> str:
        """Transcribe audio file to text"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return service name for output folder naming"""
        pass 
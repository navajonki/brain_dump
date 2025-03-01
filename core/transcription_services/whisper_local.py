from typing import Optional
from pathlib import Path
import whisper
import torch
from .base import TranscriptionService
from utils.logging import get_logger
from config import config

class WhisperLocalService(TranscriptionService):
    def __init__(self, model_name: Optional[str] = None, log_file: Optional[Path] = None):
        self.logger = get_logger(__name__, log_file)
        self.model_name = model_name or config.whisper.model
        self.model = None
        self.device = self._setup_device()
        
    @property
    def name(self) -> str:
        return f"whisper_local_{self.model_name}"
    
    # ... rest of current Whisper implementation 
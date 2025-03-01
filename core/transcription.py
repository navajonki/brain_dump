from typing import Optional
from pathlib import Path
from datetime import timedelta
import whisper
import torch
import soundfile as sf
from utils.logging import get_logger
from config import config
import time

class Transcriber:
    def __init__(self, model_name: Optional[str] = None, log_file: Optional[Path] = None):
        # Initialize logger with log file
        self.logger = get_logger(__name__, log_file)
        
        # Add debug logging
        self.logger.info(f"Config whisper model: {config.whisper.model}")
        self.logger.info(f"Provided model name: {model_name}")
        
        self.model_name = model_name or config.whisper.model
        self.logger.info(f"Using model: {self.model_name}")
        self.model = None
        
        # Check available devices
        self.logger.info(f"CUDA available: {torch.cuda.is_available()}")
        self.logger.info(f"MPS available: {torch.backends.mps.is_available()}")
        self.logger.info(f"MPS built: {torch.backends.mps.is_built()}")
        
        self.device = self._setup_device()
        self.logger.info(f"Using device: {self.device}")
    
    def _setup_device(self) -> str:
        """Setup optimal device for M1/M2 Macs or fallback to CPU"""
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            try:
                # Test MPS availability
                device = torch.device("mps")
                x = torch.ones(1, device=device)
                del x
                self.logger.info("MPS (M2 GPU) test successful")
                return "mps"
            except Exception as e:
                self.logger.warning(f"MPS available but test failed: {str(e)}")
                return "cpu"
        return "cpu"
    
    def _load_model(self):
        """Load model with optimized settings for M1/M2"""
        self.logger.info(f"Loading Whisper model: {self.model_name}")
        try:
            self.model = whisper.load_model(self.model_name)
            self.logger.info("Model loaded successfully")
            
            if self.device == "mps":
                try:
                    self.logger.info("Attempting to use M2 GPU...")
                    # Set default device to MPS
                    device = torch.device("mps")
                    
                    # Move model components to GPU explicitly
                    self.model.encoder = self.model.encoder.to(device)
                    self.model.decoder = self.model.decoder.to(device)
                    if hasattr(self.model, 'mel_filters'):
                        self.model.mel_filters = self.model.mel_filters.to(device)
                    self.logger.info("Successfully initialized M2 GPU acceleration")
                except Exception as e:
                    self.logger.warning(f"Failed to move model to MPS: {str(e)}")
                    self.device = "cpu"
                    self.model = self.model.to("cpu")
                    self.logger.info("Falling back to CPU")
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def transcribe(self, audio_path: str | Path, output_file: Optional[Path] = None) -> str:
        """Transcribe audio with optimized settings and detailed logging"""
        if not self.model:
            self._load_model()
        
        audio_path_str = str(audio_path)
        
        # Verify file exists
        if not Path(audio_path_str).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path_str}")
        
        # Get audio duration for logging
        audio_info = sf.info(audio_path_str)
        total_duration = audio_info.duration
        self.logger.info(f"Audio duration: {total_duration:.2f} seconds")
        
        self.logger.info(f"Starting transcription of {audio_path_str}")
        self.logger.info(f"Using device: {self.device}")
        
        try:
            self.logger.info("Loading audio...")
            # Load audio using soundfile
            audio, _ = sf.read(audio_path_str)
            self.logger.info("Audio loaded successfully")
            
            self.logger.info("Starting transcription...")
            start_time = time.time()
            
            # Prepare output file if needed
            if output_file:
                output_file.parent.mkdir(parents=True, exist_ok=True)
                out_file = open(output_file, "w", encoding="utf-8")
            
            # Transcribe
            result = self.model.transcribe(
                audio_path_str,
                **config.whisper.settings
            )
            
            # Process segments in real-time
            last_text = ""
            for i, segment in enumerate(result["segments"], 1):
                start_time_str = str(timedelta(seconds=int(segment["start"])))
                end_time_str = str(timedelta(seconds=int(segment["end"])))
                text = segment['text'].strip()
                line = f"[{start_time_str} --> {end_time_str}] {text}"
                last_text = text
                
                # Write to file if provided
                if output_file:
                    out_file.write(line + "\n")
                    out_file.flush()
                
                # Log progress
                self.logger.info(f"Transcribed segment {i}: {start_time_str} --> {end_time_str}: {text}")
            
            # Close output file if it was opened
            if output_file:
                out_file.close()
            
            # Calculate and log transcription speed
            elapsed = time.time() - start_time
            speed = total_duration / elapsed
            self.logger.info(f"Transcription completed in {elapsed:.1f}s ({speed:.2f}x realtime)")
            
            # Print final segment
            print("\nLast transcribed text:")
            print(f"Timestamp: {start_time_str} --> {end_time_str}")
            print(f"Text: {last_text}\n")
            
            return result["text"]
            
        except Exception as e:
            self.logger.error(f"Transcription error: {str(e)}")
            raise 
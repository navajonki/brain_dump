from typing import Optional
from pathlib import Path
import os
import mimetypes
from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
)
from .base import TranscriptionService
from utils.logging import get_logger

class DeepgramService(TranscriptionService):
    def __init__(self, 
                 model: str = "nova-3", 
                 language: str = "en",
                 log_file: Optional[Path] = None):
        self.logger = get_logger(__name__, log_file)
        
        # Initialize Deepgram client (uses DEEPGRAM_API_KEY from env)
        self.dg = DeepgramClient()
        self.model = model
        self.language = language
    
    @property
    def name(self) -> str:
        return f"deepgram_{self.model}"
        
    async def transcribe(self, audio_path: str | Path, output_file: Optional[Path] = None) -> str:
        """Transcribe audio using Deepgram's API"""
        audio_path = Path(audio_path)
        self.logger.info(f"Transcribing {audio_path} with Deepgram")
        
        try:
            with open(audio_path, 'rb') as audio:
                source = {
                    'buffer': audio.read(),
                    'mimetype': mimetypes.guess_type(audio_path)[0] or 'audio/mpeg'
                }
                
                options = PrerecordedOptions(
                    model=self.model,
                    smart_format=True,
                    language=self.language,
                    utterances=True,
                    paragraphs=True,
                    diarize=True,
                    punctuate=True  # Word timing is included by default
                )
                
                response = self.dg.listen.rest.v("1").transcribe_file(source, options)
                
                # Get transcript text and metadata
                transcript = ""
                if hasattr(response, 'results'):
                    results = response.results
                    if hasattr(results, 'channels') and results.channels:
                        channel = results.channels[0]
                        if hasattr(channel, 'alternatives') and channel.alternatives:
                            alt = channel.alternatives[0]
                            transcript = alt.transcript
                            self.logger.info(f"Got transcript of length: {len(transcript)}")
                            
                            # Write detailed transcript with timestamps
                            if output_file and hasattr(alt, 'words'):
                                with open(output_file, "w") as f:
                                    current_time = 0.0
                                    current_line = []
                                    
                                    for word in alt.words:
                                        # Start new line if more than 2 seconds have passed
                                        if word.start - current_time > 2.0 and current_line:
                                            line_text = " ".join(current_line)
                                            f.write(f"[{current_time:.2f} - {word.start:.2f}] {line_text}\n")
                                            current_line = []
                                        
                                        current_line.append(word.punctuated_word)
                                        current_time = word.end
                                    
                                    # Write final line
                                    if current_line:
                                        line_text = " ".join(current_line)
                                        f.write(f"[{current_time-2:.2f} - {current_time:.2f}] {line_text}\n")
                                    
                                    self.logger.info(f"Wrote timestamped transcript to {output_file}")
                
                return transcript
                
        except Exception as e:
            self.logger.error(f"Transcription error: {str(e)}")
            raise 
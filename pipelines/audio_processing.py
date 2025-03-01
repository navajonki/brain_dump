from pathlib import Path
from typing import Optional
from core.transcription_services import (
    TranscriptionService,
    WhisperLocalService,
    WhisperAPIService,
    DeepgramService
)
from core import TextChunker, Summarizer
from utils import OutputManager, get_logger
from config import config

class AudioPipeline:
    def __init__(
        self,
        audio_file: str,
        transcription_service: Optional[TranscriptionService] = None,
        model_name: Optional[str] = None,
        output_base: Optional[str] = None
    ):
        self.audio_path = Path(audio_file)
        
        # Use default service if none provided
        self.transcriber = transcription_service or WhisperLocalService(model_name)
        
        # Create output manager with service name in path
        self.output = OutputManager(
            base_name=self.audio_path.stem,
            variant_name=self.transcriber.name,
            output_base=output_base
        )
        
        self.chunker = TextChunker(config.max_chunk_tokens)
        self.summarizer = Summarizer()
        
        self.logger = get_logger(__name__, self.output.log_file)
        
        self.logger.info(f"Pipeline initialized for {audio_file}")
    
    async def process(self):
        """Process the audio file"""
        self.logger.info(f"Processing audio file: {self.audio_path}")
        
        try:
            # Step 1: Transcription
            self.logger.info("Starting transcription...")
            transcript = await self.transcriber.transcribe(
                self.audio_path,
                self.output.transcript_file
            )
            self.logger.info(f"Transcription complete: {len(transcript)} characters")
            
            # Step 2: Chunking
            self.logger.info("Chunking transcript...")
            chunks = self.chunker.process(transcript)
            chunk_files = self.output.save_chunks(chunks)
            self.logger.info(f"Created {len(chunks)} chunks")
            
            # Step 3: Summarization
            self.logger.info("Generating summaries...")
            processed_chunks = self.summarizer.process_chunks(chunk_files)
            self.logger.info(f"Generated summaries for {len(processed_chunks)} chunks")
            
            self.logger.info("Pipeline completed successfully")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise 
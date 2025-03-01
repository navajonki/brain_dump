from pathlib import Path
import argparse
from pipelines import AudioPipeline
from core.transcription_services import WhisperAPIService
from utils import get_logger
from tests.test_config import TEST_AUDIO, WHISPER_API_CONFIGS

logger = get_logger(__name__)

def test_whisper_api_transcription(audio_file: str = None, config_name: str = "default"):
    """Test OpenAI Whisper API transcription service"""
    audio_file = audio_file or TEST_AUDIO["short"]
    config = WHISPER_API_CONFIGS[config_name]
    
    service = WhisperAPIService(**config)
    pipeline = AudioPipeline(
        audio_file,
        transcription_service=service
    )
    pipeline.process()
    
    # Verify output
    audio_name = Path(audio_file).stem
    output_dir = Path("output") / audio_name / "whisper_api"
    assert output_dir.exists(), "Output directory not created"
    assert (output_dir / "transcripts").exists(), "Transcripts not created"
    assert (output_dir / "logs").exists(), "Logs not created"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-file", help="Path to audio file to transcribe")
    parser.add_argument("--config", choices=WHISPER_API_CONFIGS.keys(), default="default",
                      help="Which Whisper API configuration to use")
    args = parser.parse_args()
    test_whisper_api_transcription(args.audio_file, args.config) 
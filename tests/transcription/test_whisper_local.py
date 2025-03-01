from pathlib import Path
import argparse
from pipelines import AudioPipeline
from core.transcription_services import WhisperLocalService
from utils import get_logger
from tests.test_config import TEST_AUDIO

logger = get_logger(__name__)

def test_whisper_local_transcription(audio_file: str = None):
    """Test local Whisper transcription service"""
    # Use default test file if none provided
    audio_file = audio_file or TEST_AUDIO["short"]
    
    service = WhisperLocalService(model_name="base")
    pipeline = AudioPipeline(
        audio_file,
        transcription_service=service
    )
    pipeline.process()
    
    # Verify output
    audio_name = Path(audio_file).stem
    output_dir = Path("output") / audio_name / "whisper_local_base"
    assert output_dir.exists(), "Output directory not created"
    assert (output_dir / "transcripts").exists(), "Transcripts not created"
    assert (output_dir / "logs").exists(), "Logs not created"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-file", help="Path to audio file to transcribe")
    args = parser.parse_args()
    test_whisper_local_transcription(args.audio_file) 
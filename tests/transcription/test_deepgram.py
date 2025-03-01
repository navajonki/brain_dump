import asyncio
import sys
from pathlib import Path
import argparse

# Add project root to Python path
project_root = Path(__file__).parents[2]
sys.path.append(str(project_root))

from pipelines import AudioPipeline
from core.transcription_services import DeepgramService
from utils import get_logger
from tests.test_config import TEST_AUDIO, DEEPGRAM_CONFIGS

logger = get_logger(__name__)

async def test_deepgram_transcription(audio_file: str = None, config_name: str = None):
    """Test Deepgram transcription service"""
    # Use first config if none specified
    config_name = config_name or next(iter(DEEPGRAM_CONFIGS))
    audio_file = audio_file or TEST_AUDIO["medium"]
    
    config = DEEPGRAM_CONFIGS[config_name]
    logger.info(f"Using Deepgram config: {config_name}")
    
    service = DeepgramService(**config)
    pipeline = AudioPipeline(
        audio_file,
        transcription_service=service
    )
    await pipeline.process()
    
    # Verify output
    audio_name = Path(audio_file).stem
    output_dir = Path("output") / audio_name / f"deepgram_{config['model']}"
    assert output_dir.exists(), "Output directory not created"
    assert (output_dir / "transcripts").exists(), "Transcripts not created"
    assert (output_dir / "logs").exists(), "Logs not created"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-file", help="Path to audio file to transcribe")
    parser.add_argument("--config", choices=DEEPGRAM_CONFIGS.keys(), default="nova",
                      help="Which Deepgram configuration to use")
    args = parser.parse_args()
    asyncio.run(test_deepgram_transcription(args.audio_file, args.config)) 
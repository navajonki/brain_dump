import pytest
from unittest.mock import Mock, patch
from graveyard.zettel import (
    clean_text_for_markdown,
    chunk_text,
    num_tokens,
    process_audio_note,
    transcribe_audio
)

def test_clean_text_for_markdown():
    text = "Here is some ```code``` block"
    cleaned = clean_text_for_markdown(text)
    assert "` ` `" in cleaned
    assert "```" not in cleaned

def test_chunk_text():
    text = "short text"
    chunks = chunk_text(text, max_tokens=5)
    assert len(chunks) > 0
    assert all(num_tokens(chunk) <= 5 for chunk in chunks)

@pytest.mark.asyncio
async def test_process_audio_note():
    with patch('zettel.transcribe_audio') as mock_transcribe:
        mock_transcribe.return_value = "Test transcript"
        
        with patch('zettel.generate_summary_and_topics') as mock_summary:
            mock_summary.return_value = ("Test summary", ["topic1"])
            
            await process_audio_note(
                "test.wav",
                "test_session",
                output_dir="test_output",
                topic_summaries_dir="test_summaries"
            )
            
            mock_transcribe.assert_called_once()
            assert mock_summary.called 

def test_transcribe_audio():
    # ... test setup ...
    result = transcribe_audio(test_audio_file, model_name="base")  # Use "base" model
    # ... test assertions ... 
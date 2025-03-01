from typing import List, Tuple
from pathlib import Path
import yaml
import openai
from config import config, SUMMARY_AND_TOPICS_PROMPT
import os
import logging

logger = logging.getLogger(__name__)

class Summarizer:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        self.client = openai.OpenAI(api_key=api_key)
    
    def process_chunks(self, chunk_files: List[Path]) -> List[dict]:
        """Process a list of chunk files, generating summaries and topics."""
        processed = []
        for chunk_file in chunk_files:
            with open(chunk_file, 'r') as f:
                content = f.read()
            
            # Split YAML front matter from content
            _, yaml_text, chunk_text = content.split('---', 2)
            metadata = yaml.safe_load(yaml_text)
            
            summary, topics = self._generate_summary_and_topics(chunk_text.strip())
            metadata['summary'] = summary
            metadata['topics'] = topics
            
            processed.append({
                'id': metadata['id'],
                'text': chunk_text.strip(),
                'metadata': metadata
            })
            
            # Update the file with new metadata
            self._update_chunk_file(chunk_file, metadata, chunk_text)
            
        return processed
    
    def _generate_summary_and_topics(self, text: str) -> Tuple[str, List[str]]:
        """Generate summary and topics for a chunk of text."""
        logger.info("Using prompt template:")
        logger.info(SUMMARY_AND_TOPICS_PROMPT)
        
        prompt = SUMMARY_AND_TOPICS_PROMPT.format(text=text)
        logger.info("Generated prompt:")
        logger.info(prompt)
        
        response = self.client.chat.completions.create(
            model=config.openai.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=config.openai.temperature
        )
        
        try:
            import json
            data = json.loads(response.choices[0].message.content)
            return data.get("summary", ""), data.get("topics", [])
        except:
            return "Error generating summary", []
    
    def _update_chunk_file(self, file_path: Path, metadata: dict, content: str):
        """Update a chunk file with new metadata."""
        updated_content = f"""---
{yaml.dump(metadata, sort_keys=False)}---
{content}"""
        
        with open(file_path, 'w') as f:
            f.write(updated_content) 
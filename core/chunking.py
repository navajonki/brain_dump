from typing import List, Tuple
import tiktoken
import re
import requests
import openai
from pathlib import Path
from datetime import datetime
import time
from utils.logging import get_logger
from config.prompts import TOPIC_CHANGE_PROMPT, SPLIT_SENTENCE_PROMPT
from config.chunking_config import ChunkingConfig
from utils.file_ops import OutputManager

class TextChunker:
    def __init__(self, config: ChunkingConfig = None):
        self.config = config or ChunkingConfig()
        self.encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.logger = get_logger(__name__)
        self.output_mgr = None  # Will be set in process()
        
        if self.config.llm_backend == "openai":
            self.client = openai.OpenAI()
        
        # Common sentence ending patterns
        self.sentence_endings = r'[.!?][\s"\')\]]* '
        
        # Setup logs
        self.llm_log = Path("output/llm_interactions.log")
        self.metrics_log = Path("output/llm_metrics.log")
        self.llm_log.parent.mkdir(parents=True, exist_ok=True)
        
        # Timing metrics
        self.session_start = time.time()
        self.response_times = []
        self.total_prompts = 0
        
        # Text metrics (to be set in process())
        self.text_tokens = 0
        self.text_chars = 0
        self.text_words = 0
        self.transcript_name = ""
    
    def _get_metadata(self) -> dict:
        """Get metadata about the current processing run"""
        return {
            "model": self.config.model,
            "llm_backend": self.config.llm_backend,
            "max_tokens": self.config.max_tokens,
            "use_semantic_chunking": self.config.use_semantic_chunking,
            "temperature": getattr(self.config, 'temperature', None),
            "max_response_tokens": getattr(self.config, 'max_response_tokens', None),
            "ollama_url": getattr(self.config, 'ollama_url', None)
        }
    
    def _write_metrics_header(self, text: str, transcript_path: str = "unknown"):
        """Write initial metrics header"""
        self.text_tokens = len(self.encoder.encode(text))
        self.text_chars = len(text)
        self.text_words = len(text.split())
        self.transcript_name = Path(transcript_path).name
        
        with open(self.metrics_log, "a") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"Processing started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Transcript: {self.transcript_name}\n")
            f.write(f"Model: {self.config.model} ({self.config.llm_backend})\n")
            f.write(f"Text metrics:\n")
            f.write(f"  - Tokens: {self.text_tokens:,}\n")
            f.write(f"  - Characters: {self.text_chars:,}\n")
            f.write(f"  - Words: {self.text_words:,}\n")
            f.write(f"{'='*80}\n\n")
    
    def _log_metrics(self, response_time: float):
        """Log timing metrics after each LLM response"""
        self.response_times.append(response_time)
        self.total_prompts += 1
        
        avg_response_time = sum(self.response_times) / len(self.response_times)
        total_elapsed_mins = (time.time() - self.session_start) / 60
        
        # Estimate remaining prompts based on text length and chunk size
        estimated_total_prompts = self.text_tokens / self.config.max_tokens * 2  # rough estimate
        remaining_prompts = estimated_total_prompts - self.total_prompts
        estimated_remaining_mins = (remaining_prompts * avg_response_time) / 60
        
        with open(self.metrics_log, "a") as f:
            f.write(f"Prompt {self.total_prompts}:\n")
            f.write(f"  Response time: {response_time:.2f}s\n")
            f.write(f"  Average response time: {avg_response_time:.2f}s\n")
            f.write(f"  Total elapsed time: {total_elapsed_mins:.1f}m\n")
            f.write(f"  Estimated remaining time: {estimated_remaining_mins:.1f}m\n")
            f.write("-" * 40 + "\n")
    
    def _log_llm_interaction(self, prompt: str, response: str):
        """Log LLM prompt and response with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.llm_log, "a") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Backend: {self.config.llm_backend}\n")
            f.write(f"Model: {self.config.model}\n")
            f.write(f"\nPROMPT:\n{prompt}\n")
            f.write(f"\nRESPONSE:\n{response}\n")
            f.write(f"{'='*80}\n")
    
    def _call_llm(self, prompt: str, chunk_num: int = 0) -> str:
        """Make a request to the configured LLM backend with timing"""
        try:
            if not self.config.use_semantic_chunking:
                return ""
            
            start_time = time.time()
            response = ""
            
            if self.config.llm_backend == "ollama":
                api_response = requests.post(self.config.ollama_url, json={
                    "model": self.config.model,
                    "prompt": prompt,
                    "stream": False
                })
                api_response.raise_for_status()
                response = api_response.json()['response'].strip()
                
            elif self.config.llm_backend == "openai":
                api_response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_response_tokens
                )
                response = api_response.choices[0].message.content.strip()
            
            response_time = time.time() - start_time
            
            # Log interaction and metrics if we have an output manager
            if self.output_mgr:
                self.output_mgr.log_llm_interaction(
                    chunk_num=chunk_num,
                    prompt=prompt,
                    response=response,
                    metadata=self._get_metadata()
                )
                
                self.output_mgr.log_llm_metrics(
                    chunk_num=chunk_num,
                    metrics={
                        "response_time": f"{response_time:.2f}s",
                        "prompt_tokens": len(self.encoder.encode(prompt)),
                        "response_tokens": len(self.encoder.encode(response))
                    },
                    metadata=self._get_metadata()
                )
            
            return response
                
        except Exception as e:
            self.logger.error(f"LLM API error: {e}")
            if self.output_mgr:
                self.output_mgr.log_llm_interaction(
                    chunk_num=chunk_num,
                    prompt=prompt,
                    response=f"ERROR: {str(e)}",
                    metadata=self._get_metadata()
                )
            return ""
    
    def _detect_topic_change(self, current_text: str, next_sentence: str, chunk_num: int) -> bool:
        """Use configured LLM to detect topic changes"""
        if not self.config.use_semantic_chunking:
            return False
            
        try:
            prompt = TOPIC_CHANGE_PROMPT.format(
                text=current_text,
                next_sentence=next_sentence
            )
            result = self._call_llm(prompt, chunk_num).lower()
            is_new_topic = result == 'true'
            
            if is_new_topic:
                self.logger.info(f"Topic change detected at: {next_sentence[:50]}...")
            
            return is_new_topic
            
        except Exception as e:
            self.logger.error(f"Error detecting topic change: {e}")
            return False
    
    def process(self, text: str, transcript_path: str = "unknown") -> List[str]:
        """Split text into chunks based on semantic boundaries."""
        # Initialize output manager
        self.output_mgr = OutputManager(
            input_file=transcript_path,
            model_name=self.config.model,
            output_type="chunks"
        )
        
        self.logger.info("Starting text chunking...")
        
        # First split by double newlines (paragraph breaks)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        self.logger.info(f"Found {len(paragraphs)} paragraphs")
        
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_num = 1
        
        for paragraph in paragraphs:
            # Split paragraph into sentences
            sentences = re.split(self.sentence_endings, paragraph)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            for sentence in sentences:
                sentence_tokens = len(self.encoder.encode(sentence))
                
                # Check for topic change if we have content
                if current_chunk:
                    current_text = ' '.join(current_chunk)
                    is_new_topic = self._detect_topic_change(current_text, sentence, chunk_num)
                    
                    if is_new_topic or current_length + sentence_tokens > self.config.max_tokens:
                        chunks.append(current_text)
                        current_chunk = []
                        current_length = 0
                        chunk_num += 1
                
                # Handle long sentences
                if sentence_tokens > self.config.max_tokens:
                    self.logger.warning(f"Long sentence found: {sentence[:100]}...")
                    sub_sentences = self._split_long_sentence(sentence, chunk_num)
                    for sub in sub_sentences:
                        sub_tokens = len(self.encoder.encode(sub))
                        if current_length + sub_tokens > self.config.max_tokens and current_chunk:
                            chunks.append(' '.join(current_chunk))
                            current_chunk = []
                            current_length = 0
                            chunk_num += 1
                        current_chunk.append(sub)
                        current_length += sub_tokens
                else:
                    current_chunk.append(sentence)
                    current_length += sentence_tokens
        
        # Add final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        self.logger.info(f"Created {len(chunks)} chunks")
        
        # Save chunks with metadata
        metadata = {
            **self._get_metadata(),
            "processing_start": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_chunks": len(chunks),
            "input_text_metrics": {
                "tokens": len(self.encoder.encode(text)),
                "characters": len(text),
                "words": len(text.split())
            }
        }
        
        return self.output_mgr.save_chunks(chunks, metadata)
    
    def _split_long_sentence(self, text: str, chunk_num: int) -> List[str]:
        """Use LLM to split a long sentence into coherent parts."""
        try:
            prompt = SPLIT_SENTENCE_PROMPT.format(text=text)
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            
            # Parse numbered list from response
            parts = response.choices[0].message.content.strip().split('\n')
            parts = [p.strip().split('. ', 1)[1] if '. ' in p else p for p in parts if p.strip()]
            return parts
            
        except Exception as e:
            self.logger.error(f"Error splitting long sentence: {e}")
            # Fallback to simple splitting on punctuation
            return [p.strip() for p in re.split('[,;]', text) if p.strip()] 
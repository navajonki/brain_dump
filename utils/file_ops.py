import os
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import yaml
import sys
import inspect
import uuid


def load_json_file(file_path: str) -> Dict[str, Any]:
    """
    Load JSON data from a file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary containing the JSON data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    with open(file_path, 'r') as f:
        return json.load(f)

class OutputManager:
    """
    Manages output files for LLM interactions, metrics, and chunks.
    """
    
    def __init__(self, input_file: str, model_name: str, output_type: str = "chunks"):
        """
        Initialize the output manager.
        
        Args:
            input_file: Path to the input file
            model_name: Name of the model being used
            output_type: Type of output (chunks, summaries, etc.)
        """
        self.input_file = input_file
        self.model_name = model_name
        self.output_type = output_type
        
        # Get the project root directory (absolute path)
        self.project_root = self._get_project_root()
        
        # Get base name from input file
        self.base_name = Path(input_file).stem
        
        # Extract source name from the base name (e.g., JCS, techcrunch)
        # This assumes the base name format is something like "JCS_transcript_short"
        self.source_name = self.base_name.split('_')[0]
        
        # Create output directory structure: [project_root]/output/[source_name]/[model_name]/
        self.output_dir = self.project_root / "output" / self.source_name / model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.chunks_dir = self.output_dir / "chunks"
        self.logs_dir = self.output_dir / "logs"
        self.metadata_dir = self.output_dir / "metadata"  # New directory for session metadata
        self.chunks_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)
        
        # Set up log files in the logs directory
        self.debug_log = self.logs_dir / f"debug_{self.base_name}_{model_name}.log"
        self.llm_interactions_log = self.logs_dir / f"llm_interactions_{self.base_name}_{model_name}.log"
        self.llm_metrics_log = self.logs_dir / f"llm_metrics_{self.base_name}_{model_name}.log"
        self.final_summary_file = self.output_dir / f"final_summary_{self.base_name}_{model_name}.json"
        
        # Initialize session metadata
        self.session_metadata = {
            "session_id": str(uuid.uuid4()),
            "source_file": input_file,
            "model": model_name,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "transcript_metadata": {},  # Will store timestamp ranges, speakers, etc.
            "chunk_count": 0,
            "total_facts": 0,
            "context": {},  # Will store session context like location, participants, etc.
        }
        
        # Save initial session metadata
        self._save_session_metadata()
        
        # Initialize debug log
        with open(self.debug_log, 'w') as f:
            f.write(f"Debug Log\n")
            f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input file: {input_file}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Project root: {self.project_root}\n")
            f.write(f"Output directory: {self.output_dir}\n")
            f.write(f"Session ID: {self.session_metadata['session_id']}\n")
            f.write("="*80 + "\n\n")
        
        # Initialize LLM interactions log
        with open(self.llm_interactions_log, 'w') as f:
            f.write(f"LLM Interactions Log\n")
            f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input file: {input_file}\n")
            f.write(f"Model: {model_name}\n")
            f.write("="*80 + "\n\n")
        
        # Initialize LLM metrics log
        with open(self.llm_metrics_log, 'w') as f:
            f.write(f"LLM Metrics Log\n")
            f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input file: {input_file}\n")
            f.write(f"Model: {model_name}\n")
            f.write("="*80 + "\n\n")
    
    def _get_project_root(self) -> Path:
        """
        Get the absolute path to the project root directory.
        
        Returns:
            Path object representing the project root directory
        """
        # Method 1: Use the location of this file to determine the project root
        # Go up from utils/file_ops.py to the project root
        current_file = Path(inspect.getfile(inspect.currentframe())).resolve()
        utils_dir = current_file.parent
        project_root = utils_dir.parent
        
        # Verify this is the project root by checking for key files
        if (project_root / "main.py").exists() or (project_root / "setup.py").exists():
            return project_root
            
        # Method 2: Check if the module was imported from a specific location
        for module in sys.modules.values():
            if hasattr(module, '__file__') and module.__file__:
                module_path = Path(module.__file__).resolve()
                if "zettelkasten" in str(module_path):
                    # Find the zettelkasten directory
                    parts = module_path.parts
                    for i, part in enumerate(parts):
                        if part == "zettelkasten":
                            zettelkasten_path = Path(*parts[:i+1])
                            if (zettelkasten_path / "main.py").exists() or (zettelkasten_path / "setup.py").exists():
                                return zettelkasten_path
        
        # Method 3: Hardcoded fallback for this specific project
        # This ensures we always have a valid project root
        fallback_path = Path("/Users/zbodnar/Stories/brain_dump/zettelkasten")
        if fallback_path.exists():
            return fallback_path
            
        # Method 4: Last resort - use current working directory
        # but log a warning
        current_dir = Path.cwd().absolute()
        print(f"WARNING: Could not determine project root. Using current directory: {current_dir}")
        return current_dir
    
    def log_debug(self, message: str):
        """Log a debug message"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        with open(self.debug_log, 'a') as f:
            f.write(f"{timestamp} - {message}\n")
    
    def log_llm_interaction(self, message: str, prompt: str, response: str, metadata: dict = None):
        """
        Log an interaction with the LLM.
        
        Args:
            message: A message describing the interaction
            prompt: The prompt sent to the LLM
            response: The response received from the LLM
            metadata: Additional metadata about the interaction
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        with open(self.llm_interactions_log, 'a') as f:
            f.write(f"{timestamp} - {message}\n\n")
            f.write(f"PROMPT:\n{prompt}\n\n")
            f.write(f"RESPONSE:\n{response}\n\n")
            
            if metadata:
                f.write(f"METADATA:\n{json.dumps(metadata, indent=2)}\n")
            
            f.write("\n" + "="*80 + "\n\n")
    
    def log_llm_metrics(self, chunk_num: int, metrics: dict, metadata: dict = None):
        """
        Log metrics about LLM usage.
        
        Args:
            chunk_num: The chunk number
            metrics: Metrics to log
            metadata: Additional metadata
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        with open(self.llm_metrics_log, 'a') as f:
            f.write(f"{timestamp} - CHUNK: {chunk_num}\n")
            f.write(f"METRICS: {json.dumps(metrics, indent=2)}\n")
            
            if metadata:
                f.write(f"METADATA: {json.dumps(metadata, indent=2)}\n")
            
            f.write("\n" + "-"*40 + "\n\n")
    
    def get_chunk_path(self, chunk_num: int) -> str:
        """
        Get the path for a chunk file.
        
        Args:
            chunk_num: The chunk number
            
        Returns:
            Path to the chunk file
        """
        # Format chunk number with leading zeros
        formatted_num = f"{chunk_num:03d}"
        
        # Return path to chunk file with model name prefix
        return str(self.chunks_dir / f"{self.model_name}_chunk_{formatted_num}.md")
    
    def write_chunk_to_file(self, chunk_num: int, chunk_data: Dict) -> str:
        """Write a chunk to a markdown file with enhanced formatting.
        
        Args:
            chunk_num: The chunk number
            chunk_data: Dictionary containing chunk data
            
        Returns:
            str: Path to the created file
        """
        # Create output directory if it doesn't exist
        os.makedirs(self.chunks_dir, exist_ok=True)
        
        # Generate chunk filename
        chunk_filename = f"{self.model_name}_chunk_{chunk_num:03d}"
        chunk_path = os.path.join(self.chunks_dir, f"{chunk_filename}.md")
        
        # Extract data
        chunk_id = chunk_data.get("chunk_id", f"chunk_{chunk_num}")
        source_file = chunk_data.get("source_file", "unknown")
        text = chunk_data.get("text", "")
        facts = chunk_data.get("facts", [])
        tags = chunk_data.get("tags", [])
        topics = chunk_data.get("topics", [])
        entities = chunk_data.get("entities", [])
        window_num = chunk_data.get("window_num", chunk_num)
        total_windows = chunk_data.get("total_windows", 1)
        timestamp_range = chunk_data.get("timestamp_range", {})
        speakers = chunk_data.get("speakers", [])
        
        # Generate session ID if not present
        if not hasattr(self, 'session_id'):
            self.session_id = str(uuid.uuid4())
        
        # Create markdown content
        content = f"""---
id: {chunk_id}
source_file: {os.path.basename(source_file)}
session_id: {self.session_id}
chunk_type: atomic
created_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
---

# {chunk_id}

## Key Information
"""
        
        # Add facts with metadata
        if facts:
            content += "\n### Facts\n\n"
            for i, fact in enumerate(facts, 1):
                if isinstance(fact, dict):
                    fact_text = fact.get("text", "")
                    confidence = fact.get("confidence", 1.0)
                    source = fact.get("source", "extraction")
                    temporal_info = fact.get("temporal_info", {})
                    fact_entities = fact.get("entities", {})
                    
                    content += f"**Fact {i}** (confidence: {confidence:.2f})\n"
                    content += f"- {fact_text}\n"
                    
                    if temporal_info:
                        content += f"- *Temporal Context:* {temporal_info}\n"
                    
                    if fact_entities:
                        content += "- *Entities:*\n"
                        for entity_type, entities in fact_entities.items():
                            if entities:
                                content += f"  - {entity_type}: {', '.join(entities)}\n"
                    
                    content += "\n"
                else:
                    content += f"**Fact {i}**\n- {fact}\n\n"

        # Add topics and tags
        content += "\n## Topics and Tags\n"
        if topics:
            content += "\n**Topics:**\n"
            for topic in topics:
                content += f"- {topic}\n"
        
        if tags:
            content += "\n**Tags:**\n"
            for tag in tags:
                content += f"- {tag}\n"
        
        if entities:
            content += "\n**Named Entities:**\n"
            if isinstance(entities, dict):
                for entity_type, entity_list in entities.items():
                    if entity_list:
                        content += f"\n*{entity_type}:*\n"
                        for entity in entity_list:
                            content += f"- {entity}\n"
            else:
                for entity in entities:
                    content += f"- {entity}\n"

        # Add links to related chunks
        content += "\n## Related Chunks\n"
        if window_num > 1:
            content += f"\n- Previous: [{self.model_name}_chunk_{window_num-1:03d}.md]({self.model_name}_chunk_{window_num-1:03d}.md)"
        if window_num < total_windows:
            content += f"\n- Next: [{self.model_name}_chunk_{window_num+1:03d}.md]({self.model_name}_chunk_{window_num+1:03d}.md)"
        
        # Add semantic relationships if available
        if "related_chunks" in chunk_data and chunk_data["related_chunks"]:
            content += "\n\n**Semantically Related:**\n"
            for relation in chunk_data["related_chunks"]:
                chunk_id = relation.get("id", "")
                rel_type = relation.get("relationship_type", "")
                score = relation.get("score", 0.0)
                if chunk_id and rel_type and score:
                    content += f"- [{chunk_id}]({chunk_id}.md) ({rel_type}, score: {score:.2f})\n"

        # Add source information
        content += "\n## Source\n"
        content += f"\n- **File:** {source_file}"
        content += f"\n- **Chunk:** {window_num}/{total_windows}"
        
        if timestamp_range:
            start_time = timestamp_range.get("start")
            end_time = timestamp_range.get("end")
            if start_time is not None and end_time is not None:
                content += f"\n- **Time Range:** {start_time:.2f}s - {end_time:.2f}s"
        
        if speakers:
            content += "\n- **Speakers:** " + ", ".join(speakers)

        # Add original text in collapsible section
        content += "\n\n<details>\n<summary>Original Text</summary>\n\n```\n"
        content += text
        content += "\n```\n</details>\n"
        
        # Add metadata in collapsible section
        metadata = {
            "chunk_id": chunk_id,
            "source_file": source_file,
            "session_id": self.session_id,
            "processing_time": chunk_data.get("processing_time", 0),
            "window_num": window_num,
            "total_windows": total_windows,
            "tags_count": len(tags),
            "topics_count": len(topics),
            "facts_count": len(facts),
            "timestamp_range": timestamp_range,
            "speakers": speakers,
            "created_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        content += "\n<details>\n<summary>Metadata</summary>\n\n```json\n"
        content += json.dumps(metadata, indent=2)
        content += "\n```\n</details>\n"
        
        # Write markdown file
        with open(chunk_path, "w") as f:
            f.write(content)
        
        # Also save as JSON for programmatic access
        json_path = os.path.join(self.chunks_dir, f"{chunk_filename}.json")
        with open(json_path, "w") as f:
            json.dump(chunk_data, f, indent=2)
        
        return chunk_path
    
    def write_final_summary(self, metrics: dict):
        """
        Write a final summary of the processing.
        
        Args:
            metrics: Metrics to include in the summary
        """
        # Add timestamp
        metrics["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Write to JSON file
        with open(self.final_summary_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Also log to debug log with visual separator
        self.log_debug("\n" + "="*50)
        self.log_debug("FINAL PROCESSING SUMMARY")
        self.log_debug("="*50 + "\n")
        
        for key, value in metrics.items():
            self.log_debug(f"{key}: {value}")
        
        self.log_debug("\n" + "="*50)
        
        # Also write a human-readable version
        human_readable = self.output_dir / f"summary_{self.base_name}_{self.model_name}.txt"
        with open(human_readable, 'w') as f:
            f.write("PROCESSING SUMMARY\n")
            f.write("="*50 + "\n\n")
            f.write(f"Input file: {self.input_file}\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Completed: {metrics.get('timestamp', 'unknown')}\n\n")
            
            f.write("METRICS\n")
            f.write("-"*50 + "\n")
            
            for key, value in metrics.items():
                if key != "timestamp":
                    f.write(f"{key}: {value}\n")
            
            f.write("\n" + "="*50 + "\n")

    def _save_session_metadata(self):
        """Save session metadata to a file"""
        metadata_file = self.metadata_dir / f"session_{self.session_metadata['session_id']}.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.session_metadata, f, indent=2)

    def update_session_context(self, context: dict):
        """Update session context with new information"""
        self.session_metadata["context"].update(context)
        self._save_session_metadata()

    def update_transcript_metadata(self, metadata: dict):
        """Update transcript metadata with timestamp ranges, speakers, etc."""
        self.session_metadata["transcript_metadata"].update(metadata)
        self._save_session_metadata() 
"""
Utilities for loading test fixtures for chunking unit tests.
"""

import json
import os
import yaml
from pathlib import Path
from typing import Dict, List, Any, Union

# Constants for test fixtures
FIXTURES_DIR = Path(__file__).parent
INPUT_DIR = FIXTURES_DIR / "input"
EXPECTED_DIR = FIXTURES_DIR / "expected"
RESPONSES_DIR = FIXTURES_DIR / "responses"

def load_transcript(name: str) -> str:
    """Load a transcript test file by name."""
    file_path = INPUT_DIR / "transcripts" / f"{name}.txt"
    with open(file_path, "r") as f:
        return f.read()

def load_config(name: str) -> Dict[str, Any]:
    """Load a test config file by name."""
    file_path = INPUT_DIR / "config" / f"{name}.yaml"
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def load_windows(name: str) -> List[Dict[str, Any]]:
    """Load expected window data by name."""
    file_path = EXPECTED_DIR / "windows" / f"windows_{name}.json"
    with open(file_path, "r") as f:
        return json.load(f)

def load_expected_facts(name: str) -> List[Dict[str, Any]]:
    """Load expected facts by name."""
    file_path = EXPECTED_DIR / "first_pass" / f"{name}.json"
    with open(file_path, "r") as f:
        return json.load(f)

def load_expected_refined_facts(name: str) -> List[Dict[str, Any]]:
    """Load expected refined facts by name."""
    file_path = EXPECTED_DIR / "second_pass" / f"{name}.json"
    with open(file_path, "r") as f:
        return json.load(f)

def load_llm_response(response_type: str) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Load a mock LLM response by type."""
    file_path = RESPONSES_DIR / "llm" / f"{response_type}_response.json"
    with open(file_path, "r") as f:
        return json.load(f)

def load_fallback_response(response_type: str) -> Union[Dict[str, Any], List[Dict[str, Any]], str]:
    """Load a fallback response by type."""
    file_path = RESPONSES_DIR / "fallbacks" / f"{response_type}_response"
    
    # Handle both JSON and text files
    if os.path.exists(file_path + ".json"):
        with open(file_path + ".json", "r") as f:
            return json.load(f)
    else:
        with open(file_path + ".txt", "r") as f:
            return f.read()

# Function to create all transcript fixtures
def get_all_transcripts() -> List[str]:
    """Get a list of all available transcript names."""
    return [f.stem for f in (INPUT_DIR / "transcripts").glob("*.txt")]

# Function to create all config fixtures
def get_all_configs() -> List[str]:
    """Get a list of all available config names."""
    return [f.stem for f in (INPUT_DIR / "config").glob("*.yaml")]
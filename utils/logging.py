import logging
from pathlib import Path
from typing import Optional
import sys
import os

def get_logger(name: str, log_file: Optional[Path] = None) -> logging.Logger:
    """Get a logger with optional file output"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels
    
    # Clear existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers = []
    
    # Create a console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(console_handler)
    
    # Add file handler if log_file is provided
    if log_file:
        # Create logs directory if it doesn't exist
        log_dir = log_file.parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)
    else:
        # Create a default log file in the logs directory
        logs_dir = Path("logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        default_log_file = logs_dir / f"{name.replace('.', '_')}.log"
        file_handler = logging.FileHandler(default_log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)
    
    return logger 
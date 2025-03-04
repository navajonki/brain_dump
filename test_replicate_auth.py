#!/usr/bin/env python3
"""
Simple test script to verify Replicate authentication.
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

logger.info("Environment variables loaded from .env")

# Try reading the token directly from .env file
env_path = Path('.env')
direct_token = None

if env_path.exists():
    logger.info(f".env file exists at {env_path.absolute()}")
    with open(env_path, 'r') as f:
        for line in f:
            if line.strip().startswith('REPLICATE_API_TOKEN='):
                direct_token = line.strip().split('=', 1)[1].strip()
                # Remove quotes if present
                if direct_token.startswith('"') and direct_token.endswith('"'):
                    direct_token = direct_token[1:-1]
                elif direct_token.startswith("'") and direct_token.endswith("'"):
                    direct_token = direct_token[1:-1]
                    
                logger.info(f"Read token directly from .env file")
                break
else:
    logger.warning(".env file not found")

# Also check environment variable
env_token = os.environ.get("REPLICATE_API_TOKEN")
if env_token:
    logger.info("REPLICATE_API_TOKEN found in environment variables")
    masked_env = env_token[:4] + "..." + env_token[-4:] if len(env_token) > 8 else "****"
    logger.info(f"Env token: {masked_env}")
else:
    logger.warning("REPLICATE_API_TOKEN not found in environment variables")

# Use direct token if available, otherwise use environment variable
token = direct_token or env_token
if not token:
    logger.error("No Replicate API token found")
    sys.exit(1)

masked_token = token[:4] + "..." + token[-4:] if len(token) > 8 else "****"
logger.info(f"Using token: {masked_token}")

try:
    import replicate
    logger.info("Replicate package imported successfully")
except ImportError:
    logger.error("Failed to import replicate package")
    sys.exit(1)

# Test the client
try:
    # Create client with explicit token
    client = replicate.Client(api_token=token)
    logger.info("Created Replicate client with explicit token")
    
    # Test with a basic model call - something simple and fast
    logger.info("Testing with a simple model prediction...")
    
    # Try a well-established model - Llama 2 should be available
    output = client.run(
        "replicate/llama-2-70b-chat:2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1",
        input={
            "prompt": "Hello, world!",
            "system_prompt": "You are a helpful assistant.",
            "temperature": 0.01,
            "max_tokens": 10
        }
    )
    
    response = "".join(output)
    logger.info(f"Successfully received response: {response[:30]}...")
    logger.info("✅ Authentication test successful!")
    
except Exception as e:
    logger.error(f"Error with Replicate API: {e}")
    logger.error("❌ Authentication test failed")
    sys.exit(1)
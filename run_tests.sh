#!/bin/bash
# run_tests.sh - Script to run various tests with different backends and configurations
# Usage: ./run_tests.sh

# Set the default input file
INPUT_FILE="output/JCS/deepgram_nova-3/transcripts/JCS_transcript_short.txt"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print section headers
print_header() {
    echo -e "\n${YELLOW}========================================"
    echo -e "  $1"
    echo -e "========================================${NC}\n"
}

# Function to run a test and check its result
run_test() {
    local test_name="$1"
    local command="$2"
    
    print_header "Running $test_name"
    echo -e "Command: ${GREEN}$command${NC}\n"
    
    # Run the command
    eval $command
    
    # Check the result
    if [ $? -eq 0 ]; then
        echo -e "\n${GREEN}✓ Test completed successfully${NC}\n"
    else
        echo -e "\n${RED}✗ Test failed${NC}\n"
    fi
}

# Make sure the script is executable
chmod +x "$0"

# =============================================
# Direct Backend Tests
# =============================================

# Test Replicate backend directly
# run_test "Replicate Backend Direct Test" "python tests/test_replicate.py --direct"

# Test OpenAI backend (uncomment if you have a test script for it)
# run_test "OpenAI Backend Direct Test" "python tests/test_openai.py"

# Test Ollama backend (uncomment if you have a test script for it)
# run_test "Ollama Backend Direct Test" "python tests/test_ollama.py"

# =============================================
# Chunking Tests with Different Backends
# =============================================

# Test with Replicate backend - Mistral Instruct
# run_test "Chunking with Replicate (Mistral Instruct)" "python tests/test_chunking_v2.py $INPUT_FILE --config config/chunking/replicate_mistral_instruct.yaml"

# Test with Replicate backend - Llama 3 8B
run_test "Chunking with Replicate (Llama 3 8B)" "python tests/test_chunking_v2.py $INPUT_FILE --config config/chunking/replicate_llama3_8b.yaml"

# Test with OpenAI backend
# run_test "Chunking with OpenAI (GPT-3.5)" "python tests/test_chunking_v2.py $INPUT_FILE --config config/chunking/openai_gpt35.yaml"

# Test with Ollama backend
# run_test "Chunking with Ollama (Mistral)" "python tests/test_chunking_v2.py $INPUT_FILE --config config/chunking/ollama_mistral.yaml"

# =============================================
# Additional Tests
# =============================================

# Test with a different input file
# OTHER_INPUT="output/JCS/deepgram_nova-3/transcripts/JCS_transcript_full.txt"
# run_test "Chunking with Replicate (Different Input)" "python tests/test_chunking_v2.py $OTHER_INPUT --config config/chunking/replicate_llama3_8b.yaml"

# Test Replicate with streaming (direct test)
# run_test "Replicate Streaming Test" "python tests/test_replicate_streaming.py"

echo -e "\n${GREEN}All tests completed!${NC}" 
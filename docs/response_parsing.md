# Response Parsing

This document describes the response parsing system used in the chunking module.

## Overview

The response parsing system provides a robust way to handle, validate, and normalize responses from LLMs. It uses [Pydantic](https://pydantic-docs.helpmanual.io/) for schema validation and includes specialized parsers for different types of responses.

## Architecture

The system consists of three main components:

1. **JSONParser**: Core class for parsing and cleaning JSON responses
2. **Schema Models**: Pydantic models defining the expected structure of responses
3. **ResponseParserFactory**: Factory for creating specialized parsers

```
┌────────────────┐     ┌─────────────────┐
│  TextChunker   │────▶│ ParserFactory   │
└────────────────┘     └────────┬────────┘
                               │
                       ┌───────▼────────┐
                       │   JSONParser   │
                       └───────┬────────┘
                               │
                       ┌───────▼────────┐
                       │ Schema Models  │
                       └────────────────┘
```

## Key Features

### 1. Robust JSON Handling

The system handles various JSON formats and errors:

- Code block extraction (e.g., ```json ... ```)
- Malformed JSON repair
- Trailing/missing commas
- Unquoted keys
- JavaScript-style comments
- Single vs double quotes

### 2. Schema Validation

Uses Pydantic models to validate responses against expected schemas:

```python
class AtomicFact(BaseModel):
    """Schema for an atomic fact."""
    text: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source: str = "extraction"
    temporal_info: str = ""
    entities: Union[List[str], Dict[str, List[str]]] = []
```

### 3. Specialized Parsers

The factory provides parsers for different response types:

- `first_pass_parser`: For atomic fact extraction
- `second_pass_parser`: For fact refinement
- `tagging_parser`: For fact tagging
- `batch_tagging_parser`: For batch tagging
- `relationship_parser`: For relationship analysis
- `global_check_parser`: For consistency checking

## Usage

### Basic parsing

```python
from core.chunking.parsers import ResponseParserFactory

# Get a specialized parser
parser = ResponseParserFactory.get_first_pass_parser()

# Parse a response
parsed_data = parser.parse(llm_response)
```

### Custom schema validation

```python
from core.chunking.parsers import JSONParser
from pydantic import BaseModel, Field

# Define a custom schema
class CustomResponse(BaseModel):
    field1: str
    field2: int = Field(ge=0)
    field3: bool = False

# Create a parser with the custom schema
parser = JSONParser(schema=CustomResponse)

# Parse and validate
validated_data = parser.parse(response)
```

## Error Handling

The system provides graceful degradation:

1. First attempts to validate against schema
2. On validation failure, returns extracted JSON without validation
3. For text responses, extracts structured content (bullets, numbered lists)
4. As a last resort, returns the original response

## Extending the System

To add support for new response types:

1. Define a new Pydantic model in `parsers.py`
2. Add a factory method in `ResponseParserFactory`
3. Use the new parser in your code

For example:

```python
class NewResponseType(BaseModel):
    field1: str
    field2: int

# Add to factory
@staticmethod
def get_new_response_parser() -> JSONParser:
    return JSONParser(schema=NewResponseType)
```
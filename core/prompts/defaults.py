"""
Default prompt templates with metadata.

This module provides standard prompt definitions with proper documentation and
standardized parameter naming.
"""
from typing import Dict, Any

# General prompts
SUMMARY_AND_TOPICS = {
    "description": "Extract summary and topics from text",
    "required_params": ["text"],
    "version": "1.0",
    "tags": ["summary", "topics", "extraction"],
    "template": """
You are an assistant that extracts key information from text.
Given the following text delimited by triple backticks:
```
{text}
```
1. Write a concise summary (1-2 sentences).
2. Extract 3-5 top-level topics or keywords that categorize this text.

Return your answer in JSON format like:
{
  "summary": "...",
  "topics": ["Topic1", "Topic2", ...]
}
"""
}

TOPIC_AGGREGATION = {
    "description": "Aggregate multiple summaries into a unified topic summary",
    "required_params": ["topic", "summaries"],
    "version": "1.0",
    "tags": ["summary", "topics", "aggregation"],
    "template": """
The topic is: {topic}
Below are the summaries of various chunks related to this topic:
{summaries}

Please create one cohesive summary (2-3 sentences) that unifies these points.
"""
}

TOPIC_CHANGE = {
    "description": "Analyze if a sentence represents a topic change",
    "required_params": ["text", "next_sentence"],
    "version": "1.0",
    "tags": ["topic", "segmentation"],
    "template": """Analyze if the following sentence represents a topic change from the previous text. 
Keep in mind that this may be a part of a conversation. 
Utterances shouldn't be considered new topics.
Consider both content and context. Respond with only 'true' or 'false'.

Previous text: "{text}"

Next sentence: "{next_sentence}"

Is this a new topic?"""
}

SPLIT_SENTENCE = {
    "description": "Split a long sentence into smaller, coherent parts",
    "required_params": ["text"],
    "version": "1.0",
    "tags": ["sentence", "segmentation"],
    "template": """Split this long sentence into smaller, coherent parts while preserving meaning.
Format each part as a numbered list.

Sentence: "{text}"
"""
}

# Atomic chunking prompts
FIRST_PASS = {
    "description": "Extract atomic facts from a text window",
    "required_params": ["window_text"],
    "version": "1.0",
    "tags": ["chunking", "extraction", "atomic"],
    "template": """Here is a portion of a transcript:

{window_text}

List all distinct, atomic pieces of information found in this text, each as a separate bullet point. 
Include any relevant metadata (e.g., references to people, places, or times).
Focus on extracting factual information, not interpretations.
Each bullet point should contain exactly one piece of information.
Be comprehensive - don't miss any details mentioned in the text.


Sometimes in spoken conversation, people will use present tense to refer to something that happened in the past. Prioritize consistency of the overall timeline over the exact wording of the past tense.
Important: Make each fact stand on its own without ambiguity. Avoid pronouns like "he", "she", or "it" - instead use the specific names or objects they refer to.

Format your response as a simple bullet point list with each fact on a new line starting with '- '.
"""
}

SECOND_PASS = {
    "description": "Extract additional atomic facts from a text window",
    "required_params": ["window_text", "facts_text"],
    "version": "1.0",
    "tags": ["chunking", "extraction", "atomic"],
    "template": """Here is a portion of a transcript:

{window_text}

And here is the list of atomic facts already extracted from this text:

{facts_text}

Is there any additional context regarding the people, places, or timeline mentioned in the text that is not captured above?
List any further atomic facts that were missed in the first extraction.
Each fact should be a single, atomic piece of information.

Important: Make each fact stand on its own without ambiguity. Avoid pronouns like "he", "she", or "it" - instead use the specific names or objects they refer to.

Format your response as a simple bullet point list with each new fact on a new line starting with '- '.
If no additional facts are needed, respond with "No additional facts needed."
"""
}

GLOBAL_CHECK = {
    "description": "Ensure continuity and resolve contradictions across facts",
    "required_params": ["facts_text"],
    "version": "1.0",
    "tags": ["chunking", "validation", "atomic"],
    "template": """Here is a list of atomic facts extracted from a transcript:

{facts_text}

Please review these facts and:
1. Identify any contradictory facts that need resolution
2. Identify any facts that are related and should be linked
3. Identify any redundant facts that could be merged

Format your response as a JSON object with the following structure:
{
  "contradictions": [
    {"fact1": "...", "fact2": "...", "resolution": "..."}
  ],
  "related_facts": [
    {"facts": ["...", "..."], "relationship": "..."}
  ],
  "redundant_facts": [
    {"facts": ["...", "..."], "merged_fact": "..."}
  ]
}

If there are no issues to report in a category, include an empty array.
"""
}

TAGGING = {
    "description": "Tag and categorize an atomic fact",
    "required_params": ["fact"],
    "version": "1.0",
    "tags": ["chunking", "tagging", "atomic"],
    "template": """Here is an atomic fact extracted from a transcript:

{fact}

Please analyze this fact and:
1. Identify 3-5 relevant tags that categorize this information (e.g., "personal_history", "education", "career", "family", "opinion", etc.)
2. Determine the main topic this fact relates to
3. Identify any entities mentioned (people, places, organizations, etc.)

Format your response as a JSON object with the following structure:
{
  "tags": ["tag1", "tag2", "tag3"],
  "topic": "main_topic",
  "entities": {
    "people": ["name1", "name2"],
    "places": ["place1", "place2"],
    "organizations": ["org1", "org2"],
    "other": ["entity1", "entity2"]
  }
}
"""
}

RELATIONSHIP = {
    "description": "Analyze relationships between atomic facts",
    "required_params": ["facts_text"],
    "version": "1.0",
    "tags": ["chunking", "relationship", "atomic"],
    "template": """Here is a set of atomic facts extracted from a transcript:

{facts_text}

Please analyze these facts and identify relationships between them. For each fact, identify which other facts are directly related to it.

Format your response as a JSON object where each key is the index of a fact (starting from 1), and the value is an array of indices of related facts:

{
  "1": [3, 5, 8],
  "2": [4, 7],
  ...
}

Two facts are related if:
1. They refer to the same person, place, or event
2. One fact provides context or additional information about another
3. They are part of the same narrative thread or topic
4. They have a cause-effect relationship

Only include strong relationships, not tenuous connections.
"""
}

# Model-specific prompts
MISTRAL_PROMPTS = {
    "first_pass": {
        "description": "Extract atomic facts from a text window using Mistral function calling",
        "required_params": ["window_text"],
        "model": "mistral",
        "version": "1.0",
        "tags": ["chunking", "extraction", "atomic", "function-calling"],
        "template": """[AVAILABLE_TOOLS]
[{{
    "type": "function",
    "function": {{
        "name": "extract_atomic_facts",
        "description": "Extract distinct, atomic pieces of information from text",
        "parameters": {{
            "type": "object",
            "properties": {{
                "facts": {{
                    "type": "array",
                    "items": {{
                        "type": "object",
                        "properties": {{
                            "fact": {{
                                "type": "string",
                                "description": "A single, atomic piece of information"
                            }},
                            "confidence": {{
                                "type": "number",
                                "description": "Confidence score between 0 and 1"
                            }}
                        }},
                        "required": ["fact", "confidence"]
                    }},
                    "description": "List of extracted atomic facts"
                }}
            }},
            "required": ["facts"]
        }}
    }}
}}]
[/AVAILABLE_TOOLS]
[INST]
Extract all distinct, atomic pieces of information from this text. Each fact should:
- Contain exactly one piece of information
- Be self-contained without ambiguity
- Use specific names instead of pronouns
- Include relevant metadata (people, places, times)
- Maintain timeline consistency

Here is the text:

{window_text}
[/INST]"""
    },
    
    "second_pass": {
        "description": "Extract additional atomic facts using Mistral function calling",
        "required_params": ["window_text", "facts_text"],
        "model": "mistral",
        "version": "1.0",
        "tags": ["chunking", "extraction", "atomic", "function-calling"],
        "template": """[AVAILABLE_TOOLS]
[{{
    "type": "function",
    "function": {{
        "name": "extract_additional_facts",
        "description": "Extract additional atomic facts not covered in the first pass",
        "parameters": {{
            "type": "object",
            "properties": {{
                "additional_facts": {{
                    "type": "array",
                    "items": {{
                        "type": "object",
                        "properties": {{
                            "fact": {{
                                "type": "string",
                                "description": "A single, atomic piece of information"
                            }},
                            "related_to": {{
                                "type": "array",
                                "items": {{
                                    "type": "string"
                                }},
                                "description": "List of original facts this new fact provides context for"
                            }}
                        }},
                        "required": ["fact"]
                    }},
                    "description": "List of additional atomic facts"
                }}
            }},
            "required": ["additional_facts"]
        }}
    }}
}}]
[/AVAILABLE_TOOLS]
[INST]
Here is a portion of text and the facts already extracted from it.

Text:
{window_text}

Already extracted facts:
{facts_text}

Extract any additional context or atomic facts that were missed in the first extraction. Focus on:
- Implicit information that provides important context
- Relationships between people, places, or events
- Timeline details that weren't explicitly captured
- Background information that helps understand the facts

If no additional facts are needed, return an empty list.
[/INST]"""
    }
}

# Dictionary of all default prompts
DEFAULT_PROMPTS = {
    # General prompts
    "summary_and_topics": SUMMARY_AND_TOPICS,
    "topic_aggregation": TOPIC_AGGREGATION,
    "topic_change": TOPIC_CHANGE,
    "split_sentence": SPLIT_SENTENCE,
    
    # Atomic chunking prompts
    "first_pass": FIRST_PASS,
    "second_pass": SECOND_PASS,
    "global_check": GLOBAL_CHECK,
    "tagging": TAGGING,
    "relationship": RELATIONSHIP,
}

# Dictionary of model-specific prompts
MODEL_SPECIFIC_PROMPTS = {
    "mistral": {
        "first_pass": MISTRAL_PROMPTS["first_pass"],
        "second_pass": MISTRAL_PROMPTS["second_pass"],
    }
}
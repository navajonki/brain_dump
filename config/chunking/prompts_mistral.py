"""
Prompt templates for the atomic chunking process using Mistral's function calling capabilities.
These prompts are designed to work with Mistral 0.3+ and use the function calling format.
"""

FIRST_PASS_PROMPT = """[AVAILABLE_TOOLS]
[{
    "type": "function",
    "function": {
        "name": "extract_atomic_facts",
        "description": "Extract distinct, atomic pieces of information from text",
        "parameters": {
            "type": "object",
            "properties": {
                "facts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "fact": {
                                "type": "string",
                                "description": "A single, atomic piece of information"
                            },
                            "confidence": {
                                "type": "number",
                                "description": "Confidence score between 0 and 1"
                            }
                        },
                        "required": ["fact", "confidence"]
                    },
                    "description": "List of extracted atomic facts"
                }
            },
            "required": ["facts"]
        }
    }
}]
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

SECOND_PASS_PROMPT = """[AVAILABLE_TOOLS]
[{
    "type": "function",
    "function": {
        "name": "extract_additional_facts",
        "description": "Extract additional atomic facts not covered in the first pass",
        "parameters": {
            "type": "object",
            "properties": {
                "additional_facts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "fact": {
                                "type": "string",
                                "description": "A single, atomic piece of information"
                            },
                            "related_to": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                },
                                "description": "List of original facts this new fact provides context for"
                            }
                        },
                        "required": ["fact"]
                    },
                    "description": "List of additional atomic facts"
                }
            },
            "required": ["additional_facts"]
        }
    }
}]
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

TAGGING_PROMPT = """[AVAILABLE_TOOLS]
[{
    "type": "function",
    "function": {
        "name": "analyze_fact",
        "description": "Analyze a fact to extract tags, topics, and entities",
        "parameters": {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "3-5 relevant tags categorizing this information"
                },
                "topic": {
                    "type": "string",
                    "description": "The main topic this fact relates to"
                },
                "entities": {
                    "type": "object",
                    "properties": {
                        "people": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Names of people mentioned"
                        },
                        "places": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Places mentioned"
                        },
                        "organizations": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Organizations mentioned"
                        },
                        "dates": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Dates or time periods mentioned"
                        },
                        "other": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Other notable entities"
                        }
                    },
                    "required": ["people", "places", "organizations", "dates", "other"]
                }
            },
            "required": ["tags", "topic", "entities"]
        }
    }
}]
[/AVAILABLE_TOOLS]
[INST]
Analyze this atomic fact and extract tags, topics, and entities:

{fact}
[/INST]"""

RELATIONSHIP_PROMPT = """[AVAILABLE_TOOLS]
[{
    "type": "function",
    "function": {
        "name": "analyze_relationships",
        "description": "Analyze relationships between facts",
        "parameters": {
            "type": "object",
            "properties": {
                "relationships": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "fact_index": {
                                "type": "integer",
                                "description": "Index of the fact being analyzed"
                            },
                            "related_facts": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "index": {
                                            "type": "integer",
                                            "description": "Index of the related fact"
                                        },
                                        "relationship_type": {
                                            "type": "string",
                                            "enum": ["provides_context", "same_event", "cause_effect", "same_entity", "temporal"],
                                            "description": "Type of relationship between the facts"
                                        },
                                        "strength": {
                                            "type": "number",
                                            "minimum": 0,
                                            "maximum": 1,
                                            "description": "Strength of the relationship (0-1)"
                                        }
                                    },
                                    "required": ["index", "relationship_type", "strength"]
                                }
                            }
                        },
                        "required": ["fact_index", "related_facts"]
                    }
                }
            },
            "required": ["relationships"]
        }
    }
}]
[/AVAILABLE_TOOLS]
[INST]
Analyze the relationships between these facts. For each fact, identify which other facts are related and how:

{facts_text}

Consider these types of relationships:
- provides_context: One fact provides background or context for another
- same_event: Facts describe the same event
- cause_effect: One fact led to or caused another
- same_entity: Facts involve the same person, place, or thing
- temporal: Facts are temporally related (before/after)

Only include strong relationships (strength > 0.5).
[/INST]"""

GLOBAL_CHECK_PROMPT = """[AVAILABLE_TOOLS]
[{
    "type": "function",
    "function": {
        "name": "global_fact_check",
        "description": "Perform global consistency check on extracted facts",
        "parameters": {
            "type": "object",
            "properties": {
                "redundancies": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "facts": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of redundant facts"
                            },
                            "merged_fact": {
                                "type": "string",
                                "description": "Single fact that combines the information"
                            }
                        },
                        "required": ["facts", "merged_fact"]
                    }
                },
                "contradictions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "facts": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of contradicting facts"
                            },
                            "resolution": {
                                "type": "string",
                                "description": "Resolved version of the facts"
                            },
                            "confidence": {
                                "type": "number",
                                "description": "Confidence in the resolution (0-1)"
                            }
                        },
                        "required": ["facts", "resolution", "confidence"]
                    }
                },
                "timeline_issues": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "facts": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Facts with timeline inconsistencies"
                            },
                            "issue": {
                                "type": "string",
                                "description": "Description of the timeline issue"
                            },
                            "correction": {
                                "type": "string",
                                "description": "Corrected version maintaining timeline consistency"
                            }
                        },
                        "required": ["facts", "issue", "correction"]
                    }
                }
            },
            "required": ["redundancies", "contradictions", "timeline_issues"]
        }
    }
}]
[/AVAILABLE_TOOLS]
[INST]
Perform a global consistency check on these extracted facts:

{facts_text}

Check for:
1. Redundant facts that could be merged
2. Contradicting information that needs resolution
3. Timeline inconsistencies that need correction

Provide specific corrections and maintain high confidence in any changes.
[/INST]""" 
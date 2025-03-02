"""
Default prompts for chunking operations.

This module contains the default prompts used by the chunking system.
"""

# First pass prompt to extract atomic facts from a window
FIRST_PASS_PROMPT = """Here is a portion of a transcript:

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

# Second pass prompt to extract additional atomic facts from a window
SECOND_PASS_PROMPT = """Here is a portion of a transcript:

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

# Global check prompt to ensure continuity and resolve contradictions
GLOBAL_CHECK_PROMPT = """Here is a list of atomic facts extracted from a transcript:

{facts_text}

Please review these facts and:
1. Identify any contradictory facts that need resolution
2. Identify any facts that are related and should be linked
3. Identify any redundant facts that could be merged

Format your response as a JSON object with the following structure:
{{
  "contradictions": [
    {{"fact1": "...", "fact2": "...", "resolution": "..."}}
  ],
  "related_facts": [
    {{"facts": ["...", "..."], "relationship": "..."}}
  ],
  "redundant_facts": [
    {{"facts": ["...", "..."], "merged_fact": "..."}}
  ]
}}

If there are no issues to report in a category, include an empty array.
"""

# Tagging and topic identification prompt
TAGGING_PROMPT = """Here is an atomic fact extracted from a transcript:

{fact}

Please analyze this fact and:
1. Identify 3-5 relevant tags that categorize this information (e.g., "personal_history", "education", "career", "family", "opinion", etc.)
2. Determine the main topic this fact relates to
3. Identify any entities mentioned (people, places, organizations, etc.)

Format your response as a JSON object with the following structure:
{{
  "tags": ["tag1", "tag2", "tag3"],
  "topic": "main_topic",
  "entities": {{
    "people": ["name1", "name2"],
    "places": ["place1", "place2"],
    "organizations": ["org1", "org2"],
    "other": ["entity1", "entity2"]
  }}
}}
"""

# Relationship analysis prompt
RELATIONSHIP_PROMPT = """Here is a set of atomic facts extracted from a transcript:

{facts_text}

Please analyze these facts and identify relationships between them. For each fact, identify which other facts are directly related to it.

Format your response as a JSON object where each key is the index of a fact (starting from 1), and the value is an array of indices of related facts:
"""

# Batch tagging prompt for tagging multiple facts at once
BATCH_TAGGING_PROMPT = """Here are multiple atomic facts extracted from a transcript:

{facts}

Please analyze EACH fact and provide the following for EACH one:
1. Identify 3-5 relevant tags that categorize the information (e.g., "personal_history", "education", "career", "family", "opinion", etc.)
2. Determine the main topic each fact relates to
3. Identify any entities mentioned (people, places, organizations, etc.)
4. Assess the sentiment (positive, negative, or neutral)
5. Rate the importance on a scale of 1-10 (where 10 is extremely important)

Format your response as a JSON object with the following structure:

{
  "fact_tags": {
    "1": {
      "tags": ["tag1", "tag2", "tag3"],
      "topics": ["main_topic"],
      "entities": {
        "people": ["name1", "name2"],
        "places": ["place1", "place2"],
        "organizations": ["org1", "org2"],
        "other": ["entity1", "entity2"]
      },
      "sentiment": "positive/negative/neutral",
      "importance": 7
    },
    "2": {
      "tags": ["tag1", "tag2", "tag3"],
      "topics": ["main_topic"],
      "entities": {
        "people": ["name1", "name2"],
        "places": ["place1", "place2"],
        "organizations": ["org1", "org2"],
        "other": ["entity1", "entity2"]
      },
      "sentiment": "positive/negative/neutral",
      "importance": 5
    },
    // ... and so on for each fact
  }
}

Where the keys "1", "2", etc. correspond to the fact numbers in the list above.
IMPORTANT: Make sure to include an entry for EACH fact in the input list.
"""

{{
  "1": [3, 5, 8],
  "2": [4, 7],
  ...
}}

Two facts are related if:
1. They refer to the same person, place, or event
2. One fact provides context or additional information about another
3. They are part of the same narrative thread or topic
4. They have a cause-effect relationship

Only include strong relationships, not tenuous connections.
""" 
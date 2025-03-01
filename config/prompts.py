SUMMARY_AND_TOPICS_PROMPT = """
You are an assistant that extracts key information from text.
Given the following text delimited by triple backticks:
```
{text}
```
1. Write a concise summary (1-2 sentences).
2. Extract 3-5 top-level topics or keywords that categorize this text.

Return your answer in JSON format like:
{{
  "summary": "...",
  "topics": ["Topic1", "Topic2", ...]
}}
"""

TOPIC_AGGREGATION_PROMPT = """
The topic is: {topic}
Below are the summaries of various chunks related to this topic:
{summaries}

Please create one cohesive summary (2-3 sentences) that unifies these points.
"""

TOPIC_CHANGE_PROMPT = """Analyze if the following sentence represents a topic change from the previous text. 
Keep in mind that this may be a part of a conversation. 
Utterances shouldn't be considered new topics.
Consider both content and context. Respond with only 'true' or 'false'.

Previous text: "{text}"

Next sentence: "{next_sentence}"

Is this a new topic?"""

SPLIT_SENTENCE_PROMPT = """Split this long sentence into smaller, coherent parts while preserving meaning.
Format each part as a numbered list.

Sentence: "{text}"
""" 
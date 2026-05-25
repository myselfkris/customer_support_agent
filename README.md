# YouTube Reply Identifier Agent

This agent takes comments from one YouTube promo video and identifies who the creator should reply to first.

## Project Structure
- `raw_comments.csv`: Contains the exported raw YouTube comments from a public video (includes commenter name, text, timestamp, etc.).
- `main.py`: Script to load raw comments, call the LLM to analyze them, and output prioritized response suggestions.
- `prompt.txt`: The system prompt instructing the LLM on how to classify the comments, assess buying intent, and formulate replies.
- `README.md`: Project scope and setup instructions.

## Locked Output Schema
The prioritized responses will have the following fixed columns:
- `commenter_name`: The name or handle of the YouTube user.
- `comment_text`: The raw text of their comment.
- `intent_level`: Analysis of their buying or enrollment intent (`High`, `Medium`, `Low`, or `None`).
- `intent_reason`: The rationale behind the assigned intent level.
- `recommended_reply`: A drafted response to their query.
- `urgency`: The response priority (`High`, `Medium`, or `Low`).
- `topic_or_course`: The specific topic or course component they are asking about.

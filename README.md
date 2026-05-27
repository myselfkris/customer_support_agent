# YouTube Reply Identifier Agent

This agent is a two-part pipeline that extracts comments from a YouTube video and uses an LLM to identify high-value comments that the creator should reply to first.

## Project Structure
- `extract_comments.py`: Script to fetch all comments (including deep replies) from a YouTube video using the official YouTube Data API v3 and save them as `raw_comments.csv`.
- `raw_comments.csv`: Contains the exported raw YouTube comments (includes commenter name, text, parent IDs, timestamp, etc.).
- `main.py`: Script to load raw comments, call the LLM to analyze them, and output prioritized response suggestions.
- `prompt.txt`: The system prompt instructing the LLM on how to classify the comments, assess buying intent, and formulate replies.
- `README.md`: Project scope and setup instructions.

## Pipeline Usage

### Step 1: Extract Comments
Run the extraction script to download the raw comments from a video:
```bash
pip install google-api-python-client
set YOUTUBE_API_KEY=your_youtube_api_key
python extract_comments.py --url "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Step 2: Analyze Comments
Run the main script to process comments through the LLM agent:
```bash
python main.py
```

## Locked Output Schema
The prioritized responses will have the following fixed columns:
- `commenter_name`: The name or handle of the YouTube user.
- `comment_text`: The raw text of their comment.
- `intent_level`: Analysis of their buying or enrollment intent (`High`, `Medium`, `Low`, or `None`).
- `intent_reason`: The rationale behind the assigned intent level.
- `recommended_reply`: A drafted response to their query.
- `urgency`: The response priority (`High`, `Medium`, or `Low`).
- `topic_or_course`: The specific topic or course component they are asking about.


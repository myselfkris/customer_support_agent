# Skill 6: Reliability + Memory Notes

Implemented in `agents/reliable_agent.py`.

## What Changed

- Added `ReliableAgent`, a wrapper around the Skill 5 LangGraph app.
- Added input validation for empty messages and messages over 5,000 characters.
- Added lightweight prompt-injection refusal for obvious attempts to override system/developer instructions.
- Added short-term per-user memory for the last 10 conversation turns.
- Added per-user in-memory rate limiting.
- Added retry logic around transient agent failures.
- Added graceful fallback after retries are exhausted.
- Added structured JSONL logs at `logs/skill6_agent_events.jsonl`.
- Added an offline failure-injection eval that does not call Gemini.

## How To Run

```bash
python agents/reliable_agent.py --eval
```

For live interactive mode:

```bash
python agents/reliable_agent.py
```

## Clear Note

Skill 6 is intentionally a wrapper, not a rewrite of Skill 5. The LangGraph agent still owns classification, routing, tools, RAG, and escalation. The new Skill 6 layer owns operational behavior around the agent: validating inputs before they reach the model, remembering recent conversation context, slowing abusive traffic, retrying temporary failures, falling back cleanly, and logging every turn for debugging.

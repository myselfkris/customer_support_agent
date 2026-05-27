"""
main.py
--------
YouTube Reply Identifier Agent

Pipeline:
  1. Load raw_comments.csv
  2. Load system prompt from prompt.txt
  3. Send ALL comments in ONE single API call to Gemini
  4. Parse the returned JSON array
  5. Save structured results to analyzed_comments.csv

Usage:
  python main.py

Output:
  analyzed_comments.csv  <- same folder, ready to review
"""

import os
import csv
import json
import re

# ─────────────────────────────────────────────────────────────
# Load local .env file if present
# ─────────────────────────────────────────────────────────────
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
if os.path.exists(_env_path):
    with open(_env_path, 'r', encoding='utf-8') as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith('#') and '=' in _line:
                _key, _val = _line.split('=', 1)
                os.environ[_key.strip()] = _val.strip().strip('\'"')

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL_NAME     = "gemini-2.5-flash"   # 1 call for all comments = no quota issues

# Output CSV columns (locked schema)
OUTPUT_COLUMNS = [
    "commenter_name",
    "comment_text",
    "intent_level",
    "intent_reason",
    "recommended_reply",
    "urgency",
    "topic_or_course",
]


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def load_comments(csv_path: str) -> list[dict]:
    """Loads all rows from raw_comments.csv."""
    comments = []
    if not os.path.exists(csv_path):
        print(f"[ERROR] File not found: {csv_path}")
        return comments
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            comments.append(row)
    return comments


def load_prompt(prompt_path: str) -> str:
    """Loads the system prompt from prompt.txt."""
    if not os.path.exists(prompt_path):
        print(f"[ERROR] Prompt file not found: {prompt_path}")
        return ""
    with open(prompt_path, mode='r', encoding='utf-8') as f:
        return f.read().strip()


def build_batch_message(comments: list[dict]) -> str:
    """
    Formats all comments into a single user message.
    Each comment is clearly numbered and labeled so the model
    can process them all at once and return an ordered JSON array.
    """
    lines = [f"Analyze the following {len(comments)} YouTube comments:\n"]
    for i, row in enumerate(comments, start=1):
        name = row.get('commenter_name', '').strip()
        text = row.get('comment_text', '').strip()
        lines.append(f"--- Comment {i} ---")
        lines.append(f"commenter_name: {name}")
        lines.append(f"comment_text: {text}")
        lines.append("")
    return "\n".join(lines)


def parse_json_response(raw_text: str) -> list[dict] | None:
    """
    Parses the model's response as a JSON array.
    Handles accidental markdown fences gracefully.
    Returns None if parsing fails.
    """
    text = raw_text.strip()

    # Strip markdown fences if the model added them despite instructions
    if text.startswith("```"):
        lines = text.splitlines()
        # Remove opening and closing fence lines
        inner = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```"):
                continue
            inner.append(line)
        text = "\n".join(inner).strip()

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        # Model returned a dict instead of array — wrap it
        if isinstance(data, dict):
            return [data]
        return None
    except json.JSONDecodeError as e:
        # Try to extract a JSON array using regex as a last resort
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        print(f"[PARSE ERROR] {e}")
        print(f"[RAW RESPONSE SNIPPET] {text[:300]}")
        return None


def call_gemini_batch(client, system_prompt: str, user_message: str) -> tuple[str, str]:
    """
    Sends a single batch request to the Gemini API for all comments.
    Returns (raw_text, finish_reason).
    """
    from google.genai import types

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.2,
            max_output_tokens=16384,  # enough for 52 detailed comment analyses
            thinking_config=types.ThinkingConfig(
                thinking_budget=0,    # disable thinking tokens — they eat output budget
            ),
        ),
    )
    # Extract finish reason for truncation detection
    finish_reason = "UNKNOWN"
    try:
        finish_reason = str(response.candidates[0].finish_reason)
    except Exception:
        pass
    return response.text, finish_reason


def save_results(results: list[dict], output_path: str):
    """Writes all analyzed comment rows to analyzed_comments.csv."""
    with open(output_path, mode='w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=OUTPUT_COLUMNS,
            extrasaction='ignore',
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"[OK] Saved {len(results)} rows -> {output_path}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    base_dir    = os.path.dirname(os.path.abspath(__file__))
    csv_path    = os.path.join(base_dir, 'raw_comments.csv')
    prompt_path = os.path.join(base_dir, 'prompt.txt')
    output_path = os.path.join(base_dir, 'analyzed_comments.csv')

    print("=" * 55)
    print("  YouTube Reply Identifier Agent")
    print("=" * 55)

    # --- Validate API key ---
    if not GEMINI_API_KEY:
        print("[ERROR] GEMINI_API_KEY is not set. Add it to your .env file.")
        return

    # --- Load inputs ---
    print(f"\n[1/4] Loading comments from: {csv_path}")
    comments = load_comments(csv_path)
    if not comments:
        print("[ERROR] No comments loaded. Exiting.")
        return
    print(f"      Loaded {len(comments)} rows.")

    print(f"\n[2/4] Loading prompt from:   {prompt_path}")
    system_prompt = load_prompt(prompt_path)
    if not system_prompt:
        print("[ERROR] System prompt is empty. Exiting.")
        return
    print(f"      Prompt loaded ({len(system_prompt)} chars).")

    # --- Initialize Gemini client ---
    try:
        from google import genai
        client = genai.Client(api_key=GEMINI_API_KEY)
    except ImportError:
        print("[ERROR] Missing library. Run:  pip install google-genai")
        return

    # --- Build the single batch message ---
    print(f"\n[3/4] Sending all {len(comments)} comments in ONE API call to {MODEL_NAME}...")
    user_message = build_batch_message(comments)

    try:
        raw_response, finish_reason = call_gemini_batch(client, system_prompt, user_message)
        print(f"      Response received ({len(raw_response)} chars, finish_reason={finish_reason}).")
        if "MAX_TOKENS" in finish_reason or "LENGTH" in finish_reason:
            print("[WARNING] Response was cut off by token limit! Partial results may be saved.")
    except Exception as e:
        print(f"[ERROR] API call failed: {e}")
        return

    # --- Parse response ---
    print("\n[4/4] Parsing JSON response...")
    results = parse_json_response(raw_response)

    if not results:
        print("[ERROR] Could not parse JSON from model response.")
        print("Raw response saved to: raw_response_debug.txt")
        with open(os.path.join(base_dir, 'raw_response_debug.txt'), 'w', encoding='utf-8') as f:
            f.write(raw_response)
        return

    print(f"      Parsed {len(results)} comment analyses.")

    # Ensure all required keys are present in every row
    for row in results:
        for col in OUTPUT_COLUMNS:
            row.setdefault(col, "")

    # --- Save results ---
    save_results(results, output_path)

    # --- Summary table ---
    print("\n" + "=" * 55)
    high   = sum(1 for r in results if r.get('urgency') == 'High')
    medium = sum(1 for r in results if r.get('urgency') == 'Medium')
    low    = sum(1 for r in results if r.get('urgency') == 'Low')
    print(f"  Total analyzed : {len(results)}")
    print(f"  High urgency   : {high}")
    print(f"  Medium urgency : {medium}")
    print(f"  Low urgency    : {low}")
    print(f"\n  Output file    : analyzed_comments.csv")
    print("=" * 55)


if __name__ == "__main__":
    main()

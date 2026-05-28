"""
main.py
--------
YouTube Reply Identifier Agent

Pipeline:
  1. Load raw_comments.csv
  2. Load system prompt from prompt.txt
  3. Send comments in BATCHES to Gemini (default: 10 per batch)
  4. Parse the returned JSON arrays from each batch
  5. Merge all batch results and save to analyzed_comments.csv

Usage:
  python main.py

Output:
  analyzed_comments.csv  <- same folder, ready to review
"""

import os
import csv
import json
import re
import time

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
MODEL_NAME     = "gemini-2.5-flash"

# Batching config — how many comments to send per API call
BATCH_SIZE = 10   # Safe for output tokens; each batch produces ~1,600 output tokens

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
    Formats a batch of comments into a single user message.
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
    Sends a single batch request to the Gemini API.
    Returns (raw_text, finish_reason).
    """
    from google.genai import types

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.2,
            max_output_tokens=4096,  # Enough for ~10 comments per batch
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

    # --- Split comments into batches ---
    total = len(comments)
    num_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE  # ceiling division
    print(f"\n[3/4] Processing {total} comments in {num_batches} batches (batch size = {BATCH_SIZE})...")

    all_results = []
    failed_batches = []

    for batch_num in range(num_batches):
        start_idx = batch_num * BATCH_SIZE
        end_idx   = min(start_idx + BATCH_SIZE, total)
        batch     = comments[start_idx:end_idx]

        print(f"\n   -- Batch {batch_num + 1}/{num_batches} (comments {start_idx + 1}-{end_idx}) --")

        user_message = build_batch_message(batch)

        try:
            raw_response, finish_reason = call_gemini_batch(client, system_prompt, user_message)
            print(f"      Response: {len(raw_response)} chars, finish_reason={finish_reason}")

            if "MAX_TOKENS" in finish_reason or "LENGTH" in finish_reason:
                print(f"      [WARNING] Batch {batch_num + 1} was truncated by token limit!")
                failed_batches.append(batch_num + 1)

            # Parse this batch's response
            batch_results = parse_json_response(raw_response)

            if batch_results:
                print(f"      Parsed {len(batch_results)} analyses.")
                all_results.extend(batch_results)
            else:
                print(f"      [ERROR] Failed to parse batch {batch_num + 1}. Skipping.")
                failed_batches.append(batch_num + 1)

                # Save the raw response for debugging
                debug_file = os.path.join(base_dir, f'raw_response_debug_batch{batch_num + 1}.txt')
                with open(debug_file, 'w', encoding='utf-8') as f:
                    f.write(raw_response)
                print(f"      Raw response saved to: {debug_file}")

        except Exception as e:
            print(f"      [ERROR] API call failed for batch {batch_num + 1}: {e}")
            failed_batches.append(batch_num + 1)

        # Small delay between batches to respect rate limits (free tier: 15 RPM)
        if batch_num < num_batches - 1:
            print("      Waiting 4s before next batch (rate limit safety)...")
            time.sleep(4)

    # --- Parse and save results ---
    print(f"\n[4/4] Merging results from all batches...")

    if not all_results:
        print("[ERROR] No results from any batch. Check API key and connectivity.")
        return

    print(f"      Total parsed: {len(all_results)} comment analyses.")

    # Ensure all required keys are present in every row
    for row in all_results:
        for col in OUTPUT_COLUMNS:
            row.setdefault(col, "")

    # --- Save results ---
    save_results(all_results, output_path)

    # --- Summary table ---
    print("\n" + "=" * 55)
    high   = sum(1 for r in all_results if r.get('urgency') == 'High')
    medium = sum(1 for r in all_results if r.get('urgency') == 'Medium')
    low    = sum(1 for r in all_results if r.get('urgency') == 'Low')
    print(f"  Total analyzed : {len(all_results)}")
    print(f"  High urgency   : {high}")
    print(f"  Medium urgency : {medium}")
    print(f"  Low urgency    : {low}")
    if failed_batches:
        print(f"\n  WARNING Failed batches: {failed_batches}")
        print(f"  Missing comments : ~{len(failed_batches) * BATCH_SIZE}")
    print(f"\n  Output file    : analyzed_comments.csv")
    print("=" * 55)


if __name__ == "__main__":
    main()

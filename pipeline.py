"""
pipeline.py
-----------
Unified Pipeline: YouTube URL → Google Sheet Link

This is the GLUE that chains the 3 existing scripts into ONE function call.
The web app (server.py) will call run_pipeline() — that's it.

Data flows IN-MEMORY (no intermediate CSV files):
  YouTube API → Python list → Gemini API → Python list → Google Sheets API

Usage:
    from pipeline import run_pipeline

    sheet_url = run_pipeline("https://youtube.com/watch?v=VIDEO_ID")

    # With progress tracking (for the web app):
    def my_callback(stage, message, data):
        print(f"[{stage}] {message}")

    sheet_url = run_pipeline(url, on_progress=my_callback)
"""

import os
import time
import sys
from typing import Callable, Optional


# ─────────────────────────────────────────────────────────────
# Import functions from the 3 existing scripts
# (Python runs their module-level code once — loads .env, sets globals)
# ─────────────────────────────────────────────────────────────
from extract_comments import (
    extract_video_id,
    build_youtube_client,
    fetch_all_comment_threads,
)

from main import (
    load_prompt,
    build_batch_message,
    call_gemini_batch,
    parse_json_response,
    BATCH_SIZE,
    OUTPUT_COLUMNS,
    GEMINI_API_KEY,
)

from upload_to_sheets import (
    authenticate,
    apply_header_formatting,
    apply_basic_filter,
    apply_color_coding,
    add_legend_sheet,
    CREDENTIALS_PATH,
    SHEET_TITLE,
    PERSONAL_EMAIL,
    INTENT_ORDER,
    SPREADSHEET_ID,
)


# ─────────────────────────────────────────────────────────────
# PROGRESS CALLBACK
# ─────────────────────────────────────────────────────────────
# The web app will pass its own callback to push updates
# to the browser via WebSocket. For CLI usage, we just print.
#
# stage:   "extract" | "analyze" | "upload" | "done" | "error"
# message: Human-readable status text
# data:    Optional dict with structured info (counts, URLs, etc.)
# ─────────────────────────────────────────────────────────────
ProgressFn = Callable[[str, str, Optional[dict]], None]


def _default_progress(stage: str, message: str, data: dict = None):
    """Default progress handler — prints to console."""
    print(f"  [{stage.upper():>8}] {message}")


# ─────────────────────────────────────────────────────────────
# THE ONE FUNCTION THAT DOES EVERYTHING
# ─────────────────────────────────────────────────────────────
def run_pipeline(
    youtube_url: str,
    on_progress: ProgressFn = _default_progress,
    creator_email: str = "",
) -> str:
    """
    Input:  YouTube video URL (or bare video ID)
    Output: Google Sheet URL string

    on_progress() is called at every meaningful step so the
    frontend can show real-time updates to the user.

    Raises:
        ValueError        — invalid URL, no comments, missing API key
        RuntimeError      — all AI analysis batches failed
        FileNotFoundError — prompt.txt or credentials.json missing
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # ══════════════════════════════════════════════════════════
    # STEP 1 — EXTRACT COMMENTS FROM YOUTUBE
    # ══════════════════════════════════════════════════════════
    on_progress("extract", "Parsing YouTube URL...", None)

    video_id = extract_video_id(youtube_url)
    if not video_id:
        raise ValueError(f"Could not extract a video ID from: {youtube_url}")

    on_progress("extract", f"Video ID: {video_id} — fetching comments...",
                {"video_id": video_id})

    youtube = build_youtube_client()
    comments = fetch_all_comment_threads(youtube, video_id)

    if not comments:
        raise ValueError("No comments found on this video (or comments are disabled)")

    root_count = sum(1 for c in comments if c.get("parent_id") == "root")
    reply_count = len(comments) - root_count

    on_progress(
        "extract",
        f"Extracted {len(comments)} comments ({root_count} top-level, {reply_count} replies)",
        {"total": len(comments), "root": root_count, "replies": reply_count},
    )

    # ══════════════════════════════════════════════════════════
    # STEP 2 — ANALYZE WITH GEMINI AI
    # ══════════════════════════════════════════════════════════
    on_progress("analyze", "Loading AI prompt...", None)

    prompt_path = os.path.join(base_dir, "prompt.txt")
    system_prompt = load_prompt(prompt_path)
    if not system_prompt:
        raise FileNotFoundError("prompt.txt not found or empty")

    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set in .env")

    from google import genai
    client = genai.Client(api_key=GEMINI_API_KEY)

    # Split into batches and process
    total = len(comments)
    num_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    all_results = []
    failed_batches = []

    on_progress(
        "analyze",
        f"Analyzing {total} comments in {num_batches} batches...",
        {"total": total, "num_batches": num_batches},
    )

    for batch_num in range(num_batches):
        start_idx = batch_num * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, total)
        batch = comments[start_idx:end_idx]

        on_progress(
            "analyze",
            f"Batch {batch_num + 1}/{num_batches} (comments {start_idx + 1}-{end_idx})...",
            {
                "batch": batch_num + 1,
                "num_batches": num_batches,
                "progress_pct": round((batch_num + 1) / num_batches * 100),
            },
        )

        user_message = build_batch_message(batch)

        try:
            raw_response, finish_reason = call_gemini_batch(
                client, system_prompt, user_message
            )
            batch_results = parse_json_response(raw_response)
            if batch_results:
                all_results.extend(batch_results)
            else:
                failed_batches.append(batch_num + 1)
        except Exception as e:
            failed_batches.append(batch_num + 1)
            on_progress("analyze", f"Batch {batch_num + 1} failed: {e}", None)

        # Rate-limit safety: 4s between batches (free tier = 15 RPM)
        if batch_num < num_batches - 1:
            time.sleep(4)

    if not all_results:
        raise RuntimeError("Analysis failed — no results from any batch. Check API key.")

    # Fill missing columns + sort by intent (High first)
    for row in all_results:
        for col in OUTPUT_COLUMNS:
            row.setdefault(col, "")

    all_results.sort(
        key=lambda r: INTENT_ORDER.get(r.get("intent_level", ""), 4)
    )

    high = sum(1 for r in all_results if r.get("intent_level") == "High")
    med = sum(1 for r in all_results if r.get("intent_level") == "Medium")

    on_progress(
        "analyze",
        f"Analysis complete! {len(all_results)} comments ({high} High, {med} Medium intent)",
        {"analyzed": len(all_results), "high": high, "medium": med,
         "failed_batches": failed_batches},
    )

    # ══════════════════════════════════════════════════════════
    # STEP 3 — UPLOAD TO GOOGLE SHEETS
    # ══════════════════════════════════════════════════════════
    on_progress("upload", "Authenticating with Google Sheets...", None)

    sheets_client = authenticate(CREDENTIALS_PATH)

    import gspread

    on_progress("upload", "Preparing spreadsheet...", None)

    if SPREADSHEET_ID:
        # Reuse existing sheet (clears old data, keeps the same URL)
        spreadsheet = sheets_client.open_by_key(SPREADSHEET_ID)
        try:
            worksheet = spreadsheet.worksheet("Comment Analysis")
            worksheet.clear()
        except gspread.exceptions.WorksheetNotFound:
            worksheet = spreadsheet.sheet1
            worksheet.update_title("Comment Analysis")
    else:
        # Create a brand new sheet
        spreadsheet = sheets_client.create(SHEET_TITLE)
        worksheet = spreadsheet.sheet1
        worksheet.update_title("Comment Analysis")

    # Build the 2D values array: [header_row, data_row_1, data_row_2, ...]
    headers = list(OUTPUT_COLUMNS)
    all_values = [headers]
    for row in all_results:
        all_values.append([row.get(h, "") for h in headers])

    on_progress("upload", f"Uploading {len(all_results)} rows...", None)
    worksheet.update(all_values, "A1")

    # Apply formatting
    on_progress("upload", "Applying formatting & color coding...", None)
    apply_header_formatting(spreadsheet, worksheet)
    apply_basic_filter(
        spreadsheet, worksheet,
        num_rows=len(all_results), num_cols=len(headers),
    )
    apply_color_coding(spreadsheet, worksheet, all_results, headers)

    # Legend sheet — delete old one first (for clean re-runs)
    try:
        old_legend = spreadsheet.worksheet("Legend")
        spreadsheet.del_worksheet(old_legend)
    except Exception:
        pass
    try:
        add_legend_sheet(spreadsheet)
    except Exception:
        pass

    # Share with creator's email (from the web form) + personal fallback
    share_email = creator_email if creator_email else PERSONAL_EMAIL
    try:
        spreadsheet.share(share_email, perm_type="user", role="writer")
        on_progress("upload", f"Sheet shared with {share_email}", None)
    except Exception:
        pass  # Already shared from a previous run

    # Also share with personal email if different
    if creator_email and creator_email != PERSONAL_EMAIL:
        try:
            spreadsheet.share(PERSONAL_EMAIL, perm_type="user", role="writer")
        except Exception:
            pass

    sheet_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet.id}"

    on_progress(
        "done",
        f"Sheet ready!",
        {
            "sheet_url": sheet_url,
            "total_analyzed": len(all_results),
            "high_intent": high,
            "medium_intent": med,
        },
    )

    return sheet_url


# ─────────────────────────────────────────────────────────────
# CLI FALLBACK — run the pipeline directly from the terminal
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        url = input("Enter YouTube URL or Video ID: ").strip()
    else:
        url = sys.argv[1]

    print("=" * 55)
    print("  YouTube Reply Agent — Unified Pipeline")
    print("=" * 55)

    try:
        result_url = run_pipeline(url)
        print(f"\n{'=' * 55}")
        print(f"  SUCCESS! {result_url}")
        print(f"{'=' * 55}")
    except Exception as e:
        print(f"\n[FATAL] {e}")
        sys.exit(1)

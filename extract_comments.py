"""
extract_comments.py
--------------------
This script extracts all comments (top-level + replies) from any YouTube video
and saves them in the exact CSV structure used by the YouTube Reply Agent.

HOW IT WORKS:
  1. You give it a YouTube video URL or Video ID.
  2. It calls the official YouTube Data API v3.
  3. It handles pagination automatically (API returns max 100 per page).
  4. It also fetches deep replies (>5 replies per comment).
  5. It saves everything into raw_comments.csv with the correct schema.

SETUP (one-time):
  1. Go to https://console.cloud.google.com/
  2. Create a new project (or use existing)
  3. Enable "YouTube Data API v3"
  4. Go to Credentials → Create API Key
  5. Paste that key below or set it as an environment variable:
       set YOUTUBE_API_KEY=your_key_here   (Windows)

USAGE:
  python extract_comments.py
  python extract_comments.py --video_id dQw4w9WgXcQ
  python extract_comments.py --url "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

OUTPUT:
  raw_comments.csv  ← same folder, ready to be used by main.py
"""

import os
import csv
import json
import time
import argparse
import datetime
from urllib.parse import urlparse, parse_qs

# ─────────────────────────────────────────────────────────────
# CONFIGURATION — fill in your API key here OR use env variable
# ─────────────────────────────────────────────────────────────
API_KEY = os.environ.get("YOUTUBE_API_KEY", "YOUR_API_KEY_HERE")

# The exact CSV columns matching raw_comments.csv schema
CSV_COLUMNS = [
    "comment_id",
    "parent_id",
    "commenter_name",
    "comment_text",
    "like_count",
    "timestamp",
    "time_text",
]


# ─────────────────────────────────────────────────────────────
# YOUTUBE API SETUP
# ─────────────────────────────────────────────────────────────
def build_youtube_client():
    """
    Builds the official Google API client for YouTube.
    Requires: pip install google-api-python-client
    """
    try:
        from googleapiclient.discovery import build
    except ImportError:
        print("[ERROR] Missing library. Run:")
        print("        pip install google-api-python-client")
        raise

    return build("youtube", "v3", developerKey=API_KEY)


# ─────────────────────────────────────────────────────────────
# VIDEO ID EXTRACTION
# ─────────────────────────────────────────────────────────────
def extract_video_id(url_or_id: str) -> str:
    """
    Accepts either:
      - A full YouTube URL: https://www.youtube.com/watch?v=abc123
      - A short URL:        https://youtu.be/abc123
      - Just the video ID:  abc123

    Returns the 11-character video ID.
    """
    if "youtube.com" in url_or_id or "youtu.be" in url_or_id:
        parsed = urlparse(url_or_id)
        if parsed.hostname == "youtu.be":
            return parsed.path.lstrip("/")
        query_params = parse_qs(parsed.query)
        return query_params.get("v", [url_or_id])[0]
    return url_or_id  # already a bare ID


# ─────────────────────────────────────────────────────────────
# TIMESTAMP CONVERSION
# ─────────────────────────────────────────────────────────────
def iso_to_unix(iso_string: str) -> int:
    """
    Converts YouTube's ISO 8601 timestamp: "2023-05-24T18:30:00.000Z"
    into a Unix timestamp (seconds since 1970): 1684953000

    Why Unix?  It's language-neutral, sortable, and easy to compute
    "how many days ago" without timezone headaches.
    """
    # Strip microseconds if present, handle the Z (UTC) suffix
    iso_clean = iso_string.replace("Z", "+00:00")
    try:
        dt = datetime.datetime.fromisoformat(iso_clean)
    except ValueError:
        # fallback for older Python versions
        dt = datetime.datetime.strptime(iso_string, "%Y-%m-%dT%H:%M:%S.%fZ")
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return int(dt.timestamp())


def unix_to_relative(unix_ts: int) -> str:
    """
    Converts a Unix timestamp to a human-readable relative string.
    Example: 1685059200 → "3 years ago"

    This replicates the 'time_text' column seen in raw_comments.csv.
    """
    now = int(time.time())
    diff = now - unix_ts

    if diff < 60:
        return "just now"
    elif diff < 3600:
        mins = diff // 60
        return f"{mins} minute{'s' if mins > 1 else ''} ago"
    elif diff < 86400:
        hrs = diff // 3600
        return f"{hrs} hour{'s' if hrs > 1 else ''} ago"
    elif diff < 2592000:  # 30 days
        days = diff // 86400
        return f"{days} day{'s' if days > 1 else ''} ago"
    elif diff < 31536000:  # 365 days
        months = diff // 2592000
        return f"{months} month{'s' if months > 1 else ''} ago"
    else:
        years = diff // 31536000
        return f"{years} year{'s' if years > 1 else ''} ago"


# ─────────────────────────────────────────────────────────────
# CORE EXTRACTION: TOP-LEVEL COMMENTS
# ─────────────────────────────────────────────────────────────
def fetch_all_comment_threads(youtube, video_id: str) -> list[dict]:
    """
    Fetches ALL top-level comment threads for a video.

    KEY CONCEPT — Pagination:
      YouTube's API returns max 100 comments per request.
      If a video has 586 comments, you need ~6 API requests.
      Each response includes a 'nextPageToken' — a cursor pointing
      to the next batch. We loop until there's no more token.

    API call: youtube.commentThreads().list(...)
      - part="snippet,replies" → get comment text AND inline replies
      - videoId               → which video
      - maxResults=100        → get as many as possible per request
      - order="time"          → newest first (matches raw_comments.csv)
    """
    all_rows = []
    page_token = None
    page_num = 1

    print(f"\n📡 Fetching comments for video: {video_id}")

    while True:
        print(f"   Page {page_num}... ", end="", flush=True)

        # Build the API request
        request_kwargs = {
            "part": "snippet,replies",
            "videoId": video_id,
            "maxResults": 100,
            "order": "time",        # newest first
            "textFormat": "plainText",  # no HTML tags in comment_text
        }
        if page_token:
            request_kwargs["pageToken"] = page_token

        response = youtube.commentThreads().list(**request_kwargs).execute()

        # Process each comment thread in this page
        for item in response.get("items", []):
            top_comment_snip = item["snippet"]["topLevelComment"]["snippet"]
            top_comment_id   = item["snippet"]["topLevelComment"]["id"]
            reply_count      = item["snippet"]["totalReplyCount"]

            # ── Top-level comment row ──────────────────────────
            ts = iso_to_unix(top_comment_snip["publishedAt"])
            all_rows.append({
                "comment_id":     top_comment_id,
                "parent_id":      "root",           # root = no parent
                "commenter_name": top_comment_snip["authorDisplayName"],
                "comment_text":   top_comment_snip["textDisplay"],
                "like_count":     top_comment_snip["likeCount"],
                "timestamp":      ts,
                "time_text":      unix_to_relative(ts),
            })

            # ── Inline replies (up to 5 returned automatically) ──
            inline_replies = item.get("replies", {}).get("comments", [])
            for reply in inline_replies:
                r_snip = reply["snippet"]
                r_ts   = iso_to_unix(r_snip["publishedAt"])
                all_rows.append({
                    "comment_id":     reply["id"],
                    "parent_id":      top_comment_id,  # links reply → parent
                    "commenter_name": r_snip["authorDisplayName"],
                    "comment_text":   r_snip["textDisplay"],
                    "like_count":     r_snip["likeCount"],
                    "timestamp":      r_ts,
                    "time_text":      unix_to_relative(r_ts),
                })

            # ── Deep replies (>5 replies need a separate API call) ──
            if reply_count > len(inline_replies):
                deep = fetch_deep_replies(youtube, top_comment_id, len(inline_replies))
                all_rows.extend(deep)

        fetched = len(response.get("items", []))
        print(f"got {fetched} threads")

        # Pagination: check if more pages exist
        page_token = response.get("nextPageToken")
        if not page_token:
            break  # No more pages — we're done
        page_num += 1

    return all_rows


# ─────────────────────────────────────────────────────────────
# DEEP REPLY FETCHER (for comments with >5 replies)
# ─────────────────────────────────────────────────────────────
def fetch_deep_replies(youtube, parent_comment_id: str, already_fetched: int) -> list[dict]:
    """
    When a comment has MORE than 5 replies, the commentThreads() call
    only gives you the first 5 inline. To get the rest, you must use
    the comments().list() endpoint with the parentId filter.

    This is the difference between the two API endpoints:
      commentThreads().list() → top-level comments (with up to 5 replies)
      comments().list()       → replies to a specific comment
    """
    all_replies = []
    page_token = None
    fetched = 0

    while True:
        request_kwargs = {
            "part": "snippet",
            "parentId": parent_comment_id,
            "maxResults": 100,
            "textFormat": "plainText",
        }
        if page_token:
            request_kwargs["pageToken"] = page_token

        response = youtube.comments().list(**request_kwargs).execute()

        for reply in response.get("items", []):
            fetched += 1
            if fetched <= already_fetched:
                continue  # skip duplicates already captured inline

            r_snip = reply["snippet"]
            r_ts   = iso_to_unix(r_snip["publishedAt"])
            all_replies.append({
                "comment_id":     reply["id"],
                "parent_id":      parent_comment_id,
                "commenter_name": r_snip["authorDisplayName"],
                "comment_text":   r_snip["textDisplay"],
                "like_count":     r_snip["likeCount"],
                "timestamp":      r_ts,
                "time_text":      unix_to_relative(r_ts),
            })

        page_token = response.get("nextPageToken")
        if not page_token:
            break

    return all_replies


# ─────────────────────────────────────────────────────────────
# CSV WRITER — outputs the exact schema as raw_comments.csv
# ─────────────────────────────────────────────────────────────
def save_to_csv(rows: list[dict], output_path: str):
    """
    Writes all comment rows to a CSV file.

    WHY csv.DictWriter?
      - It uses column names as headers automatically
      - extrasaction='ignore' means if the API returns bonus fields,
        they won't cause crashes — we only write what we defined
      - encoding='utf-8-sig' adds a BOM so Excel opens it correctly
        (without it, Excel shows garbled Chinese/Arabic/emoji text)
    """
    with open(output_path, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=CSV_COLUMNS,
            extrasaction="ignore",  # silently drop any extra API fields
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ Saved {len(rows)} rows to: {output_path}")


# ─────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Extract YouTube comments into raw_comments.csv"
    )
    parser.add_argument(
        "--video_id",
        default=None,
        help="YouTube Video ID (e.g. dQw4w9WgXcQ)",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="Full YouTube URL (e.g. https://www.youtube.com/watch?v=dQw4w9WgXcQ)",
    )
    parser.add_argument(
        "--output",
        default="raw_comments.csv",
        help="Output CSV filename (default: raw_comments.csv)",
    )
    args = parser.parse_args()

    # Determine the video ID
    if args.url:
        video_id = extract_video_id(args.url)
    elif args.video_id:
        video_id = args.video_id
    else:
        # Interactive fallback
        user_input = input("Enter YouTube Video ID or URL: ").strip()
        video_id = extract_video_id(user_input)

    if not video_id:
        print("[ERROR] Could not determine Video ID. Exiting.")
        return

    if API_KEY == "YOUR_API_KEY_HERE":
        print("[ERROR] Please set your YouTube API key:")
        print("  Option 1: Edit extract_comments.py and replace YOUR_API_KEY_HERE")
        print("  Option 2: set YOUTUBE_API_KEY=your_key   (then rerun)")
        return

    # Build output path relative to this script
    base_dir   = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(base_dir, args.output)

    print(f"🎬 Video ID : {video_id}")
    print(f"📄 Output   : {output_path}")

    # Run the extraction
    youtube = build_youtube_client()
    rows    = fetch_all_comment_threads(youtube, video_id)

    print(f"\n📊 Total rows collected: {len(rows)}")
    print(f"   - Root comments : {sum(1 for r in rows if r['parent_id'] == 'root')}")
    print(f"   - Replies       : {sum(1 for r in rows if r['parent_id'] != 'root')}")

    save_to_csv(rows, output_path)
    print("\n🚀 Done! Run main.py next to analyze the comments with the LLM.")


if __name__ == "__main__":
    main()

"""
upload_to_sheets.py
--------------------
Uploads analyzed_comments.csv to a NEW Google Sheet automatically.

Features:
  - Creates a brand new Google Spreadsheet
  - Uploads all data sorted by intent_level (High -> Medium -> Low -> None)
  - Adds filter views on intent_level and urgency columns
  - Applies conditional formatting (color coding):
      High   intent -> light red/orange  (#fce8e6)
      Medium intent -> light yellow      (#fef7e0)
      Low    intent -> light gray        (#f1f3f4)
      None   intent -> white             (no highlight)

Requirements:
  pip install gspread google-auth

Setup (one-time):
  1. Go to https://console.cloud.google.com/
  2. Create a project (or use existing)
  3. Enable "Google Sheets API" and "Google Drive API"
  4. Create a Service Account -> download JSON key
  5. Save the JSON key as: credentials.json (in this folder)
  6. Add GOOGLE_SERVICE_ACCOUNT_JSON=credentials.json to your .env file

Usage:
  python upload_to_sheets.py
"""

import os
import csv
import json

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
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
CSV_PATH        = os.path.join(BASE_DIR, 'analyzed_comments.csv')
CREDENTIALS_PATH = os.environ.get(
    'GOOGLE_SERVICE_ACCOUNT_JSON',
    os.path.join(BASE_DIR, 'credentials.json')
)
SHEET_TITLE      = 'YouTube Comment Analysis'

# Your personal Gmail — the sheet will be shared with you as editor
PERSONAL_EMAIL   = 'comkrishjavvaji672@gmail.com'

# If you already have a blank Google Sheet, paste its ID here (from the URL).
# Leave empty ('') to attempt creating a new sheet (may fail if Drive quota exceeded).
# Sheet URL format: https://docs.google.com/spreadsheets/d/SPREADSHEET_ID/edit
SPREADSHEET_ID   = os.environ.get('SPREADSHEET_ID', '')

# Column index mapping (0-based in Python, 1-based in Sheets API)
# commenter_name=A(0), comment_text=B(1), intent_level=C(2),
# intent_reason=D(3), recommended_reply=E(4), urgency=F(5), topic_or_course=G(6)
INTENT_COL_IDX  = 2   # Column C (0-based)
URGENCY_COL_IDX = 5   # Column F (0-based)

# Color palette (RGB 0-1 scale)
COLORS = {
    'High':   {'red': 0.988, 'green': 0.910, 'blue': 0.902},  # #fce8e6 soft red/orange
    'Medium': {'red': 0.996, 'green': 0.969, 'blue': 0.878},  # #fef7e0 soft yellow
    'Low':    {'red': 0.945, 'green': 0.953, 'blue': 0.957},  # #f1f3f4 soft gray
    'None':   {'red': 1.0,   'green': 1.0,   'blue': 1.0  },  # white
}

# Sort order for intent_level
INTENT_ORDER = {'High': 0, 'Medium': 1, 'Low': 2, 'None': 3}


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def load_and_sort_csv(csv_path: str) -> tuple[list[str], list[dict]]:
    """Loads CSV and returns (headers, rows) sorted by intent_level."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with open(csv_path, mode='r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        rows = list(reader)

    rows.sort(key=lambda r: INTENT_ORDER.get(r.get('intent_level', ''), 4))
    return list(headers), rows


def authenticate(credentials_path: str):
    """Returns an authorized gspread client using a service account JSON."""
    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except ImportError:
        print("[ERROR] Missing libraries. Run:  pip install gspread google-auth")
        raise

    scopes = [
        'https://spreadsheets.google.com/feeds',
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive',
    ]

    if not os.path.exists(credentials_path):
        raise FileNotFoundError(
            f"\n[ERROR] credentials.json not found at: {credentials_path}\n"
            "Please follow the setup steps in the docstring above."
        )

    creds = Credentials.from_service_account_file(credentials_path, scopes=scopes)
    client = gspread.authorize(creds)
    return client


def create_sheet_and_upload(client, title: str, headers: list[str], rows: list[dict]):
    """Creates a new Google Spreadsheet and uploads all data."""
    import gspread

    print(f"   Creating new spreadsheet: '{title}'...")
    spreadsheet = client.create(title)
    worksheet = spreadsheet.sheet1
    worksheet.update_title('Comment Analysis')

    # Build 2D list: header row + data rows
    all_values = [headers]
    for row in rows:
        all_values.append([row.get(h, '') for h in headers])

    print(f"   Uploading {len(rows)} rows...")
    worksheet.update('A1', all_values)

    return spreadsheet, worksheet


def apply_header_formatting(spreadsheet, worksheet):
    """Bolds the header row and freezes it."""
    sheet_id = worksheet._properties['sheetId']

    requests = [
        # Bold + background color for header row
        {
            "repeatCell": {
                "range": {
                    "sheetId": sheet_id,
                    "startRowIndex": 0,
                    "endRowIndex": 1,
                },
                "cell": {
                    "userEnteredFormat": {
                        "backgroundColor": {"red": 0.263, "green": 0.263, "blue": 0.263},
                        "textFormat": {
                            "bold": True,
                            "foregroundColor": {"red": 1.0, "green": 1.0, "blue": 1.0},
                            "fontSize": 10,
                        },
                    }
                },
                "fields": "userEnteredFormat(backgroundColor,textFormat)",
            }
        },
        # Freeze header row
        {
            "updateSheetProperties": {
                "properties": {
                    "sheetId": sheet_id,
                    "gridProperties": {"frozenRowCount": 1},
                },
                "fields": "gridProperties.frozenRowCount",
            }
        },
        # Auto-resize all columns
        {
            "autoResizeDimensions": {
                "dimensions": {
                    "sheetId": sheet_id,
                    "dimension": "COLUMNS",
                    "startIndex": 0,
                    "endIndex": 7,
                }
            }
        },
    ]

    spreadsheet.batch_update({"requests": requests})
    print("   Header formatting applied.")


def apply_basic_filter(spreadsheet, worksheet, num_rows: int, num_cols: int):
    """Adds a basic filter (dropdown arrows) to all columns."""
    sheet_id = worksheet._properties['sheetId']

    requests = [
        {
            "setBasicFilter": {
                "filter": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": 0,
                        "endRowIndex": num_rows + 1,  # header + data rows
                        "startColumnIndex": 0,
                        "endColumnIndex": num_cols,
                    }
                }
            }
        }
    ]

    spreadsheet.batch_update({"requests": requests})
    print("   Basic filters added (intent_level + urgency dropdowns ready).")


def apply_color_coding(spreadsheet, worksheet, rows: list[dict], headers: list[str]):
    """Applies row-level background colors based on intent_level value."""
    sheet_id = worksheet._properties['sheetId']
    num_cols  = len(headers)
    requests  = []

    for row_idx, row in enumerate(rows, start=1):  # row 0 is the header
        intent = row.get('intent_level', '').strip()
        color  = COLORS.get(intent, COLORS['None'])

        requests.append({
            "repeatCell": {
                "range": {
                    "sheetId": sheet_id,
                    "startRowIndex": row_idx,
                    "endRowIndex": row_idx + 1,
                    "startColumnIndex": 0,
                    "endColumnIndex": num_cols,
                },
                "cell": {
                    "userEnteredFormat": {
                        "backgroundColor": color,
                    }
                },
                "fields": "userEnteredFormat.backgroundColor",
            }
        })

    # Batch in chunks of 100 to avoid request size limits
    chunk_size = 100
    for i in range(0, len(requests), chunk_size):
        chunk = requests[i:i + chunk_size]
        spreadsheet.batch_update({"requests": chunk})
        print(f"   Color coding: applied rows {i+1}–{min(i+chunk_size, len(requests))}...")

    print("   Color coding complete.")


def add_legend_sheet(spreadsheet):
    """Adds a small Legend sheet explaining the color codes."""
    try:
        legend = spreadsheet.add_worksheet(title='Legend', rows=10, cols=3)
    except Exception:
        return  # Skip if it fails silently

    legend.update([
        ['Intent Level', 'Color',         'Meaning'],
        ['High',         'Red / Orange',  'Ready to buy or direct purchase signal'],
        ['Medium',       'Yellow',        'Informational interest / pre-purchase question'],
        ['Low',          'Gray',          'General engagement, praise, or off-topic'],
        ['None',         'White',         'Spam, filler, or creator self-replies'],
    ])

    sheet_id = legend._properties['sheetId']
    color_rows = [
        (1, COLORS['High']),
        (2, COLORS['Medium']),
        (3, COLORS['Low']),
        (4, COLORS['None']),
    ]

    requests = []
    for row_idx, color in color_rows:
        requests.append({
            "repeatCell": {
                "range": {
                    "sheetId": sheet_id,
                    "startRowIndex": row_idx,
                    "endRowIndex": row_idx + 1,
                    "startColumnIndex": 0,
                    "endColumnIndex": 3,
                },
                "cell": {"userEnteredFormat": {"backgroundColor": color}},
                "fields": "userEnteredFormat.backgroundColor",
            }
        })

    # Bold header
    requests.append({
        "repeatCell": {
            "range": {"sheetId": sheet_id, "startRowIndex": 0, "endRowIndex": 1},
            "cell": {"userEnteredFormat": {"textFormat": {"bold": True}}},
            "fields": "userEnteredFormat.textFormat.bold",
        }
    })

    spreadsheet.batch_update({"requests": requests})
    print("   Legend sheet created.")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 58)
    print("  YouTube Comments -> Google Sheets Uploader")
    print("=" * 58)

    # 1. Load & sort CSV
    print("\n[1/5] Loading and sorting analyzed_comments.csv...")
    try:
        headers, rows = load_and_sort_csv(CSV_PATH)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    print(f"      Loaded {len(rows)} rows, sorted by intent_level.")

    # 2. Authenticate
    print(f"\n[2/5] Authenticating with Google Sheets API...")
    print(f"      Using credentials: {CREDENTIALS_PATH}")
    try:
        client = authenticate(CREDENTIALS_PATH)
    except FileNotFoundError as e:
        print(e)
        return
    except Exception as e:
        print(f"[ERROR] Authentication failed: {e}")
        return
    print("      Authenticated successfully.")

    # 3. Open existing sheet OR create a new one
    print(f"\n[3/5] Setting up Google Sheet...")
    try:
        import gspread
        if SPREADSHEET_ID:
            print(f"      Opening existing sheet: {SPREADSHEET_ID}")
            spreadsheet = client.open_by_key(SPREADSHEET_ID)
            # Clear and reuse first sheet
            try:
                worksheet = spreadsheet.worksheet('Comment Analysis')
                worksheet.clear()
                print("      Cleared existing 'Comment Analysis' sheet.")
            except gspread.exceptions.WorksheetNotFound:
                worksheet = spreadsheet.sheet1
                worksheet.update_title('Comment Analysis')
        else:
            print("      No SPREADSHEET_ID set — attempting to create a new sheet...")
            print("      (If this fails with a quota error, follow the instructions below)")
            spreadsheet = client.create(SHEET_TITLE)
            worksheet = spreadsheet.sheet1
            worksheet.update_title('Comment Analysis')

        # Upload all data
        all_values = [headers]
        for row in rows:
            all_values.append([row.get(h, '') for h in headers])
        print(f"      Uploading {len(rows)} rows...")
        worksheet.update(all_values, 'A1')
        print("      Upload complete.")

    except Exception as e:
        print(f"[ERROR] Failed to set up sheet: {e}")
        if 'quota' in str(e).lower() or '403' in str(e):
            print("")
            print("  FIX: Your service account Drive quota is exceeded.")
            print("  Do these 3 steps:")
            print("  1. Go to https://sheets.new to create a blank Google Sheet")
            print("  2. Click Share -> paste this email as Editor:")
            print(f"     sheets-uploader@youtubedatascrapper-497513.iam.gserviceaccount.com")
            print("  3. Copy the Sheet ID from the URL and set it in upload_to_sheets.py:")
            print("     SPREADSHEET_ID = 'your-sheet-id-here'")
        return

    # 4. Apply formatting
    print(f"\n[4/5] Applying formatting...")
    try:
        apply_header_formatting(spreadsheet, worksheet)
        apply_basic_filter(spreadsheet, worksheet, num_rows=len(rows), num_cols=len(headers))
        apply_color_coding(spreadsheet, worksheet, rows, headers)
        add_legend_sheet(spreadsheet)
    except Exception as e:
        print(f"[ERROR] Formatting failed: {e}")
        print("        Data was uploaded but formatting may be incomplete.")

    # 5. Share with your personal Gmail as editor
    print(f"\n[5/5] Sharing spreadsheet with {PERSONAL_EMAIL}...")
    try:
        spreadsheet.share(PERSONAL_EMAIL, perm_type='user', role='writer')
        print(f"      Shared with {PERSONAL_EMAIL} as Editor.")
        print(f"      Check your Google Drive — the sheet will appear there.")
    except Exception as e:
        print(f"      [WARNING] Could not share automatically: {e}")
        print(f"      Manually share the sheet with: {PERSONAL_EMAIL}")

    url = f"https://docs.google.com/spreadsheets/d/{spreadsheet.id}"
    print(f"\n{'=' * 58}")
    print(f"  SUCCESS! Your Google Sheet is ready:")
    print(f"  {url}")
    print(f"{'=' * 58}\n")


if __name__ == "__main__":
    main()

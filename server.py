"""
server.py
---------
FastAPI backend for the YouTube Reply Agent web app.

This is the BRIDGE between the browser and the pipeline:
  Browser  ←—WebSocket—→  server.py  ——calls——→  pipeline.py

How it works:
  1. Browser opens a WebSocket connection to /ws/analyze
  2. Browser sends: {"youtube_url": "https://..."}
  3. Server runs run_pipeline() in a background thread
  4. Every time on_progress fires, server pushes the update to the browser
  5. Final message includes the Google Sheet URL
  6. Connection closes

Why WebSocket instead of HTTP?
  - The pipeline takes 2-5 minutes
  - HTTP would timeout after ~30 seconds
  - WebSocket keeps the connection alive and lets us stream progress

Usage:
  uvicorn server:app --reload --port 8000
  Then open: http://localhost:8000
"""

import os
import json
import asyncio
import traceback
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from pipeline import run_pipeline


# ─────────────────────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────────────────────
app = FastAPI(title="YouTube Reply Agent")

# Thread pool for running the blocking pipeline in background
# (FastAPI is async, but our pipeline uses blocking I/O)
executor = ThreadPoolExecutor(max_workers=2)

# Serve static frontend files (index.html, style.css, app.js)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Create static directory if it doesn't exist yet
os.makedirs(STATIC_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────

@app.get("/")
async def serve_frontend():
    """Serves the main HTML page when user opens the app."""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# Mount static files AFTER the root route so / serves index.html
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ─────────────────────────────────────────────────────────────
# WEBSOCKET ENDPOINT — The core of the real-time experience
# ─────────────────────────────────────────────────────────────

@app.websocket("/ws/analyze")
async def websocket_analyze(ws: WebSocket):
    """
    WebSocket endpoint that runs the full pipeline and streams
    progress updates back to the browser in real-time.

    Protocol:
      Client sends:  {"youtube_url": "https://..."}
      Server sends:  {"stage": "extract", "message": "...", "data": {...}}
                     {"stage": "analyze", "message": "...", "data": {...}}
                     {"stage": "upload",  "message": "...", "data": {...}}
                     {"stage": "done",    "message": "...", "data": {"sheet_url": "..."}}
                 or: {"stage": "error",   "message": "Something went wrong"}
    """
    await ws.accept()

    try:
        # 1. Wait for the client to send the YouTube URL
        raw = await ws.receive_text()
        payload = json.loads(raw)
        youtube_url = payload.get("youtube_url", "").strip()
        creator_email = payload.get("creator_email", "").strip()

        if not youtube_url:
            await ws.send_json({
                "stage": "error",
                "message": "No YouTube URL provided",
                "data": None,
            })
            await ws.close()
            return

        # 2. Create a progress callback that sends updates via WebSocket
        #
        #    KEY CONCEPT: run_pipeline() is synchronous (blocking).
        #    WebSocket.send_json() is async (non-blocking).
        #    We use asyncio.run_coroutine_threadsafe() to bridge the two:
        #    the pipeline runs in a thread, and each progress update is
        #    safely pushed onto the async event loop.
        loop = asyncio.get_event_loop()

        def send_progress(stage: str, message: str, data: dict = None):
            """Called by pipeline — pushes update to browser via WebSocket."""
            future = asyncio.run_coroutine_threadsafe(
                ws.send_json({
                    "stage": stage,
                    "message": message,
                    "data": data,
                }),
                loop,
            )
            # Wait for the send to complete (so we don't flood the socket)
            future.result(timeout=10)

        # 3. Run the pipeline in a background thread
        #    (because it's blocking — API calls, time.sleep, etc.)
        sheet_url = await asyncio.get_event_loop().run_in_executor(
            executor,
            lambda: run_pipeline(
                youtube_url,
                on_progress=send_progress,
                creator_email=creator_email,
            ),
        )

        # 4. Pipeline finished successfully — the "done" message was
        #    already sent by run_pipeline via on_progress.
        #    Close the connection cleanly.

    except WebSocketDisconnect:
        print("[WS] Client disconnected")
    except Exception as e:
        # Send error to browser before closing
        try:
            await ws.send_json({
                "stage": "error",
                "message": str(e),
                "data": None,
            })
        except Exception:
            pass
        print(f"[WS ERROR] {traceback.format_exc()}")
    finally:
        try:
            await ws.close()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────
# HEALTH CHECK (useful for deployment monitoring)
# ─────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}

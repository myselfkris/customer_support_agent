"""Shared test setup: fully offline (no Gemini, no network, no real database).

Environment variables are set at import time, BEFORE any app module is imported,
because ``db/base.py`` and ``app/config.py`` read them when first imported.
``load_dotenv()`` does not override variables that are already set, so the
developer's ``.env`` cannot switch these off.
"""

import os
import tempfile
from pathlib import Path

_TMP_DIR = Path(tempfile.mkdtemp(prefix="csa_tests_"))

os.environ["MOCK_MODE"] = "true"
os.environ["GEMINI_API_KEY"] = ""
os.environ["RELATIONAL_DATABASE_URL"] = f"sqlite:///{(_TMP_DIR / 'test.db').as_posix()}"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import app.main as main_module  # noqa: E402
from agents.reliable_agent import ReliableAgent  # noqa: E402
from app.auth import require_customer  # noqa: E402


class RecordingApp:
    """Stand-in for the LangGraph app: counts calls, never raises, never calls out."""

    def __init__(self):
        self.calls = 0

    def invoke(self, state: dict) -> dict:
        self.calls += 1
        return {"final_response": "ok"}


@pytest.fixture
def stub_app():
    return RecordingApp()


@pytest.fixture
def client(monkeypatch, tmp_path, stub_app):
    """TestClient with auth and the agent replaced.

    Not used as a context manager, so the app's ``lifespan`` (init_db + real
    agent) never runs. A fresh ReliableAgent per test keeps rate-limit and
    memory state from leaking between tests.
    """
    agent = ReliableAgent(app=stub_app, log_path=tmp_path / "events.jsonl")
    monkeypatch.setattr(main_module, "_agent", agent)
    main_module.app.dependency_overrides[require_customer] = lambda: 1
    try:
        yield TestClient(main_module.app)
    finally:
        main_module.app.dependency_overrides.clear()

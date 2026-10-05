"""Message rules for POST /chat.

Rules:
  - Up to 5000 characters is allowed; 5001 is rejected.
  - More than 200 consecutive spaces (' ' only) is rejected.
  - An empty message is rejected; a message of only spaces is rejected.

"Rejected" means HTTP 422 and the agent is never called.
"""

import pytest


def post_chat(client, message):
    return client.post("/chat", json={"message": message})


# ── Length ────────────────────────────────────────────────────────────────────

def test_exactly_5000_chars_allowed(client, stub_app):
    response = post_chat(client, "a" * 5000)
    assert response.status_code == 200
    assert stub_app.calls == 1


def test_5001_chars_rejected(client, stub_app):
    response = post_chat(client, "a" * 5001)
    assert response.status_code == 422
    assert stub_app.calls == 0


# ── Consecutive spaces ────────────────────────────────────────────────────────

def test_200_consecutive_spaces_allowed(client, stub_app):
    response = post_chat(client, "hi" + " " * 200 + "there")
    assert response.status_code == 200
    assert stub_app.calls == 1


def test_201_consecutive_spaces_rejected(client, stub_app):
    response = post_chat(client, "hi" + " " * 201 + "there")
    assert response.status_code == 422
    assert stub_app.calls == 0


def test_many_spaces_split_by_text_allowed(client, stub_app):
    response = post_chat(client, "a" + " " * 150 + "b" + " " * 150 + "c")
    assert response.status_code == 200
    assert stub_app.calls == 1


def test_tabs_do_not_count_toward_space_run(client, stub_app):
    response = post_chat(client, "hi" + "\t" * 300 + "there")
    assert response.status_code == 200
    assert stub_app.calls == 1


# ── Empty / blank ─────────────────────────────────────────────────────────────

def test_empty_message_rejected(client, stub_app):
    response = post_chat(client, "")
    assert response.status_code == 422
    assert stub_app.calls == 0


@pytest.mark.parametrize("count", [1, 5, 200])
def test_spaces_only_rejected(client, stub_app, count):
    response = post_chat(client, " " * count)
    assert response.status_code == 422
    assert stub_app.calls == 0


def test_missing_message_field_rejected(client, stub_app):
    response = client.post("/chat", json={})
    assert response.status_code == 422
    assert stub_app.calls == 0

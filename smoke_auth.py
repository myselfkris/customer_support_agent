"""Chunk 5 smoke test — verify auth works end to end.

Covers:
  1. Password hashing (roundtrip + wrong password rejected)
  2. JWT (roundtrip + tampered token rejected)
  3. API keys (prefix + determinism + unknown key rejected)
  4. HTTP doors: /admin/login, /admin/approvals, /chat
"""

from fastapi.testclient import TestClient

from app.auth import (
    create_access_token,
    create_admin,
    decode_token,
    generate_api_key,
    hash_api_key,
    hash_password,
    issue_api_key,
    verify_api_key,
    verify_password,
)
from app.main import app
from db.base import init_db

FAILED = []


def check(name: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


print("=" * 60)
print("CHUNK 5 SMOKE TEST — auth")
print("=" * 60)

init_db()  # ensure auth tables exist before touching the DB (mirrors app lifespan)

# ── 1. Password hashing ────────────────────────────────────────
h = hash_password("supersecret")
check("hash_password returns bcrypt-style hash", h.startswith("$2"), h[:25] + "...")
check("verify_password accepts correct password", verify_password("supersecret", h))
check("verify_password rejects wrong password", not verify_password("wrong", h))

# ── 2. JWT ─────────────────────────────────────────────────────
token = create_access_token(1)
payload = decode_token(token)
check("decode_token roundtrips valid token", payload is not None and payload["sub"] == "1")
check("decode_token rejects tampered token", decode_token(token + "x") is None)
check("decode_token carries role=admin", payload is not None and payload.get("role") == "admin")

# ── 3. API keys ────────────────────────────────────────────────
raw, key_hash = generate_api_key()
check("generate_api_key has prefix", raw.startswith("cust_live_"), raw[:16] + "...")
check("hash_api_key is deterministic", hash_api_key(raw) == key_hash)
check("verify_api_key returns None for unknown key", verify_api_key("cust_live_bogus") is None)

# ── 4. HTTP doors (needs seeded admin + customer key) ──────────
admin_id = create_admin("admin", "adminpass123")
print(f"  (seeded admin id={admin_id})")
raw_key, key_id = issue_api_key(1)  # customer 1 = Alice from db/seed.py
print(f"  (issued API key for customer 1: {raw_key[:16]}...)")

with TestClient(app) as client:
    r = client.post("/admin/login", json={"username": "admin", "password": "wrong"})
    check("/admin/login rejects wrong password (401)", r.status_code == 401, f"got {r.status_code}")

    r = client.post("/admin/login", json={"username": "admin", "password": "adminpass123"})
    check("/admin/login returns token (200)", r.status_code == 200, f"got {r.status_code}")
    access = r.json()["access_token"] if r.status_code == 200 else None

    r = client.get("/admin/approvals")
    check("/admin/approvals without token (401)", r.status_code == 401, f"got {r.status_code}")

    if access:
        r = client.get("/admin/approvals", headers={"Authorization": f"Bearer {access}"})
        check("/admin/approvals with valid token (200)", r.status_code == 200, f"got {r.status_code}")

    r = client.post("/chat", json={"message": "hello"})
    check("/chat without API key (401)", r.status_code == 401, f"got {r.status_code}")

    r = client.post("/chat", json={"message": "hello"}, headers={"X-API-Key": raw_key})
    check("/chat with valid API key (200)", r.status_code == 200, f"got {r.status_code}")

print("=" * 60)
if FAILED:
    print(f"FAILED: {len(FAILED)} — {FAILED}")
else:
    print("ALL AUTH SMOKE TESTS PASSED")
print("=" * 60)

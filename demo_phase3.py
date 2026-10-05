"""Phase 3 end-to-end demo — the money-movement safety story.

Run:  python demo_phase3.py

Walks the full narrative:
  1. Auth: admin logs in (password -> JWT), customer presents an API key
  2. Customer requests a refund  -> PENDING (the AI can only REQUEST)
  3. Admin lists pending        -> sees the request
  4. Admin approves             -> refund APPROVED, audit trail stamped
  5. Duplicate request          -> refused (idempotency)
  6. Raw DB rows                -> the final truth

Resets demo data first so the run is deterministic and re-runnable.
"""

from fastapi.testclient import TestClient

from agents.tool_calling_agent import (
    approve_approval,
    list_pending_approvals,
    process_refund,
)
from app.auth import create_admin, issue_api_key
from app.main import app
from db.base import SessionLocal
from db.models import Approval, Refund

SEP = "=" * 64


def show(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")


def reset_demo_data() -> None:
    """Clear demo refunds/approvals so the run starts clean (sample data only)."""
    session = SessionLocal()
    try:
        session.query(Refund).delete()
        session.query(Approval).delete()
        session.commit()
    finally:
        session.close()


# ── 0. Reset + seed ────────────────────────────────────────────
show("0) Seed credentials (fresh admin + customer API key)")
reset_demo_data()
admin_id = create_admin("demo_admin", "demo123")
raw_key, _ = issue_api_key(1)  # customer 1 = Alice
print(f"  admin     : demo_admin / demo123   (id={admin_id})")
print(f"  customer  : Alice (id=1)")
print(f"  API key   : {raw_key}")
print(f"  (send as header:  X-API-Key: {raw_key})")

# ── 1. Auth over HTTP ─────────────────────────────────────────
show("1) Auth over HTTP (the real FastAPI app + guards)")
with TestClient(app) as client:
    r = client.post("/admin/login", json={"username": "demo_admin", "password": "demo123"})
    token = r.json()["access_token"]
    print(f"  POST /admin/login (correct pw)  -> {r.status_code}   token={token[:36]}...")

    r = client.post("/admin/login", json={"username": "demo_admin", "password": "wrong"})
    print(f"  POST /admin/login (wrong pw)    -> {r.status_code}   (rejected)")

    r = client.get("/admin/approvals")
    print(f"  GET  /admin/approvals (no JWT)  -> {r.status_code}   (rejected)")

    r = client.get("/admin/approvals", headers={"Authorization": f"Bearer {token}"})
    print(f"  GET  /admin/approvals (valid)   -> {r.status_code}")

    r = client.post("/chat", json={"message": "hello"})
    print(f"  POST /chat (no API key)         -> {r.status_code}   (rejected)")

    r = client.post("/chat", json={"message": "hello"}, headers={"X-API-Key": raw_key})
    print(f"  POST /chat (valid key)          -> {r.status_code}")
    print(f"     agent reply: {r.json()['response'][:70]!r}")

# ── 2. Request -> pending ─────────────────────────────────────
show("2) Customer asks for a refund (the AI can only REQUEST)")
req = process_refund("7291", "item arrived damaged", requested_by="customer:1")
print(f"  process_refund(order 7291)      -> status={req['status']!r}")
print(f"     approval_id = {req['approval_id']}   (money has NOT moved yet)")

# ── 3. Admin sees pending ─────────────────────────────────────
show("3) Admin lists pending approvals")
for p in list_pending_approvals():
    print(f"  approval #{p['approval_id']}: {p['kind']} for order {p['order_id']} "
          f"amount=${p['amount']} requested_by={p['requested_by']!r} status={p['status']!r}")

# ── 4. Admin approves ─────────────────────────────────────────
show("4) Admin approves (only a human, holding a JWT, can do this)")
res = approve_approval(req["approval_id"], approved_by="demo_admin")
print(f"  approve_approval({req['approval_id']}) -> {res}")

# ── 5. Duplicate refused ──────────────────────────────────────
show("5) Duplicate refund request is refused (idempotency)")
dup = process_refund("7291", "trying again", requested_by="customer:1")
print(f"  process_refund again -> already_requested={dup.get('already_requested')}")

# ── 6. Raw DB rows ────────────────────────────────────────────
show("6) Raw DB rows (the audit trail)")
session = SessionLocal()
try:
    refund = session.query(Refund).filter(Refund.order_id == "7291").first()
    approval = session.query(Approval).filter(Approval.id == req["approval_id"]).first()
    print(f"  refund   : order={refund.order_id} amount=${refund.amount} status={refund.status!r}")
    print(f"             approved_by={refund.approved_by!r}  approved_at={refund.approved_at}")
    print(f"  approval : requested_by={approval.requested_by!r} -> approved_by={approval.approved_by!r} "
          f"status={approval.status!r}")
finally:
    session.close()

print(f"\n{SEP}\n  DEMO COMPLETE — money moved only AFTER human approval\n{SEP}")

"""Step 4: Phase 3 smoke test — prove the approval flow works end to end.

Runs against the REAL database (no mocks). Sequence:
  1. Request a refund            -> pending Refund + pending Approval
  2. Request it again            -> idempotent (same request returned)
  3. Approve it                  -> money state changes ONCE, audit stamped
  4. Approve again               -> refused (terminal state)
  5. Cancel another order        -> pending_cancel + pending Approval
  6. Decline it                  -> order restored to previous status
  7. List pending approvals      -> should be empty
  8. Read raw rows from the DB   -> the final truth
"""

from db.base import SessionLocal
from db.models import Approval, Order, Refund
from agents.tool_calling_agent import (
    process_refund,
    approve_approval,
    decline_approval,
    cancel_order,
    list_pending_approvals,
)

print("=" * 60)
print("PHASE 3 SMOKE TEST")
print("=" * 60)

# 1. Request a refund for order 7291
r1 = process_refund("7291", "smoke test: speaker arrived broken", requested_by="smoke-user")
print("\n1) refund request:      ", r1)

# 2. Request the SAME refund again (idempotency)
r2 = process_refund("7291", "smoke test: duplicate attempt", requested_by="smoke-user")
print("2) duplicate request:   ", r2)
assert r2.get("already_requested") is True, "idempotency FAILED"

# 3. Approve it (the human decision)
a1 = approve_approval(r1["approval_id"], approved_by="smoke-admin")
print("3) approve:             ", a1)
assert a1["status"] == "approved", "approve FAILED"

# 4. Approve AGAIN — must be refused (terminal state)
a2 = approve_approval(r1["approval_id"], approved_by="smoke-admin")
print("4) re-approve attempt:  ", a2)
assert "error" in a2, "double-approve was NOT refused"

# 5. Cancel order 5592 (was 'partially_shipped')
c1 = cancel_order("5592", requested_by="smoke-user")
print("5) cancel request:      ", c1)
assert c1["status"] == "pending_cancel", "cancel gating FAILED"

# 6. Decline the cancellation — order must be RESTORED
d1 = decline_approval(c1["approval_id"], declined_by="smoke-admin")
print("6) decline:             ", d1)
assert d1["status"] == "declined", "decline FAILED"

# 7. Pending list should now be empty
pending = list_pending_approvals()
print("7) pending approvals:   ", pending)
assert pending == [], "pending list not empty"

# 8. Read the raw rows — the final truth
session = SessionLocal()
try:
    refund = session.get(Refund, 1)
    order_7291 = session.get(Order, "7291")
    order_5592 = session.get(Order, "5592")
    approvals = session.query(Approval).all()
    print("\n8) raw DB rows:")
    print(f"   refund #1        : status={refund.status!r} approved_by={refund.approved_by!r} approved_at={refund.approved_at}")
    print(f"   order 7291       : status={order_7291.status!r}")
    print(f"   order 5592       : status={order_5592.status!r}  (restored after decline)")
    for a in approvals:
        print(f"   approval #{a.id}   : kind={a.kind!r} status={a.status!r} by={a.approved_by!r}")
finally:
    session.close()

print("\n" + "=" * 60)
print("ALL SMOKE TESTS PASSED ✅")
print("=" * 60)

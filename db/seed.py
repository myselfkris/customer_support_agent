"""Seed the database with sample customers and orders.

Mirrors the sample data that used to live as hardcoded dicts in
tool_calling_agent.py, so behavior stays consistent.

Run:
    python -m db.seed
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from db.base import SessionLocal, init_db
from db.models import Customer, Order


def seed_data(session: Session) -> None:
    """Insert sample data once (idempotent: skips if already present)."""
    if session.query(Customer).count() > 0:
        print("[seed] Data already present — skipping.")
        return

    alice = Customer(name="Alice Johnson", email="alice@example.com")
    bob = Customer(name="Bob Smith", email="bob@example.com")
    session.add_all([alice, bob])
    session.flush()  # assign auto IDs so we can link orders to customers

    orders = [
        Order(id="7291", customer_id=alice.id, status="shipped", tracking_number="TRK-998877", items="Wireless Mouse, USB-C Cable", amount=49.99),
        Order(id="3310", customer_id=alice.id, status="delivered", items="Bluetooth Speaker", amount=59.99),
        Order(id="5592", customer_id=bob.id, status="partially_shipped", items="Laptop Stand, Monitor Arm", amount=79.99),
        Order(id="1234", customer_id=bob.id, status="processing", items="Mechanical Keyboard", amount=89.99),
    ]
    session.add_all(orders)
    session.commit()

    print(f"[seed] Seeded {session.query(Customer).count()} customers, {session.query(Order).count()} orders.")


if __name__ == "__main__":
    init_db()
    session = SessionLocal()
    try:
        seed_data(session)
    finally:
        session.close()

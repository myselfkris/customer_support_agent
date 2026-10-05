"""SQLAlchemy data models for the customer support domain.

These replace the hardcoded dicts in tool_calling_agent.py with real,
persistent, relational data.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.base import Base


def _now() -> datetime:
    """Current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)


class Customer(Base):
    """A customer of the store."""

    __tablename__ = "customers"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    email: Mapped[str] = mapped_column(String(255), unique=True)

    orders: Mapped[list["Order"]] = relationship(back_populates="customer")


class Order(Base):
    """A customer's order.

    Status lifecycle (Phase 3):
      - Normal: ``processing`` -> ``shipped`` | ``delivered`` | ``partially_shipped``
      - Cancel: ``<current>`` -> ``pending_cancel`` -> ``cancelled``

    A cancellation only reaches ``cancelled`` after a human approves the pending
    ``Approval`` request — the tool itself only sets ``pending_cancel``.
    """

    __tablename__ = "orders"

    id: Mapped[str] = mapped_column(String(20), primary_key=True)  # e.g. "7291"
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"))
    status: Mapped[str] = mapped_column(String(30), default="processing")
    tracking_number: Mapped[str | None] = mapped_column(String(50), nullable=True)
    items: Mapped[str] = mapped_column(Text, default="")  # comma-separated for simplicity
    amount: Mapped[float] = mapped_column(Float, default=0.0)

    customer: Mapped["Customer"] = relationship(back_populates="orders")
    refunds: Mapped[list["Refund"]] = relationship(back_populates="order")
    approvals: Mapped[list["Approval"]] = relationship(back_populates="order")


class Refund(Base):
    """A refund request against an order.

    Lifecycle: ``pending`` -> ``approved`` | ``declined``.

    ``approved_by`` and ``approved_at`` record WHO released the money and WHEN
    (the Phase 3 audit trail).
    """

    __tablename__ = "refunds"

    id: Mapped[int] = mapped_column(primary_key=True)
    order_id: Mapped[str] = mapped_column(ForeignKey("orders.id"))
    amount: Mapped[float] = mapped_column(Float)
    reason: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(30), default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    approved_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    order: Mapped["Order"] = relationship(back_populates="refunds")


class Approval(Base):
    """A human-in-the-loop decision awaiting an authenticated admin.

    Phase 3's core safety primitive. Every refund request and cancellation
    creates one ``Approval`` row (status ``pending``). Money/order state only
    changes when an authenticated admin flips it to ``approved`` — otherwise it
    is ``declined``. Once resolved, rows are immutable (no double applies).

    ``kind``: ``"refund"`` or ``"cancel"``.
    ``requested_by``: the authenticated customer (from the API key) who asked.
    ``approved_by``:  the authenticated admin (from the JWT) who decided.
    ``previous_status``: for cancels, the order status before the request, so a
    declined cancellation can restore it.
    """

    __tablename__ = "approvals"

    id: Mapped[int] = mapped_column(primary_key=True)
    kind: Mapped[str] = mapped_column(String(20))  # "refund" | "cancel"
    order_id: Mapped[str] = mapped_column(ForeignKey("orders.id"))
    refund_id: Mapped[int | None] = mapped_column(ForeignKey("refunds.id"), nullable=True)
    previous_status: Mapped[str | None] = mapped_column(String(30), nullable=True)
    amount: Mapped[float] = mapped_column(Float, default=0.0)
    reason: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(30), default="pending")  # pending|approved|declined
    requested_by: Mapped[str] = mapped_column(String(128))
    approved_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    order: Mapped["Order"] = relationship(back_populates="approvals")


class Ticket(Base):
    """A support ticket."""

    __tablename__ = "tickets"

    id: Mapped[int] = mapped_column(primary_key=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"))
    subject: Mapped[str] = mapped_column(String(200))
    status: Mapped[str] = mapped_column(String(30), default="open")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Conversation(Base):
    """A persisted conversation message (used for durable memory in Phase 4)."""

    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[str] = mapped_column(String(128))
    role: Mapped[str] = mapped_column(String(20))  # "customer" | "agent"
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class AdminUser(Base):
    """A staff member who can approve/decline money decisions.

    Stores a *hash* of the password — never the password itself.
    """

    __tablename__ = "admin_users"

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(100), unique=True)
    password_hash: Mapped[str] = mapped_column(String(255))  # bcrypt hash, never plaintext
    role: Mapped[str] = mapped_column(String(30), default="admin")


class CustomerApiKey(Base):
    """A hashed API key that proves a customer's identity.

    Stores ``key_hash`` (never the raw key) so a DB leak does not expose usable
    keys. ``key_prefix`` is a short readable fragment the customer can recognize
    their own key by. ``revoked`` kills a leaked key without deleting the row.
    """

    __tablename__ = "customer_api_keys"

    id: Mapped[int] = mapped_column(primary_key=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"))
    key_hash: Mapped[str] = mapped_column(String(255), unique=True)
    key_prefix: Mapped[str] = mapped_column(String(16))
    revoked: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

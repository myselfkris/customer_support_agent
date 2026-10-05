"""Authentication: API keys for customers, JWT for admins.

Chunk 5. Two doors, two credentials:

  - Customers prove identity with a hashed API key  -> require_customer -> customer_id
  - Admins prove identity with a password -> JWT    -> require_admin   -> admin claims

Never stores a raw password or a raw API key — only hashes.
"""

from __future__ import annotations

import hashlib
import secrets
import time

import bcrypt
import jwt
from fastapi import Header, HTTPException, status
from sqlalchemy.orm import Session

from app.config import settings
from db.base import SessionLocal, init_db
from db.models import AdminUser, CustomerApiKey


# ── Password hashing (admins) ─────────────────────────────────
def hash_password(plain: str) -> str:
    """Turn a plain password into a salted one-way hash (store THIS)."""
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """Does this password open this lock? Constant-time check."""
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except ValueError:
        return False


# ── JWT (admins) ──────────────────────────────────────────────
def create_access_token(admin_id: int) -> str:
    """Issue a short-lived token after a successful login."""
    now = int(time.time())
    payload = {
        "sub": str(admin_id),
        "role": "admin",
        "iat": now,
        "exp": now + settings.access_token_expire_minutes * 60,
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_token(token: str) -> dict | None:
    """Verify signature + expiry. Returns claims, or None if forged/expired."""
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except jwt.PyJWTError:
        return None


# ── API keys (customers) ──────────────────────────────────────
def generate_api_key() -> tuple[str, str]:
    """Return (raw_key_to_give_customer, hash_to_store)."""
    raw = settings.api_key_prefix + secrets.token_urlsafe(32)
    return raw, hash_api_key(raw)


def hash_api_key(raw: str) -> str:
    """One-way fingerprint of a high-entropy key. A fast hash is fine here."""
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def verify_api_key(raw: str) -> int | None:
    """Hash what the customer sent, look it up. Returns customer_id or None."""
    key_hash = hash_api_key(raw)
    session = SessionLocal()
    try:
        row = (
            session.query(CustomerApiKey)
            .filter(CustomerApiKey.key_hash == key_hash)
            .first()
        )
        if row is None or row.revoked:
            return None
        return row.customer_id
    finally:
        session.close()


# ── Account management (seed/demo; not exposed over HTTP) ─────
def create_admin(username: str, password: str, role: str = "admin") -> int:
    """Create (or reset) an admin user. Returns its id."""
    init_db()
    session = SessionLocal()
    try:
        user = session.query(AdminUser).filter(AdminUser.username == username).first()
        if user is None:
            user = AdminUser(username=username, password_hash=hash_password(password), role=role)
            session.add(user)
        else:
            user.password_hash = hash_password(password)
            user.role = role
        session.commit()
        session.refresh(user)
        return user.id
    finally:
        session.close()


def issue_api_key(customer_id: int) -> tuple[str, int]:
    """Create a new API key for a customer. Returns (raw_key, key_row_id)."""
    init_db()
    raw, key_hash = generate_api_key()
    session = SessionLocal()
    try:
        row = CustomerApiKey(
            customer_id=customer_id,
            key_hash=key_hash,
            key_prefix=raw[: len(settings.api_key_prefix) + 8],
            revoked=False,
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        return raw, row.id
    finally:
        session.close()


def authenticate_admin(username: str, password: str) -> dict | None:
    """Check credentials. Returns {id, username, role} or None."""
    session = SessionLocal()
    try:
        user = session.query(AdminUser).filter(AdminUser.username == username).first()
        if user is None or not verify_password(password, user.password_hash):
            return None
        return {"id": user.id, "username": user.username, "role": user.role}
    finally:
        session.close()


# ── FastAPI guards (the two doors) ────────────────────────────
def require_customer(x_api_key: str | None = Header(default=None)) -> int:
    """Door #1 (lobby): read the API key, return the customer_id."""
    if not x_api_key:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Missing API key")
    customer_id = verify_api_key(x_api_key)
    if customer_id is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Invalid or revoked API key")
    return customer_id


def require_admin(authorization: str | None = Header(default=None)) -> dict:
    """Door #2 (vault): read the Bearer token, return the claims."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")
    payload = decode_token(authorization.removeprefix("Bearer "))
    if payload is None or payload.get("role") != "admin":
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")
    return payload

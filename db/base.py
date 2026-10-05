"""Database engine, session factory, and declarative base.

Uses SQLAlchemy so the SAME models work with SQLite (development) and
PostgreSQL (production). Switching databases is a one-line config change:

    Development:  sqlite:///./customer_support.db      (default, zero setup)
    Production:   postgresql://user:pass@host:port/customer_support
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

load_dotenv()

DEFAULT_DB_URL = "sqlite:///./customer_support.db"
DATABASE_URL = os.getenv("RELATIONAL_DATABASE_URL", DEFAULT_DB_URL)

IS_SQLITE = DATABASE_URL.startswith("sqlite")

# SQLite disallows cross-thread access by default; the FastAPI app runs the
# agent in worker threads, so we must allow it.
connect_args = {"check_same_thread": False} if IS_SQLITE else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args, echo=False)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    """Declarative base — every model inherits from this."""


def init_db() -> None:
    """Create all tables if they don't exist.

    Development convenience. In production this is handled by Alembic
    migrations (versioned, reversible schema changes).
    """
    from db import models  # noqa: F401  (import registers models with metadata)

    Base.metadata.create_all(bind=engine)

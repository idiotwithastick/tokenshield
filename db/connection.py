"""
TokenShield Database Connection
=================================
PostgreSQL connection via SQLAlchemy for Replit deployment.
Falls back to SQLite for local development.

Author: Wesley Foreman (wforeman58@gmail.com)
Copyright 2026. All rights reserved.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session, declarative_base

Base = declarative_base()

# Replit provides DATABASE_URL env var for PostgreSQL
# Fall back to SQLite for local dev
_DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "sqlite:///./tokenshield_dev.db"
)

# SQLAlchemy setup — pool args only for PostgreSQL, not SQLite
_engine_kwargs = {"pool_pre_ping": True}
if _DATABASE_URL.startswith("postgresql"):
    _engine_kwargs["pool_size"] = 5
    _engine_kwargs["max_overflow"] = 10
if _DATABASE_URL.startswith("sqlite"):
    _engine_kwargs["connect_args"] = {"check_same_thread": False}

_engine = create_engine(_DATABASE_URL, **_engine_kwargs)

_SessionLocal = sessionmaker(bind=_engine, autocommit=False, autoflush=False)


def get_db() -> Session:
    """Get a database session. Use as dependency injection in FastAPI."""
    db = _SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables. Called at startup."""
    from . import models  # noqa: F401 — registers models with Base
    Base.metadata.create_all(bind=_engine)


def get_engine():
    """Get the SQLAlchemy engine (for testing/admin)."""
    return _engine

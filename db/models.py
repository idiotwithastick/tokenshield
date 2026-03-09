"""
TokenShield Database Models
=============================
SQLAlchemy models for PostgreSQL (Replit) / SQLite (local dev).

Tables:
  - cached_responses: Proxy response cache with signatures
  - api_keys: User API keys with tier and rate limits
  - usage_log: Per-request usage tracking

Author: Wesley Foreman (wforeman58@gmail.com)
Copyright 2026. All rights reserved.
"""

import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Float, Text, Boolean,
    DateTime, ForeignKey, Index
)
from sqlalchemy.orm import relationship
from .connection import Base


class CachedResponse(Base):
    """Cached proxy response with thermosolve signature."""
    __tablename__ = "cached_responses"

    id = Column(String(64), primary_key=True)
    user_id = Column(String(64), nullable=False, index=True)
    provider = Column(String(32), nullable=False)  # anthropic, openai, google

    # Prompt signature for similarity matching
    prompt_hash = Column(String(128), index=True)
    sig_n = Column(Integer, default=0)
    sig_s = Column(Float, default=0.0)
    sig_ds = Column(Float, default=0.0)
    sig_phi = Column(Float, default=0.0)

    # Cached response (full JSON body)
    response_body = Column(Text, nullable=False)

    # Token counts
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)

    # Usage tracking
    hit_count = Column(Integer, default=0)
    last_hit_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("idx_cache_provider_hash", "provider", "prompt_hash"),
        Index("idx_cache_user", "user_id"),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "provider": self.provider,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "hit_count": self.hit_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class APIKey(Base):
    """User API key with tier and usage tracking."""
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String(128), unique=True, nullable=False, index=True)
    user_email = Column(String(256), nullable=False, index=True)
    tier = Column(String(32), default="free")  # free, pro, team
    enabled = Column(Boolean, default=True)

    # Rate limits (per day)
    request_limit = Column(Integer, default=100)  # Max proxy requests per day

    # Usage counters
    requests_today = Column(Integer, default=0)
    requests_reset_at = Column(DateTime, default=datetime.utcnow)
    total_tokens_saved = Column(Integer, default=0)
    total_requests = Column(Integer, default=0)
    total_cache_hits = Column(Integer, default=0)

    # Stripe
    stripe_customer_id = Column(String(128), nullable=True)
    stripe_subscription_id = Column(String(128), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    usage_logs = relationship("UsageLog", back_populates="api_key")

    def is_rate_limited(self) -> bool:
        now = datetime.utcnow()
        if self.requests_reset_at and (now - self.requests_reset_at).days >= 1:
            self.requests_today = 0
            self.requests_reset_at = now
            return False
        return self.requests_today >= self.request_limit

    @staticmethod
    def generate_key() -> str:
        return f"sk-ts-{uuid.uuid4().hex}"


class UsageLog(Base):
    """Per-request usage log for analytics."""
    __tablename__ = "usage_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    api_key_id = Column(Integer, ForeignKey("api_keys.id"), nullable=False, index=True)
    endpoint = Column(String(64), nullable=False)
    provider = Column(String(32), default="")
    cached = Column(Boolean, default=False)
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    tokens_saved = Column(Integer, default=0)
    response_ms = Column(Float, default=0.0)
    status_code = Column(Integer, default=200)
    timestamp = Column(DateTime, default=datetime.utcnow)

    api_key = relationship("APIKey", back_populates="usage_logs")

    __table_args__ = (
        Index("idx_usage_timestamp", "timestamp"),
        Index("idx_usage_provider", "provider"),
    )


# Tier configurations
TIER_LIMITS = {
    "free": {"requests_per_day": 100},
    "pro": {"requests_per_day": 5_000},
    "team": {"requests_per_day": 50_000},
}

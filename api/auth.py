"""
TokenShield Authentication & Rate Limiting
============================================
API key validation, rate limiting, and user management.

Author: Wesley Foreman (wforeman58@gmail.com)
Copyright 2026. All rights reserved.
"""

from datetime import datetime, timezone

from fastapi import HTTPException, Depends
from sqlalchemy.orm import Session

from db.connection import get_db
from db.models import APIKey, TIER_LIMITS


async def get_proxy_key(
    x_proxy_key: str = None,
    db: Session = Depends(get_db),
) -> APIKey:
    """
    Validate proxy key from X-Proxy-Key header.
    Returns the APIKey record or raises 401/403.
    """
    if not x_proxy_key or not x_proxy_key.startswith("sk-ts-"):
        raise HTTPException(status_code=401, detail="Invalid proxy key format")

    api_key = db.query(APIKey).filter(APIKey.key == x_proxy_key).first()

    if not api_key:
        raise HTTPException(status_code=401, detail="Invalid proxy key")

    if not api_key.enabled:
        raise HTTPException(status_code=403, detail="Proxy key disabled")

    return api_key


def check_rate_limit(api_key: APIKey, db: Session):
    """Check and update rate limit. Raises 429 if exceeded."""
    # Use naive UTC (SQLite strips timezone info on read)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    if api_key.requests_reset_at and (now - api_key.requests_reset_at).days >= 1:
        api_key.requests_today = 0
        api_key.requests_reset_at = now
        db.commit()

    if api_key.requests_today >= api_key.request_limit:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Rate limit exceeded",
                "limit": api_key.request_limit,
                "tier": api_key.tier,
                "upgrade_url": "/dashboard",
            },
        )

    api_key.requests_today += 1
    db.commit()


def create_api_key(email: str, tier: str, db: Session) -> APIKey:
    """Create a new proxy API key for a user."""
    limits = TIER_LIMITS.get(tier, TIER_LIMITS["free"])

    key = APIKey(
        key=APIKey.generate_key(),
        user_email=email,
        tier=tier,
        request_limit=limits["requests_per_day"],
    )
    db.add(key)
    db.commit()
    db.refresh(key)
    return key

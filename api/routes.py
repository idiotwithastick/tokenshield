"""
TokenShield API Routes
========================
Status, savings, signup, and cache management endpoints.

Author: Wesley Foreman (wforeman58@gmail.com)
Copyright 2026. All rights reserved.
"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import func, Integer, case

from db.connection import get_db
from db.models import APIKey, UsageLog, TIER_LIMITS
from api.auth import check_rate_limit

router = APIRouter(prefix="/v1")


# ─── Request Models ───────────────────────────────────────────────

class SignupRequest(BaseModel):
    email: str = Field(..., min_length=5, max_length=256)


# ─── /v1/signup ───────────────────────────────────────────────────

@router.post("/signup")
async def signup(
    req: SignupRequest,
    db: Session = Depends(get_db),
):
    """Create a free account. Returns proxy API key."""
    from api.auth import create_api_key

    existing = db.query(APIKey).filter(APIKey.user_email == req.email).first()
    if existing:
        raise HTTPException(
            status_code=409,
            detail="An account already exists for this email.",
        )

    api_key = create_api_key(email=req.email, tier="free", db=db)

    return {
        "api_key": api_key.key,
        "tier": api_key.tier,
        "request_limit": api_key.request_limit,
        "message": "Store your proxy key securely. It cannot be retrieved later.",
    }


# ─── /v1/status ───────────────────────────────────────────────────

@router.get("/status")
async def status(
    request: Request,
    x_proxy_key: str = None,
    db: Session = Depends(get_db),
):
    """Account stats, cache stats, savings summary."""
    api_key = _get_key(_extract_key(request, x_proxy_key), db)

    from api.proxy import get_shared_gateway
    cache_stats = get_shared_gateway().get_cache_stats()

    return {
        "account": {
            "email": api_key.user_email,
            "tier": api_key.tier,
            "requests_today": api_key.requests_today,
            "request_limit": api_key.request_limit,
            "total_requests": api_key.total_requests,
            "total_cache_hits": api_key.total_cache_hits,
            "total_tokens_saved": api_key.total_tokens_saved,
        },
        "cache": {
            "entries": cache_stats["entries"],
            "hit_rate": cache_stats["hit_rate"],
            "exact_hits": cache_stats["exact_hits"],
            "near_hits": cache_stats["near_hits"],
            "total_misses": cache_stats["misses"],
        },
        "engine": {
            "status": "active",
            "execution": "CPU",
        },
    }


# ─── /v1/savings ──────────────────────────────────────────────────

@router.get("/savings")
async def savings(
    request: Request,
    x_proxy_key: str = None,
    db: Session = Depends(get_db),
):
    """Detailed token savings breakdown by provider."""
    api_key = _get_key(_extract_key(request, x_proxy_key), db)

    # Query usage logs for this user grouped by provider
    logs = (
        db.query(
            UsageLog.provider,
            func.count(UsageLog.id).label("total"),
            func.sum(UsageLog.tokens_saved).label("saved"),
            func.sum(UsageLog.input_tokens).label("input"),
            func.sum(UsageLog.output_tokens).label("output"),
            func.sum(case((UsageLog.cached == True, 1), else_=0)).label("hits"),
        )
        .filter(UsageLog.api_key_id == api_key.id)
        .group_by(UsageLog.provider)
        .all()
    )

    by_provider = {}
    total_saved = 0
    total_used = 0

    for row in logs:
        provider = row.provider or "unknown"
        saved = int(row.saved or 0)
        inp = int(row.input or 0)
        out = int(row.output or 0)
        total_saved += saved
        total_used += inp + out
        by_provider[provider] = {
            "requests": int(row.total or 0),
            "tokens_saved": saved,
            "tokens_used": inp + out,
            "cache_hits": int(row.hits or 0),
        }

    return {
        "total_tokens_saved": total_saved,
        "total_tokens_used": total_used,
        "savings_percent": round(total_saved / max(total_saved + total_used, 1) * 100, 1),
        "by_provider": by_provider,
    }


# ─── /v1/cache/clear ─────────────────────────────────────────────

@router.delete("/cache/clear")
async def clear_cache(
    request: Request,
    x_proxy_key: str = None,
    db: Session = Depends(get_db),
):
    """Clear all cached responses for this user."""
    api_key = _get_key(_extract_key(request, x_proxy_key), db)
    from api.proxy import get_shared_gateway
    removed = get_shared_gateway().clear_user_cache(api_key.user_email)
    return {"cleared": removed, "message": f"Removed {removed} cached responses."}


# ─── Helpers ──────────────────────────────────────────────────────

def _extract_key(request: Request, query_param: Optional[str] = None) -> Optional[str]:
    """Extract proxy key from X-Proxy-Key header OR query param. Header wins."""
    key = request.headers.get("x-proxy-key")
    if not key:
        key = query_param
    return key


def _get_key(proxy_key: Optional[str], db: Session) -> APIKey:
    """Validate proxy key."""
    key = proxy_key
    if not key:
        raise HTTPException(401, detail="Missing X-Proxy-Key header")
    if not key.startswith("sk-ts-"):
        raise HTTPException(401, detail="Invalid proxy key format")

    api_key = db.query(APIKey).filter(APIKey.key == key).first()
    if not api_key:
        raise HTTPException(401, detail="Invalid proxy key")
    if not api_key.enabled:
        raise HTTPException(403, detail="Proxy key disabled")
    return api_key

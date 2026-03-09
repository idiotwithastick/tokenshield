"""
TokenShield Proxy Routes
==========================
Drop-in replacement endpoints for Claude, OpenAI, and Google APIs.
Users change only the base URL. Everything else stays the same.

Flow per request:
  1. Validate proxy key (X-Proxy-Key header)
  2. Extract provider key from original auth header
  3. Normalize prompt → thermosolve → cache check
  4. HIT → return cached response body (zero API tokens)
  5. MISS → forward to real API → cache → return

Author: Wesley Foreman (wforeman58@gmail.com)
Copyright 2026. All rights reserved.
"""

import json
import time
from typing import Optional

from fastapi import APIRouter, Request, HTTPException, Depends, Header
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from core.gateway import EnforcementGateway
from core.providers import AnthropicAdapter, OpenAIAdapter, GoogleAdapter
from db.connection import get_db
from db.models import APIKey, UsageLog
from api.auth import get_proxy_key, check_rate_limit

router = APIRouter()

# Singleton gateway — shared with routes.py via get_shared_gateway()
_gateway = EnforcementGateway()


def get_shared_gateway() -> EnforcementGateway:
    """Return THE singleton gateway. Used by routes.py for stats/clear."""
    return _gateway


def _get_gateway() -> EnforcementGateway:
    return _gateway


# ─── Anthropic Proxy (POST /v1/messages) ─────────────────────────

@router.post("/v1/messages")
async def proxy_anthropic(
    request: Request,
    db: Session = Depends(get_db),
):
    """Drop-in replacement for api.anthropic.com/v1/messages."""
    return await _handle_proxy(
        request=request,
        adapter=AnthropicAdapter,
        db=db,
        endpoint="/v1/messages",
    )


# ─── OpenAI Proxy (POST /v1/chat/completions) ────────────────────

@router.post("/v1/chat/completions")
async def proxy_openai(
    request: Request,
    db: Session = Depends(get_db),
):
    """Drop-in replacement for api.openai.com/v1/chat/completions."""
    return await _handle_proxy(
        request=request,
        adapter=OpenAIAdapter,
        db=db,
        endpoint="/v1/chat/completions",
    )


# ─── Google Proxy (POST /v1beta/models/{model}:generateContent) ──

@router.post("/v1beta/models/{model}:generateContent")
async def proxy_google(
    model: str,
    request: Request,
    db: Session = Depends(get_db),
):
    """Drop-in replacement for Google Gemini API."""
    return await _handle_proxy(
        request=request,
        adapter=GoogleAdapter,
        db=db,
        endpoint=f"/v1beta/models/{model}:generateContent",
        model=model,
    )


# ─── Shared Handler ──────────────────────────────────────────────

async def _handle_proxy(
    request: Request,
    adapter,
    db: Session,
    endpoint: str,
    model: str = None,
):
    """
    Shared proxy handler for all providers.
    """
    t0 = time.perf_counter()

    # 1. Auth: validate proxy key from X-Proxy-Key header
    headers = dict(request.headers)
    proxy_key_str = headers.get("x-proxy-key", "")
    if not proxy_key_str:
        raise HTTPException(401, detail="Missing X-Proxy-Key header")

    api_key = _validate_proxy_key(proxy_key_str, db)
    check_rate_limit(api_key, db)

    # 2. Extract provider key from original auth header
    provider_key = adapter.extract_auth_key(headers)
    if not provider_key:
        raise HTTPException(
            401,
            detail=f"Missing provider API key. Include the original auth header for {adapter.provider}.",
        )

    # 3. Parse request body
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, detail="Invalid JSON body")

    # 4. Normalize prompt for thermosolve
    normalized = adapter.extract_prompt(body)
    if not normalized.strip():
        raise HTTPException(400, detail="Could not extract prompt from request body")

    # 5. Physics-first cache check
    gateway = _get_gateway()
    cached, cache_id, attestation = gateway.check_cache(
        normalized_prompt=normalized,
        provider=adapter.provider,
        user_id=api_key.user_email,
    )

    # Check if blocked by CBFs
    if attestation.blocked:
        _log_usage(db, api_key, endpoint, adapter.provider, False, 0, 0, 0,
                   (time.perf_counter() - t0) * 1000, 403)
        raise HTTPException(403, detail="Request blocked by content safety checks")

    if cached is not None:
        # CACHE HIT — return cached response body
        total_ms = (time.perf_counter() - t0) * 1000
        tokens_saved = cached.get("input_tokens", 0) + cached.get("output_tokens", 0)

        # Update account stats
        api_key.total_cache_hits += 1
        api_key.total_tokens_saved += tokens_saved
        api_key.total_requests += 1
        db.commit()

        _log_usage(db, api_key, endpoint, adapter.provider, True,
                   cached.get("input_tokens", 0), cached.get("output_tokens", 0),
                   tokens_saved, total_ms, 200)

        response = JSONResponse(
            content=cached["response_body"],
            status_code=200,
        )
        response.headers["X-Cache-Status"] = "HIT"
        response.headers["X-Tokens-Saved"] = str(tokens_saved)
        response.headers["X-Proxy-Time-Ms"] = f"{total_ms:.1f}"
        return response

    # CACHE MISS — forward to provider
    try:
        if adapter.provider == "google" and model:
            result = await adapter.forward(body, provider_key, model=model, extra_headers=headers)
        else:
            result = await adapter.forward(body, provider_key, extra_headers=headers)
    except Exception as e:
        total_ms = (time.perf_counter() - t0) * 1000
        _log_usage(db, api_key, endpoint, adapter.provider, False, 0, 0, 0,
                   total_ms, 502)
        raise HTTPException(502, detail=f"Provider error: {str(e)}")

    total_ms = (time.perf_counter() - t0) * 1000

    # Cache the response for future reuse
    if result.status_code == 200:
        gateway.cache_response(
            cache_id=cache_id,
            provider=adapter.provider,
            normalized_prompt=normalized,
            response_body=result.body,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            user_id=api_key.user_email,
        )

    # Update account stats
    api_key.total_requests += 1
    db.commit()

    tokens_used = result.input_tokens + result.output_tokens
    _log_usage(db, api_key, endpoint, adapter.provider, False,
               result.input_tokens, result.output_tokens, 0, total_ms,
               result.status_code)

    response = JSONResponse(
        content=result.body,
        status_code=result.status_code,
    )
    response.headers["X-Cache-Status"] = "MISS"
    response.headers["X-Tokens-Used"] = str(tokens_used)
    response.headers["X-Proxy-Time-Ms"] = f"{total_ms:.1f}"
    return response


# ─── Helpers ──────────────────────────────────────────────────────

def _validate_proxy_key(key: str, db: Session) -> APIKey:
    """Validate a proxy API key."""
    if not key.startswith("sk-ts-"):
        raise HTTPException(401, detail="Invalid proxy key format. Expected sk-ts-...")

    api_key = db.query(APIKey).filter(APIKey.key == key).first()
    if not api_key:
        raise HTTPException(401, detail="Invalid proxy key")
    if not api_key.enabled:
        raise HTTPException(403, detail="Proxy key disabled")
    return api_key


def _log_usage(db: Session, api_key: APIKey, endpoint: str, provider: str,
               cached: bool, input_tokens: int, output_tokens: int,
               tokens_saved: int, response_ms: float, status_code: int):
    """Log proxy usage."""
    log = UsageLog(
        api_key_id=api_key.id,
        endpoint=endpoint,
        provider=provider,
        cached=cached,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        tokens_saved=tokens_saved,
        response_ms=response_ms,
        status_code=status_code,
    )
    db.add(log)
    db.commit()

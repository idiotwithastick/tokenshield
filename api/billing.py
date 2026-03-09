"""
TokenShield Billing (Stripe)
===============================
Stripe Checkout for upgrades + webhook for tier changes.

Author: Wesley Foreman (wforeman58@gmail.com)
Copyright 2026. All rights reserved.
"""

import os
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from db.connection import get_db
from db.models import APIKey, TIER_LIMITS

router = APIRouter(prefix="/billing")

STRIPE_SECRET = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")

# Price IDs from Stripe Dashboard
STRIPE_PRICES = {
    "pro": os.environ.get("STRIPE_PRICE_PRO", ""),
    "team": os.environ.get("STRIPE_PRICE_TEAM", ""),
}


@router.post("/checkout")
async def create_checkout(
    tier: str,
    api_key_header: str,
    db: Session = Depends(get_db),
):
    """Create a Stripe Checkout session for upgrading."""
    if not STRIPE_SECRET:
        raise HTTPException(status_code=503, detail="Billing not configured")

    if tier not in ("pro", "team"):
        raise HTTPException(status_code=400, detail="Invalid tier. Choose 'pro' or 'team'.")

    api_key = db.query(APIKey).filter(APIKey.key == api_key_header).first()
    if not api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        import stripe
        stripe.api_key = STRIPE_SECRET

        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[{
                "price": STRIPE_PRICES[tier],
                "quantity": 1,
            }],
            mode="subscription",
            success_url=os.environ.get("BASE_URL", "https://tokenshield.repl.co") + "/dashboard?upgraded=true",
            cancel_url=os.environ.get("BASE_URL", "https://tokenshield.repl.co") + "/dashboard?cancelled=true",
            client_reference_id=api_key.key,
            customer_email=api_key.user_email,
            metadata={"tier": tier, "api_key_id": str(api_key.id)},
        )

        return {"checkout_url": session.url}

    except ImportError:
        raise HTTPException(status_code=503, detail="Stripe not installed. Run: pip install stripe")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Billing error: {str(e)}")


@router.post("/webhook")
async def stripe_webhook(request: Request, db: Session = Depends(get_db)):
    """Handle Stripe webhook events (subscription created/updated/cancelled)."""
    if not STRIPE_SECRET or not STRIPE_WEBHOOK_SECRET:
        raise HTTPException(status_code=503, detail="Billing not configured")

    try:
        import stripe
        stripe.api_key = STRIPE_SECRET

        payload = await request.body()
        sig_header = request.headers.get("stripe-signature", "")

        event = stripe.Webhook.construct_event(
            payload, sig_header, STRIPE_WEBHOOK_SECRET
        )
    except ImportError:
        raise HTTPException(status_code=503, detail="Stripe not installed")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Webhook error: {str(e)}")

    # Handle subscription events
    if event["type"] in ("checkout.session.completed", "customer.subscription.updated"):
        _handle_subscription_change(event, db)
    elif event["type"] == "customer.subscription.deleted":
        _handle_subscription_cancelled(event, db)

    return {"status": "ok"}


def _handle_subscription_change(event, db: Session):
    """Upgrade user tier when subscription is created/updated."""
    data = event["data"]["object"]
    metadata = data.get("metadata", {})
    tier = metadata.get("tier", "pro")
    api_key_id = metadata.get("api_key_id")

    if not api_key_id:
        # Try client_reference_id (the API key string)
        ref_id = data.get("client_reference_id")
        if ref_id:
            api_key = db.query(APIKey).filter(APIKey.key == ref_id).first()
        else:
            return
    else:
        api_key = db.query(APIKey).filter(APIKey.id == int(api_key_id)).first()

    if not api_key:
        return

    limits = TIER_LIMITS.get(tier, TIER_LIMITS["free"])
    api_key.tier = tier
    api_key.request_limit = limits["requests_per_day"]
    api_key.stripe_customer_id = data.get("customer")
    api_key.stripe_subscription_id = data.get("subscription") or data.get("id")
    db.commit()


def _handle_subscription_cancelled(event, db: Session):
    """Downgrade to free tier when subscription is cancelled."""
    data = event["data"]["object"]
    customer_id = data.get("customer")

    if not customer_id:
        return

    api_key = db.query(APIKey).filter(APIKey.stripe_customer_id == customer_id).first()
    if not api_key:
        return

    limits = TIER_LIMITS["free"]
    api_key.tier = "free"
    api_key.request_limit = limits["requests_per_day"]
    api_key.stripe_subscription_id = None
    db.commit()

"""
TokenShield — Token-Saving Proxy for Agentic AI
==================================================
Drop-in proxy for Claude, OpenAI, and Google APIs.
Caches responses by semantic similarity. Saves tokens.

Author: Wesley Foreman (wforeman58@gmail.com)
Copyright 2026. All rights reserved.
"""

import os
import sys
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from db.connection import init_db
from api.proxy import router as proxy_router
from api.routes import router as status_router
from api.billing import router as billing_router


@asynccontextmanager
async def lifespan(app):
    """Startup/shutdown lifecycle."""
    init_db()
    print("TokenShield started. Cache engine active.")
    yield


app = FastAPI(
    title="TokenShield",
    description="Token-saving proxy for agentic AI workflows",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url=None,
    lifespan=lifespan,
)

# Proxy routes (provider-compatible endpoints)
app.include_router(proxy_router)
# Status/savings/signup routes
app.include_router(status_router)
# Billing routes
app.include_router(billing_router)


# Static portal
_static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")


@app.get("/")
async def landing():
    p = os.path.join(_static_dir, "index.html")
    if os.path.exists(p):
        return FileResponse(p)
    return {"name": "TokenShield", "version": "1.0.0", "docs": "/api/docs"}


@app.get("/docs")
async def docs_page():
    p = os.path.join(_static_dir, "docs.html")
    if os.path.exists(p):
        return FileResponse(p)
    return {"redirect": "/api/docs"}


@app.get("/playground")
async def playground_page():
    p = os.path.join(_static_dir, "playground.html")
    if os.path.exists(p):
        return FileResponse(p)
    return {"message": "Coming soon"}


@app.get("/dashboard")
async def dashboard_page():
    p = os.path.join(_static_dir, "dashboard.html")
    if os.path.exists(p):
        return FileResponse(p)
    return {"message": "Coming soon"}


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)

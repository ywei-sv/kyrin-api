"""Kyrin API — app factory."""
import os
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.routers import chat, search, crawl, anime, chats, rag as rag_router, youtube


# ── Optional API Key Middleware ──────────────────────────
# Set KYRIN_REQUIRE_API_KEY=1 and KYRIN_API_KEY to require key on non-CORS routes
_REQUIRE_KEY = os.environ.get("KYRIN_REQUIRE_API_KEY", "")
_SERVER_KEY = os.environ.get("KYRIN_API_KEY", "")

PUBLIC_PATHS = {"/api/health", "/docs", "/openapi.json", "/redoc"}


def create_app() -> FastAPI:
    app = FastAPI(title="Kyrin API", version="1.0.0", docs_url="/docs")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Optional: API key check middleware
    if _REQUIRE_KEY and _SERVER_KEY:

        @app.middleware("http")
        async def api_key_middleware(request: Request, call_next):
            # Skip public paths and OPTIONS
            if request.method == "OPTIONS" or request.url.path in PUBLIC_PATHS:
                return await call_next(request)
            auth = request.headers.get("Authorization", "")
            # Parse "Bearer <key>" or raw key in header
            key = auth.removeprefix("Bearer ").strip() if auth else ""
            if not key:
                key = request.headers.get("X-API-Key", "")
            if key != _SERVER_KEY:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Missing or invalid API key. Set Authorization: Bearer <key>"},
                )
            return await call_next(request)

    app.include_router(chat.router, prefix="/api")
    app.include_router(search.router, prefix="/api")
    app.include_router(crawl.router, prefix="/api")
    app.include_router(anime.router, prefix="/api")
    app.include_router(chats.router, prefix="/api")
    app.include_router(rag_router.router, prefix="/api")
    app.include_router(youtube.router, prefix="/api")
    return app


app = create_app()


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "service": "kyrin-api",
        "version": "1.0.0",
        "model": os.environ.get("KYRIN_MODEL", "deepseek-v4-flash"),
    }

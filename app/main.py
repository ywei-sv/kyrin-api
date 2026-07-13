"""Kyrin API — app factory."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware
from slowapi.errors import RateLimitExceeded
from app.routers import chat, search, crawl, anime, chats, rag as rag_router, youtube


# ── Optional API Key Middleware ──────────────────────────
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

    # ── Rate Limiting (optional) ──────────────────────────
    # Set KYRIN_RATE_LIMIT=60/minute (default: disabled)
    rate_limit_str = os.environ.get("KYRIN_RATE_LIMIT", "")
    if rate_limit_str:
        limiter = Limiter(
            key_func=get_remote_address,
            default_limits=[rate_limit_str],
            enabled=True,
        )
        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        app.add_middleware(SlowAPIMiddleware)

    # ── API Key Check Middleware (optional) ────────────────
    if _REQUIRE_KEY and _SERVER_KEY:

        @app.middleware("http")
        async def api_key_middleware(request: Request, call_next):
            if (
                request.method == "OPTIONS"
                or request.url.path in PUBLIC_PATHS
            ):
                return await call_next(request)
            auth = request.headers.get("Authorization", "")
            key = auth.removeprefix("Bearer ").strip() if auth else ""
            if not key:
                key = request.headers.get("X-API-Key", "")
            if key != _SERVER_KEY:
                return JSONResponse(
                    status_code=401,
                    content={
                        "detail": "Missing or invalid API key. Set Authorization: Bearer <key>"
                    },
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

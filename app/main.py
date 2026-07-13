"""Kyrin API — pure API server (no static frontend).
Frontend (kyrin-landing) connects via Vite proxy at :5270."""
import os
from contextlib import asynccontextmanager
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
import httpx

from app.routers import chat, search, crawl, anime, chats, rag as rag_router, youtube


# ── Config ──────────────────────────────────────────────
_REQUIRE_KEY = os.environ.get("KYRIN_REQUIRE_API_KEY", "") in ("1", "true", "yes", "on")
_SERVER_KEY = os.environ.get("KYRIN_API_KEY", "")
PUBLIC_PATHS = {"/api/health", "/api/models", "/docs", "/openapi.json", "/redoc"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: run migrations and warmup."""
    from app.routers.chats import migrate_json_to_sqlite
    from app.routers.rag import warmup_rag

    await migrate_json_to_sqlite()
    await warmup_rag()
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Kyrin API", version="1.0.0", docs_url="/docs", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Rate Limiting (optional) ──────────────────────────
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
            if request.method == "OPTIONS" or request.url.path in PUBLIC_PATHS:
                return await call_next(request)
            auth = request.headers.get("Authorization", "")
            key = auth.removeprefix("Bearer ").strip() if auth else ""
            if not key:
                key = request.headers.get("X-API-Key", "")
            if key != _SERVER_KEY:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Missing or invalid API key. Set Authorization: Bearer <key>"},
                )
            return await call_next(request)

    # ── API Routes ───────────────────────────────────────
    app.include_router(chat.router, prefix="/api")
    app.include_router(search.router, prefix="/api")
    app.include_router(crawl.router, prefix="/api")
    app.include_router(anime.router, prefix="/api")
    app.include_router(chats.router, prefix="/api")
    app.include_router(rag_router.router, prefix="/api")
    app.include_router(youtube.router, prefix="/api")

    # ── API-only routes ──────────────────────────────────
    @app.get("/api/health")
    async def health():
        return {
            "status": "ok",
            "service": "kyrin-api",
            "version": "1.0.0",
            "model": os.environ.get("KYRIN_MODEL", "deepseek-v4-flash"),
        }

    @app.get("/api/models")
    async def list_models():
        """Fetch available models from OpenCode, with fallback."""
        api_key = os.environ.get("KYRIN_API_KEY", "")
        base_url = os.environ.get("KYRIN_BASE_URL", "https://opencode.ai/zen/go/v1")
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    f"{base_url.rstrip('/v1')}/v1/models",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                resp.raise_for_status()
                data = resp.json()
                return {"models": [{"id": m.get("id"), "name": m.get("id")} for m in data.get("data", [])]}
        except Exception:
            return {"models": [
                {"id": "deepseek-v4-flash", "name": "DeepSeek V4 Flash"},
                {"id": "mimo-v2.5", "name": "MiMo V2.5"},
                {"id": "qwen3.7-plus", "name": "Qwen 3.7 Plus"},
            ]}

    return app


app = create_app()

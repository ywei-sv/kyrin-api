"""Kyrin API — app factory."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import chat, search, crawl, anime, chats, rag as rag_router, youtube


def create_app() -> FastAPI:
    app = FastAPI(title="Kyrin API", version="1.0.0", docs_url="/docs")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
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

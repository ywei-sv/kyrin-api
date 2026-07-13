"""
Kyrin API — AI harness backend
FastAPI server providing LLM proxy, web search, crawl, anime identification, and vision.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import chat, search, crawl, anime

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


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "service": "kyrin-api",
        "version": "1.0.0",
        "model": os.environ.get("KYRIN_MODEL", "deepseek-v4-flash"),
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)

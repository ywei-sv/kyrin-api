"""
Web search — multi-engine dispatcher inspired by open-webui's retrieval system.
Supports SearXNG, DuckDuckGo, and custom external API, with domain filtering,
result deduplication, and query rewriting.
"""
import os
import re
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
import httpx

router = APIRouter()

# ── Config ──────────────────────────────────────────────────────────────
SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://127.0.0.1:8080")
DDGS_AVAILABLE = False  # optional: pip install duckduckgo_search
try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    pass

WEB_SEARCH_RESULT_COUNT = int(os.environ.get("WEB_SEARCH_RESULT_COUNT", "5"))
# Comma-separated domain allowlist — empty = no filter
WEB_SEARCH_DOMAIN_FILTER = [d.strip() for d in os.environ.get("WEB_SEARCH_DOMAIN_FILTER", "").split(",") if d.strip()]

# ── Models ──────────────────────────────────────────────────────────────
from pydantic import BaseModel


class SearchResult(BaseModel):
    title: str
    url: str
    content: str
    engine: str = ""
    score: float = 1.0


# ── Helpers ─────────────────────────────────────────────────────────────
def _deduplicate(results: list[SearchResult]) -> list[SearchResult]:
    seen = set()
    out = []
    for r in results:
        if r.url not in seen:
            seen.add(r.url)
            out.append(r)
    return out


def _filter_domain(results: list[SearchResult]) -> list[SearchResult]:
    if not WEB_SEARCH_DOMAIN_FILTER:
        return results
    return [r for r in results if any(d in r.url for d in WEB_SEARCH_DOMAIN_FILTER)]


def _rewrite_query(query: str) -> str:
    """Light query rewriting: strip boilerplate, add year context."""
    q = query.strip()
    # Remove common conversational prefixes
    for prefix in ["help me ", "i want to ", "can you ", "please ", "tell me about "]:
        if q.lower().startswith(prefix):
            q = q[len(prefix):]
    return q.strip()


# ── Engine: SearXNG ─────────────────────────────────────────────────────
async def _search_searxng(query: str, count: int) -> list[SearchResult]:
    params = {
        "q": query,
        "format": "json",
        "language": "en",
        "categories": "general",
        "pageno": 1,
    }
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(f"{SEARXNG_URL}/search", params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"SearXNG error: {e}")

    results = data.get("results", [])
    # Sort by score descending
    results.sort(key=lambda x: x.get("score", 0), reverse=True)
    return [
        SearchResult(
            title=r.get("title", ""),
            url=r.get("url", ""),
            content=r.get("content", ""),
            engine=r.get("engine", "searxng"),
            score=r.get("score", 0),
        )
        for r in results[:count]
    ]


# ── Engine: DuckDuckGo (fallback) ───────────────────────────────────────
async def _search_ddg(query: str, count: int) -> list[SearchResult]:
    if not DDGS_AVAILABLE:
        raise HTTPException(status_code=501, detail="DuckDuckGo search not available (install duckduckgo_search)")
    try:
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        def _sync():
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=count))

        with ThreadPoolExecutor() as pool:
            raw = await asyncio.get_event_loop().run_in_executor(pool, _sync)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"DuckDuckGo error: {e}")

    return [
        SearchResult(
            title=r.get("title", ""),
            url=r.get("href", ""),
            content=r.get("body", ""),
            engine="duckduckgo",
        )
        for r in raw[:count]
    ]


# ── Engine selector ─────────────────────────────────────────────────────
ENGINES = {
    "searxng": _search_searxng,
    "ddg": _search_ddg,
}


# ── Endpoint ────────────────────────────────────────────────────────────
@router.get("/search")
async def search(
    q: str = Query(..., description="Search query"),
    limit: int = Query(WEB_SEARCH_RESULT_COUNT, ge=1, le=50),
    engine: str = Query("searxng", description="Search engine"),
    rewrite: bool = Query(True, description="Auto-rewrite query"),
    fallback: bool = Query(True, description="Auto-fallback to DuckDuckGo when SearXNG fails"),
):
    """Search web with engine selection, dedup, domain filtering, and auto-fallback."""
    query = _rewrite_query(q) if rewrite else q.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Empty query")

    searcher = ENGINES.get(engine)
    if not searcher:
        raise HTTPException(status_code=400, detail=f"Unknown engine: {engine}. Choose: {', '.join(ENGINES)}")

    # Try primary engine, with fallback chain
    results = []
    engines_tried = []
    last_error = None

    primary_engines = [engine]
    if fallback and engine == "searxng" and DDGS_AVAILABLE:
        primary_engines.append("ddg")

    for eng in primary_engines:
        searcher_fn = ENGINES.get(eng)
        if not searcher_fn:
            continue
        try:
            engines_tried.append(eng)
            batch = await searcher_fn(query, limit)
            if batch:
                results = batch
                engine = eng  # report which engine actually returned results
                break
        except HTTPException:
            raise
        except Exception as e:
            last_error = str(e)
            continue  # try next engine in chain

    if not results:
        detail = f"All engines failed" if len(engines_tried) > 1 else f"{engine} error"
        if last_error:
            detail += f": {last_error}"
        raise HTTPException(status_code=502, detail=detail)

    results = _filter_domain(results)
    results = _deduplicate(results)

    return {
        "query": query,
        "engine": engine,
        "engines_tried": engines_tried,
        "fallback_used": len(engines_tried) > 1,
        "results": [r.model_dump() for r in results],
    }

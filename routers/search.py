"""
Web search via self-hosted SearXNG instance.
"""
import os

from fastapi import APIRouter, HTTPException, Query
import httpx

router = APIRouter()

SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://127.0.0.1:8080")


@router.get("/search")
async def search(
    q: str = Query(..., description="Search query"),
    limit: int = Query(5, ge=1, le=50),
):
    """Search the web via SearXNG. Returns structured results."""
    params = {"q": q, "format": "json", "language": "en", "categories": "general"}
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(f"{SEARXNG_URL}/search", params=params)
            resp.raise_for_status()
            data = resp.json()
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"SearXNG unreachable: {e}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    results = data.get("results", [])[:limit]
    return {
        "query": q,
        "number_of_results": data.get("number_of_results", 0),
        "results": [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
                "engine": r.get("engine", ""),
            }
            for r in results
        ],
    }

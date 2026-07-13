"""
Web page crawler — fetches and extracts content from URLs.
Uses jina.ai public reader as a fallback for static sites.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import httpx

router = APIRouter()


class CrawlRequest(BaseModel):
    url: str
    max_chars: int = 15000


class CrawlResponse(BaseModel):
    url: str
    title: str
    content: str
    content_length: int
    truncated: bool


@router.post("/crawl", response_model=CrawlResponse)
async def crawl(req: CrawlRequest):
    """Fetch and extract content from a web page."""
    url = req.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL is required")
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    if url in ("https://", "http://"):
        raise HTTPException(status_code=400, detail="Invalid URL")

    try:
        # Try direct fetch first
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            resp = await client.get(
                url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/125.0.0.0 Safari/537.36"
                    )
                },
            )
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            if "text/html" in content_type or "text/plain" in content_type:
                raw = resp.text
            else:
                # Fallback to jina.ai reader
                return await _jina_read(url, req.max_chars)

    except Exception:
        # Fallback to jina.ai reader
        return await _jina_read(url, req.max_chars)

    # Extract title
    title = url
    if "<title>" in raw:
        start = raw.index("<title>") + 7
        end = raw.index("</title>", start)
        title = raw[start:end].strip()[:200]

    # Strip tags for text content
    import re
    text = raw
    # Remove script/style blocks first
    text = re.sub(r"<script[^>]*>[\s\S]*?</script>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<style[^>]*>[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
    # Remove HTML tags (handle nested quotes in attributes)
    text = re.sub(r"<[^>]*>", " ", text)
    # Decode common HTML entities
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = re.sub(r"&#(\d+);", lambda m: chr(int(m.group(1))), text)
    text = re.sub(r"\s+", " ", text).strip()

    truncated = len(text) > req.max_chars
    content = text[: req.max_chars]

    return CrawlResponse(
        url=url, title=title, content=content,
        content_length=len(text), truncated=truncated,
    )


async def _jina_read(url: str, max_chars: int) -> CrawlResponse:
    """Fallback: use r.jina.ai public reader."""
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        resp = await client.get(
            f"https://r.jina.ai/{url}",
            headers={"Accept": "text/plain", "X-With-Links-Summary": "true"},
        )
        resp.raise_for_status()
        text = resp.text
        title = url
        for prefix in ["Title: ", "Title:"]:
            if text.startswith(prefix):
                end = text.find("\n")
                title = text[len(prefix) : end].strip()[:200]
                text = text[end:].strip()
                break

    truncated = len(text) > max_chars
    content = text[:max_chars]
    return CrawlResponse(
        url=url, title=title, content=content,
        content_length=len(text), truncated=truncated,
    )

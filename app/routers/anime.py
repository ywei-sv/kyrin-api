"""
Anime identification via trace.moe API.
Accepts image URL or base64-encoded image.
"""
import os
import base64
from fastapi import APIRouter, HTTPException, Form, File, UploadFile
from pydantic import BaseModel
import httpx

router = APIRouter()

TRACE_MOE = os.environ.get("TRACE_MOE_URL", "https://api.trace.moe")


class AnimeSearchResponse(BaseModel):
    title_native: str | None = None
    title_romaji: str | None = None
    title_english: str | None = None
    episode: int | None = None
    similarity: float | None = None
    timestamp_from: float | None = None
    timestamp_to: float | None = None
    image_url: str | None = None
    video_url: str | None = None
    is_adult: bool = False
    other_matches: list[dict] = []


@router.post("/anime-search", response_model=list[AnimeSearchResponse])
async def anime_search(
    url: str | None = Form(None),
    file: UploadFile | None = File(None),
    anilist_info: bool = Form(True),
):
    """Identify anime from an image. Accepts image URL, uploaded file, or query param."""
    return await _do_anime_search(url, file, anilist_info)


@router.get("/anime-search", response_model=list[AnimeSearchResponse])
async def anime_search_get(
    url: str | None = None,
    anilist_info: bool = True,
):
    """GET variant — accepts ?url= for browser convenience."""
    return await _do_anime_search(url, None, anilist_info)


async def _do_anime_search(
    url: str | None,
    file: UploadFile | None,
    anilist_info: bool,
) -> list[AnimeSearchResponse]:
    image_url = url

    # Handle file upload → convert to data URI
    if file and not image_url:
        contents = await file.read()
        b64 = base64.b64encode(contents).decode()
        mime = file.content_type or "image/jpeg"
        image_url = f"data:{mime};base64,{b64}"

    if not image_url:
        raise HTTPException(status_code=400, detail="Provide 'url' or upload an image")

    params = {"anilistInfo": "true" if anilist_info else "false"}
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{TRACE_MOE}/search",
                params={"url": image_url, **params},
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"trace.moe unreachable: {e}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    results = data.get("result", [])
    if not results:
        return []

    def _map(r: dict) -> AnimeSearchResponse:
        a = r.get("anilist") or {}
        title = a.get("title") or {}
        return AnimeSearchResponse(
            title_native=title.get("native"),
            title_romaji=title.get("romaji"),
            title_english=title.get("english"),
            episode=r.get("episode"),
            similarity=r.get("similarity"),
            timestamp_from=r.get("from"),
            timestamp_to=r.get("to"),
            image_url=r.get("image"),
            video_url=r.get("video"),
            is_adult=a.get("isAdult", False),
        )

    top = _map(results[0])
    others = [_map(r) for r in results[1:4]]

    return [top, *others]

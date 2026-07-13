"""
YouTube transcript tool — fetches transcript/subtitles from YouTube videos.
Triggered by frontend when user types "yt: <url>" or "youtube: <url>".
"""

import re
from fastapi import APIRouter, HTTPException, Query
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound

router = APIRouter()

VIDEO_ID_RE = re.compile(
    r"(?:v=|/v/|youtu\.be/|/embed/|/shorts/|/watch\?v=)([a-zA-Z0-9_-]{11})"
)

_yt_api = YouTubeTranscriptApi()


def extract_video_id(url: str) -> str | None:
    """Extract YouTube video ID from various URL formats."""
    m = VIDEO_ID_RE.search(url)
    return m.group(1) if m else None


def _try_languages(video_id: str, langs: list[str]) -> list | None:
    """Try to get transcript, falling back through languages."""
    all_ = list(dict.fromkeys([*langs, "en"]))
    for l in all_:
        try:
            result = _yt_api.fetch(video_id, languages=[l], preserve_formatting=False)
            return list(result)
        except NoTranscriptFound:
            continue
        except Exception:
            continue
    return None


@router.get("/youtube/transcript")
async def get_transcript(
    url: str = Query(..., description="YouTube video URL"),
    lang: str = Query("th,en", description="Language codes (comma-separated)"),
    max_chars: int = Query(8000, description="Max transcript characters"),
):
    """Fetch YouTube video transcript as text."""
    video_id = extract_video_id(url)
    if not video_id:
        raise HTTPException(400, "Invalid YouTube URL — could not extract video ID")

    langs = [l.strip() for l in lang.split(",") if l.strip()]

    try:
        raw = _try_languages(video_id, langs)
    except Exception as e:
        raise HTTPException(400, f"Failed to fetch transcript: {e}")

    if not raw:
        raise HTTPException(404, "No transcript found for this video")

    # Format as plain text with timestamps
    lines = []
    char_count = 0
    for entry in raw:
        text = (entry.text or "").strip()
        if not text:
            continue
        seconds = int(entry.start or 0)
        mins = seconds // 60
        secs = seconds % 60
        line = f"[{mins}:{secs:02d}] {text}"
        if char_count + len(line) > max_chars:
            lines.append("… [transcript truncated]")
            break
        lines.append(line)
        char_count += len(line)

    full_text = "\n".join(lines)

    return {
        "video_id": video_id,
        "lines": len(lines),
        "language": langs[0] if langs else "en",
        "text": full_text,
    }

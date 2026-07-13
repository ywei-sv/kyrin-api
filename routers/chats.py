"""
Chat session persistence — saves/loads chats as JSON files.
"""
import os
import json
import glob
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

DATA_DIR = os.environ.get("KYRIN_CHATS_DIR", os.path.expanduser("~/.kyrin/chats"))


class ChatSession(BaseModel):
    id: str
    title: str
    messages: list[dict]
    updatedAt: int | float


def _ensure_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def _path(sid: str) -> str:
    return os.path.join(DATA_DIR, f"{sid}.json")


@router.get("/chats")
async def list_chats():
    """List all saved chat sessions, sorted by updatedAt desc."""
    _ensure_dir()
    sessions = []
    for fp in sorted(glob.glob(os.path.join(DATA_DIR, "*.json")), reverse=True):
        try:
            with open(fp) as f:
                data = json.load(f)
                sessions.append(data)
        except Exception:
            continue
    sessions.sort(key=lambda s: s.get("updatedAt", 0), reverse=True)
    return sessions


@router.get("/chats/{chat_id}")
async def get_chat(chat_id: str):
    """Get a single chat session."""
    p = _path(chat_id)
    if not os.path.exists(p):
        raise HTTPException(status_code=404, detail="Chat not found")
    try:
        with open(p) as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chats")
async def save_chat(session: ChatSession):
    """Save (create or overwrite) a chat session."""
    _ensure_dir()
    p = _path(session.id)
    try:
        with open(p, "w") as f:
            json.dump(session.model_dump(), f, indent=2, ensure_ascii=False)
        return {"status": "ok", "id": session.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/chats/{chat_id}")
async def delete_chat(chat_id: str):
    """Delete a chat session."""
    p = _path(chat_id)
    if not os.path.exists(p):
        raise HTTPException(status_code=404, detail="Chat not found")
    os.remove(p)
    return {"status": "deleted", "id": chat_id}
